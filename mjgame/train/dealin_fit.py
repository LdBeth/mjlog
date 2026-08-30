#!/usr/bin/env python3
"""M14 — fit the learned deal-in read (and its tenpai head) in MLX.

    train/.venv/bin/python train/dealin_fit.py --lane runs/dealin/700000 \\
        --out weights/dealin-0829.json [--hidden 32,32 --epochs 20 \\
        --pos-weight auto --focal 0 --lr 1e-3 --l2 1e-4 --batch 8192]

WHAT IS BEING FITTED.  P(opponent i rons tile type t | public state), as a
sigmoid over a small MLP of `scripts/dealin_export.ts`'s frozen feature columns,
plus a second head for P(that opponent is tenpai) over the tenpai columns.  The
labels are the ORACLE's, read off the live table by the calibration recorder:
`y = 1` iff the type was in that opponent's ron set at that instant, `tt = 1`
iff they were actually tenpai.  Nothing here re-derives a feature; the lane is
the contract (see `src/ai/dealin.ts`'s "one state builder, two consumers").

THE THREE THINGS THAT MAKE THE NUMBER HONEST

1.  SPLIT BY HANCHAN SEED, never by row.  Every cell of one decision shares a
    board, and a random row split would put the same board on both sides and
    report a holdout number that is really a training number.  Odd seed =
    holdout, exactly as `scripts/calibrate_fit.ts` splits.

2.  THE SUBSAMPLING IS UNDONE BY WEIGHT.  The export keeps every positive and a
    `negKeep` share of the negatives; each surviving negative therefore stands
    for `1/negKeep` of them, and that is its weight in the loss.  Without it the
    head would learn the sampled base rate (~6%) rather than the real one
    (~0.6%), and every probability it serves would be an order of magnitude hot.

3.  THE BASELINE IS IN THE LANE.  `lane.pc.f32` is the closed-form model's own
    `dealinP` for the very same cells.  Every holdout table prints the two side
    by side; a head that does not beat that column on BCE and Brier, in every
    junme band and against riichi/副露/quiet alike, has not earned a promotion.

pos_weight vs focal: `--pos-weight auto` scales the positive term by
(Σ negative weight / Σ positive weight) so the two classes contribute equally —
the usual fix for a 0.6% base rate.  `--focal 2` instead down-weights easy cells
by (1−p)^γ.  They compose; `--pos-weight 1 --focal 0` is plain weighted BCE.

AND THE POSITIVE WEIGHT IS UNDONE BEFORE THE BLOCK IS WRITTEN.  Up-weighting the
positive class by pi moves the optimum from logit(p) to logit(p) + log(pi), i.e.
it calibrates the head for a table where deal-ins are as common as safe tiles.
`AugmentedHeuristic.riskOf` multiplies this probability by a payment in POINTS,
so an uncorrected head would price every tile ~150x hot and the seat would fold
the whole game.  `train_head` subtracts log(pi) from the output bias — exact,
since the shift is a constant on the logit — and every number reported below,
and every weight shipped, is in the measure the seat serves in.

WHAT IT WRITES
    --out                 {"dealin": {fv, dealin, tenpai}} — a ktune block,
                          f32-exact through `export_mlp_block`
    --out + ".meta.json"  the fit's own record: sizes, hyper-parameters, and
                          every holdout table as data
    lane.pred.f32         the first 10k rows' LOGITS through the numpy
                          double-accumulate reference, for
                          `scripts/dealin_report.ts` to reproduce bit for bit in
                          TypeScript.  Not `mx` output: the parity contract is
                          the sequential-sum loop in `train/common.py`.
"""

import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SmallMlp, export_mlp_block, mlp_forward_np  # noqa: E402

PRED_ROWS = 10000


# ---------------------------------------------------------------------------
# the lane
# ---------------------------------------------------------------------------


def _read_meta(path):
    """(header dict, int64 [N, C] of the per-row columns)."""
    with open(path) as f:
        head = json.loads(f.readline())
        rows = [json.loads(line) for line in f if line.strip()]
    return head, np.asarray(rows, dtype=np.int64)


def load_lane(prefix):
    head, meta = _read_meta(f"{prefix}.meta.jsonl")
    n_feat = len(head["features"])
    X = np.fromfile(f"{prefix}.X.f32", dtype="<f4").reshape(-1, n_feat)
    y = np.fromfile(f"{prefix}.y.u8", dtype=np.uint8).astype(np.float32)
    pc = np.fromfile(f"{prefix}.pc.f32", dtype="<f4")
    if not (len(X) == len(y) == len(pc) == len(meta)):
        raise SystemExit(
            f"lane rows disagree: X {len(X)}, y {len(y)}, pc {len(pc)}, meta {len(meta)}"
        )

    thead, tmeta = _read_meta(f"{prefix}.tmeta.jsonl")
    n_tfeat = len(thead["tenpaiFeatures"])
    T = np.fromfile(f"{prefix}.T.f32", dtype="<f4").reshape(-1, n_tfeat)
    tt = np.fromfile(f"{prefix}.tt.u8", dtype=np.uint8).astype(np.float32)
    if not (len(T) == len(tt) == len(tmeta)):
        raise SystemExit(f"tenpai rows disagree: T {len(T)}, tt {len(tt)}, meta {len(tmeta)}")
    return head, meta, X, y, pc, T, tt, tmeta


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def bce_with_logits(logits, y, w, pos_weight, gamma):
    """Weighted BCE on logits, with an optional focal modulation.

    The stable form: max(z,0) - z*y + log1p(exp(-|z|)).  The positive class is
    scaled by `pos_weight`, every row by `w` (the 1/keep de-biasing).
    """
    z = logits.reshape(-1)
    base = mx.maximum(z, 0) - z * y + mx.log1p(mx.exp(-mx.abs(z)))
    # Per-class scaling: `pos_weight` on the positive term only, which is what
    # keeps the negatives' contribution unchanged.
    cls = 1.0 + (pos_weight - 1.0) * y
    loss = base * cls
    if gamma > 0:
        p = mx.sigmoid(z)
        pt = p * y + (1 - p) * (1 - y)
        loss = loss * mx.power(1 - pt, gamma)
    return mx.sum(loss * w) / mx.sum(w)


def train_head(X, y, w, dims, args, tag, seed):
    model = SmallMlp(dims)
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=args.lr)

    pos_w = args.pos_weight
    if pos_w == "auto":
        wp = float(np.sum(w[y > 0]))
        wn = float(np.sum(w[y == 0]))
        pos_w = wn / wp if wp > 0 else 1.0
    pos_w = float(pos_w)

    def loss_fn(m, xb, yb, wb):
        out = m(xb)
        loss = bce_with_logits(out, yb, wb, pos_w, args.focal)
        if args.l2 > 0:
            reg = mx.array(0.0)
            for layer in m.ordered_layers:
                reg = reg + mx.sum(layer.weight * layer.weight)
            loss = loss + args.l2 * reg
        return loss

    step = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(seed)
    n = len(X)
    Xm = mx.array(X)
    ym = mx.array(y)
    wm = mx.array(w)
    history = []
    for ep in range(args.epochs):
        order = rng.permutation(n)
        total = 0.0
        nb = 0
        for i in range(0, n, args.batch):
            idx = mx.array(order[i : i + args.batch])
            loss, grads = step(model, Xm[idx], ym[idx], wm[idx])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            total += float(loss)
            nb += 1
        history.append(total / max(1, nb))
        print(f"  [{tag}] epoch {ep + 1:3d}/{args.epochs}  loss {history[-1]:.6f}", flush=True)

    # PRIOR CORRECTION, and it is not optional.  Up-weighting the positive class
    # by pi shifts the optimum from logit(p) to logit(p) + log(pi): the head is
    # then calibrated for a table where deal-ins are as common as safe tiles,
    # which is not the table anyone plays at.  `riskOf` multiplies this
    # probability by a payment in points, so a head that is 30x hot prices every
    # tile 30x hot.  Subtracting log(pi) from the output bias undoes exactly that
    # (the shift is a constant on the logit), and what ships is calibrated in the
    # measure it is served in.  `--pos-weight 1` makes this a no-op.
    if pos_w != 1.0:
        last = model.ordered_layers[-1]
        last.bias = last.bias - math.log(pos_w)
        mx.eval(model.parameters())
        print(f"  [{tag}] 事前補正: 出力バイアスから log({pos_w:.4g}) を引いた", flush=True)
    return model, pos_w, history


def forward_np(model, X, batch=65536):
    """Logits for scoring (mx path — the tables do not need bit-exactness)."""
    out = []
    for i in range(0, len(X), batch):
        out.append(np.asarray(model(mx.array(X[i : i + batch]))).reshape(-1))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _bce(p, y, w):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(np.sum(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def _brier(p, y, w):
    return float(np.sum(w * (p.astype(np.float64) - y) ** 2) / np.sum(w))


def score_block(p, y, w):
    return {"n": int(len(y)), "rate": float(np.sum(w * y) / np.sum(w)),
            "bce": _bce(p, y, w), "brier": _brier(p, y, w)}


BANDS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 1.0]


def reliability(p, y, w):
    """10 bands of predicted probability: what was claimed, what happened."""
    out = []
    edges = [0.0] + BANDS
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo)
        if not m.any():
            out.append({"lo": lo, "hi": hi, "n": 0})
            continue
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "n": int(m.sum()),
                "pred": float(np.sum(w[m] * p[m]) / np.sum(w[m])),
                "truth": float(np.sum(w[m] * y[m]) / np.sum(w[m])),
            }
        )
    return out


def print_reliability(name, table):
    print(f"  {name}")
    print("    帯                n        予測      実際")
    for b in table:
        if b["n"] == 0:
            continue
        print(
            f"    [{b['lo']:.4f},{b['hi']:.4f})  {b['n']:9d}  "
            f"{100 * b['pred']:8.3f}%  {100 * b['truth']:8.3f}%"
        )


def strata(meta, junme_edges=(6, 9, 12)):
    """(name, mask) pairs: junme buckets, then 立直/副露/静か."""
    j = meta[:, 1]
    c = meta[:, 2]
    out = []
    lo = 0
    for hi in junme_edges:
        out.append((f"巡目 {lo + 1}–{hi}", (j > lo) & (j <= hi)))
        lo = hi
    out.append((f"巡目 {lo + 1}+", j > lo))
    for code, name in ((0, "立直"), (1, "副露"), (2, "静か")):
        out.append((name, c == code))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description="M14 learned deal-in read")
    ap.add_argument("--lane", required=True, help="prefix written by dealin_export.ts")
    ap.add_argument("--out", required=True, help="ktune JSON to write")
    ap.add_argument("--hidden", default="32,32")
    ap.add_argument("--tenpai-hidden", default="16")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--pos-weight", default="auto")
    ap.add_argument("--focal", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260829)
    args = ap.parse_args(argv)

    t0 = time.time()
    head, meta, X, y, pc, T, tt, tmeta = load_lane(args.lane)
    fv = int(head["fv"])
    keep = float(head["negKeep"])
    print(f"レーン {args.lane}: 放銃 {len(X)}行 × {X.shape[1]}列, 聴牌 {len(T)}行 × {T.shape[1]}列")

    # (1) the split: by hanchan seed, never by row.
    hold = (meta[:, 0] % 2) == 1
    thold = (tmeta[:, 0] % 2) == 1
    w = np.where(y > 0, 1.0, 1.0 / keep).astype(np.float32)
    tw = np.ones(len(tt), dtype=np.float32)
    print(
        f"  学習 {int((~hold).sum())} / 検証 {int(hold.sum())} 行 "
        f"(奇数の種を検証に回す), 正例率 {100 * float(np.sum(w * y) / np.sum(w)):.3f}%"
    )

    dims = [X.shape[1]] + [int(h) for h in args.hidden.split(",") if h] + [1]
    tdims = [T.shape[1]] + [int(h) for h in args.tenpai_hidden.split(",") if h] + [1]
    model, pos_w, hist = train_head(
        X[~hold], y[~hold], w[~hold], dims, args, "放銃", args.seed
    )
    tmodel, tpos_w, thist = train_head(
        T[~thold], tt[~thold], tw[~thold], tdims, args, "聴牌", args.seed + 1
    )

    # (2) the holdout, against the closed form that is already shipping.
    ph = 1.0 / (1.0 + np.exp(-forward_np(model, X[hold]).astype(np.float64)))
    yh = y[hold].astype(np.float64)
    wh = w[hold].astype(np.float64)
    pch = pc[hold].astype(np.float64)
    learned = score_block(ph, yh, wh)
    baseline = score_block(pch, yh, wh)
    print("\n検証 (奇数の種):")
    print(f"  行 {learned['n']}  実際の放銃率 {100 * learned['rate']:.4f}%")
    print(f"  BCE   学習ヘッド {learned['bce']:.6f}   計算 {baseline['bce']:.6f}")
    print(f"  Brier 学習ヘッド {learned['brier']:.8f}   計算 {baseline['brier']:.8f}")
    print_reliability("学習ヘッド", reliability(ph, yh, wh))
    print_reliability("計算 (基準)", reliability(pch, yh, wh))

    cuts = {}
    print("\n  層別 (BCE 学習 / 計算):")
    for name, m in strata(meta[hold]):
        if not m.any():
            continue
        a = score_block(ph[m], yh[m], wh[m])
        b = score_block(pch[m], yh[m], wh[m])
        cuts[name] = {"learned": a, "computed": b}
        flag = "◎" if a["bce"] < b["bce"] else "×"
        print(
            f"    {name:10s} n={a['n']:8d}  {a['bce']:.6f} / {b['bce']:.6f}  "
            f"Brier {a['brier']:.8f} / {b['brier']:.8f}  {flag}"
        )

    pth = 1.0 / (1.0 + np.exp(-forward_np(tmodel, T[thold]).astype(np.float64)))
    tth = tt[thold].astype(np.float64)
    twh = tw[thold].astype(np.float64)
    tprior = T[thold][:, head["tenpaiFeatures"].index("tpPrior")].astype(np.float64)
    tlearned = score_block(pth, tth, twh)
    tbase = score_block(tprior, tth, twh)
    print("\n聴牌ヘッド (立直の行は除いてある):")
    print(f"  行 {tlearned['n']}  実際の聴牌率 {100 * tlearned['rate']:.3f}%")
    print(f"  BCE   学習 {tlearned['bce']:.6f}   計算の事前確率 {tbase['bce']:.6f}")
    print(f"  Brier 学習 {tlearned['brier']:.6f}   計算の事前確率 {tbase['brier']:.6f}")
    print_reliability("学習ヘッド", reliability(pth, tth, twh))

    # (3) the blocks, f32-exact.
    block = export_mlp_block(model.ordered_layers, model.acts, fv)
    tblock = export_mlp_block(tmodel.ordered_layers, tmodel.acts, fv)
    out = {"dealin": {"fv": fv, "dealin": block, "tenpai": tblock}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
        f.write("\n")

    # (4) the parity fixture: the first rows' LOGITS through the double-accumulate
    # reference, which is the loop `src/ai/mlp.ts` runs.  `scripts/dealin_report.ts`
    # must reproduce these bit for bit.
    n_pred = min(PRED_ROWS, len(X))
    pred = mlp_forward_np(block, X[:n_pred]).reshape(-1).astype("<f4")
    pred.tofile(f"{args.lane}.pred.f32")

    meta_out = {
        "lane": args.lane,
        "out": args.out,
        "fv": fv,
        "negKeep": keep,
        "rows": int(len(X)),
        "tenpaiRows": int(len(T)),
        "holdoutRows": int(hold.sum()),
        "dims": dims,
        "tenpaiDims": tdims,
        "epochs": args.epochs,
        "lr": args.lr,
        "l2": args.l2,
        "batch": args.batch,
        "posWeight": pos_w,
        "tenpaiPosWeight": tpos_w,
        "focal": args.focal,
        "seed": args.seed,
        "trainLoss": hist,
        "tenpaiTrainLoss": thist,
        "holdout": {"learned": learned, "computed": baseline},
        "reliability": {
            "learned": reliability(ph, yh, wh),
            "computed": reliability(pch, yh, wh),
        },
        "strata": cuts,
        "tenpai": {"learned": tlearned, "prior": tbase,
                   "reliability": reliability(pth, tth, twh)},
        "predRows": int(n_pred),
        "seconds": round(time.time() - t0, 1),
    }
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta_out, f, indent=1)
        f.write("\n")
    print(f"\n書き出し: {args.out}  (+ .meta.json, {args.lane}.pred.f32 {n_pred}行)")


if __name__ == "__main__":
    main()
