#!/usr/bin/env python3
"""M13 — fit the fold head off an ε-flip contextual-bandit lane.

    train/.venv/bin/python train/fold_fit.py \
        --in runs/fold/lane.jsonl --out weights/fold-MMDD.json \
        [--hidden 16] [--epochs 40] [--lr 1e-3] [--l2 1e-4] [--clip 20]

WHAT THE LANE IS.  `selfplay --foldcalib=PATH --fold-eps=X` plays the 計算 seat
with its push/fold verdict FLIPPED with probability X, and writes one row per
gated fold decision: the 37 features the head reads (`src/ai/fold.ts`
`FOLD_FEATURES`), the verdict the seat's own rule produced, the action actually
taken, the propensity `p` of that action (1−ε or ε), and the round's own
settlement.  There is no supervised label anywhere in it: nothing in a mahjong
log says what folding would have paid on a hand that pushed.  The flips are the
only counterfactual evidence that exists, and everything below is an estimator
built on them.

THE REWARD (plan D7), stated once and obeyed everywhere: `delta / 1000`, the
round's `deltas[0]` in thousands of points.  Nothing else.  The lane also
carries `vio0` (dojo ledger entries charged to the seat in that round), `won`,
`dealtIn` and `endJunme`; those are DATA, printed in the report and never in the
objective.  The project's reward philosophy is that violations are minimised as
a byproduct of long-term reward, never by per-decision shaping, and a term for
them here would be exactly that shaping.

THE INDEPENDENCE ASSUMPTION.  Every fold decision inside one 局 shares that
局's single settlement, so two flipped decisions in one round are credited with
the same number and the estimator treats them as independent draws.  That is
false, and the size of the lie is REPORTED rather than assumed away: the
"複数反転局" fraction below is the share of rounds carrying more than one flip.
At ε = 0.05 over ~7 gated decisions a round it is a few percent, and the bias it
introduces is second-order in ε.  A lane at ε = 0.3 should not be read this way.

THE ESTIMATOR (plan B5/D9), in three steps.

  (1) An OUTCOME MODEL q̂(x, a) — SmallMlp(38 → 32 → 1) on the 37 features plus
      the action indicator, fitted with a Huber loss (δ = 8, i.e. a mangan in
      the reward's own units) so a haneman deal-in does not own the fit.

  (2) A DOUBLY-ROBUST per-decision advantage of folding over pushing,

          Δ̂ = [q̂(x,F) − q̂(x,P)]
              + (1[a=F]/p − 1[a=P]/p) · (r − q̂(x,a))

      — the model's difference, corrected by the importance-weighted residual of
      whichever action was actually logged.  It is consistent if EITHER the
      model or the propensities are right, which is the reason for the shape:
      the propensities here are exact by construction (the engine drew them),
      so the correction term is what carries the estimate and the model only
      reduces its variance.  The weights are clipped at `--clip` (default 20,
      i.e. 1/ε at ε = 0.05) so one lucky flip cannot dominate.

  (3) THE HEAD, SmallMlp(37 → hidden → 1), trained to predict `sign(Δ̂)` with a
      BCE weighted by `|Δ̂|`: a decision the estimator is confident about pulls
      hard, one it cannot separate pulls barely at all.  Adam, L2 on the
      weights, and an INIT THAT IS THE OLD GATE — see below.

THE INIT IS THE INCUMBENT.  `INIT_FOLD` (src/ai/fold.ts) is one linear layer
with `w[margin] = −1`: `out > 0 ⇔ margin < 0`, the hand-written comparison
exactly.  A hidden layer cannot be that layer, so it is WIDENED instead:
hidden unit 0 is `relu(−margin)` and the output layer reads it with weight +1
and everything else 0, which gives `out = relu(−margin) > 0 ⇔ margin < 0` —
the same decision boundary, including the tie.  The remaining hidden units get
small random weights and a zero output weight, so they are silent at epoch 0
and alive from the first gradient step.  The script VERIFIES this numerically
before training (sign equality on every row, not value equality — `relu` is not
the identity) and refuses to continue if it fails.

OUTPUT.  `--out` gets `{"fold": <block>}`, ready to paste beside a champion's
sections or to hand to `--ktune` on its own, and `<out>.meta.json` gets the
provenance and the holdout report.
"""

from __future__ import annotations

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

KIND = "mjgame-fold"
VERSION = 1
# Must equal `FOLD_FV` in src/ai/fold.ts. A lane or a block at another feature
# version describes different columns and is refused, never coerced.
FOLD_FV = 1


# ---------------------------------------------------------------------------
# the lane
# ---------------------------------------------------------------------------


def load_lane(path: str):
    """(header, X [N,37] f32, A [N] u8 fold=1, P [N] f32, R [N] f32, S [N] i64, meta)."""
    xs, acts, ps, rs, seeds = [], [], [], [], []
    vio, flips, keys, verdicts = [], [], [], []
    header = None
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if header is None:
                header = obj
                if header.get("kind") != KIND:
                    raise SystemExit(f"{path}: 押し引きの較正記録ではありません")
                if header.get("v") != VERSION:
                    raise SystemExit(f"{path}: 版が違います (v{header.get('v')})")
                if header.get("fv") != FOLD_FV:
                    raise SystemExit(f"{path}: 特徴量版が違います (fv{header.get('fv')})")
                continue
            xs.append(obj["x"])
            acts.append(1 if obj["taken"] else 0)
            verdicts.append(1 if obj["verdict"] else 0)
            ps.append(obj["p"])
            rs.append(obj["delta"] / 1000.0)  # D7: the reward, in thousands
            seeds.append(obj["s"])
            vio.append(obj["vio0"])
            flips.append(obj["flipped"])
            keys.append((obj["s"], obj["k"], obj["b"]))
    if header is None:
        raise SystemExit(f"{path}: 空のファイルです")
    X = np.asarray(xs, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(header["features"]):
        raise SystemExit(f"{path}: 特徴量の幅が頭書きと違います")
    meta = {
        "keys": keys,
        "flips": np.asarray(flips, dtype=np.int32),
        "vio0": np.asarray(vio, dtype=np.int32),
        "verdict": np.asarray(verdicts, dtype=np.int32),
    }
    return (
        header,
        X,
        np.asarray(acts, dtype=np.int32),
        np.asarray(ps, dtype=np.float32),
        np.asarray(rs, dtype=np.float32),
        np.asarray(seeds, dtype=np.int64),
        meta,
    )


def multi_flip_fraction(keys, flips) -> tuple[float, int, int]:
    """Share of 局 in which more than one decision was flipped (the D7 caveat)."""
    per = {}
    for k, f in zip(keys, flips):
        per[k] = per.get(k, 0) + int(f)
    rounds = len(per)
    multi = sum(1 for v in per.values() if v > 1)
    return (multi / rounds if rounds else 0.0), multi, rounds


# ---------------------------------------------------------------------------
# (1) the outcome model
# ---------------------------------------------------------------------------


def huber(pred, target, delta: float):
    d = pred - target
    a = mx.abs(d)
    return mx.where(a <= delta, 0.5 * d * d, delta * (a - 0.5 * delta))


def fit_q(X, A, R, *, epochs: int, lr: float, seed: int = 13):
    """SmallMlp(38 -> 32 -> 1) on [features, action], Huber δ=8."""
    mx.random.seed(seed)
    net = SmallMlp([X.shape[1] + 1, 32, 1])
    opt = optim.Adam(learning_rate=lr)
    inp = mx.array(np.concatenate([X, A.reshape(-1, 1).astype(np.float32)], axis=1))
    tgt = mx.array(R.reshape(-1, 1))

    def loss_fn(model):
        return mx.mean(huber(model(inp), tgt, 8.0))

    grad_fn = nn.value_and_grad(net, loss_fn)
    last = float("nan")
    for ep in range(epochs):
        loss, grads = grad_fn(net)
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state)
        last = float(loss)
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"  q̂ epoch {ep + 1:3d}/{epochs}  huber {last:.4f}")
    return net, last


def q_of(net, X, a: int) -> np.ndarray:
    """q̂(x, a) for a fixed action, as float64 numpy."""
    col = np.full((X.shape[0], 1), float(a), dtype=np.float32)
    out = net(mx.array(np.concatenate([X, col], axis=1)))
    mx.eval(out)
    return np.asarray(out, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# (2) the doubly-robust advantage
# ---------------------------------------------------------------------------


def dr_advantage(qF, qP, A, P, R, clip: float):
    """Δ̂ per decision: positive ⇒ FOLDING is estimated to pay better."""
    w = np.minimum(1.0 / np.maximum(P, 1e-12), clip)
    isF = (A == 1).astype(np.float64)
    isP = 1.0 - isF
    qa = np.where(A == 1, qF, qP)
    resid = R.astype(np.float64) - qa
    return (qF - qP) + (isF * w - isP * w) * resid


def dr_value(policy_fold, qF, qP, A, P, R, clip: float) -> float:
    """DR value of a DETERMINISTIC policy: `policy_fold` is its 0/1 fold decision."""
    w = np.minimum(1.0 / np.maximum(P, 1e-12), clip)
    qpi = np.where(policy_fold == 1, qF, qP)
    agree = (policy_fold == A).astype(np.float64)
    qa = np.where(A == 1, qF, qP)
    return float(np.mean(qpi + agree * w * (R.astype(np.float64) - qa)))


def snips(policy_fold, A, P, R) -> tuple[float, float]:
    """(SNIPS value, ESS) — no outcome model, so `fold_report.ts` can reproduce it."""
    w = np.where(policy_fold == A, 1.0 / np.maximum(P, 1e-12), 0.0)
    s1 = float(w.sum())
    if s1 <= 0:
        return float("nan"), 0.0
    s2 = float((w * w).sum())
    return float((w * R.astype(np.float64)).sum() / s1), (s1 * s1 / s2 if s2 > 0 else 0.0)


# ---------------------------------------------------------------------------
# (3) the head
# ---------------------------------------------------------------------------


def init_head(n_in: int, hidden: int, margin_idx: int, seed: int = 1313) -> SmallMlp:
    """INIT_FOLD, widened: hidden unit 0 is `relu(−margin)`, output reads it +1."""
    if hidden < 1:
        raise SystemExit("--hidden は1以上")
    mx.random.seed(seed)
    net = SmallMlp([n_in, hidden, 1])
    rng = np.random.default_rng(seed)
    W0 = np.zeros((hidden, n_in), dtype=np.float32)
    W0[0, margin_idx] = -1.0
    if hidden > 1:
        # Small, so the units are near-linear and un-saturated when the output
        # layer first learns to read them; zero would be a dead ReLU.
        W0[1:, :] = rng.normal(0.0, 0.05, size=(hidden - 1, n_in)).astype(np.float32)
    b0 = np.zeros((hidden,), dtype=np.float32)
    W1 = np.zeros((1, hidden), dtype=np.float32)
    W1[0, 0] = 1.0
    b1 = np.zeros((1,), dtype=np.float32)
    layers = net.ordered_layers
    layers[0].weight = mx.array(W0)
    layers[0].bias = mx.array(b0)
    layers[1].weight = mx.array(W1)
    layers[1].bias = mx.array(b1)
    mx.eval(net.parameters())
    return net


def head_block(net: SmallMlp) -> dict:
    return export_mlp_block(net.ordered_layers, net.acts, FOLD_FV)


def verify_epoch0(block: dict, X: np.ndarray, margin_idx: int) -> int:
    """Sign equality with the old gate on every row. Returns the mismatch count.

    Checked through `mlp_forward_np`, the SAME reference `src/ai/mlp.ts` and
    `native/mlp.c` are pinned to — so passing here means the engine agrees too.
    """
    out = mlp_forward_np(block, X).reshape(-1)
    want = X[:, margin_idx] < 0
    got = out > 0
    return int(np.count_nonzero(want != got))


def train_head(net, X, label, weight, *, epochs, lr, l2, seed=1313):
    """BCE(sigmoid(out), label) weighted by |Δ̂|, plus L2 on every weight."""
    mx.random.seed(seed)
    opt = optim.Adam(learning_rate=lr)
    inp = mx.array(X)
    y = mx.array(label.reshape(-1, 1).astype(np.float32))
    w = mx.array(weight.reshape(-1, 1).astype(np.float32))
    wsum = float(weight.sum()) or 1.0

    def loss_fn(model):
        logit = model(inp)
        bce = nn.losses.binary_cross_entropy(logit, y, with_logits=True, reduction="none")
        reg = mx.zeros(())
        for layer in model.ordered_layers:
            reg = reg + mx.sum(layer.weight * layer.weight)
        return mx.sum(bce * w) / wsum + l2 * reg

    grad_fn = nn.value_and_grad(net, loss_fn)
    for ep in range(epochs):
        loss, grads = grad_fn(net)
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state)
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"  head epoch {ep + 1:3d}/{epochs}  loss {float(loss):.5f}")
    return net


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="M13 fold head, fitted off an ε-flip lane")
    ap.add_argument("--in", dest="inp", required=True, help="selfplay --foldcalib の JSONL")
    ap.add_argument("--out", required=True, help='書き出し先 ({"fold": block})')
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--q-epochs", type=int, default=None, help="既定は --epochs と同じ")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=20.0, help="IPS 重みの上限 (既定 20 = 1/ε)")
    args = ap.parse_args()

    t0 = time.time()
    header, X, A, P, R, S, meta = load_lane(args.inp)
    feats = list(header["features"])
    if "margin" not in feats:
        raise SystemExit("頭書きに margin 列がありません")
    margin_idx = feats.index("margin")
    n = X.shape[0]
    if n == 0:
        raise SystemExit(f"{args.inp}: 行がありません")

    # Split by seed parity: even = fit, odd = holdout. A game is one seed, so no
    # round can straddle the split and no decision can leak its own round's
    # settlement into the other side.
    hold = (S % 2) == 1
    fit = ~hold
    frac, multi, rounds = multi_flip_fraction(meta["keys"], meta["flips"])
    print(f"■ 記録 {args.inp}")
    print(f"  行 {n}  局 {rounds}  ε={header['eps']}  判定={header['head']}  fv={header['fv']}")
    print(f"  当てはめ {int(fit.sum())}行 / 検証 {int(hold.sum())}行 (偶数種/奇数種)")
    print(f"  反転 {int(meta['flips'].sum())}行   複数反転局 {multi}/{rounds} ({100 * frac:.1f}%)")
    print(f"  ※ 局内の判断は同じ収支を共有します — 上の割合が独立性の嘘の大きさです")
    print(f"  席0の違反 {int(meta['vio0'].sum())}件 (記録のみ — 目的関数には入りません)")
    if int(meta["flips"].sum()) == 0:
        print("  ⚠ 反転が1件もありません — 反実仮想がないので Δ̂ は模型そのものです")

    # (1) the outcome model, fitted on the FIT split alone.
    print("■ (1) 結果模型 q̂(x,a)  SmallMlp(38→32→1)  Huber δ=8")
    qnet, qloss = fit_q(X[fit], A[fit], R[fit], epochs=args.q_epochs or args.epochs, lr=args.lr)

    qF = q_of(qnet, X, 1)
    qP = q_of(qnet, X, 0)

    # (2) the DR advantage of folding, per decision.
    print("■ (2) 二重頑健な優位 Δ̂ = [q̂(F)−q̂(P)] + (1[a=F]/p − 1[a=P]/p)(r − q̂(a))")
    adv = dr_advantage(qF, qP, A, P, R, args.clip)
    print(
        f"  Δ̂: 平均 {adv.mean():+.4f}  中央 {np.median(adv):+.4f}  "
        f"降り優位 {100 * float((adv > 0).mean()):.1f}%"
    )

    # (3) the head.
    label = (adv > 0).astype(np.float32)
    weight = np.abs(adv).astype(np.float32)
    net = init_head(X.shape[1], args.hidden, margin_idx)
    block = head_block(net)
    bad = verify_epoch0(block, X, margin_idx)
    print(f"■ (3) 初期値 ≡ 旧ゲート の検査: 不一致 {bad}/{n} 行  " + ("OK" if bad == 0 else "NG"))
    if bad != 0:
        raise SystemExit("初期ヘッドが旧ゲートと一致しません — 学習を中止します")
    print(f"■ 学習  SmallMlp({X.shape[1]}→{args.hidden}→1)  BCE(|Δ̂| 重み) + L2 {args.l2}")
    train_head(net, X[fit], label[fit], weight[fit], epochs=args.epochs, lr=args.lr, l2=args.l2)
    block = head_block(net)

    # ---- the holdout report
    head_fold = (mlp_forward_np(block, X).reshape(-1) > 0).astype(np.int32)
    gate_fold = (X[:, margin_idx] < 0).astype(np.int32)
    always_push = np.zeros(n, dtype=np.int32)
    always_fold = np.ones(n, dtype=np.int32)
    print("■ 検証 (奇数種) — 局収支 (千点) の推定値")
    print("  方策            DR       SNIPS      ESS     降り率")
    rows = []
    for name, pol in (
        ("旧ゲート", gate_fold),
        ("学習ヘッド", head_fold),
        ("常に押し", always_push),
        ("常に降り", always_fold),
    ):
        v = dr_value(pol[hold], qF[hold], qP[hold], A[hold], P[hold], R[hold], args.clip)
        sn, ess = snips(pol[hold], A[hold], P[hold], R[hold])
        rate = 100 * float(pol[hold].mean())
        rows.append({"policy": name, "dr": v, "snips": sn, "ess": ess, "foldRate": rate})
        print(f"  {name:<10} {v:+9.4f} {sn:+10.4f} {ess:9.1f}  {rate:6.1f}%")
    agree = float((head_fold[hold] == gate_fold[hold]).mean())
    print(f"  ヘッドと旧ゲートの一致 {100 * agree:.1f}%")
    print("  ※ SNIPS は scripts/fold_report.ts --fold=… と 1e-6 まで一致するはずの数です")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"fold": block}, fh, indent=2)
        fh.write("\n")
    metaout = {
        "kind": "mjgame-fold-fit",
        "lane": os.path.abspath(args.inp),
        "laneHeader": header,
        "rows": int(n),
        "fitRows": int(fit.sum()),
        "holdoutRows": int(hold.sum()),
        "rounds": rounds,
        "multiFlipRounds": multi,
        "multiFlipFraction": frac,
        "flips": int(meta["flips"].sum()),
        "vio0": int(meta["vio0"].sum()),
        "hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "l2": args.l2,
        "clip": args.clip,
        "qHuber": qloss,
        "advMean": float(adv.mean()),
        "holdout": rows,
        "gateAgreement": agree,
        "seconds": round(time.time() - t0, 2),
    }
    with open(args.out + ".meta.json", "w") as fh:
        json.dump(metaout, fh, indent=2)
        fh.write("\n")
    print(f"■ 書き出し {args.out}  (+ .meta.json)  {time.time() - t0:.1f}s")
    print("  昇格には対戦評価が要ります: paired --ktune=<champion+fold> --ktune-b=champion")


if __name__ == "__main__":
    main()
