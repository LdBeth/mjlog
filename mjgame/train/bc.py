#!/usr/bin/env python3
"""Behavior cloning for the mjgame policy.

    python bc.py --data 'runs/*.jsonl' --out weights/ \
        --epochs 10 --batch 4096 --lr 3e-4 --val 0.05

Trains only the policy head; the value head (output 79) is sliced off inside
`masked_cross_entropy`, so it stays at its random init and contributes no
gradient.

BC IS A v3-WIDTH TRAINER, deliberately.  Feature v4 hangs a 64-wide attention
summary of the rivers off the end of the policy input, and that encoder is a
second parameter tree with its own forward (ppo.py); cloning a heuristic's
discards does not need it and would only train it against a teacher that never
saw it.  So this trains the plain 1674-wide MLP over the dense features that a
v4 trajectory still carries verbatim, and writes a v3 weight set.  Run
`train/widen4.py` on the result to get the v4 net PPO wants -- the migration is
function-preserving, so the cloned policy is unchanged by it.
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from common import (
    ACTIONS,
    FEATURE_VERSION,
    PLANE_SCALAR_DIM,
    PolicyValueNet,
    export_weights,
    load_trajectories,
    masked_argmax,
    masked_cross_entropy,
)


def evaluate(model: PolicyValueNet, X, mask, y, batch: int):
    """Return (mean loss, top-1 accuracy) over a held-out split."""
    n = X.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    total_loss = 0.0
    correct = 0
    for i in range(0, n, batch):
        xb = mx.array(X[i : i + batch])
        mb = mx.array(mask[i : i + batch])
        yb = mx.array(y[i : i + batch].astype(np.int32))
        out = model(xb)
        loss = masked_cross_entropy(out, mb, yb)
        pred = masked_argmax(out, mb)
        mx.eval(loss, pred)
        bs = xb.shape[0]
        total_loss += float(loss) * bs
        correct += int((np.array(pred) == y[i : i + batch]).sum())
    return total_loss / n, correct / n


def main() -> None:
    ap = argparse.ArgumentParser(description="Behavior cloning on mjgame trajectories")
    ap.add_argument("--data", required=True, help="glob for trajectory JSONL files")
    ap.add_argument("--out", required=True, help="output weights directory")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val", type=float, default=0.05, help="validation fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--save-every",
        type=int,
        default=0,
        metavar="N",
        help="also export weights every N epochs (0 = only at the end)",
    )
    args = ap.parse_args()

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    traj = load_trajectories(args.data)
    n = len(traj)
    n_matches = traj.net.shape[0]
    print(
        f"loaded {n} decisions from {n_matches} match(es), "
        f"{traj.round_deltas.shape[0]} round(s) in {time.time() - t0:.1f}s"
    )

    perm = rng.permutation(n)
    n_val = int(round(n * args.val))
    n_val = min(max(n_val, 1 if args.val > 0 and n > 1 else 0), n - 1 if n > 1 else 0)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Xtr, mtr, ytr = traj.X[train_idx], traj.mask[train_idx], traj.action[train_idx]
    Xva, mva, yva = traj.X[val_idx], traj.mask[val_idx], traj.action[val_idx]
    # The data decides the width; the constant only has to agree with it.  The
    # loader already rejects anything that is not the current feature version,
    # so a disagreement here means the constants drifted, not that the file is
    # stale.  The comparison is against the DENSE width (planes ++ scalars),
    # which is what `traj.X` holds in every feature version -- v4's token stream
    # lives in `traj.seq` and this trainer does not read it.
    input_dim = int(traj.X.shape[1])
    if input_dim != PLANE_SCALAR_DIM:
        raise SystemExit(
            f"data is {input_dim} wide but feature v{FEATURE_VERSION}'s planes ++ "
            f"scalars is {PLANE_SCALAR_DIM} -- common.py's contract constants and "
            "the engine's encoder disagree"
        )
    print(
        f"train {Xtr.shape[0]} / val {Xva.shape[0]}  "
        f"(input={input_dim}, actions={ACTIONS}, features=v{FEATURE_VERSION})"
    )

    model = PolicyValueNet(input_dim)
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=args.lr)

    def loss_fn(m, xb, mb, yb):
        return masked_cross_entropy(m(xb), mb, yb)

    grad_fn = nn.value_and_grad(model, loss_fn)

    ntr = Xtr.shape[0]
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(ntr)
        running, seen = 0.0, 0
        te = time.time()
        for i in range(0, ntr, args.batch):
            sel = order[i : i + args.batch]
            xb = mx.array(Xtr[sel])
            mb = mx.array(mtr[sel])
            yb = mx.array(ytr[sel].astype(np.int32))
            loss, grads = grad_fn(model, xb, mb, yb)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            bs = len(sel)
            running += float(loss) * bs
            seen += bs
        tr_loss = running / max(seen, 1)
        line = f"epoch {epoch:3d}/{args.epochs}  train_loss {tr_loss:.4f}"
        if Xva.shape[0]:
            va_loss, va_acc = evaluate(model, Xva, mva, yva, args.batch)
            line += f"  val_loss {va_loss:.4f}  val_top1 {va_acc * 100:.2f}%"
        else:
            line += "  val_loss n/a  val_top1 n/a"
        print(f"{line}  ({time.time() - te:.1f}s)")

        if args.save_every and epoch % args.save_every == 0 and epoch != args.epochs:
            export_weights(model, args.out)
            print(f"  checkpoint -> {args.out}")

    export_weights(model, args.out)
    print(f"wrote weights to {args.out}")


if __name__ == "__main__":
    main()
