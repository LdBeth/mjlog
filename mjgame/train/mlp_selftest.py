#!/usr/bin/env python3
"""Write `test/fixtures/mlp-parity.json` — the trainer's half of the MLP contract.

Three deterministic random nets (numpy RNG, fixed seed) and a handful of random
input vectors, run through `common.mlp_forward_np` — the explicit-loop
double-accumulate / float32-store reference.  `test/mlp_test.ts` replays the
fixture through `mlpForward` and demands EXACT equality, so this file is what
pins the Python side of the pipeline to the engine that consumes its weights.

Everything is float32 before anything is computed: the weights and the inputs
are generated as float64, immediately cast, and BOTH the JSON and the reference
consume the cast values.  A fixture whose reference saw numbers the JSON does
not carry would prove nothing.

    train/.venv/bin/python train/mlp_selftest.py [--out PATH]

Only numpy is used here (nothing calls MLX), but `common` imports mlx at module
level, so the venv still needs it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import export_mlp_block, mlp_forward_np  # noqa: E402

# (dims, acts) — small on purpose: the fixture is committed, and 8 inputs is
# already enough to catch a reordered sum.
NETS = [
    ([4, 1], ["none"]),
    ([6, 5, 3], ["relu", "none"]),
    ([8, 7, 6, 2], ["relu", "relu", "none"]),
]
N_INPUTS = 5
SEED = 20260829
FV = 1


def build(rng: np.random.Generator, dims, acts) -> dict:
    layers = []
    for i in range(len(dims) - 1):
        n_in, n_out = dims[i], dims[i + 1]
        # A spread of magnitudes: a scale of 1 keeps relu killing about half the
        # units, and the bias offset keeps the "none" layer off zero.
        w = (rng.standard_normal((n_out, n_in)) * 1.0).astype(np.float32)
        b = (rng.standard_normal(n_out) * 0.5).astype(np.float32)
        layers.append((w, b))
    return export_mlp_block(layers, acts, FV)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(here, os.pardir, "test", "fixtures", "mlp-parity.json")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.normpath(default))
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    nets = []
    for dims, acts in NETS:
        spec = build(rng, dims, acts)
        xs = (rng.standard_normal((N_INPUTS, dims[0])) * 1.5).astype(np.float32)
        ys = np.asarray([mlp_forward_np(spec, x) for x in xs], dtype=np.float32)
        # Round-trip check: what the JSON carries is what the reference ran on.
        again = np.asarray(
            [mlp_forward_np(json.loads(json.dumps(spec)), x) for x in xs],
            dtype=np.float32,
        )
        assert np.array_equal(ys.view(np.int32), again.view(np.int32)), "json round-trip differs"
        nets.append(
            {
                "spec": spec,
                "inputs": [[float(v) for v in x] for x in xs],
                "outputs": [[float(v) for v in y] for y in ys],
            }
        )

    doc = {
        "note": "written by train/mlp_selftest.py; float32-exact, see src/ai/mlp.ts",
        "seed": SEED,
        "nets": nets,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, indent=1))
        fh.write("\n")
    total = sum(len(n["inputs"]) for n in nets)
    print(f"{args.out}: {len(nets)} nets, {total} vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
