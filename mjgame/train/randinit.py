#!/usr/bin/env python3
"""Export an untrained, randomly initialized policy.

    python randinit.py --out weights/

Lets the TypeScript side smoke-test its inference path (manifest parsing,
blob layout, masked argmax) before any trajectory data exists.
"""

from __future__ import annotations

import argparse
import os

import mlx.core as mx

from common import (
    BLOB_BYTES,
    BLOB_NAME,
    FEATURE_VERSION,
    INPUT_DIM,
    MANIFEST_NAME,
    PolicyValueNet,
    export_weights,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a random-init mjgame policy")
    ap.add_argument("--out", required=True, help="output weights directory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    model = PolicyValueNet()
    mx.eval(model.parameters())
    export_weights(model, args.out)

    size = os.path.getsize(os.path.join(args.out, BLOB_NAME))
    assert size == BLOB_BYTES, f"{BLOB_NAME} is {size} bytes, expected {BLOB_BYTES}"
    print(
        f"wrote {os.path.join(args.out, MANIFEST_NAME)} "
        f"(feature v{FEATURE_VERSION}, input {INPUT_DIM})"
    )
    print(f"wrote {os.path.join(args.out, BLOB_NAME)} ({size} bytes)")


if __name__ == "__main__":
    main()
