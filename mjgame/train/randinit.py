#!/usr/bin/env python3
"""Export an untrained, randomly initialized policy.

    python randinit.py --out weights/

Lets the TypeScript side smoke-test its inference path (manifest parsing,
blob layout, masked argmax) before any trajectory data exists.

From feature v4 on that is TWO blobs: policy.f32 (1738 inputs = the dense
features ++ the river encoder's 64-wide z) and attn.f32 (the encoder itself).
Both are random here, which is the point -- the smoke test is of the loader and
the shapes, not of the play.
"""

from __future__ import annotations

import argparse
import os

import mlx.core as mx

from common import (
    ATTN_BLOB_NAME,
    ATTN_BYTES,
    BLOB_BYTES,
    BLOB_NAME,
    FEATURE_VERSION,
    INPUT_DIM,
    MANIFEST_NAME,
    PolicyValueNet,
    export_attn,
    export_weights,
    random_attn,
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
    export_attn(random_attn(args.seed), args.out)

    size = os.path.getsize(os.path.join(args.out, BLOB_NAME))
    assert size == BLOB_BYTES, f"{BLOB_NAME} is {size} bytes, expected {BLOB_BYTES}"
    asize = os.path.getsize(os.path.join(args.out, ATTN_BLOB_NAME))
    assert asize == ATTN_BYTES, f"{ATTN_BLOB_NAME} is {asize} bytes, expected {ATTN_BYTES}"
    print(
        f"wrote {os.path.join(args.out, MANIFEST_NAME)} "
        f"(feature v{FEATURE_VERSION}, input {INPUT_DIM})"
    )
    print(f"wrote {os.path.join(args.out, BLOB_NAME)} ({size} bytes)")
    print(f"wrote {os.path.join(args.out, ATTN_BLOB_NAME)} ({asize} bytes)")


if __name__ == "__main__":
    main()
