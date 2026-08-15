"""Migrate a feature-v3 weight set to feature-v4, WITHOUT changing what it computes.

    python train/widen4.py --in weights --out weights-v4

v4 adds the ATTENTION RIVER ENCODER (train/V4_SPEC.md): a small self-attention
net reads the four rivers as a token stream and folds them into a 64-vector z,
which is CONCATENATED onto the end of the existing input.  Every v3 column keeps
its index, so a v3 net is exactly a v4 net whose 64 new columns are zero:

    policy  v3 (1674)                   v4 (1738)
      x[   0:1674]  planes ++ scalars    x[   0:1674]  (copied)
                                         x[1674:1738]  z  (ZERO)

    critic  v3 (1844)                   v4 (1908)
      x[   0:1674]  planes ++ scalars    x[   0:1674]  (copied)
      x[1674:1844]  oracle               x[1674:1844]  (copied)
                                         x[1844:1908]  z  (ZERO)

so for `fc1.weight`, which is [512, in] row-major, the whole surgery is

    new[:, :old_width] = old            # everything v3 had
    new[:, old_width:] = 0              # the 64 z columns

and fc1's bias, fc2, fc3, the auxiliary shanten rows and every critic layer
after fc1 are copied through BYTE-IDENTICALLY -- they never saw the input width.

THE ASYMMETRY IS THE DESIGN, NOT AN OVERSIGHT.  attn.f32 is written RANDOM --
every parameter ~ normal(0, 0.02) from --seed, u and the biases included -- while
the columns that consume its output are written ZERO.  Zeroing both sides would
look tidier and would be a dead saddle: the gradient reaching the encoder flows
through those fc1 columns (so it is 0 if they are 0) and the gradient on the
columns is proportional to z (so it is 0 if z is 0).  Exactly one side must be
nonzero for the path to wake up, and it has to be the encoder, because the zero
side is what makes the migration function-preserving.  z != 0 with zero consumer
columns gives BOTH: the output is unchanged for any input at all, and the column
gradients are alive from the first minibatch.  Do not "fix" this.

That the output is unchanged is not asserted by construction, it is MEASURED,
the same three ways `widen.py` measures it:

  * STRUCTURE: the v3 columns are bit-identical, the 64 new columns are exactly
    0.0, everything after fc1's weight matrix is unchanged, aux.f32 is a byte
    copy, and attn.f32 is the right size, all-finite and NOT all-zero.
  * EXACT: an IEEE-double reference forward of both nets differs by exactly 0.
    The v4 net is fed RANDOM values in the z columns -- that is the whole point:
    64 terms of 0 * (random) cannot move a running total, but 64 terms of
    (nonzero) * (random) would show up immediately.
  * RUNTIME: the mlx float32 forward agrees to `--tol` RELATIVE to the output
    magnitude.  Not absolute: float32 matmul is non-associative and mlx blocks a
    1738-wide reduction differently from a 1674-wide one, so the last ulp or two
    genuinely moves while the exact check above stays at 0.

The encoder's own forward is NOT exercised here, and that is not a gap: its
output is multiplied by zero columns, so no value it could produce changes the
migrated net's function.  What must hold is that the columns really are zero,
which is checked twice (structurally, and by feeding noise through them).

One-directional and guarded on both ends: a directory that is already v4 is
refused (re-running would re-zero columns the net has since learned) and so is
anything that is not recognisably v3.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import NamedTuple, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402

from common import (  # noqa: E402
    ATTN_BLOB_NAME,
    ATTN_BYTES,
    ATTN_D,
    ATTN_FLOATS,
    ATTN_TENSORS,
    AUX_BLOB_NAME,
    BLOB_NAME,
    CRITIC_BLOB_NAME,
    CRITIC_INPUT,
    CRITIC_MANIFEST_NAME,
    FEATURE_VERSION,
    HIDDEN1,
    INPUT_DIM,
    MANIFEST_NAME,
    ORACLE_LEN,
    PLANES,
    SCALAR_LEN,
    V3_CRITIC_INPUT,
    V3_INPUT_DIM,
    critic_manifest_for,
    load_attn,
    load_critic,
    load_weights,
    manifest_for,
)

VERIFY_SAMPLES = 256
VERIFY_TOL = 1e-5

# The init the spec freezes for the fresh encoder.  Not a CLI knob: all three
# implementations have to be able to reproduce a migration from --seed alone.
ATTN_INIT_STD = 0.02


class Refused(SystemExit):
    """A guard tripped: wrong input version, or already migrated.  Exit code 2."""

    def __init__(self, msg: str) -> None:
        super().__init__(2)
        self.msg = msg


# ---------------------------------------------------------------------------
# Blob surgery
# ---------------------------------------------------------------------------


def _read_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as e:
            raise Refused(f"{path}: bad JSON: {e}") from e


def append_columns(old: np.ndarray, new_width: int) -> np.ndarray:
    """Widen a [rows, w] matrix to [rows, new_width], the extra columns at 0.0."""
    if old.ndim != 2 or old.shape[1] >= new_width:
        raise ValueError(
            f"expected a [rows, <{new_width}] matrix, got {tuple(old.shape)}"
        )
    new = np.zeros((old.shape[0], new_width), dtype=old.dtype)
    new[:, : old.shape[1]] = old
    return new


def _widen_blob(blob: np.ndarray, old_width: int, new_width: int) -> np.ndarray:
    """Return the v4 blob for a v3 `blob`, touching fc1's weight and nothing else.

    The blob is [out][in] weights then [out] bias, per layer, in order; fc1's
    weight is therefore the leading `512 * in` floats and everything after it is
    copied through verbatim -- which is why the tail of the output file is
    byte-identical to the tail of the input file.
    """
    n_fc1_w = HIDDEN1 * old_width
    if blob.size < n_fc1_w:
        raise Refused(
            f"blob has {blob.size} floats, too small for a {old_width}-wide fc1"
        )
    fc1_w = blob[:n_fc1_w].reshape(HIDDEN1, old_width)
    rest = blob[n_fc1_w:]
    new_w = append_columns(np.ascontiguousarray(fc1_w, dtype=np.float32), new_width)
    return np.concatenate([new_w.reshape(-1), rest]).astype("<f4")


def widen_policy(indir: str, outdir: str) -> Tuple[int, int]:
    """policy.f32 + manifest.json: v3 -> v4.  Returns (old width, new width)."""
    mpath = os.path.join(indir, MANIFEST_NAME)
    if not os.path.exists(mpath):
        raise Refused(f"no {MANIFEST_NAME} in {indir} -- not a weight set")
    manifest = _read_manifest(mpath)

    feats = manifest.get("features")
    layers = manifest.get("layers") or []
    width = int(layers[0]["in"]) if layers and "in" in layers[0] else -1

    if width == INPUT_DIM or os.path.exists(os.path.join(indir, ATTN_BLOB_NAME)):
        raise Refused(
            f"{indir} is ALREADY feature v{FEATURE_VERSION} (input {width}"
            + (f", {ATTN_BLOB_NAME} present" if
               os.path.exists(os.path.join(indir, ATTN_BLOB_NAME)) else "")
            + ") -- nothing to widen.\n"
            "  Re-running the migration would re-zero the z columns the net has "
            "since learned AND throw away the trained encoder, so this is refused "
            "rather than made idempotent-by-copy."
        )
    if width != V3_INPUT_DIM or not isinstance(feats, dict) or (
        feats.get("planes") != PLANES or feats.get("scalars") != SCALAR_LEN
    ):
        raise Refused(
            f"{indir} is not a feature-v3 weight set: features={feats!r}, "
            f"input width {width}; expected "
            f"{{'planes': {PLANES}, 'scalars': {SCALAR_LEN}}} / {V3_INPUT_DIM}.\n"
            "  Only v3 -> v4 is implemented here; a v2 set goes through "
            "train/widen.py first."
        )

    blob_name = manifest.get("blob", BLOB_NAME)
    bpath = os.path.join(indir, blob_name)
    if not os.path.exists(bpath):
        raise Refused(f"no {blob_name} next to {mpath}")
    blob = np.fromfile(bpath, dtype="<f4")

    new_blob = _widen_blob(blob, V3_INPUT_DIM, INPUT_DIM)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, BLOB_NAME), "wb") as fh:
        fh.write(new_blob.tobytes())
    with open(os.path.join(outdir, MANIFEST_NAME), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest_for(INPUT_DIM), separators=(",", ":")))
    return V3_INPUT_DIM, INPUT_DIM


def write_attn(outdir: str, seed: int) -> str:
    """Write a fresh random attn.f32 -- see the module docstring on the asymmetry.

    Deliberately NOT `common.random_attn` + `export_attn`: this writes the file
    tensor by tensor in `ATTN_TENSORS` order so the migration's output depends
    only on numpy's stream and the spec's order, not on how the trainer's module
    happens to lay its parameters out today.  The verifier then reads it back
    with the trainer's own loader, which is what ties the two together.
    """
    rng = np.random.default_rng(seed)
    chunks = [
        rng.normal(0.0, ATTN_INIT_STD, size=shape).astype("<f4").reshape(-1)
        for _, shape in ATTN_TENSORS
    ]
    blob = np.concatenate(chunks).astype("<f4")
    if blob.size != ATTN_FLOATS:
        raise SystemExit(f"attn blob has {blob.size} floats, expected {ATTN_FLOATS}")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, ATTN_BLOB_NAME)
    with open(path, "wb") as fh:
        fh.write(blob.tobytes())
    return path


def copy_aux(indir: str, outdir: str) -> bool:
    """aux.f32 rides on fc3 ([24, 256]), so migration is a byte copy."""
    src = os.path.join(indir, AUX_BLOB_NAME)
    if not os.path.exists(src):
        return False
    os.makedirs(outdir, exist_ok=True)
    shutil.copyfile(src, os.path.join(outdir, AUX_BLOB_NAME))
    return True


def widen_critic(indir: str, outdir: str) -> Optional[Tuple[int, int]]:
    """critic.f32 + critic.json: v3 -> v4, or None when the dir has no critic."""
    mpath = os.path.join(indir, CRITIC_MANIFEST_NAME)
    if not os.path.exists(mpath):
        return None
    manifest = _read_manifest(mpath)

    feats = manifest.get("features")
    layers = manifest.get("layers") or []
    width = int(layers[0]["in"]) if layers and "in" in layers[0] else -1

    if width == CRITIC_INPUT:
        raise Refused(
            f"{mpath} is already feature v{FEATURE_VERSION} (input {width})"
        )
    if width != V3_CRITIC_INPUT or not isinstance(feats, dict) or (
        feats.get("planes") != PLANES or feats.get("scalars") != SCALAR_LEN
    ):
        raise Refused(
            f"{mpath} is not a feature-v3 critic: features={feats!r}, input width "
            f"{width}; expected {V3_CRITIC_INPUT} "
            f"({V3_INPUT_DIM} policy + oracle {ORACLE_LEN})"
        )

    blob_name = manifest.get("blob", CRITIC_BLOB_NAME)
    bpath = os.path.join(indir, blob_name)
    if not os.path.exists(bpath):
        raise Refused(f"no {blob_name} next to {mpath}")
    blob = np.fromfile(bpath, dtype="<f4")

    new_blob = _widen_blob(blob, V3_CRITIC_INPUT, CRITIC_INPUT)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, CRITIC_BLOB_NAME), "wb") as fh:
        fh.write(new_blob.tobytes())
    with open(os.path.join(outdir, CRITIC_MANIFEST_NAME), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(critic_manifest_for(CRITIC_INPUT), separators=(",", ":")))
    return V3_CRITIC_INPUT, CRITIC_INPUT


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class Report(NamedTuple):
    """One net's verification result; `ok` is the AND of all three checks."""

    ok: bool
    structure: str        # "" when clean, else what is wrong
    exact: float          # max |Δ| of the float64 reference forward
    runtime: float        # max |Δ| of the mlx float32 forward
    scale: float          # max |output| of the original net, for the relative bound


def _forward(net, x: np.ndarray) -> np.ndarray:
    out = net(mx.array(np.ascontiguousarray(x, dtype=np.float32)))
    mx.eval(out)
    return np.array(out, copy=True).reshape(x.shape[0], -1)


def _forward64(net, x: np.ndarray) -> np.ndarray:
    """The same MLP in IEEE double, where the added 0-products are provably inert."""
    h = np.asarray(x, dtype=np.float64)
    layers = net.ordered_layers
    # errstate: numpy 2.x on Accelerate raises spurious divide/overflow/invalid
    # warnings out of BLAS matmul (the accumulator leaves FP exception flags set
    # even when every operand and every result is finite -- checked).  Muted here
    # so a clean PASS does not print three lines of noise; the finiteness that
    # actually matters is asserted just below.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for i, layer in enumerate(layers):
            w = np.array(layer.weight, copy=True).astype(np.float64)
            b = np.array(layer.bias, copy=True).astype(np.float64)
            h = h @ w.T + b
            if i < len(layers) - 1:
                h = np.maximum(h, 0.0)
    if not np.all(np.isfinite(h)):
        raise SystemExit("verify: float64 reference forward produced a non-finite value")
    return h.reshape(x.shape[0], -1)


def _check_structure(old, new, old_width: int, new_width: int) -> str:
    """Bit-level check of the column map on fc1 and of every other parameter."""
    w3 = np.array(old.fc1.weight, copy=True)
    w4 = np.array(new.fc1.weight, copy=True)
    problems = []
    if w4.shape != (HIDDEN1, new_width):
        problems.append(f"fc1 is {w4.shape}, expected {(HIDDEN1, new_width)}")
        return "; ".join(problems)
    if not np.array_equal(w4[:, :old_width], w3):
        problems.append(f"columns 0..{old_width - 1} not copied bit-exactly")
    nz = int(np.count_nonzero(w4[:, old_width:]))
    if nz:
        problems.append(
            f"{nz} non-zero weight(s) in the {new_width - old_width} new z columns"
        )

    for name, a, b in [
        ("fc1.bias", old.fc1.bias, new.fc1.bias),
        ("fc2.weight", old.fc2.weight, new.fc2.weight),
        ("fc2.bias", old.fc2.bias, new.fc2.bias),
        ("fc3.weight", old.fc3.weight, new.fc3.weight),
        ("fc3.bias", old.fc3.bias, new.fc3.bias),
    ]:
        if not np.array_equal(np.array(a, copy=True), np.array(b, copy=True)):
            problems.append(f"{name} changed")
    return "; ".join(problems)


def _verify(old, new, x_old: np.ndarray, x_new: np.ndarray, tol: float) -> Report:
    structure = _check_structure(old, new, x_old.shape[1], x_new.shape[1])
    exact = float(np.max(np.abs(_forward64(old, x_old) - _forward64(new, x_new))))
    o32 = _forward(old, x_old)
    runtime = float(np.max(np.abs(o32 - _forward(new, x_new))))
    scale = float(np.max(np.abs(o32)))
    ok = (not structure) and exact == 0.0 and runtime <= tol * max(1.0, scale)
    return Report(ok, structure, exact, runtime, scale)


def _embed(x3: np.ndarray, rng) -> np.ndarray:
    """v3 rows ++ RANDOM z, which is what makes a non-zero column visible."""
    z = rng.standard_normal((x3.shape[0], ATTN_D)).astype(np.float32)
    return np.concatenate([x3, z], axis=1)


def verify_policy(indir: str, outdir: str, rng, tol: float) -> Report:
    """Old net on v3 inputs vs new net on the same inputs ++ noise in the z block."""
    old = load_weights(indir)
    new = load_weights(outdir)
    if old.input_dim != V3_INPUT_DIM or new.input_dim != INPUT_DIM:
        raise SystemExit(
            f"verify: widths are {old.input_dim} -> {new.input_dim}, expected "
            f"{V3_INPUT_DIM} -> {INPUT_DIM}"
        )
    x3 = rng.standard_normal((VERIFY_SAMPLES, V3_INPUT_DIM)).astype(np.float32)
    return _verify(old, new, x3, _embed(x3, rng), tol)


def verify_critic(indir: str, outdir: str, rng, tol: float) -> Report:
    """Same, on the critic: its oracle block keeps its place, z lands after it."""
    old = load_critic(indir)
    new = load_critic(outdir)
    if old is None or new is None:
        raise SystemExit("verify: critic disappeared between write and read")
    if old.input_dim != V3_CRITIC_INPUT or new.input_dim != CRITIC_INPUT:
        raise SystemExit(
            f"verify: critic widths are {old.input_dim} -> {new.input_dim}, expected "
            f"{V3_CRITIC_INPUT} -> {CRITIC_INPUT}"
        )
    c3 = rng.standard_normal((VERIFY_SAMPLES, V3_CRITIC_INPUT)).astype(np.float32)
    return _verify(old, new, c3, _embed(c3, rng), tol)


def verify_attn(outdir: str) -> str:
    """"" when the written encoder is loadable, finite and awake, else what is wrong.

    "Awake" is the migration-specific claim: an all-zero encoder would pass every
    forward-equality check above (its output is multiplied by zero columns) and
    would leave the whole v4 path at the dead saddle the docstring describes.
    """
    path = os.path.join(outdir, ATTN_BLOB_NAME)
    size = os.path.getsize(path) if os.path.exists(path) else -1
    if size != ATTN_BYTES:
        return f"{ATTN_BLOB_NAME} is {size} bytes, expected {ATTN_BYTES}"
    enc = load_attn(outdir)
    if enc is None:
        return f"{ATTN_BLOB_NAME} vanished between write and read"
    problems = []
    for (name, shape), t in zip(ATTN_TENSORS, enc.ordered_tensors):
        a = np.array(t, copy=True)
        if a.shape != shape:
            problems.append(f"{name} is {a.shape}, expected {shape}")
            continue
        if not np.all(np.isfinite(a)):
            problems.append(f"{name} has non-finite values")
        if not np.any(a):
            problems.append(f"{name} is all zeros (the dead-saddle init)")
    return "; ".join(problems)


def verify_aux(indir: str, outdir: str) -> str:
    """aux.f32 must be a BYTE copy: the heads live on fc3 and never saw the width."""
    src = os.path.join(indir, AUX_BLOB_NAME)
    dst = os.path.join(outdir, AUX_BLOB_NAME)
    if not os.path.exists(src):
        return ""
    if not os.path.exists(dst):
        return f"{AUX_BLOB_NAME} was not copied"
    with open(src, "rb") as a, open(dst, "rb") as b:
        if a.read() != b.read():
            return f"{AUX_BLOB_NAME} differs from the source"
    return ""


def _report(name: str, what: str, rep: Report, tol: float) -> None:
    bound = tol * max(1.0, rep.scale)
    rel = rep.runtime / rep.scale if rep.scale else 0.0
    print(
        f"{'PASS' if rep.ok else 'FAIL'} {name}: exact(f64) max|Δ| = {rep.exact:g}  "
        f"runtime(f32) max|Δ| = {rep.runtime:.3e} over {VERIFY_SAMPLES} random inputs "
        f"({what}, |out| <= {rep.scale:.1f}, {rel:.1e} relative, bound {bound:.3e})"
    )
    if rep.structure:
        print(f"     structure: {rep.structure}")
    elif not rep.ok:
        print("     structure: clean -- the failure is numerical, not a bad column map")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            f"Widen a feature-v3 weight set to feature-v{FEATURE_VERSION} "
            f"({V3_INPUT_DIM} -> {INPUT_DIM} policy inputs, {V3_CRITIC_INPUT} -> "
            f"{CRITIC_INPUT} critic inputs) and write a fresh random "
            f"{ATTN_BLOB_NAME}, without changing the function it computes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"The {ATTN_D} new input columns are written as exact zeros and verified\n"
            f"as such by forwarding random batches through both nets; the encoder\n"
            f"itself is written RANDOM (normal(0, {ATTN_INIT_STD}) from --seed) on\n"
            "purpose -- see the module docstring.  A mismatch exits 1, a\n"
            "wrong-version or already-migrated input directory exits 2."
        ),
    )
    ap.add_argument("--in", dest="indir", required=True, help="v3 weights directory")
    ap.add_argument("--out", dest="outdir", required=True, help="v4 weights directory")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed: fixes BOTH the new encoder's weights and the verification "
        "batch (independent streams, so one does not shift the other)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=VERIFY_TOL,
        help=(
            f"output difference accepted from the float32 runtime, RELATIVE to the "
            f"output magnitude (default {VERIFY_TOL:g}); the float64 reference "
            f"forward must match exactly regardless"
        ),
    )
    args = ap.parse_args()

    indir, outdir = args.indir, args.outdir
    if not os.path.isdir(indir):
        raise Refused(f"--in {indir} is not a directory")
    if os.path.abspath(indir) == os.path.abspath(outdir):
        raise Refused(
            "--in and --out are the same directory; widening in place would "
            "destroy the only copy of the v3 net if verification failed"
        )

    # Two independent streams off one seed: the weights the migration writes must
    # not move when the verification batch does, and vice versa.
    seed_w, seed_v = np.random.SeedSequence(args.seed).spawn(2)

    old_w, new_w = widen_policy(indir, outdir)
    print(
        f"policy: fc1 [{HIDDEN1}, {old_w}] -> [{HIDDEN1}, {new_w}]  "
        f"(z columns {old_w}:{new_w} = 0); fc1.bias/fc2/fc3 copied verbatim"
    )

    write_attn(outdir, seed_w)
    print(
        f"attn:   {ATTN_BLOB_NAME} written fresh, {ATTN_FLOATS} floats "
        f"({ATTN_BYTES} bytes), all ~ normal(0, {ATTN_INIT_STD}) from --seed "
        f"{args.seed}"
    )

    if copy_aux(indir, outdir):
        print(f"aux:    {AUX_BLOB_NAME} copied (rows live on fc3, unaffected by width)")
    else:
        print(f"aux:    no {AUX_BLOB_NAME} in {indir} -- skipped")

    crit = widen_critic(indir, outdir)
    if crit is not None:
        print(
            f"critic: fc1 [{HIDDEN1}, {crit[0]}] -> [{HIDDEN1}, {crit[1]}]  "
            f"(oracle block stays at {V3_INPUT_DIM}:{V3_CRITIC_INPUT}, z appended "
            f"at {V3_CRITIC_INPUT}:{CRITIC_INPUT})"
        )
    else:
        print(f"critic: no {CRITIC_MANIFEST_NAME} in {indir} -- skipped")

    rng = np.random.default_rng(seed_v)
    failed = False

    rep = verify_policy(indir, outdir, rng, args.tol)
    _report("policy", "logits+value", rep, args.tol)
    failed |= not rep.ok

    if crit is not None:
        rep = verify_critic(indir, outdir, rng, args.tol)
        _report("critic", "value", rep, args.tol)
        failed |= not rep.ok

    bad = verify_attn(outdir)
    print(
        f"{'FAIL' if bad else 'PASS'} attn:   {len(ATTN_TENSORS)} tensors reload "
        f"through common.load_attn, finite and non-zero"
        + (f"\n     {bad}" if bad else "")
    )
    failed |= bool(bad)

    bad = verify_aux(indir, outdir)
    if os.path.exists(os.path.join(indir, AUX_BLOB_NAME)):
        print(f"{'FAIL' if bad else 'PASS'} aux:    byte-identical copy"
              + (f"\n     {bad}" if bad else ""))
    failed |= bool(bad)

    if failed:
        print(
            "verification FAILED -- the widened net does not compute the same "
            f"function; {outdir} is left in place for inspection but MUST NOT be "
            "trained from",
            file=sys.stderr,
        )
        return 1
    print(f"wrote v{FEATURE_VERSION} weights to {outdir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as e:
        print(f"widen4.py: {e.msg}", file=sys.stderr)
        sys.exit(e.code)
