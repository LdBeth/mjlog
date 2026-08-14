"""Migrate a feature-v2 weight set to feature-v3, WITHOUT changing what it computes.

    python train/widen.py --in weights --out weights-v3

v3 adds twelve planes and three scalars to the encoder.  Every v2 feature keeps
its meaning and its position WITHIN ITS SECTION, so a v2 net is exactly a v3 net
whose new input columns are zero: widening is pure surgery on the first layer's
weight matrix, and nothing else in the blob moves at all.

The input layouts, side by side:

    v2 (1263)                       v3 (1674)
    x[   0:1224]  planes 0..35  ->  x[   0:1224]  planes 0..35   (copied)
                                    x[1224:1632]  planes 36..47  (ZERO)
    x[1224:1263]  scalars 0..38 ->  x[1632:1671]  scalars 0..38  (copied)
                                    x[1671:1674]  scalars 39..41 (ZERO)

and for the oracle critic, whose input is the policy input ++ 170 oracle values:

    v2 (1433)                       v3 (1844)
    x[   0:1263]  policy       ->   the map above
    x[1263:1433]  oracle       ->   x[1674:1844]                 (copied)

so for `fc1.weight`, which is [512, in] row-major,

    new[:,    0:1224] = old[:,    0:1224]      # v2 planes
    new[:, 1224:1632] = 0                      # planes 36..47
    new[:, 1632:1671] = old[:, 1224:1263]      # v2 scalars
    new[:, 1671:1674] = 0                      # scalars 39..41
    new[:, 1674:1844] = old[:, 1263:1433]      # oracle (critic only)

fc1's bias, fc2, fc3 and the auxiliary shanten rows are copied through
BYTE-IDENTICALLY -- they never saw the input width.

Because the added columns are exactly zero, the widened net's output for an
embedded v2 input equals the original's for any values at all in the new
positions.  That is not asserted by construction, it is MEASURED: the tool
forwards 256 random inputs through both nets -- filling the new positions with
random values precisely so that a non-zero column would show up -- and refuses
to report success unless three things hold.

  * STRUCTURE: the mapped columns are bit-identical to the originals, the new
    columns are exactly 0.0, and every blob byte after fc1's weight matrix is
    unchanged.
  * EXACT: an IEEE-double reference forward of both nets differs by exactly 0.
    This is the real claim -- summing 411 extra terms that are all exactly
    0 * x = 0 cannot move a running total.
  * RUNTIME: the mlx float32 forward agrees to `--tol` RELATIVE to the output
    magnitude.  Not absolute: float32 matmul is non-associative and mlx blocks
    a 1674-wide reduction differently from a 1263-wide one, so the last couple
    of ulps genuinely move.  On the seeded v2 set that is ~1e-4 absolute on
    logits of magnitude ~400 -- 2.7e-7 relative, i.e. two ulps, while the exact
    check above is 0.  A real bug (a column that is not zero) is many orders of
    magnitude bigger than that and trips every one of the three.

The tool is deliberately one-directional and guarded on both ends: it refuses a
directory that is already v3 (so re-running a migration script cannot zero out
learned columns) and refuses anything that is not recognisably v2.
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
    PLANE_LEN,
    PLANES,
    SCALAR_LEN,
    TILE_TYPES,
    V2_CRITIC_INPUT,
    V2_INPUT_DIM,
    V2_PLANE_FLOATS,
    V2_SCALAR_FLOATS,
    critic_manifest_for,
    load_critic,
    load_weights,
    manifest_for,
)

# The v2 feature block, spelled out so "is this v2?" is a comparison and not a
# guess.  36 planes x 34 + 39 scalars = 1263.
V2_PLANES = 36
V2_SCALARS = V2_SCALAR_FLOATS  # 39

VERIFY_SAMPLES = 256
VERIFY_TOL = 1e-5


class Refused(SystemExit):
    """A guard tripped: wrong input version, or already migrated.  Exit code 2."""

    def __init__(self, msg: str) -> None:
        super().__init__(2)
        self.msg = msg


# ---------------------------------------------------------------------------
# The column map
# ---------------------------------------------------------------------------


def embed_columns(old: np.ndarray, oracle: bool) -> np.ndarray:
    """Widen a [rows, v2_width] matrix to [rows, v3_width] along axis 1.

    Column-for-column the mapping described in the module docstring; every
    column of the result that is not fed by `old` is left at exactly 0.0.
    """
    v2_width = V2_CRITIC_INPUT if oracle else V2_INPUT_DIM
    v3_width = CRITIC_INPUT if oracle else INPUT_DIM
    if old.ndim != 2 or old.shape[1] != v2_width:
        raise ValueError(
            f"expected a [rows, {v2_width}] matrix, got {tuple(old.shape)}"
        )
    new = np.zeros((old.shape[0], v3_width), dtype=old.dtype)
    new[:, 0:V2_PLANE_FLOATS] = old[:, 0:V2_PLANE_FLOATS]
    new[:, PLANE_LEN : PLANE_LEN + V2_SCALARS] = old[:, V2_PLANE_FLOATS:V2_INPUT_DIM]
    if oracle:
        new[:, INPUT_DIM : INPUT_DIM + ORACLE_LEN] = old[:, V2_INPUT_DIM:V2_CRITIC_INPUT]
    return new


def embed_inputs(x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    """Place v2 input rows into v3 rows, keeping whatever `x3` holds elsewhere.

    Same map as `embed_columns` but along the FEATURE axis of a batch, and
    non-destructive in the new positions: the verifier hands in random noise
    there on purpose, so that any column the surgery failed to zero shows up as
    a difference instead of hiding behind a convenient 0.
    """
    out = np.array(x3, dtype=np.float32, copy=True)
    out[:, 0:V2_PLANE_FLOATS] = x2[:, 0:V2_PLANE_FLOATS]
    out[:, PLANE_LEN : PLANE_LEN + V2_SCALARS] = x2[:, V2_PLANE_FLOATS:V2_INPUT_DIM]
    return out


# ---------------------------------------------------------------------------
# Blob surgery
# ---------------------------------------------------------------------------


def _read_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as e:
            raise Refused(f"{path}: bad JSON: {e}") from e


def _widen_blob(blob: np.ndarray, oracle: bool) -> np.ndarray:
    """Return the v3 blob for a v2 `blob`, touching fc1's weight and nothing else.

    The blob is [out][in] weights then [out] bias, per layer, in order; fc1's
    weight is therefore the leading `512 * in` floats and everything after it is
    copied through verbatim -- which is why the tail of the output file is
    byte-identical to the tail of the input file.
    """
    v2_width = V2_CRITIC_INPUT if oracle else V2_INPUT_DIM
    n_fc1_w = HIDDEN1 * v2_width
    if blob.size < n_fc1_w:
        raise Refused(f"blob has {blob.size} floats, too small for a {v2_width}-wide fc1")
    fc1_w = blob[:n_fc1_w].reshape(HIDDEN1, v2_width)
    rest = blob[n_fc1_w:]
    new_w = embed_columns(np.ascontiguousarray(fc1_w, dtype=np.float32), oracle)
    return np.concatenate([new_w.reshape(-1), rest]).astype("<f4")


def widen_policy(indir: str, outdir: str) -> Tuple[int, int]:
    """policy.f32 + manifest.json: v2 -> v3.  Returns (old width, new width)."""
    mpath = os.path.join(indir, MANIFEST_NAME)
    if not os.path.exists(mpath):
        raise Refused(f"no {MANIFEST_NAME} in {indir} -- not a weight set")
    manifest = _read_manifest(mpath)

    feats = manifest.get("features")
    layers = manifest.get("layers") or []
    width = int(layers[0]["in"]) if layers and "in" in layers[0] else -1

    if width == INPUT_DIM or (isinstance(feats, dict) and feats.get("planes") == PLANES):
        raise Refused(
            f"{indir} is ALREADY feature v{FEATURE_VERSION} "
            f"({feats} / input {width}) -- nothing to widen.\n"
            "  Re-running the migration would re-zero columns the net has since "
            "learned, so this is refused rather than made idempotent-by-copy."
        )
    if width != V2_INPUT_DIM or not isinstance(feats, dict) or (
        feats.get("planes") != V2_PLANES or feats.get("scalars") != V2_SCALARS
    ):
        raise Refused(
            f"{indir} is not a feature-v2 weight set: features={feats!r}, "
            f"input width {width}; expected "
            f"{{'planes': {V2_PLANES}, 'scalars': {V2_SCALARS}}} / {V2_INPUT_DIM}.\n"
            "  Only v2 -> v3 is implemented; anything older has to be retrained."
        )

    blob_name = manifest.get("blob", BLOB_NAME)
    bpath = os.path.join(indir, blob_name)
    if not os.path.exists(bpath):
        raise Refused(f"no {blob_name} next to {mpath}")
    blob = np.fromfile(bpath, dtype="<f4")

    new_blob = _widen_blob(blob, oracle=False)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, BLOB_NAME), "wb") as fh:
        fh.write(new_blob.tobytes())
    with open(os.path.join(outdir, MANIFEST_NAME), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest_for(INPUT_DIM), separators=(",", ":")))
    return V2_INPUT_DIM, INPUT_DIM


def copy_aux(indir: str, outdir: str) -> bool:
    """aux.f32 rides on fc3 ([24, 256]), so migration is a byte copy."""
    src = os.path.join(indir, AUX_BLOB_NAME)
    if not os.path.exists(src):
        return False
    os.makedirs(outdir, exist_ok=True)
    shutil.copyfile(src, os.path.join(outdir, AUX_BLOB_NAME))
    return True


def widen_critic(indir: str, outdir: str) -> Optional[Tuple[int, int]]:
    """critic.f32 + critic.json: v2 -> v3, or None when the dir has no critic."""
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
    if width != V2_CRITIC_INPUT or not isinstance(feats, dict) or (
        feats.get("planes") != V2_PLANES or feats.get("scalars") != V2_SCALARS
    ):
        raise Refused(
            f"{mpath} is not a feature-v2 critic: features={feats!r}, input width "
            f"{width}; expected {V2_CRITIC_INPUT} "
            f"({V2_PLANES}x{TILE_TYPES} + {V2_SCALARS} + oracle {ORACLE_LEN})"
        )

    blob_name = manifest.get("blob", CRITIC_BLOB_NAME)
    bpath = os.path.join(indir, blob_name)
    if not os.path.exists(bpath):
        raise Refused(f"no {blob_name} next to {mpath}")
    blob = np.fromfile(bpath, dtype="<f4")

    new_blob = _widen_blob(blob, oracle=True)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, CRITIC_BLOB_NAME), "wb") as fh:
        fh.write(new_blob.tobytes())
    with open(os.path.join(outdir, CRITIC_MANIFEST_NAME), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(critic_manifest_for(CRITIC_INPUT), separators=(",", ":")))
    return V2_CRITIC_INPUT, CRITIC_INPUT


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
    # actually matters is asserted by the caller comparing exact zeros.
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


def _check_structure(old, new, oracle: bool) -> str:
    """Bit-level check of the column map on fc1 and of every other parameter."""
    w2 = np.array(old.fc1.weight, copy=True)
    w3 = np.array(new.fc1.weight, copy=True)
    problems = []
    if not np.array_equal(w3[:, 0:V2_PLANE_FLOATS], w2[:, 0:V2_PLANE_FLOATS]):
        problems.append("planes 0..35 columns not copied bit-exactly")
    if not np.array_equal(
        w3[:, PLANE_LEN : PLANE_LEN + V2_SCALARS], w2[:, V2_PLANE_FLOATS:V2_INPUT_DIM]
    ):
        problems.append("v2 scalar columns not copied bit-exactly")
    nz = int(np.count_nonzero(w3[:, V2_PLANE_FLOATS:PLANE_LEN]))
    if nz:
        problems.append(f"{nz} non-zero weight(s) in the new plane columns")
    nz = int(np.count_nonzero(w3[:, PLANE_LEN + V2_SCALARS : INPUT_DIM]))
    if nz:
        problems.append(f"{nz} non-zero weight(s) in the new scalar columns")
    if oracle and not np.array_equal(
        w3[:, INPUT_DIM:CRITIC_INPUT], w2[:, V2_INPUT_DIM:V2_CRITIC_INPUT]
    ):
        problems.append("oracle columns not copied bit-exactly")

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


def _verify(old, new, x_old: np.ndarray, x_new: np.ndarray, oracle: bool, tol: float):
    structure = _check_structure(old, new, oracle)
    exact = float(np.max(np.abs(_forward64(old, x_old) - _forward64(new, x_new))))
    o32 = _forward(old, x_old)
    runtime = float(np.max(np.abs(o32 - _forward(new, x_new))))
    scale = float(np.max(np.abs(o32)))
    ok = (not structure) and exact == 0.0 and runtime <= tol * max(1.0, scale)
    return Report(ok, structure, exact, runtime, scale)


def verify_policy(indir: str, outdir: str, rng, tol: float) -> Report:
    """Old net on v2 inputs vs new net on the same inputs EMBEDDED into v3.

    The new input positions are filled with RANDOM values, not zeros: that is
    the whole point of the test.  If any of the 411 added columns were non-zero
    the noise would flow straight through into the logits.
    """
    old = load_weights(indir)
    new = load_weights(outdir)
    if old.input_dim != V2_INPUT_DIM or new.input_dim != INPUT_DIM:
        raise SystemExit(
            f"verify: widths are {old.input_dim} -> {new.input_dim}, expected "
            f"{V2_INPUT_DIM} -> {INPUT_DIM}"
        )
    x2 = rng.standard_normal((VERIFY_SAMPLES, V2_INPUT_DIM)).astype(np.float32)
    noise = rng.standard_normal((VERIFY_SAMPLES, INPUT_DIM)).astype(np.float32)
    return _verify(old, new, x2, embed_inputs(x2, noise), False, tol)


def verify_critic(indir: str, outdir: str, rng, tol: float) -> Report:
    """Same, with one shared random oracle block appended on both sides."""
    old = load_critic(indir)
    new = load_critic(outdir)
    if old is None or new is None:
        raise SystemExit("verify: critic disappeared between write and read")
    if old.input_dim != V2_CRITIC_INPUT or new.input_dim != CRITIC_INPUT:
        raise SystemExit(
            f"verify: critic widths are {old.input_dim} -> {new.input_dim}, expected "
            f"{V2_CRITIC_INPUT} -> {CRITIC_INPUT}"
        )
    x2 = rng.standard_normal((VERIFY_SAMPLES, V2_INPUT_DIM)).astype(np.float32)
    noise = rng.standard_normal((VERIFY_SAMPLES, INPUT_DIM)).astype(np.float32)
    oracle = rng.standard_normal((VERIFY_SAMPLES, ORACLE_LEN)).astype(np.float32)
    c2 = np.concatenate([x2, oracle], axis=1)
    c3 = np.concatenate([embed_inputs(x2, noise), oracle], axis=1)
    return _verify(old, new, c2, c3, True, tol)


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
            f"Widen a feature-v2 weight set to feature-v{FEATURE_VERSION} "
            f"({V2_INPUT_DIM} -> {INPUT_DIM} policy inputs, {V2_CRITIC_INPUT} -> "
            f"{CRITIC_INPUT} critic inputs) without changing the function it computes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "New input columns are written as exact zeros and verified as such by\n"
            "forwarding random batches through both nets; a mismatch exits 1, a\n"
            "wrong-version or already-migrated input directory exits 2."
        ),
    )
    ap.add_argument("--in", dest="indir", required=True, help="v2 weights directory")
    ap.add_argument("--out", dest="outdir", required=True, help="v3 weights directory")
    ap.add_argument(
        "--seed", type=int, default=0, help="RNG seed for the verification batch"
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
            "destroy the only copy of the v2 net if verification failed"
        )

    old_w, new_w = widen_policy(indir, outdir)
    print(
        f"policy: fc1 [{HIDDEN1}, {old_w}] -> [{HIDDEN1}, {new_w}]  "
        f"(planes {V2_PLANES}->{PLANES}, scalars {V2_SCALARS}->{SCALAR_LEN}); "
        "fc1.bias/fc2/fc3 copied verbatim"
    )
    if copy_aux(indir, outdir):
        print(f"aux:    {AUX_BLOB_NAME} copied (rows live on fc3, unaffected by width)")
    else:
        print(f"aux:    no {AUX_BLOB_NAME} in {indir} -- skipped")

    crit = widen_critic(indir, outdir)
    if crit is not None:
        print(
            f"critic: fc1 [{HIDDEN1}, {crit[0]}] -> [{HIDDEN1}, {crit[1]}]  "
            f"(oracle block moved {V2_INPUT_DIM}:{V2_CRITIC_INPUT} -> "
            f"{INPUT_DIM}:{CRITIC_INPUT})"
        )
    else:
        print(f"critic: no {CRITIC_MANIFEST_NAME} in {indir} -- skipped")

    rng = np.random.default_rng(args.seed)
    failed = False

    rep = verify_policy(indir, outdir, rng, args.tol)
    _report("policy", "logits+value", rep, args.tol)
    failed |= not rep.ok

    if crit is not None:
        rep = verify_critic(indir, outdir, rng, args.tol)
        _report("critic", "value", rep, args.tol)
        failed |= not rep.ok

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
        print(f"widen.py: {e.msg}", file=sys.stderr)
        sys.exit(e.code)
