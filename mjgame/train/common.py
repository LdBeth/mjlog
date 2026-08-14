"""Shared model / IO code for the mjgame RL agent.

Everything here implements the frozen contracts shared with the TypeScript
engine (see README.md).  Nothing in this module may drift from them:

  * trajectory JSONL   -- the engine's output, this trainer's input
  * manifest.json + policy.f32 -- this trainer's output, the engine's input
"""

from __future__ import annotations

import base64
import glob as _glob
import json
import os
import sys
from typing import Iterable, List, NamedTuple, Sequence, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Frozen contract constants
# ---------------------------------------------------------------------------

FEATURE_VERSION = 2                      # "v" on every "d" line of a trajectory
PLANES = 36
TILE_TYPES = 34
PLANE_LEN = PLANES * TILE_TYPES          # 1224 int8 values
SCALAR_LEN = 39                          # 39 little-endian float32 values = 156 bytes
INPUT_DIM = PLANE_LEN + SCALAR_LEN       # 1263
ACTIONS = 78                             # policy logits
OUTPUT_DIM = ACTIONS + 1                 # 79 = 78 logits + 1 value

# The v1 widths, kept only so a stale dataset can be NAMED when it is rejected.
V1_PLANE_LEN = 748
V1_SCALAR_BYTES = 132

HIDDEN1 = 512
HIDDEN2 = 256


def manifest_for(input_dim: int = INPUT_DIM) -> dict:
    """The manifest describing an `input_dim`-wide net of the fixed architecture.

    `version` is the FILE format's, which is 1; the feature layout is what the
    `features` block names, and the engine checks that block against its own
    encoder before it will load the blob.
    """
    return {
        "version": 1,
        "arch": "mlp",
        "features": {"planes": PLANES, "scalars": SCALAR_LEN},
        "actions": ACTIONS,
        "layers": [
            {"in": input_dim, "out": HIDDEN1, "act": "relu"},
            {"in": HIDDEN1, "out": HIDDEN2, "act": "relu"},
            {"in": HIDDEN2, "out": OUTPUT_DIM, "act": "none"},
        ],
        "blob": "policy.f32",
    }


MANIFEST = manifest_for()

MANIFEST_NAME = "manifest.json"
BLOB_NAME = "policy.f32"


def blob_floats(input_dim: int = INPUT_DIM) -> int:
    """Total float32 count in policy.f32, derived from the manifest."""
    return sum(l["in"] * l["out"] + l["out"] for l in manifest_for(input_dim)["layers"])


BLOB_FLOATS = blob_floats()
BLOB_BYTES = BLOB_FLOATS * 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PolicyValueNet(nn.Module):
    """1263 -> 512 -> relu -> 256 -> relu -> 79 (78 action logits + 1 value).

    mlx.nn.Linear stores `weight` as [out, in], which is exactly the row-major
    [out][in] layout the blob contract asks for, so export is a straight dump.

    The input width is a parameter rather than a constant so callers derive it
    from what they actually have — `X.shape[1]` when training, the manifest's
    first layer when loading — and a width mismatch surfaces as a shape error
    on the data instead of silently reshaping the wrong feature version.
    """

    def __init__(self, input_dim: int = INPUT_DIM) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.fc1 = nn.Linear(self.input_dim, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, OUTPUT_DIM)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)

    @property
    def ordered_layers(self) -> List[nn.Linear]:
        """Layers in blob order."""
        return [self.fc1, self.fc2, self.fc3]


def split_head(out: mx.array):
    """Split a [B, 79] forward pass into (policy logits [B, 78], value [B])."""
    return out[..., :ACTIONS], out[..., ACTIONS]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def masked_cross_entropy(logits: mx.array, mask: mx.array, target: mx.array) -> mx.array:
    """Mean cross-entropy over the legal-action support.

    `logits` may be [B, 78] or the raw [B, 79] network output (the value head
    is sliced off and therefore never contributes a gradient).  `mask` is a
    boolean [B, 78]; illegal entries are set to -inf before the log-softmax so
    they carry zero probability and zero gradient.  `target` is int [B].
    """
    if logits.shape[-1] == OUTPUT_DIM:
        logits = logits[..., :ACTIONS]
    neg_inf = mx.array(float("-inf"), dtype=logits.dtype)
    masked = mx.where(mask, logits, neg_inf)
    logp = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
    picked = mx.take_along_axis(logp, target.reshape(-1, 1).astype(mx.int32), axis=-1)
    return -picked.mean()


def masked_argmax(logits: mx.array, mask: mx.array) -> mx.array:
    """Top-1 action index restricted to the legal set."""
    if logits.shape[-1] == OUTPUT_DIM:
        logits = logits[..., :ACTIONS]
    neg_inf = mx.array(float("-inf"), dtype=logits.dtype)
    return mx.argmax(mx.where(mask, logits, neg_inf), axis=-1)


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------


class Trajectories(NamedTuple):
    """Decision-level tensors plus the episode bookkeeping RL will want.

    Unpacks in contract order:
        X, mask, action, seat, match, net, violations
    with a few extra fields appended for future use.  New fields are only ever
    APPENDED, so positional unpacking of the prefix and attribute access both
    keep working across format additions.
    """

    X: np.ndarray            # [n, 1263] float32  planes ++ scalars
    mask: np.ndarray         # [n, 78]  bool      legal actions
    action: np.ndarray       # [n]      int32     chosen action
    seat: np.ndarray         # [n]      int32     acting seat, 0-3
    match: np.ndarray        # [n]      int32     index into net/violations
    net: np.ndarray          # [m, 4]   float32   final settlement, uma applied
    violations: np.ndarray   # [m, 4]   float32   penalty totals
    scores: np.ndarray       # [m, 4]   float32   raw final scores
    rnd: np.ndarray          # [n]      int32     index into round_deltas
    round_deltas: np.ndarray     # [r, 4] float32 per-round point deltas
    round_outcome: np.ndarray    # [r]    object  "agari" | "draw"
    round_match: np.ndarray      # [r]    int32   index into net/violations
    kyoku: np.ndarray        # [n]      int32
    honba: np.ndarray        # [n]      int32
    junme: np.ndarray        # [n]      int32
    round_viol: np.ndarray   # [r, 4]   float32  penalty points incurred IN round k
    has_round_viol: bool     #          True iff EVERY "r" line carried "viol"

    def __len__(self) -> int:
        return int(self.X.shape[0])


def _stale_hint(n_planes: int, n_scalar_bytes: int) -> str:
    """Name the feature version a decision line looks like, when we recognise it."""
    if n_planes == V1_PLANE_LEN or n_scalar_bytes == V1_SCALAR_BYTES:
        return " -- v1 data -- re-record with the current engine"
    return ""


def _decode_planes(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    if len(raw) != PLANE_LEN:
        raise ValueError(
            f"planes: expected {PLANE_LEN} bytes (feature v{FEATURE_VERSION}), "
            f"got {len(raw)}{_stale_hint(len(raw), 0)}"
        )
    return np.frombuffer(raw, dtype=np.int8)


def _decode_scalars(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    if len(raw) != SCALAR_LEN * 4:
        raise ValueError(
            f"scalars: expected {SCALAR_LEN * 4} bytes (feature v{FEATURE_VERSION}), "
            f"got {len(raw)}{_stale_hint(0, len(raw))}"
        )
    return np.frombuffer(raw, dtype="<f4")


def _expand_files(pattern: Union[str, Sequence[str]]) -> List[str]:
    pats: Iterable[str] = [pattern] if isinstance(pattern, str) else pattern
    files: List[str] = []
    for p in pats:
        hits = sorted(_glob.glob(p))
        if not hits and os.path.exists(p):
            hits = [p]
        files.extend(hits)
    # de-dup, keep order
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def load_trajectories(pattern: Union[str, Sequence[str]]) -> Trajectories:
    """Read trajectory JSONL file(s) into flat arrays.

    `pattern` is a glob (or list of globs / paths).  Lines of one match are
    contiguous: every "d"/"r" line of a match precedes its "m" line, so we
    accumulate decisions and close them out when the "m" arrives.

    An "r" line MAY carry `"viol":[4]` — the 評価点 penalty points incurred
    during that round, per absolute seat, as positive magnitudes.  It lands in
    `round_viol`, zero-filled for lines that lack it, and `has_round_viol` is
    True only when EVERY round line in the whole dataset carried it: a mixed
    dataset cannot be timestamped consistently, so it counts as not having the
    field at all and the caller must fall back to the match-level totals.
    """
    files = _expand_files(pattern)
    if not files:
        raise FileNotFoundError(f"no trajectory files matched: {pattern!r}")

    feats: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    actions: List[int] = []
    seats: List[int] = []
    match_ids: List[int] = []
    round_ids: List[int] = []
    kyokus: List[int] = []
    honbas: List[int] = []
    junmes: List[int] = []

    nets: List[List[float]] = []
    viols: List[List[float]] = []
    scores: List[List[float]] = []

    r_deltas: List[List[float]] = []
    r_outcome: List[str] = []
    r_match: List[int] = []
    r_viol: List[List[float]] = []
    r_viol_lines = 0       # "r" lines that actually carried "viol"

    match_idx = 0          # index of the match currently being accumulated
    open_decisions = 0     # decisions seen since the last "m" line

    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: bad JSON: {e}") from e
                kind = rec.get("k")

                if kind == "d":
                    # v1 lines carry no "v" at all, which is exactly what makes
                    # a missing field mean 1 rather than "trust it".
                    ver = int(rec.get("v", 1))
                    if ver != FEATURE_VERSION:
                        raise ValueError(
                            f"{path}:{lineno}: v{ver} data -- re-record with the current "
                            f"engine (this trainer reads feature v{FEATURE_VERSION}: "
                            f"{PLANE_LEN} plane bytes + {SCALAR_LEN * 4} scalar bytes)"
                        )
                    planes = _decode_planes(rec["planes"]).astype(np.float32)
                    sc = _decode_scalars(rec["scalars"]).astype(np.float32)
                    feats.append(np.concatenate([planes, sc]))

                    m = np.zeros(ACTIONS, dtype=bool)
                    legal = rec["mask"]
                    if not legal:
                        raise ValueError(f"{path}:{lineno}: empty legal mask")
                    for a in legal:
                        if not 0 <= a < ACTIONS:
                            raise ValueError(f"{path}:{lineno}: action {a} out of range")
                        m[a] = True
                    a = int(rec["a"])
                    if not m[a]:
                        raise ValueError(f"{path}:{lineno}: chosen action {a} not in mask")
                    masks.append(m)
                    actions.append(a)

                    seats.append(int(rec["seat"]))
                    match_ids.append(match_idx)
                    round_ids.append(len(r_deltas))
                    kyokus.append(int(rec.get("kyoku", 0)))
                    honbas.append(int(rec.get("honba", 0)))
                    junmes.append(int(rec.get("junme", 0)))
                    open_decisions += 1

                elif kind == "r":
                    r_deltas.append([float(v) for v in rec["deltas"]])
                    r_outcome.append(str(rec["outcome"]))
                    r_match.append(match_idx)
                    rv = rec.get("viol")
                    if rv is None:
                        r_viol.append([0.0] * 4)
                    else:
                        if len(rv) != 4:
                            raise ValueError(
                                f"{path}:{lineno}: viol has {len(rv)} entries, expected 4"
                            )
                        r_viol.append([float(v) for v in rv])
                        r_viol_lines += 1

                elif kind == "m":
                    scores.append([float(v) for v in rec["scores"]])
                    nets.append([float(v) for v in rec["net"]])
                    viols.append([float(v) for v in rec["violations"]])
                    match_idx += 1
                    open_decisions = 0

                else:
                    raise ValueError(f"{path}:{lineno}: unknown record kind {kind!r}")

    if open_decisions:
        # Truncated file (crash / still-running writer): keep the decisions but
        # give them a zero-filled match record so indices stay in bounds.
        print(
            f"warning: {open_decisions} decision(s) with no closing 'm' line; "
            "padding match outcome with zeros",
            file=sys.stderr,
        )
        scores.append([0.0] * 4)
        nets.append([0.0] * 4)
        viols.append([0.0] * 4)

    if not feats:
        raise ValueError("no decision records found")

    return Trajectories(
        X=np.stack(feats).astype(np.float32),
        mask=np.stack(masks),
        action=np.asarray(actions, dtype=np.int32),
        seat=np.asarray(seats, dtype=np.int32),
        match=np.asarray(match_ids, dtype=np.int32),
        net=np.asarray(nets, dtype=np.float32).reshape(-1, 4),
        violations=np.asarray(viols, dtype=np.float32).reshape(-1, 4),
        scores=np.asarray(scores, dtype=np.float32).reshape(-1, 4),
        rnd=np.asarray(round_ids, dtype=np.int32),
        round_deltas=np.asarray(r_deltas, dtype=np.float32).reshape(-1, 4),
        round_outcome=np.asarray(r_outcome, dtype=object),
        round_match=np.asarray(r_match, dtype=np.int32),
        kyoku=np.asarray(kyokus, dtype=np.int32),
        honba=np.asarray(honbas, dtype=np.int32),
        junme=np.asarray(junmes, dtype=np.int32),
        round_viol=np.asarray(r_viol, dtype=np.float32).reshape(-1, 4),
        has_round_viol=bool(r_deltas) and r_viol_lines == len(r_deltas),
    )


# ---------------------------------------------------------------------------
# Weight export
# ---------------------------------------------------------------------------


def load_weights(path: str) -> PolicyValueNet:
    """Read manifest.json + policy.f32 back into a `PolicyValueNet`.

    The exact inverse of `export_weights`: `path` is either the directory that
    holds the pair or the manifest.json inside it.  The manifest is checked
    field-by-field against `manifest_for(input_dim)` — where `input_dim` is
    taken from the manifest's own first layer, so a net trained on a different
    feature width still loads and only DISAGREEMENTS with the fixed
    architecture (layer count, hidden sizes, action count, feature block) are
    errors.  The blob is then sliced in the same order it was written, with no
    transpose: mlx.nn.Linear's `weight` is [out, in], which is the blob layout.
    """
    if os.path.isdir(path):
        mdir, mpath = path, os.path.join(path, MANIFEST_NAME)
    else:
        mdir, mpath = os.path.dirname(path) or ".", path
    if not os.path.exists(mpath):
        raise ValueError(f"no {MANIFEST_NAME} at {mpath}")

    with open(mpath, "r", encoding="utf-8") as fh:
        try:
            manifest = json.load(fh)
        except json.JSONDecodeError as e:
            raise ValueError(f"{mpath}: bad JSON: {e}") from e

    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"{mpath}: manifest has no layers")
    try:
        input_dim = int(layers[0]["in"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{mpath}: first layer has no usable 'in': {e}") from e

    want = manifest_for(input_dim)
    for key in ("version", "arch", "features", "actions", "blob"):
        if manifest.get(key) != want[key]:
            raise ValueError(
                f"{mpath}: {key} is {manifest.get(key)!r}, expected {want[key]!r}"
            )
    if len(layers) != len(want["layers"]):
        raise ValueError(
            f"{mpath}: {len(layers)} layer(s), expected {len(want['layers'])}"
        )
    for i, (got, exp) in enumerate(zip(layers, want["layers"])):
        for key in ("in", "out", "act"):
            gv = got.get(key) if isinstance(got, dict) else None
            if key != "act":
                gv = int(gv) if gv is not None else None
            if gv != exp[key]:
                raise ValueError(
                    f"{mpath}: layer {i} {key} is {gv!r}, expected {exp[key]!r}"
                )

    blob_path = os.path.join(mdir, manifest["blob"])
    if not os.path.exists(blob_path):
        raise ValueError(f"no {manifest['blob']} next to {mpath}")
    blob = np.fromfile(blob_path, dtype="<f4")
    need = blob_floats(input_dim)
    if blob.size != need:
        raise ValueError(
            f"{blob_path}: {blob.size} float32 ({blob.size * 4} bytes), "
            f"expected {need} ({need * 4} bytes) for input width {input_dim}"
        )

    model = PolicyValueNet(input_dim)
    off = 0
    for layer, spec in zip(model.ordered_layers, want["layers"]):
        n_w = spec["out"] * spec["in"]
        w = blob[off : off + n_w].reshape(spec["out"], spec["in"])
        off += n_w
        b = blob[off : off + spec["out"]]
        off += spec["out"]
        layer.weight = mx.array(np.ascontiguousarray(w, dtype=np.float32))
        layer.bias = mx.array(np.ascontiguousarray(b, dtype=np.float32))
    assert off == need, f"consumed {off} of {need} floats"
    mx.eval(model.parameters())
    return model


def export_weights(model: PolicyValueNet, outdir: str) -> str:
    """Write manifest.json + policy.f32 into `outdir`, byte-exactly.

    policy.f32 is, per layer in order: weight [out][in] row-major, then bias
    [out]; all little-endian float32, concatenated with no header or padding.
    """
    os.makedirs(outdir, exist_ok=True)

    manifest = manifest_for(getattr(model, "input_dim", INPUT_DIM))
    chunks: List[np.ndarray] = []
    for layer, spec in zip(model.ordered_layers, manifest["layers"]):
        w = np.array(layer.weight, copy=True).astype("<f4")
        b = np.array(layer.bias, copy=True).astype("<f4")
        if w.shape != (spec["out"], spec["in"]):
            raise ValueError(f"weight shape {w.shape} != {(spec['out'], spec['in'])}")
        if b.shape != (spec["out"],):
            raise ValueError(f"bias shape {b.shape} != {(spec['out'],)}")
        chunks.append(np.ascontiguousarray(w).reshape(-1))
        chunks.append(np.ascontiguousarray(b).reshape(-1))

    blob = np.concatenate(chunks).astype("<f4")
    want = blob_floats(manifest["layers"][0]["in"])
    if blob.size != want:
        raise ValueError(f"blob has {blob.size} floats, expected {want}")

    blob_path = os.path.join(outdir, BLOB_NAME)
    with open(blob_path, "wb") as fh:
        fh.write(blob.tobytes())

    with open(os.path.join(outdir, MANIFEST_NAME), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, separators=(",", ":")))

    return outdir
