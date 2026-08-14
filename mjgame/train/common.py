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

FEATURE_VERSION = 3                      # "v" on every "d" line of a trajectory
PLANES = 48
TILE_TYPES = 34
PLANE_LEN = PLANES * TILE_TYPES          # 1632 int8 values
SCALAR_LEN = 42                          # 42 little-endian float32 values = 168 bytes
INPUT_DIM = PLANE_LEN + SCALAR_LEN       # 1674
ACTIONS = 78                             # policy logits
OUTPUT_DIM = ACTIONS + 1                 # 79 = 78 logits + 1 value

# v3 is a strict SUPERSET of v2: the v2 planes are planes 0..35 unchanged and the
# v2 scalars are scalars 0..38 unchanged, so the v2 input vector is an exact
# prefix of each v3 SECTION (not of the whole vector -- the new planes sit
# between the old planes and the old scalars).  Written out, the v3 input is
#
#     [ planes 0..35   v2, 1224 floats ]   x[   0:1224]  <- v2 x[   0:1224]
#     [ planes 36..47  new,  408 floats ]   x[1224:1632]  <- new
#     [ scalars 0..38  v2,    39 floats ]   x[1632:1671]  <- v2 x[1224:1263]
#     [ scalars 39..41 new,    3 floats ]   x[1671:1674]  <- new
#
# which is exactly the column map `train/widen.py` implements to migrate a v2
# weight set into a v3 one without changing the function it computes.
V2_PLANE_LEN = 1224                      # 36 planes x 34
V2_SCALAR_BYTES = 156                    # 39 float32
V2_INPUT_DIM = V2_PLANE_LEN + V2_SCALAR_BYTES // 4        # 1263
V2_PLANE_FLOATS = V2_PLANE_LEN                            # v2 planes, as inputs
V2_SCALAR_FLOATS = V2_SCALAR_BYTES // 4                   # 39

# The v1 widths, kept only so a stale dataset can be NAMED when it is rejected.
V1_PLANE_LEN = 748
V1_SCALAR_BYTES = 132

HIDDEN1 = 512
HIDDEN2 = 256

# ---------------------------------------------------------------------------
# Privileged (oracle) side channel -- TRAINING ONLY, never exported to the engine
# ---------------------------------------------------------------------------
#
# The engine may attach an `"o"` field to a "d" line: base64 of int8[170], five
# 34-wide planes of information the acting seat CANNOT see --
#
#     plane 0..2  the three opponents' concealed hands, in RELATIVE seat order
#                 (shimocha, toimen, kamicha), counts per tile type
#     plane 3     the hidden remainder (live wall + dead wall + everything else
#                 unseen by the actor)
#     plane 4     the ura-dora indicators
#
# and a `"sh"` field: the three opponents' CURRENT shanten, same relative order,
# -1 for a complete (agari-shape) hand.
#
# These feed two things and only two things:
#   * the ORACLE CRITIC (critic.json/critic.f32) -- a separate value network
#     that sees policy features ++ oracle features.  Asymmetric actor-critic:
#     the baseline may peek, the policy may not, so per-hand luck is absorbed
#     by the critic instead of being charged to the policy's advantage.
#   * the AUXILIARY SPEED HEADS -- 3x8 shanten logits hung off the policy net's
#     last layer, predicting each opponent's shanten from PUBLIC features only.
#     The oracle supplies the LABEL; the input stays public, so the head is a
#     representation shaper, not an information leak.
ORACLE_PLANES = 5
ORACLE_LEN = ORACLE_PLANES * TILE_TYPES  # 170 int8 values
CRITIC_INPUT = INPUT_DIM + ORACLE_LEN    # 1844
V2_CRITIC_INPUT = V2_INPUT_DIM + ORACLE_LEN               # 1433, for widen.py

# Auxiliary shanten heads: one 8-way classifier per opponent.  The label is
# clip(shanten, 0, 7) -- note that clipping at 0 MERGES the "-1 = already
# complete" case into the tenpai class, which is deliberate: an actor cannot
# act on the difference (the hand is about to be declared either way) and it
# keeps the class set a plain 0..7 shanten ladder.  7 is the cap, not "7 or
# more is impossible" -- a 6-shanten-or-worse hand only appears on turn 1.
AUX_OPP = 3
AUX_CLASSES = 8
AUX_OUT = AUX_OPP * AUX_CLASSES          # 24 extra fc3 rows during training

# Sentinel written into `opp_shanten` for a decision line that carried no "sh".
AUX_MISSING = -9


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

# Trainer-private companions, written NEXT TO the pair above.  The engine's
# loader reads manifest.json/policy.f32 and nothing else, so these three may
# change shape freely without touching the frozen contract.
CRITIC_MANIFEST_NAME = "critic.json"
CRITIC_BLOB_NAME = "critic.f32"
AUX_BLOB_NAME = "aux.f32"


def critic_manifest_for(input_dim: int = CRITIC_INPUT) -> dict:
    """Arch descriptor for the oracle critic, analogous to `manifest_for`.

    Same file format as the policy manifest -- `blob` is a headerless
    [out][in]+bias little-endian float32 dump -- with `arch` set to
    "mlp-critic" so the two can never be mistaken for each other, and the
    input width carried both at top level and in layer 0 (the `features`
    block says what those inputs ARE).
    """
    return {
        "version": 1,
        "arch": "mlp-critic",
        "input": input_dim,
        "features": {
            "planes": PLANES,
            "scalars": SCALAR_LEN,
            "oracle_planes": ORACLE_PLANES,
        },
        "layers": [
            {"in": input_dim, "out": HIDDEN1, "act": "relu"},
            {"in": HIDDEN1, "out": HIDDEN2, "act": "relu"},
            {"in": HIDDEN2, "out": 1, "act": "none"},
        ],
        "blob": CRITIC_BLOB_NAME,
    }


def blob_floats(input_dim: int = INPUT_DIM) -> int:
    """Total float32 count in policy.f32, derived from the manifest."""
    return sum(l["in"] * l["out"] + l["out"] for l in manifest_for(input_dim)["layers"])


BLOB_FLOATS = blob_floats()
BLOB_BYTES = BLOB_FLOATS * 4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PolicyValueNet(nn.Module):
    """1674 -> 512 -> relu -> 256 -> relu -> 79 (78 action logits + 1 value).

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
    """Split a [B, >=79] forward pass into (policy logits [B, 78], value [B]).

    A net carrying auxiliary rows is wider than 79; the extra columns are not
    part of either return value here -- ask `split_aux` for those.
    """
    return out[..., :ACTIONS], out[..., ACTIONS]


def split_aux(out: mx.array) -> mx.array:
    """The auxiliary shanten logits of a [B, 103] forward pass, as [B, 3, 8].

    Raises if the net has no aux rows, so a caller can never silently train an
    aux loss against policy logits.
    """
    width = out.shape[-1]
    if width != OUTPUT_DIM + AUX_OUT:
        raise ValueError(
            f"net is {width} wide; auxiliary heads need {OUTPUT_DIM + AUX_OUT} "
            f"({OUTPUT_DIM} policy+value ++ {AUX_OUT} aux)"
        )
    return out[..., OUTPUT_DIM:].reshape(-1, AUX_OPP, AUX_CLASSES)


def widen_for_aux(model: PolicyValueNet, aux=None, scale: float = 0.01, seed=None):
    """Grow `model.fc3` from 79 to 103 rows in place, returning `model`.

    The first 79 rows are untouched, so the policy the widened net computes is
    bit-identical to the narrow one; `export_weights` slices them straight back
    out.  `aux` is an (weight [24, 256], bias [24]) pair -- e.g. from
    `load_aux` -- or None for a small random init, which starts the auxiliary
    heads near-uniform so their first gradients do not shove the shared trunk.
    """
    fc3 = model.fc3
    have = int(fc3.weight.shape[0])
    if have == OUTPUT_DIM + AUX_OUT:
        return model
    if have != OUTPUT_DIM:
        raise ValueError(f"fc3 has {have} rows, expected {OUTPUT_DIM}")

    if aux is None:
        if seed is not None:
            mx.random.seed(int(seed))
        w_aux = mx.random.normal((AUX_OUT, HIDDEN2)) * scale
        b_aux = mx.zeros((AUX_OUT,))
    else:
        w_np, b_np = aux
        w_aux = mx.array(np.ascontiguousarray(w_np, dtype=np.float32))
        b_aux = mx.array(np.ascontiguousarray(b_np, dtype=np.float32))
        if w_aux.shape != (AUX_OUT, HIDDEN2) or b_aux.shape != (AUX_OUT,):
            raise ValueError(
                f"aux rows are {tuple(w_aux.shape)}/{tuple(b_aux.shape)}, expected "
                f"{(AUX_OUT, HIDDEN2)}/{(AUX_OUT,)}"
            )

    fc3.weight = mx.concatenate([fc3.weight, w_aux.astype(fc3.weight.dtype)], axis=0)
    fc3.bias = mx.concatenate([fc3.bias, b_aux.astype(fc3.bias.dtype)], axis=0)
    mx.eval(model.parameters())
    return model


class CriticNet(nn.Module):
    """1844 -> 512 -> relu -> 256 -> relu -> 1: the PRIVILEGED baseline.

    Same shape and same blob layout as the policy trunk, but a different input
    (policy features ++ oracle planes) and a single output, and it is never
    handed to the engine.  Its whole job is to explain away the variance the
    policy cannot see -- who was tenpai, what the wall held -- so the advantage
    the policy trains on is closer to "was that a good decision" and further
    from "did that hand run well".
    """

    def __init__(self, input_dim: int = CRITIC_INPUT) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.fc1 = nn.Linear(self.input_dim, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, 1)

    def __call__(self, x: mx.array) -> mx.array:
        """[B, 1844] -> [B]; the trailing width-1 axis is dropped here so every
        caller regresses against a [B] target and none of them has to guess."""
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)[..., 0]

    @property
    def ordered_layers(self) -> List[nn.Linear]:
        return [self.fc1, self.fc2, self.fc3]


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

    X: np.ndarray            # [n, 1674] float32  planes ++ scalars
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
    oracle: np.ndarray       # [n, 170] float32  privileged planes, 0 when absent
    opp_shanten: np.ndarray  # [n, 3]   int32    opponents' shanten, -9 when absent
    has_oracle: bool         #          True iff EVERY "d" line had BOTH

    def __len__(self) -> int:
        return int(self.X.shape[0])


def _stale_hint(n_planes: int, n_scalar_bytes: int) -> str:
    """Name the feature version a decision line looks like, when we recognise it.

    DATA is never migrated -- a v2 trajectory is missing the twelve new planes
    and three new scalars outright, so there is nothing to widen it with and the
    only fix is to re-record.  WEIGHTS are a different story: a v2 net is a v3
    net with zeroed columns, which is what `train/widen.py` builds, so the hint
    points at it to spare a from-scratch retrain.
    """
    if n_planes == V2_PLANE_LEN or n_scalar_bytes == V2_SCALAR_BYTES:
        return (
            " -- v2 data -- re-record with the v3 engine "
            "(weights can be migrated with train/widen.py)"
        )
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


def _decode_oracle(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    if len(raw) != ORACLE_LEN:
        raise ValueError(
            f"oracle: expected {ORACLE_LEN} bytes ({ORACLE_PLANES} planes x "
            f"{TILE_TYPES}), got {len(raw)}"
        )
    return np.frombuffer(raw, dtype=np.int8)


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

    A "d" line MAY likewise carry the privileged pair `"o"` (base64 int8[170],
    five oracle planes) and `"sh"` (three opponents' shanten, relative order,
    -1 for a complete hand).  They land in `oracle` / `opp_shanten`, filled with
    zeros and `AUX_MISSING` respectively for lines that lack them, and
    `has_oracle` is True only when EVERY decision line carried BOTH -- same
    all-or-nothing rule as `has_round_viol`, and for the same reason: a half
    oracle-labelled batch would train a critic on two different input
    distributions.  Consumers that do not know about these fields (bc.py) are
    unaffected; the fields are appended, never inserted.
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
    oracles: List[np.ndarray] = []
    shantens: List[np.ndarray] = []
    oracle_lines = 0       # "d" lines that carried BOTH "o" and "sh"

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
                        extra = (
                            "  (v2 WEIGHTS can be migrated with train/widen.py; "
                            "v2 DATA cannot -- the new planes were never recorded)"
                            if ver == 2
                            else ""
                        )
                        raise ValueError(
                            f"{path}:{lineno}: v{ver} data -- re-record with the current "
                            f"engine (this trainer reads feature v{FEATURE_VERSION}: "
                            f"{PLANE_LEN} plane bytes + {SCALAR_LEN * 4} scalar bytes)"
                            f"{extra}"
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

                    ob = rec.get("o")
                    sh = rec.get("sh")
                    if ob is None:
                        oracles.append(np.zeros(ORACLE_LEN, dtype=np.float32))
                    else:
                        oracles.append(_decode_oracle(ob).astype(np.float32))
                    if sh is None:
                        shantens.append(
                            np.full(AUX_OPP, AUX_MISSING, dtype=np.int32)
                        )
                    else:
                        if len(sh) != AUX_OPP:
                            raise ValueError(
                                f"{path}:{lineno}: sh has {len(sh)} entries, "
                                f"expected {AUX_OPP}"
                            )
                        shantens.append(np.asarray(sh, dtype=np.int32))
                    if ob is not None and sh is not None:
                        oracle_lines += 1

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
        oracle=np.stack(oracles).astype(np.float32),
        opp_shanten=np.stack(shantens).astype(np.int32),
        has_oracle=oracle_lines == len(feats),
    )


# ---------------------------------------------------------------------------
# Weight export
# ---------------------------------------------------------------------------


def _check_features(mpath: str, feats, input_dim: int, oracle: bool = False) -> None:
    """Validate a manifest's `features` block for INTERNAL CONSISTENCY only.

    Deliberately NOT a comparison against this module's `PLANES`/`SCALAR_LEN`:
    the loaders have to stay able to read HISTORICAL weight sets so `widen.py`
    can migrate them, and a v2 manifest describes a perfectly well-formed net —
    it is just narrower than today's features.  So all that is required here is
    that the block agrees with the width the layers declare:

        planes * 34 + scalars [+ oracle_planes * 34] == input_dim

    Rejecting a net that is too narrow FOR THE DATA AT HAND is the caller's job
    (bc.py / ppo.py compare against the loaded trajectories and name widen.py);
    doing it here would make the migration tool unable to open its own input.
    """
    if not isinstance(feats, dict):
        raise ValueError(f"{mpath}: features is {feats!r}, expected an object")
    keys = ("planes", "scalars") + (("oracle_planes",) if oracle else ())
    vals = {}
    for key in keys:
        v = feats.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError(
                f"{mpath}: features.{key} is {v!r}, expected a non-negative integer"
            )
        vals[key] = v
    if set(feats) != set(keys):
        raise ValueError(
            f"{mpath}: features has keys {sorted(feats)}, expected {sorted(keys)}"
        )
    width = vals["planes"] * TILE_TYPES + vals["scalars"]
    if oracle:
        width += vals["oracle_planes"] * TILE_TYPES
    if width != input_dim:
        parts = f"{vals['planes']}x{TILE_TYPES} + {vals['scalars']}"
        if oracle:
            parts += f" + {vals['oracle_planes']}x{TILE_TYPES}"
        raise ValueError(
            f"{mpath}: features says {parts} = {width} inputs but layer 0 takes "
            f"{input_dim} -- the manifest disagrees with itself"
        )


def load_weights(path: str) -> PolicyValueNet:
    """Read manifest.json + policy.f32 back into a `PolicyValueNet`.

    The exact inverse of `export_weights`: `path` is either the directory that
    holds the pair or the manifest.json inside it.  The manifest is checked
    field-by-field against `manifest_for(input_dim)` — where `input_dim` is
    taken from the manifest's own first layer, so a net trained on a different
    feature width still loads and only DISAGREEMENTS with the fixed
    architecture (layer count, hidden sizes, action count) are errors.  The
    `features` block is checked for internal consistency only (see
    `_check_features`), which is what lets a HISTORICAL v2 weight set be opened
    by `train/widen.py` and migrated; a net too narrow for the data being
    trained on is rejected by the trainer, not here.  The blob is then sliced in
    the same order it was written, with no transpose: mlx.nn.Linear's `weight`
    is [out, in], which is the blob layout.
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
    for key in ("version", "arch", "actions", "blob"):
        if manifest.get(key) != want[key]:
            raise ValueError(
                f"{mpath}: {key} is {manifest.get(key)!r}, expected {want[key]!r}"
            )
    _check_features(mpath, manifest.get("features"), input_dim)
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

    A net carrying AUXILIARY ROWS (fc3 widened to 79+24 for the shanten heads)
    exports its first 79 rows and only those, so the blob the engine loads is
    byte-identical to what a plain 79-row net of the same weights would write.
    The sliced-off rows go to `aux.f32` beside it, which is trainer-private and
    exists purely so a later run can resume the heads instead of re-learning
    them from a random init.
    """
    os.makedirs(outdir, exist_ok=True)

    manifest = manifest_for(getattr(model, "input_dim", INPUT_DIM))
    chunks: List[np.ndarray] = []
    for layer, spec in zip(model.ordered_layers, manifest["layers"]):
        w = np.array(layer.weight, copy=True).astype("<f4")
        b = np.array(layer.bias, copy=True).astype("<f4")
        if w.shape[0] > spec["out"]:
            w = np.ascontiguousarray(w[: spec["out"]])
            b = np.ascontiguousarray(b[: spec["out"]])
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

    fc3 = model.ordered_layers[-1]
    extra = int(fc3.weight.shape[0]) - OUTPUT_DIM
    if extra > 0:
        if extra != AUX_OUT:
            raise ValueError(
                f"fc3 has {extra} extra row(s), expected {AUX_OUT} auxiliary rows"
            )
        export_aux(
            np.array(fc3.weight, copy=True)[OUTPUT_DIM:],
            np.array(fc3.bias, copy=True)[OUTPUT_DIM:],
            outdir,
        )

    return outdir


# ---------------------------------------------------------------------------
# Auxiliary head rows (trainer-private)
# ---------------------------------------------------------------------------


def export_aux(weight, bias, outdir: str) -> str:
    """Write aux.f32: [24][256] weights row-major, then the [24] bias."""
    os.makedirs(outdir, exist_ok=True)
    w = np.ascontiguousarray(np.asarray(weight), dtype="<f4")
    b = np.ascontiguousarray(np.asarray(bias), dtype="<f4")
    if w.shape != (AUX_OUT, HIDDEN2):
        raise ValueError(f"aux weight shape {w.shape} != {(AUX_OUT, HIDDEN2)}")
    if b.shape != (AUX_OUT,):
        raise ValueError(f"aux bias shape {b.shape} != {(AUX_OUT,)}")
    path = os.path.join(outdir, AUX_BLOB_NAME)
    with open(path, "wb") as fh:
        fh.write(np.concatenate([w.reshape(-1), b.reshape(-1)]).astype("<f4").tobytes())
    return path


def load_aux(path: str):
    """Read aux.f32 back as (weight [24, 256], bias [24]), or None if absent.

    `path` is the directory or the blob itself.  A wrong-sized file is an
    error, not a miss: silently starting the heads from scratch because the
    blob was stale would be indistinguishable from resuming them.
    """
    blob_path = os.path.join(path, AUX_BLOB_NAME) if os.path.isdir(path) else path
    if not os.path.exists(blob_path):
        return None
    blob = np.fromfile(blob_path, dtype="<f4")
    need = AUX_OUT * HIDDEN2 + AUX_OUT
    if blob.size != need:
        raise ValueError(
            f"{blob_path}: {blob.size} float32, expected {need} "
            f"({AUX_OUT}x{HIDDEN2} weights + {AUX_OUT} bias)"
        )
    w = blob[: AUX_OUT * HIDDEN2].reshape(AUX_OUT, HIDDEN2)
    b = blob[AUX_OUT * HIDDEN2 :]
    return np.ascontiguousarray(w, dtype=np.float32), np.ascontiguousarray(
        b, dtype=np.float32
    )


# ---------------------------------------------------------------------------
# Oracle critic weights (trainer-private)
# ---------------------------------------------------------------------------


def export_critic(net: CriticNet, outdir: str) -> str:
    """Write critic.json + critic.f32 into `outdir`, same layout as the policy pair."""
    os.makedirs(outdir, exist_ok=True)
    manifest = critic_manifest_for(getattr(net, "input_dim", CRITIC_INPUT))

    chunks: List[np.ndarray] = []
    for layer, spec in zip(net.ordered_layers, manifest["layers"]):
        w = np.array(layer.weight, copy=True).astype("<f4")
        b = np.array(layer.bias, copy=True).astype("<f4")
        if w.shape != (spec["out"], spec["in"]):
            raise ValueError(f"weight shape {w.shape} != {(spec['out'], spec['in'])}")
        if b.shape != (spec["out"],):
            raise ValueError(f"bias shape {b.shape} != {(spec['out'],)}")
        chunks.append(np.ascontiguousarray(w).reshape(-1))
        chunks.append(np.ascontiguousarray(b).reshape(-1))

    blob = np.concatenate(chunks).astype("<f4")
    with open(os.path.join(outdir, CRITIC_BLOB_NAME), "wb") as fh:
        fh.write(blob.tobytes())
    with open(os.path.join(outdir, CRITIC_MANIFEST_NAME), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, separators=(",", ":")))
    return outdir


def load_critic(path: str):
    """Read critic.json + critic.f32 back into a `CriticNet`, or None if absent.

    `path` is the directory or the critic.json inside it.  A MISSING pair
    returns None -- that is the "no critic yet, start fresh" case and the
    caller reports it -- while a pair that is present but disagrees with the
    architecture raises.  As with `load_weights`, the `features` block only has
    to agree with its own layer widths, so a v2 critic (1433 = 36x34 + 39 +
    5x34) still loads for `train/widen.py` to migrate; ppo.py is what rejects a
    critic too narrow for the trajectories in hand.
    """
    if os.path.isdir(path):
        mdir, mpath = path, os.path.join(path, CRITIC_MANIFEST_NAME)
    else:
        mdir, mpath = os.path.dirname(path) or ".", path
    if not os.path.exists(mpath):
        return None

    with open(mpath, "r", encoding="utf-8") as fh:
        try:
            manifest = json.load(fh)
        except json.JSONDecodeError as e:
            raise ValueError(f"{mpath}: bad JSON: {e}") from e

    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"{mpath}: critic manifest has no layers")
    try:
        input_dim = int(layers[0]["in"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{mpath}: first layer has no usable 'in': {e}") from e

    want = critic_manifest_for(input_dim)
    for key in ("version", "arch", "blob"):
        if manifest.get(key) != want[key]:
            raise ValueError(
                f"{mpath}: {key} is {manifest.get(key)!r}, expected {want[key]!r}"
            )
    _check_features(mpath, manifest.get("features"), input_dim, oracle=True)
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
    need = sum(l["in"] * l["out"] + l["out"] for l in want["layers"])
    if blob.size != need:
        raise ValueError(
            f"{blob_path}: {blob.size} float32 ({blob.size * 4} bytes), "
            f"expected {need} ({need * 4} bytes) for input width {input_dim}"
        )

    net = CriticNet(input_dim)
    off = 0
    for layer, spec in zip(net.ordered_layers, want["layers"]):
        n_w = spec["out"] * spec["in"]
        w = blob[off : off + n_w].reshape(spec["out"], spec["in"])
        off += n_w
        b = blob[off : off + spec["out"]]
        off += spec["out"]
        layer.weight = mx.array(np.ascontiguousarray(w, dtype=np.float32))
        layer.bias = mx.array(np.ascontiguousarray(b, dtype=np.float32))
    assert off == need, f"consumed {off} of {need} floats"
    mx.eval(net.parameters())
    return net
