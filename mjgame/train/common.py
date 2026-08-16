"""Shared model / IO code for the mjgame RL agent.

Everything here implements the frozen contracts shared with the TypeScript
engine (see README.md).  Nothing in this module may drift from them:

  * trajectory JSONL   -- the engine's output, this trainer's input
  * manifest.json + policy.f32 + attn.f32 -- this trainer's output, the
    engine's input (attn.f32 only from feature v4 on; see train/V4_SPEC.md)
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

FEATURE_VERSION = 4                      # "v" on every "d" line of a trajectory
PLANES = 48
TILE_TYPES = 34
PLANE_LEN = PLANES * TILE_TYPES          # 1632 int8 values
SCALAR_LEN = 42                          # 42 little-endian float32 values = 168 bytes
PLANE_SCALAR_DIM = PLANE_LEN + SCALAR_LEN                 # 1674 -- what a "d" line carries
ACTIONS = 78                             # policy logits
OUTPUT_DIM = ACTIONS + 1                 # 79 = 78 logits + 1 value

# ---------------------------------------------------------------------------
# Feature v4: the attention river encoder (train/V4_SPEC.md is the frozen spec)
# ---------------------------------------------------------------------------
#
# v4 changes NOTHING about the planes (48) and scalars (42): a "d" line still
# carries exactly the 1674 dense values above, and a v3 digest of them still
# passes.  What v4 adds is a second, SEQUENTIAL view of the same board -- one
# token per river entry, all four seats, relative seat order, each river
# truncated to its first 24 discards -- which a small self-attention encoder
# folds into a 64-vector z that is CONCATENATED onto the dense features:
#
#     policy fc1 input  = [planes+scalars 1674][z 64]            = 1738
#     critic fc1 input  = [planes+scalars 1674][oracle 170][z 64] = 1908
#
# so every v3 column keeps its index and the new capacity is a suffix.  That is
# what makes `train/widen4.py` a function-preserving migration.
SEQ_MAX = 96                             # 4 seats x 24 entries, the token cap
RIVER_MAX = 24                           # per-river truncation; also the idx scale
TOK_BYTES = 4                            # packed token: [type, seatRel, idx, flags]
TOK_DENSE = 42                           # onehot34 ++ onehot4 ++ idx/24 ++ 3 flags
ATTN_D = 64                              # d_model of the encoder, and |z|
ATTN_HEADS = 4
ATTN_HEAD_DIM = ATTN_D // ATTN_HEADS     # 16
ATTN_SCALE = 0.25                        # frozen 1/4 -- which is also 1/sqrt(16)

# Packed-token field bounds.  A value outside them is DEFINED, not rejected: the
# dense expansion simply sets no one-hot bit for it (V4_SPEC.md, "Out-of-range
# inputs"), which is what an equality-against-arange one-hot does for free.  The
# loader counts them anyway -- defined behaviour for a broken recorder is still a
# broken recorder.
TOK_TYPE_MAX = TILE_TYPES - 1            # 33
TOK_SEAT_MAX = 3
TOK_IDX_MAX = RIVER_MAX - 1              # 23
TOK_FLAG_MAX = 7                         # bit0 tsumogiri | bit1 riichi | bit2 called

INPUT_DIM = PLANE_SCALAR_DIM + ATTN_D    # 1738 -- policy fc1 width
V3_INPUT_DIM = PLANE_SCALAR_DIM          # 1674 -- the v3 policy fc1 width

# The encoder's weights ride BESIDE policy.f32 (same directory, same [out][in]
# row-major + bias little-endian float32 convention); the engine loads them
# through `rlnet_attn_create`, so unlike aux.f32/critic.f32 this one is NOT
# trainer-private.
ATTN_BLOB_NAME = "attn.f32"

# attn.f32, in file order.  This list IS the format: `export_attn` writes it top
# to bottom and `load_attn` reads it back the same way, so the two can only ever
# disagree by disagreeing with the spec.
ATTN_TENSORS = (
    ("W_in", (ATTN_D, TOK_DENSE)), ("b_in", (ATTN_D,)),
    ("Wq", (ATTN_D, ATTN_D)), ("bq", (ATTN_D,)),
    ("Wk", (ATTN_D, ATTN_D)), ("bk", (ATTN_D,)),
    ("Wv", (ATTN_D, ATTN_D)), ("bv", (ATTN_D,)),
    ("Wo", (ATTN_D, ATTN_D)), ("bo", (ATTN_D,)),
    ("u", (ATTN_D,)),
    ("Wz", (ATTN_D, ATTN_D)), ("bz", (ATTN_D,)),
)

ATTN_FLOATS = sum(int(np.prod(shape)) for _, shape in ATTN_TENSORS)   # 23616
ATTN_BYTES = ATTN_FLOATS * 4

# !! SPEC ARITHMETIC.  V4_SPEC.md states "Total floats: 64*42+64 + 4*(64*64+64)
# + 64 + 64*64+64 = 23,872".  The expression is right and the total is not: it
# evaluates to 23,616.  23,872 is what the same expression gives with a
# [64][46] W_in, i.e. the total was computed against a 46-wide dense token that
# the spec never describes (the enumerated expansion is 34 + 4 + 1 + 3 = 42, and
# W_in is written [64][42] in the tensor list).  The TENSOR LIST is authoritative
# here -- it is what is actually serialised -- so this file is 23,616 floats /
# 94,464 bytes.  The spec's total line needs the correction.
assert ATTN_FLOATS == 23616, ATTN_FLOATS

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
# [planes+scalars 1674][oracle 170][z 64] -- z is APPENDED at the end, so every
# v3 critic column keeps its index (V4_SPEC.md "Network input layout").
V3_CRITIC_INPUT = PLANE_SCALAR_DIM + ORACLE_LEN           # 1844, the v3 critic width
CRITIC_INPUT = V3_CRITIC_INPUT + ATTN_D                   # 1908
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


def has_attn(input_dim: int) -> bool:
    """Does an `input_dim`-wide net take the v4 attention suffix?

    Width IS the discriminator: v3 and v4 have the same planes/scalars, so the
    only thing that separates a 1674-wide policy from a 1738-wide one is the
    64-column z block on the end (and 1844 vs 1908 on the critic).
    """
    return input_dim in (INPUT_DIM, CRITIC_INPUT)


def manifest_for(input_dim: int = INPUT_DIM) -> dict:
    """The manifest describing an `input_dim`-wide net of the fixed architecture.

    `version` is the FILE format's, which is 1; the feature layout is what the
    `features` block names, and the engine checks that block against its own
    encoder before it will load the blob.

    v4 (input 1738) adds ONE top-level key and NOTHING inside `features`:

        "attn": "attn.f32"      the river encoder's weights, beside policy.f32

    `features` stays {"planes": 48, "scalars": 42} byte-for-byte, because v4 did
    not touch the planes or the scalars -- the engine's existing check of that
    block therefore keeps passing unchanged.  The PRESENCE of `attn` is what
    tells a v3 weight set (no attn, fc1 1674) from a v4 one (attn, fc1 1738),
    which is exactly how `checkManifest` in src/rl/net.ts reads it: it derives
    the expected layer-0 width from that key alone.  No separate version field
    is written, because two discriminators that could disagree are worse than
    one.
    """
    m = {
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
    if has_attn(input_dim):
        m["attn"] = ATTN_BLOB_NAME
    return m


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

    The v4 critic's trailing 64 z columns are NOT named in `features`: the
    encoder is the policy's (the critic reads a stop-gradient copy of its
    output, see ppo.py), the critic never owns a copy of attn.f32, and the
    width already says whether they are there.  `_check_features` therefore
    accepts both 1844 and 1908 against the same block.
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
    """1738 -> 512 -> relu -> 256 -> relu -> 79 (78 action logits + 1 value).

    The 1738 is [planes+scalars 1674][z 64]: this module is the plain MLP and
    knows nothing about where z came from, which is exactly why a v3 net
    (1674) still instantiates and still loads.

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
    """1908 -> 512 -> relu -> 256 -> relu -> 1: the PRIVILEGED baseline.

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
        """[B, 1908] -> [B]; the trailing width-1 axis is dropped here so every
        caller regresses against a [B] target and none of them has to guess."""
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)[..., 0]

    @property
    def ordered_layers(self) -> List[nn.Linear]:
        return [self.fc1, self.fc2, self.fc3]


# ---------------------------------------------------------------------------
# The river encoder's parameters
# ---------------------------------------------------------------------------


class AttnEncoder(nn.Module):
    """The v4 river encoder's WEIGHTS -- a container, not a forward pass.

    Six affine maps and one learned query vector, exactly the tensors
    `ATTN_TENSORS` names:

        w_in  42 -> 64     the dense token embedding (relu after)
        wq/wk/wv  64 -> 64 the 4-head attention projections
        wo    64 -> 64     the per-token output map
        u     [64]         the pooling query
        wz    64 -> 64     p -> z

    The FORWARD lives in `ppo.py` (`encode_tokens`), not here, for the same
    reason `masked_logp` does: this module is the file format's shape and the
    optimizer's parameter tree, and every consumer that only needs to move the
    bytes around -- widen4.py, an exporter, a checkpoint diff -- can then use it
    without pulling in the batching, masking and padding conventions of one
    particular trainer.

    mlx.nn.Linear stores `weight` as [out, in], which is the blob layout, so
    export is a straight dump in `ordered_tensors` order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w_in = nn.Linear(TOK_DENSE, ATTN_D)
        self.wq = nn.Linear(ATTN_D, ATTN_D)
        self.wk = nn.Linear(ATTN_D, ATTN_D)
        self.wv = nn.Linear(ATTN_D, ATTN_D)
        self.wo = nn.Linear(ATTN_D, ATTN_D)
        self.u = mx.zeros((ATTN_D,))
        self.wz = nn.Linear(ATTN_D, ATTN_D)

    @property
    def ordered_tensors(self) -> List[mx.array]:
        """Every parameter, in attn.f32 order."""
        return [
            self.w_in.weight, self.w_in.bias,
            self.wq.weight, self.wq.bias,
            self.wk.weight, self.wk.bias,
            self.wv.weight, self.wv.bias,
            self.wo.weight, self.wo.bias,
            self.u,
            self.wz.weight, self.wz.bias,
        ]

    def set_ordered(self, arrays: Sequence[np.ndarray]) -> "AttnEncoder":
        """Load from a list in `ATTN_TENSORS` order, checking every shape."""
        if len(arrays) != len(ATTN_TENSORS):
            raise ValueError(
                f"{len(arrays)} tensor(s), expected {len(ATTN_TENSORS)}"
            )
        vals = []
        for (name, shape), a in zip(ATTN_TENSORS, arrays):
            a = np.ascontiguousarray(a, dtype=np.float32)
            if a.shape != shape:
                raise ValueError(f"{name} is {a.shape}, expected {shape}")
            vals.append(mx.array(a))
        (
            self.w_in.weight, self.w_in.bias,
            self.wq.weight, self.wq.bias,
            self.wk.weight, self.wk.bias,
            self.wv.weight, self.wv.bias,
            self.wo.weight, self.wo.bias,
            self.u,
            self.wz.weight, self.wz.bias,
        ) = vals
        mx.eval(self.parameters())
        return self


def export_attn(enc: AttnEncoder, outdir: str) -> str:
    """Write attn.f32 into `outdir`: `ATTN_TENSORS` in order, LE float32, no header."""
    os.makedirs(outdir, exist_ok=True)
    chunks: List[np.ndarray] = []
    for (name, shape), t in zip(ATTN_TENSORS, enc.ordered_tensors):
        a = np.array(t, copy=True).astype("<f4")
        if a.shape != shape:
            raise ValueError(f"attn {name} is {a.shape}, expected {shape}")
        chunks.append(np.ascontiguousarray(a).reshape(-1))
    blob = np.concatenate(chunks).astype("<f4")
    if blob.size != ATTN_FLOATS:
        raise ValueError(f"attn blob has {blob.size} floats, expected {ATTN_FLOATS}")
    path = os.path.join(outdir, ATTN_BLOB_NAME)
    with open(path, "wb") as fh:
        fh.write(blob.tobytes())
    return path


def load_attn(path: str):
    """Read attn.f32 back into an `AttnEncoder`, or None if there is none.

    `path` is the directory or the blob itself.  A MISSING file returns None --
    "this is a v3 weight set, migrate it" is the caller's message to write, with
    the width of its policy in hand -- while a file that is present and the
    wrong size raises, because silently re-initialising an encoder would be
    indistinguishable from resuming one.
    """
    blob_path = os.path.join(path, ATTN_BLOB_NAME) if os.path.isdir(path) else path
    if not os.path.exists(blob_path):
        return None
    blob = np.fromfile(blob_path, dtype="<f4")
    if blob.size != ATTN_FLOATS:
        raise ValueError(
            f"{blob_path}: {blob.size} float32 ({blob.size * 4} bytes), expected "
            f"{ATTN_FLOATS} ({ATTN_BYTES} bytes) = "
            + " + ".join(f"{n}{list(s)}" for n, s in ATTN_TENSORS)
        )
    arrays, off = [], 0
    for _, shape in ATTN_TENSORS:
        n = int(np.prod(shape))
        arrays.append(np.ascontiguousarray(blob[off : off + n].reshape(shape)))
        off += n
    assert off == ATTN_FLOATS, f"consumed {off} of {ATTN_FLOATS} floats"
    return AttnEncoder().set_ordered(arrays)


def random_attn(seed=None, std: float = 0.02) -> AttnEncoder:
    """An encoder with every parameter ~ normal(0, `std`), biases and u included.

    This is the init `train/widen4.py` writes, and the reason it is NOT zeros is
    the argument in V4_SPEC.md's Migration section: the consumer columns on the
    other side of z are zero, so if the encoder were zero too the whole path
    would sit at a saddle with no gradient on either side.
    """
    rng = np.random.default_rng(seed)
    return AttnEncoder().set_ordered(
        [rng.normal(0.0, std, size=shape).astype(np.float32) for _, shape in ATTN_TENSORS]
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def masked_cross_entropy(
    logits: mx.array,
    mask: mx.array,
    target: mx.array,
    weight: mx.array | None = None,
) -> mx.array:
    """Mean cross-entropy over the legal-action support.

    `logits` may be [B, 78] or the raw [B, 79] network output (the value head
    is sliced off and therefore never contributes a gradient).  `mask` is a
    boolean [B, 78]; illegal entries are set to -inf before the log-softmax so
    they carry zero probability and zero gradient.  `target` is int [B].

    `weight`, when given, is a per-sample [B] importance factor and the result
    is the weighted mean sum(w*ce)/sum(w).  BC uses it to counter class
    imbalance (call decisions are ~4% of a teacher dataset, so plain mean CE
    buys accuracy on discards and resolves claim-window near-ties toward
    pass); omitting it is the old behavior everywhere else.
    """
    if logits.shape[-1] == OUTPUT_DIM:
        logits = logits[..., :ACTIONS]
    neg_inf = mx.array(float("-inf"), dtype=logits.dtype)
    masked = mx.where(mask, logits, neg_inf)
    logp = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
    picked = mx.take_along_axis(logp, target.reshape(-1, 1).astype(mx.int32), axis=-1)
    if weight is None:
        return -picked.mean()
    w = weight.reshape(-1)
    return -(picked.reshape(-1) * w).sum() / w.sum()


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
    seq: np.ndarray          # [n, 96, 4] int8   packed river tokens, zero-padded
    seq_len: np.ndarray      # [n]      int32    valid token count, 0..96
    round_kyoku: np.ndarray  # [r]      int32    the "r" line's own kyoku, -1 if absent
    round_honba: np.ndarray  # [r]      int32    the "r" line's own honba, -1 if absent
    has_round_id: bool       #          True iff EVERY "r" line named its round

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
    # v3 is NOT nameable here: its planes and scalars are byte-identical to v4's,
    # so a v3 line never reaches a size mismatch -- it is caught by its "v" in
    # `load_trajectories`, which is where the widen4.py hint lives.
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


def _decode_seq(b64: str):
    """The "seq" field -> ([n_tok, 4] int8, tokens clamped away, fields out of range).

    The stream is base64 of an Int8Array, four bytes per river entry
    ([type, seatRel, idx, flags]).  Two of the three checks here are the spec's
    own out-of-range rules (V4_SPEC.md), which every implementation shares:

      * more than SEQ_MAX tokens is CLAMPED to SEQ_MAX, not rejected;
      * a type/seatRel/idx outside its range is legal input and simply sets no
        one-hot bit in the dense expansion.

    So neither raises.  Both are COUNTED instead and reported once per load,
    because while the forward is defined for them, a recorder that produces them
    at scale is broken and the number is the only way to notice.  A length that
    is not a whole number of tokens IS an error: it means the stream is
    misaligned and every field after the break is garbage.
    """
    raw = base64.b64decode(b64)
    if len(raw) % TOK_BYTES:
        raise ValueError(
            f"seq: {len(raw)} bytes is not a whole number of {TOK_BYTES}-byte tokens"
        )
    n_tok = len(raw) // TOK_BYTES
    tok = np.frombuffer(raw, dtype=np.int8).reshape(n_tok, TOK_BYTES)
    clamped = max(n_tok - SEQ_MAX, 0)
    if clamped:
        tok = tok[:SEQ_MAX]
    oor = 0
    if tok.size:
        bounds = np.array(
            [TOK_TYPE_MAX, TOK_SEAT_MAX, TOK_IDX_MAX, TOK_FLAG_MAX], dtype=np.int16
        )
        t16 = tok.astype(np.int16)
        oor = int(np.count_nonzero(((t16 < 0) | (t16 > bounds)).any(axis=1)))
    return tok, clamped, oor


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

    An "r" line MAY likewise carry `"kyoku"`/`"honba"` — the round it reports,
    named with the same pair its "d" lines carry.  They land in `round_kyoku` /
    `round_honba`, filled with -1 for lines that lack them, and `has_round_id`
    follows the same all-or-nothing rule: True only when EVERY round line in the
    dataset named itself, because a half-labelled dataset would need two
    different joins and neither could check the other.  With it, a consumer
    joins a decision to its round directly on (match, kyoku, honba); without it
    the only join available is positional, which cannot see a round that holds
    no decision at all (see `decision_round` in ppo.py).

    A "d" line MAY likewise carry the privileged pair `"o"` (base64 int8[170],
    five oracle planes) and `"sh"` (three opponents' shanten, relative order,
    -1 for a complete hand).  They land in `oracle` / `opp_shanten`, filled with
    zeros and `AUX_MISSING` respectively for lines that lack them, and
    `has_oracle` is True only when EVERY decision line carried BOTH -- same
    all-or-nothing rule as `has_round_viol`, and for the same reason: a half
    oracle-labelled batch would train a critic on two different input
    distributions.  Consumers that do not know about these fields (bc.py) are
    unaffected; the fields are appended, never inserted.

    Feature v4 adds `"seq"` — base64 of the packed river tokens, four int8 per
    entry — and it is NOT optional: `"v":4` means the field is there.  It lands
    in `seq` zero-padded to [n, 96, 4] with the true count in `seq_len`, which
    is the shape the batched encoder wants and is cheap besides (384 bytes a
    decision against the 6.5 kB of planes).  A v4 line that carries no "seq" at
    all is read as an EMPTY river and counted, and the count is reported once at
    the end of the load: an empty river is a real, common state (every decision
    before the first discard) and JSON has no way to tell "no tokens" from
    "field omitted", so refusing the whole dataset over it would be a worse
    failure than saying so out loud.
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
    seqs: List[np.ndarray] = []
    seq_lens: List[int] = []
    seq_missing = 0        # v4 "d" lines with no "seq" field at all
    seq_clamped = 0        # tokens dropped by the SEQ_MAX clamp
    seq_oor = 0            # tokens with a field outside its documented range

    nets: List[List[float]] = []
    viols: List[List[float]] = []
    scores: List[List[float]] = []

    r_deltas: List[List[float]] = []
    r_outcome: List[str] = []
    r_match: List[int] = []
    r_viol: List[List[float]] = []
    r_viol_lines = 0       # "r" lines that actually carried "viol"
    r_kyoku: List[int] = []
    r_honba: List[int] = []
    r_id_lines = 0         # "r" lines that carried BOTH "kyoku" and "honba"

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
                        extra = ""
                        if ver == 3:
                            extra = (
                                "  (v3 WEIGHTS can be migrated with "
                                "train/widen4.py; v3 DATA cannot -- the planes and "
                                "scalars are identical, but the river token stream "
                                "'seq' was never recorded, and it is an input the "
                                "encoder cannot invent)"
                            )
                        elif ver == 2:
                            extra = (
                                "  (v2 WEIGHTS can be migrated with train/widen.py "
                                "then train/widen4.py; v2 DATA cannot -- the new "
                                "planes were never recorded)"
                            )
                        raise ValueError(
                            f"{path}:{lineno}: v{ver} data -- re-record with the current "
                            f"engine (this trainer reads feature v{FEATURE_VERSION}: "
                            f"{PLANE_LEN} plane bytes + {SCALAR_LEN * 4} scalar bytes "
                            f"+ up to {SEQ_MAX} river tokens)"
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

                    sq = rec.get("seq")
                    if sq is None:
                        seq_missing += 1
                        tok = np.zeros((0, TOK_BYTES), dtype=np.int8)
                    else:
                        try:
                            tok, n_clamp, n_oor = _decode_seq(sq)
                        except ValueError as e:
                            raise ValueError(f"{path}:{lineno}: {e}") from e
                        seq_clamped += n_clamp
                        seq_oor += n_oor
                    padded = np.zeros((SEQ_MAX, TOK_BYTES), dtype=np.int8)
                    padded[: tok.shape[0]] = tok
                    seqs.append(padded)
                    seq_lens.append(int(tok.shape[0]))

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
                    rk, rh = rec.get("kyoku"), rec.get("honba")
                    if rk is None or rh is None:
                        r_kyoku.append(-1)
                        r_honba.append(-1)
                    else:
                        r_kyoku.append(int(rk))
                        r_honba.append(int(rh))
                        r_id_lines += 1
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

    if seq_missing:
        print(
            f"warning: {seq_missing} of {len(feats)} v{FEATURE_VERSION} decision "
            "line(s) carried no 'seq' field; read as an EMPTY river.  The contract "
            "says a v4 line carries 'seq' (possibly the empty string) -- if this is "
            "not just the pre-first-discard decisions, the recorder is dropping the "
            "token stream and the encoder is being trained on nothing.",
            file=sys.stderr,
        )
    if seq_clamped or seq_oor:
        print(
            f"warning: river tokens: {seq_clamped} dropped by the {SEQ_MAX}-token "
            f"clamp, {seq_oor} with a field outside its range (read as 'no one-hot "
            "bit', per V4_SPEC.md).  Both are DEFINED behaviour, and both mean the "
            "recorder is emitting something the encoder cannot use.",
            file=sys.stderr,
        )

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
        seq=np.stack(seqs).astype(np.int8),
        seq_len=np.asarray(seq_lens, dtype=np.int32),
        round_kyoku=np.asarray(r_kyoku, dtype=np.int32),
        round_honba=np.asarray(r_honba, dtype=np.int32),
        has_round_id=bool(r_deltas) and r_id_lines == len(r_deltas),
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

    v4 widens the tolerance by exactly one alternative: the declared features
    may account for the width either directly (a v3 net) or with the encoder's
    64-wide z block appended (a v4 net).  Both are well-formed nets and
    `train/widen4.py` has to be able to open the first to write the second.
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
    if input_dim not in (width, width + ATTN_D):
        parts = f"{vals['planes']}x{TILE_TYPES} + {vals['scalars']}"
        if oracle:
            parts += f" + {vals['oracle_planes']}x{TILE_TYPES}"
        raise ValueError(
            f"{mpath}: features says {parts} = {width} inputs ({width + ATTN_D} "
            f"with the v{FEATURE_VERSION} attention block) but layer 0 takes "
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
    # "attn" is checked only for a v4-width net, and there it is REQUIRED: the
    # engine's loader derives the expected fc1 width from that key's presence
    # alone, so a 1738-wide manifest without it is one the engine would refuse.
    keys = ("version", "arch", "actions", "blob")
    if has_attn(input_dim):
        keys += ("attn",)
    for key in keys:
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

    The v4 manifest NAMES attn.f32 but this function does not write it: the
    encoder is a separate parameter tree with its own optimizer story, so
    `export_attn(enc, outdir)` is a separate call the trainer makes beside this
    one.  A directory holding a 1738-wide policy.f32 and no attn.f32 is
    therefore a bug, and both loaders say so.
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
