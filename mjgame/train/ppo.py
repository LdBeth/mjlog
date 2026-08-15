#!/usr/bin/env python3
"""PPO (clipped surrogate) on mjgame self-play trajectories.

    python train/ppo.py --data 'runs/ppo/iter3.s*.jsonl' \
        --init weights/ --out weights/ --epochs 3 --batch 4096 --lr 1e-4

One hanchan is one episode.  The reward is a stream of ROUND-level payouts plus
one terminal settlement term, all in 1000-point (uma-net) units:

    R[k][s] = round_deltas[k][s]/1000 - viol_lambda * round_viol[k][s]
    U[m][s] = net[m][s] - sum_k round_deltas[k][s]/1000     <- uma / oka / trunc

The terminal term is not modelled, it is *measured*: the engine's `settlement`
derives `net` from the final scores, so subtracting the raw round sum leaves
whatever uma, return-point offset and truncation it applied, exactly.

WHY THE PENALTY IS ROUND-TIMESTAMPED.  Avoiding 禁じ手 must be a BYPRODUCT of
playing well, never a per-decision shaping term bolted onto the loss: a
per-decision penalty teaches "do not emit this action", which is exactly the
rule-following-by-construction that the dojo engine deliberately refuses to
hard-code.  What the agent is asked to learn instead is that dojo-violating
lines cost 評価点, and 評価点 are money like any other.  So the penalty enters
the reward at the coarsest granularity that still carries CAUSAL TIMING — the
round in which it was incurred.  Charging it only at match end (the old
behaviour, kept as the fallback below) smears a 1st-round 現物切り over all
eight rounds' decisions; charging it per decision would collapse the credit
assignment into a rule table.  Round granularity keeps the credit inside the
hand that earned it and lets the value head explain the rest.

If the dataset's "r" lines lack `viol` (pre-round-viol recorder), ppo.py warns
and folds `violations[m][s]` — the match total — into U instead, which
reproduces the old undiscounted return exactly.  The two are never both
applied; that would double-charge every penalty.

ADVANTAGES are GAE(gamma, lambda) over each (match, seat) decision sequence,
computed in value-head units.  A round's reward is attached to the last
decision that seat made at or before the end of that round, so a penalty is
never credited to a decision that had not happened yet.  With --gamma 1
--gae-lambda 1, A_t telescopes back to (MC return-to-go - V(s_t)) — the
previous behaviour, and the correctness anchor for the whole scan.

IMPORTANT ASSUMPTION.  The trajectories must have been collected by SAMPLING
from softmax(logits, T = 1) of exactly the `--init` weights — the engine's
`--temp=1` with `--weights` pointing at that same directory.  This trainer
recomputes the behaviour log-probs from `--init` rather than reading them off
the file, so a mismatch (different weights, greedy argmax, a different
temperature) silently poisons every importance ratio.  The two forward passes
agree across languages to ~3e-5, which is negligible for the ratio.

The value head arrives from behaviour cloning UNTRAINED (random init), so the
first iteration's advantages are essentially G plus noise; advantage
normalisation is what keeps that from exploding the first update.

The value head predicts G / RETURN_SCALE, not G.  Raw returns are ±60-ish, and
an MSE against them is ~300 while the surrogate is ~0.1 — at that ratio the
value gradient owns the shared trunk and drags the policy arbitrarily far
(the PPO clip bounds only the surrogate term, not trunk drift; measured on
iteration 1: KL 0.31 in the first epoch, eval win rate −3pp).  Dividing the
target by a FIXED constant makes both losses O(1).  The constant is part of
the value head's units and must never change between iterations: V_old from
`--init` is read in the same scaled units the previous run trained.

Two KL backstops guard the update, both against the mean over each minibatch:
past 1.5x --target-kl the policy terms (surrogate + entropy) are dropped and
the remaining minibatches fit the value head only; past 4x the update aborts
outright, since value-only steps still move the policy through the trunk.

THE ORACLE CRITIC.  When the trajectories carry the privileged "o"/"sh" fields
(see common.py), the baseline moves OUT of the policy net entirely and into a
separate 1908-wide critic that sees the policy's features PLUS the opponents'
hands, the hidden remainder and the ura indicators.  This is asymmetric
actor-critic: only the baseline peeks, and it is used only to subtract, so the
policy stays exactly as blind as it will be at inference time while the
variance from "what the wall happened to hold" is explained away instead of
being charged to the policy gradient.  Two things follow, and both are the
point:

  * the policy's in-tensor value head (output 78) STOPS RECEIVING GRADIENT.
    It was the loss that owned the shared trunk -- the whole RETURN_SCALE
    apparatus above exists to keep it from dragging the policy -- and with the
    critic carrying the baseline the trunk is freed for the policy and its
    auxiliary heads.  Output 78 is now vestigial: it drifts with the trunk,
    nothing reads it, the engine ignores it.
  * the critic is trained by plain MSE against the same TD(lambda) targets on
    its OWN optimizer.  It never touches the policy's parameters, so a large
    critic loss cannot move the policy at all -- only through the advantage it
    computes on the NEXT iteration.

AUXILIARY SPEED HEADS.  fc3 grows 24 rows during training (79 -> 103), read as
3 opponents x 8 shanten classes, and is asked to predict each opponent's
current shanten FROM PUBLIC FEATURES ONLY -- the oracle supplies the label, not
the input.  This deliberately shapes the trunk toward 速度読み: in 雀鬼流 the
read that matters is how fast the table is, not precisely what each player is
waiting on, and a trunk that can answer "how close is kamicha" is already
carrying most of what push/fold needs.  `export_weights` slices those 24 rows
off, so the blob the engine loads is unchanged; they persist in aux.f32.

Without the oracle fields, everything above is skipped and the ORIGINAL shared
value-head path runs verbatim -- same losses, same prints, same weights out.

THE RIVER ENCODER (feature v4, train/V4_SPEC.md).  Every decision line now
carries "seq": one token per river entry, all four seats, relative seat order,
each river truncated to its first 24 discards.  A 4-head self-attention encoder
(42 -> 64, learned-query pooling, 64 out) folds that stream into a z the two
nets read as extra input columns:

    policy fc1 input = [planes+scalars 1674][z 64]             = 1738
    critic fc1 input = [planes+scalars 1674][oracle 170][z 64]  = 1908

THE ENCODER BELONGS TO THE POLICY OPTIMIZER, AND ONLY TO IT.  The policy and
the encoder are updated together as one parameter tree (one Adam, one
value_and_grad over the pair), so z is shaped by the surrogate that actually
needs it.  The critic reads STOP-GRADIENT(z): it gets the representation, it
never gets to write it.

That is a training-side decision and it is deliberate.  The aux-coef incident
is the precedent -- an auxiliary loss on a shared trunk, left with a free hand,
annexes capacity from the objective that matters and the symptom shows up
somewhere else entirely -- and the critic is the strongest annexer available
here: it is the only loss in this trainer with a privileged input (the
opponents' actual hands) and a target it can drive to near-zero.  Give it
gradient into the encoder and the encoder learns the features that make ORACLE
regression easy, which is precisely the information the policy will not have at
inference time.  The critic does not need it either: it already sees the truth
the encoder is trying to infer.  So the flow of value is one-way, z is a
POLICY-side summary that the baseline is merely allowed to condition on, and
the critic's own loss remains unable to move a single policy-side parameter --
the same guarantee the separate-optimizer split already gave for the MLP.

MEMORY.  The attention scores are [B, 4, L, L] with L up to 96, so a 4096
minibatch materialises ~600 MB for that tensor alone (plus its backward).  If
the machine starts swapping, halve --batch; nothing about the update depends on
the batch size except the noise in it.
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
    ATTN_BLOB_NAME,
    ATTN_D,
    ATTN_HEAD_DIM,
    ATTN_HEADS,
    ATTN_SCALE,
    AUX_CLASSES,
    AUX_MISSING,
    AUX_OPP,
    CRITIC_BLOB_NAME,
    CRITIC_INPUT,
    CRITIC_MANIFEST_NAME,
    FEATURE_VERSION,
    INPUT_DIM,
    ORACLE_LEN,
    PLANE_SCALAR_DIM,
    RIVER_MAX,
    SEQ_MAX,
    TILE_TYPES,
    TOK_DENSE,
    AttnEncoder,
    CriticNet,
    PolicyValueNet,
    export_attn,
    export_critic,
    export_weights,
    load_attn,
    load_aux,
    load_critic,
    load_trajectories,
    load_weights,
    split_aux,
    split_head,
    widen_for_aux,
)

# Value-head units: V predicts G / RETURN_SCALE.  Frozen across iterations.
RETURN_SCALE = 20.0

# Additive mask for softmax over padded positions.  FINITE on purpose: -inf
# would make a row of all-pad keys (an empty river inside a non-empty batch)
# softmax to NaN, and a NaN cannot be selected away afterwards -- `mx.where`
# would still propagate it through the discarded branch's gradient.  At -1e9 the
# same row softmaxes to a harmless uniform over garbage, exp(-1e9 - max) is
# exactly 0.0 in float32 so masked keys contribute EXACTLY nothing, and the L=0
# case is then fixed up by an honest select.
MASK_NEG = -1e9


# ---------------------------------------------------------------------------
# The river encoder's forward (V4_SPEC.md "Encoder forward" -- the frozen one)
# ---------------------------------------------------------------------------


def token_dense(tok: mx.array) -> mx.array:
    """Packed [B, L, 4] int -> dense [B, L, 42], the spec's expansion exactly.

        concat(onehot34(type), onehot4(seatRel), idx/24, tsumogiri, riichiDecl,
               calledAway)

    The one-hots are equality against an arange, which gives the spec's
    out-of-range rule for free: a `type` of 34 or -1 matches nothing and sets no
    bit, rather than wrapping onto a real tile.  `idx/24` is a float32 divide of
    an exactly-representable small integer, which is bit-identical to the
    spec's "double, rounded once" over the whole int8 range (checked: all 256
    values agree in both numpy and mlx).
    """
    ttype = tok[..., 0]
    srel = tok[..., 1]
    idx = tok[..., 2]
    flags = tok[..., 3]
    oh_type = (ttype[..., None] == mx.arange(TILE_TYPES)).astype(mx.float32)
    oh_seat = (srel[..., None] == mx.arange(4)).astype(mx.float32)
    pos = (idx.astype(mx.float32) / float(RIVER_MAX))[..., None]
    bits = mx.stack(
        [((flags // (1 << b)) % 2).astype(mx.float32) for b in range(3)], axis=-1
    )
    return mx.concatenate([oh_type, oh_seat, pos, bits], axis=-1)


def encode_tokens(enc: AttnEncoder, tok: mx.array, lens: mx.array) -> mx.array:
    """[B, L, 4] packed tokens + [B] lengths -> z [B, 64].

    The spec is written for one sequence at a time; this is that, batched, and
    the batching is the only thing that needs an argument.  `tok` is padded to
    the batch's LONGEST river, and every pad position is removed three times
    over, because each removal covers a different way it could leak:

      * as an input   -- the dense row is zeroed, so a pad token never even
                         reaches W_in with the all-zeros token's one-hots;
      * as a KEY      -- masked out of the attention softmax, so no real token
                         attends to it;
      * as a QUERY    -- masked out of the pooling softmax, so its (finite,
                         garbage) m never reaches p.

    Masked positions get weight exp(-1e9 - max) = 0.0 exactly, and adding exact
    zeros to a float sum cannot change it, so a padded batch and a batch of one
    compute the same z to within matmul reduction order (asserted at 1e-6 in the
    padding test).

    L = 0 for a whole sample -- every decision before the first discard of the
    hand -- is the spec's special case: p = 0, so z = bz.  It is applied as a
    select over the batch rather than a branch, so a mixed batch is one graph.
    """
    B, L = int(tok.shape[0]), int(tok.shape[1])
    bz = enc.wz.bias
    if L == 0:
        return mx.broadcast_to(bz, (B, ATTN_D))

    valid = mx.arange(L)[None, :] < lens.reshape(-1, 1)            # [B, L] bool
    x = token_dense(tok) * valid[..., None].astype(mx.float32)     # [B, L, 42]

    h = nn.relu(enc.w_in(x))                                       # [B, L, 64]

    def split_heads(t: mx.array) -> mx.array:
        # Contiguous 16-column slices of the 64, per the spec's head split.
        return t.reshape(B, L, ATTN_HEADS, ATTN_HEAD_DIM).transpose(0, 2, 1, 3)

    q = split_heads(enc.wq(h))                                     # [B, H, L, 16]
    k = split_heads(enc.wk(h))
    v = split_heads(enc.wv(h))

    scores = (q @ k.transpose(0, 1, 3, 2)) * ATTN_SCALE            # [B, H, L, L]
    scores = mx.where(valid[:, None, None, :], scores, MASK_NEG)
    attn = mx.softmax(scores, axis=-1)
    o = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, ATTN_D)     # concat heads
    m = enc.wo(o)                                                  # [B, L, 64]

    pool = mx.where(valid, (m * enc.u).sum(axis=-1), MASK_NEG)     # [B, L]
    alpha = mx.softmax(pool, axis=-1)
    p = (alpha[..., None] * m).sum(axis=1)                         # [B, 64]

    z = enc.wz(p)
    empty = (lens.reshape(-1, 1) == 0)
    return mx.where(empty, mx.broadcast_to(bz, (B, ATTN_D)), z)


def seq_batch(SEQ: np.ndarray, LEN: np.ndarray, sel):
    """One minibatch of tokens, trimmed to ITS longest river.

    `SEQ` is stored [n, 96, 4] so a minibatch is a plain gather, but almost no
    batch actually needs 96 columns and the attention cost is quadratic in them,
    so the padding is cut back to the batch maximum before anything is built.
    """
    lens = LEN[sel]
    lmax = int(lens.max()) if lens.size else 0
    return (
        mx.array(SEQ[sel][:, :lmax].astype(np.int32)),
        mx.array(lens.astype(np.int32)),
    )


class Actor(nn.Module):
    """The policy net and its river encoder as ONE parameter tree.

    They are updated by one optimizer from one loss, so they are one module:
    `nn.value_and_grad(actor, ...)` then returns gradients for both, and there
    is no way to accidentally step one without the other.  The critic is NOT in
    here -- it has its own optimizer and its own loss, and only ever sees a
    stop-gradient copy of this encoder's output.
    """

    def __init__(self, net: PolicyValueNet, enc: AttnEncoder) -> None:
        super().__init__()
        self.net = net
        self.enc = enc

    def __call__(self, x: mx.array, tok: mx.array, lens: mx.array):
        """(network output, z) for a minibatch of dense features ++ tokens."""
        z = encode_tokens(self.enc, tok, lens)
        return self.net(mx.concatenate([x, z], axis=1)), z


# ---------------------------------------------------------------------------
# Masked policy helpers
# ---------------------------------------------------------------------------


def masked_logp(logits: mx.array, mask: mx.array) -> mx.array:
    """Log-softmax over the legal support; illegal entries come back -inf."""
    if logits.shape[-1] != ACTIONS:
        logits = logits[..., :ACTIONS]
    neg_inf = mx.array(float("-inf"), dtype=logits.dtype)
    masked = mx.where(mask, logits, neg_inf)
    return masked - mx.logsumexp(masked, axis=-1, keepdims=True)


def gather(logp: mx.array, action: mx.array) -> mx.array:
    """log pi(a|s) for the taken action, [B]."""
    idx = action.reshape(-1, 1).astype(mx.int32)
    return mx.take_along_axis(logp, idx, axis=-1).reshape(-1)


def masked_entropy(logp: mx.array, mask: mx.array) -> mx.array:
    """Mean entropy of the masked softmax.

    `logp` is -inf off the support, so p*logp there is 0*(-inf) = NaN.  Both
    factors are therefore *selected* to 0 with `where` before they ever meet:
    a select has no NaN to propagate, and its gradient on the discarded branch
    is a clean zero.
    """
    zero = mx.zeros_like(logp)
    safe = mx.where(mask, logp, zero)
    p = mx.where(mask, mx.exp(safe), zero)
    return -(p * safe).sum(axis=-1).mean()


# ---------------------------------------------------------------------------
# Auxiliary shanten heads
# ---------------------------------------------------------------------------


def aux_ce_acc(aux_logits: mx.array, labels: mx.array, valid: mx.array):
    """(masked mean cross-entropy, masked top-1 accuracy) over 3 x 8 heads.

    `aux_logits` is [B, 3, 8], `labels` int [B, 3] already clipped into 0..7,
    `valid` float [B, 3] with 0 where the line carried no label.  The mean is
    over VALID (sample, opponent) pairs, so a batch that is half unlabelled
    contributes half as many terms rather than half-strength ones; a batch with
    none at all yields a hard zero and therefore a zero gradient.
    """
    logp = aux_logits - mx.logsumexp(aux_logits, axis=-1, keepdims=True)
    idx = labels.reshape(-1, AUX_OPP, 1).astype(mx.int32)
    picked = mx.take_along_axis(logp, idx, axis=-1).reshape(-1, AUX_OPP)
    denom = mx.maximum(valid.sum(), 1.0)
    l_aux = -(picked * valid).sum() / denom
    hit = (mx.argmax(aux_logits, axis=-1) == labels).astype(mx.float32)
    acc = (hit * valid).sum() / denom
    return l_aux, acc


def xo_batch(X, O, sel, z: mx.array) -> mx.array:
    """[X ++ oracle ++ z] for one minibatch, assembled on the fly.

    Materialising the full 1908-wide matrix would double the trainer's peak
    memory over a big collection batch for no benefit -- the critic only ever
    sees it a minibatch at a time.

    `z` is passed in ALREADY DETACHED (see `detached_z`).  The critic's own
    `value_and_grad` could not reach the encoder in any case, since it
    differentiates the critic's parameters alone; the stop-gradient is stated
    here as well because "the critic does not write the encoder" is a design
    claim, not an accident of which module the optimizer was handed.
    """
    return mx.concatenate([mx.array(X[sel]), mx.array(O[sel]), z], axis=1)


def detached_z(enc: AttnEncoder, tok: mx.array, lens: mx.array) -> mx.array:
    """The encoder's output as a CONSTANT: evaluated, cut off from any tape."""
    z = mx.stop_gradient(encode_tokens(enc, tok, lens))
    mx.eval(z)
    return z


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def decision_round(traj):
    """Absolute round index per decision, [n] int64.

    `traj.rnd` is NOT it.  The recorder buffers a match's round results and
    writes every "r" line at match end, after all of that match's "d" lines, so
    the loader's running round counter is frozen at the match's FIRST round for
    every decision in it.  The join has to be rebuilt, and there are two ways to
    do it depending on what the dataset carries.

    CURRENT DATA.  Every "r" line names its own round with the same
    (kyoku, honba) its "d" lines carry, so the join is a direct lookup on
    (match, kyoku, honba) -- see `_round_index_by_id`.  This is the only correct
    way when the population is MIXED: recording wraps just the neural seats, so
    a round can end with no "d" line at all (an opponent wins before any
    recorded seat acts), and such a round is invisible to any positional scheme.

    OLD DATA (no "kyoku"/"honba" on the "r" lines).  All that is left is
    position: (kyoku, honba) is constant through a round, changes at every round
    boundary (honba increments on renchan, kyoku advances and honba resets
    otherwise) and so never repeats non-consecutively within a match, so
    counting those blocks off the match's first round recovers the index --
    PROVIDED every round contains at least one decision.  The block count is
    checked against the match's "r" line count, which is both the assertion that
    the reconstruction is right and the detector for the decision-less round it
    cannot handle.  There is deliberately no attempt to patch that case here:
    consecutive decision-less rounds are ambiguous in principle, and guessing
    would misalign credit silently.  Re-record with the current engine.
    """
    if traj.has_round_id:
        return _round_index_by_id(traj)
    n = len(traj)
    m, ky, hb = traj.match, traj.kyoku, traj.honba
    new_blk = np.ones(n, dtype=bool)
    if n > 1:
        new_blk[1:] = (m[1:] != m[:-1]) | (ky[1:] != ky[:-1]) | (hb[1:] != hb[:-1])
    blk = np.cumsum(new_blk) - 1                       # global block id

    n_matches = traj.net.shape[0]
    base_blk = np.zeros(n_matches, dtype=np.int64)
    first = np.zeros(n_matches, dtype=bool)
    for i in range(n):
        mi = int(m[i])
        if not first[mi]:
            first[mi] = True
            base_blk[mi] = blk[i]

    idx = traj.rnd.astype(np.int64) + (blk - base_blk[m.astype(np.int64)])

    # Cross-check: blocks per match must equal that match's round-line count.
    n_blocks = np.zeros(n_matches, dtype=np.int64)
    for mi in np.unique(m):
        n_blocks[int(mi)] = int(np.unique(blk[m == mi]).size)
    n_rlines = np.bincount(traj.round_match, minlength=n_matches)[:n_matches]
    bad = np.where(first & (n_blocks != n_rlines))[0]
    if bad.size:
        raise ValueError(
            f"round reconstruction disagrees with the 'r' lines for "
            f"{bad.size} match(es) (first: match {int(bad[0])}, "
            f"{int(n_blocks[bad[0]])} (kyoku,honba) block(s) vs "
            f"{int(n_rlines[bad[0]])} round line(s)) -- if the data was "
            f"collected with a MIXED population, this is the decision-less "
            f"round that positional reconstruction cannot see; re-record with "
            f"an engine that writes 'kyoku'/'honba' on the 'r' lines"
        )
    return idx


def _round_index_by_id(traj):
    """`decision_round` for data whose "r" lines name their own round.

    A round is identified by (match, kyoku, honba), which the "r" line and its
    "d" lines both carry verbatim, so the mapping is a lookup and needs no
    inference at all.  Two things are still checked, because a bad label is
    worse than no label:

      * the pairs must be UNIQUE within a match, or the identifier does not
        identify (the engine guarantees this -- honba increments on renchan and
        on a draw, kyoku advances otherwise -- so a duplicate means the recorder
        is broken);
      * every decision's pair must appear among its match's "r" lines, i.e. the
        decision pairs are a SUBSET of the round pairs.  The reverse is NOT
        required: a round with no decision in it is normal under a mixed
        population, and its reward is attached by `gae_returns` to the last
        decision at-or-before it (see there for the leading-hole rule).
    """
    m_r = traj.round_match.astype(np.int64)
    ky_r = traj.round_kyoku.astype(np.int64)
    hb_r = traj.round_honba.astype(np.int64)
    m_d = traj.match.astype(np.int64)
    ky_d = traj.kyoku.astype(np.int64)
    hb_d = traj.honba.astype(np.int64)

    if m_r.size == 0:
        raise ValueError("no 'r' lines at all -- no round can be attached")

    # A single int64 key per (match, kyoku, honba).  All three are non-negative
    # and small; the radices come from the data so the packing cannot collide.
    nk = int(max(ky_r.max(), ky_d.max(initial=0))) + 1
    nh = int(max(hb_r.max(), hb_d.max(initial=0))) + 1
    if min(ky_r.min(), hb_r.min()) < 0:
        raise ValueError("an 'r' line carries a negative kyoku/honba")

    def pack(m, k, h):
        return (m * nk + k) * nh + h

    rkey = pack(m_r, ky_r, hb_r)
    dkey = pack(m_d, ky_d, hb_d)

    order = np.argsort(rkey, kind="stable")
    skey = rkey[order]
    dup = np.flatnonzero(skey[1:] == skey[:-1])
    if dup.size:
        i = int(order[dup[0]])
        raise ValueError(
            f"'r' lines repeat a (kyoku,honba) inside one match "
            f"({dup.size} duplicate(s); first: match {int(m_r[i])}, kyoku "
            f"{int(ky_r[i])} honba {int(hb_r[i])}) -- the round labels do not "
            f"identify a round, so no join is possible"
        )

    pos = np.clip(np.searchsorted(skey, dkey), 0, skey.size - 1)
    hit = skey[pos] == dkey
    if not np.all(hit):
        miss = np.flatnonzero(~hit)
        i = int(miss[0])
        bad_m = np.unique(m_d[miss])
        raise ValueError(
            f"decision (kyoku,honba) is not among the 'r' lines for "
            f"{bad_m.size} match(es) (first: match {int(m_d[i])}, a decision "
            f"at kyoku {int(ky_d[i])} honba {int(hb_d[i])} with no matching "
            f"round line) -- the decisions must be a SUBSET of the rounds"
        )
    return order[pos].astype(np.int64)


def round_rewards(traj, viol_lambda: float):
    """(R, U, starts, ends) — the episode's reward decomposition, in uma points.

    `R[k][s]` is what round k pays seat s and `U[m][s]` the terminal remainder
    (uma / oka / sub-1000 truncation), so an episode's total return is
    `sum_k R[k][s] + U[m][s]` — which is `net[m][s]` minus the dojo charge, no
    matter which of the two penalty timings is in force.

    Rounds of one match are contiguous and in play order in `round_deltas`;
    contiguity is verified, not assumed.  `starts`/`ends` are the first and last
    round index of each match, -1 for a match with no closed round.

    The dojo charge is applied ONCE.  With per-round data it rides on R, where
    it is timestamped to the hand that incurred it; without, it is folded into
    U, which reproduces the old match-terminal behaviour exactly.
    """
    rd = traj.round_deltas.astype(np.float64) / 1000.0    # [r, 4]
    rmatch = traj.round_match
    n_rounds = rd.shape[0]
    n_matches = traj.net.shape[0]

    if n_rounds and np.any(np.diff(rmatch) < 0):
        raise ValueError("round_match is not non-decreasing -- matches are interleaved")

    starts = np.full(n_matches, -1, dtype=np.int64)
    ends = np.full(n_matches, -1, dtype=np.int64)
    for k in range(n_rounds):
        m = int(rmatch[k])
        if starts[m] < 0:
            starts[m] = k
        ends[m] = k

    match_total = np.zeros((n_matches, 4), dtype=np.float64)
    for m in range(n_matches):
        if starts[m] >= 0:
            match_total[m] = rd[starts[m] : ends[m] + 1].sum(axis=0)
    U = traj.net.astype(np.float64) - match_total         # [m, 4]

    if traj.has_round_viol:
        R = rd - viol_lambda * traj.round_viol.astype(np.float64)
    else:
        R = rd.copy()
        U = U - viol_lambda * traj.violations.astype(np.float64)
    return R, U, starts, ends


def usable(traj, i_rnd):
    """Decisions whose round actually closed (a truncated file has some that did not)."""
    n_rounds = traj.round_deltas.shape[0]
    keep = (i_rnd >= 0) & (i_rnd < n_rounds)
    keep[keep] &= traj.round_match[i_rnd[keep]] == traj.match[keep]
    return keep


def episode_groups(match_of, seat_of):
    """Decision positions grouped per (match, seat), each in temporal order.

    File order is play order, so a stable sort by (match, seat) leaves every
    group's members in the order that seat actually acted.
    """
    if match_of.size == 0:
        return []
    key = match_of.astype(np.int64) * 4 + seat_of.astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    cuts = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1], [True])))
    return [order[cuts[i] : cuts[i + 1]] for i in range(cuts.size - 1)]


def gae_returns(traj, keep, i_rnd, R, U, V_old, starts, ends, gamma, lam):
    """(A, Gv, G) — GAE advantages, TD(lambda) value targets, MC returns-to-go.

    Everything but `G` is in VALUE-HEAD UNITS (reward / RETURN_SCALE), which is
    what `V_old` already predicts; `G` is handed back in raw uma points purely
    so the diagnostic print keeps meaning the same thing it always did.

    REWARD ATTRIBUTION.  Each (match, seat) episode is a sequence of decisions
    whose round indices are non-decreasing.  The loop walks EVERY round of the
    match, `starts[m]..ends[m]`, not just the rounds this episode has decisions
    in, so no round's payout can be dropped.  Round k's reward goes to the last
    decision that seat made at or before the END of round k, i.e. the last one
    with round <= k; a round in which the seat never acted therefore pays out at
    its previous decision rather than vanishing.

    Rewards with no decision at-or-before them attach to the seat's FIRST
    decision instead (`max(p, 0)`), and the terminal U rides on the last.  That
    case is not hypothetical any more.  With a mixed population the recorder
    wraps only the neural seats, so a whole round — including the match's FIRST
    round, and any run of consecutive rounds — can close before a recorded seat
    ever acts.  Those rounds' payouts are then LUMPED onto the first recorded
    decision of the episode, which is the deliberate choice: the points are
    real, they belong to this hanchan's return, and the earliest decision is the
    earliest state from which the agent could have influenced anything.  Pushing
    them into U instead would move them to the LAST decision, i.e. discount them
    as if they were earned at the end, and dropping them would break the
    identity `sum_k R + U == net` that the assertion below checks.  Under
    gamma = 1 the placement is exactly neutral in the return; under gamma < 1 it
    is the least-wrong of the three.

    The per-episode total is then asserted against the episode's own return,
    which is what catches an attribution bug before it silently becomes a bad
    gradient.

    GAE.  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) with V(terminal) = 0, and
    A_t = sum_l (gamma*lambda)^l delta_{t+l} by a reversed scan.  At gamma = 1,
    lambda = 1 the deltas telescope and A_t = (sum_{t' >= t} r_t') - V(s_t):
    exactly the Monte-Carlo advantage this trainer used before GAE existed.
    """
    kept = np.flatnonzero(keep)
    n = kept.size
    m_of = traj.match[kept].astype(np.int64)
    s_of = traj.seat[kept].astype(np.int64)
    r_of = i_rnd[kept].astype(np.int64)

    A = np.zeros(n, dtype=np.float64)
    Gv = np.zeros(n, dtype=np.float64)
    G = np.zeros(n, dtype=np.float64)
    if n == 0:
        return A, Gv, G

    Rs = R / RETURN_SCALE
    Us = U / RETURN_SCALE
    step = np.zeros(n, dtype=np.float64)        # per-decision reward, scaled
    v = V_old.astype(np.float64)

    groups = episode_groups(m_of, s_of)

    for grp in groups:
        m, s = int(m_of[grp[0]]), int(s_of[grp[0]])
        seq_rnd = r_of[grp]
        if seq_rnd.size > 1 and np.any(np.diff(seq_rnd) < 0):
            raise ValueError(
                f"match {m} seat {s}: decisions are not in round order -- file "
                "order is not play order, reward attribution would be wrong"
            )
        lo, hi = int(starts[m]), int(ends[m])
        want = 0.0
        if lo >= 0:
            for k in range(lo, hi + 1):
                p = int(np.searchsorted(seq_rnd, k, side="right")) - 1
                step[grp[max(p, 0)]] += Rs[k, s]
                want += Rs[k, s]
        step[grp[-1]] += Us[m, s]
        want += Us[m, s]
        got = float(step[grp].sum())
        if abs(got - want) > 1e-4:
            raise AssertionError(
                f"match {m} seat {s}: attached reward {got:+.6f} != episode "
                f"return {want:+.6f} (scaled units)"
            )

    for grp in groups:
        adv = 0.0
        ret = 0.0
        v_next = 0.0                            # V(terminal) = 0
        for j in range(grp.size - 1, -1, -1):
            t = grp[j]
            delta = step[t] + gamma * v_next - v[t]
            adv = delta + gamma * lam * adv
            ret = step[t] + ret                 # undiscounted MC return-to-go
            A[t] = adv
            G[t] = ret
            v_next = v[t]

    Gv = A + v                                  # TD(lambda) value targets
    return A, Gv, G * RETURN_SCALE


# ---------------------------------------------------------------------------
# Behaviour policy
# ---------------------------------------------------------------------------


def behaviour_pass(actor: Actor, X, SEQ, LEN, mask, action, batch: int):
    """(old_logp, V_old) from the frozen collection policy, as numpy.

    The collection policy is the MLP *and* the encoder that produced the z it
    was conditioned on, so both come from `--init`: recomputing the behaviour
    log-probs with a different encoder would poison every importance ratio in
    exactly the way the module docstring warns about for the weights.
    """
    n = X.shape[0]
    lp = np.empty(n, dtype=np.float32)
    vv = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        sel = slice(i, i + batch)
        xb = mx.array(X[sel])
        mb = mx.array(mask[sel])
        ab = mx.array(action[sel].astype(np.int32))
        tok, lens = seq_batch(SEQ, LEN, sel)
        out, _ = actor(xb, tok, lens)
        logits, v = split_head(out)
        picked = gather(masked_logp(logits, mb), ab)
        mx.eval(picked, v)
        lp[sel] = np.array(picked)
        vv[sel] = np.array(v)
    return lp, vv


def critic_pass(critic: CriticNet, enc: AttnEncoder, X, O, SEQ, LEN, batch: int):
    """V_old from the FROZEN --critic-init net over [X ++ oracle ++ z], as numpy.

    A fresh critic makes this noise, exactly as the value head was noise on
    iteration 1 of the shared-head path; advantage normalisation below is what
    keeps that first update scaled like a policy-gradient step regardless.
    """
    n = X.shape[0]
    vv = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        sel = slice(i, i + batch)
        tok, lens = seq_batch(SEQ, LEN, sel)
        v = critic(xo_batch(X, O, sel, detached_z(enc, tok, lens)))
        mx.eval(v)
        vv[sel] = np.array(v)
    return vv


def main() -> None:
    ap = argparse.ArgumentParser(description="PPO on mjgame self-play trajectories")
    ap.add_argument("--data", required=True, nargs="+", help="glob(s) for trajectory JSONL")
    ap.add_argument("--init", required=True, help="weights the rollouts were COLLECTED with")
    ap.add_argument("--out", required=True, help="where to export the updated weights")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--critic-init",
        default=None,
        dest="critic_init",
        help="directory holding critic.json/critic.f32 for the oracle critic "
        "(default: --init); a missing pair means a fresh critic",
    )
    ap.add_argument("--clip", type=float, default=0.2, help="PPO ratio clip epsilon")
    ap.add_argument(
        "--vf",
        type=float,
        default=0.5,
        help="value loss coefficient (shared-head fallback only -- the oracle "
        "critic trains on its own optimizer and its own pure MSE)",
    )
    ap.add_argument(
        "--aux-coef",
        type=float,
        default=0.5,
        dest="aux_coef",
        help="weight of the auxiliary opponent-shanten loss on the policy net",
    )
    ap.add_argument("--ent", type=float, default=0.01, help="entropy bonus coefficient")
    ap.add_argument(
        "--viol-lambda",
        type=float,
        default=1.0,
        dest="viol_lambda",
        help="uma-points charged per 雀鬼流 penalty point, in the round that incurred it",
    )
    ap.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="reward discount across a hanchan's rounds (1.0 = undiscounted)",
    )
    ap.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        dest="gae_lambda",
        help="GAE trace decay; 1.0 makes the advantage the plain Monte-Carlo one",
    )
    ap.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        dest="target_kl",
        help="approx-KL budget per update; 1.5x freezes the policy terms, 4x aborts",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    traj = load_trajectories(args.data)
    print(
        f"loaded {len(traj)} decisions from {traj.net.shape[0]} match(es), "
        f"{traj.round_deltas.shape[0]} round(s) in {time.time() - t0:.1f}s"
    )

    dense_dim = int(traj.X.shape[1])
    if dense_dim != PLANE_SCALAR_DIM:
        raise SystemExit(
            f"data is {dense_dim} wide but feature v{FEATURE_VERSION}'s planes ++ "
            f"scalars is {PLANE_SCALAR_DIM} -- common.py's contract constants and "
            "the engine's encoder disagree"
        )

    model = load_weights(args.init)
    if model.input_dim != INPUT_DIM:
        raise SystemExit(
            f"--init net takes {model.input_dim} inputs but feature "
            f"v{FEATURE_VERSION} is {dense_dim} dense ++ {ATTN_D} from the river "
            f"encoder = {INPUT_DIM}.\n"
            f"  {args.init} は旧特徴量の重みです / that weight set predates the "
            f"current feature layout.\n"
            f"  Migrate it, function-preserving, with:\n"
            f"      python train/widen4.py --in {args.init} --out <v{FEATURE_VERSION}-dir>\n"
            f"  (the loaders keep reading historical weight sets on purpose, so "
            f"this check -- not load_weights -- is what stops the mismatch)"
        )

    # The encoder that produced the z the rollouts were conditioned on.  It is
    # not optional and it is not initialisable here: a fresh encoder would make
    # every recomputed behaviour log-prob a log-prob of a policy that never
    # played, and the importance ratios would be silently meaningless.
    enc = load_attn(args.init)
    if enc is None:
        raise SystemExit(
            f"--init {args.init} has a v{FEATURE_VERSION} policy ({INPUT_DIM} "
            f"inputs) but no {ATTN_BLOB_NAME}.\n"
            f"  The 64 z columns are fed by an encoder that has to be the one the "
            f"rollouts used; starting a fresh one here would poison every "
            f"importance ratio.\n"
            f"  `python train/widen4.py --in <v3-dir> --out {args.init}` writes both."
        )
    actor = Actor(model, enc)
    tok_pct = 100.0 * float((traj.seq_len > 0).mean())
    print(
        f"encoder: {ATTN_BLOB_NAME} from {args.init} "
        f"({TOK_DENSE}->{ATTN_D}, {ATTN_HEADS} heads x {ATTN_HEAD_DIM}); "
        f"rivers {traj.seq_len.mean():.1f} tokens mean, {int(traj.seq_len.max())} max "
        f"of {SEQ_MAX}, {tok_pct:.1f}% of decisions non-empty"
    )

    # ---- baseline selection -------------------------------------------------
    # The oracle path needs the privileged fields on EVERY decision line; a
    # partially-labelled batch would train the critic on two different input
    # distributions, so the loader's all-or-nothing flag decides, and anything
    # short of all of it falls back to the shared value head end to end.
    use_oracle = traj.has_oracle
    critic = None
    critic_dir = args.critic_init or args.init
    if use_oracle:
        aux0 = load_aux(args.init)
        widen_for_aux(model, aux0, seed=args.seed)
        print(
            f"aux heads: {'loaded from ' + args.init if aux0 is not None else 'fresh random init'} "
            f"({AUX_OPP} opponents x {AUX_CLASSES} shanten classes, coef {args.aux_coef})"
        )
        critic = load_critic(critic_dir)
        if critic is None:
            critic = CriticNet(CRITIC_INPUT)
            mx.eval(critic.parameters())
            print(
                f"critic: no {CRITIC_MANIFEST_NAME} in {critic_dir} -- fresh init "
                f"({CRITIC_INPUT} inputs); V_old is noise this iteration"
            )
        else:
            print(f"critic: loaded from {critic_dir} ({critic.input_dim} inputs)")
        if critic.input_dim != CRITIC_INPUT:
            raise SystemExit(
                f"critic takes {critic.input_dim} inputs but the data gives "
                f"{dense_dim} + {ORACLE_LEN} + {ATTN_D} = {CRITIC_INPUT}.\n"
                f"  {critic_dir} の critic は旧特徴量です / that critic predates the "
                f"current feature layout.\n"
                f"  `python train/widen4.py --in {critic_dir} --out <v{FEATURE_VERSION}-dir>` "
                f"migrates policy, aux, critic and the encoder together;\n"
                f"  or delete {CRITIC_MANIFEST_NAME}/{CRITIC_BLOB_NAME} to start the "
                f"critic from a fresh init."
            )
    else:
        print(
            "WARNING: the trajectories carry no oracle side channel ('o'/'sh' on "
            "every 'd' line) -- falling back to the SHARED VALUE HEAD baseline "
            "and training no auxiliary shanten heads.  This is the pre-oracle "
            "behaviour, not a degraded one, but the policy gradient keeps the "
            "hand-luck variance a privileged critic would have absorbed; "
            "re-record with an engine that emits 'o'/'sh' to enable it."
        )

    if not traj.has_round_viol:
        print(
            "warning: no per-round 'viol' on the 'r' lines -- charging the dojo "
            "penalty at MATCH END from `violations` instead (the pre-round-viol "
            "behaviour).  Credit assignment is smeared over the whole hanchan; "
            "re-record with the current engine to timestamp it per round."
        )
    if not traj.has_round_id:
        print(
            "warning: the 'r' lines do not name their round ('kyoku'/'honba') -- "
            "falling back to POSITIONAL round reconstruction, which is only "
            "correct if every round contains at least one recorded decision.  "
            "Mixed-population data (recording wraps the neural seats only) can "
            "violate that; re-record with the current engine."
        )
    R, U, r_starts, r_ends = round_rewards(traj, args.viol_lambda)

    i_rnd = decision_round(traj)
    keep = usable(traj, i_rnd)
    dropped = int((~keep).sum())
    if dropped:
        print(f"warning: dropping {dropped} decision(s) whose round never closed")
    X, mask, action = traj.X[keep], traj.mask[keep], traj.action[keep]
    SEQ, LEN = traj.seq[keep], traj.seq_len[keep]
    n = X.shape[0]
    if n == 0:
        raise SystemExit("no usable decisions after return attribution")

    # Oracle side channel, if this dataset has one.  `aux_valid` is 0 for a
    # missing label, which is the sentinel and not a shanten value; the clip
    # would otherwise turn -9 into a confident "tenpai" target.
    if use_oracle:
        O = traj.oracle[keep]
        sh = traj.opp_shanten[keep]
        aux_lab = np.clip(sh, 0, AUX_CLASSES - 1).astype(np.int32)
        aux_valid = (sh != AUX_MISSING).astype(np.float32)
    else:
        O = aux_lab = aux_valid = None

    # Behaviour log-probs, from the UNMODIFIED --init net (widening fc3 leaves
    # the first 79 rows alone, so the policy it computes is unchanged).  Taken
    # before the optimizer exists so the training net starts from exactly these
    # weights; frozen numpy from here on.  The baseline it returns is used only
    # by the shared-head path -- under the oracle critic V_old comes from the
    # frozen critic instead, and the policy's own value head is never read.
    tb = time.time()
    old_logp_np, V_old = behaviour_pass(actor, X, SEQ, LEN, mask, action, args.batch)
    if use_oracle:
        V_old = critic_pass(critic, enc, X, O, SEQ, LEN, args.batch)

    # Everything the losses see is in value-head units (reward / RETURN_SCALE),
    # which is what V_old already predicts; `G` comes back in raw uma points for
    # the print only.  `Gv = A + V_old` is the TD(lambda) value target.
    A, Gv_all, G = gae_returns(
        traj, keep, i_rnd, R, U, V_old, r_starts, r_ends, args.gamma, args.gae_lambda
    )
    A = A.astype(np.float32)
    Gv = Gv_all.astype(np.float32)

    print(
        f"returns  mean {G.mean():+.3f}  std {G.std():.3f}  "
        f"[{G.min():+.1f}, {G.max():+.1f}]   "
        f"V_old mean {V_old.mean():+.3f} (x{RETURN_SCALE:g})  ({time.time() - tb:.1f}s)"
    )
    print(
        f"gae  gamma={args.gamma:g}  lambda={args.gae_lambda:g}  "
        f"penalty={'per-round' if traj.has_round_viol else 'match-terminal'}  "
        f"baseline={'oracle-critic' if use_oracle else 'shared-head'}  "
        f"adv(pre-norm) mean {A.mean():+.4f}  std {A.std():.4f}"
    )
    if use_oracle:
        lab_pct = 100.0 * float(aux_valid.mean())
        print(
            f"oracle  {ORACLE_LEN} privileged inputs/decision  "
            f"critic {critic.input_dim}->512->256->1  "
            f"aux labels {lab_pct:.1f}% of {n * AUX_OPP} (opponent, decision) pairs"
        )

    # The value head is random-init out of behaviour cloning, so V_old is noise
    # on iteration 1.  Normalising the advantage is what keeps that first update
    # scaled like a policy-gradient step instead of like the raw ±60 returns.
    A = (A - A.mean()) / max(float(A.std()), 1e-8)

    print(
        f"ppo  n={n}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}  "
        f"clip={args.clip}  ent={args.ent}  viol_lambda={args.viol_lambda}  "
        + (
            f"aux_coef={args.aux_coef}  [oracle critic: own Adam, pure MSE; "
            "--vf unused, policy value head untrained]"
            if use_oracle
            else f"vf={args.vf}  [shared value head]"
        )
    )

    opt = optim.Adam(learning_rate=args.lr)
    copt = optim.Adam(learning_rate=args.lr) if use_oracle else None

    # `pi_coef` gates the policy terms (surrogate + entropy): 1.0 normally, 0.0
    # once the KL budget is spent, which turns the step into a pure value fit.
    def loss_fn(m, xb, tokb, lenb, mb, ab, oldb, adv, gb, pi_coef):
        out, _ = m(xb, tokb, lenb)
        logits, v = split_head(out)
        lp_all = masked_logp(logits, mb)
        lp = gather(lp_all, ab)
        ratio = mx.exp(lp - oldb)
        l_pi = -mx.minimum(
            ratio * adv,
            mx.clip(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv,
        ).mean()
        l_v = mx.square(v - gb).mean()
        ent = masked_entropy(lp_all, mb)
        total = pi_coef * (l_pi - args.ent * ent) + args.vf * l_v
        kl = (oldb - lp).mean()
        clipfrac = (mx.abs(ratio - 1.0) > args.clip).astype(mx.float32).mean()
        return total, (l_pi, l_v, ent, kl, clipfrac)

    def policy_loss_fn(m, xb, tokb, lenb, mb, ab, oldb, adv, lab, valid):
        """Oracle path: surrogate + entropy + auxiliary shanten, NO value term.

        The baseline lives in the critic, so nothing here reads output 78 and
        no gradient reaches it -- the policy's own value head is vestigial from
        this point on and drifts only as the shared trunk moves under it.

        This is also the ONLY loss the river encoder ever sees: `m` is the
        `Actor`, so the gradient of the surrogate, the entropy bonus and the
        auxiliary heads flows back through z into W_in/Wq/Wk/Wv/Wo/u/Wz, and
        nothing else does.
        """
        out, _ = m(xb, tokb, lenb)
        lp_all = masked_logp(out, mb)
        lp = gather(lp_all, ab)
        ratio = mx.exp(lp - oldb)
        l_pi = -mx.minimum(
            ratio * adv,
            mx.clip(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv,
        ).mean()
        ent = masked_entropy(lp_all, mb)
        l_aux, a_acc = aux_ce_acc(split_aux(out), lab, valid)
        total = l_pi - args.ent * ent + args.aux_coef * l_aux
        kl = (oldb - lp).mean()
        clipfrac = (mx.abs(ratio - 1.0) > args.clip).astype(mx.float32).mean()
        return total, (l_pi, l_aux, a_acc, ent, kl, clipfrac)

    def critic_loss_fn(c, xob, gb):
        """Plain MSE against the same TD(lambda) targets, on the critic alone."""
        return mx.square(c(xob) - gb).mean()

    # `actor`, not `model`: the encoder trains with the policy, on this optimizer.
    grad_fn = nn.value_and_grad(actor, policy_loss_fn if use_oracle else loss_fn)
    cgrad_fn = nn.value_and_grad(critic, critic_loss_fn) if use_oracle else None

    frozen = False
    aborted = False
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(n)
        acc = np.zeros(7 if use_oracle else 6, dtype=np.float64)
        crit_acc = 0.0
        seen = 0
        te = time.time()
        for i in range(0, n, args.batch):
            sel = order[i : i + args.batch]
            bs = len(sel)
            xb = mx.array(X[sel])
            mb = mx.array(mask[sel])
            ab = mx.array(action[sel].astype(np.int32))
            oldb = mx.array(old_logp_np[sel])
            adv = mx.array(A[sel])
            gb = mx.array(Gv[sel])
            tokb, lenb = seq_batch(SEQ, LEN, sel)

            if use_oracle:
                # The critic trains every minibatch, on its own optimizer and
                # its own parameters: it cannot move the policy at all this
                # iteration, only the advantages of the next one.  Its z is a
                # detached snapshot of the encoder as the policy has it RIGHT
                # NOW -- read, never written.
                zc = detached_z(actor.enc, tokb, lenb)
                l_v, cgrads = cgrad_fn(critic, xo_batch(X, O, sel, zc), gb)
                copt.update(critic, cgrads)
                mx.eval(critic.parameters(), copt.state, l_v)
                crit_acc += bs * float(l_v)

                lab = mx.array(aux_lab[sel])
                valid = mx.array(aux_valid[sel])
                if frozen:
                    # Frozen means the POLICY PARAMETERS STOP MOVING -- which
                    # takes the aux heads and the encoder with it, since their
                    # gradient flows through the same trunk.  Forward only, for
                    # the stats; the critic above keeps training either way.
                    total, aux = policy_loss_fn(
                        actor, xb, tokb, lenb, mb, ab, oldb, adv, lab, valid
                    )
                    mx.eval(total, *aux)
                else:
                    (total, aux), grads = grad_fn(
                        actor, xb, tokb, lenb, mb, ab, oldb, adv, lab, valid
                    )
                    opt.update(actor, grads)
                    mx.eval(actor.parameters(), opt.state, total, *aux)
                kl_batch = float(aux[4])
            else:
                pi_coef = mx.array(0.0 if frozen else 1.0)
                (total, aux), grads = grad_fn(
                    actor, xb, tokb, lenb, mb, ab, oldb, adv, gb, pi_coef
                )
                opt.update(actor, grads)
                mx.eval(actor.parameters(), opt.state, total, *aux)
                kl_batch = float(aux[3])

            acc += bs * np.array([float(total)] + [float(a) for a in aux])
            seen += bs
            if kl_batch > 4.0 * args.target_kl:
                print(
                    f"epoch {epoch}: KL {kl_batch:+.5f} > 4x target "
                    f"{args.target_kl} -- aborting the update ("
                    + (
                        "the critic alone cannot pull the policy back"
                        if use_oracle
                        else "value-only steps are still dragging the policy "
                        "through the trunk"
                    )
                    + ")"
                )
                aborted = True
                break
            if not frozen and kl_batch > 1.5 * args.target_kl:
                frozen = True
                print(
                    f"epoch {epoch}: KL {kl_batch:+.5f} > 1.5x target "
                    f"{args.target_kl} -- "
                    + (
                        "policy and aux updates frozen, training the critic only"
                        if use_oracle
                        else "policy terms frozen, fitting value only"
                    )
                )
        stats = acc / max(seen, 1)
        if use_oracle:
            loss, l_pi, l_aux, a_acc, ent, kl, clipfrac = stats
            print(
                f"epoch {epoch:3d}/{args.epochs}  loss {loss:+.4f}  "
                f"L_pi {l_pi:+.5f}  L_aux {l_aux:.4f}  "
                f"aux acc {a_acc * 100:.2f}%  critic L_v {crit_acc / max(seen, 1):.4f}  "
                f"H {ent:.4f}  KL {kl:+.5f}  clip {clipfrac * 100:.2f}%  "
                f"({time.time() - te:.1f}s){'  [policy frozen]' if frozen else ''}"
            )
        else:
            loss, l_pi, l_v, ent, kl, clipfrac = stats
            print(
                f"epoch {epoch:3d}/{args.epochs}  loss {loss:+.4f}  "
                f"L_pi {l_pi:+.5f}  L_v {l_v:.4f}  H {ent:.4f}  "
                f"KL {kl:+.5f}  clip {clipfrac * 100:.2f}%  "
                f"({time.time() - te:.1f}s){'  [policy frozen]' if frozen else ''}"
            )
        if aborted:
            break

    export_weights(model, args.out)
    export_attn(actor.enc, args.out)
    if use_oracle:
        export_critic(critic, args.out)
        print(
            f"wrote weights to {args.out} (policy.f32 sliced back to "
            f"{ACTIONS + 1} outputs; {ATTN_BLOB_NAME} + aux.f32 + "
            f"critic.json/critic.f32 beside it)"
        )
    else:
        print(f"wrote weights to {args.out} (policy.f32 + {ATTN_BLOB_NAME})")


if __name__ == "__main__":
    main()
