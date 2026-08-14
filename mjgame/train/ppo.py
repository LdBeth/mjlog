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
separate 1433-wide critic that sees the policy's features PLUS the opponents'
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
    AUX_CLASSES,
    AUX_MISSING,
    AUX_OPP,
    CRITIC_BLOB_NAME,
    CRITIC_MANIFEST_NAME,
    FEATURE_VERSION,
    INPUT_DIM,
    ORACLE_LEN,
    CriticNet,
    PolicyValueNet,
    export_critic,
    export_weights,
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


def xo_batch(X, O, sel) -> mx.array:
    """[X ++ oracle] for one minibatch, assembled on the fly.

    Materialising the full 1433-wide matrix would double the trainer's peak
    memory over a big collection batch for no benefit -- the critic only ever
    sees it a minibatch at a time.
    """
    return mx.concatenate([mx.array(X[sel]), mx.array(O[sel])], axis=1)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def decision_round(traj):
    """Absolute round index per decision, [n] int64.

    `traj.rnd` is NOT it.  The recorder buffers a match's round results and
    writes every "r" line at match end, after all of that match's "d" lines, so
    the loader's running round counter is frozen at the match's FIRST round for
    every decision in it.  What actually separates rounds inside a match is
    (kyoku, honba), which is constant through a round, changes at every round
    boundary (honba increments on renchan, kyoku advances and honba resets
    otherwise) and so never repeats non-consecutively within a match.  Counting
    those blocks off the match's first round recovers the true index.

    The block count is checked against the match's "r" line count, which is the
    assertion that this reconstruction is right.
    """
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
            f"{int(n_rlines[bad[0]])} round line(s))"
        )
    return idx


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
    whose round indices are non-decreasing.  Round k's reward goes to the last
    decision that seat made at or before the END of round k, i.e. the last one
    with round <= k; a round in which the seat never acted therefore pays out at
    its previous decision rather than vanishing.  Rewards with no decision
    at-or-before them (a seat whose first decision is in a later round — not
    something mahjong produces, but cheap to be right about) attach to the first
    decision, and the terminal U rides on the last.  The per-episode total is
    then asserted against the episode's own return, which is what catches an
    attribution bug before it silently becomes a bad gradient.

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


def behaviour_pass(model: PolicyValueNet, X, mask, action, batch: int):
    """(old_logp, V_old) from the frozen collection policy, as numpy."""
    n = X.shape[0]
    lp = np.empty(n, dtype=np.float32)
    vv = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        xb = mx.array(X[i : i + batch])
        mb = mx.array(mask[i : i + batch])
        ab = mx.array(action[i : i + batch].astype(np.int32))
        logits, v = split_head(model(xb))
        picked = gather(masked_logp(logits, mb), ab)
        mx.eval(picked, v)
        lp[i : i + batch] = np.array(picked)
        vv[i : i + batch] = np.array(v)
    return lp, vv


def critic_pass(critic: CriticNet, X, O, batch: int) -> np.ndarray:
    """V_old from the FROZEN --critic-init net over [X ++ oracle], as numpy.

    A fresh critic makes this noise, exactly as the value head was noise on
    iteration 1 of the shared-head path; advantage normalisation below is what
    keeps that first update scaled like a policy-gradient step regardless.
    """
    n = X.shape[0]
    vv = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        sel = slice(i, i + batch)
        v = critic(xo_batch(X, O, sel))
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

    input_dim = int(traj.X.shape[1])
    if input_dim != INPUT_DIM:
        raise SystemExit(
            f"data is {input_dim} wide but feature v{FEATURE_VERSION} is {INPUT_DIM} "
            "-- common.py's contract constants and the engine's encoder disagree"
        )

    model = load_weights(args.init)
    if model.input_dim != input_dim:
        raise SystemExit(
            f"--init net takes {model.input_dim} inputs but the data is {input_dim} "
            f"wide (feature v{FEATURE_VERSION}).\n"
            f"  {args.init} は旧特徴量の重みです / that weight set predates the "
            f"current feature layout.\n"
            f"  Migrate it, function-preserving, with:\n"
            f"      python train/widen.py --in {args.init} --out <v{FEATURE_VERSION}-dir>\n"
            f"  (the loaders keep reading historical weight sets on purpose, so "
            f"this check -- not load_weights -- is what stops the mismatch)"
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
            critic = CriticNet(input_dim + ORACLE_LEN)
            mx.eval(critic.parameters())
            print(
                f"critic: no {CRITIC_MANIFEST_NAME} in {critic_dir} -- fresh init "
                f"({input_dim + ORACLE_LEN} inputs); V_old is noise this iteration"
            )
        else:
            print(f"critic: loaded from {critic_dir} ({critic.input_dim} inputs)")
        if critic.input_dim != input_dim + ORACLE_LEN:
            raise SystemExit(
                f"critic takes {critic.input_dim} inputs but the data gives "
                f"{input_dim} + {ORACLE_LEN} = {input_dim + ORACLE_LEN}.\n"
                f"  {critic_dir} の critic は旧特徴量です / that critic predates the "
                f"current feature layout.\n"
                f"  `python train/widen.py --in {critic_dir} --out <v{FEATURE_VERSION}-dir>` "
                f"migrates policy, aux and critic together;\n"
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
    R, U, r_starts, r_ends = round_rewards(traj, args.viol_lambda)

    i_rnd = decision_round(traj)
    keep = usable(traj, i_rnd)
    dropped = int((~keep).sum())
    if dropped:
        print(f"warning: dropping {dropped} decision(s) whose round never closed")
    X, mask, action = traj.X[keep], traj.mask[keep], traj.action[keep]
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
    old_logp_np, V_old = behaviour_pass(model, X, mask, action, args.batch)
    if use_oracle:
        V_old = critic_pass(critic, X, O, args.batch)

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
    def loss_fn(m, xb, mb, ab, oldb, adv, gb, pi_coef):
        logits, v = split_head(m(xb))
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

    def policy_loss_fn(m, xb, mb, ab, oldb, adv, lab, valid):
        """Oracle path: surrogate + entropy + auxiliary shanten, NO value term.

        The baseline lives in the critic, so nothing here reads output 78 and
        no gradient reaches it -- the policy's own value head is vestigial from
        this point on and drifts only as the shared trunk moves under it.
        """
        out = m(xb)
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

    grad_fn = nn.value_and_grad(model, policy_loss_fn if use_oracle else loss_fn)
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

            if use_oracle:
                # The critic trains every minibatch, on its own optimizer and
                # its own parameters: it cannot move the policy at all this
                # iteration, only the advantages of the next one.
                l_v, cgrads = cgrad_fn(critic, xo_batch(X, O, sel), gb)
                copt.update(critic, cgrads)
                mx.eval(critic.parameters(), copt.state, l_v)
                crit_acc += bs * float(l_v)

                lab = mx.array(aux_lab[sel])
                valid = mx.array(aux_valid[sel])
                if frozen:
                    # Frozen means the POLICY PARAMETERS STOP MOVING -- which
                    # takes the aux heads with it, since their gradient flows
                    # through the same trunk.  Forward only, for the stats; the
                    # critic above keeps training either way.
                    total, aux = policy_loss_fn(
                        model, xb, mb, ab, oldb, adv, lab, valid
                    )
                    mx.eval(total, *aux)
                else:
                    (total, aux), grads = grad_fn(
                        model, xb, mb, ab, oldb, adv, lab, valid
                    )
                    opt.update(model, grads)
                    mx.eval(model.parameters(), opt.state, total, *aux)
                kl_batch = float(aux[4])
            else:
                pi_coef = mx.array(0.0 if frozen else 1.0)
                (total, aux), grads = grad_fn(model, xb, mb, ab, oldb, adv, gb, pi_coef)
                opt.update(model, grads)
                mx.eval(model.parameters(), opt.state, total, *aux)
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
    if use_oracle:
        export_critic(critic, args.out)
        print(
            f"wrote weights to {args.out} (policy.f32 sliced back to "
            f"{ACTIONS + 1} outputs; aux.f32 + critic.json/critic.f32 beside it)"
        )
    else:
        print(f"wrote weights to {args.out}")


if __name__ == "__main__":
    main()
