# mjgame training (Python / MLX)

The learning side of the mjgame RL agent. The TypeScript engine plays games and writes trajectory
JSONL; this trainer reads that and writes a weight blob the engine loads back for inference. Nothing
here imports or runs the engine — the two sides meet only at the two file contracts described below.

Runs on Apple silicon via [MLX](https://ml-explore.github.io/mlx/). Verified on an M4 Pro with the
macOS system Python 3.9 and mlx 0.29.3.

## Setup

```sh
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
```

## Commands

Export an untrained, randomly initialized network so the engine can smoke-test its inference path
before any data exists:

```sh
python randinit.py --out weights/
```

Behavior cloning on recorded trajectories:

```sh
python bc.py --data 'runs/*.jsonl' --out weights/ \
    --epochs 10 --batch 4096 --lr 3e-4 --val 0.05
```

Quote the glob so the shell hands the pattern through rather than expanding it. `--save-every N`
also writes a checkpoint into `--out` every N epochs; `--seed` makes the split, shuffle, and init
reproducible. Each epoch prints train loss, validation loss, and validation top-1 accuracy (both
computed over the legal-action support only). Weights are exported when training ends.

PPO on self-play rollouts — see the PPO section below:

```sh
python ppo.py --data 'runs/ppo/iter1.s*.jsonl' --init weights/ --out weights/ --epochs 3
```

Loading data in your own script:

```python
from common import load_trajectories, PolicyValueNet, export_weights
traj = load_trajectories('runs/*.jsonl')
```

## File contracts

**Trajectory JSONL (input).** One JSON object per line, three kinds. A _decision_ is
`{"k":"d","v":2,"seat":0-3,"kyoku":n,"honba":n,"junme":n,
"planes":…,"scalars":…,"mask":[legal action indices],"a":chosen}`,
where `v` is the feature version, `planes` is base64 of `int8[1224]` (36 planes × 34 tile types) and
`scalars` is base64 of little-endian `float32[39]`. Lines without a `v` are v1 (748/132 bytes) and
are rejected: re-record them. A _round end_ is `{"k":"r","deltas":[4],
"outcome":"agari"|"draw","viol":[4]}`, where the optional `viol` is the 評価点 penalty points
each absolute seat incurred **during that round** (positive magnitudes). A
_match end_ is `{"k":"m","scores":[4],"net":[4],
"violations":[4]}`, where `net` is the per-seat
final settlement with uma applied and `violations` the per-seat penalty totals. All `d`/`r` lines of
one match precede its `m` line, and one file may hold many matches; `load_trajectories` relies on
that contiguity to attribute each decision to a match. It returns flat arrays: `X[n,1263]` float32
(planes cast to float, concatenated with scalars), `mask[n,78]` bool, `action[n]` int32,
per-decision `seat`/`match` index arrays, and per-match `net[m,4]`/`violations[m,4]` — the last two
are unused by behavior cloning and exist for the RL work to come. Round lines land in
`round_deltas[r,4]`/`round_outcome[r]`/`round_match[r]` plus `round_viol[r,4]` (zero-filled for
lines with no `viol`), and the flag `has_round_viol` is True only when *every* round line in the
dataset carried the field — a mixed dataset counts as not having it, since it cannot be timestamped
consistently.

**Weights (output).** Two files in the output directory. `manifest.json` describes the fixed
architecture: input 1263 → 512 relu → 256 relu → 79 linear, where the 79 outputs are 78 action
logits plus one value. `policy.f32` is the raw parameters with no header: for each layer in order,
the weight matrix row-major `[out][in]` followed by the bias `[out]`, all little-endian float32,
concatenated. That is 1263·512+512 + 512·256+256 + 256·79+79 floats = 3,195,196 bytes.
`mlx.nn.Linear` already stores `weight` as `[out, in]`, so the export is a direct dump with no
transpose.

Behavior cloning trains the policy head only. The value head is sliced off inside
`masked_cross_entropy`, so it receives no gradient and stays at its random initialization — the
engine should ignore output 79 until an RL stage actually trains it.

## PPO

`ppo.py` is the RL stage: clipped-surrogate PPO over self-play trajectories, training both heads.

```sh
python ppo.py --data 'runs/ppo/iter1.s*.jsonl' --init weights/ --out weights/ \
    --epochs 3 --batch 4096 --lr 1e-4 --clip 0.2 --vf 0.5 --ent 0.01 --viol-lambda 1.0 \
    --gamma 1.0 --gae-lambda 0.95
```

`--data` takes one or more globs. `--init` is the directory the rollouts were **collected** with —
it is the behaviour policy, not just a warm start — and `--out` is where the updated weights go
(the same directory is fine; the loop overwrites in place). `--epochs` is passes over the collected
batch, not over fresh data.

**The on-policy assumption.** `ppo.py` does not read behaviour log-probs off the trajectory file;
it recomputes them by forward-passing the frozen `--init` net. So the rollouts must have been
collected by *sampling* from `softmax(logits, T = 1)` of exactly those weights — the engine's
`--seats=nnnn --weights=<init>/manifest.json --temp=1`. Greedy play (`--temp=0`), a different
temperature, or a different weight directory silently poisons every importance ratio. The
TypeScript and MLX forward passes agree to about 3e-5, which is far below anything the ratio cares
about.

**Reward.** One hanchan is one episode. The reward is a stream of round payouts plus one terminal
settlement term, in uma-net points (episode magnitude ~±60):

```
R[k][seat] = round_deltas[k][seat]/1000 - viol_lambda * round_viol[k][seat]   # round k pays out
U[m][seat] = net[m][seat] - sum_k round_deltas[k][seat]/1000                  # uma / oka / trunc
```

The terminal term is not modelled but *measured*: the engine's `settlement` derives `net` from the
final scores, so subtracting the raw round sum leaves exactly whatever uma, return-point offset and
sub-1000 truncation the ruleset applied. `--viol-lambda` is the exchange rate between a dojo penalty
point and an uma point; single violations are worth 1/3/5/10 points, so the default 1.0 makes a
medium 禁じ手 cost about as much as a 3-point placement swing.

**Why the penalty is round-timestamped.** Dojo-violation avoidance has to *emerge* from the
long-horizon reward — it is a byproduct of playing well, never a per-decision shaping term. A
per-decision penalty would teach "do not emit this action", which is the rule-following-by-
construction the engine deliberately refuses to hard-code (禁じ手 are ledgered, not blocked). So the
charge enters the reward at the coarsest granularity that still carries causal timing: the round in
which it was incurred. Charging it only at match end smears a 1st-round 現物切り across all eight
rounds' decisions; round granularity keeps the credit inside the hand that earned it.

If the `r` lines carry no `viol` (data from a pre-`viol` recorder), `ppo.py` prints a warning and
folds `violations[m][seat]` — the match total — into `U` instead, which reproduces the old
undiscounted return *exactly* (verified to 5e-6 on `runs/ppo/smoke.jsonl`). The two are never both
applied; that would double-charge every penalty.

**Round index reconstruction.** `traj.rnd` cannot be used directly: the recorder buffers a match's round
results and emits every `r` line at match end, after all of that match's `d` lines, so the loader's
running counter is pinned to the match's first round. `ppo.py` reconstructs the true index by
counting `(kyoku, honba)` blocks off that first round — the pair is constant through a round and
changes at every boundary — and cross-checks the block count against the match's `r` line count,
raising if they disagree.

**Reward attribution.** Each (match, seat) pair is one decision sequence; file order is play order,
so a stable sort by (match, seat) already leaves each sequence temporal. Round `k`'s reward is
attached to the last decision that seat made **at or before the end of round k** — so a penalty is
never credited to a decision that had not happened yet, and a round in which the seat happened not
to act pays out at its previous decision rather than vanishing. The terminal `U` rides on the
episode's last decision. Every episode's attached total is asserted equal to its own return
(`sum_k R + U`) to 1e-4; that assertion is what catches an attribution bug before it becomes a bad
gradient.

**Advantages.** GAE(`--gamma`, `--gae-lambda`) over those sequences, computed in value-head units
(reward / `RETURN_SCALE`, the units `V_old` already predicts):

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)      V(terminal) = 0
A_t     = sum_l (gamma*lambda)^l delta_{t+l}     (reversed scan, per episode)
Gv_t    = A_t + V_old_t                          TD(lambda) value target
```

`Gv` is what the value loss regresses on. At `--gamma 1 --gae-lambda 1` the deltas telescope and
`A_t = (MC return-to-go) - V(s_t)` — the pre-GAE behaviour of this trainer, and the correctness
anchor for the scan (checked to 2e-16 on synthetic data). The default `--gae-lambda 0.95` trades a
little bias for much less variance from the seven-or-so rounds of noise downstream of any decision.

`A` is then normalized to zero mean and unit variance over the whole collected batch. The value head
arrives from behavior cloning *untrained* (random init), so on iteration 1 `V_old` is noise and the
returns alone drive the sign; normalization is what keeps that first update scaled like a
policy-gradient step instead of like raw ±60 returns. Expect `L_v` to be large and to fall over the
first few iterations as the critic catches up. The `gae` line printed before training reports the
config, which penalty timing is in force (`per-round` / `match-terminal`), and the pre-normalization
advantage mean/std.

Each epoch prints total loss, the policy surrogate `L_pi`, value loss `L_v`, masked policy entropy
`H`, approximate KL from the behaviour policy, and the fraction of samples outside the clip band.
KL creeping past roughly 0.02 or a clip fraction past ~30% on freshly collected on-policy data means
`--epochs` or `--lr` is too high for the batch.

### The loop

`ppo_loop.sh` alternates collection and updates from the mjgame directory:

```sh
sh train/ppo_loop.sh <iters> [games-per-iter=400] [shards=4] [start-seed=50000]
```

Per iteration it fans `games-per-iter / shards` hanchan out over that many parallel `deno run …
selfplay` processes (all four seats neural, `--temp=1`, current `weights/`), waits, then runs one
`ppo.py --init weights --out weights --epochs 3`. Shard seeds are `start + iter*10000 + shard*1000`,
so no two shards or iterations ever replay the same deal; a shard must therefore stay under 1000
games, which the script enforces. Rollouts land in `runs/ppo/iter<i>.s<j>.jsonl` (truncated at the
start of each iteration) with the deno stdout beside them in `.log`.
