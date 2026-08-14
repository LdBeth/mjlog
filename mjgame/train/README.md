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

Migrating a feature-v2 weight set to v3 — see [Feature layout](#feature-layout-v3):

```sh
python train/widen.py --in weights-v2/ --out weights/
```

Loading data in your own script:

```python
from common import load_trajectories, PolicyValueNet, export_weights
traj = load_trajectories('runs/*.jsonl')
```

## Feature layout (v3)

The encoder is **48 planes × 34 tile types + 42 scalars = 1674** policy inputs; the oracle critic
takes those same 1674 followed by the 170 privileged values, `1674 + 170 = 1844`.

**Plane semantics are not documented here** — `src/rl/features.ts` is the authoritative map, and it
is the only place the numbering is maintained. What this side of the contract fixes is the *shape*
and the *order*, because those are what the blob's first layer is indexed by.

v3 is a strict superset of v2: every v2 feature keeps its meaning and its index **within its
section**, and the new features are appended to the end of each section. Since the input vector is
planes-then-scalars, the v2 vector is a prefix of each v3 section but *not* of the whole v3 vector —
the new planes sit between the old planes and the old scalars:

| v3 columns | contents | source |
| --- | --- | --- |
| `0 : 1224` | planes 0–35 | v2 `0 : 1224` |
| `1224 : 1632` | planes 36–47 (new) | — |
| `1632 : 1671` | scalars 0–38 | v2 `1224 : 1263` |
| `1671 : 1674` | scalars 39–41 (new) | — |
| `1674 : 1844` | oracle planes 0–4 (critic only) | v2 `1263 : 1433` |

### Migrating v2 weights

A v2 net *is* a v3 net whose new input columns are zero, so `widen.py` migrates one by surgery on
the first layer's weight matrix and nothing else:

```
new[:,    0:1224] = old[:,    0:1224]        # v2 planes
new[:, 1224:1632] = 0                        # planes 36..47
new[:, 1632:1671] = old[:, 1224:1263]        # v2 scalars
new[:, 1671:1674] = 0                        # scalars 39..41
new[:, 1674:1844] = old[:, 1263:1433]        # oracle block, critic only
```

`fc1`'s bias, `fc2`, `fc3` and `aux.f32` never saw the input width and are copied through
byte-identically (`aux.f32` is a file copy; the policy/critic blob tails after `fc1.weight` compare
equal with `cmp`). One invocation migrates the policy, the aux rows and the critic together:

```sh
python train/widen.py --in weights-v2/ --out weights/
```

It verifies itself before reporting success, three ways, and exits 1 if any of them fails:

- **structure** — the mapped columns are bit-identical, the added columns are exactly `0.0`, and no
  other parameter changed;
- **exact** — a float64 reference forward of both nets over 256 random inputs differs by exactly 0
  (summing extra terms that are all `0 × x` cannot move a running total);
- **runtime** — the mlx float32 forward agrees to `--tol` (default `1e-5`) *relative to the output
  magnitude*. Relative, not absolute, because float32 matmul is not associative and mlx blocks a
  1674-wide reduction differently from a 1263-wide one: on a real weight set that is ~1e-4 absolute
  on logits of magnitude ~400, i.e. two ulps, while the exact check is 0.

The new input positions are filled with **random** values in all three, never zeros — that is what
makes a column the surgery failed to zero show up rather than hide.

Guards, both exiting 2: the tool refuses a `--in` directory that is already v3 (re-running a
migration script would re-zero columns the net has since learned) and refuses anything that is not
recognisably v2 (only v2 → v3 is implemented). `--in` and `--out` may not be the same directory.

`common.py`'s loaders deliberately keep reading historical weight sets — `load_weights` /
`load_critic` check the manifest's `features` block only for internal consistency
(`planes·34 + scalars [+ oracle_planes·34] == layer 0's input width`), never against the current
constants, which is what lets `widen.py` open its own input. Rejecting a net that is too narrow *for
the data at hand* is the trainer's job instead: `ppo.py` compares both `--init` and `--critic-init`
against the loaded trajectories' width and names `widen.py` in the error. (`bc.py` has no `--init` —
it always builds a fresh net at the data's width — so it can only ever reject stale *data*.)

## File contracts

**Trajectory JSONL (input).** One JSON object per line, three kinds. A _decision_ is
`{"k":"d","v":3,"seat":0-3,"kyoku":n,"honba":n,"junme":n,
"planes":…,"scalars":…,"mask":[legal action indices],"a":chosen}`,
where `v` is the feature version, `planes` is base64 of `int8[1632]` (48 planes × 34 tile types) and
`scalars` is base64 of little-endian `float32[42]`. Older lines are rejected with their version
named: v2 is 1224/156 bytes, and a line with no `v` at all is v1 (748/132). **Data is never
migrated** — the planes v3 added were simply never recorded — so a stale dataset must be
re-recorded; stale *weights* are a different matter, see [Feature layout](#feature-layout-v3)
below. A _round end_ is `{"k":"r","deltas":[4],
"outcome":"agari"|"draw","viol":[4]}`, where the optional `viol` is the 評価点 penalty points
each absolute seat incurred **during that round** (positive magnitudes). A
_match end_ is `{"k":"m","scores":[4],"net":[4],
"violations":[4]}`, where `net` is the per-seat
final settlement with uma applied and `violations` the per-seat penalty totals. All `d`/`r` lines of
one match precede its `m` line, and one file may hold many matches; `load_trajectories` relies on
that contiguity to attribute each decision to a match. It returns flat arrays: `X[n,1674]` float32
(planes cast to float, concatenated with scalars), `mask[n,78]` bool, `action[n]` int32,
per-decision `seat`/`match` index arrays, and per-match `net[m,4]`/`violations[m,4]` — the last two
are unused by behavior cloning and exist for the RL work to come. Round lines land in
`round_deltas[r,4]`/`round_outcome[r]`/`round_match[r]` plus `round_viol[r,4]` (zero-filled for
lines with no `viol`), and the flag `has_round_viol` is True only when *every* round line in the
dataset carried the field — a mixed dataset counts as not having it, since it cannot be timestamped
consistently.

A `d` line MAY also carry the **oracle side channel**: `"o"`, base64 of `int8[170]` — five 34-wide
planes the acting seat cannot see (the three opponents' concealed hands in *relative* seat order
shimocha/toimen/kamicha, the hidden remainder, the ura-dora indicators) — and `"sh"`, the same three
opponents' current shanten in the same order, `-1` for an already-complete hand. They land in
`oracle[n,170]` float32 (zero-filled when absent) and `opp_shanten[n,3]` int32 (`-9` when absent),
with `has_oracle` True only when *every* decision line carried **both** — same all-or-nothing rule
as `has_round_viol`, and for the same reason: half a batch of privileged features would train the
critic on two different input distributions. These fields are appended to `Trajectories`, never
inserted, so positional unpacking and every existing consumer (`bc.py`) are unaffected.

**Weights (output).** Two files in the output directory. `manifest.json` describes the fixed
architecture: input 1674 → 512 relu → 256 relu → 79 linear, where the 79 outputs are 78 action
logits plus one value. `policy.f32` is the raw parameters with no header: for each layer in order,
the weight matrix row-major `[out][in]` followed by the bias `[out]`, all little-endian float32,
concatenated. That is 1674·512+512 + 512·256+256 + 256·79+79 floats = 4,036,924 bytes.
`mlx.nn.Linear` already stores `weight` as `[out, in]`, so the export is a direct dump with no
transpose.

**Trainer-private companions.** PPO with the oracle side channel writes three more files into the
same directory, none of which the engine reads or needs: `aux.f32` (the 24 auxiliary output rows
sliced off fc3 — `24·256` weights then the `24` bias, same little-endian layout) and the pair
`critic.json` + `critic.f32` (the oracle critic, `1844 → 512 relu → 256 relu → 1`, `arch`
`"mlp-critic"`, same headerless `[out][in]`+bias blob layout as the policy). `manifest.json` and
`policy.f32` keep the frozen 79-output shape no matter what: `export_weights` slices fc3 back to its
first 79 rows, and exporting a net that never grew any is byte-identical to what it always wrote
(verified with `cmp`).

Behavior cloning trains the policy head only. The value head is sliced off inside
`masked_cross_entropy`, so it receives no gradient and stays at its random initialization — and
under the oracle critic (below) no stage ever trains it either, since the baseline lives in a
separate net. The engine must always ignore output 79.

## PPO

`ppo.py` is the RL stage: clipped-surrogate PPO over self-play trajectories.

```sh
python ppo.py --data 'runs/ppo/iter1.s*.jsonl' --init weights/ --out weights/ \
    --epochs 3 --batch 4096 --lr 1e-4 --clip 0.2 --ent 0.01 --aux-coef 0.5 \
    --viol-lambda 1.0 --gamma 1.0 --gae-lambda 0.95
```

`--data` takes one or more globs. `--init` is the directory the rollouts were **collected** with —
it is the behaviour policy, not just a warm start — and `--out` is where the updated weights go
(the same directory is fine; the loop overwrites in place). `--epochs` is passes over the collected
batch, not over fresh data. `--critic-init` is where the oracle critic is loaded from and defaults
to `--init`, so the in-place loop round-trips policy, aux rows and critic together with no extra
flags.

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

`A` is then normalized to zero mean and unit variance over the whole collected batch. The baseline
arrives *untrained* on iteration 1 — random-init value head out of behavior cloning, or a fresh
critic — so `V_old` is noise and the returns alone drive the sign; normalization is what keeps that
first update scaled like a policy-gradient step instead of like raw ±60 returns. Expect the value
loss to be large and to fall over the first few iterations as the baseline catches up. The `gae`
line printed before training reports the config, which penalty timing is in force (`per-round` /
`match-terminal`), which baseline is active (`oracle-critic` / `shared-head`), and the
pre-normalization advantage mean/std.

Each epoch prints total loss, the policy surrogate `L_pi`, the value/critic loss, masked policy
entropy `H`, approximate KL from the behaviour policy, and the fraction of samples outside the clip
band (plus `L_aux` and `aux acc` on the oracle path). KL creeping past roughly 0.02 or a clip
fraction past ~30% on freshly collected on-policy data means `--epochs` or `--lr` is too high for
the batch.

### Oracle critic and auxiliary speed heads

When every `d` line carries the privileged `"o"`/`"sh"` pair, the baseline moves **out of the policy
net** and into a separate critic that sees the policy's 1674 features **plus** the 170 oracle ones:
`1844 → 512 relu → 256 relu → 1`, its own Adam at the same `--lr`, plain MSE against the same
`Gv` targets, exported as `critic.json` + `critic.f32`. This is asymmetric actor-critic — only the
baseline peeks, and it is used only to *subtract*, so the policy stays exactly as blind at training
time as it is at inference time while the variance from what the wall happened to hold is explained
away instead of being charged to the policy gradient. `V_old` for GAE comes from the **frozen**
`--critic-init` net, exactly as `V_old` used to come from the frozen `--init` value head; a fresh
critic makes it noise on the first iteration, which is what the advantage normalization above
handles.

Two consequences, and both are the point:

- The policy loss loses its value term entirely: `total = L_pi − ent·H + aux_coef·L_aux`. `--vf` is
  unused on this path. The critic's gradient never touches a policy parameter, so a large critic
  loss cannot drag the shared trunk — only the *next* iteration's advantages.
- **Output 78 (the in-tensor value head) is vestigial.** It receives no gradient from any loss and
  drifts only as the trunk moves under it. Nothing reads it: the engine ignores output 79, and this
  trainer takes its baseline from the critic. It stays in the blob solely to keep the frozen
  79-output weights contract byte-compatible.

**Auxiliary speed heads.** During training `fc3` grows 24 rows (79 → 103), read as 3 opponents × 8
shanten classes, and predicts each opponent's *current* shanten **from public features only** — the
oracle supplies the label, never the input, so this is representation shaping, not leakage. The
target is `clip(shanten, 0, 7)`; note that clipping at 0 merges the `-1` "already complete" case
into the tenpai class, which is deliberate (an actor cannot act on the difference) and keeps the
classes a plain 0..7 ladder. The loss is masked cross-entropy averaged over *labelled* (decision,
opponent) pairs, weighted by `--aux-coef` (default 0.5), and each epoch reports its top-1 accuracy.
The motivation is 雀鬼流 doctrine: 速度読み beats 待ち読み — a trunk that can answer "how close is
kamicha" already carries most of what push/fold needs. `export_weights` slices those 24 rows off, so
the engine's blob is unchanged; they persist in `aux.f32` and are reloaded from `--init` on the next
iteration (the run prints whether they were loaded or freshly initialized, and likewise for the
critic).

**KL guards under the critic.** Unchanged in shape, but "frozen" now means the policy parameters
genuinely stop moving: past 1.5× `--target-kl` the policy *and* aux updates are skipped outright
(there is no value term left to fit through the trunk) while the critic keeps training on every
minibatch; past 4× the whole update aborts, critic included.

**No oracle fields?** `ppo.py` prints a loud warning and runs the original shared-value-head path
end to end — same losses, same prints, same weights out (verified bit-identical against the
pre-oracle trainer on `runs/ppo/smoke.jsonl`). Nothing about the old path was deleted.

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
