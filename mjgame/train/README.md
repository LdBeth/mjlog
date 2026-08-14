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
"outcome":"agari"|"draw"}`. A
_match end_ is `{"k":"m","scores":[4],"net":[4],
"violations":[4]}`, where `net` is the per-seat
final settlement with uma applied and `violations` the per-seat penalty totals. All `d`/`r` lines of
one match precede its `m` line, and one file may hold many matches; `load_trajectories` relies on
that contiguity to attribute each decision to a match. It returns flat arrays: `X[n,1263]` float32
(planes cast to float, concatenated with scalars), `mask[n,78]` bool, `action[n]` int32,
per-decision `seat`/`match` index arrays, and per-match `net[m,4]`/`violations[m,4]` — the last two
are unused by behavior cloning and exist for the RL work to come.

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
