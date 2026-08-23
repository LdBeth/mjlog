# CLAUDE.md — mjgame

A 1-player 雀鬼流 (jankiryu) riichi mahjong game with a TUI, a rule-violation
ledger (禁じ手 are recorded and priced, never blocked), and an RL training
pipeline (TypeScript inference + MLX trainer). Runs on **Deno, not Node/npm**.

## Tasks

```sh
deno task play        # TUI game (needs a tty)
deno task selfplay    # headless matches; --seats=…, --jobs=N, --record=…, --export=…
deno task bench       # timing subset of selfplay
deno task check       # typecheck src/ + scripts/
deno task test        # full suite (~555 tests, ~50s; compiles native libs via clang)
deno task tune        # 感性 vector tuning via paired runs
# champion seat: --seats=khhh --ktune=weights/champion.json (計算 calibrated + hand block)
deno task build-kernel  # native/mjkernel  (shanten/ukeire/shape-mass)
deno task build-native  # native/librlnet  (policy net via Accelerate)
```

CLI commands: `play` / `selfplay` / `paired` / `bench`. Full flag reference
lives in `src/cli/usage.ts`. Notable: `--jobs=N` (selfplay only) plays games in
worker threads with **byte-identical** output to sequential; `--export=PATH`
(play/selfplay) writes a Tenhou mjlog XML + `.mjgame.json` sidecar per game;
`--jobs` with `--calibrate` is refused (the calibration writer has no per-game
buffering seam). Flags a command would silently ignore are rejected by
`argError` in `src/cli/args.ts` — keep it that way when adding flags.

## Layout

- **Core engine** (`src/*.ts`): `round.ts` is the game master (synchronous
  generator); `table.ts` the authoritative state; `legal.ts` action
  enumeration; `observe.ts` builds the per-seat `Observation` — **the hot
  path**, everything derived (shanten, ukeire, waits, danger, discardInfo,
  lazy penalty preview) is computed once here and shared by all consumers.
  `yaku.ts`/`fu.ts`/`score.ts` scoring; `score.ts finalStandings` is the ONE
  settlement (violators rank below clean seats, 起家 tie-break) — the TUI and
  trainer consume it; never re-implement ranking. `export.ts` writes Tenhou
  XML (`sc` base = last accepted REACH snapshot; `INIT` carries the
  at-deal pot; `owari` on the last result) — exported games render through
  `../mjrender`.
- **CLI/harness**: `src/main.ts` is a ~500-line dispatcher. `src/cli/`
  (args/usage/die), `src/harness.ts` (policy construction — `makePolicy` takes
  an options object; `openArm`/`playGame`/`closeArm`: policies are built **once
  per run** and `reset(seed)` per game, `NeuralPolicy.close()` frees native
  nets), `src/paired.ts` (A/B driver), `src/selfplay_worker.ts` (worker shard;
  thread-safety notes in `harness.ts` — worker 0 warms mjkernel's lazy tables
  before others spawn).
- **`src/ai/`**: 計算/感性/規律 layers. `heuristic.ts` base CPU (compliance
  filter + protected hooks; per-decision facts cached on `Ctx`/`DecisionMemo`);
  `computed.ts` the 計算 engine (flat scratch-row hot path backed by the
  `mj_shape_masses` kernel); `planner.ts` C7 targets; `augmented.ts`
  Reads providers + the subclass; `consumer.ts` 感性 curves; `standings.ts`
  順位効用; `calibration.ts` offline-fit records (opponent deal-in model);
  `handvalue.ts` the M11 own-hand model (P(win) counting chain × value, ~15
  scalars fitted against actual hand outcomes) consumed by the fold gate and
  the discard score when a `--ktune` file carries a `hand` block — absent ⇒
  bit-identical to before; `handcalib.ts` its recorder (`--handcalib=PATH`,
  labels written at round end). Fit: `scripts/hand_fit.ts`, check:
  `scripts/hand_report.ts`. Consumption scalars `pushScale`/`evWeight` are
  set by paired sweeps, not by the fit (see `runs/hand/SWEEP.md`). FIT ONLY
  on a lane played WITHOUT the hand block: labels recorded under the folding
  policy are censored by its own folds, and a refit on them measured +0.11
  WORSE. `weights/champion.json` (tracked; the only files under weights/ in
  git are the three ktune JSONs) is the shipped baseline, pinned by
  `test/champion_test.ts` — a deliberate change regenerates the pins there.
- **`src/penalty/`**: `rules.ts` predicates, `preview.ts` the speculative
  referee (same `runHook`, guarded mutate-and-rollback). `heuristic.ts`
  keeps deliberate hand copies of a few predicates at a *different pricing
  scale* — do not "deduplicate" them onto the preview; that is a tuning
  decision, not a refactor.
- **`src/rl/`**: v4 feature encoder, TS reference forward + FFI fast path,
  78-slot action space, trajectory recording (`TrajectoryWriter` has a
  buffering mode for workers; `RecordingPolicy` reuses `NeuralPolicy`'s
  `lastEncoding`).
- **`src/tui/`**: term/input/screen/glyph/widgets/app. The `--timer` countdown
  is display-only pressure: nothing in the ledger prices the clock (see
  Decisions). The JIS number row (`1-9 0 - ^ ¥`, `\` = ¥) discards hand slot
  1–13 in one keystroke; `t` is the drawn tile; ←→/Enter remain.
- **`native/`**: `mjkernel.cc` (ABI 2: `mj_shanten`, `mj_ukeire_mask`,
  `mj_shape_masses`) and `rlnet.c`. Built with `-ffp-contract=off` — required
  for bit-exact parity with TS.

## Invariants (tests enforce these; do not weaken them)

- **Determinism**: same invocation ⇒ byte-identical output, including
  trajectories and exports; `--jobs=N` ⇒ byte-identical to `--jobs=1`.
- **Frozen contracts** with the MLX trainer: trajectory JSONL line schemas and
  encodings, weight-file formats, plane/scalar/seq orders, the 78-slot action
  space, lowest-slot tie-break, one RNG draw per decision. Buffer *reuse* is
  fine (serialization copies by value); format changes are not.
- **Kernel parity**: every native symbol must match its TS reference
  **bit-for-bit** (no tolerance) — the TS path stays as reference and no-FFI
  fallback. FFI is gated on `MJGAME_NATIVE=1` + `--allow-ffi`; a stale dylib
  degrades silently to TS unless the env var demands native.
- **Pinned fingerprints**: `computed_test`/`calibration_test` pin whole-hanchan
  decision streams. A legitimate behavior change must regenerate them
  deliberately, with the reason documented in the test.
- **Reward philosophy**: violations are minimized as a byproduct of long-term
  reward — no per-decision penalty shaping.

## Decisions

- **長考/腰 (the time-based Tier-B penalties) were REMOVED 2026-08-23**: a
  keyboard TUI cannot meet physical-table timing norms (3s / 1.2s), only the
  human seat ever had a clock, and one ledger entry demotes the seat below
  every clean seat for the whole hanchan under `finalStandings`. Do not re-add
  without a timer-bank-based design.

## Training

`sh train/ppo_loop.sh <iters> [games=400] [jobs=4] [start-seed=50000]` —
one deno process per iteration using `--jobs`, one dataset file
`runs/ppo/iterN.jsonl`, then `train/ppo.py --init weights --out weights`.
Rollouts must be sampled at `--temp=1` from exactly the weights being updated.
`train/V4_SPEC.md` documents the encoder/weights contract.
