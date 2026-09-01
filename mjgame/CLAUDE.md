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
deno task test        # full suite (~640 tests, ~60s; compiles native libs via clang)
deno task tune        # 感性 vector tuning via paired runs
deno task freeze      # league snapshot: --label=MMDD [--ktune=FILE] [--plan] → weights/league/
deno task arena       # riichi.dev bot: --token-file=PATH [--ranked] [--games=N]
                      #   [--brain=champion|tsumogiri] [--ktune=PATH] [--log=PATH]
# champion vs the frozen field: --seats=khhh --ktune=weights/champion.json
deno task build-kernel  # native/mjkernel  (shanten/ukeire/shape-mass)
deno task build-native  # native/librlnet  (policy net via Accelerate)
deno task build-mlp     # native/libmjmlp  (小さな学習ヘッド, bit-exact)
```

CLI commands: `play` / `selfplay` / `paired` / `bench`. Full flag reference
lives in `src/cli/usage.ts`. Notable: `--jobs=N` (selfplay only) plays games in
worker threads with **byte-identical** output to sequential; `--export=PATH`
(play/selfplay) writes a Tenhou mjlog XML + `.mjgame.json` sidecar per game;
`--jobs` with `--calibrate` is refused (the calibration writer has no per-game
buffering seam), and so is `--handcalib` / `--foldcalib` / `--evcalib` (all
three buffer per ROUND). `--foldcalib=PATH` (+ `--fold-eps=X`) records the M13
push/fold lane off a "k" seat 0; without `--fold-eps` it changes nothing it
records. `--evcalib=PATH` records M15b's EV核 lane off a "k" seat 0.
`--calibrate` is also refused beside a ktune carrying M14's `dealin` block, and
`--evcalib` beside one carrying an `ev` block (in both cases the lane must be
recorded on the plain read, never under the component being fitted). Flags a
command would silently ignore are rejected by `argError` in `src/cli/args.ts` —
keep it that way when adding flags.

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
  `scripts/hand_report.ts`. `mlp.ts` the shared small-MLP runtime (one
  forward pass, TS / native / numpy, bit-exact); `fold.ts` the M13 push/fold
  head over it, `foldcalib.ts` its ε-flip bandit lane
  (`--foldcalib=PATH --fold-eps=X`, rewards written at round end). Fit:
  `train/fold_fit.py`, check: `scripts/fold_report.ts`.
  `dealin.ts` the M14 learned deal-in read — two small MLPs (P(ron on each of
  the 34 types) per opponent, plus a tenpai head) served as `Reads.dealinP` /
  `tenpaiP` in place of the closed-form 計算 estimate, with `dealinValue` still
  computed's (D5). Features are FROZEN and public-current-state only: the 河読み
  ban is a test (permute a river, every feature must be bit-identical), and the
  river reaches the tenpai head as a BAG. Absent ⇒ bit-identical play; `{}`
  throws (a learned head has no identity). Lane: calibration v3 (a superset of
  v2 — old lanes still re-score) → `scripts/dealin_export.ts` →
  `train/dealin_fit.py` → `scripts/dealin_report.ts`, which reproduces the
  trainer's logits in TypeScript bit for bit. FIT ONLY on a lane recorded
  WITHOUT the block (`--calibrate` refuses one), for `handvalue.ts`'s reason:
  labels recorded under the head are censored by its own reads.
  Consumption scalars `pushScale`/`evWeight` are
  set by paired sweeps, not by the fit (see `runs/hand/SWEEP.md`). FIT ONLY
  on a lane played WITHOUT the hand block: labels recorded under the folding
  policy are censored by its own folds, and a refit on them measured +0.11
  WORSE.
  `evcalib.ts` M15b's EV核 lane (`--evcalib=PATH`, one packed
  `mjev_eval_rest` wire per turn decision + the 局's labels; the WRITER owns
  the `EvCore`, the recording seat carries none, and the header stamps
  `evAbi` + `engineHash` so a rebuilt DP makes the reproduction check a
  NOTICE and never a refusal). Fit: `scripts/ev_fit.ts`.
  `ev.ts` / `evparams.ts` / `evlayout.ts` / `evpack.ts` the M15
  expected-value core's TypeScript half — the FFI front door (no
  fallback: an `ev` block REQUIRES `native/libmjev`), the `ev` ktune
  block, the ONE wire layout mirrored offset-for-offset by
  `native/mjev.cc`, and the pure-TS packer that turns an `Observation`
  plus three protected policy hooks (`dealinProbOf`, `dealinCostPts`,
  `hiddenInfoOf`) into that wire. Absent block ⇒ no FFI is touched and
  the seat is bit-identical.
  `weights/champion.json` is BY CONVENTION the current champion: the
  TUI 助言 advisor always reads it (`src/main.ts`; play's CPU seats carry no
  vector) — never point it elsewhere; improve the file. Promotion requires
  controlled paired evidence (opponents held fixed) and deliberately
  regenerates the `test/champion_test.ts` pins. Since 2026-08-30 it is the
  arena-proven vector: the M10 computed calibration, the 最終形 riichi head,
  the 色読み sense trio, `liveYakuhai` 200, `keepTriplet` 1, the M14 learned
  deal-in `dealin` block with `augment.floor` 0.25 — no M11 hand
  block (the post-epoch sweep removed it, see Decisions;
  `hand-calibrated.json` stays archived). Tracked under
  weights/: the three ktune JSONs plus `weights/league/` (frozen snapshots,
  see Decisions).
- **`src/net/`**: the riichi.dev (RiichiLab) MJAI bridge — the arena owns the
  game; the bot answers `request_action` messages whose `possible_actions` is
  the authoritative legal set (an off-list reply is a chombo, recorded on the
  bot's profile — the one unforgivable failure). `arena.ts` transport
  (WebSocketStream Bearer auth, `--log` wire tap, last-ditch in-protocol
  fallback); `mjai.ts` notation arithmetic (the arena uses mjgame's own
  136-id scheme, reds = ids 16/52/88 ⇒ `ARENA_CFG`); `shadow.ts` rebuilds a
  REAL `Table` per kyoku from the event stream (proxy ids for concealed
  opponent tiles, exact-id resync from each request's decoded observation)
  so the champion's `Observation` comes from the real `observe()` — referee
  on, zero duplicated derivation; `champion.ts` maps possible_actions ↔
  engine `Action`s, runs the champion `decide`, handles the reach two-step,
  and surrenders the kyoku to the tsumogiri fallback on any desync. The
  loopback parity test (`test/arena_shadow_test.ts`) asserts shadow
  observations equal the engine's field-by-field — the champion DEGRADES
  SILENTLY on bad observations, so "same action chosen" is never enough;
  `test/arena_replay_test.ts` replays a real captured wire log
  (`test/fixtures/arena-validate-0827.jsonl`). Bot tokens live outside the
  repo (`--token-file`); never log or commit them. Ranked play uses
  `weights/arena.json` — champion.json plus ONLY the arena-only heuristic
  overrides (`bufferTight`/`bufferLow` = 1:
  the 持ち点8000未満 buffer defends a HOME ledger rule the arena does not
  have; wire-log replay showed it was the dominant cause of folding live
  tenpai vs a single riichi). Do not fold arena-only overrides back into
  champion.json.
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
  `mj_shape_masses`), `mjev.cc` (ABI 1: M15's expected-value core —
  scorer, shanten, and the discard/fold DP; re-entrant, every cache
  inside the context, no globals and no warm-up, so each worker dlopens
  its own image and each seat owns its ctx), `rlnet.c`, and `mlp.c` (ABI 1: the small learned heads
  behind `src/ai/mlp.ts` — plain C loops, NOT Accelerate, because a head's sign
  decides a fold and BLAS reorders sums; `test/mlp_native_test.ts` grades it at
  zero tolerance). Built with `-ffp-contract=off` — required for bit-exact
  parity with TS.

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
  *Exception (M15, owner decision 2026-08-30)*: `native/mjev.cc` has NO
  TypeScript twin; its scorer and shanten are bit-exact against
  `yaku.ts`/`fu.ts`/`score.ts`/`kernel.ts`, the DP is graded by analytic
  cases and paired play. A ktune carrying `ev` REQUIRES the dylib +
  `--allow-ffi` and refuses `MJGAME_NATIVE=0`; absent, the seat is
  bit-identical and no FFI is touched. Determinism holds the way it does
  everywhere else here: the DP is a pure function of its inputs, with a
  fixed exploration order, a node cap that truncates deterministically,
  per-context caches and no threads — so `--jobs=N` identity is unaffected.
- **Pinned fingerprints**: `computed_test`/`calibration_test` pin whole-hanchan
  decision streams. A legitimate behavior change must regenerate them
  deliberately, with the reason documented in the test.
- **Reward philosophy**: violations are minimized as a byproduct of long-term
  reward — no per-decision penalty shaping.

## Decisions

- **M15 — 計算の極致 (2026-08-30, owner-directed)**: the 計算 seat's discard
  choice was still a hand-written linear surrogate (`−1000·shanten +
  12·ukeire + 4·types + 60·dora + …`) glued to a defence term by two
  multipliers — nothing in it a probability or a point value. Two earlier
  results were symptoms of the FORM being wrong rather than of the weights:
  M10's post-calibration searches all returned the defaults ("the surface is
  flat"), and M11's correct value estimate had no place to act when poured in
  through one scalar. The replacement is an exact expected-value DP in C++
  (`native/mjev.cc`, `libmjev`) over the state `(hand counts, own draws taken,
  aka held, riichi)`: exact enumeration to 2向聴 with a closed-form tail
  beyond, a fold option at every state, and a root that reports `{total, dama,
  riichi, foldLine}` per candidate plus `bestPush`/`bestFold`.
  **D1 pool by net absorption** — own future draws are a uniform sequence
  without replacement from `publicUnseen`, keyed by the hand's counts.
  **D2 re-entrant** — the shanten group-word cache lives INSIDE the context
  (no globals, no lock, no warm-up), unlike `mjkernel.cc`.
  **D3 `ev` refuses what it supersedes** — sub-switches
  `ev.discard`/`riichi`/`calls`; `discard` refuses
  `consumer`/`hand`/`fold`/`--foldcalib`, `riichi` refuses the M12 `riichi`
  block, so unit C is graded as `champion − riichi + ev` and the substitution
  is explicit.
  **D4 units are points** — `pointsPerScore` (4 = 1/`augment.lambda`) converts
  `dojoCost`/`senseLineTax`/the planner bonuses into points, so a 4000-score
  veto is a 16,000-point one.
  **D5 deterministic budget** — `maxNodes` 200k value states and a fixed
  exploration order; beyond the budget every state takes the closed-form tail,
  so the answer stays a pure function of the inputs.
  **D6 the "h" seat is untouched** — `frozen.ts` gains nothing.
  **D7 hidden information is first-class and optional** — `drawDist[K][34]`,
  `poolOverride[34]`, `uraDist[34]`, `nextDoraDist[34]`, each behind its own
  flag so absence stays distinguishable from a uniform vector. The oracle's
  one-hots and a future LEARNED hidden-information module reach the engine
  through the same `hiddenInfoOf` hook and the engine never learns which
  producer filled them; `riichiNextDraw` is deliberately NOT consumed (it is
  per-opponent sequential information).
  THE CONSTITUTION HOLDS: 計算 stays exact counting over public facts,
  opponent hazards are POPULATION rates over M14's
  `tenpaiP`/`dealinP`/`dealinValue`/`expLoss` (never sequential inference),
  and the doctrine filters (`compliantDiscards`, `keepTriplet`, `dojoCost`,
  `senseLineTax`, `hasYakuProspect`, the kan vetoes) stay OUTSIDE the price as
  vetoes — `chooseDiscard` argmaxes the DP's price over the set the filters
  already narrowed, and the root deal-in cost is packed by TypeScript
  (`dealinCostPts` = `riskOf`'s arithmetic in points, 安全 free) so no dojo
  ruling gets a second home in C++.
  APPROXIMATIONS, all flagged in the code: the pool is off by ≤1 copy on
  draw-then-cut-same-type paths (D1); calls and rinshan draws do not shift
  turns; aka are not depleted from the pool; future (non-root) discards take a
  base deal-in rate rather than a per-tile estimate; the fold line ignores
  held 現物 beyond the root; ura is a counting expectation unless `uraDist` is
  supplied; 流局 carries no 連荘 value.
  STATUS: units A–D are BUILT. A = scorer + shanten in C++; B = discard +
  fold (the packer, `computeFold`, `chooseDiscard`, the three hooks, the
  wiring tests, a 決定/秒 + nodes line on `bench`); C = riichi — inside
  `wantRiichi`'s four unchanged gates the verdict is
  `ev.riichi[ty] > ev.dama[ty] + riichiMargin` for the CHOSEN tile and the
  M12 head is not consulted (a non-finite pair means "riichi is not on offer
  for that candidate" and reads as no); D = calls — `chooseCall` keeps
  `hasYakuProspect` and the referee's compliance test as VETOES but REPLACES
  the "must improve shanten" acceptance rule with
  `bestPush(post-call) − V_pass > callMargin` (a 役牌ポン that buys no step
  can still be worth taking), and `chooseKan` prices the 暗槓 the dojo
  already cleared as `eval_rest(post-kan, kanDoraOn) − bestPush >
  callMargin`; `mandatoryKan` is untouched and the C7 planner's `chooseCall`
  override keeps precedence while a target is locked. The extra call/kan
  roots are counted in `evStats`, so the bench line stays honest.
  FIRST GRADE (2026-08-30, unit B only: champion + `ev{riichi:false,
  calls:false}` vs champion, 600 paired games, seed 20901): 道場順位差
  **+0.475 [+0.362, +0.588]** — the EV seat is clearly WORSE (席0 平均点
  24,951 vs 31,038; 和了率 ~9% vs ~21%; 聴牌率 ~20% vs ~32%; 放銃率 ~10% vs
  ~12%; 平均和了打点 8,300-11,300 vs 6,300; 8000点未満 43 vs 25). Getting
  here took four engine rounds, each found by replaying decisions against the
  champion (scratchpad diag scripts): a tail that priced worse shanten higher
  (rung collapse), a root deal-in cost ADDED instead of subtracted, the fold
  option absorbed into `O_TOTAL` (every candidate collapsed to one number), a
  shape-arena starvation freezing shapes at 0, the growth term tripled (my
  plan text), and a single tail calibration that was 2-3× too generous at
  3向聴+ (now `TAIL_CAL[s]` per rung, measured against the exact DP: advance
  beats keep on 100%/93% of random 3/4向聴 roots). The engine is now
  INTERNALLY CONSISTENT (seam ratio ~1.0 at 1-4向聴, no truncation at
  `maxNodes` 60000, `handOutlook` parity for the tail chain, scorer bit-exact,
  ~600 決定/秒). What remains is the MODEL: the seat builds expensive slow
  hands — E[value|win] is realistic (a doraless standard hand ≈ 1,800, a
  七対子 shape ≈ 7,800) while the incumbent's flat 4,200-7,000 value never
  discriminates, so speed wins there and value wins here; and the population
  hazard scalars (`ronFactor` 0.5 gives P(win) 0.89 from a ryanmen tenpai in
  12 turns — ≈0.19 would give 0.65; `oppGrowth` retires a third of a quiet
  hand's mass) were never fitted to THIS model — a 18-cell screen over
  ronFactor × oppGrowth × dealinRate moved nothing (all cells 3.0-3.2 on a
  60-game smoke). Next lever: FIT the `ev` scalars (and the value/speed
  realism) against recorded ground truth — the handcalib lane records
  HandFacts + won/tenpai/dealt-in per decision — rather than screening; and
  audit whether the exact path's P(win) from tenpai matches the champion's
  realised rates. The champion is UNTOUCHED — `champion_test` asserts
  `k.ev === undefined` until the owner's word, no frozen/league pin moves at
  any step, and `weights/ev-default.json` is a bench/smoke vector, never a
  candidate.

  M15b FIT — THE LANE AND THE FITTER ARE BUILT, THE FIT IS ON HOLD
  (2026-08-30). The FIRST GRADE's next lever was "fit the `ev` scalars against
  recorded ground truth rather than screening", and the machinery for it now
  exists end to end. What does NOT exist is a fitted vector worth carrying: a
  full review of `native/mjev.cc` found defects (root-discard furiten
  unpriced, the tail/exact seam still deciding the fold, the fold verdict
  inert), so every number below is an artefact of an engine that is being
  repaired. NOTHING WAS GRADED — no paired run against the champion, no
  candidate in `weights/` (the fit's output sits in `runs/ev/ev-0830-MOOT.json`
  under that name on purpose), and the champion is untouched.
  THE LANE (`ai/evcalib.ts`, `--evcalib=PATH`, selfplay/paired, "k" seat 0, no
  `--jobs`) writes one line per TURN decision of seat 0: the FULL packed
  `mjev_eval_rest` wire of the resting 13-tile shape the seat chose — `ints`
  (208) and `dbls` (320, hidden block zero), which is by construction
  everything the DP is allowed to see (`evlayout.ts` IS that specification) —
  plus the 局's ground truth for that seat (`won`/`winPoints`/`dealtIn`/
  `dealtInPoints`/`oppWon`/`outcome`/`endJunme`/`tenpaiEnd`) and the bucket
  keys (`sh`/`junme`/`T`). `tenpaiEnd` is judged the way the 局 ended: the
  table's own 流局 judgement, 1 when we won, and `shanten(seat 0's hand) ≤ 0`
  when somebody else ended it.
  THE M11 LESSON, ENFORCED: `--evcalib` REFUSES any `ev` block (sub-switches
  included) — the lane must be the PLAIN champion's continuation of the hand,
  because a lane played by the DP is censored by the DP's own folds. So the
  recording seat holds no core at all: `evFactsForRest` builds the mode-1
  facts through the same hooks an `ev` seat uses (`dealinProbOf`/
  `dealinCostPts`/`threatOf`/`expLossOf`/`standingsOf`/`hiddenInfoOf`) and
  packs them into a scratch `EvWire`, and the WRITER owns the one `EvCore`
  that evaluates them. `--evcalib` with the lane on plays BIT-IDENTICALLY to
  the same seed without it (a test), and `paired` strips it from the control
  arm.
  THE ENGINE'S IDENTITY IS IN THE HEADER (v2): `evAbi` and `engineHash`, the
  sha256 of `native/mjev.cc` at record time. The two are treated differently
  ON PURPOSE. The ABI governs whether the wire can be read at all and is a
  REFUSAL; the engine hash governs only whether the stored `pT`/`pW`/`eV`/
  `eCost` still reproduce, and is a NOTICE — `ev_fit.ts` skips its
  reproduction check, prints why, and fits on. A record carries two
  independent things, and only one of them is the engine's: the WIRE and the
  LABELS are facts about the game and survive any repair to the DP, while the
  four stored predictions are one engine's answers and are SUPPOSED to change
  when it is corrected. Refusing the lane would throw away the half that is
  still true. v1 lanes (recorded before the hash) read the same way.
  THE FITTER (`scripts/ev_fit.ts`) optimises SIX population scalars —
  `ronFactor`, `oppHazard`, `oppGrowth`, `dealinRate`, `tsumoShare`,
  `foldHazard` — against `L = BCE(pW, won) + BCE(pT, tenpaiEnd)` on the train
  half, by replaying each stored wire through a real `EvCore` built from the
  candidate vector (the honesty rule: never a second implementation of the
  model). Bounded-logit parametrisation, so three scalars whose lower bound is
  exactly 0 stay reachable and no step can propose a negative hazard;
  deterministic Nelder-Mead, because the DP's fold and riichi choices are
  argmaxes and the surface has a step at every decision boundary. `meanUkeire`
  and the eight `value*` scalars are NOT fitted (the tail's shape was
  calibrated against the exact DP in C++, and 打点 is `handvalue.ts`'s lane);
  `eCost` is REPORTED, never fitted — it is an expectation over a continuation
  that did not happen and has no per-record Bernoulli label. Split by seed
  parity. Reservoir sampling, seeded, so a 448 MB lane answers from 4,000 rows.
  THE ONE MEASUREMENT THAT SURVIVES the engine review, because it is a
  statement about the MODEL and not about the DP's bugs: at `DEFAULT_EV` the
  core's P(win) for the shape the champion actually chose is over-predicted
  everywhere and catastrophically so far from tenpai. Holdout (6,000 rows,
  odd seeds, 2,000-半荘 lane): predicted 和了 25.1% against a realised 15.0%;
  predicted 聴牌-at-end 53.0% against 37.9%. The 向聴 × 残り自摸 audit says
  where: at 聴牌 the prediction is close and honest (T6-10: 53.6% vs 48.1%),
  at 1向聴 it is 20-30% relative high, and at 3向聴+ with 11+ draws left it
  predicts 40.7% and the realised rate is 10.7% — the shapeless-hand tail is
  the wrong end of the model, exactly where M15's `TAIL_CAL` work stopped.
  E[打点|和了] is the half that is ALREADY RIGHT (0.90-1.16× realised at
  聴牌/1向聴/2向聴; only 3向聴+ is off at 0.54×), which is the same finding the
  FIRST GRADE recorded from the other direction. A 279-evaluation fit moved
  holdout BCE 0.462 → 0.386 (和了) and 0.730 → 0.534 (聴牌) with
  `ronFactor` 0.50 → 0.61, `oppHazard` 0.12 → 0.35, `oppGrowth` 0.040 →
  0.075, `dealinRate` 0.050 → 0.020, `tsumoShare` 0.30 → 0.025,
  `foldHazard` 0.010 → 0.026 — READ AS DIRECTION ONLY, since the engine those
  numbers correct is the one under repair.
  THE LANE ON DISK is `runs/ev/lane-800000.jsonl`: 2,000 半荘, 264,035 rows,
  448 MB, seed 800000, `khhh` + `weights/champion.json`, v1 (recorded before
  the hash landed). Its wires and labels stay valid; its stored predictions do
  not, and the fitter now says so instead of dying.
  COST, and a flag for whoever owns the repair: `mjev_eval_rest` on this lane
  cost 0.93 ms per call at 17:50 and 246-353 ms at 18:25 after the rewrite —
  250-300× — which is why the lane was NOT re-recorded under the new header
  (2,000 半荘 would be ~21 h) and why a 280-evaluation fit is days rather than
  15 minutes. A fresh, hashed lane is cheap again the moment a rest evaluation
  is back in the ~1 ms range.
  **REVIEW 2026-08-30** (a full read of the integration, after the first grade;
  every finding below carries a test that fails without its fix):
  FATAL — a folding seat threw the PUSH tile: `computeFold` decides by
  `bestFold > bestPush` and `chooseDiscard` then ranked every candidate by
  `O_TOTAL`, which is the push line and holds no fold option; the two argmaxes
  are different tiles by construction. FIXED — `evPriceOf` ranks by
  `O_FOLDLINE` while `ctx.folding`. This alone can account for a large part of
  the +0.475: the seat folded and kept feeding the shape it had abandoned.
  FATAL — an ev seat CRASHED near 河底: 計算's `wallComposition` is
  `unseen × wallRemaining/unseenTotal`, so on the last discards of every hand
  that gets there the vector is all zeros (wall 0) or sums to 0.99999998 in
  float32 (wall 1), and `hiddenInfoOf` handed that over as a REPLACEMENT pool;
  `mjev.cc` refuses `Nroot < 1` and `evEvalDiscard` turns the refusal into a
  throw mid-match — `headless(3, 8191, "kkkk", {ev:{}})` reproduced it, i.e.
  the DEFAULT ev configuration, in most hands. FIXED — the channel is filled
  only when the composition holds at least one tile as the DP counts it (same
  sum, same order as `parseEval`); below four live tiles `T` is 0, so nothing
  in that pool is ever drawn and the uniform `unseen` prices the identical
  hand. ⚑ The native side still treats an empty pool as fatal rather than as
  "no override"; a future producer that sends one has no other guard.
  MATERIAL — the planner's `keepBonus`/`drawBonus` rode on the EV path, and
  `planKeep` 5000 × `pointsPerScore` 4 = a 20,000-point steering term over a
  price whose whole hand is usually worth less. FIXED — both hooks are off the
  EV path (the DP prices the shape; C4/C5's hidden information reaches it
  through `hiddenInfoOf`'s channels instead, and C6 has no channel by D7 and is
  dropped). `dojoCost`/`senseLineTax` stay, as vetoes-by-price.
  MATERIAL — `O_TOTAL` is `max(dama, riichi)`, but the declaration can still
  be refused by `riichiBanned` (地獄単騎/即引っかけ) or the referee, or simply
  not be on offer. FIXED — such a candidate is priced at its `O_DAMA`.
  MATERIAL — `kanWorthIt` compared `eval_rest` (a hold, paying no discard
  cost) against `bestPush` (whose every candidate paid `−costIn`), so on a
  loud table every kan looked cheaper than every push by the price of a
  deal-in. FIXED — the cheapest held type's `dealinCostPts` comes off the hold
  line first.
  MATERIAL — `hiddenInfoOf`'s channels were reused for HYPOTHETICAL roots
  (post-call, post-kan), where "the next own draw" is not what we would draw.
  FIXED — `evPolicyFacts` takes `hidden` as a parameter; hypotheticals pass
  null, the real root and the PASS line keep it.
  MATERIAL — `akaHeld` ignored aka in our OWN melds (`score.ts` counts hand +
  melds) and read `obs.hand` even when the root overrode it. FIXED —
  `EvFactOpts.tiles` names the root's own tiles and melds are counted;
  `akaUnseen` is read off the Observation alone. (⚑ `mjev.cc` then clamps
  `I_AKA_HELD` to the CONCEALED 5p count, so a melded aka is still dropped
  inside the engine — a modelling gap, not a wiring one.)
  MATERIAL — `dealinCostPts` zeroed the 安全 rung on its own while `riskOf`
  charged `w.danger["安全"]`, so any vector pricing that rung gave one tile
  two prices in one decision. FIXED — base and augmented no-read branches are
  `riskOf` term for term; the 安全 exit stays only where an ESTIMATE could
  outrank the proof.
  MINOR — the D4 equivalence (`dealinCostPts` = `riskOf` in points) holds only
  while `augment.lambda = 1/pointsPerScore`; nothing enforced it. FIXED —
  `makePolicy` refuses `pointsPerScore × lambda ≠ 1` before `buildEv`.
  INVESTIGATED, NO CHANGE — the root count rule. A 槓 is ONE set: its fourth
  tile is paid for by the rinshan draw, so a post-kan discard root holds
  `14 − 3·melds` concealed tiles exactly as a pon's does (measured over 40
  hanchan of the engine's own stream). The TypeScript test is now that
  equality rather than `hand.length % 3 === 2` — the same rule `mjev.cc`'s
  `parseEval` states — and `evRestRoot` guards the rest root the same way.
  Subtracting the kan count on either side would refuse every kan hand the
  engine produces; both files carry the reasoning at the check.
  SUITE BUDGET — with the crash fixed, `ev_wiring`'s identity arms stopped
  ending early and ran the DP at the default `maxNodes` 60,000: one test took
  over ten minutes. They now carry `maxNodes: 250` like the unit C/D
  aggregates already did (7s), which does not weaken them — the claim is that
  the block REACHES the seat, and a truncated DP is still the DP.
  The native half's own findings are listed at their fixes in `native/mjev.cc`.
  SECOND GRADE (2026-08-30, after the full review + the brute-force oracle;
  champion + `ev{fitted scalars of weights/ev-0830b.json, riichi:false,
  calls:false}` vs champion, 600 paired games, seed 20901): 道場順位差
  **+0.215 [+0.103, +0.327]** — still worse, but half the first grade's
  +0.475 (席0 平均点 27,734 vs 31,038; A優位 164 / B優位 235; 8000点未満
  38 vs 25). Runtime 76 min (0.3 半荘/秒: per-candidate independent search at
  maxNodes 60000 is ~22 決定/秒). What the oracle settled: the recursion is
  exact (0 ulp vs brute force on 240 tiny instances); what remains is the
  MODEL — the closed-form tail (every root at 4向聴+, and 3向聴 roots when the
  budget is short) is calibrated only for silent reads (exact/tail 1.9-10×
  with live tenpaiP), the hazard scalars were fitted on the champion's play,
  and the seat still wins 12% vs 21% at equal 聴牌率. Scalar fits: `deno run …
  scripts/ev_fit.ts --in=runs/ev/lane-800000.jsonl --base=default`.
  QUALITY-FIRST PASS (2026-08-31, owner: "sacrifice speed for the best
  result"): pruning widened (keep-ALL shanten-keeping discards at ≤1向聴,
  top-6+dora at 2向聴, 2 待ち替え candidates), safe cross-candidate sharing
  restored (shape GEOMETRY shared — pool-independent; mass/edges per
  candidate generation; 1.39× with byte-identical nodes), the tail's hazard
  half now rides `turnValue`'s own arithmetic (cal scales the win term only;
  live-reads exact/tail within 3-11% where the tail governs), defaults
  maxNodes 1.2M / exactShanten 3 (≈3.6-7 決定/秒, p95 0.9s). Smoke 3.00 →
  2.55; 放銃 9.9%, 聴牌 36.5%. Residual divergence is a MODEL question: at
  巡1-4 the exact DP prefers the wider worse-shanten shape 45/232 times
  (champion never) — acceptance compounds while the win-value half
  under-discriminates; the next lever is the win-value model, not the
  search. Pruning residual = the 待ち替え acceptance-mass gate (8): 6% mean
  at T=2 on 1向聴 rests; gate→0 costs 25× nodes for 0.7%.
  THIRD GRADE (2026-09-01, quality-first engine + fitted scalars, discard/
  fold only, 600 paired games): 道場順位差 **+0.103 [−0.007, +0.214]** — the
  CI touches zero for the first time (+0.475 → +0.215 → +0.103 across the
  three grades; 席0平均点 29,326 vs 31,038; 違反 39 vs 26, mostly 8000点未満
  37 vs 25). ~4.3h per arm at 0.1 半荘/秒. The full vector (EV riichi +
  calls, `champ-ev-full.json`) is graded separately.
  FULL-VECTOR GRADE (2026-09-01, same engine, champion − riichi +
  `ev{fitted}` with ALL THREE switches on, 600 paired games): 道場順位差
  **+0.012 [−0.104, +0.127]** — statistical PARITY with the champion
  (席0平均点 30,907 vs 31,038; A優位 202 / B優位 209; 違反 36 vs 26, the gap
  is 8000点未満 35 vs 25 — the home dojo buffer the arena does not have).
  The EV riichi + call decisions recovered what the discard-only vector
  left on the table (+0.103 → +0.012). ~6h per arm.
- **M14 — the learned deal-in read (`ai/dealin.ts`, 2026-08-29,
  owner-directed)**: the other half of the pair M13 opened. Where the fold head
  learns the DECISION, this learns the EVIDENCE: end-to-end
  P(opponent i rons type t | public state) from a 54-column feature row per
  (opponent, tile type), plus a 22-column tenpai head, served as `Reads.dealinP`
  / `tenpaiP` in place of the counting model's closed form. `dealinValue` stays
  computed's, rebuilt for ALL 34 types through `valueOnType` (D5) — the value
  model is not being learned, and computed leaves a zero wherever its own q ≤ 0
  that `riskOf`'s `?? expLoss` fallback would not rescue.
  SWITCH SEMANTICS with ONE difference from `hand`/`riichi`/`sense`/`fold`: no
  `dealin` section ⇒ no head is built, no trace is requested and the seat plays
  bit-for-bit its prior game, but `{}` THROWS ("dealin ブロックには重みが要り
  ます") — a learned head has no weight setting that reproduces the counting
  model, so absent is the switch and an empty block can only be a mistake.
  Routed to "k" seats ONLY (D11); the two heads are BUILT objects (they may hold
  a native context), so `makePolicy` builds them once per seat outside the
  `withReads` rebuild closure — building inside it would allocate a native
  context per hanchan and free none — and frees them in the seat's `close()`.
  FEATURES ARE FROZEN and public-current-state only: the 河読み ban is enforced
  by a river-permutation test (permute a river keeping bag and 現物 ⇒ every
  feature bit-identical), and the river reaches the tenpai head as a BAG.
  THE LANE is calibration v3, a strict SUPERSET of v2 (`CALIB_ACCEPTED {2,3}`,
  so the 598 MB v2 lanes and `calibrate_fit`/`calibrate_report` keep working):
  new public-state fields `un`/`oh`/`ak`/`sc`/`ri`/`rj`, per opponent `gb`
  (現物) and `rb` — the deliberate addition to the brief's field list: each
  opponent's own discards as a BAG of type counts, because the tenpai head's
  river columns cannot be derived from anything else the record carries and
  `gb` is contaminated for a declarer; it carries no ORDER, by the same ban.
  `fh` (a digest of the served feature rows) is written only when the recording
  seat is running the heads — which the CLI refuses (D6: `--calibrate` beside a
  `dealin` block, the third member of the `--jobs`/`--curriculum` family) — so
  the first lane carries no digest and `dealin_export` says so out loud.
  `train/dealin_fit.py` undoes the export's negative subsampling by weight
  (`1/keep`) and, when `--pos-weight` ≠ 1, applies the PRIOR CORRECTION:
  log(pos_weight) is subtracted from the output bias after training, because
  `riskOf` multiplies the probability by a payment in points and a head
  calibrated for a balanced table would price every tile hot. Holdout is seed
  parity; `scripts/dealin_report.ts` reproduces the trainer's logits in
  TypeScript bit for bit.
  THE `floor` KNOB is the promotion plan's second half: `augment.floor` (0.5
  today) keeps the rule ladder as a lower bound under the estimate, so 安全
  stays a proof no head may price. As the head is graded the floor steps
  0.5 → 0.25 → 0 — a separate, deliberate owner edit that must never ride in
  with the block. `champion_test` asserts the block is present and the floor
  is 0.25 (the promoted, arena-tested values); a different floor is a new
  promotion, not a merge.
  FIRST TRAINING ROUND (2026-08-29): v3 lane of 2,000 半荘 (371,731 decisions,
  3.99M (opponent, tile) rows, 0.59% positives; tenpai rows 1.03M). Holdout
  (odd seeds): learned P(ron) BCE 0.02402 vs computed 0.02461, Brier 0.005500
  vs 0.005530, better in EVERY stratum (巡目 × riichi/furo/quiet); tenpai head
  BCE 0.179 vs the prior's 0.197. TS reproduces the trainer's logits exactly
  (10,000/10,000). Paired grades, 600 games vs champion, floor sweep:
  0.5 → +0.008 [−0.073, +0.090]; 0.25 → −0.015 [−0.098, +0.068];
  0 → −0.015 [−0.102, +0.072]; violations flat. NEUTRAL at home — note the
  paired SD is ~1.0 vs the usual 0.4 (the learned reads move many more
  discards), so this field needs ~6× the games for the usual CI; the
  prediction gain is real but small (2.5% BCE) and the arena is where the
  ladder's blind spots (closed flushes, quiet tables) actually cost. Nothing
  ships in the champion; INSTALLED in `weights/arena.json` the same day
  (owner: "install the improved component and get ready to arena test") with
  `augment.floor` 0.25 — the arena is the test the home grade could not give.
  Replay bench on two ranked logs: 133/1,581 decisions change, 81 at 巡≤5,
  122 shanten-neutral (tie reorderings under the learned risk row).
  ARENA RESULT (50 ranked games, 535 局, 2026-08-29/30; logs
  `runs/arena/ranked-0829-dealin*.jsonl`) vs the pooled pre-M14 baseline
  (15 games / 122 局): 放銃率 10.8% vs 13.9%, 平均放銃打点 4,924 vs 7,871,
  満貫以上の放銃 21% of feeds vs 59%, 放銃失点/局 534 vs 1,097 — the feed
  cost the ladder could not see roughly HALVED. Bought with 和了率 22.8% vs
  28.7% and 平均点 25,282 vs 29,107; 平均順位 2.560 [2.26, 2.86] vs 2.667 —
  placement neutral at n=50 (blocks ran 2.44 → 2.59, noise). The head does
  what the fit predicted; the promotion question is which criterion decides:
  placement needs several hundred arena games, feed cost/局 is already
  decisive relative to its own variance.
  PROMOTED 2026-08-30 (owner: "go ahead and promote 0.25"): champion.json =
  arena.json minus the buffer overrides — the `dealin` block plus
  `augment.floor` 0.25, the exact vector that played the 50 games. 0.5 was
  never rejected (home grades could not separate the floors; 0.25 had the
  best point estimates and the fewest 8000点未満 violations, 25 vs 32) — it
  is simply untested in the arena. Champion pins re-captured; the frozen
  "h" seat and the league pins are UNTOUCHED (frozen.ts has no dealin
  support and `learnedReads` lives only in the k-branch), so seat 0 of
  `khhh` no longer equals the field. `dealin_wiring_test` states its
  "absent" claims against the champion with the block stripped.

- **M13 — the fold head (`ai/fold.ts`, 2026-08-29, owner-directed)**: the
  push/fold gate's whole verdict was one hand-written comparison,
  `push·gain < 0.5·pressure·risk`, and every arena mangan feed was on one side
  of it. `computeFold` is refactored so that comparison's number,
  `margin = push·gain − 0.5·pressure·risk`, is FEATURE 0 of a 37-wide vector
  (the gate's five parts, the resting shape's `HandFacts`, per-seat threat and
  `expLoss`, the M11 outlook, the scoreboard, the hand's own defensive capacity
  — safe/low/unassessed types and 現物 counts vs each riichi — and the 色読み
  field pressure). SWITCH SEMANTICS, as `hand`/`riichi`/`sense`: no `fold`
  block ⇒ `computeFold` takes an early return running the incumbent expression
  character for character (no feature built, no rng touched, no pin moved);
  `fold: {}` ⇒ `INIT_FOLD`, one linear layer with `w[margin] = −1`, so
  `forward(x)[0] > 0` IS `margin < 0` — the identity is STRUCTURAL, not
  measured. Routed to "k" seats ONLY (the frozen "h" letter is the baseline
  this head must beat); built once per seat in `makePolicy` and freed in the
  seat's `close()`, because it may hold a native context.
  THE LANE is a contextual bandit, not a supervised set: no log says what
  folding would have paid, so `--foldcalib=PATH --fold-eps=X` flips the verdict
  with probability ε on a stream of the seat's OWN (derived
  `imul(seed, 0x9E3779B1) + 13`, made only when ε>0, at most one draw per
  decision — `shouldFold` is memoised) and writes the propensity down beside
  the action. At ε 0 the recorder is invisible: same games, same bytes, no
  draw. REWARD = the round's `deltas[0]/1000` and nothing else (D7); `vio0`,
  `won`, `dealtIn` are recorded as DATA — a violation term here would be
  exactly the per-decision shaping the reward philosophy forbids. The
  per-round settlement is shared by every decision inside it, so the
  independence assumption is stated in both the recorder and the fit, and the
  multi-flip-round fraction is REPORTED rather than assumed away.
  `train/fold_fit.py` fits q̂(x,a) (Huber δ=8), forms the doubly-robust
  advantage of folding, and trains the head on `sign(Δ̂)` weighted `|Δ̂|` from
  an init that is `INIT_FOLD` widened with a pass-through `relu(−margin)` unit
  (epoch 0 ≡ the old gate, verified numerically in the script before training).
  Holdout is seed parity. NO VECTOR SHIPS IT YET: `champion_test` asserts
  `k.fold === undefined`, and promotion needs the pre-registered paired grade
  (道場順位差 negative, 95% CI clear of zero, violations flat) and the owner's
  word.
  FIRST TRAINING ROUND (2026-08-29, same day): lane of 3,000 半荘 at ε=0.05
  (203,458 gated decisions, 10,150 flips, 6.3% of rounds with >1 flip;
  reproduction 0 mismatches). The lane's own quadrant table already says the
  home field carries almost no fold signal: pushing where the gate folds costs
  ~31 points on average (−1,509 vs −1,478), folding where it pushes ~324.
  Holdout DR: old gate −0.688, always-push −0.687, always-fold −0.835, fitted
  head (37→16→1, 40 epochs) −0.723; the paired grade agreed — champion+fold vs
  champion **+0.108 [+0.037, +0.180]**, −1,978 点/半荘, violations +10: a clear
  REGRESSION. A regularised refit (8 hidden, L2 1e-2, 10 epochs) collapsed to
  1.8% fold rate with DR ≈ always-push. Reading: against three frozen copies of
  itself deal-ins are cheap and the round-level reward is shared by ~7
  decisions, so there is nothing for a head to beat the gate on at home; the
  fold decision's cost shows only in the arena, which cannot be randomised.
  Nothing ships; the next lever is a reward that prices the arena's feed
  distribution (e.g. 道場順位/final-standings reward, or a lane against a
  pusher field), not more epochs.

- **暗刻を崩して七対子に向かわない — the `keepTriplet` guard (2026-08-28,
  owner's replay review)**: `kernel.shanten` is the MIN over standard/chiitoi,
  so to the discard score a concealed triplet is a pair plus a spare and
  cutting the third copy is free on the min line while the standard line
  loses a step; the sense's `chiitoiTax` never reaches the case because it
  exempts `sh < 2`, and the shape a break leaves is five pairs = 1向聴 (its
  comment argues only the SIX-pair exemption — the mismatch is on the table
  for the next sense re-grade; the trio was graded under the `< 2` guard).
  Verified on the four ranked wire logs (5,254 live discards): 25 breaks of a
  held triplet where the kept shape rode the pairs line strictly below
  standard, 17 of them not even into tenpai (`5m5m5m 6m6m` cutting 5m at
  4巡目). Fix: a CANDIDATE FILTER in `decide` beside `compliantDiscards`
  (never a price — no fitted core or planner malus can outbid it), gated by
  the `keepTriplet` heuristic weight: a 3→2 cut is struck when its standard
  shanten is worse than the best on offer AND the shape it leaves is chiitoi-
  below-standard at 1向聴 or worse. Exempt: breaking INTO 七対子聴牌 (単騎 is
  sanctioned), a break that keeps the best standard shanten (a standard-form
  decision), melded hands, and FOLDING hands (the triplet may be the only
  現物). Isolated on the same logs it moves 12/5,347 decisions, every one a
  triplet break replaced at equal-or-better standard shanten. DEFAULT 0
  (`frozen.ts` carries an explicit 0, the league snapshot inherits the default 0; no pin moved); ships in `weights/arena.json`
  as 1. Home paired grade (600 games, khhh, champion+guard vs champion, seed
  20828): 道場順位差 −0.027 [−0.059, +0.006], A優位 28 / B優位 19 / 同着 553,
  violations flat — a small gain, CI touching zero. Promoted into
  champion.json 2026-08-29 with the rest of the arena vector.

- **The assessor was fed DOUBLE-COUNTED own tiles (engine bug, found
  2026-08-28 in the ranked arena)**: `Table.visibleCounts(seat)` includes the
  seat's own hand (right for the ukeire `live` field), but `observe.ts`
  passed it to `assessDanger` as the PUBLIC count while also passing
  `ownCounts`, so a held pair killed every wait shape (`当たり形:なし` ⇒ 安全)
  and the champion — which honours an explicit 安全 as a proof above its
  deal-in estimate — threw a live 単騎 East into a 3-pon toitoi for 12,000.
  Fixed to mjrender's own contract (public counts + hand separately).
  Isolated on the 0827 wire log the fix changes 2.6% of live discards, every
  one a tile the FIXED build rates 危険度低 (the old build had them at 安全
  through the killed-shapes cap). The bug was SYMMETRIC at home
  (shared `observe()`), so no paired grade could see it — only the arena,
  where opponents do not share our bugs. Shared code ⇒ all fingerprint pins
  moved and were re-captured the same day by the owner's word (the 08-27
  precedent); frozen-h ≡ frozen-0825 re-verified across the re-capture.
  The same replay pass ("old vs new on real decisions": the live `> dahai`
  replies ARE the old build) shaped three sense/discipline corrections:
  the sense's honor surcharge is now `HONOR_SHARE` 0.5 and zero once <2
  copies remain unseen (at hot≈1.0 the folding bot had hoarded five honors
  while cutting live middle tiles); each opponent's void score is scaled by
  an honor-retention factor in the `fieldSense` reduction (lane-measured:
  ≥4 honors discarded ⇒ ×0.5, 2–3 ⇒ ×0.8 — late-game precision of the
  [0.5,0.7) bin 6.6%→12.6%; `fieldSenseDetail` untouched so the recorded
  lane keeps its semantics); and a `liveYakuhai` heuristic weight (default
  0) surcharges a 役牌-for-anyone honor with 0 public copies at junme ≥ 6
  when an opponent has called AND the assessor holds no entry for the tile
  (2/16 such releases were ronned at mangan+ across 35 arena games; under a
  riichi the assessor already prices it, and stacking made the replay swap a
  中 honor for a 高 number tile). Arena heat runs 2–3× hotter than home
  (SL-bot rivers are suit-skewed: hot≥0.7 on 31.6% of late decisions vs
  11.0% at home) — the home bench under-measures sense consumption costs.

- **色読み — the 感性 field sense (`ai/sense.ts`, 2026-08-28, owner-directed)**:
  the 0827 ranked arena batch (30 games, avg 2.60) fed 10 mangan+ deal-ins,
  and the owner's replay review found the cluster was 染め手 (mostly CLOSED
  flushes — invisible to both the danger assessor, which needs a riichi/furo
  threat to run, and the computed `dealinP`), plus false 七対子 commitment
  (`kernel.shanten` is the MIN over standard/chiitoi, so four early pairs flip
  the discard chooser onto the pairs line as an artifact). The owner's doctrine:
  色読み is NOT a river read (forbidden) — it is the 感性 sense of the 場,
  トイツ場 and 染め場, exposed as FIELD facts only (per-suit heat + one pairing
  scalar, never "seat N holds X"). Facts are fixed arithmetic; consumption is
  three ktune weights (`sense` block — a switch like `hand`/`riichi`, absent ⇒
  bit-identical play): `someRisk` (suit-heat surcharge added at every `riskOf`
  exit, the explicit 安全 proof included — that proof is against ASSESSED
  threats; the sense's own proof is the dye source's discards), `somePressure`
  (un-zeros the fold gate's quiet-table early-out), `chiitoiTax` (line tax
  beside `dojoCost` on BOTH scoreDiscard paths). Two measurement lessons are
  built in: the void score is DEFICIT below uniform expectation (share-based
  and raw-void scores both called 40%+ of mid-game decisions hot on the 0827
  wire replay), and consumption prices heat only above `HEAT_BAR` 0.35 (linear
  consumption cost +0.08 道場順位 per arm at home; thresholded it is free:
  {someRisk 200, somePressure 0.5, chiitoiTax 500} graded +0.008 ±0.051 over
  600 paired games, violations flat). Promoted into champion.json (and the
  frozen "h" seat) 2026-08-29 after the arena vector held ~1600 on riichi.dev.

- **The sense is ORACLE-CALIBRATED, and the 場 is per-player (owner doctrine,
  2026-08-28)**: `scripts/sense_lane.ts` plays headless hanchan and records
  `fieldSenseDetail` evidence next to truth read off the live Table (opponent
  suit shares, concealed pairs + pon count, shanten) with round outcomes
  joined by (kyoku, honba); `scripts/sense_fit.ts` renders reliability tables
  (recommends — the paired sweep still decides). The 0828 lane (600 半荘, 90k
  rows): dye heat is genuinely calibrated — P(dye-committed | heat ≥ 0.7) =
  17.5% mid-game vs a 1.8% base rate, monotone across bins — and few-honors-
  discarded sharpens it further (9.8% vs 5.5%), an unused corroborator noted
  for later. トイツ場 is monotone (18.9% → 51.7% P(field pairs) across bins)
  ONLY under a pon-aware truth label: a pon consumes the pair it proves, so
  concealed-pair counts anti-calibrate the pon-heavy bins — a label lesson,
  not a fact defect. The doctrine refinement in the same pass: the 場 is not
  shared by all four — usually one player is OUT of the field the other three
  are in — so own-hand pairing was REMOVED from the toitsuba field evidence
  (own pairs are the temptation to commit, never proof the field pairs; the
  alignment reading comes from the chiitoi tax's `1 − toitsuba` scale; the
  lane's observational slice is consistent — aligned pairing ran ~250
  points/round ahead of lone pairing, though junme/pon confounds keep that
  a direction, not a confirmation). Re-graded under the final formula: the
  trio {someRisk 200,
  somePressure 0.5, chiitoiTax 500} is −0.017 ±0.053 over 600 fresh paired
  games — free at home.

- **The dojo's own rules were CORRECTED 2026-08-27 (owner ruling), and the
  frozen seats play under the corrected dojo**: (1) 持ち点8000点未満 is judged
  at 終局, not per 局 — the rule moved to a new `on-game-end` hook (fired by
  the match drivers on the last round's table; its violations are appended to
  the MATCH ledger after the last `ledgerCut`, so no per-round slice owns
  them and the "m" line totals still count them), and `bufferScale` engages
  南入以降 only. (2) The call gate's 対々和 clause no longer passes any
  chi-free shape — the concealed rest must hold ≥ `4 − melds` pair-or-better
  types (arena wire logs: 15/36 pons had no other justification, one at
  5向聴) — and a バック needs its third tile LIVE. Both changes are shared
  code, so ALL fingerprint pins were re-captured that day — the ONE
  sanctioned exception to the frozen/league never-regenerate rule, by the
  owner's explicit direction ("apply fixes to both h agent"); the
  frozen-h ≡ frozen-0825 equality was re-verified across the re-capture.

- **最終形リーチ doctrine (2026-08-27 owner ruling, refined same day)**:
  immediate riichi is allowed for any tenpai with acceptance strictly > 2
  live tiles AND more hand than riichi(+平和)のみ — cheapness is priced by
  the M11 value model (`out.value` under `valueRiichi` + half a
  `valuePerDora`, dealer-scaled), not by a dora count — and always for the
  sanctioned 単騎 (ドラ単騎, 七対子単騎, 四暗刻単騎, 国士 shapes). Everything
  else HOLDS, released after ~2 own turns tenpai without the wait improving.
  Implemented as M12 head features: `holdShape` (the doctrine verdict —
  a conjunction the linear surface cannot build, computed by
  `riichiHoldShape`), `tenpaiHeld` (own turns at an unimproved wait — the
  base policy's only cross-decision state: feature-only, dead without a
  head, cleared by `reset`, and the arena bridge resets per kyoku), and
  `improvable` (informational: a live draw could rebuild a strictly wider
  wait, via `waitUpgradeExists`). INIT weights carry 0 for all three, so a
  headless vector still declares unconditionally; the doctrine ships as the
  `riichi` block in champion.json / arena.json
  (`{bias 0.1, holdShape −1, tenpaiHeld 0.5}`); since the 2026-08-29 epoch
  the frozen "h" seat carries the same head as a frozen object
  (`FROZEN_RIICHI`).

- **The M11 hand block was REMOVED from the champion 2026-08-25**: the
  post-epoch sweep re-grade (pre-registered rule: promote only on 道場順位差
  negative with 95% CI clear of zero) measured EVERY (pushScale, evWeight)
  cell WORSE than the no-hand-block incumbent — 31/31 completed cells
  positive with CIs clear of zero, harm monotone in pushScale (+0.18 @1500 →
  +0.52 @100000) — for BOTH the 08-23 fit and a fresh refit on a frozen-field
  lane, while the refit predicted held-out hand outcomes essentially
  identically to the 08-23 fit. So the defect is STRUCTURAL, not predictive:
  ev = pwin×value is ~4.4x flatter across shanten than the incumbent push
  table (`value` cancels most of pwin's gradient), and a multiplicative
  pushScale can match the push LEVEL but never that SHAPE, so every setting
  trades under-pushing good hands against over-pushing bad ones.
  `champion.json` is the M10 computed calibration alone; do NOT re-inject
  computed information into hand-tuned rules through a single scalar — feed
  it to a learned decision rule (the M12+ direction). Re-adding a hand block
  takes new controlled paired evidence, and `champion_test` fails loudly if
  one reappears by merge accident. Full grids: `runs/hand/SWEEP.md` addendum.

- **League of frozen snapshots (adopted 2026-08-25)**: at each meaningful
  improvement of the default "k" seat, freeze the champion configuration —
  `deno task freeze --label=MMDD [--ktune=FILE] [--plan]` writes
  `weights/league/frozen-MMDD.json`, a COMPLETE resolved ktune (the script
  self-checks the dump against the live seat before writing) — and grade
  future candidates against MIXED fields of past snapshots via `--table`. A
  monoculture of the current self manufactures style-specific overfitting,
  and that matters most for learned heads (M12's riichi head onward).
  `test/league_test.ts` pins every snapshot, and league pins NEVER regenerate
  — the opposite of `champion_test`'s, which regenerate on deliberate
  promotion (the champion is the present; the league is the past). When a new
  default field breaks a league pin, the fix is adding the explicit old value
  to that snapshot's JSON, never re-pinning. `frozen-0825` is snapshot #1
  (the 08-25 default seat); `frozen-0829` is the promoted champion and the
  same seat as the frozen "h" letter since the 08-29 re-bind, pins equal by
  construction.

- **The "h" seat was RE-BOUND to the champion 2026-08-29 (second epoch,
  owner's word: "we get a stable rank of 1600 on riichi.dev with k agent,
  time to promote it to h agents again")**: `champion.json` became the
  arena vector minus the arena-only buffer overrides, and `src/ai/frozen.ts`
  became a complete frozen copy of THAT champion — generated through the
  merge functions, never transcribed — now including `FROZEN_RIICHI` and
  `FROZEN_SENSE`; `weights/league/frozen-0829.json` is its snapshot and
  frozen-h ≡ frozen-0829 was verified on all three seeds; frozen-0825's pins
  did not move (no shared code changed). DEFAULT_* constants did not move: a
  bare "k" seat still plays the default game, so "h" is no longer a copy of
  the defaults but of the champion. champion/frozen pins re-captured;
  runs/ numbers before 08-29 were measured against the 08-25 h and are not
  comparable forward. The first epoch, 2026-08-25: the original hand-written
  heuristic agent was retired; the letter builds a FROZEN copy
  (`src/ai/frozen.ts` — complete weight objects, snapshotted, configurable by
  nothing; `test/frozen_test.ts` pins it and that pin NEVER regenerates
  except by the owner's explicit word). Vectors route to "k" seats only; `loadTable`/`argError` refuse
  configuration aimed at an "h" seat. Consequences: numbers in `runs/`
  recorded before the epoch were measured against the old h population and are
  not comparable forward, and `--ktune-opp` under `paired` now requires an
  incumbent control (`--ktune-b`/`--consumer-b`) — the default hhhh control
  arm is frozen and no vector reaches it.

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
