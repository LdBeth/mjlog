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
  WORSE. `weights/champion.json` is BY CONVENTION the current champion: the
  TUI 助言 advisor always reads it (`src/main.ts`; play's CPU seats carry no
  vector) — never point it elsewhere; improve the file. Promotion requires
  controlled paired evidence (opponents held fixed) and deliberately
  regenerates the `test/champion_test.ts` pins. Since 2026-08-25 it is the
  M10 computed calibration ALONE — the post-epoch sweep removed the M11 hand
  block (see Decisions); `hand-calibrated.json` stays archived. Tracked under
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
  `weights/arena.json` — the champion's computed block and 最終形 riichi
  block plus arena-only heuristic overrides (`bufferTight`/`bufferLow` = 1:
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
  violations flat — a small gain, CI touching zero. Promotion into
  champion.json is the owner's call.

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
  600 paired games, violations flat). NO vector ships it yet — adding `sense`
  to champion.json is a promotion (pins regenerate) awaiting the owner's word.

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
  (`{bias 0.1, holdShape −1, tenpaiHeld 0.5}`), never in the frozen seats.

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
  to that snapshot's JSON, never re-pinning. `frozen-0825` is snapshot #1:
  the same seat as the frozen "h" letter, pins equal by construction.

- **The "h" seat was RE-BOUND 2026-08-25 (epoch)**: the original hand-written
  heuristic agent is retired; the letter now builds a FROZEN copy of the
  default 計算 seat (`src/ai/frozen.ts` — complete weight objects, snapshotted,
  configurable by nothing; `test/frozen_test.ts` pins it and that pin NEVER
  regenerates). Vectors route to "k" seats only; `loadTable`/`argError` refuse
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
