// The headless driver: seat letters → policies, and a run of matches over them.
//
// Everything here is CLI-independent on purpose — `main.ts` only translates
// flags into the options below, and tests (and `scripts/`) drive the same
// functions without spawning a process.
//
// LIFETIME. A seat is built ONCE per run and re-seeded per match
// (`SeatPolicy.reset`), never rebuilt: an "n" seat's constructor re-reads ~4MB
// of weights and mallocs a native context, so per-match construction leaked one
// context per hanchan — hundreds of them over a shard. What a fresh
// construction used to give the run, `reset` gives it instead: the seat's rng
// re-seeded, its per-hand caches dropped, and its Reads provider rebuilt (the
// noise / curriculum providers carry an rng of their own, and its stream must
// start where a newly built seat's did, or a run would stop reproducing).

import { RandomPolicy } from "./ai/random.ts";
import type { HeuristicWeights } from "./ai/heuristic.ts";
import {
  AugmentedHeuristic,
  calibrationReads,
  curriculumReads,
  mergeAugmented,
  noisyReads,
  oracleReads,
  parseChannels,
} from "./ai/augmented.ts";
import type { AugmentedWeights, OracleChannel, ReadsProvider } from "./ai/augmented.ts";
import type { CalibrationWriter } from "./ai/calibration.ts";
import type { CalibRecord } from "./ai/calibration.ts";
import { computedReads, mergeComputed } from "./ai/computed.ts";
import {
  buildDealinHeads,
  closeDealinHeads,
  dealinRecordExtras,
  learnedReads,
  mergeDealin,
} from "./ai/dealin.ts";
import { buildEv, closeEv } from "./ai/ev.ts";
import type { EvCore } from "./ai/ev.ts";
import { mergeEv } from "./ai/evparams.ts";
import { mergeFold } from "./ai/fold.ts";
import type { FoldSample } from "./ai/fold.ts";
import type { FoldCalibrationWriter } from "./ai/foldcalib.ts";
import { buildMlp, closeMlp } from "./ai/mlp.ts";
import type { ComputedTraceRef, ComputedWeights } from "./ai/computed.ts";
import { parseConsumerParams } from "./ai/consumer.ts";
import type { ConsumerParams } from "./ai/consumer.ts";
import {
  FROZEN_AUGMENT,
  FROZEN_COMPUTED,
  FROZEN_HEURISTIC,
  FROZEN_RIICHI,
  FROZEN_SENSE,
} from "./ai/frozen.ts";
import type { HandCalibrationWriter, HandSample } from "./ai/handcalib.ts";
import type { EvCalibrationWriter, EvSample } from "./ai/evcalib.ts";
import { mergeHand } from "./ai/handvalue.ts";
import type { HandWeights } from "./ai/handvalue.ts";
import { mergeRiichi } from "./ai/riichi.ts";
import type { RiichiWeights } from "./ai/riichi.ts";
import { DEFAULT_STANDINGS_WEIGHTS } from "./ai/standings.ts";
import { die } from "./cli/die.ts";
import { makeDojoHooks } from "./dojo.ts";
import { encodeOracle } from "./rl/features.ts";
import { NeuralPolicy } from "./rl/policy.ts";
import { RecordingPolicy, TrajectoryWriter, writeMatchEnd } from "./rl/record.ts";
import type { LineCounts } from "./rl/record.ts";
import { runMatchSync } from "./match.ts";
import type { MatchResult } from "./match.ts";
import type { SyncPolicy } from "./policy.ts";
import { DOJO_HEADLESS, JANKI } from "./rules.ts";
import { scorer } from "./score.ts";
import { kindString, resolveTable } from "./spec.ts";
import type { KTune, SeatSpec, TableSpec } from "./spec.ts";
import type { Table } from "./table.ts";
import { SEATS } from "./types.ts";
import type { RoundOutcome, Seat } from "./types.ts";

/** Where `--weights` points by default: what the trainer writes. */
export const DEFAULT_WEIGHTS = "weights/manifest.json";

/**
 * What `--calibrate` asks the oracle for, fixed and independent of `--oracle`:
 * the ron mask (C1), tenpai (C2) and the payment (C3) — the three truths the
 * 計算 model makes a claim about. Tying it to the user's channel set would let a
 * `--oracle=none` run write records with no truth in them.
 */
const CALIBRATION_CHANNELS = parseChannels("C1,C2,C3")!;

// The seat/table description layer lives in `spec.ts`; re-exported here because
// this module has always been where the CLI and the tests read them from.
export { loadConsumer, loadKtune, loadTable, resolveTable } from "./spec.ts";
export type { KTune, SeatSpec, TableSpec } from "./spec.ts";

/** What an "o" seat needs: the live-Table tap and the channels it may read. */
export interface OracleWiring {
  get: () => Table | null;
  channels: Set<OracleChannel>;
  /**
   * Oracle fading: per decision, per information group, the probability that the
   * truth is withheld. 0 (the default) leaves the provider untouched.
   */
  noise?: number;
}

/**
 * Everything one seat is built from: its `SeatSpec` — the complete, plain-JSON
 * description of WHO the seat is (`spec.ts`) — plus the wiring only the driver
 * holds: a name, an rng seed, the live-Table tap, and the recording sinks.
 */
export interface MakePolicyOptions extends SeatSpec {
  name: string;
  /** The seat's own rng seed, and what `reset` is handed for the first match. */
  seed: number;
  /** The live-Table tap; absent in `play`, which has no hidden-info seats. */
  oracle?: OracleWiring;
  /** M10a: where a "k" seat's calibration records go. */
  calibrate?: (rec: CalibRecord) => void;
  /**
   * M11: where this seat's 手牌価値 samples go. Unlike `calibrate` it needs no
   * tap on the Table — the prediction is a function of the Observation, and the
   * LABEL arrives later, from the round outcome, through the writer.
   */
  handSink?: (rec: HandSample) => void;
  /**
   * M13: where this seat's fold samples go, and how often the verdict is
   * flipped on the way out. A "k" seat only (plan D11 — the frozen "h" letter
   * takes no head and no lane).
   *
   * `foldEps` 0 (or absent) draws no random numbers, so a lane recorded at ε=0
   * is bit-identical to a run with no lane at all — the recorder is an
   * observer, exactly as `calibrate` and `handSink` are.
   */
  foldSink?: (rec: FoldSample) => void;
  foldEps?: number;
  /**
   * M15b: where this seat's EV-core calibration wires go. A "k" seat only, and
   * only on a vector with NO `ev` block — `makePolicy` refuses the combination
   * below, because a lane recorded under the DP is censored by the DP's own
   * folds (the M11 lesson, measured).
   *
   * Like `handSink` it needs no tap on the Table and perturbs nothing: the
   * policy packs the wire it would have handed a core and hands it to the
   * writer instead.
   */
  evSink?: (rec: EvSample) => void;
}

/**
 * A seat that outlives the match it plays: the policy itself, plus the two
 * things only its builder knows how to do — put it back into the state a freshly
 * constructed one would be in, and release whatever native memory it holds.
 */
export interface SeatPolicy {
  readonly policy: SyncPolicy;
  /** Construction-equivalent state for a new match, under a new seed. */
  reset(seed: number): void;
  /** Release native resources. Idempotent. */
  close(): void;
}

/** A seat whose whole state is its rng and its per-hand caches. */
function reseeded(policy: SyncPolicy & { reset(seed: number): void }): SeatPolicy {
  return { policy, reset: (s) => policy.reset(s), close: () => {} };
}

/**
 * A seat that reads through a provider chain, rebuilt on every reset.
 *
 * `noisyReads` / `curriculumReads` hold an rng seeded at BUILD time, so the
 * schedule of dropped information groups is a property of the chain instance.
 * Rebuilding it is what makes match N of a long run play exactly the game a
 * one-match run of the same seed plays — which is the invariant `paired` rests
 * on. The seat reads through a stable indirection so the rebuild never has to
 * touch (or replace) the policy object itself.
 */
function withReads(
  build: () => ReadsProvider,
  make: (provider: ReadsProvider) => AugmentedHeuristic,
): SeatPolicy {
  let cur = build();
  const policy = make((obs) => cur(obs));
  return {
    policy,
    reset(s) {
      cur = build();
      policy.reset(s);
    },
    close: () => {},
  };
}

/** Seat letter → policy. Seeded per seat so a match seed reproduces exactly. */
export function makePolicy(o: MakePolicyOptions): SeatPolicy {
  const { kind, name, seed, oracle } = o;
  // 順位効用 (`--standings`). Only the heuristic family has a push/fold gate to
  // scale, so only "h" and "k" can carry it; the caller has already decided WHICH
  // seat is allowed to (see `headless`), because a layer applied to every seat at
  // once would move both sides of a paired measurement.
  const rank = o.standings ? { standings: DEFAULT_STANDINGS_WEIGHTS } : undefined;
  // M11 手牌価値. The one `--ktune` section that reaches an "h" seat as well as a
  // "k" one, deliberately: it is not a 感性 vector but a decision MODEL, and the
  // thing it replaces (the push table, the linear dora term) is code both seat
  // kinds share — so withholding it from "h" would leave the model untestable
  // against the baseline it is meant to beat. Absent (no `hand` section in the
  // file) it stays undefined and neither seat kind changes by a single bit.
  const hand = o.ktune?.hand && mergeHand(o.ktune.hand);
  // M12 riichi head. Same cross-kind reach as `hand`, for the same reason: a
  // decision MODEL, not a 感性 vector — the pre-fit lane and the baseline it
  // must beat are both "h" seats, so withholding it from "h" would leave it
  // untestable. Absent (no `riichi` section) neither seat kind changes by a
  // bit. With `--ktune-opp` the opponents' file may carry its own block —
  // nothing extra to wire, since each seat's vector fully describes that seat.
  const riichiHead = o.ktune?.riichi && mergeRiichi(o.ktune.riichi);
  if (kind === "r") return reseeded(new RandomPolicy(name, seed));
  if (kind === "k") {
    // 計算. The same consumption terms as an "o" seat, fed by exact counting
    // over the Observation instead of by the engine's hidden state — so unlike
    // "o" this seat needs no tap on the Table and is legal in `play` too.
    // `--oracle` / `--noise` configure `oracleReads` and do not reach it.
    //
    // `--ktune` reaches THIS seat and only this seat: the outcome tuner grades a
    // candidate 感性 vector against an untouched baseline, so anything the
    // vector could also move on the other side of the comparison would make the
    // measurement circular. A file that names `planner` outranks `--plan`,
    // because a vector is a complete description of a seat.
    //
    // M10a: with `calibrate` on, the provider also fills a trace out-param on
    // every call. The trace is written to a file and never read by the policy,
    // so the seat below is the same seat either way — that is what the paired
    // self-diff test in `test/calibration_test.ts` pins.
    if (o.calibrate && !oracle) {
      die("--calibrate は selfplay / paired 専用です (真値を読む Table の tap が要ります)");
    }
    // M14's deal-in heads. Same ownership rules as `foldHead` below — BUILT
    // objects that may hold a native context, so they are constructed exactly
    // once here (never inside `build`, which `withReads` re-runs every hanchan:
    // a head built there would allocate a native context per game and free
    // none of them) and freed in the seat's `close()`. "k" only (plan D11).
    // `mergeDealin` throws on a malformed block — and on `{}`, which unlike
    // every other section has NO identity: absent is the switch.
    const dealinHeads = o.ktune?.dealin ? buildDealinHeads(mergeDealin(o.ktune.dealin)) : undefined;
    // The seat's own value model, resolved once: `learnedReads` prices its
    // rebuilt `dealinValue` (D5) through the same weights `computedReads`
    // merges internally, so the head values a type exactly as this seat does.
    const cw = dealinHeads
      ? mergeComputed({ planner: o.plan ?? false, ...o.ktune?.computed })
      : undefined;
    const build = (): ReadsProvider => {
      const traceRef: ComputedTraceRef = { t: null };
      const computed = computedReads(
        { planner: o.plan ?? false, ...o.ktune?.computed },
        o.calibrate || dealinHeads ? traceRef : undefined,
      );
      // M14: the learned read REPLACES computed's `dealinP`/`tenpaiP`, reading
      // the trace `computed` fills — hence the same `traceRef`. It sits UNDER
      // the recorder and the curriculum: those measure or mix what the seat
      // actually consumes, which is this.
      const reads = dealinHeads && cw
        ? learnedReads(dealinHeads, cw, computed, traceRef)
        : computed;
      // M9c curriculum. Still the 計算 seat — the reader is what changes: each
      // information group is answered by the oracle with probability 1−E and by
      // the counting reader with probability E, per decision. E=1 returns
      // `computed` itself, so the seat this trains for and the seat it is finally
      // graded as are the same object. Headless only, like every oracle path: no
      // `oracle` wiring (i.e. `play`) means the curriculum is silently impossible,
      // and `parseArgs` refuses the flag there rather than letting it be ignored.
      if (o.calibrate) {
        return calibrationReads(
          reads,
          traceRef,
          // The recorder's own channel set, deliberately NOT `--oracle`: the three
          // truths the model makes claims about (tenpai, the ron mask, the
          // payment), whatever an "o" seat elsewhere in the run was allowed.
          oracleReads(oracle!.get, scorer, CALIBRATION_CHANNELS),
          o.calibrate,
          // Only a seat RUNNING the heads can say what rows it was served, so
          // the digest (`fh`) is gated on them: on a plain computed seat the
          // callback is absent and the recorder costs exactly what it always
          // did. (`--calibrate` beside a `dealin` block is refused at the CLI
          // — D6 — so this is the in-code path only.)
          dealinHeads ? dealinRecordExtras : undefined,
        );
      }
      if (o.curriculum !== undefined && oracle) {
        return curriculumReads(
          oracleReads(oracle.get, scorer, oracle.channels),
          reads,
          o.curriculum,
        );
      }
      return reads;
    };
    // M15's expected-value core. Same ownership rule as the fold head below —
    // built exactly once HERE, never inside `build`, freed in the seat's
    // `close()` — and built FIRST, so a refused layout is refused before any
    // other component has allocated anything. It is the strictest of the
    // family in two further ways.
    //
    // IT REFUSES WHAT IT SUPERSEDES (plan D3). `mergeEv` is pure data, so the
    // conflicts below are decided BEFORE `buildEv` is ever called: a layout
    // that is refused must be refused on a machine with no dylib, or the
    // diagnostic a user gets for a bad table would be "build the library
    // first". A block that would be loaded and then never consulted is a
    // silent no-op, which is the one thing the flag discipline does not allow.
    //
    // AND IT HAS NO FALLBACK. `buildEv` THROWS on a missing/stale `libmjev`,
    // on a missing `--allow-ffi`, and under `MJGAME_NATIVE=0`; a seat that
    // quietly went back to the linear surrogate would be graded as the DP.
    const evParams = o.ktune?.ev ? mergeEv(o.ktune.ev) : undefined;
    if (evParams) {
      // THROWN, not `die`d: `die` exits the process, and this refusal has to
      // be assertable from a test that never spawns one (`mergeFold` /
      // `mergeDealin` refuse a malformed block the same way, and the CLI turns
      // the exception into a message).
      const both = (other: string): never => {
        throw new Error(
          `ev ブロック と ${other} は併用できません (M15 D3: EV核が置き換える側なので、` +
            `${other} は積まれても一度も参照されません — 黙って無効になるくらいなら拒否します)`,
        );
      };
      // M15b, and NOT gated on a sub-switch: the calibration lane must be the
      // PLAIN champion's continuation of the hand, so any `ev` block at all
      // disqualifies the seat from recording it.
      if (o.evSink) both("--evcalib (M15b EV核レーン)");
      if (evParams.discard) {
        // The DP owns the discard score core AND the push/fold verdict, so
        // every incumbent owner of either is refused while `ev.discard` is on.
        if (o.consumer) both("consumer (M9 の学習消費器)");
        if (o.ktune?.hand) both("hand ブロック (M11 手牌価値)");
        if (o.ktune?.fold) both("fold ブロック (M13 押し引きヘッド)");
        if (o.foldSink) both("--foldcalib (M13 押し引きレーン)");
      }
      // `ev.riichi` replaces the declare-vs-damaten decision inside the gates,
      // which is exactly what the M12 head does (plan D3: unit C is graded as
      // `champion − riichi + ev`, so the substitution has to be explicit).
      if (evParams.riichi && o.ktune?.riichi) both("riichi ブロック (M12 リーチヘッド)");

      // D4 の換算率が本当に換算率であること。`dealinCostPts` は `riskOf` と
      // 「同じ式の単位違い」であることを契約にしていて、その等式が成り立つのは
      // λ = 1/pointsPerScore のときだけです (λ は線形サロゲートの消費スカラ、
      // pointsPerScore は DP の交換レート — 別々に調整できてしまう)。ずれた
      // まま走ると、同じ牌が同じ判断の中で二つの値段を持つことになるので、
      // 積む前に断ります。dylib は要りません (mergeEv と同じく純データ)。
      const lambda = mergeAugmented(o.ktune?.augment).lambda;
      const product = evParams.pointsPerScore * lambda;
      if (Math.abs(product - 1) > 1e-9) {
        throw new Error(
          `ev.pointsPerScore × augment.lambda は 1 でなければなりません ` +
            `(${evParams.pointsPerScore} × ${lambda} = ${product}): ` +
            `dealinCostPts は riskOf を points に換算しただけのものという契約 (M15 D4) が` +
            `この等式に乗っています — どちらか一方だけを動かすと、同じ牌が同じ判断の中で` +
            `二つの値段を持ちます`,
        );
      }
    }
    const evCore: EvCore | undefined = evParams ? buildEv(evParams) : undefined;
    // M13's fold head. Unlike `hand`/`riichi` this is a BUILT object — it may
    // hold a native context — so it is constructed exactly once, here, and
    // freed in the seat's `close()` below. And unlike them it reaches a "k"
    // seat ONLY (plan D11): the frozen "h" letter is the baseline this head has
    // to beat, and a baseline that moved with the candidate would measure
    // nothing. `mergeFold` throws on a malformed block; the loader's caller
    // turns that into a die message.
    const foldHead = o.ktune?.fold ? buildMlp(mergeFold(o.ktune.fold)) : undefined;
    const seat = withReads(build, (reads) =>
      new AugmentedHeuristic(name, seed, reads, {
        // 順位効用 is merged AFTER the tuned vector so the two compose: a
        // `--ktune` file tunes the terms 順位効用 then scales.
        weights: rank ? { ...o.ktune?.heuristic, ...rank } : o.ktune?.heuristic,
        augment: o.ktune?.augment,
        // M9 composes AFTER `--ktune`, and orthogonally to it: the curves
        // replace the CONSUMPTION of the evidence, while the tuned vector still
        // decides what the evidence says (`riskOf`'s ladder, the fold scales,
        // the 計算 reader's own constants).
        consumer: o.consumer,
        hand,
        riichi: riichiHead,
        sense: o.ktune?.sense,
        handSink: o.handSink,
        evSink: o.evSink,
        fold: foldHead,
        foldSink: o.foldSink,
        ev: evCore,
        ...(o.foldEps ? { foldExplore: { eps: o.foldEps } } : {}),
      }));
    if (!foldHead && !dealinHeads && !evCore) return seat;
    // `withReads` returns a no-op `close`; the heads are the one thing this seat
    // owns that the process would otherwise leak per seat, per run.
    return {
      ...seat,
      close: () => {
        if (foldHead) closeMlp(foldHead);
        if (dealinHeads) closeDealinHeads(dealinHeads);
        // Idempotent by contract (`closeEv`), because `closeArm` is.
        if (evCore) closeEv(evCore);
      },
    };
  }
  if (kind === "o") {
    // The oracle reads the round in play through `MatchOptions.tableRef`. There
    // is no such tap in `play` — and a CPU that could see the human's hand is
    // not something to offer by accident.
    if (!oracle) {
      die(
        "o席 (オラクル増補) は selfplay / bench / paired 専用です。\n" +
          "play では隠蔽情報を読む席は作れません。",
      );
    }
    const eps = oracle.noise ?? 0;
    const build = (): ReadsProvider => {
      const truth = oracleReads(oracle.get, scorer, oracle.channels);
      return eps > 0 ? noisyReads(truth, eps) : truth;
    };
    return withReads(build, (reads) => new AugmentedHeuristic(name, seed, reads));
  }
  if (kind === "n") {
    // Eager load: a seat that cannot think is better refused at startup than
    // discovered mid-hanchan.
    let p: NeuralPolicy;
    try {
      p = new NeuralPolicy(name, seed, o.weights ?? DEFAULT_WEIGHTS, { temperature: o.temp ?? 0 });
    } catch (e) {
      die(
        `${e instanceof Error ? e.message : String(e)}\n` +
          `n席 (学習済みポリシー) には重みが要ります。先に trainer を回すか、\n` +
          `\`python train/randinit.py\` で初期重みを作ってから --weights=PATH を指定してください。`,
      );
    }
    // The one seat kind that holds memory the process does not free on its own.
    return { policy: p, reset: (s) => p.reset(s), close: () => p.close() };
  }
  // "h" — 2026-08-29 EPOCH (first bound 2026-08-25). The letter builds a
  // FROZEN copy of the champion (`ai/frozen.ts`): its riichi head and sense
  // block are frozen objects, not ktune input. Nothing configurable reaches
  // it — no ktune, no hand block, no consumer/standings/planner/curriculum — which is
  // the property that makes it a baseline: `resolveTable` routes the vectors
  // to "k" seats only, and `loadTable`/`argError` refuse the attempt loudly.
  // The one thing that still composes is `handSink`: a recording tap, not a
  // policy knob — the seat plays identically with or without it.
  const frozen = (): ReadsProvider => computedReads(FROZEN_COMPUTED);
  return withReads(frozen, (reads) =>
    new AugmentedHeuristic(name, seed, reads, {
      weights: FROZEN_HEURISTIC,
      augment: FROZEN_AUGMENT,
      riichi: FROZEN_RIICHI,
      sense: FROZEN_SENSE,
      // Spelled out even though they equal today's constructor defaults: a
      // default is a LIVE value, and the frozen seat may not reference one —
      // if these ever drift for "k" experimentation, this seat must not move.
      kuitan: true,
      dojo: true,
      epsilon: 0,
      handSink: o.handSink,
    }));
}

export interface HeadlessOptions {
  /** Manifest path handed to any "n" seat. */
  weights?: string;
  /** Softmax temperature for any "n" seat; omitted or 0 = greedy. */
  temp?: number;
  /** Trajectory JSONL to record into; one writer for the whole run. */
  record?: string;
  /**
   * A ready-made trajectory sink, used INSTEAD of opening `record`. Only the
   * `--jobs` workers pass it (with a buffering writer): the dataset file belongs
   * to the main thread, and a worker that opened `record` itself would truncate
   * the run's one file N times over. Ignored when null/absent.
   */
  writer?: TrajectoryWriter | null;
  /** Record every seat (BC teacher data), not just the "n" ones. */
  recordAll?: boolean;
  /** Hidden-information channels any "o" seat may read. */
  oracle?: Set<OracleChannel>;
  /** Per-decision, per-group dropout applied to those channels (0 = off). */
  noise?: number;
  /**
   * M9c: SEAT 0's reader becomes an oracle→計算 curriculum at this dropout rate.
   * Seat 0 and seat 0 alone, for the same reason `standings` and `consumer` are:
   * a reader handed to every seat would move both sides of a paired comparison.
   * `undefined` leaves every seat exactly as it was.
   */
  curriculum?: number;
  /** Engage the C7 planner on any "k" seat. */
  plan?: boolean;
  /**
   * Engage 順位効用 on SEAT 0, and on seat 0 alone. Deliberately narrower than
   * every other option here: the layer applies to any heuristic-family seat, so
   * a run-wide switch would move all four of them at once and there would be
   * nothing left to measure the one against.
   */
  standings?: boolean;
  /**
   * Tuned 感性 vector for any "k" seat. Every other seat kind ignores it — see
   * `makePolicy`, and `pairedRun`, which deliberately withholds it from the
   * control arm.
   */
  ktune?: KTune;
  /**
   * The OPPONENTS' (seats 1–3) tuning vector, distinct from the subject's
   * `ktune`. Absent, the opponents share the subject's `--ktune` exactly as
   * before — bit-for-bit; the option exists so a SECOND competent population can
   * be built, which is what measuring whether fitted parameters transfer needs.
   *
   * Unlike `ktuneB` this is not a control-arm knob: the opponents are the
   * ENVIRONMENT, so `pairedRun` hands this to BOTH arms unchanged.
   */
  ktuneOpp?: KTune;
  /**
   * `pairedRun` ONLY (`headless` ignores it): an EXPLICIT control table — the
   * `--table-b` form of the incumbent comparison. Set, the control arm is built
   * from these four specs instead of from `ktuneB`/`consumerB` (which the CLI
   * refuses alongside it), and `pairedRun` REFUSES the pair unless seats 1–3
   * match arm A's resolved specs exactly: the opponents are the environment,
   * and the environment must be identical in both arms.
   */
  tableB?: TableSpec;
  /**
   * `pairedRun` ONLY (`headless` ignores it): the CONTROL arm's 感性 vector.
   *
   * The scalar-space half of the variance-reduction instrument (M10d), and the
   * exact mirror of `consumerB` below — same seats, same policy dice, same
   * curves, same oracle/curriculum wiring, only the `--ktune` file differs. What
   * `paired` then measures is candidate MINUS INCUMBENT, which is the comparison
   * a search actually asks about; against plain `hhhh` a twelve-scalar nudge is
   * swamped by the ~1.4 per-seed SD of two different players' rank difference.
   * The two may compose: with both set, arm B gets its own vector AND its own
   * curves.
   */
  ktuneB?: KTune;
  /**
   * M9's learned consumer, for SEAT 0 and seat 0 alone — the same narrowing as
   * `standings`, and the same reason: a curve set applied to every seat would
   * move both sides of a paired measurement at once.
   */
  consumer?: ConsumerParams;
  /**
   * `pairedRun` ONLY (`headless` ignores it): the CONTROL arm's curve set.
   *
   * Setting it changes what `paired` measures, and it is the whole point of the
   * option. Against the default control (`hhhh`, no curves) a small perturbation
   * of a curve set is swamped: the two arms are different players, most games
   * diverge on the first turn, and the per-seed SD of the rank difference is
   * ~1.4 — so a 1000-seed run carries a ±0.08 CI, which cannot rank neighbours
   * that differ by 0.02. Against an INCUMBENT the two arms are the same player
   * up to a nudge, most games stay bit-identical to the end, and the difference
   * is measured where it actually occurs. Same seeds, same policy dice, same
   * `--ktune`, same oracle/curriculum wiring — only the curve file differs.
   */
  consumerB?: ConsumerParams;
  /**
   * M10a: SEAT 0's calibration recorder, one file for the whole run.
   *
   * A WRITER rather than a path, because `pairedRun` plays one game at a time and
   * a path would truncate the file at every seed. It is also why the control arm
   * never receives it — `pairedRun` strips it explicitly, the same discipline
   * `record` follows.
   */
  calibrate?: CalibrationWriter;
  /**
   * M11: SEAT 0's 手牌価値 recorder, one file for the whole run.
   *
   * A WRITER and not a path, for `calibrate`'s reason (a path would truncate at
   * every seed of a `paired` run) plus one of its own: a sample is only half a
   * record when the policy emits it. The label — did that hand cash, for how
   * much, did it deal in instead — is written by `endRound`, so the writer has
   * to be the object that spans the decision AND the outcome.
   *
   * Honoured for a "k" OR an "h" seat 0, unlike `calibrate`: the model is
   * heuristic-family code, and the pre-fit lane is played by plain `hhhh`.
   */
  handCalib?: HandCalibrationWriter;
  /**
   * M13: SEAT 0's fold recorder, one file for the whole run. A WRITER and not a
   * path, for `handCalib`'s two reasons (a path would truncate at every seed of
   * a `paired` run, and the label is the 局's own settlement, which only an
   * object spanning the decision AND the outcome can attach).
   *
   * A "k" seat 0 only — narrower than `handCalib`'s k-or-h, because the head
   * this lane fits reaches "k" alone (plan D11).
   */
  foldCalib?: FoldCalibrationWriter;
  /**
   * M15b: SEAT 0's EV-core recorder, one file for the whole run. A WRITER and
   * not a path, for `handCalib`'s two reasons — and for a third of its own: the
   * writer OWNS the `EvCore` every wire is evaluated through, so it is the only
   * object in the run that may be built once and freed once.
   *
   * A "k" seat 0 only, and (enforced in `makePolicy`) only on a vector with no
   * `ev` block.
   */
  evCalib?: EvCalibrationWriter;
  /**
   * M13: the ε of the fold-flip lane, 0 < ε < 1. 0 (the default) is a pure
   * observation lane: the recorder writes what the incumbent gate decided and
   * nothing is perturbed. Only meaningful with `foldCalib`; the CLI refuses the
   * flag on its own rather than letting it be ignored.
   */
  foldEps?: number;
}

/**
 * A run's four seats, its trajectory writer and its window onto hidden state,
 * all of them built once. `playGame` re-seeds the seats and plays one match;
 * `closeArm` releases the file and the native contexts.
 *
 * `pairedRun` holds TWO of these open at once — that is why an arm is a value
 * rather than a scope inside `headless`.
 */
export interface Arm {
  /** Kind letters, for reports — derived from the table ("khhh"). */
  readonly seats: string;
  /** The four resolved specs this arm was built from. */
  readonly table: TableSpec;
  /** What the engine plays: the seats, wrapped in recorders where asked for. */
  readonly policies: SyncPolicy[];
  /** The same seats, unwrapped, for `reset` and `close`. */
  readonly built: SeatPolicy[];
  readonly writer: TrajectoryWriter | null;
  /** The tap the recorder and any oracle-reading seat share. */
  readonly ref: { t: Table | null };
  /** Non-null only when something in this arm actually reads the Table. */
  readonly tableRef: { t: Table | null } | undefined;
  /** Set only when seat 0 is a "k" seat under a calibration run. */
  readonly calibrate: CalibrationWriter | undefined;
  /** Set only when seat 0 is a heuristic-family seat under a 手牌価値 lane. */
  readonly handCalib: HandCalibrationWriter | undefined;
  /** Set only when seat 0 is a "k" seat under an M13 fold lane. */
  readonly foldCalib: FoldCalibrationWriter | undefined;
  /** Set only when seat 0 is a "k" seat under an M15b EV核 lane. */
  readonly evCalib: EvCalibrationWriter | undefined;
}

/**
 * Open one arm — from a legacy seats-string (whose per-seat conventions
 * `resolveTable` reproduces bit for bit) or from an explicit `TableSpec`, the
 * modular form: four complete seats, each with its own components and weights.
 * A string arm is DEFINED as `openArm(resolveTable(seats, opts), opts)`, so
 * the two paths cannot drift.
 */
export function openArm(seats: string | TableSpec, opts: HeadlessOptions = {}): Arm {
  const table = typeof seats === "string" ? resolveTable(seats, opts) : seats;
  const kinds = kindString(table);
  // One file, one handle, every seat and every match of the run: the trainer
  // reads a single stream and the "r"/"m" lines terminate each match in it.
  // …unless the caller brought its own sink (a `--jobs` worker does).
  const writer = opts.writer ?? (opts.record ? new TrajectoryWriter(opts.record) : null);
  // The recorder's window onto hidden state. `runMatchSync` points it at the
  // round in play, so `ref.t` is non-null for the whole life of a decision;
  // outside a round nobody calls the tap.
  // ...and the oracle seats' window onto the same thing. One tap serves both.
  const ref: { t: Table | null } = { t: null };
  const oracleSeats = table.some((s) => s.kind === "o");
  // The curriculum's oracle half needs the same tap, on a "k" seat that would
  // otherwise never ask for it.
  const curriculumOn = table.some((s) => s.kind === "k" && s.curriculum !== undefined);
  // …and so does the calibration recorder, for the truth half of its records.
  const calibrateOn = opts.calibrate !== undefined && table[0].kind === "k";
  // M11's lane. No tap needed — the writer reads the Table through `onRoundEnd`
  // — and "h" qualifies as well as "k": the pre-fit lane is plain `hhhh`.
  const handCalibOn = opts.handCalib !== undefined &&
    (table[0].kind === "k" || table[0].kind === "h");
  // M13's lane. "k" ONLY: the head the lane fits is routed to "k" seats alone,
  // and a lane recorded off the frozen baseline would fit a rule that seat can
  // never carry.
  const foldCalibOn = opts.foldCalib !== undefined && table[0].kind === "k";
  // M15b's lane. "k" ONLY, like M13's: the core the lane fits is routed to "k"
  // seats alone. (The `ev`-block refusal is `makePolicy`'s, where the vector is.)
  const evCalibOn = opts.evCalib !== undefined && table[0].kind === "k";
  const wiring: OracleWiring = {
    get: () => ref.t,
    channels: opts.oracle ?? new Set(),
    noise: opts.noise ?? 0,
  };
  const built = table.map((spec, seat) =>
    makePolicy({
      // The seat IS its spec — kind, vector, components, weights — plus the
      // wiring only this driver holds.
      ...spec,
      name: `${spec.kind.toUpperCase()}${seat}`,
      // A placeholder: the construction seed only ever seeds the seat's rng,
      // and `playGame` re-seeds it — including before the FIRST match, so no
      // match is ever played on this number.
      seed: seat,
      oracle: wiring,
      calibrate: calibrateOn && seat === 0 ? opts.calibrate!.record : undefined,
      handSink: handCalibOn && seat === 0 ? opts.handCalib!.record : undefined,
      foldSink: foldCalibOn && seat === 0 ? opts.foldCalib!.record : undefined,
      foldEps: foldCalibOn && seat === 0 ? opts.foldEps : undefined,
      evSink: evCalibOn && seat === 0 ? opts.evCalib!.record : undefined,
    })
  );
  // Record ONLY neural seats: ppo.py recomputes behavior logp from --init, so a
  // heuristic seat's "d" lines would be treated as samples from the neural
  // policy and silently poison every importance ratio. Heuristic seats still
  // play (and appear in "r"/"m" lines); they just never emit decisions.
  // `recordAll` overrides for BC teacher datasets, whose consumer (bc.py) never
  // computes a ratio.
  const policies = built.map((b, seat) =>
    writer && (opts.recordAll || table[seat].kind === "n")
      ? new RecordingPolicy(b.policy, writer, (sq) => encodeOracle(ref.t!, sq as Seat))
      : b.policy
  );
  return {
    seats: kinds,
    table,
    policies,
    built,
    writer,
    ref,
    tableRef: writer || oracleSeats || curriculumOn || calibrateOn ? ref : undefined,
    calibrate: calibrateOn ? opts.calibrate : undefined,
    handCalib: handCalibOn ? opts.handCalib : undefined,
    foldCalib: foldCalibOn ? opts.foldCalib : undefined,
    evCalib: evCalibOn ? opts.evCalib : undefined,
  };
}

/**
 * One match on `seed`, with every seat put back into the state a seat built for
 * this match alone would have been in. The per-seat seeds are `seed*4+seat`,
 * which is what makes two arms on one seed roll the same policy dice.
 */
export function playGame(arm: Arm, seed: number): MatchResult {
  arm.calibrate?.beginGame(seed);
  arm.handCalib?.beginGame(seed);
  arm.foldCalib?.beginGame(seed);
  arm.evCalib?.beginGame(seed);
  for (const seat of SEATS) arm.built[seat].reset(seed * 4 + seat);
  // Without the hooks the ledger is always empty and the stats line would
  // report "違反 0件" no matter what actually happened.
  const hooks = makeDojoHooks(DOJO_HEADLESS);
  // M11's labels are attached at the end of the 局, and `onRoundEnd` is the one
  // seam that sees it — so the recorder is COMPOSED onto the dojo's hook rather
  // than replacing it, and strictly after it: the ledger entries the referee
  // adds there are part of the outcome the samples are labelled against.
  const handCalib = arm.handCalib;
  // M13's labels arrive at the same seam and are attached in the same order:
  // dojo referee first (its entries are part of the outcome), then the 手牌価値
  // recorder, then the fold recorder. Both recorders read a finished Table and
  // write nothing the other can see, so the order is a convention, not a
  // dependency — but it is a FIXED convention, so two lanes of one run
  // interleave the same way every time.
  const foldCalib = arm.foldCalib;
  const evCalib = arm.evCalib;
  const r = runMatchSync(arm.policies, {
    seed,
    cfg: JANKI,
    dojo: DOJO_HEADLESS,
    scorer,
    tableRef: arm.tableRef,
    ...hooks,
    ...(handCalib || foldCalib || evCalib
      ? {
        onRoundEnd: (t: Table, outcome: RoundOutcome) => {
          hooks.onRoundEnd(t, outcome);
          handCalib?.endRound(t, outcome);
          foldCalib?.endRound(t, outcome);
          evCalib?.endRound(t, outcome);
        },
      }
      : {}),
  });
  // Round and match lines close the match out: a policy never sees a result, so
  // only the driver can write them.
  if (arm.writer) writeMatchEnd(arm.writer, r, JANKI);
  return r;
}

export function closeArm(arm: Arm): void {
  arm.writer?.close();
  for (const b of arm.built) b.close();
}

/** What a headless run reports back, however many threads played it. */
export interface RunReport {
  /** One entry per game, ALWAYS in game order (seed + 0, seed + 1, …). */
  results: MatchResult[];
  /** Wall clock of the run, seconds × 1000. Not part of any byte-identity claim. */
  ms: number;
  traj: LineCounts | null;
}

export function headless(
  games: number,
  seed: number,
  seats: string | TableSpec = "hhhh",
  opts: HeadlessOptions = {},
): RunReport {
  const results: MatchResult[] = [];
  const arm = openArm(seats, opts);
  const t0 = performance.now();
  try {
    for (let g = 0; g < games; g++) results.push(playGame(arm, seed + g));
    return { results, ms: performance.now() - t0, traj: arm.writer?.stats() ?? null };
  } finally {
    closeArm(arm);
  }
}

// ---------------------------------------------------------------------------
// --jobs: the same run, over N Deno Workers
// ---------------------------------------------------------------------------
//
// SHARDING is round-robin: game i (seed + i) goes to worker `i % jobs`. Contiguous
// blocks would be simpler, but they make the in-order emission below buffer a
// whole block — worker 0's last game gates every game worker 1 ever played — and
// they shard badly when hanchan lengths differ. Round-robin keeps the workers in
// lockstep, so the reorder window stays a few games wide.
//
// IDENTITY. A worker plays with its OWN arm, and a game is a pure function of
// its seed (`playGame` re-seeds every seat to construction-equivalent state), so
// which thread played game i cannot be observed in its result. What COULD be
// observed is the order things are written in, so nothing is written from a
// worker: each game's trajectory lines are buffered as text and shipped back
// with the MatchResult, and the main thread appends them in game order and
// exports/aggregates from `results[]`, which is filled by index. The only line
// of a `--jobs=N` run that differs from `--jobs=1` is the 所要 timing.
//
// THREAD SAFETY of the native halves (checked against the sources, 2026-08):
//   * native/rlnet.c — every byte of state lives in a malloc'd `rlnet_ctx`
//     reached through the caller's handle; no globals, no function statics. Each
//     worker builds its own "n" seats and therefore its own contexts, and
//     Accelerate/cblas is thread-safe. Safe.
//   * native/mjkernel.cc — says so itself ("NOT re-entrant and not thread-safe"):
//     `g_suitTab` / `g_honorTab` are lazily allocated and lazily MEMOISED without
//     a lock, and a dlopen'd image is shared by every worker in the process. Two
//     hazards, both handled or benign:
//       (a) the allocation. `if (!g_suitTab) calloc(…)` raced by two threads
//           leaks a 16MB table and can leave one thread filling the table the
//           other abandoned. Closed here rather than argued about: worker 0 warms
//           the kernel and only then reports "ready", and the remaining workers
//           are spawned after that — the tables exist before a second thread can
//           run.
//       (b) the memoisation, `tab[idx] = computeWord(…)`. `computeWord` is a pure
//           function of the index, so two threads racing on one slot store
//           IDENTICAL bytes, into an 8-byte-aligned uint64 (calloc is 16-byte
//           aligned, the offset is idx*8). On aarch64 such accesses are
//           single-copy atomic, so a reader sees either 0 (recompute, same
//           answer) or the finished word. Formally a data race, observably a
//           no-op — and crucially it cannot change an answer, so `--jobs` cannot
//           perturb the results it is claimed to reproduce.

/**
 * The cloneable half of `HeadlessOptions` — what a worker can be handed through
 * `postMessage`. `record` is replaced by a boolean (the path stays on the main
 * thread), `writer`/`calibrate` are live objects and never cross, and
 * `ktuneB`/`consumerB` belong to `pairedRun`, which does not shard.
 *
 * `ktune` and `ktuneOpp` are plain JSON and deliberately DO cross: a sharded
 * run must build the same four seats every worker would.
 */
export type ArmSpec = Omit<
  HeadlessOptions,
  | "record"
  | "writer"
  | "calibrate"
  | "handCalib"
  | "foldCalib"
  | "evCalib"
  | "ktuneB"
  | "consumerB"
  | "tableB"
>;

/** The one message a worker receives: its arm, and the games it owns. */
export interface ShardInit {
  /**
   * The RESOLVED table — the main thread runs `resolveTable` (or takes the
   * caller's explicit specs) exactly once, and every worker builds from the
   * same four specs. Plain JSON, so `postMessage` carries it verbatim.
   */
  table: TableSpec;
  opts: ArmSpec;
  /** This worker's games, in the order it plays them. `i` is the game INDEX. */
  games: Array<{ i: number; seed: number }>;
  /** Buffer trajectory lines and ship them back with each game. */
  record: boolean;
}

/** What a worker posts back. */
export type ShardOut =
  | { k: "ready" }
  | { k: "game"; i: number; result: MatchResult; traj: string; counts: LineCounts }
  | { k: "done" }
  | { k: "error"; message: string };

interface Shard {
  worker: Worker;
  /** Resolves once the arm is built (and, for worker 0, the kernel warmed). */
  ready: Promise<void>;
  done: Promise<void>;
}

function launchShard(
  url: string,
  init: ShardInit,
  onGame: (m: Extract<ShardOut, { k: "game" }>) => void,
): Shard {
  const worker = new Worker(url, { type: "module" });
  let readyOk!: () => void;
  let readyErr!: (e: unknown) => void;
  const ready = new Promise<void>((res, rej) => {
    readyOk = res;
    readyErr = rej;
  });
  // A shard that dies before it is ready must reject BOTH promises or a caller
  // awaiting `ready` would hang; marking `ready` handled here keeps the
  // rejection from surfacing as an unhandled one for the shards nobody awaits.
  ready.catch(() => {});
  let doneOk!: () => void;
  let doneErr!: (e: unknown) => void;
  const done = new Promise<void>((res, rej) => {
    doneOk = res;
    doneErr = rej;
  });
  const fail = (e: unknown) => {
    readyErr(e);
    doneErr(e);
  };
  worker.onmessage = (ev: MessageEvent<ShardOut>) => {
    const m = ev.data;
    if (m.k === "game") onGame(m);
    else if (m.k === "ready") readyOk();
    else if (m.k === "done") {
      readyOk();
      doneOk();
    } else fail(new Error(m.message));
  };
  worker.onerror = (ev) => {
    ev.preventDefault();
    fail(new Error(ev.message || "self-play worker failed"));
  };
  worker.postMessage(init);
  return { worker, ready, done };
}

/**
 * `headless`, played over `jobs` workers. Same arguments, same report, same
 * bytes on disk — see the sharding/identity note above.
 *
 * `opts.calibrate` is NOT accepted (the CLI refuses the combination): the
 * recorder is a live writer fed from inside a decision, and there is no
 * per-game seam to buffer it at. `opts.handCalib` is refused for the same
 * reason, and for one more: its samples are only labelled at `onRoundEnd`, so
 * even a per-game buffer would have to cross the worker boundary half-written.
 */
export async function headlessParallel(
  games: number,
  seed: number,
  seats: string | TableSpec,
  jobs: number,
  opts: HeadlessOptions = {},
): Promise<RunReport> {
  if (opts.calibrate) throw new Error("headlessParallel: --calibrate cannot be sharded");
  if (opts.handCalib) throw new Error("headlessParallel: --handcalib cannot be sharded");
  if (opts.evCalib) throw new Error("headlessParallel: --evcalib cannot be sharded");
  // Resolved HERE, once: every shard receives the same four specs, so a table
  // and its seats-string form shard identically by construction.
  const table = typeof seats === "string" ? resolveTable(seats, opts) : seats;
  const n = Math.max(1, Math.min(Math.floor(jobs), games));
  const {
    record,
    writer: _w,
    calibrate: _c,
    handCalib: _hc,
    evCalib: _ec,
    ktuneB: _kb,
    consumerB: _cb,
    ...spec
  } = opts;
  const results: MatchResult[] = new Array(games);
  // The trajectory file is opened HERE, once, and written in game order.
  const writer = record ? new TrajectoryWriter(record) : null;
  // Games whose lines have arrived but whose predecessors have not.
  const held = new Map<number, { traj: string; counts: LineCounts }>();
  let next = 0;
  const flush = () => {
    while (next < games) {
      const p = held.get(next);
      if (!p) break;
      held.delete(next);
      writer?.writeRaw(p.traj, p.counts);
      next++;
    }
  };
  const onGame = (m: Extract<ShardOut, { k: "game" }>) => {
    results[m.i] = m.result;
    if (writer) {
      held.set(m.i, { traj: m.traj, counts: m.counts });
      flush();
    }
  };

  const url = new URL("./selfplay_worker.ts", import.meta.url).href;
  const shards: Shard[] = [];
  const t0 = performance.now();
  try {
    for (let w = 0; w < n; w++) {
      const mine: Array<{ i: number; seed: number }> = [];
      for (let i = w; i < games; i += n) mine.push({ i, seed: seed + i });
      shards.push(launchShard(url, { table, opts: spec, games: mine, record: !!record }, onGame));
      // Worker 0 alone is allowed to race with nobody: it warms the native
      // kernel's lazy tables before anyone else exists. See hazard (a) above.
      if (w === 0) await shards[0].ready;
    }
    await Promise.all(shards.map((s) => s.done));
    if (next !== games && writer) {
      throw new Error(`--jobs: ${games - next} 半荘ぶんの軌跡が届きませんでした`);
    }
    return { results, ms: performance.now() - t0, traj: writer?.stats() ?? null };
  } finally {
    for (const s of shards) s.worker.terminate();
    writer?.close();
  }
}
