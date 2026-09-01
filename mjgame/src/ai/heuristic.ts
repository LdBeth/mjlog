// The baseline CPU: efficiency-first, with a push/fold gate driven by
// mjrender's danger assessor and a dojo-aware filter on which actions it will
// even consider.
//
// THE COMPLIANCE FILTER. `finalStandings` ranks one ledger entry below every
// clean seat regardless of score, so a 禁じ手 is not an expensive move — it is a
// losing one, and pricing it (which is all `dojoCost` can do) only decides how
// expensive the loss was. So when the referee's own predicates are reachable
// (`Observation.preview`, wired by every real driver) this policy asks them
// first and DROPS every action that would be ledgered, choosing by score only
// among what is left. `dojoCost` stays for the fallthrough: when literally every
// candidate is charged, something has to be picked, and the prices are how.
//
// The filter lives where a subclass cannot reach around it: the discard
// candidate set is narrowed inside `chooseDiscard` (private) before scoring, and
// the call/kan hooks (protected, and overridden by the C7 planner) only PROPOSE
// — `decide` vetoes what they return.
//
// The shape of the decision — score every legal action, take the argmax — is
// the same shape a learned policy will use, so replacing this with a network
// changes `score`, not the plumbing. Every magic number lives in one weights
// object so the thing can be tuned or ablated without surgery.
//
// It is deterministic by default: no Date.now, no Math.random. `epsilon` (off
// unless asked for) exists so self-play can generate varied trajectories.

import type { DangerLevel } from "mjrender/danger.ts";
import type { Meld, Tile } from "mjrender/model.ts";
import { rankOfType, suitOfType, tileType } from "mjrender/tiles.ts";
import { countsFromTiles, shanten, ukeireTypes } from "../kernel.ts";
import { isHonor, isYaochu } from "../tiles.ts";
import type { Observation } from "../observe.ts";
import type { ActionPreview } from "../penalty/preview.ts";
import { pickLesserEvil, violationPoints } from "../penalty/preview.ts";
import type { SyncPolicy } from "../policy.ts";
import type { ConsumerParams } from "./consumer.ts";
import { scoreDiscard as consumeEvidence } from "./consumer.ts";
import {
  chiitoiShanten,
  dyeEff,
  fieldSense,
  HONOR_SHARE,
  mergeSense,
  senseActive,
} from "./sense.ts";
import type { FieldSense, SenseWeights } from "./sense.ts";
import type { ContextEvidence, EvidenceHooks } from "./evidence.ts";
import { assembleCandidate, assembleContext } from "./evidence.ts";
import { genbutsuSets } from "./computed.ts";
import type { FoldFacts, FoldSample } from "./fold.ts";
import { decideFold, FOLD_INPUTS, foldVector } from "./fold.ts";
import type { EvCore } from "./ev.ts";
import { evEvalDiscard, evEvalRest } from "./ev.ts";
import {
  DBLS_LEN,
  INTS_LEN,
  O_BEST_FOLD,
  O_BEST_PUSH,
  O_DAMA,
  O_FOLDLINE,
  O_NODES,
  O_RIICHI,
  O_STRIDE,
  O_TOTAL,
  O_TRUNC,
  R_NODES,
  R_TRUNC,
} from "./evlayout.ts";
import type { EvFacts, EvHidden, EvWire } from "./evpack.ts";
import { evFactsFromObservation, packEvInputs } from "./evpack.ts";
import type { EvSample } from "./evcalib.ts";
import { DEFAULT_EV } from "./evparams.ts";
import type { Mlp } from "./mlp.ts";
import type { HandSample } from "./handcalib.ts";
import type { HandFacts, HandOutlook, HandWeights } from "./handvalue.ts";
import { DEFAULT_HAND, handOutlook } from "./handvalue.ts";
import type { RiichiFeatures, RiichiWeights } from "./riichi.ts";
import { decideRiichi } from "./riichi.ts";
import { doraTypesOf, publicUnseen, valueHonorsOf } from "./planner.ts";
import type { StandingsWeights } from "./standings.ts";
import { standingsScales } from "./standings.ts";
import type { Rng } from "../rng.ts";
import { sfc32 } from "../rng.ts";
import type { Action, Violation } from "../types.ts";

export interface HeuristicWeights {
  /** Per shanten step. Deliberately dominates every other efficiency term. */
  shanten: number;
  /** Per live ukeire tile (copies of a useful type not yet visible). */
  ukeire: number;
  /** Per distinct ukeire type — breadth of the wait, not just its count. */
  ukeireType: number;
  /** Per dora (including aka) retained in the concealed hand. */
  dora: number;
  /** Retaining a concealed value-honor pair: a yaku seed, and a safe holding. */
  yakuhaiPair: number;
  /** Per lone honor still held, scaled by how late it is. */
  isolatedHonor: number;
  /** Cost of a discard by assessed danger level. */
  danger: Record<DangerLevel, number>;
  /**
   * 第一打字牌切り and 不聴時ドラ切り, as costs rather than hard filters: a
   * ledger entry is a price, not a wall, and the engine will happily let the
   * CPU pay it if literally every alternative is worse. Both are set above a
   * shanten step so in practice they decide the discard.
   */
  firstHonor: number;
  notenDora: number;
  /**
   * 片和了り. Set above a shanten step on purpose: when a clean tenpai is
   * available this only steers between equals, but when every tenpai is split
   * and riichi cannot cure it, outweighing `shanten` is what makes the policy
   * break the shape and give the hand up — which for an open hand is the only
   * move left.
   */
  katagari: number;
  /**
   * 後付け: the discard would leave an OPEN hand tenpai on nothing that scores.
   * Weighted like `firstHonor` because it buys exactly that — a Tier A medium
   * ledger entry — and unlike 片和了り there is no riichi available to cure it.
   */
  yakulessTenpai: number;
  /** Any tedashi once 不聴時ドラ切り has been called on us. */
  tsumogiriLock: number;
  /** Efficiency is scaled by this while folding. */
  foldEfficiency: number;
  /** Danger is scaled by this while folding. */
  foldDanger: number;
  /**
   * The 持ち点8000未満 buffer (see `bufferScale`): from 南入 on, push is scaled
   * by `bufferTight` within one assumed deal-in of the 8000 line and by
   * `bufferLow` within two. (East is exempt — the rule is judged on the FINAL
   * scores, 2026-08-27 ruling, and an early stack has a whole hanchan to
   * recover in.) These exist to protect a LEDGER RULE of the home dojo; an
   * environment without that rule (the riichi.dev arena) sets both to 1 —
   * measured on ranked wire logs, the buffer was the dominant cause of folding
   * live tenpai against a single riichi at ordinary mid-game stacks.
   */
  bufferTight: number;
  bufferLow: number;
  /**
   * 生牌の役牌 surcharge: risk points (the `w.danger` currency) added to a
   * mid-game release of a value honor NOBODY has shown yet, while at least one
   * opponent is open. DEFAULT 0 — the term is off until a vector asks for it,
   * so the seat plays bit-for-bit its prior game.
   *
   * Why it is not already priced: the 計算 deal-in estimate needs a threat to
   * model, and one chi or pon sits below the assessor's activation, so a live
   * 白 released at 8巡目 against a single-meld hand costs a few percent on
   * paper. Across 35 arena games (2026-08-28) the bot's 巡目 ≥ 6 releases of a
   * 0-copies-visible value honor into a table with an open hand were ronned
   * 2/16 times at 満貫 or better — about 1,300 points per discard, against
   * ~216 for an ordinary middle tile. This is the flat charge for that gap, not
   * a model of it: a 役牌 nobody can spare is the tile an open hand is waiting
   * on, and the assessor cannot see the wait yet.
   */
  liveYakuhai: number;
  /**
   * 暗刻を崩して七対子に向かわない — the triplet guard. Nonzero ⇒ ON: a discard
   * that breaks a concealed triplet (3 → 2) is struck from the candidate set
   * when the shape it leaves rides the 七対子 line (chiitoi shanten strictly
   * below standard) at 1向聴 or worse AND some other discard keeps a better
   * standard shanten. DEFAULT 0 — off, the seat plays bit-for-bit its prior
   * game.
   *
   * Why the chooser needs it: `kernel.shanten` is the MIN over standard and
   * chiitoi, so to the score a triplet is a pair plus a spare — cutting the
   * third copy costs nothing on the min line while the standard line loses a
   * whole step. The sense's `chiitoiTax` does not reach it: that tax exempts
   * `sh < 2`, and five pairs (the shape the break leaves) is 1向聴. Arena wire
   * logs 2026-08-28: 6 of 12 triplet-breaking discards traded a standard step
   * for the pairs line (`5m5m5m 6m6m` cutting 5m at 4巡目). The exceptions are
   * the doctrine's own: breaking INTO 七対子聴牌 stays allowed (七対子単騎 is a
   * sanctioned wait), a break that keeps the best standard shanten is a
   * standard-form decision and untouched, a melded hand is exempt by
   * construction, and a FOLDING hand is exempt — its triplet may be the only
   * 現物 it holds. Never a price: a filter, so no fitted core or planner malus
   * can buy its way around it (the same reasoning as `compliantDiscards`).
   */
  keepTriplet: number;
  /**
   * 順位効用. Absent by DEFAULT, and absent means off: every scale the layer
   * produces is 1 and the policy is bit-for-bit the point-EV agent it has always
   * been. Present, it prices this seat's points by what they do to the FINAL
   * PLACEMENT — see `standings.ts` — and reaches the decision through exactly
   * two multipliers, on the push/fold gate and on the price of danger.
   */
  standings?: StandingsWeights;
}

export const DEFAULT_WEIGHTS: HeuristicWeights = {
  shanten: 1000,
  ukeire: 12,
  ukeireType: 4,
  dora: 60,
  yakuhaiPair: 40,
  isolatedHonor: 6,
  danger: { "安全": 0, "危険度低": 30, "危険度中": 90, "危険度高": 200 },
  firstHonor: 4000,
  notenDora: 2500,
  katagari: 1500,
  yakulessTenpai: 4000,
  tsumogiriLock: 2500,
  foldEfficiency: 0.05,
  foldDanger: 10,
  bufferTight: 0.35,
  bufferLow: 0.7,
  liveYakuhai: 0,
  keepTriplet: 0,
};

/**
 * A partial over the defaults, `danger` merged level-wise: spreading a partial
 * wholesale would let an override drop a level, and a missing level scores
 * every discard NaN — which silently degrades to "discard the first tile in
 * hand". The constructor and `scripts/freeze.ts` (which must dump EXACTLY what
 * a seat would play under) both resolve through here.
 */
export function mergeHeuristic(w?: Partial<HeuristicWeights>): HeuristicWeights {
  return {
    ...DEFAULT_WEIGHTS,
    ...w,
    danger: { ...DEFAULT_WEIGHTS.danger, ...w?.danger },
  };
}

export interface HeuristicOptions {
  weights?: Partial<HeuristicWeights>;
  /** 喰いタン. Only affects whether an open tanyao counts as a confirmed yaku. */
  kuitan?: boolean;
  /**
   * Obey the dojo 禁じ手. With a referee preview on the Observation this is
   * literal — every action the ledger would charge for is dropped from the
   * choice set by `penalty/preview.ts`, which runs the ledger's OWN predicates
   * rather than an imitation of them. Without one (a hand-built Observation, a
   * driver that passed no `DojoConfig`) it degrades to the older behaviour: the
   * rules the policy can see for itself, priced by `dojoCost`.
   */
  dojo?: boolean;
  /** Probability of taking a uniformly random legal action instead. */
  epsilon?: number;
  /**
   * M9. The learned consumer of the evidence vector (`consumer.ts`). ABSENT BY
   * DEFAULT, and absent means the hand-written arithmetic below runs unchanged,
   * bit for bit. Present, it replaces the discard score CORE and nothing else:
   * the compliance filter, the `dojoCost` fallthrough pricing and the riichi
   * decision are outside it on both paths, and `initFromWeights` makes the two
   * paths agree exactly.
   */
  consumer?: ConsumerParams;
  /**
   * M11. 手牌価値 — the own-hand value model of `handvalue.ts`, already merged.
   * ABSENT BY DEFAULT, and absent means every number below is the one this class
   * has always produced, bit for bit: the push table, the dealer nudge, the
   * late-junme zeroing and the linear dora term only step aside for a `hand`
   * block. Present, it replaces exactly two things — what the push/fold gate
   * calls "how much this hand is worth carrying forward", and the value half of
   * the discard score — with `pwin × points` computed by ONE function that the
   * offline fit also calls.
   */
  hand?: HandWeights;
  /**
   * M12. The riichi head (`riichi.ts`) — the learned declare-vs-damaten
   * decision. ABSENT BY DEFAULT, and absent means `wantRiichi` behaves bit for
   * bit as it always has: the four gates, then declare. Present, it replaces
   * exactly one thing — the unconditional "declare" inside the region the
   * gates admit — with `decideRiichi` over the features of the post-discard
   * shape. The gates themselves, `riichiBanned` and the `mustCure` override
   * stay outside it, always.
   */
  riichi?: RiichiWeights;
  /**
   * 色読み (`sense.ts`) — the 感性 field sense: トイツ場 and 染め場. ABSENT BY
   * DEFAULT, and absent (or all-zero) means no field fact is even computed and
   * every decision is the one this class has always made, bit for bit. Present
   * with live weights, it adds exactly three terms: suit-heat risk on the
   * defence side, field pressure at the fold gate, and the chiitoi-line tax on
   * the discard score — each scaled by its own weight, nothing replaced.
   */
  sense?: Partial<SenseWeights>;
  /**
   * M13. The fold head (`fold.ts`) — the learned push/fold decision. ABSENT BY
   * DEFAULT, and absent means `computeFold` ends in the comparison it always
   * ended in, bit for bit. Present, it replaces exactly one thing: the sign of
   * `margin` inside the region the two early-outs (declared riichi, zero
   * pressure) admit. `INIT_FOLD` reproduces that sign exactly, so a `{}` block
   * is the incumbent seat.
   *
   * A BUILT head and not a weight block, unlike `hand`/`riichi`/`sense`: it may
   * hold a native context, so exactly one owner may build it and exactly one
   * must free it. `harness.ts` builds it once per seat and closes it in the
   * seat's `close()`; this policy only reads it.
   */
  fold?: Mlp;
  /**
   * M13's calibration lane: flip the fold verdict with probability `eps` on a
   * stream of the policy's OWN, seeded from `seed` (see `foldRng`). Off (or
   * `eps` 0) draws nothing at all, so a lane recorded at ε=0 is bit-identical
   * to a run with no lane — which is what makes the recorder an observer.
   */
  foldExplore?: { eps: number };
  /**
   * M13's recorder, one sample per fold decision that reaches the head region
   * (`foldcalib.ts` labels them from the 局's own settlement).
   *
   * DELIBERATELY INDEPENDENT of `fold`, for `handSink`'s reason: the first lane
   * has to be played by the seat that ships, which — before the first fit — is
   * the seat with no `fold` block at all. The sink then records the INCUMBENT
   * gate's verdict, and the header says so (`head: "gate"`).
   */
  foldSink?: (rec: FoldSample) => void;
  /**
   * M11's recorder, one sample per turn decision (`handcalib.ts` labels them
   * from the outcome of the 局).
   *
   * DELIBERATELY INDEPENDENT of `hand`: the lane that fits the model has to be
   * played by the seat that ships, which — before the first fit — is the seat
   * with no `hand` block at all. So the sink watches a policy it does not move,
   * and the samples it emits are evaluated under `DEFAULT_HAND` when no weights
   * were given, which is exactly the header a writer records.
   */
  handSink?: (rec: HandSample) => void;
  /**
   * M15. The expected-value core (`ev.ts` + `native/mjev.cc`), ALREADY BUILT —
   * like `fold`, and for the same reason twice over: it holds a native context
   * and a set of reused wire buffers, so exactly one owner may build it and
   * exactly one must free it. `harness.ts` builds it once per seat and closes
   * it in the seat's `close()`; this policy only reads it.
   *
   * ABSENT BY DEFAULT, and absent means no FFI is touched and every decision is
   * the one this class has always made, bit for bit. The DECISION INTEGRATION
   * (unit B onward) lands separately — this field is the seam it will read.
   */
  ev?: EvCore;
  /**
   * M15b's recorder: one sample per turn decision, carrying the FULL packed
   * wire of the resting 13-tile shape the seat chose (`evcalib.ts` labels them
   * from the outcome of the 局).
   *
   * DELIBERATELY INDEPENDENT of `ev`, which is `handSink`'s reason raised to
   * the level of a refusal: the lane that fits the population scalars has to be
   * played by the seat that ships — the PLAIN champion, with no core at all —
   * because a lane recorded under the DP is censored by the DP's own folds.
   * The CLI refuses `--evcalib` beside an `ev` block for exactly that, and this
   * sink accordingly builds its wire through the same hooks an `ev` seat would
   * use WITHOUT needing a core to do it.
   */
  evSink?: (rec: EvSample) => void;
}

/**
 * What `outlookOf`'s caller already knows about the shape it is asking about.
 *
 * Exported because the hook is protected and a subclass cannot name its
 * parameter type otherwise.
 */
export interface OutlookOpts {
  /**
   * The discard that produced the 13-tile shape. It is the per-decision cache
   * key, and the way to `discardInfo` — which is where 後付け (a tenpai nobody
   * can ron) is recorded. Absent at the fold gate of a claim decision, where the
   * resting shape is simply the hand.
   */
  tile?: Tile;
  /** The live ukeire count and its breadth, when the caller has already paid for them. */
  ukeire?: { live: number; types: number };
}

/**
 * Everything derived once per decision and shared by the per-action scorers.
 *
 * Exported because the subclass hooks below (`riskOf`, `drawBonus`, `keepBonus`)
 * take it: a policy that reads hidden information overrides those, and cannot
 * name their parameter type otherwise.
 */
export interface Ctx {
  obs: Observation;
  open: number;
  closed: boolean;
  doraTypes: Set<number>;
  valueHonors: Set<number>;
  /**
   * Copies of each of the 34 types this seat cannot see (`publicUnseen`).
   *
   * THE ONLY LIVENESS ACCOUNT. Every "how many are left" question in the
   * decision reads this vector: the ukeire count, the 不聴時ドラ切り exception's
   * visible-copy test (`4 − unseen`), and the planner's availability model.
   * There used to be two — `Observation.ukeire[].live` for a type the resting
   * hand happened to accept and `4 − own copies` for everything else — which
   * priced the same tile differently depending on which candidate was asking,
   * and always upward, because the second one counts neither the rivers nor the
   * melds nor the indicators.
   */
  unseen: number[];
  folding: boolean;
  /** Riichi is on the table this turn — which puts a yaku on every wait. */
  canRiichi: boolean;
  eff: number;
  def: number;
}

/**
 * Everything the consumer path needs that is fixed for one decision: the bound
 * hooks and the per-decision context evidence, assembled once and shared by
 * every candidate. Null whenever no consumer is set, which is what makes the
 * old path free.
 */
interface EvidenceRun {
  hooks: EvidenceHooks;
  context: ContextEvidence;
}

/**
 * The per-decision memo of the four PURE per-observation quantities.
 *
 * All four are closed forms over one `Observation` and were computed two or
 * three times per decision — `shouldFold` from `context` and again from the C7
 * planner's `updatePlan`, `pressureOf`/`bufferScale`/`standingsOf` from inside
 * `shouldFold` and again from the evidence assembler. Keyed on the Observation's
 * IDENTITY rather than cleared by hand: `decide` receives one object and every
 * caller inside the decision is handed that same object, so a new decision
 * invalidates the memo by construction and nothing has to remember to.
 *
 * It memoizes the CALL, never the method: `pressureOf` and `bufferScale` are
 * hooks an augmented policy overrides, so the first call still dispatches
 * virtually and the memo holds whatever that policy answered.
 */
interface DecisionMemo {
  obs: Observation;
  pressure?: number;
  buffer?: number;
  standings?: { gain: number; risk: number };
  fold?: boolean;
  /** M11: `threatOf`, the per-seat half of `pressure`. */
  threat?: number[];
  /** 色読み: the field, sensed once per decision — never when the sense is off. */
  sense?: FieldSense;
  /** M11: the board facts every `HandFacts` of this decision is built from. */
  basis?: HandBasis;
  /** M11: one entry per resting shape, keyed by the discard that produced it. */
  outlooks?: Map<number, HandEntry>;
  /** M13: the fold head's 37 facts — built only when a head or a sink exists. */
  foldFacts?: FoldFacts;
  /** M15: the EV core's whole verdict on this decision — one native call. */
  ev?: EvResult;
  /**
   * M15 unit D: the PASS side of a call — what holding the current 13-tile
   * hand is worth (`mjev_eval_rest`). One per decision, because every call on
   * offer is compared against the same alternative.
   */
  evRest?: number;
  /**
   * M15 unit D: the ACCEPT side, keyed by the meld the call would make.
   *
   * A SEPARATE map rather than more fields on `ev`: `ev` is the verdict on THIS
   * hand, and a caller reading `memo.ev.bestPush` must never be handed the
   * value of a hypothetical post-call shape. The key is the meld's tile ids,
   * which name the call uniquely inside one decision.
   */
  evCalls?: Map<string, number>;
}

/**
 * M11. Everything a `HandFacts` needs that is a property of the BOARD rather
 * than of the candidate discard, assembled once per decision — the fold gate and
 * every discard candidate ask the same questions of it.
 *
 * Four of these fields shadow `Ctx`'s, and that is not an oversight: the gate is
 * one of the two callers and runs BEFORE a `Ctx` exists, since `Ctx.folding` is
 * precisely what the gate decides. Built lazily, so a policy with no `hand` and
 * no `handSink` never pays for a line of it.
 */
interface HandBasis {
  /** Melds INCLUDING ankan — what `shanten`/`ukeireTypes` count. */
  melds: number;
  /** Melds EXCLUDING ankan — what "open" means to the value model. */
  open: number;
  closed: boolean;
  doraTypes: Set<number>;
  valueHonors: Set<number>;
  unseen: number[];
  unseenTotal: number;
  /** Dora (aka included) sitting in our own melds: constant across candidates. */
  meldDora: number;
}

/** M11: one resting shape's facts, and the model's verdict on them. */
interface HandEntry {
  facts: HandFacts;
  out: HandOutlook;
}

/**
 * What the last `decide` weighed when it chose a discard. Purely a RECORD of
 * the scoring that happened anyway (no extra probes, nothing the choice reads
 * back), kept so a driver can show the seat's reasoning — the TUI's 助言 row
 * renders the advisor seat's trace. Null until a discard has been scored.
 */
export interface DiscardTrace {
  /** Defensive mode: the push/fold gate said to abandon the hand. */
  folding: boolean;
  /** Every candidate the filter let through, best first. */
  candidates: Array<{ tile: Tile; shanten: number; score: number }>;
  chosen: Tile;
  riichi: boolean;
  /** 片和了り cure: riichi was forced by the dojo, not chosen on merit. */
  mustCure: boolean;
  /**
   * M15. What `score` is denominated in. Absent (the incumbent path) it is the
   * hand-written score unit; `"points"` is the EV core's own currency.
   *
   * The TUI reads score DIFFERENCES between candidates and never the absolute
   * number, so both units render — but a reader comparing a logged trace across
   * two seats has to know which scale it is looking at, and a factor of
   * `pointsPerScore` between them is not something to infer from magnitudes.
   */
  units?: "points";
  /** M15: the fold line the chosen discard was compared against, in points. */
  foldValue?: number;
  /** M15: what declaring on the chosen discard is worth, in points (unit C reads it). */
  riichiValue?: number;
}

/**
 * M15. One evaluation of the expected-value core, copied out of the wire buffer.
 *
 * COPIED, deliberately: `EvCore.out` is reused by the next decision (it is the
 * only per-decision allocation the FFI path makes), so a memo holding the live
 * buffer would answer the NEXT board's numbers to this board's questions. Four
 * 34-vectors per decision is the price of a memo that cannot lie.
 */
export interface EvResult {
  /** Per type: the price of letting it go, in points. −Infinity outside `candMask`. */
  total: Float64Array;
  /** Per type: the value of the resulting shape played damaten. */
  dama: Float64Array;
  /** Per type: the value of declaring riichi on it (unit C's input). */
  riichi: Float64Array;
  /** Per type: what abandoning the hand after this discard is worth. */
  foldLine: Float64Array;
  /** The best push line and the best fold line over the priced candidates. */
  bestPush: number;
  bestFold: number;
  /** Search accounting (plan D5): value states visited, and whether the cap bit. */
  nodes: number;
  truncated: boolean;
}

/**
 * The fold lane's stream, derived from the seat's own seed.
 *
 * A SEPARATE stream, not a fork of `rng`: `Rng.fork` is the wall's device and
 * the flip must be independent of how many numbers `epsilon` consumed. The
 * derivation is `Math.imul(seed, 0x9E3779B1) + 13`, the golden-ratio odd
 * multiplier `rng.ts` itself uses (`imul`, so a seat seed of `seed*4+seat`
 * cannot silently lose bits to float rounding the way `*` would), offset by 13
 * for M13 so the fold stream and any later head's stream differ on every seed.
 */
function foldStream(seed: number): number {
  return (Math.imul(seed, 0x9e3779b1) + 13) >>> 0;
}

export class HeuristicPolicy implements SyncPolicy {
  /** See `DiscardTrace`. Overwritten by every decision that scores discards. */
  lastTrace: DiscardTrace | null = null;
  /**
   * INSTRUMENTATION, for `bench` and nothing else: how many decisions this seat
   * has been asked for, and what the EV core spent answering them.
   *
   * Deliberately NOT cleared by `reset` — a bench run wants the whole run's
   * total, and a per-match counter would report the last hanchan. Nothing in
   * any decision reads either object, so they cannot move a game: `RunReport`
   * keeps its byte-identity claims because the counters are not in it.
   */
  decisions = 0;
  readonly evStats = { calls: 0, nodes: 0, truncated: 0, maxNodes: 0 };
  readonly name: string;
  readonly sync = true;
  protected w: HeuristicWeights;
  private kuitan: boolean;
  private dojo: boolean;
  private epsilon: number;
  private rng: Rng;
  /** M9's learned consumer, or null for the hand-written score. */
  private consumer: ConsumerParams | null;
  /** M11's model, or null for the push table and the linear dora term. */
  private hand: HandWeights | null;
  /**
   * The weights the RECORDER evaluates under. Identical to `hand` when one was
   * given; `DEFAULT_HAND` when none was, so a lane recorded off an unmodified
   * seat still carries a well-defined prediction to fit against.
   */
  private handW: HandWeights;
  /** M12's head, or null for the unconditional declare. */
  private riichiHead: RiichiWeights | null;
  /** 色読み weights, and the guard that keeps a zero vector costing zero work. */
  private sw: SenseWeights;
  private senseOn: boolean;
  private handSink: ((rec: HandSample) => void) | null;
  /** M15b's recorder, and the one scratch wire it packs into (see `evSink`). */
  private evSink: ((rec: EvSample) => void) | null;
  private evWire: EvWire | null;
  /** M13's head, or null for the incumbent `margin < 0`. Owned by the builder. */
  private foldHead: Mlp | null;
  /**
   * M15's EV core, or null for the hand-written surrogate. Owned by the
   * builder (`harness.ts`), read by the subclass — hence `protected`.
   */
  protected readonly ev: EvCore | null;
  /** The head's input row, reused for the life of the seat (see `decideFold`). */
  private foldX: Float32Array | null;
  /** ε of the fold-flip lane; 0 (the default) draws no random numbers at all. */
  private foldEps: number;
  /**
   * The lane's OWN random stream, and the reason it is its own: the flip must
   * not consume from `this.rng`, whose draws are `epsilon`'s and whose position
   * is part of what makes a seed reproduce a run. Made only when ε > 0, so a
   * seat with no lane holds null and cannot draw by accident.
   *
   * Seeded from the seat's seed by `foldStream` — a fixed derivation, so the
   * flip schedule is a function of the match seed like everything else.
   */
  private foldRng: Rng | null = null;
  private foldSink: ((rec: FoldSample) => void) | null;
  private memo: DecisionMemo | null = null;
  /**
   * The `tenpaiHeld` counter (see `riichiFeatures`): how long the current
   * kyoku's tenpai has sat on an unimproving wait. Per-kyoku by key, advanced
   * at most once per 巡, cleared by `reset` — the only cross-decision state
   * the base policy carries, and it feeds a FEATURE, never a gate: with no
   * riichi head in the ktune vector it is dead weight and the policy is
   * bit-for-bit its stateless self.
   */
  private riichiHold: { key: string; junme: number; mass: number; held: number } | null = null;

  constructor(name: string, seed: number, opts: HeuristicOptions = {}) {
    this.name = name;
    this.w = mergeHeuristic(opts.weights);
    this.kuitan = opts.kuitan ?? true;
    this.dojo = opts.dojo ?? true;
    this.epsilon = opts.epsilon ?? 0;
    this.consumer = opts.consumer ?? null;
    this.hand = opts.hand ?? null;
    this.handW = opts.hand ?? DEFAULT_HAND;
    this.riichiHead = opts.riichi ?? null;
    this.sw = mergeSense(opts.sense);
    this.senseOn = senseActive(this.sw);
    this.handSink = opts.handSink ?? null;
    this.evSink = opts.evSink ?? null;
    // One pair of buffers for the life of the seat, exactly as `EvCore` keeps
    // its own: the sink hands them to the writer, which COPIES before returning.
    this.evWire = opts.evSink
      ? { ints: new Int32Array(INTS_LEN), dbls: new Float64Array(DBLS_LEN) }
      : null;
    this.foldHead = opts.fold ?? null;
    this.ev = opts.ev ?? null;
    this.foldX = this.foldHead || opts.foldSink ? new Float32Array(FOLD_INPUTS) : null;
    this.foldEps = opts.foldExplore?.eps ?? 0;
    this.foldSink = opts.foldSink ?? null;
    this.rng = sfc32(seed);
    if (this.foldEps > 0) this.foldRng = sfc32(foldStream(seed));
  }

  /**
   * The evidence assembler's window onto this instance's own methods. Built as
   * closures rather than passed as a `this` reference so that the hook set is a
   * NAMED, minimal contract — and, more to the point, so that the calls dispatch
   * virtually: an `AugmentedHeuristic`'s `riskOf`/`pressureOf` overrides fill
   * the evidence automatically, with no cooperation from `evidence.ts`.
   */
  private evidenceHooks(): EvidenceHooks {
    return {
      handWithout: (ctx, tile) => this.handWithout(ctx, tile),
      riskOf: (ctx, tile) => this.riskOf(ctx, tile),
      drawBonus: (ctx, tile) => this.drawBonus(ctx, tile),
      keepBonus: (ctx, tile) => this.keepBonus(ctx, tile),
      pressureOf: (obs) => this.pressure(obs),
      bufferScale: (obs) => this.buffer(obs),
      standings: (obs) => this.standingsOf(obs),
    };
  }

  reset(seed: number): void {
    this.rng = sfc32(seed);
    // The lane's stream restarts with the match, exactly as `rng` does — and
    // stays null while ε is 0, so a headless seat cannot be handed a stream it
    // would never draw from.
    if (this.foldEps > 0) this.foldRng = sfc32(foldStream(seed));
    this.memo = null;
    this.riichiHold = null;
  }

  /** The memo slot for this decision, freshly emptied when the board moved. */
  private cache(obs: Observation): DecisionMemo {
    const m = this.memo;
    if (m && m.obs === obs) return m;
    return (this.memo = { obs });
  }

  /** `pressureOf`, once per decision. */
  private pressure(obs: Observation): number {
    const m = this.cache(obs);
    return m.pressure ??= this.pressureOf(obs);
  }

  /** `bufferScale`, once per decision. */
  private buffer(obs: Observation): number {
    const m = this.cache(obs);
    return m.buffer ??= this.bufferScale(obs);
  }

  /** `threatOf`, once per decision. */
  private threat(obs: Observation): number[] {
    const m = this.cache(obs);
    return m.threat ??= this.threatOf(obs);
  }

  /** 色読み, once per decision — and never at all while the sense is off. */
  private senseOf(obs: Observation): FieldSense {
    const m = this.cache(obs);
    return m.sense ??= fieldSense(obs);
  }

  // ------------------------------------------------------------- 色読み (感性)

  /**
   * The 染め場 surcharge on one tile, in `w.danger` currency. This is the one
   * defensive price that fires where the assessor does NOT look: a quiet table
   * (no riichi, no furo threat) with a silent flush growing on it prices every
   * tile 0 through the rule ladder, and that quiet is exactly where the
   * 2026-08-27 arena batch fed its mangan cluster. Honors take the hottest
   * suit's heat — the dyer is holding them. Types the dye's own source already
   * discarded stay free (`FieldSense.safe`): the field evidence proves them out.
   *
   * Protected so `AugmentedHeuristic.riskOf` can ADD it around its estimate —
   * the estimate models assessed threats, and this term models a threat the
   * assessor has no entry for, so the two compose instead of competing.
   */
  protected senseRisk(ctx: Ctx, tile: Tile): number {
    if (!this.senseOn || this.sw.someRisk === 0) return 0;
    const f = this.senseOf(ctx.obs);
    if (f.hot === 0) return 0;
    const ty = tileType(tile);
    const s = ty < 9 ? 0 : ty < 18 ? 1 : ty < 27 ? 2 : 3;
    if (s === 3) {
      // Hot honors: free only when the hottest suit's source let this type go.
      const hotSuit = f.someba.indexOf(f.hot);
      if (f.safe[hotSuit].has(ty)) return 0;
      // シャンポン不成立 ⇒ 無価格. An honor deals in as シャンポン or 単騎 only,
      // and a 染め手 is built on the PAIR: with fewer than two copies left
      // outside our own hand nobody can hold one, and the bare 単騎 that
      // remains is not what the field evidence is arguing for. `ctx.unseen`
      // (`publicUnseen`) is exactly "not in a river, meld or indicator, and
      // not in our hand" — the count this test wants.
      if (ctx.unseen[ty] < 2) return 0;
      // 字牌は本命ではない — see `HONOR_SHARE`. Pricing the honor at the full
      // suit heat made the agent hoard honors and shed live middle tiles.
      return this.sw.someRisk * HONOR_SHARE * dyeEff(f.hot);
    }
    if (f.safe[s].has(ty)) return 0;
    return this.sw.someRisk * dyeEff(f.someba[s]);
  }

  /**
   * 生牌の役牌: the flat surcharge on letting an untouched value honor go, in
   * the same `w.danger` currency as the ladder. See `HeuristicWeights.liveYakuhai`
   * for the arena measurement behind it; zero by default, so this is one
   * comparison on the ordinary path.
   *
   * Every clause has to hold. The tile is an honor, and a VALUE honor to
   * SOMEONE — the round wind, any dragon, or any OPPONENT's seat wind (our own
   * seat wind is worthless to the three seats that could ron us, unless it is
   * also the round wind). No copy is public, so the hand that wants it can
   * still be holding a pair of it and waiting シャンポン. It is 巡目 ≥ 6, so a
   * held 役牌 is being held on purpose rather than waiting to be cut. And
   * somebody has actually called, which is what makes the 役牌 a plausible
   * wait rather than a lone honor in four closed hands.
   */
  protected liveYakuhaiRisk(ctx: Ctx, tile: Tile): number {
    const w = this.w.liveYakuhai;
    if (w === 0) return 0;
    const ty = tileType(tile);
    if (ty < 27) return 0;
    const obs = ctx.obs;
    if (obs.junme < 6) return 0;
    // Only where nobody is looking. Under a declared riichi or an activated
    // furo threat the assessor already prices a live 役牌 (its honor rule is
    // 危険度高 at ≤1 copy public); stacking this on top made the arena replay
    // swap a 中-rated honor for a 高-rated number tile INTO the riichi. The
    // surcharge exists for the 1-meld quiet table the assessor never enters.
    if (this.dangerLevelOf(ctx, tile) !== undefined) return 0;

    // 三元牌はいつでも役牌; 場風も同じ; 自風は他家のものだけ数える。
    let value = ty >= 31 || ty === obs.roundWind;
    if (!value) {
      for (let o = 1; o < 4 && !value; o++) {
        if (ty === 27 + ((obs.seatWind - 27 + o) % 4)) value = true;
      }
    }
    if (!value) return 0;

    // 生牌 test on PUBLIC copies only. `4 − unseen` counts rivers, melds,
    // indicators AND our own hand (see `Ctx.unseen`), so our own copies come
    // back out: holding a pair of 白 does not make the 白 any less live to the
    // seat that would ron it.
    let own = 0;
    for (const t of obs.hand) if (tileType(t) === ty) own++;
    if (4 - ctx.unseen[ty] - own !== 0) return 0;

    // Open melds only — an ankan leaves the hand closed (same reading as
    // `furoThreats`); the evidence was "somebody has called".
    for (let o = 1; o < 4; o++) if (obs.melds[o].some((m) => m.kind !== "ankan")) return w;
    return 0;
  }

  /**
   * The 感性 surcharges, together — everything ADDED to a risk reading rather
   * than computed from the assessed threats. Both terms price hands the
   * assessor holds no entry for, so they compose with any estimate instead of
   * competing with it, and both are exactly 0 unless a vector armed them.
   */
  protected surcharge(ctx: Ctx, tile: Tile): number {
    return this.senseRisk(ctx, tile) + this.liveYakuhaiRisk(ctx, tile);
  }

  /** The 染め場 term of the fold gate's pressure: a fully dyed field, weight 1 ⇒ one riichi. */
  protected sensePressure(obs: Observation): number {
    if (!this.senseOn || this.sw.somePressure === 0) return 0;
    return this.sw.somePressure * dyeEff(this.senseOf(obs).hot);
  }

  /**
   * The chiitoi-line tax on one discard candidate, in score units.
   *
   * `kernel.shanten` is the MIN across standard/chiitoi, so four early pairs
   * silently flip the whole discard chooser onto the pairs line — commitment as
   * an artifact of taking the min, with no judgment about whether the field
   * pairs at all. The tax makes that flip cost something outside a トイツ場:
   * `chiitoiTax × (standard − min) × (1 − toitsuba)`, only while the kept shape
   * actually rides the chiitoi line (its chiitoi shanten IS the min, strictly
   * below standard) and only at 2向聴 or worse — a hand at six pairs has
   * completed its commitment, and taxing it out of tenpai would be absurd.
   * Melded hands are exempt by construction (no meld shape can reach 七対子;
   * the kernel already prices them standard-only).
   *
   * Subtracted in BOTH `scoreDiscard` paths beside `dojoCost`: like the dojo
   * price it is a judgment no fitted core is allowed to move.
   */
  protected senseLineTax(ctx: Ctx, tile: Tile, sh: number): number {
    if (!this.senseOn || this.sw.chiitoiTax === 0) return 0;
    if (sh < 2 || ctx.obs.melds[0].length > 0) return 0;
    const counts = countsFromTiles(this.handWithout(ctx, tile));
    if (chiitoiShanten(counts) !== sh) return 0;
    const std = shanten(counts, 0, false);
    if (std <= sh) return 0;
    return this.sw.chiitoiTax * (std - sh) * (1 - this.senseOf(ctx.obs).toitsuba);
  }

  decide(obs: Observation): Action {
    this.decisions++;
    const { legal } = obs;
    if (legal.length === 1) return legal[0];
    if (this.epsilon > 0 && this.rng.float() < this.epsilon) {
      return legal[this.rng.int(legal.length)];
    }

    // A win is always taken. 片和了り and the 8000点未満 rule could argue
    // otherwise; neither is worth modelling before the hand values are.
    const win = legal.find((a) => a.t === "tsumo") ?? legal.find((a) => a.t === "ron");
    if (win) return win;

    const ctx = this.context(obs);

    // The veto sits HERE, not inside the hooks: `chooseKan` and `chooseCall` are
    // protected and a subclass (the C7 planner) overrides them, so a filter
    // applied inside them would be one the plan could talk its way out of.
    const kan = this.chooseKan(ctx, legal);
    if (kan && this.compliant(ctx, kan)) return kan;

    // 立直後カン見送り is the one rule that fires on an OMISSION: while in
    // riichi, passing up a kan that leaves the wait alone is itself the foul.
    const forced = this.mandatoryKan(ctx, legal);
    if (forced) return forced;

    const call = this.chooseCall(ctx, legal);
    if (call && this.compliant(ctx, call)) return call;

    const discard = this.chooseDiscard(ctx, legal);
    if (discard) return discard;

    return legal.find((a) => a.t === "pass") ?? legal[0];
  }

  // ------------------------------------------------------- compliance filter

  /** The referee's hypothetical judgement, when the driver wired one up. */
  private referee(ctx: Ctx): ActionPreview | undefined {
    return this.dojo ? ctx.obs.preview : undefined;
  }

  /** Would the ledger stay silent on this call/kan? True when it cannot be asked. */
  private compliant(ctx: Ctx, a: Action): boolean {
    const pv = this.referee(ctx);
    if (!pv) return true;
    if (a.t === "pon" || a.t === "chi") return pv.call(a).length === 0;
    if (a.t === "daiminkan") {
      return pv.call(a).length === 0 && pv.kan(a, ctx.obs.drawn).length === 0;
    }
    if (a.t === "ankan" || a.t === "kakan") return pv.kan(a, ctx.obs.drawn).length === 0;
    return true;
  }

  /**
   * The kan the dojo requires. Declining is charged (立直後カン見送り), so the
   * only question is whether accepting is charged too — and if both are, the
   * cheaper option wins with the tie going to declining, which is `pickLesserEvil`
   * called with the decline first.
   */
  private mandatoryKan(ctx: Ctx, legal: Action[]): Action | null {
    const pv = this.referee(ctx);
    if (!pv || !ctx.obs.riichi[0] || ctx.obs.drawn === null) return null;
    const skip = pv.skipKan(ctx.obs.drawn);
    if (skip.length === 0) return null;
    let best: { a: Action; vs: Violation[] } | null = null;
    for (const a of legal) {
      if (a.t !== "ankan") continue;
      const vs = pv.kan(a, ctx.obs.drawn);
      if (!best || violationPoints(vs) < violationPoints(best.vs)) best = { a, vs };
    }
    if (!best) return null;
    return pickLesserEvil(skip, best.vs) === "b" ? best.a : null;
  }

  // ---------------------------------------------------------------- context

  private context(obs: Observation): Ctx {
    const folding = this.shouldFold(obs);
    return {
      obs,
      open: obs.melds[0].length,
      closed: obs.melds[0].every((m) => m.kind === "ankan"),
      doraTypes: doraTypesOf(obs),
      valueHonors: valueHonorsOf(obs.roundWind, obs.seatWind),
      unseen: publicUnseen(obs),
      folding,
      canRiichi: obs.legal.some((a) => a.t === "discard" && a.riichi),
      eff: folding ? this.w.foldEfficiency : 1,
      // 順位効用 rides on `def` and not on `eff`: how rank-sensitive this seat's
      // points are is a statement about the PRICE OF DANGER, so a protected lead
      // pays more for every risky tile and folds earlier, while a hopeless
      // deficit pays less and pushes tiles a point-EV agent would never let go.
      // Off (the default) the factor is exactly 1 and this is the old value.
      def: (folding ? this.w.foldDanger : 1) * this.standingsOf(obs).risk,
    };
  }

  /**
   * 順位効用's two multipliers, or the neutral pair when the layer is off — which
   * it is unless a weights object asked for it, so this is `1 × 1` and changes
   * nothing by default.
   *
   * Memoized per decision (`DecisionMemo`): `Ctx.def`, the push/fold gate and
   * the evidence assembler all ask for it, and the closed form behind it is ten
   * `rankStats` evaluations — not free enough to run three times for one
   * discard.
   */
  protected standingsOf(obs: Observation): { gain: number; risk: number } {
    const w = this.w.standings;
    if (!w) return { gain: 1, risk: 1 };
    const m = this.cache(obs);
    return m.standings ??= standingsScales(obs, w);
  }

  /**
   * Push/fold. `push` is how much this hand is worth carrying forward, `pressure`
   * is how loud the table is. Folding is not all-or-nothing — it re-weights the
   * discard score rather than switching to a different algorithm.
   *
   * Protected, not because the base policy shares it, but because a subclass
   * that runs work BEFORE `decide` (the C7 planner, which must not re-plan while
   * the hand is being abandoned) has no other way to ask. Pure — and memoized on
   * the Observation for exactly that reason: the planner asks, and then `context`
   * asks again for the same board.
   */
  protected shouldFold(obs: Observation): boolean {
    const m = this.cache(obs);
    return m.fold ??= this.computeFold(obs);
  }

  /**
   * The push/fold gate, as the code has always computed it — rearranged so its
   * verdict is a NUMBER rather than a boolean (M13's D3).
   *
   * `margin = push·gain − 0.5·pressure·risk`, so `margin < 0` is the incumbent
   * `push·gain < 0.5·pressure·risk` (`a − b < 0 ⇔ a < b` for finite doubles,
   * including the tie), and the five parts it was built from come out with it
   * as features. NOTHING here is new arithmetic: everything below the signature
   * is the old body verbatim.
   *
   * `computeFold` does NOT call this on the plain path — see the early return
   * there, which keeps the incumbent expression literally intact.
   */
  private foldMargin(
    obs: Observation,
    pressure: number,
  ): { margin: number; push: number; gain: number; risk: number; buffer: number } {
    let push: number;
    if (this.hand) {
      const r = this.restingShape(obs);
      push = this.outlookOf(obs, r.rest, r.sh, { tile: r.tile }).ev / this.hand.pushScale;
    } else {
      push = obs.shanten <= 0 ? 1.0 : obs.shanten === 1 ? 0.45 : obs.shanten === 2 ? 0.15 : 0;
      push += 0.12 * obs.doraCount;
      if (obs.shanten >= 2 && obs.junme >= 10) push = 0;
      if (obs.seatWind === 27) push += 0.08;
    }
    const buffer = this.buffer(obs);
    push *= buffer;
    const st = this.standingsOf(obs);
    return {
      margin: push * st.gain - 0.5 * pressure * st.risk,
      push,
      gain: st.gain,
      risk: st.risk,
      buffer,
    };
  }

  /**
   * The 37 facts the head reads, once per decision.
   *
   * Built ONLY when a head or a sink exists — a seat with neither never touches
   * `genbutsuSets`, never asks `handEntry` for an outlook it would not otherwise
   * have wanted, and never allocates the row. Everything here is either already
   * memoised on the decision (`threat`, `sensePressure`, `handEntry`) or a
   * single pass over the hand.
   */
  private foldFeatures(
    obs: Observation,
    parts: { margin: number; push: number; gain: number; risk: number; buffer: number },
    pressure: number,
  ): FoldFacts {
    const m = this.cache(obs);
    if (m.foldFacts) return m.foldFacts;

    // The hand, through the SAME memoised entry the discard score and the M11
    // recorder use — under `handW`, i.e. `DEFAULT_HAND` when the seat carries
    // no `hand` block, exactly as the hand lane is recorded.
    const r = this.restingShape(obs);
    const entry = this.handEntry(obs, r.rest, r.sh, { tile: r.tile });
    const f = entry.facts;

    // Threat: the declared half is a public count, the assessed furo half comes
    // from the danger entries (the same seats `pressureOf` sums), and the
    // per-seat vector is the hook — which an augmented seat answers with
    // `tenpaiP` rather than with 1/0.5.
    let oppRiichi = 0;
    for (let s = 1; s < 4; s++) if (obs.riichi[s]) oppRiichi++;
    const furo = new Set<number>();
    for (const d of obs.danger.values()) {
      for (const detail of d.details) if (detail.kind === "furo") furo.add(detail.seat);
    }
    const th = this.threat(obs);
    const el = this.expLossOf(obs);

    // The scoreboard, in thousands and relative to the field.
    let top = -Infinity, bottom = Infinity;
    for (let s = 1; s < 4; s++) {
      if (obs.scores[s] > top) top = obs.scores[s];
      if (obs.scores[s] < bottom) bottom = obs.scores[s];
    }

    // Defensive capacity: what the hand could throw if it did fold. Distinct
    // TYPES, not tiles — three copies of one safe tile buy one turn of safety
    // each, and the assessor speaks in types.
    const seen = new Set<number>();
    let safeTypes = 0, lowTypes = 0, unassessed = 0;
    const gb = genbutsuSets(obs);
    const riichiRel: number[] = [];
    for (let s = 1; s < 4; s++) if (obs.riichi[s]) riichiRel.push(s);
    const gbCount = riichiRel.map(() => 0);
    let gbAll = 0;
    for (const t of obs.hand) {
      const ty = tileType(t);
      if (seen.has(ty)) continue;
      seen.add(ty);
      const lvl = obs.danger.get(ty)?.level;
      if (lvl === undefined) unassessed++;
      else if (lvl === "安全") safeTypes++;
      else if (lvl === "危険度低") lowTypes++;
      let all = riichiRel.length > 0;
      for (let i = 0; i < riichiRel.length; i++) {
        if (gb[riichiRel[i]].has(ty)) gbCount[i]++;
        else all = false;
      }
      if (all) gbAll++;
    }

    return m.foldFacts = {
      margin: parts.margin,
      push: parts.push,
      pressure,
      gain: parts.gain,
      risk: parts.risk,
      buffer: parts.buffer,
      shanten: f.shanten,
      ukeire: f.ukeire,
      ukeireTypes: f.ukeireTypes,
      dora: f.dora,
      junme: f.junme,
      turnsLeft: f.turnsLeft,
      dealer: f.dealer ? 1 : 0,
      open: f.open,
      oppRiichi,
      furoThreats: furo.size,
      threat0: th[0] ?? 0,
      threat1: th[1] ?? 0,
      threat2: th[2] ?? 0,
      expLoss0: (el[0] ?? 0) / 1000,
      expLoss1: (el[1] ?? 0) / 1000,
      expLoss2: (el[2] ?? 0) / 1000,
      pwin: entry.out.pwin,
      value: entry.out.value / 1000,
      ev: entry.out.ev / 1000,
      score: obs.scores[0] / 1000,
      leadTop: (obs.scores[0] - top) / 1000,
      leadBottom: (obs.scores[0] - bottom) / 1000,
      kyoku: obs.kyoku,
      honba: obs.honba,
      kyotaku: obs.kyotaku,
      safeTypes,
      lowTypes,
      unassessedTypes: unassessed,
      genbutsuAll: gbAll,
      genbutsuMin: gbCount.length === 0 ? 0 : Math.min(...gbCount),
      sensePressure: this.sensePressure(obs),
    };
  }

  /**
   * What a deal-in to each opponent is expected to cost, in points, relative
   * order. HOOK: the base policy reads no opponent model and answers zeros;
   * `AugmentedHeuristic` answers the 計算 reader's `expLoss`.
   *
   * A FEATURE-ONLY hook — nothing in the base policy's arithmetic reads it, and
   * it is called only when the fold head or its recorder is attached.
   */
  protected expLossOf(_obs: Observation): readonly number[] {
    return [0, 0, 0];
  }

  // ---------------------------------------------------------------- EV核 (M15)
  //
  // Three protected hooks and one memoised call. Everything the DP is told
  // about the OPPONENTS crosses through the hooks, so the doctrine stays in
  // TypeScript (plan §1: "packed by TS so the doctrine stays in TS") and the
  // C++ side never grows a second, invisible home for a dojo ruling.

  /**
   * Σ_i P(opponent i rons this tile NOW) — the probability half of what the DP
   * is told about the root discard.
   *
   * HOOK. The base policy reads no opponent model and answers 0, which is not
   * the claim "nothing can deal in": it is "this seat holds no estimate". The
   * COST half below still carries the rule ladder, so a base seat's root
   * discard is priced exactly the way `riskOf` prices it, only in points.
   * `AugmentedHeuristic` answers the 計算/M14 reader's `dealinP`.
   */
  protected dealinProbOf(_ctx: Ctx, _tile: Tile): number {
    return 0;
  }

  /**
   * What letting this tile go NOW costs, in POINTS: `riskOf`'s arithmetic with
   * the units changed (plan D4).
   *
   * WHY A SECOND HOOK RATHER THAN `riskOf × pointsPerScore`. `riskOf` answers
   * in score units, and the conversion is only exact because `augment.lambda`
   * is 1/`pointsPerScore` by construction — a coincidence of two defaults, not
   * a contract. Naming the points figure separately makes the DP's input a
   * quantity in its own right: a vector that retunes λ (which is a CONSUMPTION
   * scalar of the linear surrogate) then moves the surrogate without silently
   * rescaling the DP's deal-in cost.
   *
   * IDENTICAL TO `riskOf`, TERM FOR TERM (2026-08-30 review). This body used to
   * zero the ladder at 安全 on its own — "no estimate outranks a proof" — but
   * the base policy HOLDS no estimate: its whole price IS the ladder, and
   * `riskOf` charges `w.danger["安全"]` there like every other rung. With the
   * shipped vector's 0 the two agreed by coincidence; a tuned vector that
   * priced 安全 at all made the DP and the surrogate disagree about the same
   * tile in the same decision. The 安全 exit belongs where the estimate is —
   * `AugmentedHeuristic` keeps it, in both of its own two methods, because
   * there it means "no ESTIMATE outranks a proof".
   *
   * The 感性 surcharges are ADDED for the reason `riskOf` documents — they
   * price hands the assessor holds no entry for. 順位効用's `risk` multiplies
   * the whole thing, as `Ctx.def` does on the incumbent path.
   */
  protected dealinCostPts(ctx: Ctx, tile: Tile): number {
    const pps = this.pointsPerScore();
    // Spelled out rather than `this.riskOf(...) * pps`: an override of `riskOf`
    // alone must not silently redefine the DP's input (that is what the second
    // hook exists for), so the two bodies are the same EXPRESSION, not a call.
    const ladder = this.ruleRisk(this.dangerLevelOf(ctx, tile));
    return (ladder + this.surcharge(ctx, tile)) * pps * this.standingsOf(ctx.obs).risk;
  }

  /**
   * The DP's hidden-information overrides (plan D7), or null for the computed
   * seat — which is the base policy, always: 計算 is exact counting over public
   * facts, and a channel here is by definition not a public fact.
   *
   * HOOK. `AugmentedHeuristic` maps whatever its `Reads` carry; a future
   * learned hidden-information module returns soft distributions through the
   * same signature, and the engine never learns which producer filled them.
   */
  protected hiddenInfoOf(_obs: Observation): EvHidden | null {
    return null;
  }

  /**
   * Points per hand-written score unit (plan D4) — the exchange rate that lets
   * `dojoCost`, `senseLineTax` and the two bonus hooks stand beside a price in
   * points. `DEFAULT_EV`'s value when no core is loaded, so the hooks above are
   * callable (and testable) on a machine with no dylib.
   */
  protected pointsPerScore(): number {
    return this.ev?.params.pointsPerScore ?? DEFAULT_EV.pointsPerScore;
  }

  /**
   * The light `Ctx` the two pricing hooks are called through.
   *
   * It cannot be the real one. `context()` builds `Ctx` with `folding` already
   * decided, and the fold verdict is one of the things the DP DECIDES — asking
   * for a `Ctx` here would be a cycle. Everything `riskOf`/`surcharge` actually
   * read is present and identical (`obs`, `unseen`, `doraTypes`,
   * `valueHonors`); the two multipliers `eff`/`def` are 1 because the DP is
   * handed the un-multiplied facts and applies 順位効用 itself.
   */
  private evCtx(obs: Observation): Ctx {
    return {
      obs,
      open: obs.melds[0].length,
      closed: obs.melds[0].every((m) => m.kind === "ankan"),
      doraTypes: doraTypesOf(obs),
      valueHonors: valueHonorsOf(obs.roundWind, obs.seatWind),
      unseen: publicUnseen(obs),
      folding: false,
      canRiichi: obs.legal.some((a) => a.t === "discard" && a.riichi),
      eff: 1,
      def: 1,
    };
  }

  /**
   * Is there a 14-tile root to price? A turn decision (and a post-call discard)
   * rests on `3n + 2` concealed tiles; a claim decision — pon/chi/pass on
   * somebody else's tile — rests on `3n + 1` and has no discard to price.
   *
   * Unit B serves the DISCARD and the FOLD verdict only. The call comparisons
   * (`eval_rest` on the 13-tile shape vs `eval_discard` on the post-call one)
   * are unit D and are where this test stops being the whole story.
   *
   * THE EXACT COUNT, not `% 3 === 2` (2026-08-30 review). Both tests admit the
   * same well-formed roots — `14 − 3m` is `2 mod 3` for every m — but the
   * modulus also admits a malformed one, and the DP answers a malformed root
   * with a return code, which `evEvalDiscard` turns into a throw in the middle
   * of a match. The rule here is `mjev.cc`'s own (`parseEval`: `sum != (mode ==
   * 0 ? 14 : 13) - 3 * nMelds`), stated the same way on both sides.
   *
   * A KAN IS ONE SET AND NOTHING MORE. Its fourth tile is paid for by the
   * rinshan draw, so the concealed count after an 暗槓/加槓/大明槓 is what it
   * would be after a pon — 11 concealed beside one meld at a discard root, 10
   * at rest. (Measured over 40 hanchan: every kan root in the engine's own
   * stream satisfies `14 − 3m` exactly, none of them `14 − 3m − kans`.)
   */
  private evRoot(obs: Observation): boolean {
    return obs.hand.length === 14 - 3 * obs.melds[0].length;
  }

  /**
   * The same test for a 13-tile REST root (`mjev_eval_rest`, mode 1) — the PASS
   * side of a call comparison. `chooseCall` only ever runs on a claim decision,
   * where the shape is at rest by construction, so this is a guard rather than
   * a branch: a caller that gets it wrong gets "the DP has no opinion" instead
   * of a mid-match throw.
   */
  private evRestRoot(obs: Observation): boolean {
    return obs.hand.length === 13 - 3 * obs.melds[0].length;
  }

  /**
   * The core's verdict for THIS decision, or null when it is not this seat's
   * job (no core, `ev.discard` off, or no discard root to price).
   */
  private evDiscard(obs: Observation): EvResult | null {
    if (!this.ev?.params.discard || !this.evRoot(obs)) return null;
    return this.evOf(obs);
  }

  /**
   * One native evaluation, memoised on the Observation's identity like every
   * other per-decision fact — the fold gate and the discard chooser both ask,
   * and the DP is the most expensive thing in the decision by two orders of
   * magnitude.
   *
   * Protected so a test can read the numbers the seat acted on without
   * re-deriving them (which would be a second, differently-wrong witness).
   */
  /**
   * The POLICY half of one evaluation's facts — everything the Observation
   * cannot answer by counting (plan §1: "packed by TS so the doctrine stays in
   * TS"), for a root whose concealed part is `tiles`.
   *
   * Factored out of `evOf` for unit D: a call comparison prices a DIFFERENT
   * concealed shape (the hand minus the tiles the call spends) off the same
   * board, and the deal-in hooks must be asked about the tiles that shape
   * actually holds — not about the ones it just gave away.
   *
   * `hidden` IS A PARAMETER, not a call to the hook (2026-08-30 review). The
   * channels describe a FUTURE — the next own draw, the next kan-dora, the live
   * wall — and that future is a function of what the seat does next: after a
   * pon the tile `ownNextDraw` names is not what we draw, and after a kan the
   * rinshan tile comes first. Handing a hypothetical root the real root's
   * channels is not extra information, it is wrong information, so every
   * hypothetical passes `null` and prices its shape by counting.
   */
  private evPolicyFacts(obs: Observation, tiles: readonly Tile[], hidden: EvHidden | null) {
    // The root deal-in facts, per HELD TYPE. A representative tile of each type
    // is enough and is exact: every term of both hooks reads the tile through
    // `tileType`. The types not held are left at 0 — they are outside
    // `candMask` and the DP never prices them as a root discard.
    const ctx = this.evCtx(obs);
    const pIn = new Array<number>(34).fill(0);
    const costIn = new Array<number>(34).fill(0);
    const seen = new Set<number>();
    for (const t of tiles) {
      const ty = tileType(t);
      if (seen.has(ty)) continue;
      seen.add(ty);
      pIn[ty] = this.dealinProbOf(ctx, t);
      costIn[ty] = this.dealinCostPts(ctx, t);
    }
    const st = this.standingsOf(obs);
    return {
      pIn,
      costIn,
      tenpaiP: this.threat(obs),
      expLoss: this.expLossOf(obs),
      gain: st.gain,
      risk: st.risk,
      hidden,
    };
  }

  /**
   * M15b. The facts of the 13-tile REST left by `chosenTile` — `mjev_eval_rest`
   * mode 1, built through the very hooks an `ev` seat would use, and WITHOUT an
   * `EvCore`.
   *
   * That last clause is the whole reason it exists as its own method. The
   * calibration lane is recorded on a seat carrying NO `ev` block (the M11
   * lesson: never fit on a lane played by the block being fitted), so nothing
   * in this path may touch the FFI — but the wire it produces has to be the
   * wire the DP would have been handed, hook for hook, or the fit would be
   * calibrating a model against inputs the seat never presents. Everything it
   * calls (`evCtx`, `dealinProbOf`, `dealinCostPts`, `threat`, `expLossOf`,
   * `standingsOf`, `hiddenInfoOf`) is pure TypeScript, and `pointsPerScore`
   * falls back to `DEFAULT_EV`'s value when no core is loaded.
   *
   * Protected so the recorder's test can build one off a hand-made Observation.
   */
  protected evFactsForRest(obs: Observation, chosenTile: Tile): EvFacts {
    // `handWithout`'s removal, without a `Ctx`: the last copy of the id, so a
    // hand holding two of a type loses exactly one of them.
    const rest = [...obs.hand];
    const at = rest.lastIndexOf(chosenTile);
    if (at < 0) throw new Error(`evFactsForRest: ${chosenTile} は手牌にありません`);
    rest.splice(at, 1);
    return evFactsFromObservation(obs, {
      mode: 1,
      hand: countsFromTiles(rest),
      tiles: rest,
      ...this.evPolicyFacts(obs, rest, this.hiddenInfoOf(obs)),
    });
  }

  /** One evaluation's accounting, for `bench` — every native call, unit D's included. */
  private noteEv(nodes: number, truncated: boolean): void {
    this.evStats.calls++;
    this.evStats.nodes += nodes;
    if (truncated) this.evStats.truncated++;
    if (nodes > this.evStats.maxNodes) this.evStats.maxNodes = nodes;
  }

  protected evOf(obs: Observation): EvResult {
    const m = this.cache(obs);
    if (m.ev) return m.ev;
    const core = this.ev;
    if (!core) throw new Error("evOf: EV核が積まれていません (ev ブロックが無い席)");

    packEvInputs(
      core,
      evFactsFromObservation(obs, {
        mode: 0,
        ...this.evPolicyFacts(obs, obs.hand, this.hiddenInfoOf(obs)),
      }),
    );
    evEvalDiscard(core);

    const out = core.out;
    const total = new Float64Array(34);
    const dama = new Float64Array(34);
    const riichi = new Float64Array(34);
    const foldLine = new Float64Array(34);
    for (let ty = 0; ty < 34; ty++) {
      const at = ty * O_STRIDE;
      total[ty] = out[at + O_TOTAL];
      dama[ty] = out[at + O_DAMA];
      riichi[ty] = out[at + O_RIICHI];
      foldLine[ty] = out[at + O_FOLDLINE];
    }
    const nodes = out[O_NODES];
    const truncated = out[O_TRUNC] !== 0;
    this.noteEv(nodes, truncated);

    return m.ev = {
      total,
      dama,
      riichi,
      foldLine,
      bestPush: out[O_BEST_PUSH],
      bestFold: out[O_BEST_FOLD],
      nodes,
      truncated,
    };
  }

  /**
   * UNIT C. The core's verdict when the RIICHI question is the core's to answer,
   * or null when it is not (no core, `ev.riichi` off, or a decision with no
   * 14-tile root — riichi is only ever offered on one).
   *
   * A separate accessor from `evDiscard` on purpose: `ev.riichi` and
   * `ev.discard` are independent sub-switches, and a seat carrying
   * `{discard:false, riichi:true}` must get the DP's declaration verdict over
   * the incumbent's linear discard score. The evaluation itself is the SAME
   * one — `evOf` is memoised on the Observation — so a seat carrying both pays
   * for one native call, not two.
   */
  private evRiichiPrice(obs: Observation): EvResult | null {
    if (!this.ev?.params.riichi || !this.evRoot(obs)) return null;
    return this.evOf(obs);
  }

  /**
   * UNIT D. What the best push line of a hypothetical 14-tile root is worth —
   * the ACCEPT side of a call comparison, `mjev_eval_discard` on the post-call
   * hand (melds + 1, open).
   *
   * NOT memoised on `memo.ev`: that slot holds the verdict on the hand we
   * actually have, and a hypothetical must never be able to answer in its
   * place. Callers memoise by meld signature in `memo.evCalls`.
   */
  private evPushOf(obs: Observation, rest: readonly Tile[], melds: readonly Meld[]): number {
    const core = this.ev;
    if (!core) throw new Error("evPushOf: EV核が積まれていません");
    packEvInputs(
      core,
      evFactsFromObservation(obs, {
        mode: 0,
        hand: countsFromTiles([...rest]),
        tiles: rest,
        melds,
        // A POST-CALL root: `hiddenInfoOf`'s channels are about the hand we
        // have, and taking the call moves the draw order out from under them.
        ...this.evPolicyFacts(obs, rest, null),
      }),
    );
    evEvalDiscard(core);
    this.noteEv(core.out[O_NODES], core.out[O_TRUNC] !== 0);
    return core.out[O_BEST_PUSH];
  }

  /**
   * UNIT D. What HOLDING a 13-tile rest is worth (`mjev_eval_rest`) — the PASS
   * side of a call, and the post-暗槓 root (`kanDoraOn`, which is what makes a
   * kan worth anything at all).
   *
   * Returns NaN when the core refuses the root; every caller treats a
   * non-finite answer as "the DP has no opinion" and keeps the incumbent one.
   */
  private evHoldOf(
    obs: Observation,
    rest: readonly Tile[],
    melds: readonly Meld[],
    kanDoraOn = false,
    hidden: EvHidden | null = null,
  ): number {
    const core = this.ev;
    if (!core) throw new Error("evHoldOf: EV核が積まれていません");
    packEvInputs(
      core,
      evFactsFromObservation(obs, {
        mode: 1,
        hand: countsFromTiles([...rest]),
        tiles: rest,
        melds,
        kanDoraOn,
        ...this.evPolicyFacts(obs, rest, hidden),
      }),
    );
    const v = evEvalRest(core);
    this.noteEv(core.meta[R_NODES], core.meta[R_TRUNC] > 0);
    return v;
  }

  /**
   * UNIT D. The PASS line of this decision: hold the hand as it stands.
   *
   * The ONE root that is not hypothetical, so it keeps the hidden-information
   * channels: they are true of the hand that is actually sitting there. (The
   * call side is priced blind — see `evPushOf` — which leaves the comparison
   * asymmetric in the oracle's favour by exactly the value of knowing our own
   * next draw. That is the honest asymmetry: the alternative is to price the
   * pass line with a future the calling line would not have had.)
   */
  private evPassValue(obs: Observation): number {
    const m = this.cache(obs);
    if (m.evRest !== undefined) return m.evRest;
    if (!this.evRestRoot(obs)) return m.evRest = NaN;
    return m.evRest = this.evHoldOf(obs, obs.hand, obs.melds[0], false, this.hiddenInfoOf(obs));
  }

  private computeFold(obs: Observation): boolean {
    // Committed: after riichi the only legal discard is the drawn tile anyway.
    // OUTSIDE the head, permanently: a fact about the position, not a judgement.
    if (obs.riichi[0]) return false;

    // M15. With the core serving discards the verdict is a comparison of two
    // PRICES at the root — the best push line against the best fold line, both
    // in points, both out of the same sweep the discard score reads — and the
    // quiet-table early-out below is deliberately bypassed (plan §4.2):
    // "nothing loud" is a statement about the ASSESSED threats, while the DP
    // runs its own hazard sweep over `tenpaiP` and prices the fold option at
    // every state. No feature is built and no random number is drawn on this
    // path. A claim decision has no discard root, so it falls through to the
    // incumbent gate (the call comparisons are unit D).
    const evd = this.evDiscard(obs);
    if (evd) return evd.bestFold > evd.bestPush;

    const pressure = this.pressure(obs);
    // …and so is this: with nothing loud there is nothing to fold FROM.
    if (pressure === 0) return false;

    // THE PLAIN PATH — no head, no recorder, no exploration. The expression
    // below is the incumbent gate, character for character, and it is what a
    // seat with no `fold` block runs: `foldMargin` is not called, no feature is
    // built, no random number is drawn, and no pin can move.
    if (this.foldHead === null && this.foldSink === null && this.foldEps === 0) {
      let push: number;
      if (this.hand) {
        // M11. The table below is a four-step guess at exactly what `handOutlook`
        // computes — P(this hand cashes) × what it cashes for — so with the model
        // wired the guess goes and the EV comes in, divided by `pushScale` to land
        // back in the units the comparison against `pressure` is written in.
        //
        // THE TWO ADJUSTMENTS GO WITH IT, and not because they were wrong: both
        // are inside the model now, and continuously rather than as cliffs. 親 is
        // `valueDealer` on the value; "late and far from tenpai" is `turnsLeft`
        // (few own draws left to climb three levels) meeting `oppGrowth` (a table
        // that has grown ready), which prices 3向聴 on 9巡目 too instead of
        // switching at 10.
        const r = this.restingShape(obs);
        push = this.outlookOf(obs, r.rest, r.sh, { tile: r.tile }).ev / this.hand.pushScale;
      } else {
        push = obs.shanten <= 0 ? 1.0 : obs.shanten === 1 ? 0.45 : obs.shanten === 2 ? 0.15 : 0;
        push += 0.12 * obs.doraCount;
        // Late and far from tenpai is not a hand worth defending with.
        if (obs.shanten >= 2 && obs.junme >= 10) push = 0;
        // A dealer has more to lose by folding (連荘) — nudge, don't override.
        if (obs.seatWind === 27) push += 0.08;
      }

      push *= this.buffer(obs);

      // 順位効用: the same hand against the same table is worth pushing for a
      // different amount depending on what the points would DO. Off (the default)
      // both scales are 1 and this is the old `push < 0.5 * pressure`, exactly.
      const st = this.standingsOf(obs);
      return push * st.gain < 0.5 * pressure * st.risk;
    }

    // THE HEAD PATH. `foldMargin` is the same arithmetic, with its verdict left
    // as a number so the head can read it as feature 0.
    const parts = this.foldMargin(obs, pressure);
    const facts = this.foldFeatures(obs, parts, pressure);
    const x = this.foldX!;
    const verdict = this.foldHead
      ? decideFold(this.foldHead, facts, x)
      : (foldVector(facts, x), parts.margin < 0);

    // The ε-flip. ONE draw per decision at most: `shouldFold` memoises, so this
    // body runs once per Observation however many callers ask.
    let flipped = false;
    if (this.foldRng !== null && this.foldRng.float() < this.foldEps) flipped = true;
    const taken = flipped ? !verdict : verdict;

    if (this.foldSink) {
      this.foldSink({
        x: Array.from(x),
        verdict,
        taken,
        p: this.foldEps === 0 ? 1 : flipped ? this.foldEps : 1 - this.foldEps,
        flipped,
        turn: obs.drawn !== null,
      });
    }
    return taken;
  }

  /**
   * 持ち点8000未満になる打ち方禁止. The ledger judges the FINAL scores, at 終局
   * (2026-08-27 ruling) — and by then it is too late to play differently — so
   * the buffer, not the breach, is what the policy watches: in the closing
   * stretch, a stack within one deal-in of the line is one bad discard away
   * from ending the game in violation. Before 南入 the buffer stays out of it:
   * an early deal-in leaves most of a hanchan to recover in, and (as the rule
   * is judged) has broken nothing yet.
   *
   * HOOK. `expectedLoss` is what a deal-in is assumed to cost; the base policy
   * has no way to know, so it guesses. A subclass that can price the table
   * overrides this, computes the figure and calls `super` with it.
   */
  protected bufferScale(obs: Observation, expectedLoss = 6000): number {
    // 東場 is not the closing stretch. (A 東風戦 would end in East, but the
    // home dojo plays hanchan — the rule this buffer defends is a hanchan
    // ledger's.)
    if (obs.roundWind === 27) return 1;
    const buffer = obs.scores[0] - 8000;
    if (buffer < expectedLoss) return this.w.bufferTight;
    if (buffer < 2 * expectedLoss) return this.w.bufferLow;
    return 1;
  }

  /**
   * Threat volume: a declared riichi counts full, a loud open hand counts half.
   *
   * HOOK. Both figures are guesses standing in for "how likely is that seat to
   * be tenpai, and for how much" — exactly what an estimator replaces.
   */
  protected pressureOf(obs: Observation): number {
    let p = 0;
    for (let s = 1; s < 4; s++) if (obs.riichi[s]) p += 1;
    // Danger entries carry the furo threats the assessor decided were real.
    const furoSeats = new Set<number>();
    for (const d of obs.danger.values()) {
      for (const detail of d.details) {
        if (detail.kind === "furo") furoSeats.add(detail.seat);
      }
    }
    p += 0.5 * furoSeats.size;
    // 色読み: a dyed field is loud even when nobody has declared anything —
    // this term is what un-zeros the fold gate's quiet-table early-out.
    return p + this.sensePressure(obs);
  }

  /**
   * Threat volume BROKEN OUT PER SEAT, in relative order: 1 for a declared
   * riichi, 0.5 for an open hand the assessor decided was a real threat, 0 for
   * everyone else. The same two facts `pressureOf` sums, kept separate.
   *
   * HOOK, and it exists because M11 needs the vector rather than the scalar: the
   * survival term asks, per own turn, whether the TABLE ends the hand first, and
   * the answer is a function of how much of the table is ready — Σ over this,
   * grown per turn — not of a single loudness number that also carries value.
   * `AugmentedHeuristic` overrides it with `tenpaiP` for the same reason it
   * overrides `pressureOf`.
   */
  protected threatOf(obs: Observation): number[] {
    const out = [0, 0, 0];
    for (let i = 0; i < 3; i++) if (obs.riichi[i + 1]) out[i] = 1;
    // `ThreatDetail.seat` is ABSOLUTE — mjrender's assessor knows nothing of
    // this file's relative indexing — while `obs.riichi` is relative. This is
    // the conversion, the same one `observe.ts` makes for the ledger counts.
    for (const d of obs.danger.values()) {
      for (const detail of d.details) {
        if (detail.kind !== "furo") continue;
        const i = (detail.seat - obs.seat + 4) % 4 - 1;
        if (i >= 0 && out[i] < 0.5) out[i] = 0.5;
      }
    }
    return out;
  }

  // ------------------------------------------------------------ 手牌価値 (M11)

  /** M11's per-decision board facts. Built on demand; never when M11 is off. */
  private basisOf(obs: Observation): HandBasis {
    const m = this.cache(obs);
    if (m.basis) return m.basis;
    const unseen = publicUnseen(obs);
    let unseenTotal = 0;
    for (const n of unseen) unseenTotal += n;
    const doraTypes = doraTypesOf(obs);
    let meldDora = 0;
    for (const meld of obs.melds[0]) {
      for (const t of meld.tiles) {
        if (doraTypes.has(tileType(t))) meldDora++;
        if (obs.akaIds.has(t)) meldDora++;
      }
    }
    return m.basis = {
      melds: obs.melds[0].length,
      open: obs.melds[0].filter((x) => x.kind !== "ankan").length,
      closed: obs.melds[0].every((x) => x.kind === "ankan"),
      doraTypes,
      valueHonors: valueHonorsOf(obs.roundWind, obs.seatWind),
      unseen,
      unseenTotal,
      meldDora,
    };
  }

  /**
   * The model's verdict on one resting 13-tile shape, and the facts behind it.
   *
   * Memoized per decision and keyed by the discard that produced the shape (−1
   * for the hand as it already rests, on a claim decision): the gate asks about
   * the tsumogiri shape and then the score loop asks about it again, and the
   * recorder asks a third time about whichever shape won.
   */
  private handEntry(obs: Observation, rest: Tile[], sh: number, opts: OutlookOpts): HandEntry {
    const m = this.cache(obs);
    const cache = m.outlooks ??= new Map<number, HandEntry>();
    const key = opts.tile ?? -1;
    const hit = cache.get(key);
    if (hit) return hit;
    const facts = this.handFacts(obs, rest, sh, opts);
    const entry: HandEntry = { facts, out: handOutlook(facts, this.handW) };
    cache.set(key, entry);
    return entry;
  }

  /**
   * `HandFacts` for the 13-tile shape `rest`. The ONE place they are assembled —
   * the live seat and the offline fit must never be able to disagree about what
   * a hand looked like, so the recorder writes down this object verbatim rather
   * than re-deriving anything (the honesty rule `calibration.ts` states).
   */
  private handFacts(obs: Observation, rest: Tile[], sh: number, opts: OutlookOpts): HandFacts {
    const b = this.basisOf(obs);
    const counts = countsFromTiles(rest);
    // A 13-tile shape is never complete, so the clamp only ever catches a caller
    // that handed in the 14-tile reading by mistake.
    const shape = sh < 0 ? 0 : sh;

    let live = opts.ukeire?.live ?? 0;
    let types = opts.ukeire?.types ?? 0;
    if (!opts.ukeire) {
      // Correctness over the micro-cost. The base score counts ukeire only for
      // the tiles holding the best shanten (`wideOpen`), which is sound when the
      // shanten term dominates — but the model's first term IS the ukeire, and a
      // candidate priced with a zero there is priced as a dead shape.
      const tys = ukeireTypes(counts, b.melds, b.closed, shape);
      types = tys.length;
      for (const ty of tys) live += b.unseen[ty];
    }

    let dora = b.meldDora;
    for (const t of rest) {
      if (b.doraTypes.has(tileType(t))) dora++;
      if (obs.akaIds.has(t)) dora++;
    }

    let yakuhaiTriplets = 0;
    let yakuhaiPairs = 0;
    for (let ty = 0; ty < 34; ty++) {
      if (!b.valueHonors.has(ty)) continue;
      if (counts[ty] >= 3) yakuhaiTriplets++;
      else if (counts[ty] === 2) yakuhaiPairs++;
    }
    for (const meld of obs.melds[0]) {
      if (meld.kind === "chi") continue;
      if (b.valueHonors.has(tileType(meld.tiles[0]))) yakuhaiTriplets++;
    }

    const furiten = obs.furiten.permanent || obs.furiten.temporary;
    // 後付け, as the referee sees it: `discardInfo` already ran the real scorer
    // over this exact shape. Without a discard to look up (a claim decision) the
    // resting hand's own reading is the same answer.
    const yakuless = opts.tile !== undefined
      ? (obs.discardInfo.get(opts.tile)?.yakuless ?? false)
      : obs.shanten <= 0 && obs.ronnable.length === 0;

    return {
      shanten: shape,
      ukeire: live,
      ukeireTypes: types,
      unseenTotal: b.unseenTotal,
      turnsLeft: Math.floor(obs.wallRemaining / 4),
      junme: obs.junme,
      dora,
      open: b.open,
      closed: b.closed,
      riichi: obs.riichi[0],
      yakuhaiTriplets,
      yakuhaiPairs,
      honitsu: this.honitsuShape(obs, rest),
      // Above tenpai every wait is still hypothetical, so the shape is assumed
      // curable; at tenpai a closed hand only needs to not be furiten (riichi
      // supplies the yaku), while an open one needs a wait that actually scores.
      ronnable: shape > 0 ? true : b.closed ? !furiten : !yakuless,
      furiten,
      dealer: obs.seatWind === 27,
      oppTenpai: this.threat(obs),
      honba: obs.honba,
      kyotaku: obs.kyotaku,
    };
  }

  /**
   * 混一色/清一色模様 — the reading `hasYakuProspect` makes of an open shape,
   * generalized to a hand that may have no melds at all: every non-honor tile
   * outside the dominant suit is a stray, and two strays are still curable.
   * Honors are welcome, so an all-honor shape counts.
   */
  private honitsuShape(obs: Observation, rest: Tile[]): boolean {
    const bySuit = [0, 0, 0];
    let n = 0;
    const add = (t: Tile) => {
      const ty = tileType(t);
      if (isHonor(ty)) return;
      bySuit[(ty / 9) | 0]++;
      n++;
    };
    for (const t of rest) add(t);
    for (const meld of obs.melds[0]) for (const t of meld.tiles) add(t);
    return n - Math.max(bySuit[0], bySuit[1], bySuit[2]) <= 2;
  }

  /**
   * M11's verdict on the 13-tile shape `rest`.
   *
   * HOOK. A policy that can price its own hand better than a closed form over
   * public counts overrides this; everything downstream — the gate, the discard
   * score, the recorder — reads the model through here.
   *
   * DEVIATION from the M11 brief, deliberate: the first parameter is an
   * `Observation` and not a `Ctx`. The push/fold gate is one of the two callers
   * and runs before any `Ctx` exists — `Ctx.folding` is what the gate decides —
   * so a `Ctx`-taking hook could serve the discard score and nothing else, which
   * is half the model. Nothing in `HandFacts` needs more than the Observation
   * and the per-decision `HandBasis` anyway.
   */
  protected outlookOf(
    obs: Observation,
    rest: Tile[],
    sh: number,
    opts: OutlookOpts = {},
  ): HandOutlook {
    return this.handEntry(obs, rest, sh, opts).out;
  }

  /**
   * The 13 tiles this seat is resting on right now: the hand minus the drawn
   * tile on a turn decision, and the hand itself on a claim decision (where the
   * shape is already at rest and `obs.shanten` describes it).
   */
  private restingShape(obs: Observation): { rest: Tile[]; sh: number; tile?: Tile } {
    const drawn = obs.drawn;
    if (drawn === null) return { rest: obs.hand, sh: obs.shanten };
    const rest = [...obs.hand];
    rest.splice(rest.lastIndexOf(drawn), 1);
    const known = obs.discardInfo.get(drawn)?.shanten;
    const b = this.basisOf(obs);
    return { rest, sh: known ?? shanten(countsFromTiles(rest), b.melds, b.closed), tile: drawn };
  }

  // ---------------------------------------------------------------- discard

  private chooseDiscard(ctx: Ctx, legal: Action[]): Action | null {
    const discards = legal.filter((a): a is Extract<Action, { t: "discard" }> => a.t === "discard");
    if (discards.length === 0) return null;

    // One entry per distinct tile — the riichi flag is decided separately.
    const byTile = new Map<Tile, Extract<Action, { t: "discard" }>[]>();
    for (const d of discards) {
      const g = byTile.get(d.tile);
      if (g) g.push(d);
      else byTile.set(d.tile, [d]);
    }

    // THE FILTER. Ask the referee about every candidate before scoring any of
    // them, and if some subset is clean, that subset IS the choice set — no
    // score, from this class or a subclass hook, can nominate a tile outside it.
    // An empty result means every discard is charged, and the priced fallthrough
    // below (`dojoCost`) decides which charge to take.
    const clean = this.compliantDiscards(ctx, byTile);
    const candidates = this.guardTriplets(
      ctx,
      [...byTile.keys()].filter((t) => clean === null || clean.has(t)),
    );
    const shantenAfter = new Map<Tile, number>();
    let best = Infinity;
    for (const tile of candidates) {
      const s = this.shantenWithout(ctx, tile);
      shantenAfter.set(tile, s);
      if (s < best) best = s;
    }

    // M9. One assembly per decision when a consumer is set, and not a single
    // line of work when one is not.
    let run: EvidenceRun | null = null;
    if (this.consumer) {
      const hooks = this.evidenceHooks();
      run = { hooks, context: assembleContext(hooks, ctx) };
    }

    // M15. With the core on, THE PRICE IS THE DP's and `scoreDiscard` is not
    // called at all — the linear surrogate it computes is precisely what the
    // core replaces (plan §4.3). What survives beside it is only what was never
    // a valuation: the two judgments no fitted core may buy its way around
    // (`dojoCost`, `senseLineTax`), converted from score units into points by
    // `pointsPerScore` (D4) so a 4000-score veto is still a 16,000-point one.
    // The candidate SET is untouched: `compliantDiscards` and `guardTriplets`
    // ran above, and a price never re-admits what a veto struck out.
    //
    // THE PLANNER'S TWO BONUS HOOKS ARE NOT HERE (2026-08-30 review). They are
    // the INCUMBENT's steering terms — `keepBonus` alone carries C7's `planKeep`
    // 5000, which at `pointsPerScore` 4 is a 20,000-point thumb on a scale whose
    // whole hand is usually worth less than that — and steering is precisely
    // what the DP replaces: it prices the SHAPE, so "keep the tiles the plan is
    // built from" is either already true of the price or an instruction to
    // ignore it. What the two hooks knew that the DP does not is the
    // hidden-information half (C4's incoming draw, C5's coming dora), and that
    // reaches the DP properly, through `hiddenInfoOf`'s channels. C6 (a tile
    // about to become genbutsu) is the one term with no channel — plan D7 bars
    // per-opponent sequential information from the 計算 seat — and it is dropped
    // rather than smuggled in as a 20,000-point tiebreak.
    const evd = this.evDiscard(ctx.obs);
    const pps = this.pointsPerScore();

    let bestTile = candidates[0];
    let bestScore = -Infinity;
    const ranked: DiscardTrace["candidates"] = [];
    for (const tile of candidates) {
      const sh = shantenAfter.get(tile)!;
      // Ukeire is the expensive part (34 shanten probes); only the tiles that
      // actually hold the best shanten can win on it.
      const score = evd
        ? this.evPriceOf(ctx, evd, byTile, tile) -
          pps * (this.dojoCost(ctx, tile, sh) + this.senseLineTax(ctx, tile, sh))
        : this.scoreDiscard(ctx, tile, sh, sh === best, run);
      ranked.push({ tile, shanten: sh, score });
      if (score > bestScore || (score === bestScore && tile < bestTile)) {
        bestScore = score;
        bestTile = tile;
      }
    }

    // M11's lane. One sample per TURN decision, for the shape the policy
    // actually chose — never for the candidates it rejected, because the label
    // the fit needs (did this hand cash, and for how much) is a property of the
    // 局 that followed, and only the chosen shape had one. The riichi question
    // below cannot move it: `HandFacts.riichi` is the declaration already
    // standing, not the one about to be made.
    if (this.handSink && ctx.obs.drawn !== null) {
      const e = this.handEntry(
        ctx.obs,
        this.handWithout(ctx, bestTile),
        shantenAfter.get(bestTile)!,
        { tile: bestTile },
      );
      this.handSink({ facts: e.facts, pwin: e.out.pwin, value: e.out.value });
    }

    // M15b's lane, and the same discipline one layer over: one sample per TURN
    // decision, for the resting shape the policy actually chose. The WIRE is
    // recorded, not a verdict — the seat holds no core to ask — so the fit
    // replays it under any candidate parameter vector and the labels are this
    // seat's own continuation of the hand.
    if (this.evSink && this.evWire && ctx.obs.drawn !== null) {
      packEvInputs(this.evWire, this.evFactsForRest(ctx.obs, bestTile));
      this.evSink({
        ints: this.evWire.ints,
        dbls: this.evWire.dbls,
        shanten: shantenAfter.get(bestTile)!,
      });
    }

    // The evaluation the trace reports from. Normally the discard sweep's; a
    // riichi-only seat has the same evaluation without having priced with it.
    const evTrace = evd ?? this.evRiichiPrice(ctx.obs);

    const group = byTile.get(bestTile)!;
    const plain = group.find((d) => !d.riichi);
    const riichi = group.find((d) => d.riichi);
    const trace: DiscardTrace = {
      folding: ctx.folding,
      candidates: ranked.sort((a, b) => b.score - a.score || a.tile - b.tile),
      chosen: bestTile,
      riichi: false,
      mustCure: false,
      // The core's own two numbers for the tile actually chosen: what folding
      // after it is worth, and what declaring on it is worth. Unit C makes the
      // second one a DECISION (`wantRiichi` reads the same pair off the same
      // evaluation); here it is the record of what that decision saw.
      //
      // `units` marks the SCORE column only, so it is set by `evd` alone: a
      // seat carrying `{discard:false, riichi:true}` still ranks candidates
      // with the linear surrogate, and mislabelling that column would be worse
      // than leaving the two point-valued fields unlabelled.
      ...(evd ? { units: "points" as const } : {}),
      ...(evTrace
        ? {
          foldValue: evTrace.foldLine[tileType(bestTile)],
          riichiValue: evTrace.riichi[tileType(bestTile)],
        }
        : {}),
    };
    this.lastTrace = trace;
    if (riichi && !this.riichiBanned(ctx, bestTile) && this.riichiClean(ctx, riichi)) {
      // A split wait must not be left damaten. Riichi is the cure and the
      // dojo's own prescription (役なしなら即リーチ), so it overrides the
      // ordinary "is this worth declaring" judgement — but never a 禁じ手.
      const mustCure = this.dojo && (ctx.obs.discardInfo.get(bestTile)?.katagari ?? false);
      if (mustCure || this.wantRiichi(ctx, bestTile)) {
        trace.riichi = true;
        trace.mustCure = mustCure;
        return riichi;
      }
    }
    return plain ?? group[0];
  }

  /**
   * M15. WHICH of the core's four numbers prices this candidate — the one line
   * the seat is actually going to play.
   *
   * FOLDING TAKES THE FOLD LINE (2026-08-30 review; this was the fatal one).
   * `computeFold` decides by `bestFold > bestPush`, so a seat that has decided
   * to fold has decided that the fold line is the better of the two — and then
   * ranked its candidates by `O_TOTAL`, which is the PUSH line and does not
   * even contain a fold option. The two argmaxes are different tiles by
   * construction: `O_TOTAL` pays for the shape's future and `O_FOLDLINE` pays
   * for surviving without one, so the folding seat threw the tile that best
   * advanced a hand it had just abandoned. `O_FOLDLINE` is the same root
   * arithmetic (`−costIn − endCost + surv·fold`), so the two are comparable
   * numbers in the same units and the dojo/sense taxes subtract off either.
   *
   * PUSHING TAKES `O_TOTAL`, which is `max(dama, riichi)` — but the riichi half
   * of that max is only real if the seat would be ALLOWED to declare. The
   * dojo's 禁じ手 (地獄単騎, 即引っかけ) and the referee's own reading are
   * vetoes applied AFTER the tile is chosen, and the DP knows nothing about
   * either, so a candidate whose riichi is forbidden must be priced at its dama
   * line: otherwise the seat picks a tile for a declaration it will then be
   * refused, and plays the shape damaten at a price it never compared.
   * `wantRiichi`'s own preferences (furiten, 純カラ) are deliberately NOT
   * applied here — those are judgements the DP prices for itself; these two are
   * rules it cannot see.
   */
  private evPriceOf(
    ctx: Ctx,
    evd: EvResult,
    byTile: Map<Tile, Extract<Action, { t: "discard" }>[]>,
    tile: Tile,
  ): number {
    const ty = tileType(tile);
    if (ctx.folding) return evd.foldLine[ty];
    // The declaration is not what the max would take: nothing to check.
    if (!(evd.riichi[ty] > evd.dama[ty])) return evd.total[ty];
    // Not on offer at all (open, already declared, under 1000 点, wall < 4, or
    // the discard does not leave tenpai) — `legal.ts` is the authority.
    const decl = byTile.get(tile)?.find((d) => d.riichi);
    if (!decl) return evd.dama[ty];
    if (this.riichiBanned(ctx, tile) || !this.riichiClean(ctx, decl)) return evd.dama[ty];
    return evd.total[ty];
  }

  /**
   * The discards the ledger would let pass, or null when the question cannot be
   * asked (no preview wired, or the dojo leash is off) — and null again when the
   * answer is "none of them", which is the fallthrough the prices exist for.
   *
   * Each tile is judged by the plain discard: the riichi variant carries extra
   * rules (地獄単騎, 即引っかけ …) and is settled separately, once the tile is
   * chosen, by `riichiClean`.
   *
   * 片和了り and 後付け are vetoed here too, even though the preview cannot see
   * them. They are the only two ledger rules whose CHARGE lands at win time —
   * unpreviewable by construction, since declining the win would be 見逃し — but
   * whose only PREVENTION is a discard, right here. Leaving them to `dojoCost`
   * makes them a price, and a price loses: the C7 planner's `planKeep` malus
   * (5000) simply outbids both. So they are vetoes like everything else, and the
   * prices stay for the fallthrough, where ranking damage is all that is left.
   */
  private compliantDiscards(
    ctx: Ctx,
    byTile: Map<Tile, Extract<Action, { t: "discard" }>[]>,
  ): Set<Tile> | null {
    const pv = this.referee(ctx);
    if (!pv) return null;
    const ok = new Set<Tile>();
    for (const [tile, group] of byTile) {
      const a = group.find((d) => !d.riichi) ?? group[0];
      if (pv.discard(a, ctx.obs.drawn).length !== 0) continue;
      const info = ctx.obs.discardInfo.get(tile);
      // 片和了り, but only when riichi is not on offer: declaring is itself a
      // yaku, so it makes every wait scoring and the shape stops being split.
      if (!ctx.canRiichi && info?.katagari) continue;
      // 後付け: only an open hand is stuck with a yakuless tenpai; a closed one
      // can still cure the same shape by declaring.
      if (!ctx.closed && info?.yakuless) continue;
      ok.add(tile);
    }
    return ok.size > 0 ? ok : null;
  }

  /**
   * The triplet guard — see `HeuristicWeights.keepTriplet`. Returns the
   * candidates with the doctrine's vetoes removed, or the input untouched when
   * the guard is off, does not apply (open or folding hand, no triplet held),
   * or would veto everything.
   */
  private guardTriplets(ctx: Ctx, candidates: Tile[]): Tile[] {
    if (this.w.keepTriplet === 0 || ctx.folding || ctx.obs.melds[0].length > 0) return candidates;
    const held = countsFromTiles(ctx.obs.hand);
    if (!held.some((n) => n === 3)) return candidates;
    // Standard shanten of every candidate's kept shape; the best of them is
    // what a triplet break must match to stay a standard-form decision.
    const std = new Map<Tile, number>();
    let bestStd = Infinity;
    for (const tile of candidates) {
      const s = shanten(countsFromTiles(this.handWithout(ctx, tile)), 0, false);
      std.set(tile, s);
      if (s < bestStd) bestStd = s;
    }
    const kept = candidates.filter((tile) => {
      if (held[tileType(tile)] !== 3) return true;
      const s = std.get(tile)!;
      if (s <= bestStd) return true;
      const chi = chiitoiShanten(countsFromTiles(this.handWithout(ctx, tile)));
      // Breaking into 七対子聴牌 is the sanctioned exception; a break whose
      // shape does not even ride the pairs line is not this doctrine's business.
      return chi === 0 || chi >= s;
    });
    return kept.length > 0 ? kept : candidates;
  }

  /** Would declaring on this discard stay off the ledger? */
  private riichiClean(ctx: Ctx, a: Extract<Action, { t: "discard" }>): boolean {
    const pv = this.referee(ctx);
    return !pv || pv.discard(a, ctx.obs.drawn).length === 0;
  }

  /** The 13-tile shape left behind by this discard. Protected: the hooks want it. */
  protected handWithout(ctx: Ctx, tile: Tile): Tile[] {
    const rest = [...ctx.obs.hand];
    rest.splice(rest.lastIndexOf(tile), 1);
    return rest;
  }

  private shantenWithout(ctx: Ctx, tile: Tile): number {
    // The referee already worked this out for every legal discard.
    const known = ctx.obs.discardInfo.get(tile);
    if (known) return known.shanten;
    const counts = countsFromTiles(this.handWithout(ctx, tile));
    return shanten(counts, ctx.open, ctx.closed);
  }

  /**
   * One discard candidate's score.
   *
   * TWO PATHS, ONE SURROUND. With a consumer set (M9) the CORE — the efficiency
   * aggregate, the price of danger, and the two hook bonuses — is computed from
   * the named evidence vector by `consumer.ts` instead of by the arithmetic
   * below; `dojoCost` is subtracted identically either way, because a 禁じ手 is
   * priced by the ledger and not by anything a fit is allowed to move. At init
   * the two paths agree bit for bit (`initFromWeights`), which is what makes the
   * swap measurable rather than merely plausible.
   */
  private scoreDiscard(
    ctx: Ctx,
    tile: Tile,
    sh: number,
    wideOpen: boolean,
    run: EvidenceRun | null,
  ): number {
    if (this.consumer && run) {
      const ev = {
        context: run.context,
        candidate: assembleCandidate(run.hooks, ctx, tile, sh, wideOpen),
      };
      return consumeEvidence(ev, this.consumer) - this.dojoCost(ctx, tile, sh) -
        this.senseLineTax(ctx, tile, sh);
    }

    const { obs } = ctx;
    const rest = this.handWithout(ctx, tile);
    const counts = countsFromTiles(rest);

    let eff = -sh * this.w.shanten;

    // Hoisted out of the `wideOpen` branch so M11 can be handed the count this
    // block already paid for instead of enumerating the 34 probes a second time.
    let live = 0;
    let types = 0;
    if (wideOpen) {
      const tys = ukeireTypes(counts, ctx.open, ctx.closed, sh);
      types = tys.length;
      for (const ty of tys) live += ctx.unseen[ty];
      eff += live * this.w.ukeire + types * this.w.ukeireType;
    }

    if (this.hand) {
      // M11. The dora count is a stand-in for "what is this shape worth if it
      // lands" — one term of a value model, priced linearly, and with no opinion
      // at all on whether the hand will land. The model computes the whole
      // product, dora included; `evWeight` converts its points into the score
      // units the rest of this sum is written in. Everything else here —
      // shanten, ukeire, the 役牌 pair, the lone honor — is unchanged, because
      // those are counting terms and the model is a valuation term.
      eff += this.hand.evWeight *
        this.outlookOf(obs, rest, sh, {
          tile,
          ukeire: wideOpen ? { live, types } : undefined,
        }).ev;
    } else {
      // Value kept. Melded dora is constant across candidates, so hand-only is
      // enough to rank them.
      let dora = 0;
      for (const t of rest) {
        if (ctx.doraTypes.has(tileType(t))) dora++;
        if (obs.akaIds.has(t)) dora++;
      }
      eff += dora * this.w.dora;
    }

    for (let ty = 0; ty < 34; ty++) {
      if (counts[ty] >= 2 && ctx.valueHonors.has(ty)) eff += this.w.yakuhaiPair;
      if (counts[ty] === 1 && isHonor(ty)) eff -= this.w.isolatedHonor * Math.min(obs.junme, 12);
    }

    // The dojo cost is deliberately outside `ctx.eff`: folding must not make a
    // 禁じ手 cheap. The ledger charges the same either way. The two bonus hooks
    // sit outside it too, and outside `ctx.eff`/`ctx.def`: they are already in
    // score units and already know whether the policy is folding. The 色読み
    // line tax rides beside `dojoCost` on both paths for the same reason: it is
    // a judgment about which LINE the hand may commit to, and no fitted core is
    // allowed to buy its way around it.
    return ctx.eff * eff - ctx.def * this.riskOf(ctx, tile) +
      this.drawBonus(ctx, tile) - this.keepBonus(ctx, tile) -
      this.dojoCost(ctx, tile, sh) - this.senseLineTax(ctx, tile, sh);
  }

  /**
   * What letting this tile go costs defensively, in score units, BEFORE the
   * fold multiplier.
   *
   * HOOK. The base policy reads mjrender's four danger levels off the
   * observation; a policy holding a per-tile deal-in probability and a payment
   * to go with it computes the product instead. Whatever the source, "安全"
   * must stay free — that level means provably safe (genbutsu), and no estimate
   * outranks a proof. The 感性 surcharges (`surcharge`: 色読み plus the 生牌
   * 役牌 charge) are ADDED outside that contract, and deliberately so: 安全 is a
   * proof against the ASSESSED threats, while those terms price hands the
   * assessor holds no entry for — the sense's own proof test is
   * `FieldSense.safe`, the dye source's discards, and a 役牌 nobody has shown
   * is by construction genbutsu against nobody.
   */
  protected riskOf(ctx: Ctx, tile: Tile): number {
    return this.ruleRisk(this.dangerLevelOf(ctx, tile)) + this.surcharge(ctx, tile);
  }

  /**
   * The assessor's own reading of this tile, or `undefined` when it was not
   * looking at all (no declared threat on the table).
   *
   * Split from `riskOf` because an override needs BOTH halves and the two are
   * one map lookup: `AugmentedHeuristic` prices its estimate against the rule
   * ladder while also honouring an EXPLICIT 安全 as a proof, and an absent entry
   * is not that proof.
   */
  protected dangerLevelOf(ctx: Ctx, tile: Tile): DangerLevel | undefined {
    return ctx.obs.danger.get(tileType(tile))?.level;
  }

  /** The rule ladder's price for a level; an absent reading costs nothing. */
  protected ruleRisk(level: DangerLevel | undefined): number {
    return this.w.danger[level ?? "安全"];
  }

  /**
   * A bonus ADDED to this discard's score. Zero here: the base policy has no
   * one-turn lookahead. Overridden by a policy that knows what is coming off
   * the wall — the discard whose kept shape accepts the incoming tile, the one
   * that keeps a copy of a dora about to be flipped.
   */
  protected drawBonus(_ctx: Ctx, _tile: Tile): number {
    return 0;
  }

  /**
   * A malus SUBTRACTED from this discard's score: a reason to hold the tile
   * back this turn rather than spend it. Zero here. Overridden by a policy that
   * can see a tile is about to become genbutsu anyway, and would rather spend a
   * less useful safe tile first.
   */
  protected keepBonus(_ctx: Ctx, _tile: Tile): number {
    return 0;
  }

  /**
   * What the ledger would charge for this discard, in score units.
   *
   * With the compliance filter live this is mostly moot for the rules it shares
   * with the preview — a charged discard never reaches the score loop unless
   * EVERY discard is charged, and then these prices are exactly what ranks the
   * damage. The two win-time rules are different: 片和了り and 後付け fire when a
   * hand is CASHED, which a policy cannot decline without going furiten, so the
   * only prevention is here, at the discard that builds the wait.
   */
  private dojoCost(ctx: Ctx, tile: Tile, sh: number): number {
    if (!this.dojo) return 0;
    const { obs } = ctx;
    const ty = tileType(tile);
    let cost = 0;

    // 第一打字牌切り — our river is still empty when we pick this discard.
    if (obs.rivers[0].length === 0 && isHonor(ty)) cost += this.w.firstHonor;

    // 不聴時ドラ切り. Indicator dora only: the aka 5p may be cut before tenpai.
    // Charged only from 3向聴 out — 2向聴以内 is allowed.
    // 例外: an honor dora already twice in the rivers is spent.
    // `4 − unseen` IS the visible count: rivers, melds, indicators and own hand
    // are exactly what `publicUnseen` subtracts, and there are only four copies.
    if (sh > 2 && ctx.doraTypes.has(ty)) {
      if (!(isHonor(ty) && ctx.unseen[ty] <= 2)) cost += this.w.notenDora;
    }

    // 片和了り, but only when riichi is not on offer: riichi is itself a yaku,
    // so declaring it makes every wait scoring and the shape stops being split.
    if (!ctx.canRiichi && obs.discardInfo.get(tile)?.katagari) cost += this.w.katagari;

    // 後付け: an open hand tenpai on nothing that scores. A closed hand can cure
    // the same shape by declaring, so only the open case is a violation — and
    // the only prevention is not making that discard.
    if (!ctx.closed && obs.discardInfo.get(tile)?.yakuless) cost += this.w.yakulessTenpai;

    // ドラ切りをポンされた後の手出し. `legal.ts` will not stop us — the dojo
    // takes the payment instead — so the price has to be paid here.
    if (obs.tsumogiriLock && tile !== obs.drawn) cost += this.w.tsumogiriLock;

    return cost;
  }

  // ----------------------------------------------------------------- riichi

  /**
   * The waits this hand is on. Approximate when the chosen discard is not the
   * drawn tile: `obs.waits` describes the resting hand, and only a tsumogiri
   * leaves that shape untouched.
   */
  private waitsOf(obs: Observation): number[] {
    return obs.waits.length ? obs.waits : obs.ukeire.map((u) => u.type);
  }

  /** Riichi declarations the dojo forbids outright. Never overridden. */
  private riichiBanned(ctx: Ctx, discard: Tile): boolean {
    if (!this.dojo) return false;
    const { obs } = ctx;
    const waits = this.waitsOf(obs);
    if (waits.length === 0) return true;
    const live = this.liveWaits(obs, waits);

    // 地獄単騎: a lone honor wait with most copies already gone.
    if (waits.length === 1 && isHonor(waits[0]) && live <= 1) return true;

    // 即引っかけ: a wait one suji away from the tile being cut. The dojo asks
    // for a 巡 of daylight; a policy with no memory of when the shape arrived
    // simply declines.
    const dt = tileType(discard);
    if (!isHonor(dt)) {
      return waits.some((w) =>
        !isHonor(w) && suitOfType(w) === suitOfType(dt) &&
        Math.abs(rankOfType(w) - rankOfType(dt)) === 3
      );
    }
    return false;
  }

  private wantRiichi(ctx: Ctx, discard: Tile): boolean {
    const { obs } = ctx;
    const waits = this.waitsOf(obs);
    if (waits.length === 0) return false;
    if (this.liveWaits(obs, waits) === 0) return false; // 純カラ — nothing to win on

    // Furiten riichi is legal in the dojo but rarely what you want.
    if (obs.furiten.permanent || obs.furiten.temporary) return false;

    if (obs.wallRemaining < 4) return false;

    // M15 UNIT C. The four gates above are untouched — they are dojo doctrine
    // and a price never re-admits what a veto struck out — but INSIDE them the
    // question "declare or stay damaten" is a comparison of two prices for the
    // shape this discard leaves behind, and the core answers it exactly:
    // `V_riichi` against `V_dama` for the CHOSEN tile's type, both out of the
    // same sweep the discard score was read from. The M12 head is not consulted
    // (D3: a vector carrying both is refused at load), and the 最終形 doctrine
    // it encoded is re-expressed by the DP's own hold-for-upgrade branch plus
    // `riichiMargin` — whether that reproduces the 2026-08-27 ruling is what
    // the paired grade of unit C measures.
    //
    // The DP reports `O_RIICHI` as −inf (or simply equal to dama) where riichi
    // is not available for a candidate, so a non-finite pair is read as "no
    // declaration on offer here" rather than as a comparison.
    const evr = this.evRiichiPrice(obs);
    if (evr) {
      const ty = tileType(discard);
      const r = evr.riichi[ty];
      const d = evr.dama[ty];
      if (!Number.isFinite(r) || !Number.isFinite(d)) return false;
      return r > d + this.ev!.params.riichiMargin;
    }

    // M12. Absent the core, the head is consulted only INSIDE the region the
    // gates admit, and only to pick declare over damaten. Absent both,
    // declaring unconditionally is the answer the gates have always given.
    if (!this.riichiHead) return true;
    return decideRiichi(this.riichiFeatures(ctx, discard), this.riichiHead);
  }

  /**
   * The head's view of one gated-in declaration. Everything comes off the
   * memoized `handEntry` of the POST-DISCARD resting shape — the same entry the
   * M11 hooks read, so no new computation path exists — plus the one public
   * fact the outlook does not carry: who has already declared. The gates above
   * deliberately keep reading `waitsOf`'s resting-hand approximation; the head
   * is free to see the committed shape.
   */
  private riichiFeatures(ctx: Ctx, discard: Tile): RiichiFeatures {
    const { obs } = ctx;
    const rest = this.handWithout(ctx, discard);
    // sh 0: riichi is only on offer when the discard leaves tenpai.
    const { facts, out } = this.handEntry(obs, rest, 0, { tile: discard });
    let oppRiichi = 0;
    for (let s = 1; s < 4; s++) if (obs.riichi[s]) oppRiichi++;

    // 最終形 doctrine (2026-08-27): both features share one wait-mass basis —
    // live copies by `publicUnseen`, so today's mass and a hypothetical
    // upgrade's mass are comparable numbers.
    const unseen = publicUnseen(obs);
    const counts = countsFromTiles(rest);
    const nM = obs.melds[0].length; // ankan only — riichi needs a menzen hand
    const waits = ukeireTypes(counts, nM, true, 0);
    let mass = 0;
    for (const ty of waits) mass += unseen[ty];
    const improvable = this.waitUpgradeExists(counts, nM, mass, unseen) ? 1 : 0;

    // The held counter advances once per 巡 and only while the head is being
    // consulted (tenpai inside the four gates); an improvement resets it. The
    // high-water mass is what "improve" is measured against, so a wait that
    // wobbles down and back does not reset the clock.
    const key = `${obs.kyoku}:${obs.honba}`;
    const st = this.riichiHold;
    let held = 0;
    if (st && st.key === key) {
      if (obs.junme > st.junme) {
        held = mass > st.mass ? 0 : st.held + 1;
        this.riichiHold = { key, junme: obs.junme, mass: Math.max(mass, st.mass), held };
      } else {
        held = st.held;
      }
    } else {
      this.riichiHold = { key, junme: obs.junme, mass, held: 0 };
    }

    return {
      ev: out.ev / 1000,
      pwin: out.pwin,
      value: out.value / 1000,
      liveWaits: facts.ukeire,
      waitTypes: facts.ukeireTypes,
      junme: facts.junme,
      turnsLeft: facts.turnsLeft,
      dora: facts.dora,
      dealer: facts.dealer ? 1 : 0,
      oppRiichi,
      kyotaku: facts.kyotaku,
      improvable,
      tenpaiHeld: held,
      holdShape: this.riichiHoldShape(obs, counts, waits, mass, out.value, facts.dealer) ? 1 : 0,
    };
  }

  /**
   * The refined 最終形 doctrine (2026-08-27), as one verdict: should this
   * tenpai HOLD rather than declare immediately? Held ⇔ the wait's acceptance
   * is not strictly better than 2 live tiles, OR the hand is riichi(+平和)
   * only — priced by the M11 value model: `value` below the model's own
   * declared-hand baseline (`valueRiichi` plus half a dora, dealer-scaled)
   * is a hand riichi itself is most of — UNLESS the shape is a sanctioned
   * 単騎: ドラ単騎, 七対子の単騎, or a 役満形の単騎 (四暗刻単騎, 国士無双).
   * `tenpaiHeld` remains the release: ~2 own turns without improvement and
   * the head may declare anyway.
   */
  private riichiHoldShape(
    obs: Observation,
    counts: number[],
    waits: number[],
    mass: number,
    value: number,
    dealer: boolean,
  ): boolean {
    const wide = mass > 2;
    const w = this.handW;
    const cheap = value < (w.valueRiichi + 0.5 * w.valuePerDora) * (dealer ? w.valueDealer : 1);
    if (wide && !cheap) return false;

    const melds = obs.melds[0];
    const ns: number[] = [];
    const kinds: number[] = [];
    for (let ty = 0; ty < 34; ty++) {
      if (counts[ty] > 0) {
        ns.push(counts[ty]);
        kinds.push(ty);
      }
    }

    // 国士無双: a truly closed hand of nothing but 幺九 is that tenpai,
    // whichever of its tiles waits — 13面 included, so before any 単騎 test.
    if (melds.length === 0 && kinds.every(isYaochu)) return false;

    // The 単騎 exceptions: a single-type wait pairing a lone tile.
    if (waits.length === 1 && counts[waits[0]] === 1) {
      if (doraTypesOf(obs).has(waits[0])) return false; // ドラ単騎
      // 七対子聴牌: six pairs and the lone wait tile.
      if (
        melds.length === 0 &&
        ns.filter((n) => n === 2).length === 6 && ns.filter((n) => n === 1).length === 1
      ) {
        return false;
      }
      // 四暗刻単騎: four concealed triplets (暗槓 counts) and the lone wait.
      const ankan = melds.filter((m) => m.kind === "ankan").length;
      if (ankan === melds.length && ns.filter((n) => n === 3).length + ankan === 4) {
        return false;
      }
    }
    return true;
  }

  /**
   * Could a LIVE draw rebuild this tenpai onto a strictly wider wait? The
   * probe behind the head's `improvable` feature: for every unseen draw type
   * that is not already a winning tile, try every discard that keeps tenpai
   * and ask whether the new wait's live mass beats the current one. Discarding
   * into the new wait is skipped — that tenpai would be born furiten.
   *
   * 34×~13 kernel calls at worst, and only at a gated-in riichi decision —
   * the rarest decision the policy faces.
   */
  private waitUpgradeExists(
    counts: number[],
    openMelds: number,
    massNow: number,
    unseen: number[],
  ): boolean {
    const c = [...counts];
    const winsNow = new Set(ukeireTypes(c, openMelds, true, 0));
    for (let d = 0; d < 34; d++) {
      if (unseen[d] === 0 || c[d] >= 4) continue;
      // A winning tile drawn is a win, not an upgrade path.
      if (winsNow.has(d)) continue;
      c[d]++;
      for (let x = 0; x < 34; x++) {
        if (x === d || c[x] === 0) continue;
        c[x]--;
        if (shanten(c, openMelds, true) === 0) {
          const w2 = ukeireTypes(c, openMelds, true, 0);
          if (!w2.includes(x)) {
            let mass = 0;
            for (const ty of w2) mass += unseen[ty] - (ty === d ? 1 : 0);
            if (mass > massNow) {
              return true;
            }
          }
        }
        c[x]++;
      }
      c[d]--;
    }
    return false;
  }

  private liveWaits(obs: Observation, waits: number[]): number {
    return waits.reduce((n, ty) => n + (obs.ukeire.find((u) => u.type === ty)?.live ?? 0), 0);
  }

  // ------------------------------------------------------------------ calls

  /**
   * Which pon/chi to take, if any: the one that buys a shanten step, subject to
   * the open hand still having a route to a yaku.
   *
   * HOOK (protected only for that reason — the base behaviour is unchanged): a
   * policy with a locked-on target overrides this to accept only the calls its
   * plan asked for, and to decline the rest even when they are faster.
   */
  protected chooseCall(ctx: Ctx, legal: Action[]): Action | null {
    const calls = legal.filter((a) => a.t === "pon" || a.t === "chi");
    if (calls.length === 0) return null;
    if (ctx.folding) return null;

    // M15 UNIT D. With the core serving calls the ACCEPTANCE RULE changes and
    // the VETOES do not: `hasYakuProspect` (inside `callShape`) and the
    // referee's compliance test (applied by `decide`, outside this hook) strike
    // out exactly what they always struck out, and the surviving calls are
    // ranked by price instead of by shanten.
    //
    // THE SHANTEN-IMPROVEMENT REQUIREMENT IS WHAT GOES. "Only a call that buys
    // a step" is a rule about SPEED, and speed is one term of a value the DP
    // computes in full — a 役牌ポン that leaves shanten alone can still be the
    // better hand, because it converts a yakuless shape into a scoring one and
    // takes the tile off the table; conversely a step bought into a cheap,
    // open, defenceless hand can price below simply holding. So the comparison
    // is the plan's: `bestPush(post-call) − V_pass > callMargin`, with V_pass
    // the value of holding the 13 tiles we have.
    if (this.ev?.params.calls) return this.chooseCallByEv(ctx, calls);

    let best: Action | null = null;
    let bestSh = ctx.obs.shanten;
    for (const a of calls) {
      if (a.t !== "pon" && a.t !== "chi") continue;
      const after = this.shantenAfterCall(ctx, a);
      if (after === null) continue;
      if (after < bestSh) {
        bestSh = after;
        best = a;
      }
    }
    return best;
  }

  /**
   * M15 unit D: the priced call choice. The best-priced admissible call, or
   * null when none of them beats passing by `callMargin`.
   *
   * Ties go to the call that appears first in `legal`, which is the driver's
   * own deterministic order — the same tie-break discipline as the incumbent
   * loop, which keeps its first strict improvement.
   */
  private chooseCallByEv(ctx: Ctx, calls: Action[]): Action | null {
    const { obs } = ctx;
    const pass = this.evPassValue(obs);
    // The DP refused the root (an ill-formed rest can only be a caller bug, but
    // a refusal is not a reason to call): with no PASS line to compare against
    // there is no comparison, and not calling is the null action.
    if (!Number.isFinite(pass)) return null;

    const m = this.cache(obs);
    const memo = m.evCalls ?? (m.evCalls = new Map<string, number>());

    let best: Action | null = null;
    let bestValue = -Infinity;
    for (const a of calls) {
      if (a.t !== "pon" && a.t !== "chi") continue;
      const shape = this.callShape(ctx, a);
      if (!shape) continue; // 役の見込み無し / 取れない — a veto, not a price
      const key = shape.melds[shape.melds.length - 1].tiles.join(",");
      let v = memo.get(key);
      if (v === undefined) {
        v = this.evPushOf(obs, shape.rest, shape.melds);
        memo.set(key, v);
      }
      if (!Number.isFinite(v)) continue;
      if (v > bestValue) {
        bestValue = v;
        best = a;
      }
    }
    if (!best) return null;
    return bestValue - pass > this.ev!.params.callMargin ? best : null;
  }

  /**
   * The shape a call would leave — concealed rest, melds including the new one,
   * and its shanten — or null if the call is one we refuse.
   *
   * The refusals are the doctrine's, and they are the same two whichever
   * acceptance rule runs afterwards: a call whose tiles we do not hold (a
   * driver bug), and one that opens a hand with no route to a yaku at all.
   */
  private callShape(
    ctx: Ctx,
    a: Extract<Action, { t: "pon" | "chi" }>,
  ): { rest: Tile[]; melds: Meld[]; shanten: number } | null {
    const { obs } = ctx;
    const rest = [...obs.hand];
    for (const t of a.tiles) {
      const i = rest.lastIndexOf(t);
      if (i < 0) return null;
      rest.splice(i, 1);
    }
    const meld: Meld = {
      kind: a.t,
      who: obs.seat,
      fromWho: obs.seat,
      tiles: [...a.tiles, a.called].sort((x, y) => x - y),
      calledTile: a.called,
    };
    const melds = [...obs.melds[0], meld];

    if (
      this.dojo &&
      !this.hasYakuProspect(rest, melds, ctx.valueHonors, this.kuitan, publicUnseen(obs))
    ) {
      return null;
    }

    return { rest, melds, shanten: shanten(countsFromTiles(rest), melds.length, false) };
  }

  /** Shanten after taking the call, or null if the call is one we refuse. */
  private shantenAfterCall(
    ctx: Ctx,
    a: Extract<Action, { t: "pon" | "chi" }>,
  ): number | null {
    return this.callShape(ctx, a)?.shanten ?? null;
  }

  /**
   * Is there a yaku this open shape can plausibly still land on?
   *
   * GUIDANCE ONLY, and deliberately looser than the ledger's 後付け rule. That
   * rule judges the finished waiting hand with the real scorer (`yakuless` in
   * `DiscardInfo`, priced in `dojoCost`); this one only has to stop the CPU
   * from opening a hand with no route to a yaku at all. A concealed 役牌 pair —
   * the classic バック — passes here on purpose, PROVIDED the third tile is
   * still live: under the new reading the crime is the yakuless WAIT, not the
   * hopeful call, so refusing every バック would cost the policy hands the dojo
   * has no objection to — but a バック whose trigger is dead is not hopeful, it
   * is a route to nothing.
   *
   * TIGHTENED 2026-08-27 (with the owner's word, ranked wire logs as the
   * evidence): the 対々和 clause used to pass ANY chi-free shape — which made
   * every first pon self-justifying and produced open, cheap, yakuless hands
   * (15 of 36 arena pons had no other justification, one of them at 5向聴).
   * It now demands the concealed rest actually be pair-rich enough to finish
   * the triplet build.
   */
  private hasYakuProspect(
    rest: Tile[],
    melds: Meld[],
    valueHonors: Set<number>,
    kuitan: boolean,
    unseen: number[],
  ): boolean {
    const restTypes = rest.map(tileType);
    const meldTypes = melds.map((m) => m.tiles.map(tileType));

    // 役牌: already melded, or held concealed as a pair waiting on a LIVE third.
    for (const m of melds) {
      if (m.kind !== "chi" && valueHonors.has(tileType(m.tiles[0]))) return true;
    }
    const counts = new Map<number, number>();
    for (const ty of restTypes) counts.set(ty, (counts.get(ty) ?? 0) + 1);
    for (const [ty, n] of counts) {
      if (!valueHonors.has(ty)) continue;
      // A concealed triplet already IS the yaku; a pair needs its third live.
      if (n >= 3 || (n >= 2 && unseen[ty] > 0)) return true;
    }

    // 断幺九: nothing melded touches a yaochu, and the concealed part holds at
    // most one — which the discard that follows this call can throw away.
    if (kuitan) {
      const meldClean = meldTypes.every((ts) => ts.every((ty) => !isYaochu(ty)));
      if (meldClean && restTypes.filter(isYaochu).length <= 1) return true;
    }

    // 混一色/清一色: the melds sit in one suit and at most two concealed tiles
    // are stranded in another (honors are always welcome in a 混一色).
    const meldSuits = new Set(
      meldTypes.flat().filter((ty) => !isHonor(ty)).map((ty) => suitOfType(ty)),
    );
    if (meldSuits.size === 1) {
      const [suit] = [...meldSuits];
      const strays = restTypes.filter((ty) => !isHonor(ty) && suitOfType(ty) !== suit);
      if (strays.length <= 2) return true;
    }

    // 対々和: chi-free, AND the concealed rest can still supply the build.
    // `melds` already includes the call being judged, so the rest owes
    // (4 − melds) triplets + 1 pair; every one of those blocks realistically
    // grows from a pair already held, and ONE of them is allowed to arrive
    // later (a draw pairing a floater) — hence ≥ 4 − melds pair-or-better
    // types. A first pon wants three more pairs behind it, which is what a
    // hand actually built for 対々和 looks like.
    if (melds.every((m) => m.kind !== "chi")) {
      const pairTypes = [...counts.values()].filter((n) => n >= 2).length;
      if (pairTypes >= 4 - melds.length) return true;
    }

    return false;
  }

  // -------------------------------------------------------------------- kan

  /** HOOK, same reasoning as `chooseCall`; no subclass overrides it yet. */
  protected chooseKan(ctx: Ctx, legal: Action[]): Action | null {
    const { obs } = ctx;

    const ankan = legal.find((a) => a.t === "ankan");

    if (!this.dojo) {
      // Off the dojo leash, a kan is just value: take whichever is offered —
      // subject, with the core on, to it actually BEING value (unit D).
      if (ankan && ankan.t === "ankan") return this.kanWorthIt(ctx, ankan) ? ankan : null;
      return legal.find((a) => a.t === "kakan") ??
        legal.find((a) => a.t === "daiminkan") ?? null;
    }

    // 明槓 (大明槓・加槓) is a 禁じ手, and a bad idea besides: it opens a dora
    // for a table you cannot see. 暗槓 is allowed, but only 門前 + 聴牌 and
    // only when it leaves the wait alone.
    if (!ankan || ankan.t !== "ankan") return null;
    if (!ctx.closed || obs.shanten > 0) return null;
    if (this.kanChangesWait(ctx, ankan.type)) return null;
    return this.kanWorthIt(ctx, ankan) ? ankan : null;
  }

  /**
   * M15 unit D: is this 暗槓 worth taking? The vetoes above have already run —
   * this is the PRICE, and only the price: the post-kan rest (10 concealed
   * tiles, melds + 1, with the kan-dora reveal pending) against the best push
   * line of the hand as it stands.
   *
   * Answers "yes" — the incumbent verdict — whenever the core has no opinion:
   * the switch is off, this is not a 14-tile root (a 大明槓 claim has no
   * discard root to compare against, and unit D does not price one), the hand
   * is being abandoned, or either line came back non-finite. A kan the DP
   * cannot price is a kan the doctrine already cleared.
   */
  private kanWorthIt(ctx: Ctx, ankan: Extract<Action, { t: "ankan" }>): boolean {
    const { obs } = ctx;
    if (!this.ev?.params.calls || ctx.folding || !this.evRoot(obs)) return true;

    const rest: Tile[] = [];
    const four: Tile[] = [];
    for (const t of obs.hand) {
      if (tileType(t) === ankan.type && four.length < 4) four.push(t);
      else rest.push(t);
    }
    if (four.length < 4) return true; // not actually a concealed four

    const meld: Meld = {
      kind: "ankan",
      who: obs.seat,
      fromWho: obs.seat,
      tiles: four,
      calledTile: four[0],
    };
    // `kanDoraOn`: the new indicator is what a kan BUYS, and the DP prices it
    // by counting over the unseen pool. Without it the comparison could only
    // ever come out negative — a kan freezes the wait and hands the table a
    // dora. The channels of `hiddenInfoOf` do NOT ride along (`evHoldOf`
    // defaults to none): the rinshan tile comes before the draw they describe.
    const hold = this.evHoldOf(obs, rest, [...obs.melds[0], meld], true);
    const push = this.evOf(obs).bestPush;
    if (!Number.isFinite(hold) || !Number.isFinite(push)) return true;

    // THE TWO LINES HAVE TO PAY THE SAME TOLL (2026-08-30 review). `bestPush`
    // is a DISCARD root: every candidate it maxes over has already paid
    // `−costIn` for the tile it lets go. `mjev_eval_rest` is a 13-tile hold and
    // pays nothing — but the kan line does not skip the discard, it defers it
    // by one draw (kan → rinshan → discard). Comparing the two raw made every
    // kan look cheaper than every push by the price of a deal-in, which on a
    // loud table is the whole decision. The cheapest tile the post-kan hand
    // could let go is the floor of what that deferred discard will cost, so it
    // comes off the hold line before the margin is applied.
    let toll = Infinity;
    const priced = new Set<number>();
    for (const t of rest) {
      const ty = tileType(t);
      if (priced.has(ty)) continue;
      priced.add(ty);
      const c = this.dealinCostPts(ctx, t);
      if (c < toll) toll = c;
    }
    if (!Number.isFinite(toll)) toll = 0;

    return (hold - toll) - push > this.ev.params.callMargin;
  }

  /** Does declaring this ankan change what the hand is waiting on? */
  private kanChangesWait(ctx: Ctx, type: number): boolean {
    const { obs } = ctx;
    const before = new Set(obs.waits);

    const rest: Tile[] = [];
    let removed = 0;
    for (const t of obs.hand) {
      if (tileType(t) === type && removed < 4) removed++;
      else rest.push(t);
    }
    if (removed < 4) return true; // not actually a concealed four — refuse

    const counts = countsFromTiles(rest);
    const sh = shanten(counts, ctx.open + 1, ctx.closed);
    if (sh !== 0) return true;
    const after = ukeireTypes(counts, ctx.open + 1, ctx.closed, sh);
    if (after.length !== before.size) return true;
    return after.some((w) => !before.has(w));
  }
}
