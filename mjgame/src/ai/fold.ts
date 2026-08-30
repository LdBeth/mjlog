// M13 — the fold head: the push/fold gate as a learned decision rule.
//
// WHAT IT REPLACES. `HeuristicPolicy.computeFold` has always ended in one
// comparison:
//
//     push · gain  <  0.5 · pressure · risk
//
// where `push` is a four-step table over shanten plus 0.12 per dora (or the
// M11 outlook when a `hand` block is present), `pressure` counts riichi at 1
// and assessed furo threats at 0.5, and the two 順位効用 scales are 1 unless
// `--standings` is on. Every mangan the arena fed us was on one side or the
// other of that line. M11's lesson (see `CLAUDE.md`) is that computed
// information must reach a LEARNED DECISION RULE and not be squeezed through a
// scalar, and M12's riichi head is the only such rule that has shipped. This is
// the second.
//
// THE IDENTITY, and why it is structural (plan D3). The comparison above is
// rearranged, not replaced: `margin = push·gain − 0.5·pressure·risk` becomes
// FEATURE 0 of the vector, and `INIT_FOLD` is one linear layer with
// `w[margin] = −1`, everything else 0 and no bias, so
//
//     fold  ⇔  forward(x)[0] > 0  ⇔  −margin > 0  ⇔  margin < 0
//
// which is the old gate, bit for bit, for any finite feature vector. So
// `fold: {}` in a `--ktune` file is provably the incumbent seat, `fold` absent
// is the incumbent seat with no head built at all, and a trained head is a
// measurable step away from a known point rather than a new player.
//
// EVERY FEATURE IS FINITE BY CONSTRUCTION, and `foldVector` guards anyway: one
// NaN in the vector and `0 · NaN = NaN` poisons the init sum, which would break
// the equivalence silently instead of loudly (riichi.ts states the same rule).
// Absent information is 0 — no read is ever encoded as a sentinel.
//
// WHAT STAYS OUTSIDE THE HEAD, permanently. The two early-outs of
// `computeFold`: a declared riichi (the only legal discard is the drawn tile,
// so "fold" is meaningless) and a table with zero pressure (nothing to fold
// FROM). They are facts about the position, not judgements, and the head is
// consulted strictly inside the region they admit — the same discipline
// `consumer.ts` states for the discard core and `riichi.ts` for the gates.
//
// HOW IT IS FITTED. Not by imitation and not adversarially: `--foldcalib=PATH`
// plus `--fold-eps=X` plays a contextual-bandit lane in which the gate's verdict
// is FLIPPED with probability ε per decision, and `train/fold_fit.py` fits a
// doubly-robust estimate of "would the other action have paid better" off the
// round's own settlement (`deltas[0]/1000` — D7: the ledger and the game-end
// rule are recorded as data and never enter the objective; violations are
// minimised as a byproduct of long-term reward, never by per-decision shaping).

import type { Mlp, MlpSpec } from "./mlp.ts";
import { mlpForward, validateMlp } from "./mlp.ts";

/**
 * The FEATURE version. Bumping the list below bumps this, and a weight file
 * fitted against the old columns is then refused by `validateMlp` instead of
 * being fed the wrong numbers.
 */
export const FOLD_FV = 1;

/**
 * The frozen feature order. THIS ARRAY IS THE CONTRACT between the recorder,
 * the trainer and the live seat: a lane's header carries it verbatim, the
 * fitted block's first layer has one column per entry in this order, and
 * `foldVector` writes them in exactly this order. Reordering it is an `fv` bump.
 *
 * The groups, in order:
 *
 *   0-5   THE OLD GATE, taken apart. `margin` is its whole verdict as one
 *         number (< 0 ⇒ the incumbent folds) and the five parts it was built
 *         from are beside it, so the head can rebuild the comparison with
 *         different exchange rates instead of only shifting its threshold.
 *         `push` is POST-buffer (what the comparison actually used); `buffer`
 *         is the 8000点 scale that multiplied it, so the pre-buffer push is
 *         recoverable.
 *   6-13  THE HAND — `HandFacts` of the resting 13-tile shape, the same object
 *         M11's recorder writes, evaluated through the same memoised entry the
 *         discard score uses. Parameter-free counts.
 *   14-21 THE TABLE'S THREAT — how many opponents are loud, per seat, and what
 *         the 計算 reader thinks each one would collect (`expLoss`, in
 *         thousands). Zeros for a base heuristic seat with no reads.
 *   22-24 THE M11 OUTLOOK of that same shape, under whatever `hand` weights the
 *         seat carries (`DEFAULT_HAND` when it carries none, exactly as the
 *         hand recorder evaluates).
 *   25-30 THE SCOREBOARD — what the points would DO. The head gets the raw
 *         standings rather than only 順位効用's two scales.
 *   31-35 DEFENSIVE CAPACITY — can this hand actually fold? A hand of five
 *         genbutsu folds cheaply; a hand of five live middle tiles does not,
 *         and the incumbent gate cannot see the difference at all.
 *   36    色読み — the 感性 field pressure already added into `pressure`, broken
 *         out so the head can price the sensed half differently from declared
 *         riichi.
 */
export const FOLD_FEATURES: readonly string[] = [
  // 0-5 — the old gate, taken apart
  "margin",
  "push",
  "pressure",
  "gain",
  "risk",
  "buffer",
  // 6-13 — the hand
  "shanten",
  "ukeire",
  "ukeireTypes",
  "dora",
  "junme",
  "turnsLeft",
  "dealer",
  "open",
  // 14-21 — the table's threat
  "oppRiichi",
  "furoThreats",
  "threat0",
  "threat1",
  "threat2",
  "expLoss0",
  "expLoss1",
  "expLoss2",
  // 22-24 — the M11 outlook
  "pwin",
  "value",
  "ev",
  // 25-30 — the scoreboard
  "score",
  "leadTop",
  "leadBottom",
  "kyoku",
  "honba",
  "kyotaku",
  // 31-35 — defensive capacity
  "safeTypes",
  "lowTypes",
  "unassessedTypes",
  "genbutsuAll",
  "genbutsuMin",
  // 36 — 色読み
  "sensePressure",
] as const;

/** Input width of the head. 37. */
export const FOLD_INPUTS = FOLD_FEATURES.length;

/**
 * The named form of the vector — what `heuristic.ts` builds and what the
 * recorder would show a human. Field names are `FOLD_FEATURES`' entries, one
 * for one; `foldVector` is the only thing that knows the ORDER.
 */
export interface FoldFacts {
  /** `push·gain − 0.5·pressure·risk` — the incumbent gate's whole verdict. */
  margin: number;
  /** The push term AFTER the 8000点 buffer scale. */
  push: number;
  /** Threat volume: riichi 1, assessed furo 0.5, plus the 染め場 term. */
  pressure: number;
  /** 順位効用's gain scale (1 when 順位効用 is off). */
  gain: number;
  /** 順位効用's risk scale (1 when 順位効用 is off). */
  risk: number;
  /** `bufferScale` — the 持ち点8000未満 multiplier on `push`. */
  buffer: number;

  /** shanten of the resting 13-tile shape, clamped ≥ 0. */
  shanten: number;
  /** live copies the shape accepts (wait copies at tenpai). */
  ukeire: number;
  ukeireTypes: number;
  /** dora in hand + melds, aka included. */
  dora: number;
  junme: number;
  /** own draws remaining, ≈ wall/4. */
  turnsLeft: number;
  /** 1 when dealer. */
  dealer: number;
  /** open (non-ankan) melds. */
  open: number;

  /** opponents in declared riichi, 0..3. */
  oppRiichi: number;
  /** opponents the danger assessor rated a real furo threat, 0..3. */
  furoThreats: number;
  /** `threatOf` per opponent in relative order — 1 riichi, 0.5 furo, or the
   *  augmented seat's `tenpaiP`. */
  threat0: number;
  threat1: number;
  threat2: number;
  /** the 計算 reader's expected deal-in payment per opponent, ÷1000 (0 with no reads). */
  expLoss0: number;
  expLoss1: number;
  expLoss2: number;

  /** M11 `handOutlook` of the resting shape: P(this hand wins). */
  pwin: number;
  /** …what it collects, ÷1000. */
  value: number;
  /** …`pwin × value`, ÷1000. */
  ev: number;

  /** own score ÷1000. */
  score: number;
  /** own score minus the best other seat's, ÷1000 (negative ⇒ we are chasing). */
  leadTop: number;
  /** own score minus the worst other seat's, ÷1000. */
  leadBottom: number;
  /** 0 = 東1. */
  kyoku: number;
  honba: number;
  kyotaku: number;

  /** distinct own-hand types the assessor rates 安全. */
  safeTypes: number;
  /** …rates 危険度低. */
  lowTypes: number;
  /** …has no entry for at all (a quiet table assesses nothing). */
  unassessedTypes: number;
  /** distinct own-hand types that are 現物 against EVERY riichi opponent (0 if none declared). */
  genbutsuAll: number;
  /** the fewest 現物 any one riichi opponent leaves us (0 if none declared). */
  genbutsuMin: number;

  /** the 染め場 half of `pressure`, broken out. */
  sensePressure: number;
}

/**
 * The head as it rides in a ktune JSON: 37 in, 1 out. A `MlpSpec` and nothing
 * more — the fold head owns no arithmetic of its own beyond the sign test.
 */
export type FoldWeights = MlpSpec;

/**
 * THE IDENTITY HEAD — one linear layer, `w[margin] = −1`, every other weight 0,
 * bias 0, no activation. `forward(x)[0] = −margin`, so `> 0` is exactly
 * `margin < 0`: the incumbent gate, for any finite input, including the tie at
 * `margin === 0` (the old code's `<` is strict and `-0 > 0` is false).
 *
 * Frozen by construction: nothing here references a live default.
 */
export const INIT_FOLD: FoldWeights = {
  fv: FOLD_FV,
  layers: [{
    in: FOLD_INPUTS,
    out: 1,
    act: "none",
    w: FOLD_FEATURES.map((f) => (f === "margin" ? -1 : 0)),
    b: [0],
  }],
};

/**
 * A `--ktune` file's `fold` section, resolved. A partial is spread over
 * `INIT_FOLD` (so `{}` IS the identity, exactly as `mergeRiichi({})` is), and
 * the result is validated against the width and the feature version — a stale
 * or malformed block throws rather than silently mis-reading its own columns.
 *
 * THROWS (`validateMlp`'s contract): the CLI layer turns it into a die message.
 */
export function mergeFold(spec?: Partial<FoldWeights>): FoldWeights {
  return validateMlp(
    { ...INIT_FOLD, ...spec },
    { inputs: FOLD_INPUTS, outputs: 1, fv: FOLD_FV },
    "ktune.fold",
  );
}

/**
 * Facts → the float32 row the head reads, in `FOLD_FEATURES` order.
 *
 * `out` (length 37) is written and returned; without one a fresh array is
 * allocated. THE FINITE GUARD LIVES HERE and not at the call sites: every
 * feature is finite by construction today, and a future one that divides by a
 * count must not be able to poison the identity head's sum from a distance.
 */
export function foldVector(f: FoldFacts, out?: Float32Array): Float32Array {
  const x = out ?? new Float32Array(FOLD_INPUTS);
  if (x.length !== FOLD_INPUTS) {
    throw new Error(`fold 特徴量の長さ ${x.length} は ${FOLD_INPUTS} であるべきです`);
  }
  // Written positionally rather than by a name lookup: this is the hot path of
  // every fold decision, and the order below IS `FOLD_FEATURES` (the test
  // `fold_head_test.ts` asserts the two agree name by name).
  x[0] = fin(f.margin);
  x[1] = fin(f.push);
  x[2] = fin(f.pressure);
  x[3] = fin(f.gain);
  x[4] = fin(f.risk);
  x[5] = fin(f.buffer);
  x[6] = fin(f.shanten);
  x[7] = fin(f.ukeire);
  x[8] = fin(f.ukeireTypes);
  x[9] = fin(f.dora);
  x[10] = fin(f.junme);
  x[11] = fin(f.turnsLeft);
  x[12] = fin(f.dealer);
  x[13] = fin(f.open);
  x[14] = fin(f.oppRiichi);
  x[15] = fin(f.furoThreats);
  x[16] = fin(f.threat0);
  x[17] = fin(f.threat1);
  x[18] = fin(f.threat2);
  x[19] = fin(f.expLoss0);
  x[20] = fin(f.expLoss1);
  x[21] = fin(f.expLoss2);
  x[22] = fin(f.pwin);
  x[23] = fin(f.value);
  x[24] = fin(f.ev);
  x[25] = fin(f.score);
  x[26] = fin(f.leadTop);
  x[27] = fin(f.leadBottom);
  x[28] = fin(f.kyoku);
  x[29] = fin(f.honba);
  x[30] = fin(f.kyotaku);
  x[31] = fin(f.safeTypes);
  x[32] = fin(f.lowTypes);
  x[33] = fin(f.unassessedTypes);
  x[34] = fin(f.genbutsuAll);
  x[35] = fin(f.genbutsuMin);
  x[36] = fin(f.sensePressure);
  return x;
}

function fin(v: number): number {
  return Number.isFinite(v) ? v : 0;
}

/**
 * The head's verdict: FOLD when the single output is strictly positive.
 *
 * `scratch` (length 37) is filled with the vector and left filled, so a caller
 * that also has to RECORD the decision reads it back instead of building it
 * twice — which is what `computeFold` does with its per-seat `foldX`.
 */
export function decideFold(m: Mlp, f: FoldFacts, scratch: Float32Array): boolean {
  return mlpForward(m, foldVector(f, scratch))[0] > 0;
}

/**
 * One recorded fold decision, as the policy hands it to the sink. Everything
 * about the ROUND (seeds, indices, the settlement) is the writer's job — see
 * `foldcalib.ts`, which mirrors `handcalib.ts`'s division exactly.
 */
export interface FoldSample {
  /** The 37 features, in `FOLD_FEATURES` order. */
  x: number[];
  /** What the policy's rule (head or incumbent gate) said. */
  verdict: boolean;
  /** What was actually PLAYED — `verdict` unless the ε-flip fired. */
  taken: boolean;
  /** P(taken | this state) under the behaviour policy: `1−ε`, or `ε` when flipped. */
  p: number;
  /** 1 when the ε-flip fired on this decision. */
  flipped: boolean;
  /**
   * True on a DRAW decision (`obs.drawn !== null`), false on a claim decision.
   * The two are different questions — a draw decision chooses a discard, a
   * claim decision chooses whether to call — and only the policy can tell them
   * apart, so it stamps the answer here rather than leaving the recorder to
   * guess from the feature row.
   */
  turn: boolean;
}
