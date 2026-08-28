// The frozen baseline seat — the 2026-08-25 epoch.
//
// On this date the original hand-written "h" agent was retired and the seat
// letter re-bound: "h" now builds a FROZEN COPY of the default 計算 ("k")
// seat — `AugmentedHeuristic` over `computedReads`, with the three weight
// objects below, and nothing else, ever. No ktune vector, no hand block, no
// riichi head, no consumer, no standings, no planner, no curriculum reaches
// it; the flag/table layers refuse the attempt. That is what makes it a
// BASELINE: an environment that later tuning cannot silently move, which is
// the property the M11 confound proved the old baseline lacked.
//
// The three objects are COMPLETE — every field of their interfaces, written
// out — and typed as the full interfaces on purpose: the live defaults they
// were copied from (`DEFAULT_WEIGHTS`, augmented's defaults,
// `DEFAULT_COMPUTED`) keep evolving with "k", and a partial here would
// silently inherit those changes through the constructors' merges. A field
// added to an interface later becomes a COMPILE error here, forcing the
// choice of frozen value to be made explicitly.
//
// THE PIN NEVER REGENERATES. `test/frozen_test.ts` fingerprints whole hanchan
// of this seat; drift there is a bug — in this file or in a shared code path —
// never a legitimate behaviour change. Numbers in `runs/` recorded before this
// date were measured against the OLD h and are not comparable forward.
//
// ONE sanctioned exception so far, 2026-08-27: the owner re-ruled the DOJO
// itself — 持ち点8000点未満 is judged at 終局 (buffer engages 南入以降 only)
// and the call gate's 対々和/バック clauses were tightened — and directed that
// the frozen seat play under the corrected rules too. The pins were
// re-captured that day; runs/ numbers straddling it are not comparable.

import type { AugmentedWeights } from "./augmented.ts";
import type { ComputedWeights } from "./computed.ts";
import type { HeuristicWeights } from "./heuristic.ts";

/** The base evaluation weights, as `DEFAULT_WEIGHTS` stood on 2026-08-25. */
export const FROZEN_HEURISTIC: HeuristicWeights = {
  shanten: 1000,
  ukeire: 12,
  ukeireType: 4,
  dora: 60,
  yakuhaiPair: 40,
  isolatedHonor: 6,
  danger: { 安全: 0, 危険度低: 30, 危険度中: 90, 危険度高: 200 },
  firstHonor: 4000,
  notenDora: 2500,
  katagari: 1500,
  yakulessTenpai: 4000,
  tsumogiriLock: 2500,
  foldEfficiency: 0.05,
  foldDanger: 10,
  // Added to HeuristicWeights 2026-08-27 (arena buffer neutralization); these
  // are the values the 8000-line buffer had when this seat froze.
  bufferTight: 0.35,
  bufferLow: 0.7,
  // Added to HeuristicWeights 2026-08-28 (生牌の役牌 surcharge). The frozen seat
  // predates the term and must not learn it: 0 is the value that keeps this
  // seat playing exactly the game it played on 2026-08-25.
  liveYakuhai: 0,
};

/** The Reads-consumption weights, as augmented.ts's defaults stood. */
export const FROZEN_AUGMENT: AugmentedWeights = {
  lambda: 0.25,
  floor: 0.5,
  drawRealize: 150,
  futureDora: 30,
  futureGenbutsu: 120,
  planKeep: 5000,
  planAdvance: 200,
  planRelock: 1.18,
};

/** The 計算 reader's weights, as `DEFAULT_COMPUTED` stood. `planner` frozen off. */
export const FROZEN_COMPUTED: ComputedWeights = {
  junmeBuckets: [6, 9, 12],
  tenpaiPrior: [
    [0.03, 0.12, 0.25, 0.38],
    [0.06, 0.2, 0.36, 0.5],
    [0.1, 0.3, 0.48, 0.62],
    [0.15, 0.4, 0.58, 0.72],
    [0.2, 0.45, 0.65, 0.78],
  ],
  tenpaiFloor: 0.25,
  yakuhaiTenpai: 0.08,
  tenpaiOtherRiichi: 1,
  tenpaiMeldDora: 1,
  shapePrior: {
    リャンメン: 0.45,
    カンチャン: 0.19,
    ペンチャン: 0.08,
    シャンポン: 0.16,
    タンキ: 0.12,
  },
  yakuhaiShanpon: 1.5,
  honitsuHot: 1.6,
  honitsuCold: 0.5,
  toitoiPair: 1.5,
  toitoiRun: 0.6,
  sujiHalfSurvive: 0,
  sujiFullSurvive: 0,
  doraPair: 1,
  doraBridge: 1,
  dealinScale: 0.065,
  waitNormalize: false,
  expWaitMass: 1.38,
  yakuFactor: { riichi: 1, open: 0.85, damaten: 0.6 },
  valueRiichi: 7000,
  valueDamaten: 4200,
  valueOpen: 3900,
  valueHonitsu: 7700,
  valueYakuhai: 1000,
  valuePerDora: 1600,
  valueDealer: 1.5,
  valueCap: 16000,
  valuePerHonba: 300,
  planner: false,
};
