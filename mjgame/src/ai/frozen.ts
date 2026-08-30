// The frozen baseline seat — the 2026-08-29 epoch.
//
// "h" builds a FROZEN COPY of the CHAMPION as it stood on 2026-08-29 — the
// vector that held a stable ~1600 rating on riichi.dev ranked play — with the
// weight objects below and nothing else, ever. No ktune vector, no hand
// block, no consumer, no standings, no planner, no curriculum reaches it; the
// flag/table layers refuse the attempt. That is what makes it a BASELINE: an
// environment that later tuning cannot silently move.
//
// EPOCH HISTORY. 2026-08-25: the original hand-written "h" agent was retired
// and the letter re-bound to a frozen copy of the DEFAULT 計算 seat
// (uncalibrated `DEFAULT_COMPUTED`, no riichi head, no sense). 2026-08-29,
// by the owner's word ("time to promote it to h agents again"): re-bound to
// the CHAMPION — the M10 computed calibration, the 最終形 riichi head, the
// 色読み sense trio, `liveYakuhai` 200 and `keepTriplet` 1. Two things this
// changes in kind: the frozen seat now carries a riichi head and a sense
// block (both complete objects below — the seat never references a live
// default), and it is a copy of `weights/champion.json`, not of the
// DEFAULT_* constants (which did not move; a bare "k" seat still plays the
// default game). The home-dojo buffer (`bufferTight`/`bufferLow`) keeps its
// home values: the arena's 1/1 are arena-only overrides and never promote.
//
// The objects are COMPLETE — every field of their interfaces, written out —
// and typed as the full interfaces on purpose: the live defaults keep
// evolving with "k", and a partial here would silently inherit those changes
// through the constructors' merges. A field added to an interface later
// becomes a COMPILE error here, forcing the choice of frozen value to be made
// explicitly. They were GENERATED from champion.json through the same merge
// functions the constructors use, never transcribed by hand.
//
// THE PIN NEVER REGENERATES. `test/frozen_test.ts` fingerprints whole hanchan
// of this seat; drift there is a bug — in this file or in a shared code path —
// never a legitimate behaviour change. Numbers in `runs/` recorded before an
// epoch were measured against the previous h and are not comparable forward.
// Sanctioned exceptions so far: 2026-08-27 (the owner re-ruled the dojo
// itself and directed the frozen seat play under the corrected rules),
// 2026-08-28 (a shared-path engine bug — the assessor's double-counted own
// tiles), and this epoch.

import type { AugmentedWeights } from "./augmented.ts";
import type { ComputedWeights } from "./computed.ts";
import type { HeuristicWeights } from "./heuristic.ts";
import type { RiichiWeights } from "./riichi.ts";
import type { SenseWeights } from "./sense.ts";

/** The base evaluation weights: `DEFAULT_WEIGHTS` as of 2026-08-29 plus the champion's two live terms. */
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
  // Home-dojo buffer values — the arena's 1/1 are overrides that never promote.
  bufferTight: 0.35,
  bufferLow: 0.7,
  // The champion's two live discipline terms (2026-08-29 epoch).
  liveYakuhai: 200,
  keepTriplet: 1,
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

/** The 計算 reader's weights: the champion's M10 calibration, fully resolved. `planner` frozen off. */
export const FROZEN_COMPUTED: ComputedWeights = {
  junmeBuckets: [
    6,
    9,
    12,
  ],
  tenpaiPrior: [
    [
      0.008476169622625462,
      0.034012918340418204,
      0.06484941973915648,
      0.086560433313427,
    ],
    [
      0.034256309775365285,
      0.10750323831971202,
      0.1830046191173242,
      0.25414381796772045,
    ],
    [
      0.09461867303987682,
      0.28548352714751346,
      0.3734662907461872,
      0.4462928558957687,
    ],
    [
      0.15026267689139716,
      0.3838139685285288,
      0.5304661888525666,
      0.6007422173445008,
    ],
    [
      0.2930658967925937,
      0.5065206587529641,
      0.6928527303236185,
      0.8059650060323581,
    ],
  ],
  tenpaiFloor: 0.25,
  yakuhaiTenpai: 0.2058306228886018,
  tenpaiOtherRiichi: 0.5711428365918875,
  tenpaiMeldDora: 0.9979491965375276,
  shapePrior: {
    リャンメン: 0.45,
    カンチャン: 0.20377052961446657,
    ペンチャン: 0.08961927548799176,
    シャンポン: 0.11660918664922965,
    タンキ: 0.07975393562248119,
  },
  yakuhaiShanpon: 1.1955729343288835,
  honitsuHot: 1.6,
  honitsuCold: 0.9287666603741758,
  toitoiPair: 1.5,
  toitoiRun: 0.6712136369056616,
  sujiHalfSurvive: 0.019962244038466775,
  sujiFullSurvive: 0.019832688654681924,
  doraPair: 1.178733675111213,
  doraBridge: 1.047007270580399,
  dealinScale: 0.065,
  waitNormalize: true,
  expWaitMass: 1.4863907046363236,
  yakuFactor: {
    riichi: 1,
    open: 0.867177745357772,
    damaten: 0.5891695829719623,
  },
  valueRiichi: 5227.359936315767,
  valueDamaten: 3032.201916638026,
  valueOpen: 1523.5218844471492,
  valueHonitsu: 1410.5283334621101,
  valueYakuhai: 291.33660177049876,
  valuePerDora: 2041.965063394323,
  valueDealer: 1.428763959930773,
  valueCap: 16266.435799411049,
  valuePerHonba: 300,
  planner: false,
};

/** The 最終形 riichi head (M12) as the champion carries it. */
export const FROZEN_RIICHI: RiichiWeights = {
  bias: 0.1,
  ev: 0,
  pwin: 0,
  value: 0,
  liveWaits: 0,
  waitTypes: 0,
  junme: 0,
  turnsLeft: 0,
  dora: 0,
  dealer: 0,
  oppRiichi: 0,
  kyotaku: 0,
  improvable: 0,
  tenpaiHeld: 0.5,
  holdShape: -1,
};

/** The 色読み sense trio as the champion carries it. */
export const FROZEN_SENSE: SenseWeights = {
  someRisk: 200,
  somePressure: 0.5,
  chiitoiTax: 500,
};
