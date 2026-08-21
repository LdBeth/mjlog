// 計算 — the combinatorial reader. Exact counting over PUBLIC facts, nothing else.
//
// The agent is three layers: 計算 (this file), 感性 (a small learned parameter
// vector, later) and 規律 (the compliance filter, already in `heuristic.ts`).
// This layer fills the SAME `Reads` shape `oracleReads` fills with truth — but
// from an `Observation` alone, so a seat driven by it may sit at any table,
// including a human's.
//
// WHAT IS ALLOWED HERE. Counting, and only counting: suji, kabe, visible
// copies, genbutsu, meld contents, riichi declarations, junme, wall remaining,
// the unseen pool. Every one of those is a FACT on the table that any player
// may count, and the arithmetic over them is exact.
//
// WHAT IS FORBIDDEN HERE, permanently: 河読み — inferring hidden state from an
// opponent's BEHAVIOR. No tedashi/tsumogiri timing, no discard-order patterns,
// no tenpai classifier trained on river sequences, no "he is pushing so he must
// be tenpai". The project's 雀鬼流 constitution rules that out, and nothing in
// this module may reach for it. Note the deliberate consequence: P(tenpai) here
// is a BASE RATE by (meld count, junme) — a published rate, the same number for
// every opponent in that state — and never a reading of this particular player.
//
// WHERE THE JUDGMENT LIVES. Exact counting still has to be turned into a
// number, and every constant that does the turning is gathered in
// `ComputedWeights` — the future 感性 tuning surface. Nothing in the code below
// hides a magic number.
//
// HOW THE ESTIMATE IS BUILT. The C1 surrogate factors:
//
//     P(deal in to opponent i with type ty)
//       = P(i is tenpai) × P(i can ron at all) × P(i is waiting on ty | tenpai)
//
// The third factor is a DISTRIBUTION over the 34 types, not 34 independent
// verdicts (M10b): a tenpai hand rons some fixed small number of types —
// `expWaitMass`, measured at about 1.4 — however many shapes the counting leaves
// standing, so the shape masses are normalized by their own row total. The
// M10a form multiplied each type by a single population-wide constant
// (`dealinScale`) instead, which is the same thing only on the average board;
// `waitNormalize` selects between them and defaults to the old behavior.
//
// The first factor is the base-rate table. The second is wait-shape survival:
// enumerate the five primitive shapes that could be waiting on `ty`, kill the
// ones public facts REFUTE (genbutsu ⇒ furiten ⇒ all dead; four copies visible
// ⇒ シャンポン/タンキ dead; a dead bridging tile ⇒ that run shape dead; a suji
// end in their genbutsu ⇒ that リャンメン holding dead), and weight the
// survivors by how many ways the opponent could physically be holding the tiles
// each shape needs, given the unseen counts. The kill rules are exactly the ones
// `mjrender/danger.ts` uses for its `当たり形:` note (v0.8.0) — that module
// returns an ORDINAL level, and what a risk term needs is a probability, so the
// binary "shape survives" test is extended here into a graded count of holdings.
// The スジ kill is graded further (M10b): 半スジ (one bridge end passed, the
// other リャンメン still live) and 全スジ (every リャンメン that reaches the tile
// refuted) are booked into separate rows, so a fit may price the residue each
// leaves instead of discarding both. 現物 remains absolute: it is furiten, and
// furiten is not a matter of degree.
//
// MELD CONTENTS ARE PUBLIC. The reads in `MeldRead` (染め手模様 / トイトイ模様 /
// 役牌副露) condition on the tiles an opponent has turned FACE UP. That is a
// statement about their CURRENT public state — the same statement any observer
// would make about anyone showing those melds — and it is emphatically not a
// per-opponent sequential inference: nothing accumulates across turns, nothing
// remembers what THIS player did earlier, and the same melds in front of a
// different seat produce the same numbers. Compare the banned reading, which
// needs a history: "he cut ⑨筒 at 4巡 手出し, so he is going for a manzu flush".
// What is read here is only "two of his melds are manzu" — face up on the table.

import type { Meld } from "mjrender/model.ts";
import { doraFromIndicatorType, rankOfType, suitOfType, tileType } from "mjrender/tiles.ts";
import type { Observation } from "../observe.ts";
import { shapeMasses } from "../kernel.ts";
import { zeros34 } from "../tiles.ts";
import type { Reads, ReadsProvider } from "./augmented.ts";
import { publicUnseen, valueHonorsOf } from "./planner.ts";

/** The five primitive wait shapes, named as `mjrender/danger.ts` names them. */
export type WaitShape = "リャンメン" | "カンチャン" | "ペンチャン" | "シャンポン" | "タンキ";

export const WAIT_SHAPES: readonly WaitShape[] = [
  "リャンメン",
  "カンチャン",
  "ペンチャン",
  "シャンポン",
  "タンキ",
];

// ---------------------------------------------------------------------------
// the tuning surface
// ---------------------------------------------------------------------------

/**
 * Every judgment constant this module applies to its exact counts, in one
 * object. The counting itself has no free parameters; these are what turn a
 * count into a probability or a point figure, and they are what 感性 will later
 * learn. Nothing here reads behavior — a rate is not a read.
 */
export interface ComputedWeights {
  // ---- P(tenpai): the factual base-rate table -----------------------------
  /**
   * Upper bounds of the junme buckets, ascending. `[6, 9, 12]` reads as
   * 序盤 (≤6巡), 中盤 (7–9), 終盤 (10–12), 大詰 (13巡以降) — four columns.
   */
  junmeBuckets: readonly number[];
  /**
   * P(tenpai) by [meld count 0..4][junme bucket]. A PRIOR: the published rate at
   * which a hand in that public state is tenpai, applied identically to every
   * opponent. Rows are the meld count (ankan included — a kan is a committed
   * hand however closed it leaves the wait), clamped to 0..4.
   */
  tenpaiPrior: readonly (readonly number[])[];
  /**
   * Exposure threshold for the `tenpaiP` FIELD (not for `dealinP`, which uses
   * the raw prior). `AugmentedHeuristic.pressureOf` reads `tenpaiP` linearly and
   * `shouldFold` treats any pressure above zero as a table worth defending
   * against, so handing it three raw base rates would put the seat in a
   * permanent half-fold on a silent table. Below this the prior carries no
   * information the base heuristic does not already have, so it is reported as
   * ZERO. At or above it, the RAW prior is reported — a gate, not a rescale.
   *
   * The rescale this replaced (`(p − floor)/(1 − floor)`) was a bug of slope,
   * not of level: it stretched everything above the floor onto 0..1, so the
   * pressure input that drives the fold trigger arrived distorted — a 大詰 menzen
   * table at its true .38 came through as ≈.17, less than half its weight, and
   * the seat under-defended exactly the quiet late board that historically
   * killed it. A threshold decides WHETHER to listen; it has no business also
   * changing what it heard. At the tuner's upper bound of 1 the field reports
   * declared riichi and nothing else, which is still a meaningful setting.
   */
  tenpaiFloor: number;
  /**
   * Additive bump to the tenpai prior of an opponent showing a 役牌 triplet meld
   * (ankan included). Such a hand has its yaku locked from the call, so it is
   * driving straight at tenpai with nothing left to arrange — the dojo's ban on
   * 後付け makes that read stronger here than at a normal table. Applied after
   * the (副露数 × 巡目) table lookup and clamped to that column's 副露4 cell, so
   * the bump can sharpen the row but never outrun the table's own ceiling.
   */
  yakuhaiTenpai: number;
  /**
   * Multiplier on a SILENT opponent's tenpai prior when some OTHER seat has
   * declared riichi (ourselves included). A declaration changes what everyone
   * else at the table is doing — most of them stop pushing — so the population
   * of quiet hands after a riichi is not the population before it. This is a
   * statement about the table's public state, identical for every seat in it,
   * and it is not a read of the individual: the seat that declared is already
   * tenpai by rule, and this weight never touches them. Below 1 it suppresses,
   * above 1 it sharpens; 1 = the M10a behavior, and is the default.
   */
  tenpaiOtherRiichi: number;
  /**
   * Multiplier applied ONCE PER dora visible in this opponent's own melds
   * (`tenpaiMeldDora ** melded dora`). A hand that has called dora into the open
   * is a hand that has committed to a price and is driving at tenpai; a hand
   * showing melds with nothing in them is likelier to still be shopping. Face-up
   * tiles only. 1 = the M10a behavior, and is the default.
   */
  tenpaiMeldDora: number;

  // ---- P(waiting on this type | tenpai): wait-shape survival ---------------
  /**
   * Prior share of tenpai hands waiting through each shape, before the public
   * facts kill and grade them. Roughly the standard riichi wait distribution;
   * it sums to 1 by construction, and only the ratios matter.
   */
  shapePrior: Readonly<Record<WaitShape, number>>;
  /**
   * Multiplier on the シャンポン weight of a value honor (round wind, that
   * seat's wind, the three dragons). A yakuhai pair is HELD — that is a fact
   * about the ruleset, not about the player.
   */
  yakuhaiShanpon: number;
  /**
   * Multiplier on every surviving shape of a tile IN the read flush suit, or on
   * any honor, against an opponent whose melds say 染め手模様. Their hand is
   * concentrated there, so a far larger share of it is waiting tiles.
   */
  honitsuHot: number;
  /**
   * Multiplier on every surviving shape of a tile in one of the OTHER two suits
   * against the same opponent. The counterpart of `honitsuHot`: a flush hand
   * holds almost nothing outside its suit, so those tiles are close to safe —
   * this is the single largest standard-defense discount meld counting gives.
   */
  honitsuCold: number;
  /**
   * Multiplier on シャンポン/タンキ against an opponent whose melds say トイトイ
   * 模様 (every meld a pon/kan). A toitoi hand waits with pairs, so those two
   * shapes carry essentially the whole risk.
   */
  toitoiPair: number;
  /**
   * Multiplier on リャンメン/カンチャン/ペンチャン against the same opponent — a
   * hand of triplets is not holding run shapes. Damped rather than killed: the
   * read is a 模様, and a 対々 that quietly turned into a normal hand exists.
   */
  toitoiRun: number;
  /**
   * 半スジ: the share of a リャンメン holding that SURVIVES its own スジ proof
   * while the tile's other suji end is still live.
   *
   * The proof is exact and it is not in doubt — a hand holding 67筒 and waiting
   * 5-8筒 that has already discarded 8筒 is furiten. What is in doubt is the
   * WAIT: 5筒 with only 8筒 passed is a half-suji tile, and the 34筒 holding on
   * the other side is untouched, so the tile is still a リャンメン tile. This
   * weight covers the residue the binary kill throws away: the reasons a player
   * ends up holding the refuted shape anyway (a 振聴立直, a shape that changed
   * after the discard, a call that shifted the hand). 0 is the M10a behavior —
   * refuted means gone — and is the default.
   */
  sujiHalfSurvive: number;
  /**
   * 全スジ: the same residue when EVERY リャンメン that could reach the tile is
   * refuted — both ends passed for a 4/5/6, the single end for a 1/2/3/7/8/9.
   *
   * Kept separate from `sujiHalfSurvive` because the two are different claims.
   * Half suji says "one of the two リャンメン holdings is out"; full suji says
   * "no リャンメン at all", which is the classic 全スジ read and is the stronger
   * statement — a fit is free to price them apart. 0 is the M10a behavior.
   */
  sujiFullSurvive: number;
  /**
   * Multiplier on シャンポン/タンキ when the tile TYPE is a ドラ. Dora are held:
   * a player who draws a second one keeps the pair rather than breaking it, so
   * the two shapes that wait with a pair are over-represented on dora types.
   * The indicators are face up, so this is public. 1 = no conditioning.
   */
  doraPair: number;
  /**
   * Multiplier on the リャンメン mass of a holding whose BRIDGE contains a dora
   * — the 67筒 that waits 5-8筒 when 6筒 or 7筒 is dora. Such a shape is kept
   * where a plain one is broken up, and it is public for the same reason. It is
   * applied to the live holdings only (the suji-refuted residue above is already
   * a discount on a shape the facts argue against). 1 = no conditioning.
   */
  doraBridge: number;
  /**
   * Calibration of the shape sum onto an actual probability, for the
   * UN-NORMALIZED path only (`waitNormalize: false`). A fully live no-suji
   * middle tile scores ≈0.92 raw; at 0.065 that becomes ≈6%, which is where a
   * no-suji 456 sits against a riichi.
   *
   * It was doing two jobs at once, and M10b takes one of them away. A tenpai
   * hand waits on a FIXED number of tile types — about 1.4 in this population —
   * however many shapes the counting leaves alive, so the shape sum has to be
   * turned into a distribution OVER types before it is a probability. This
   * constant could only ever do that on average, across every board at once; a
   * board with many live types therefore came out systematically hot and a board
   * with few came out cold. `waitNormalize` does it per hand instead. What is
   * left for this constant is the honest half of its old job: the overall level
   * at which a consumer should take the model seriously.
   */
  dealinScale: number;
  /**
   * Normalize the wait masses PER HAND (M10b): report
   * `expWaitMass × m(ty) / Σ_live m(ty)` instead of `m(ty) × dealinScale`.
   *
   * The mahjong statement: an opponent who is tenpai is waiting on some tiles,
   * and the counting's job is to say WHICH — a ranking over types, not 34
   * independent verdicts. Dividing by the row's own total makes that explicit;
   * `dealinScale` had to guess the divisor once for the whole population.
   *
   * DEFAULT FALSE so that the shipped seat is bit-for-bit the seat M10a shipped;
   * turning it on is a level change everywhere at once, and the level is what
   * every consumer term was tuned against.
   */
  waitNormalize: boolean;
  /**
   * The number of tile TYPES a tenpai hand actually rons, on average — the total
   * probability mass the normalized row carries. Measured at 1.38 against the
   * `h` bot population (a リャンメン is 2 types, a タンキ is 1, a hand with two
   * shapes alive is more). Consumed only when `waitNormalize` is on, where it
   * IS the level: the row sums to it by construction, so `dealinScale` does not
   * multiply it a second time.
   */
  expWaitMass: number;
  /**
   * P(the hand can actually declare a ron) by public class: a declared riichi
   * always can; an open hand nearly always does (the dojo bans 後付け, so an
   * open hand without a yaku is not a hand anyone builds); a closed silent
   * tenpai often cannot, which is part of why it stayed silent.
   */
  yakuFactor: { riichi: number; open: number; damaten: number };

  // ---- what a deal-in costs: the static value model ------------------------
  /** Base ron payment of a declared riichi (riichi + 一発/裏 potential folded in). */
  valueRiichi: number;
  /** Base ron payment of a closed, silent tenpai. */
  valueDamaten: number;
  /** Base ron payment of an open hand. */
  valueOpen: number;
  /**
   * Base ron payment of an open hand whose melds say 染め手模様. Roughly double
   * the flat open base: ホンイツ alone is 2翻 open, it travels with 役牌 and
   * トイトイ, and the suit concentration drags dora along with it. Pricing such a
   * hand at the same 3900 as a タンヤオ chi was the value model's largest error.
   */
  valueHonitsu: number;
  /**
   * Added per 役牌 triplet meld visible (ankan included). A 白 pon is a han that
   * is already scored; a タンヤオ chi is not. One bump per triplet, so a double
   * yakuhai prices twice.
   */
  valueYakuhai: number;
  /** Added per dora visible in their melds, and per dora in the winning tile. */
  valuePerDora: number;
  /** Multiplier when the opponent is the dealer. */
  valueDealer: number;
  /** Ceiling on the modelled payment. */
  valueCap: number;
  /**
   * 本場 surcharge per stick, paid by the discarder on top of the hand's points.
   *
   * NOT a judgment constant: `score.ts#ronValue` charges `honba × 300` and the
   * rules say 300, so this is the one number in the model that is simply CORRECT
   * — and until M10b the model was the only place at the table that did not
   * charge it. It is named and mergeable anyway, because that is what makes the
   * correction measurable: setting it to 0 reproduces the pre-M10b seat exactly,
   * decision for decision, which is how the change was proved to move nothing
   * except hands played on a 本場. It is added AFTER `valueCap`, exactly as the
   * scorer adds it after the hand's own limit.
   */
  valuePerHonba: number;

  // ---- engagement (a switch, not a tuning constant) -----------------------
  /**
   * Engage the C7 planner for this seat. Set it and the provider reports
   * `planner: true` alongside the availability fields; leave it off and the seat
   * runs the augmented terms alone. Two eval arms, one provider.
   */
  planner: boolean;
}

export const DEFAULT_COMPUTED: ComputedWeights = {
  junmeBuckets: [6, 9, 12],
  tenpaiPrior: [
    [0.03, 0.12, 0.25, 0.38], // 門前 (副露0)
    [0.06, 0.20, 0.36, 0.50], // 副露1
    [0.10, 0.30, 0.48, 0.62], // 副露2
    [0.15, 0.40, 0.58, 0.72], // 副露3
    [0.20, 0.45, 0.65, 0.78], // 副露4
  ],
  tenpaiFloor: 0.25,
  yakuhaiTenpai: 0.08,
  tenpaiOtherRiichi: 1,
  tenpaiMeldDora: 1,
  shapePrior: {
    "リャンメン": 0.45,
    "カンチャン": 0.19,
    "ペンチャン": 0.08,
    "シャンポン": 0.16,
    "タンキ": 0.12,
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

/**
 * Defaults, then the override, with the nested records merged field-wise.
 * Exported because a calibration file's header must state the vector its
 * predictions were made with, and "the partial the CLI parsed" is not that.
 */
export function mergeComputed(w?: Partial<ComputedWeights>): ComputedWeights {
  // Nested records are merged field-wise: a partial override that dropped a
  // shape would score every hand NaN, the same trap `HeuristicPolicy` documents.
  return {
    ...DEFAULT_COMPUTED,
    ...w,
    shapePrior: { ...DEFAULT_COMPUTED.shapePrior, ...w?.shapePrior },
    yakuFactor: { ...DEFAULT_COMPUTED.yakuFactor, ...w?.yakuFactor },
    tenpaiPrior: mergeTable(DEFAULT_COMPUTED.tenpaiPrior, w?.tenpaiPrior),
  };
}

/**
 * The tenpai table, merged CELL BY CELL (M10b). The twenty cells are twenty
 * independent parameters — the measured error in them is a matter of SHAPE, not
 * of level, so a fit moves some and leaves others — and a whole-array override
 * makes stating one of them mean restating all twenty, with a dropped row
 * silently reading as `undefined ?? 0`. A missing row falls back to the default
 * table's row of the same index, a missing cell to the default cell.
 */
function mergeTable(
  def: readonly (readonly number[])[],
  ov?: readonly (readonly number[])[],
): number[][] {
  if (!ov) return def.map((r) => [...r]);
  const out: number[][] = [];
  for (let r = 0; r < Math.max(def.length, ov.length); r++) {
    const d = def[r] ?? def[def.length - 1];
    const o = ov[r];
    const row: number[] = [];
    for (let c = 0; c < Math.max(d.length, o?.length ?? 0); c++) {
      row.push(o?.[c] ?? d[c] ?? d[d.length - 1]);
    }
    out.push(row);
  }
  return out;
}

// ---------------------------------------------------------------------------
// P(tenpai): the base rate
// ---------------------------------------------------------------------------

/**
 * The factual tenpai rate for a hand in this PUBLIC state. A declared riichi is
 * tenpai by the rules of the game, so it is 1 — the only place this function
 * knows anything about a specific opponent.
 *
 * `yakuhai` is the one content condition the table takes: a visible 役牌 triplet
 * meld splits the (副露数, 巡目) cell into two populations — hands with a yaku
 * already locked, which are simply racing to tenpai, and hands still shopping for
 * one. It is a state, not a history; see `yakuhaiTenpai`.
 */
/**
 * Which cell of the (副露数 × 巡目) table a public state lands in. Split out so
 * the calibration recorder can name the cell it predicted from — a reliability
 * table indexed by anything else would be measuring the wrong partition.
 */
export function tenpaiCellOf(
  w: ComputedWeights,
  melds: number,
  junme: number,
): { row: number; col: number } {
  const row = Math.max(0, Math.min(w.tenpaiPrior.length - 1, melds));
  let col = 0;
  for (const b of w.junmeBuckets) if (junme > b) col++;
  return { row, col: Math.min(col, w.tenpaiPrior[row].length - 1) };
}

export function tenpaiPriorOf(
  w: ComputedWeights,
  melds: number,
  junme: number,
  riichi: boolean,
  yakuhai = false,
  otherRiichi = false,
  meldDora = 0,
): number {
  if (riichi) return 1;
  const cell = tenpaiCellOf(w, melds, junme);
  const row = w.tenpaiPrior[cell.row];
  const col = cell.col;
  const p0 = row[col] ?? 0;
  let p = p0;
  if (yakuhai) {
    // The table's own 副露4 row is the ceiling: a yaku in hand cannot make an
    // opponent likelier to be tenpai than the most committed public state there
    // is.
    const ceiling = w.tenpaiPrior[w.tenpaiPrior.length - 1][col] ?? 1;
    p = Math.min(p0 + w.yakuhaiTenpai, Math.max(p0, ceiling));
  }
  // M10b: two multipliers over the table cell, both functions of public state.
  // Applied AFTER the 役牌 bump and its ceiling, because that pair is a
  // statement about which population the cell describes and these two are
  // statements about the table the cell is being read at. Both default to 1, so
  // the whole clause is `p × 1 × 1` on the shipped vector.
  if (otherRiichi) p *= w.tenpaiOtherRiichi;
  if (meldDora > 0 && w.tenpaiMeldDora !== 1) p *= Math.pow(w.tenpaiMeldDora, meldDora);
  return p < 0 ? 0 : p > 1 ? 1 : p;
}

// ---------------------------------------------------------------------------
// what the melds say: the public content read
// ---------------------------------------------------------------------------

/**
 * The reading of one opponent's FACE-UP melds. Every field is a function of the
 * tiles currently on the table in front of them and of the ruleset (which honors
 * are yakuhai for that seat) — evaluate it twice on the same melds and it gives
 * the same answer, evaluate it on another seat's identical melds and it gives the
 * same answer. That is what keeps it a population prior conditioned on a public
 * event rather than a per-opponent inference: there is no history in it.
 *
 * MIRRORED, NOT IMPORTED. `mjrender/state.ts#furoThreats` computes the same three
 * reads for `mjrender/danger.ts` (v0.8.0), but over its own `GameState`, which an
 * `Observation` is not — so the thresholds are reproduced here rather than
 * reused: activation at 2 open melds or 1 yakuhai triplet; 染め手 requires every
 * meld to be honors-or-one-common-suit with at least one meld IN that suit;
 * トイトイ requires ≥2 open melds, all of them triplets. ONE deliberate
 * divergence: mjrender additionally requires ≤1 tile of the suit in the player's
 * river, and that is a river read — counting what they chose to throw away — so
 * it is left on the other side of the constitution's line. The cost is that the
 * flush read fires a little more often here than it does in the commentary tool.
 */
export interface MeldRead {
  /** Number suit of a 染め手模様 read, or null. Honors are hot alongside it. */
  honitsuSuit: "m" | "p" | "s" | null;
  /** トイトイ模様: every open meld is a pon/kan, and there are at least two. */
  toitoi: boolean;
  /** Types of the 役牌 triplet melds they are showing (ankan included). */
  yakuhai: ReadonlySet<number>;
  /** Open (non-ankan, non-nuki) meld count — what the reads activate on. */
  open: number;
}

const NO_READ: MeldRead = {
  honitsuSuit: null,
  toitoi: false,
  yakuhai: new Set<number>(),
  open: 0,
};

/**
 * Read a seat's melds. An ankan leaves the hand closed, so it does not activate
 * anything by itself, but its four tiles are face up and count as evidence for
 * every read they touch — exactly the split `furoThreats` makes.
 */
export function meldReadOf(
  melds: readonly Meld[],
  valueHonors: ReadonlySet<number>,
): MeldRead {
  const real = melds.filter((m) => m.kind !== "nuki");
  const open = real.filter((m) => m.kind !== "ankan");
  if (real.length === 0) return NO_READ;

  const isTriplet = (m: Meld) => m.kind !== "chi";
  const yakuhai = new Set<number>();
  for (const m of real) {
    if (isTriplet(m) && valueHonors.has(tileType(m.tiles[0]))) yakuhai.add(tileType(m.tiles[0]));
  }
  // Activation: two open melds, or one open yakuhai triplet — a hand that showed
  // its yaku on the first call is already committed.
  const openYakuhai = open.some((m) => isTriplet(m) && valueHonors.has(tileType(m.tiles[0])));
  if (open.length < 2 && !openYakuhai) {
    return { honitsuSuit: null, toitoi: false, yakuhai, open: open.length };
  }

  let honitsuSuit: "m" | "p" | "s" | null = null;
  if (open.length >= 2) {
    const suits = new Set(
      real.flatMap((m) => m.tiles.map((t) => suitOfType(tileType(t)))).filter((x) => x !== "z"),
    );
    if (suits.size === 1) honitsuSuit = [...suits][0] as "m" | "p" | "s";
  }
  const triplets = open.filter(isTriplet).length;
  return {
    honitsuSuit,
    toitoi: triplets >= 2 && triplets === open.length,
    yakuhai,
    open: open.length,
  };
}

// ---------------------------------------------------------------------------
// P(waiting on this type | tenpai): wait-shape survival
// ---------------------------------------------------------------------------

/** What the survival count needs to know about one opponent, all of it public. */
export interface WaitContext {
  /** Copies of each type not visible to the observer (`publicUnseen`). */
  unseen: readonly number[];
  /** Types this opponent provably cannot ron: their own discards, plus — once
   *  they have declared riichi — everything discarded by anyone since. */
  genbutsu: ReadonlySet<number>;
  /** Honor types that are yakuhai for this opponent. */
  valueHonors: ReadonlySet<number>;
  /**
   * What this opponent's face-up melds say (`meldReadOf`). Optional: absent means
   * "no content modifier", which is what a menzen opponent gets. It is a property
   * of their CURRENT public state, so it belongs to this opponent's row of
   * `dealinP` and to no one else's — a per-state modifier, not a per-player one.
   */
  read?: MeldRead;
  /**
   * Dora count per tile TYPE from the revealed indicators (aka excluded), as
   * `doraOfType` builds it. Optional: absent means "no dora conditioning", which
   * is what a caller that does not care about `doraPair`/`doraBridge` gets — the
   * indicators are face up, so supplying it reveals nothing.
   */
  dora?: readonly number[];
}

/** Ways to pick `k` copies from `n` unseen ones (n ≤ 4, k ≤ 2). */
function choose2(n: number): number {
  return n < 2 ? 0 : (n * (n - 1)) / 2;
}

/**
 * The COUNTING half of the wait model, before a single weight touches it: per
 * shape, the number of ways the opponent could be holding the tiles that shape
 * needs, as a plain integer numerator over the shape's fixed denominator
 * (リャンメン /32, カンチャン・ペンチャン /16, シャンポン /6, タンキ /4).
 *
 * SPLIT ON PURPOSE (M10a). Everything a public fact can PROVE — 現物 furiten,
 * スジ, 壁, visible copies — is folded in here, and nothing else is: no prior,
 * no multiplier, no calibration. So this object is the complete, parameter-free
 * summary of one (opponent, tile type) cell, and `combineShapes` below is a pure
 * closed-form function of it and of `ComputedWeights`. That is what lets the
 * calibration recorder cache four small integers per cell and re-evaluate the
 * whole model later against different weights, WITHOUT replaying the game and
 * without a second implementation of this arithmetic: the recorder, the live
 * seat and the offline fit all go through the two functions below.
 */
export interface ShapeBase {
  /** Σ over the LIVE リャンメン holdings of u(a)·u(b) — 0..32. */
  ryanmen: number;
  /**
   * The part of `ryanmen` whose BRIDGE contains a dora (M10b) — 0..32, and
   * always ≤ `ryanmen`, because it is a subset of the same holdings. Split out
   * rather than flagged so that a board where one bridge is dora-bearing and the
   * other is not prices the two apart; `doraBridge` multiplies this share alone.
   */
  ryanmenDora: number;
  /**
   * The リャンメン holdings スジ refutes while the tile's OTHER end is still
   * live — 半スジ, 0..16 (a half-suji tile has exactly one refuted holding).
   * Zero in M10a's arithmetic, and zero again whenever `sujiHalfSurvive` is 0;
   * carried here so a fit can price the residue without replaying a game.
   */
  ryanmenHalf: number;
  /**
   * The リャンメン holdings スジ refutes when EVERY holding that could reach the
   * tile is refuted — 全スジ, 0..32. See `sujiFullSurvive`.
   */
  ryanmenFull: number;
  /** カンチャン: u(ty−1)·u(ty+1) — 0..16. */
  kanchan: number;
  /** ペンチャン: u·u of the single holding that reaches `ty` — 0..16. */
  penchan: number;
  /** シャンポン: C(u(ty), 2) — 0, 1, 3 or 6. */
  shanpon: number;
  /** タンキ: u(ty) — 0..4. */
  tanki: number;
}

/** All five shapes refuted: 現物, or a type nothing could physically reach. */
const zeroBase = (): ShapeBase => ({
  ryanmen: 0,
  ryanmenDora: 0,
  ryanmenHalf: 0,
  ryanmenFull: 0,
  kanchan: 0,
  penchan: 0,
  shanpon: 0,
  tanki: 0,
});

/**
 * What `combineShapes` needs to know about the (opponent, type) pair beyond the
 * counts: the four CLASS facts that select a multiplier. Each is a statement
 * about public state (the ruleset's value honors, the opponent's face-up melds,
 * the revealed dora indicators), never about behavior — see the module header.
 */
export interface ShapeFlags {
  /** `ty` is a 役牌 for this opponent, so a held pair of it is likelier. */
  valueHonor: boolean;
  /** 染め手模様 suit read from their melds, or null. */
  honitsuSuit: "m" | "p" | "s" | null;
  /** トイトイ模様. */
  toitoi: boolean;
  /**
   * `ty` is a ドラ (indicator-derived; aka are tile ids, not types, so they
   * cannot enter here). A dora is held rather than discarded, which is what
   * `doraPair` prices.
   */
  doraType: boolean;
}

/** The flags a `WaitContext` implies for one type. */
export function shapeFlagsOf(ty: number, ctx: WaitContext): ShapeFlags {
  return {
    valueHonor: ctx.valueHonors.has(ty),
    honitsuSuit: ctx.read?.honitsuSuit ?? null,
    toitoi: ctx.read?.toitoi ?? false,
    doraType: (ctx.dora?.[ty] ?? 0) > 0,
  };
}

/**
 * The pure count: how many ways the opponent could hold each shape that waits on
 * `ty`, with every shape a public fact REFUTES set to zero.
 *
 * 現物 kills the cell outright (they passed it ⇒ furiten). スジ kills one
 * リャンメン holding and only that one: the (ty+1, ty+2) holding also waits on
 * ty+3, so ty+3 in their genbutsu makes it furiten. カンチャン/ペンチャン wait on
 * `ty` alone, and `ty` is known not to be genbutsu at that point, so no furiten
 * test applies to either. Everything else is a count of unseen copies, which
 * decays to zero as 壁 accumulates.
 *
 * M10b GRADES the スジ kill without deciding anything: the refuted リャンメン
 * mass is no longer discarded but BOOKED, into `ryanmenHalf` when the tile's
 * other end is still live and into `ryanmenFull` when nothing リャンメン can
 * reach the tile at all. Which of the three rows counts, and at what weight, is
 * `combineShapes`'s business; `sujiHalfSurvive = sujiFullSurvive = 0` restores
 * the binary kill exactly. Same for `ryanmenDora`, the dora-bearing share of the
 * live holdings — a partition of a count, with no judgment in it.
 */
export function shapeBaseMasses(ty: number, ctx: WaitContext): ShapeBase {
  // 現物 — they passed it, so they are furiten on it. A proof, not an estimate.
  if (ctx.genbutsu.has(ty)) return zeroBase();
  const u = (t: number) => (t < 0 || t > 33 ? 0 : (ctx.unseen[t] ?? 0));
  const base = zeroBase();
  base.shanpon = choose2(u(ty));
  base.tanki = u(ty);
  // Honors have no run shapes at all: シャンポン/タンキ IS the whole mechanism.
  if (suitOfType(ty) === "z") return base;

  const r = rankOfType(ty); // 1..9
  const dora = (t: number) => (ctx.dora?.[t] ?? 0) > 0;
  // The リャンメン holdings that can physically reach `ty`: the (ty+1, ty+2)
  // bridge for ranks 1..6 and the (ty−1, ty−2) bridge for ranks 4..9. Each one
  // carries its own スジ proof (the OTHER tile it waits on) and its own bridge.
  const holdings: { mass: number; refuted: boolean; dora: boolean }[] = [];
  if (r <= 6) {
    holdings.push({
      mass: u(ty + 1) * u(ty + 2),
      refuted: ctx.genbutsu.has(ty + 3),
      dora: dora(ty + 1) || dora(ty + 2),
    });
  }
  if (r >= 4) {
    holdings.push({
      mass: u(ty - 1) * u(ty - 2),
      refuted: ctx.genbutsu.has(ty - 3),
      dora: dora(ty - 1) || dora(ty - 2),
    });
  }
  // 全スジ is "every リャンメン that reaches this tile is refuted" — which for a
  // 1/2/3 or a 7/8/9 is one end, and for a 4/5/6 is both. 半スジ is the rest.
  const full = holdings.every((h) => h.refuted);
  for (const h of holdings) {
    if (!h.refuted) {
      base.ryanmen += h.mass;
      if (h.dora) base.ryanmenDora += h.mass;
    } else if (full) base.ryanmenFull += h.mass;
    else base.ryanmenHalf += h.mass;
  }
  if (r >= 2 && r <= 8) base.kanchan = u(ty - 1) * u(ty + 1);
  if (r === 3) base.penchan = u(ty - 1) * u(ty - 2);
  if (r === 7) base.penchan = u(ty + 1) * u(ty + 2);
  return base;
}

/**
 * The JUDGMENT half: turn the counts into weights with the tuning surface.
 *
 * A closed form in `ComputedWeights` and nothing else — the one function every
 * consumer of the wait model goes through, live seat and offline fit alike. The
 * denominators are powers of two (and the exact 6 of `C(4,2)`), so the
 * associations below are the arithmetic the module has always performed, tile
 * for tile and bit for bit.
 */
export function combineShapes(
  ty: number,
  base: ShapeBase,
  f: ShapeFlags,
  w: ComputedWeights = DEFAULT_COMPUTED,
): Record<WaitShape, number> {
  // The リャンメン mass, graded (M10b): the live holdings, the dora-bearing
  // share of them bumped, and the スジ-refuted residue re-admitted at whatever
  // weight the two suji constants give it. On the shipped vector
  // (doraBridge = 1, sujiHalfSurvive = sujiFullSurvive = 0) every added term is
  // an exact zero over integers and this is `base.ryanmen`, unchanged.
  const ryanmen = base.ryanmen + (w.doraBridge - 1) * base.ryanmenDora +
    w.sujiHalfSurvive * base.ryanmenHalf + w.sujiFullSurvive * base.ryanmenFull;
  // ドラは対子で持たれる: the two shapes that wait with a held pair, on a type
  // the indicators made valuable.
  const doraPair = f.doraType ? w.doraPair : 1;
  const out: Record<WaitShape, number> = {
    "リャンメン": w.shapePrior["リャンメン"] * (ryanmen / 32),
    "カンチャン": w.shapePrior["カンチャン"] * base.kanchan / 16,
    "ペンチャン": w.shapePrior["ペンチャン"] * base.penchan / 16,
    "シャンポン": w.shapePrior["シャンポン"] * (base.shanpon / 6) *
      (f.valueHonor ? w.yakuhaiShanpon : 1) * doraPair,
    "タンキ": w.shapePrior["タンキ"] * (base.tanki / 4) * doraPair,
  };
  applyMeldRead(out, ty, f, w);
  return out;
}

/**
 * Per shape, how strongly public counting supports "this opponent is waiting on
 * `ty` through that shape". Zero means REFUTED, not merely unlikely.
 *
 * Each survivor is graded by the number of ways the opponent could be holding
 * the tiles the shape needs, over the number of ways they could hold them if
 * nothing at all were visible — so the value is 1 when the shape is untouched
 * and decays smoothly to 0 as copies of the tiles it needs come into view. That
 * is the same kabe/count evidence `danger.ts` reports as ノーチャンス/ワン
 * チャンス, read as a count instead of as a label.
 *
 * APPROXIMATION, stated plainly: the `unseen` pool mixes the live wall with the
 * other three concealed hands and the dead wall, so these are not conditional
 * probabilities that the opponent holds those tiles — they are the hypergeo-
 * metric numerators, shared by every opponent, with the denominators dropped
 * because they are common to all shapes and all seats. Only ratios are consumed
 * (`dealinScale` sets the absolute level), so the dropped normaliser cannot
 * change an ordering. The two ends of a リャンメン are counted as a pair of
 * independent draws (`u(a)·u(b)`, max 4·4) and a シャンポン/タンキ as copies of
 * one type (`C(u,2)`, max 6, and `u`, max 4).
 *
 * Finally, if the context carries a `read`, the surviving weights are tilted by
 * what the opponent's melds are showing (`applyMeldRead`) — after the survival
 * count, never instead of it.
 */
export function waitShapeWeights(
  ty: number,
  ctx: WaitContext,
  w: ComputedWeights = DEFAULT_COMPUTED,
): Record<WaitShape, number> {
  return combineShapes(ty, shapeBaseMasses(ty, ctx), shapeFlagsOf(ty, ctx), w);
}

/**
 * Scale the surviving shapes by what this opponent's melds are showing.
 *
 * Two independent modifiers, applied on top of the survival count and never in
 * place of it: a killed shape stays killed, because a proof outranks a 模様.
 *   - 染め手模様 tilts across TYPES: the read suit and the honors up, the other
 *     two suits down. Same tile, different opponent ⇒ different row of `dealinP`,
 *     which is the whole point of conditioning on melds.
 *   - トイトイ模様 tilts across SHAPES: pairs up, runs down, for every type.
 *
 * Both are functions of melds that are face up right now (see `MeldRead`).
 */
function applyMeldRead(
  out: Record<WaitShape, number>,
  ty: number,
  read: ShapeFlags,
  w: ComputedWeights,
): void {
  const suit = suitOfType(ty);
  const flush = read.honitsuSuit === null
    ? 1
    : (suit === read.honitsuSuit || suit === "z" ? w.honitsuHot : w.honitsuCold);
  for (const s of WAIT_SHAPES) {
    if (out[s] === 0) continue;
    const pairShape = s === "シャンポン" || s === "タンキ";
    const toitoi = !read.toitoi ? 1 : (pairShape ? w.toitoiPair : w.toitoiRun);
    out[s] *= flush * toitoi;
  }
}

/**
 * `combineShapes` summed: the RAW mass supporting "waiting on `ty`", before any
 * calibration onto a probability. Not a probability itself and not comparable
 * across boards — the two functions below are what turn it into one, and they
 * differ precisely in how.
 */
function shapeMassOf(
  ty: number,
  base: ShapeBase,
  f: ShapeFlags,
  w: ComputedWeights = DEFAULT_COMPUTED,
): number {
  const shapes = combineShapes(ty, base, f, w);
  let sum = 0;
  for (const s of WAIT_SHAPES) sum += shapes[s];
  return sum;
}

/** `shapeMassOf` calibrated by the global constant: P(waiting on `ty` | tenpai)
 * in the UN-NORMALIZED reading. This is the M10a form, kept because it is what
 * `waitNormalize: false` means, and because a per-type answer that does not need
 * the other 33 types is what most callers want to reason about. */
export function waitLikelihoodFrom(
  ty: number,
  base: ShapeBase,
  f: ShapeFlags,
  w: ComputedWeights = DEFAULT_COMPUTED,
): number {
  return Math.min(1, shapeMassOf(ty, base, f, w) * w.dealinScale);
}

/**
 * The whole row at once: P(waiting on each type | tenpai) for ONE opponent.
 *
 * WHY A ROW (M10b). The un-normalized form treats 34 types as 34 independent
 * questions, and they are not: a tenpai hand waits on about 1.4 of them, no
 * matter how many the counting leaves alive. Measured against the truth the old
 * form ranked the types correctly (a monotone lift of 0.12→3.35 across the
 * predicted bands) and carried the wrong TOTAL — 0.38 types per board against a
 * true 1.38. So with `waitNormalize` on, the masses are divided by their own row
 * total and multiplied by `expWaitMass`: the ranking is untouched, the level is
 * per hand rather than per population, and the row sums to `expWaitMass` over
 * the live types by construction.
 *
 * A type public facts REFUTE has mass 0 and stays 0 through the division — a
 * proof is not a share of anything — so 現物/振聴 and the カベ kills survive
 * normalization intact.
 *
 * With `waitNormalize` off this is `waitLikelihoodFrom` applied 34 times, bit
 * for bit; the two paths call the same `shapeMassOf`.
 */
export function waitRowFrom(
  bases: readonly ShapeBase[],
  flagsOf: (ty: number) => ShapeFlags,
  w: ComputedWeights = DEFAULT_COMPUTED,
): Float64Array {
  const mass = new Float64Array(34);
  let total = 0;
  for (let ty = 0; ty < 34; ty++) {
    mass[ty] = shapeMassOf(ty, bases[ty], flagsOf(ty), w);
    total += mass[ty];
  }
  const out = new Float64Array(34);
  if (!w.waitNormalize) {
    for (let ty = 0; ty < 34; ty++) out[ty] = Math.min(1, mass[ty] * w.dealinScale);
    return out;
  }
  if (total <= 0) return out;
  for (let ty = 0; ty < 34; ty++) {
    if (mass[ty] <= 0) continue;
    out[ty] = Math.min(1, w.expWaitMass * (mass[ty] / total));
  }
  return out;
}

/** `waitShapeWeights` summed and calibrated: P(waiting on `ty` | tenpai). */
export function waitLikelihood(
  ty: number,
  ctx: WaitContext,
  w: ComputedWeights = DEFAULT_COMPUTED,
): number {
  return waitLikelihoodFrom(ty, shapeBaseMasses(ty, ctx), shapeFlagsOf(ty, ctx), w);
}

// ---------------------------------------------------------------------------
// the flat hot path: one opponent's whole row, without allocating
// ---------------------------------------------------------------------------
//
// WHY THIS EXISTS AND WHY IT IS NOT THE REFERENCE. Everything above is the
// definition — a `ShapeBase` object per (opponent, tile type), a
// `Record<WaitShape, number>` per cell, and a `flagsOf(ty)` call per type — and
// it is the shape a reader, a test and the offline fit all want. It is also
// ~300 short-lived objects per DECISION, three quarters of what the 計算 seat
// spends its time on, and self-play runs ~10^5 decisions.
//
// So the same arithmetic is written a second time, flat: the eight counts of a
// `ShapeBase` become eight slots of a `Float64Array`, the five shape weights
// become five locals, and the three `ShapeFlags` fields that cannot vary with
// the tile type (`honitsuSuit`, `toitoi`, and — through the weight vector —
// every multiplier) are read once per opponent instead of 34 times.
//
// BIT-EXACT, and that is a tested claim, not a hope: `test/computed_test.ts`
// fuzzes this against `shapeBaseMasses` + `waitRowFrom` on random boards and
// random weight vectors and demands element-for-element equality. Every
// association is copied deliberately — `prK * kanchan / 16` really is
// `(prK * kanchan) / 16` above while リャンメン really is `prR * (mass / 32)`,
// and float multiplication is not associative. `scripts/calibrate_fit.ts`'s
// `condRowInline` is the third copy of the same arithmetic, for the same
// reason and under the same discipline.

/** Doubles per tile type in a flat base row — the fields of `ShapeBase`. */
const BF = 8;
const B_RYANMEN = 0;
const B_RYANMEN_DORA = 1;
const B_RYANMEN_HALF = 2;
const B_RYANMEN_FULL = 3;
const B_KANCHAN = 4;
const B_PENCHAN = 5;
const B_SHANPON = 6;
const B_TANKI = 7;

/**
 * Length of the buffer one row evaluation fills: the 34 wait likelihoods first,
 * then the 34 × 8 parameter-free counts they were derived from. One buffer
 * rather than two because the native kernel writes both in a single crossing,
 * and because the counts are what a calibration trace wants back.
 */
export const SHAPE_ROW_LEN = 34 + 34 * BF;

/** Per-type flag bits, as the kernel's `flags[]` argument packs them. */
const F_GENBUTSU = 1;
const F_VALUE_HONOR = 2;
const F_DORA = 4;

/** Slots of the packed weight vector the kernel reads. Mirrored in mjkernel.cc. */
const SHAPE_W_LEN = 17;

/**
 * `ComputedWeights` as the flat vector the kernel takes. Packed ONCE per
 * provider — these are constants of the seat, not of the decision.
 *
 * Slot 5 is `doraBridge − 1` rather than `doraBridge`: `combineShapes` forms
 * that difference itself, and forming it here instead keeps the C side free of
 * an arithmetic step the TypeScript would have to be trusted to match.
 */
function packShapeWeights(w: ComputedWeights): Float64Array {
  const p = new Float64Array(SHAPE_W_LEN);
  p[0] = w.shapePrior["リャンメン"];
  p[1] = w.shapePrior["カンチャン"];
  p[2] = w.shapePrior["ペンチャン"];
  p[3] = w.shapePrior["シャンポン"];
  p[4] = w.shapePrior["タンキ"];
  p[5] = w.doraBridge - 1;
  p[6] = w.sujiHalfSurvive;
  p[7] = w.sujiFullSurvive;
  p[8] = w.yakuhaiShanpon;
  p[9] = w.doraPair;
  p[10] = w.honitsuHot;
  p[11] = w.honitsuCold;
  p[12] = w.toitoiPair;
  p[13] = w.toitoiRun;
  p[14] = w.dealinScale;
  p[15] = w.expWaitMass;
  p[16] = w.waitNormalize ? 1 : 0;
  return p;
}

/**
 * Process-wide scratch for the kernel's two integer arguments.
 *
 * The same argument `src/kernel.ts` makes for its 34-byte count buffer: a row
 * evaluation is a LEAF — the arrays are filled and handed straight to the FFI
 * call, and nothing in between can re-enter this module — so a fresh pair per
 * (opponent, decision) would be pure garbage, which is exactly what this path
 * exists to stop producing.
 */
const scUnseen = new Int32Array(34);
const scFlags = new Int32Array(34);

/** 染め手 suit as the kernel encodes it: 0 なし, 1 m, 2 p, 3 s (and 0 = 字 for a tile). */
function honitsuCode(s: "m" | "p" | "s" | null): number {
  return s === null ? 0 : s === "m" ? 1 : s === "p" ? 2 : 3;
}

/**
 * Fill `scUnseen` / `scFlags` from a context, or report that this context is
 * outside the kernel's integer domain (a count that is not 0..4 — which the
 * `publicUnseen` the seat feeds it never is, but a hand-built test context may
 * be). Refusing is free: the caller simply takes the TypeScript path.
 */
function packShapeCtx(ctx: WaitContext): boolean {
  const un = ctx.unseen;
  const gen = ctx.genbutsu;
  const vh = ctx.valueHonors;
  const dr = ctx.dora;
  for (let t = 0; t < 34; t++) {
    const u = un[t] ?? 0;
    if (!(u >= 0 && u <= 4) || u !== Math.floor(u)) return false;
    scUnseen[t] = u;
    scFlags[t] = (gen.has(t) ? F_GENBUTSU : 0) | (vh.has(t) ? F_VALUE_HONOR : 0) |
      ((dr?.[t] ?? 0) > 0 ? F_DORA : 0);
  }
  return true;
}

/**
 * The parameter-free counts of every tile type at once — `shapeBaseMasses`
 * written flat, into `out[34 …]`. Reads nothing but public facts, exactly as
 * the definition above does; see its doc comment for what each kill means.
 */
function shapeBasesFlat(ctx: WaitContext, out: Float64Array): void {
  const un = ctx.unseen;
  const gen = ctx.genbutsu;
  const dr = ctx.dora;
  const u = (t: number): number => (t < 0 || t > 33 ? 0 : (un[t] ?? 0));
  const isDora = (t: number): boolean => (dr?.[t] ?? 0) > 0;

  out.fill(0);
  for (let ty = 0; ty < 34; ty++) {
    if (gen.has(ty)) continue; // 現物 — furiten, so every shape is refuted
    const o = 34 + ty * BF;
    const uty = u(ty);
    out[o + B_SHANPON] = uty < 2 ? 0 : (uty * (uty - 1)) / 2;
    out[o + B_TANKI] = uty;
    if (ty >= 27) continue; // 字牌: シャンポン/タンキ IS the whole mechanism
    const r = (ty % 9) + 1; // rankOfType, inlined
    const up = r <= 6; // the (ty+1, ty+2) bridge
    const dn = r >= 4; // the (ty−1, ty−2) bridge
    const upRef = up && gen.has(ty + 3);
    const dnRef = dn && gen.has(ty - 3);
    const full = (!up || upRef) && (!dn || dnRef); // 全スジ
    if (up) {
      const m = u(ty + 1) * u(ty + 2);
      if (!upRef) {
        out[o + B_RYANMEN] += m;
        if (isDora(ty + 1) || isDora(ty + 2)) out[o + B_RYANMEN_DORA] += m;
      } else if (full) out[o + B_RYANMEN_FULL] += m;
      else out[o + B_RYANMEN_HALF] += m;
    }
    if (dn) {
      const m = u(ty - 1) * u(ty - 2);
      if (!dnRef) {
        out[o + B_RYANMEN] += m;
        if (isDora(ty - 1) || isDora(ty - 2)) out[o + B_RYANMEN_DORA] += m;
      } else if (full) out[o + B_RYANMEN_FULL] += m;
      else out[o + B_RYANMEN_HALF] += m;
    }
    if (r >= 2 && r <= 8) out[o + B_KANCHAN] = u(ty - 1) * u(ty + 1);
    if (r === 3) out[o + B_PENCHAN] = u(ty - 1) * u(ty - 2);
    if (r === 7) out[o + B_PENCHAN] = u(ty + 1) * u(ty + 2);
  }
}

/**
 * `combineShapes` + `applyMeldRead` + `waitRowFrom`, over the counts already in
 * `out[34 …]`, writing the row into `out[0 … 33]`. The three type-invariant
 * `ShapeFlags` fields and every weight are read before the loop; only
 * `valueHonor` and `doraType` are per type, and both are single lookups.
 */
function shapeMassRowFlat(ctx: WaitContext, w: ComputedWeights, out: Float64Array): void {
  const prR = w.shapePrior["リャンメン"];
  const prK = w.shapePrior["カンチャン"];
  const prP = w.shapePrior["ペンチャン"];
  const prS = w.shapePrior["シャンポン"];
  const prT = w.shapePrior["タンキ"];
  const dB = w.doraBridge - 1;
  const sH = w.sujiHalfSurvive;
  const sF = w.sujiFullSurvive;
  const hs = honitsuCode(ctx.read?.honitsuSuit ?? null);
  const toitoi = ctx.read?.toitoi ?? false;
  const run = toitoi ? w.toitoiRun : 1;
  const pair = toitoi ? w.toitoiPair : 1;
  const vh = ctx.valueHonors;
  const dr = ctx.dora;

  let total = 0;
  for (let ty = 0; ty < 34; ty++) {
    const o = 34 + ty * BF;
    const ryanmen = out[o + B_RYANMEN] + dB * out[o + B_RYANMEN_DORA] +
      sH * out[o + B_RYANMEN_HALF] + sF * out[o + B_RYANMEN_FULL];
    const doraPair = (dr?.[ty] ?? 0) > 0 ? w.doraPair : 1;
    let sRyan = prR * (ryanmen / 32);
    let sKan = prK * out[o + B_KANCHAN] / 16;
    let sPen = prP * out[o + B_PENCHAN] / 16;
    let sSha = prS * (out[o + B_SHANPON] / 6) * (vh.has(ty) ? w.yakuhaiShanpon : 1) * doraPair;
    let sTan = prT * (out[o + B_TANKI] / 4) * doraPair;
    // `applyMeldRead`, shape by shape: a killed shape stays killed.
    const suit = ty < 9 ? 1 : ty < 18 ? 2 : ty < 27 ? 3 : 0;
    const flush = hs === 0 ? 1 : (suit === hs || suit === 0 ? w.honitsuHot : w.honitsuCold);
    if (sRyan !== 0) sRyan *= flush * run;
    if (sKan !== 0) sKan *= flush * run;
    if (sPen !== 0) sPen *= flush * run;
    if (sSha !== 0) sSha *= flush * pair;
    if (sTan !== 0) sTan *= flush * pair;
    const mass = sRyan + sKan + sPen + sSha + sTan;
    out[ty] = mass;
    total += mass;
  }

  if (!w.waitNormalize) {
    const sc = w.dealinScale;
    for (let ty = 0; ty < 34; ty++) out[ty] = Math.min(1, out[ty] * sc);
    return;
  }
  if (total <= 0) {
    for (let ty = 0; ty < 34; ty++) out[ty] = 0;
    return;
  }
  const exp = w.expWaitMass;
  for (let ty = 0; ty < 34; ty++) {
    const m = out[ty];
    out[ty] = m <= 0 ? 0 : Math.min(1, exp * (m / total));
  }
}

/**
 * The TypeScript reference of the flat path: counts, then row, no allocation
 * beyond the two closures. This is what runs when the kernel is not loaded, and
 * it is what the parity fuzz judges the kernel against.
 */
export function shapeRowTS(
  ctx: WaitContext,
  w: ComputedWeights,
  out: Float64Array,
): void {
  shapeBasesFlat(ctx, out);
  shapeMassRowFlat(ctx, w, out);
}

/** One opponent's row, filled into a `SHAPE_ROW_LEN` buffer. */
export type ShapeRowFn = (ctx: WaitContext, out: Float64Array) => void;

/**
 * A prepared row evaluator for one weight vector: native when the kernel is
 * loaded and the context is inside its integer domain, `shapeRowTS` otherwise.
 * The choice is invisible — the two agree bit for bit, which is what
 * `test/kernel_native_test.ts` fuzzes.
 *
 * `native: false` forces the TypeScript, for the parity test and for anyone who
 * wants the reference on purpose.
 */
export function shapeRowEvaluator(w: ComputedWeights, native = true): ShapeRowFn {
  const packed = native ? packShapeWeights(w) : null;
  return (ctx: WaitContext, out: Float64Array): void => {
    if (packed && packShapeCtx(ctx)) {
      const suit = honitsuCode(ctx.read?.honitsuSuit ?? null);
      const toi = ctx.read?.toitoi ? 1 : 0;
      if (shapeMasses(scUnseen, scFlags, suit, toi, packed, out)) return;
    }
    shapeRowTS(ctx, w, out);
  };
}

/** The 34 `ShapeBase` objects a filled row buffer carries, for a trace. */
function basesFromRow(row: Float64Array): ShapeBase[] {
  const out: ShapeBase[] = [];
  for (let ty = 0; ty < 34; ty++) {
    const o = 34 + ty * BF;
    out.push({
      ryanmen: row[o + B_RYANMEN],
      ryanmenDora: row[o + B_RYANMEN_DORA],
      ryanmenHalf: row[o + B_RYANMEN_HALF],
      ryanmenFull: row[o + B_RYANMEN_FULL],
      kanchan: row[o + B_KANCHAN],
      penchan: row[o + B_PENCHAN],
      shanpon: row[o + B_SHANPON],
      tanki: row[o + B_TANKI],
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// public bookkeeping over an Observation
// ---------------------------------------------------------------------------

/** Relative index of the dealer, from our own seat wind. */
function dealerRel(obs: Observation): number {
  return (4 - ((obs.seatWind - 27) % 4)) % 4;
}

/** Seat wind TYPE of the seat at relative index `rel`. */
function seatWindOf(obs: Observation, rel: number): number {
  return 27 + (((obs.seatWind - 27) + rel) % 4);
}

/** Turn order within a go-around: the dealer moves first. */
function turnOrder(obs: Observation, rel: number): number {
  return (rel - dealerRel(obs) + 4) % 4;
}

/**
 * The types each seat provably cannot ron, one set per RELATIVE index 0..3.
 *
 * Two mechanisms, both public and both exact:
 *   - their own discards make them permanently furiten on those types;
 *   - once they have declared riichi their wait is frozen, so every tile
 *     discarded by ANYONE since the declaration passed them and locks them out.
 *
 * The second needs an ordering, and an `Observation` carries junme per river
 * entry rather than a global sequence. A discard is counted as "since the
 * declaration" when its junme is strictly later, or when it is in the same
 * go-around from a seat that acts after the declarer. Calls can reorder seats
 * inside a go-around, so on the rare boundary case this UNDER-claims safety —
 * the direction that costs a little caution rather than a deal-in.
 */
export function genbutsuSets(obs: Observation): Set<number>[] {
  const out = [new Set<number>(), new Set<number>(), new Set<number>(), new Set<number>()];
  for (let r = 0; r < 4; r++) {
    // A called tile was still discarded — it passed everyone with a ron on it.
    for (const e of obs.rivers[r]) out[r].add(tileType(e.tile));
  }
  for (let r = 0; r < 4; r++) {
    if (!obs.riichi[r]) continue;
    const dj = obs.riichiJunme[r];
    const order = turnOrder(obs, r);
    for (let q = 0; q < 4; q++) {
      if (q === r) continue;
      for (const e of obs.rivers[q]) {
        if (e.junme > dj || (e.junme === dj && turnOrder(obs, q) > order)) {
          out[r].add(tileType(e.tile));
        }
      }
    }
  }
  return out;
}

/** Dora count per tile TYPE from the revealed indicators (aka excluded). */
function doraOfType(obs: Observation): number[] {
  const c = zeros34();
  for (const ind of obs.doraIndicators) c[doraFromIndicatorType(tileType(ind))]++;
  return c;
}

/** Dora (indicator-derived plus aka) visible inside a seat's melds. */
function meldDora(obs: Observation, rel: number, dora: readonly number[]): number {
  let n = 0;
  for (const m of obs.melds[rel]) {
    for (const t of m.tiles) {
      n += dora[tileType(t)] ?? 0;
      if (obs.akaIds.has(t)) n++;
    }
  }
  return n;
}

// ---------------------------------------------------------------------------
// what a deal-in costs, and what class of hand is collecting it
// ---------------------------------------------------------------------------

/**
 * The public class of an opponent's hand, in the order `yakuFactor` names it.
 * A declared riichi outranks everything (it is closed by construction), so the
 * three classes partition the table exactly.
 */
export type YakuClass = 0 | 1 | 2; // 0 riichi, 1 open, 2 damaten

function yakuClassOf(riichi: boolean, open: boolean): YakuClass {
  return riichi ? 0 : open ? 1 : 2;
}

/** P(the hand can actually declare a ron) for a class. */
function yakuFactorOf(w: ComputedWeights, cls: YakuClass): number {
  return cls === 0 ? w.yakuFactor.riichi : cls === 1 ? w.yakuFactor.open : w.yakuFactor.damaten;
}

/** The public facts the static value model prices. All of them face up. */
export interface ValueFacts {
  cls: YakuClass;
  /** 染め手模様 read from their melds — only consulted for an open hand. */
  honitsu: boolean;
  /** Dora visible inside their melds, aka included. */
  meldDora: number;
  /** 役牌 triplet melds showing (ankan included). */
  yakuhai: number;
  dealer: boolean;
}

/**
 * The modelled ron payment of an opponent, before the winning tile's own dora.
 * Split out for the same reason `combineShapes` is: the calibration recorder
 * caches the FACTS and the offline fit re-prices them through this one function,
 * so the value model can never fork into two versions of itself.
 */
export function baseValueOf(w: ComputedWeights, f: ValueFacts): number {
  const base = f.cls === 0
    ? w.valueRiichi
    : f.cls === 1
    ? (f.honitsu ? w.valueHonitsu : w.valueOpen)
    : w.valueDamaten;
  let value = base + w.valuePerDora * f.meldDora + w.valueYakuhai * f.yakuhai;
  if (f.dealer) value *= w.valueDealer;
  return Math.min(w.valueCap, value);
}

/**
 * What dealing `ty` into this opponent actually costs: the modelled hand price,
 * plus the winning tile's own dora, plus the 本場 surcharge.
 *
 * THE ONE PLACE THE FIGURE IS ASSEMBLED (M10b), and there are three consumers —
 * the live seat's `dealinValue`, the recorder's re-derivation, and the report's
 * error column. The 本場 term is not a model at all: `score.ts#ronValue` charges
 * `honba × 300` and so does this, in the same order (after the hand's own cap,
 * because a 本場 is a surcharge on the payment and not part of the hand's
 * value). Until M10b the model omitted it entirely, which made every deal-in on
 * a 2本場 look 600 points cheaper than the table would actually charge.
 */
export function valueOnType(
  w: ComputedWeights,
  base: number,
  doraOnType: number,
  honba: number,
): number {
  return Math.min(w.valueCap, base + w.valuePerDora * doraOnType) + w.valuePerHonba * honba;
}

// ---------------------------------------------------------------------------
// the trace: what the model computed, on its way to computing it
// ---------------------------------------------------------------------------

/**
 * One opponent's row of the model, as INTERMEDIATES rather than as an answer.
 *
 * WHY IT EXISTS (M10a). The calibration recorder pairs this model's predictions
 * with the engine's truth so an offline fit can say WHERE the deal-in edge is
 * lost — tenpai estimation, wait location, or value. A fit needs to re-evaluate
 * the model under different weights, and replaying two thousand hanchan per
 * candidate vector is not a fit, it is a tournament. So the recorder caches this
 * instead: the parameter-FREE counts (`base`, one `ShapeBase` per tile type) and
 * the class facts that select the multipliers. `combineShapes` + `baseValueOf` +
 * `tenpaiPriorOf` then reproduce every number below in closed form.
 *
 * It is filled only when the caller asks (`ComputedTraceRef`), and it holds the
 * SAME arrays the provider returned — reading it changes nothing, and neither
 * does not reading it.
 */
export interface ComputedOppTrace {
  /** RELATIVE index into an `Observation` (1 = shimocha … 3 = kamicha). */
  rel: number;
  /** Public class of their hand — what `yakuFactor` and the value base select. */
  cls: YakuClass;
  /** Total melds (ankan included): the ROW of the tenpai table. */
  melds: number;
  /** Open (non-ankan) melds — what the content reads activate on. */
  openMelds: number;
  /** Cell of the tenpai table this prediction came from. */
  cell: { row: number; col: number };
  /** The RAW prior, before `tenpaiFloor` gates the reported field. */
  tenpaiP: number;
  /**
   * Some OTHER seat has a live riichi declaration (ourselves included, this
   * opponent excluded) — what `tenpaiOtherRiichi` conditions on.
   */
  otherRiichi: boolean;
  /** Their face-up content read. */
  read: MeldRead;
  /** Dora visible in their melds, aka included. */
  meldDora: number;
  dealer: boolean;
  /** Honor types that are 役牌 for this seat. */
  valueHonors: ReadonlySet<number>;
  /**
   * `baseValueOf` for this opponent: the winning tile's own dora and the 本場
   * surcharge are both added on top of it by `valueOnType`, so this is the hand
   * price and not the payment.
   */
  value: number;
  /** Per tile type, the parameter-free counts. Length 34. */
  base: ShapeBase[];
  /** The row the provider returned, for a bit-exact reproduction check. */
  dealinP: Float32Array;
}

/** Everything one `computedReads` call worked out, per decision. */
export interface ComputedTrace {
  junme: number;
  /** Dora count per tile TYPE from the revealed indicators (aka excluded). */
  dora: readonly number[];
  /** Length 3, in `Reads` order (index 0 = shimocha). */
  opps: ComputedOppTrace[];
}

/**
 * The out-param a caller passes to `computedReads` to receive the trace. Same
 * idiom as `MatchOptions.tableRef`: a one-field box the callee overwrites, so
 * the provider keeps the `ReadsProvider` signature the whole system is built on.
 */
export interface ComputedTraceRef {
  t: ComputedTrace | null;
}

// ---------------------------------------------------------------------------
// the provider
// ---------------------------------------------------------------------------

/**
 * A `ReadsProvider` that cheats at nothing: everything it reports is derived by
 * exact counting over the `Observation` it is handed, so a seat driven by it is
 * legal at any table, the human's TUI included.
 *
 * WHAT IT FILLS
 *   dealinP        P(tenpai) × P(waiting on the type | tenpai), per opponent,
 *                  both factors conditioned on that opponent's face-up melds
 *                  (`meldReadOf`) — the 染め手/トイトイ/役牌 content reads.
 *   dealinValue    the static value model, per type (the winning tile's own
 *                  dora count is public, so it is priced — as are melded dora,
 *                  aka included, a flush read, and 役牌 triplets).
 *   expLoss        the same figure without the per-type dora.
 *   tenpaiP        the base rate, gated by `tenpaiFloor` (see the weight).
 *   wallComposition / oppConcealed
 *                  the honest uniform posterior over the unseen pool.
 *
 * WHAT IT REFUSES TO FILL, and why absence is the honest answer: `nextDraw`,
 * `ownNextDraw`, `nextDora` and `riichiNextDraw` are all statements about the
 * ORDER of unseen tiles, and no amount of counting reveals an order. Every
 * consumer term degrades to the base heuristic when its field is missing, which
 * is exactly right — the base policy's ukeire already prices "how many of the
 * tiles I want are still out there", and `wallComposition` sharpens that.
 *
 * PURITY. Same Observation ⇒ same Reads. No RNG, no memory between calls: the
 * only state is the weights object, fixed at construction.
 *
 * `traceRef`, when supplied, receives a `ComputedTrace` of the intermediates on
 * every call (M10a calibration). It is an OUT-PARAM and nothing more: the Reads
 * returned are the same object with the same numbers either way, so a seat with
 * a recorder attached plays the identical game to one without.
 */
export function computedReads(
  w?: Partial<ComputedWeights>,
  traceRef?: ComputedTraceRef,
): ReadsProvider {
  const cw = mergeComputed(w);
  // The weight vector is a constant of the seat, so the row evaluator is built
  // once here rather than per decision — and with it the packed vector the
  // kernel reads. The scratch row is the provider's own: it is filled and
  // drained inside one iteration of the three-opponent loop below.
  const shapeRow = shapeRowEvaluator(cw);
  const row = new Float64Array(SHAPE_ROW_LEN);

  return (obs: Observation): Reads => {
    const unseen = publicUnseen(obs);
    const genbutsu = genbutsuSets(obs);
    const dora = doraOfType(obs);
    const dealer = dealerRel(obs);

    const tenpaiP: number[] = [];
    const dealinP: Float32Array[] = [];
    const dealinValue: Float32Array[] = [];
    const expLoss: number[] = [];
    const trace: ComputedTrace | null = traceRef ? { junme: obs.junme, dora, opps: [] } : null;

    for (let i = 0; i < 3; i++) {
      const rel = i + 1; // Reads index 0 = shimocha = Observation index 1
      const melds = obs.melds[rel];
      const open = melds.some((m) => m.kind !== "ankan");
      const riichi = obs.riichi[rel];
      const valueHonors = valueHonorsOf(obs.roundWind, seatWindOf(obs, rel));
      // What their melds are showing — a fact about the table right now, read the
      // same way for every seat that shows the same tiles.
      const read = meldReadOf(melds, valueHonors);
      const md = meldDora(obs, rel, dora);
      // Public state of the TABLE, not of this opponent: someone else has
      // declared. Our own declaration counts — it is the loudest one there is.
      const otherRiichi = obs.riichi.some((r, q) => r && q !== rel);
      const pT = tenpaiPriorOf(
        cw,
        melds.length,
        obs.junme,
        riichi,
        read.yakuhai.size > 0,
        otherRiichi,
        md,
      );

      // value: facts only — declared riichi, meld CONTENTS (flush read, yakuhai
      // triplets), visible melded dora (aka included, `meldDora`), and who deals.
      const cls = yakuClassOf(riichi, open);
      const value = baseValueOf(cw, {
        cls,
        honitsu: read.honitsuSuit !== null,
        meldDora: md,
        yakuhai: read.yakuhai.size,
        dealer: rel === dealer,
      });
      // The 本場 surcharge is on the payment, not on the hand: `expLoss` is the
      // per-type figure with the winning tile's own dora left out, so it is the
      // same assembly with a dora count of zero (`valueOnType`).
      expLoss.push(pT > 0 ? valueOnType(cw, value, 0, obs.honba) : 0);

      const ctx: WaitContext = {
        unseen,
        genbutsu: genbutsu[rel],
        valueHonors,
        read,
        dora,
      };
      const yaku = yakuFactorOf(cw, cls);

      const p = new Float32Array(34);
      const v = new Float32Array(34);
      // The whole row is built before any of it is consumed: the row form
      // normalizes over its own total when `waitNormalize` is on, so a per-type
      // answer is not available until every type has been counted. `shapeRow`
      // is `shapeBaseMasses` + `waitRowFrom` written flat — same numbers, no
      // allocation — and it is the kernel's entry point when one is loaded.
      shapeRow(ctx, row);
      for (let ty = 0; ty < 34; ty++) {
        const q = pT * yaku * row[ty];
        if (q <= 0) continue;
        p[ty] = q;
        v[ty] = valueOnType(cw, value, dora[ty], obs.honba);
      }
      dealinP.push(p);
      dealinValue.push(v);
      if (trace) {
        trace.opps.push({
          rel,
          cls,
          melds: melds.length,
          openMelds: read.open,
          cell: tenpaiCellOf(cw, melds.length, obs.junme),
          tenpaiP: pT,
          otherRiichi,
          read,
          meldDora: md,
          dealer: rel === dealer,
          valueHonors,
          value,
          // The trace's own copy: `row` is scratch and the next opponent
          // overwrites it, so the counts are materialized here and nowhere else.
          base: basesFromRow(row),
          dealinP: p,
        });
      }
      // The reported tenpai read is GATED, not rescaled: under the floor it is
      // reported as nothing, at or over it as the prior itself. The deal-in
      // estimate above passes through neither. See `tenpaiFloor`.
      tenpaiP.push(pT >= cw.tenpaiFloor ? Math.max(0, Math.min(1, pT)) : 0);
    }

    // ---- the availability model ------------------------------------------
    //
    // The uniform posterior. Counting cannot sharpen the distribution over
    // TYPES beyond the unseen counts — that IS the posterior when nothing about
    // the hidden tiles is known — but it CAN split those copies exactly between
    // "still in the live wall" (drawable) and "in an opponent's hand"
    // (claimable), because both population sizes are public: `wallRemaining`,
    // and 13 − 3×melds per opponent.
    //
    // So this is option (a) of the two the consumer allows: scale by the live
    // wall's share of the unseen pool, and hand `oppConcealed` the complementary
    // share. `availabilityFrom` then reads `draw(ty)/poolSize()` =
    // unseen[ty]/unseenTotal — which is exactly P(the next draw is that type),
    // the dead wall's dilution included — while `call(ty)` stays alive and is
    // now proportional to the opponents' REAL hand sizes rather than to C7P's
    // flat quarter. Option (b) (raw counts, pool = unseenTotal) gives the same
    // draw ratio but prices the call channel against the wrong population, and
    // leaves `poolSize()` meaning "unseen tiles" where `pCompleteOf` wants "the
    // pool my draws come from".
    let unseenTotal = 0;
    for (let ty = 0; ty < 34; ty++) unseenTotal += unseen[ty];
    const live = Math.max(0, Math.min(obs.wallRemaining, unseenTotal));
    const wall = new Float32Array(34);
    const oppConcealed = [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
    if (unseenTotal > 0) {
      const wallShare = live / unseenTotal;
      const handShare = [0, 1, 2].map((i) =>
        Math.max(0, 13 - 3 * obs.melds[i + 1].length) / unseenTotal
      );
      for (let ty = 0; ty < 34; ty++) {
        wall[ty] = unseen[ty] * wallShare;
        for (let i = 0; i < 3; i++) oppConcealed[i][ty] = unseen[ty] * handShare[i];
      }
    }

    const reads: Reads = {
      tenpaiP,
      dealinP,
      dealinValue,
      expLoss,
      wallComposition: wall,
      oppConcealed,
    };
    if (cw.planner) reads.planner = true;
    if (traceRef) traceRef.t = trace;
    return reads;
  };
}
