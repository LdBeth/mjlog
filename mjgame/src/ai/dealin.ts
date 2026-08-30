// M14 — the LEARNED deal-in read: P(opponent i rons tile type t | public state).
//
// WHAT IT REPLACES. `computed.ts` answers the same question in closed form:
// P(tenpai) from a (副露数 × 巡目) table, times a wait-shape survival count,
// times a yaku-class factor. Every one of those three is a hand-written model
// with hand-fitted constants, and `scripts/calibrate_report.ts` has been saying
// for two milestones WHERE it is wrong rather than HOW to fix it. M14 keeps the
// counting — the counts are the features — and hands the JUDGMENT to a small
// MLP trained on the truth the calibration lane already records (`R`, the types
// an opponent actually rons at that instant). The value half is NOT learned:
// `dealinValue` still comes from computed's `valueOnType`, so this head moves
// exactly one factor of the product (plan D5).
//
// ===========================================================================
// THE 河読み BAN — why every feature below is a statement about NOW
// ===========================================================================
//
// The dojo's constitution forbids reading the river: no "he discarded 9p at 3巡
// then 8p at 5巡, so he is collecting…". That is not a performance rule, it is
// what makes this seat's estimates the same estimates a human at the table can
// make from the board, and it is why `computed.ts` is built out of counting and
// nothing else. A learned head is exactly the thing that would break it by
// accident: give an MLP the river IN ORDER and it will find the tedashi/timing
// signal whether or not anybody asked it to.
//
// So the feature list is frozen, hand-written, and every entry is checkable
// against one question — "is this a property of the board as it stands, that
// two identical boards reached by different orders would share?":
//
//   * unseen counts, own hand counts, dora, aka, ranks, suits — pure inventory;
//   * genbutsu (a SET — furiten is a proof, and a proof has no order),
//     スジ half/full, 壁 up/down — all derived from that set and the counts;
//   * the eight `ShapeBase` counts — `computed.ts`'s own parameter-free wait
//     survival, which is a function of (unseen, genbutsu) and nothing else;
//   * the opponent's public state — declared riichi (a fact, with its junme,
//     which is public and announced), melds and what they show, visible melded
//     dora, the tenpai-table cell, scores, kyoku, honba, kyotaku.
//
// Nothing carries a river POSITION, a tedashi/tsumogiri flag, a discard order,
// or "when" a tile was thrown. `test/dealin_test.ts` pins that with a
// permutation test: shuffle a river keeping its bag and its genbutsu set, and
// every feature must come back bit-identical. The tenpai head sees the river as
// a BAG (how many honors, how many terminals, how many of each suit) for the
// same reason — a bag is a count, and a count has no order.
//
// (The one thing that IS order-derived is genbutsu-after-riichi, which
// `genbutsuSets` builds: everyone's discards since a declaration lock that
// declarer out. That is a rule of the game — the wait is frozen at the
// declaration — not a behavioural read, and computed.ts has always used it.)
//
// ===========================================================================
// ONE STATE BUILDER, TWO CONSUMERS
// ===========================================================================
//
// The head is trained OFFLINE on a v3 calibration lane and served ONLINE inside
// a decision, and the two must see the same numbers to the last bit or the
// training is measuring a player nobody ships. That is enforced structurally:
// `dealinStateOf(obs, trace)` is the ONLY way a `DealinState` is built from a
// live board, `dealinFeatures` is the ONLY way features are built from a state,
// and the calibration record carries exactly the fields a state needs — so
// `scripts/dealin_export.ts` rebuilds the state from the record and runs the
// same `dealinFeatures`. `buildCalibRecord` stores `fh`, a digest of the served
// feature rows, and the export verifies every row against it. A drift between
// the two paths is then a loud number, not a silent one.
//
// The head reads the trace `computedReads` already fills (`ComputedTraceRef`),
// which is why `learnedReads` takes the same ref the provider was built with —
// the same wiring `calibrationReads` uses, and the same loud error when a caller
// forgets it.

import { tileType } from "mjrender/tiles.ts";
import { countsFromTiles } from "../kernel.ts";
import type { Observation } from "../observe.ts";
import type { Reads, ReadsProvider } from "./augmented.ts";
import type { CalibRecord } from "./calibration.ts";
import { basesFromRecord, decode34 } from "./calibration.ts";
import type {
  ComputedTrace,
  ComputedTraceRef,
  ComputedWeights,
  ShapeBase,
  YakuClass,
} from "./computed.ts";
import { genbutsuSets, valueOnType } from "./computed.ts";
import { publicUnseen } from "./planner.ts";
import type { Mlp, MlpSpec } from "./mlp.ts";
import { buildMlp, closeMlp, mlpForwardBatch, validateMlp } from "./mlp.ts";

/**
 * FEATURE VERSION. Bumped whenever `DEALIN_FEATURES` or `TENPAI_FEATURES`
 * changes in any way — an added column, a re-ordered one, a changed scale.
 * `validateMlp` refuses a block whose `fv` is not this, so a weight file fitted
 * on an older lane is rejected instead of being fed the wrong columns.
 */
export const DEALIN_FV = 1;

/**
 * The deal-in head's input, per (opponent, tile type). FROZEN ORDER: the index
 * of every name below is a column of every lane file and of every trained
 * weight matrix, so a change here is a `DEALIN_FV` bump and a new lane.
 *
 * The scale in a name is the divisor applied to the raw quantity (`junme/18`
 * is `junme / 18`); everything else is 0/1 or a small count left raw. The
 * normalisers are the natural ceilings of the game (18 turns, 70 wall tiles,
 * 4 copies, the 32 of a リャンメン mass) rather than fitted statistics, so the
 * columns mean the same thing on every board.
 */
export const DEALIN_FEATURES: readonly string[] = [
  // --- inventory of this tile type -----------------------------------------
  "unseen/4", // copies not visible to us
  "own/4", // copies in our own hand (drawn tile included)
  "unseenFrac", // unseen[ty] / Σ unseen — the draw/hold share
  // --- what public facts PROVE ---------------------------------------------
  "genbutsu", // they passed it ⇒ furiten ⇒ they cannot ron it
  "sujiHalf", // one リャンメン holding refuted, the other still live
  "sujiFull", // every リャンメン holding that reaches it is refuted
  "kabeUp", // the bridge tile two ranks up has 0 unseen copies
  "kabeDown", // …two ranks down
  // --- computed.ts's own parameter-free wait survival, /32 ------------------
  "ryanmen/32",
  "ryanmenDora/32",
  "ryanmenHalf/32",
  "ryanmenFull/32",
  "kanchan/32",
  "penchan/32",
  "shanpon/32",
  "tanki/32",
  // --- what this tile IS ---------------------------------------------------
  "dora", // indicator-derived count on the type (0..4)
  "aka", // the type a red five lives in (5m/5p/5s)
  "valueHonorFor", // 役牌 for THIS opponent (round wind, their seat wind, dragons)
  "honor",
  "terminal",
  "rank1",
  "rank2",
  "rank3",
  "rank4",
  "rank5",
  "rank6",
  "rank7",
  "rank8",
  "rank9",
  "suitM",
  "suitP",
  "suitS",
  "suitZ",
  // --- this opponent's public state ----------------------------------------
  "riichi",
  "riichiJunme/18", // 0 when they have not declared
  "junme/18",
  "wall/70",
  "open", // they have at least one non-ankan meld
  "melds/4", // total melds, ankan included — the tenpai table's row
  "honitsuMatch", // 染め手模様 and this type is in (or hot alongside) that suit
  "honitsuOther", // 染め手模様 and this type is not
  "toitoi", // トイトイ模様
  "yakuhaiMelds", // 役牌 triplet melds showing
  "meldDora", // dora visible inside their melds, aka included
  // --- the table ------------------------------------------------------------
  "tpCell/20", // which cell of the (副露数 × 巡目) table they land in
  "tpPrior", // computed's own P(tenpai) — the model this head is beating
  "otherRiichi", // some OTHER seat has declared (us included, them excluded)
  "dealer", // this opponent is 親
  "selfDealer", // WE are 親
  "honba",
  "kyotaku",
  "scoreRel/30", // (their points − ours) / 100 / 30
  "kyoku/7",
];

/** Columns of one deal-in feature row. */
export const DEALIN_F = DEALIN_FEATURES.length;

/**
 * The tenpai head's input, per opponent. FROZEN ORDER, same contract as above.
 *
 * THE RIVER IS A BAG. The first six columns are the only place a river enters
 * this module at all, and they are counts over the opponent's own discards with
 * every trace of order removed — how many tiles, how many honors, how many
 * terminals, how many of each suit. That is public inventory (it is the same
 * number whichever order the tiles arrived in), and it is what a human means by
 * "he has been throwing honors". No position, no tedashi, no timing.
 *
 * NO `riichi` COLUMN, deliberately: a declared riichi IS tenpai by the rules of
 * the game, so `learnedReads` answers 1 without consulting the head and
 * `train/dealin_fit.py` drops those rows from the fit. A column that is always
 * 0 in training and sometimes 1 in serving is exactly the kind of dead weight
 * that silently becomes a bug.
 */
export const TENPAI_FEATURES: readonly string[] = [
  "riverLen/18",
  "riverHonors/6",
  "riverTerminals/8",
  "riverM/6",
  "riverP/6",
  "riverS/6",
  "junme/18",
  "wall/70",
  "melds/4",
  "openMelds/4",
  "yakuhaiMelds",
  "honitsu",
  "toitoi",
  "meldDora",
  "otherRiichi",
  "dealer",
  "selfDealer",
  "tpPrior",
  "tpCell/20",
  "scoreRel/30",
  "kyoku/7",
  "honba",
];

/** Columns of one tenpai feature row. */
export const TENPAI_F = TENPAI_FEATURES.length;

// ---------------------------------------------------------------------------
// the state
// ---------------------------------------------------------------------------

/**
 * One opponent's half of a decision's public state.
 *
 * Every field is either read off the `Observation` (rivers, melds, riichi) or
 * taken from the `ComputedTrace` the counting model already produced on its way
 * to its own answer (`base`, `cell`, `tenpaiP`, `value`, `read`) — nothing is
 * re-derived, which is the same honesty rule `calibration.ts` states: one
 * implementation of the arithmetic, three callers.
 */
export interface DealinOppState {
  /** RELATIVE index into an `Observation` (1 = 下家 … 3 = 上家). */
  rel: number;
  riichi: boolean;
  /** 巡目 of their declaration, −1 when they have not declared. */
  riichiJunme: number;
  /** 0/1 per type: types they provably cannot ron (`genbutsuSets`). */
  genbutsu: Uint8Array;
  /** Their own discards as a BAG of type counts — no order, see the header. */
  river: Uint8Array;
  /** The parameter-free wait counts per type, from the trace. Length 34. */
  base: readonly ShapeBase[];
  /** computed's RAW P(tenpai) prior, before `tenpaiFloor` gates anything. */
  tenpaiP: number;
  /** Cell of the (副露数 × 巡目) table. */
  cellRow: number;
  cellCol: number;
  /** Total melds, ankan included. */
  melds: number;
  /** Open (non-ankan) melds. */
  openMelds: number;
  cls: YakuClass;
  /** 役牌 triplet melds showing. */
  yakuhaiMelds: number;
  /** 染め手模様: 0 none, 1 m, 2 p, 3 s. */
  honitsu: 0 | 1 | 2 | 3;
  toitoi: boolean;
  /** Dora visible in their melds, aka included. */
  meldDora: number;
  dealer: boolean;
  /** 0/1 per type: the honors that are 役牌 for this seat. */
  valueHonors: Uint8Array;
  /** Some OTHER seat has a live declaration (we count, they do not). */
  otherRiichi: boolean;
  /** `baseValueOf` for them — what `valueOnType` turns into a payment (D5). */
  value: number;
  /** Their points ÷ 100. */
  score: number;
}

/** Everything one decision's heads read. Built once per decision. */
export interface DealinState {
  junme: number;
  wall: number;
  honba: number;
  kyotaku: number;
  kyoku: number;
  /** WE are 親. */
  selfDealer: boolean;
  /** Dora count per tile TYPE from the revealed indicators (aka excluded). */
  dora: Uint8Array;
  /** Copies of each type not visible to us (`publicUnseen`). */
  unseen: Uint8Array;
  /** Σ `unseen`. */
  unseenTotal: number;
  /** Our own hand as type counts, the drawn tile included. */
  own: Uint8Array;
  /** Red fives in our own hand. */
  aka: number;
  /** Our points ÷ 100. */
  score: number;
  /** Length 3, in `Reads` order (index 0 = 下家). */
  opps: DealinOppState[];
}

const SUIT_CODE: Record<string, 0 | 1 | 2 | 3> = { m: 1, p: 2, s: 3 };

function u8from(a: readonly number[]): Uint8Array {
  const out = new Uint8Array(34);
  for (let ty = 0; ty < 34; ty++) {
    const v = a[ty] ?? 0;
    out[ty] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  return out;
}

/**
 * The one builder. Same `Observation` + same `ComputedTrace` ⇒ same state, and
 * `scripts/dealin_export.ts` rebuilds an equal state from the record the same
 * decision wrote (that equality is what `fh` checks).
 *
 * The trace is REQUIRED rather than optional: the wait counts, the tenpai cell
 * and the value base are computed's, and re-deriving them here would be a second
 * implementation of the very arithmetic this head is being trained against.
 */
export function dealinStateOf(obs: Observation, trace: ComputedTrace): DealinState {
  const unseen = publicUnseen(obs);
  const gen = genbutsuSets(obs);
  const own = countsFromTiles(obs.hand);
  let unseenTotal = 0;
  for (let ty = 0; ty < 34; ty++) unseenTotal += unseen[ty];
  let aka = 0;
  for (const t of obs.hand) if (obs.akaIds.has(t)) aka++;
  // 親 is the seat whose wind is 東; ours is `obs.seatWind`.
  const selfDealer = obs.seatWind === 27;

  const opps: DealinOppState[] = [];
  for (let i = 0; i < 3; i++) {
    const t = trace.opps[i];
    const rel = t.rel;
    const river = new Uint8Array(34);
    // A called tile was still discarded by them — it is in their bag exactly
    // once, wherever it now sits.
    for (const e of obs.rivers[rel]) river[tileType(e.tile)]++;
    const vh = new Uint8Array(34);
    for (const ty of t.valueHonors) vh[ty] = 1;
    const gb = new Uint8Array(34);
    for (const ty of gen[rel]) gb[ty] = 1;
    opps.push({
      rel,
      riichi: obs.riichi[rel],
      riichiJunme: obs.riichi[rel] ? obs.riichiJunme[rel] : -1,
      genbutsu: gb,
      river,
      base: t.base,
      tenpaiP: t.tenpaiP,
      cellRow: t.cell.row,
      cellCol: t.cell.col,
      melds: t.melds,
      openMelds: t.openMelds,
      cls: t.cls,
      yakuhaiMelds: t.read.yakuhai.size,
      honitsu: t.read.honitsuSuit === null ? 0 : SUIT_CODE[t.read.honitsuSuit],
      toitoi: t.read.toitoi,
      meldDora: t.meldDora,
      dealer: t.dealer,
      valueHonors: vh,
      otherRiichi: t.otherRiichi,
      value: t.value,
      score: Math.round(obs.scores[rel] / 100),
    });
  }

  return {
    junme: obs.junme,
    wall: obs.wallRemaining,
    honba: obs.honba,
    kyotaku: obs.kyotaku,
    kyoku: obs.kyoku,
    selfDealer,
    dora: u8from(trace.dora as readonly number[]),
    unseen: u8from(unseen),
    unseenTotal,
    own: u8from(own),
    aka,
    score: Math.round(obs.scores[0] / 100),
    opps,
  };
}

// ---------------------------------------------------------------------------
// features
// ---------------------------------------------------------------------------

/** Red-five types: 5m, 5p, 5s — the types a 赤ドラ can be. */
const isAkaType = (ty: number): boolean => ty === 4 || ty === 13 || ty === 22;

/**
 * The 34 deal-in feature rows of one opponent, row-major into `out`
 * (`out[ty * DEALIN_F + f]`, length ≥ 34 × `DEALIN_F`).
 *
 * NO ALLOCATION: the caller owns the buffer and reuses it for the life of the
 * provider. Nothing here reads anything but `st`, which is what makes the served
 * row and the exported row the same row.
 */
export function dealinFeatures(st: DealinState, opp: number, out: Float32Array): void {
  const o = st.opps[opp];
  if (out.length < 34 * DEALIN_F) {
    throw new Error(`dealinFeatures: 出力長 ${out.length} < ${34 * DEALIN_F}`);
  }
  const un = st.unseen;
  const gb = o.genbutsu;
  const total = st.unseenTotal > 0 ? st.unseenTotal : 1;
  // Type-invariant columns, read once — the same discipline `shapeMassRowFlat`
  // applies to the weights it hoists out of its own loop.
  const riichi = o.riichi ? 1 : 0;
  const rj = o.riichiJunme >= 0 ? o.riichiJunme / 18 : 0;
  const junme = st.junme / 18;
  const wall = st.wall / 70;
  const open = o.openMelds > 0 ? 1 : 0;
  const melds = o.melds / 4;
  const toitoi = o.toitoi ? 1 : 0;
  const yakuhai = o.yakuhaiMelds;
  const meldDora = o.meldDora;
  const tpCell = (o.cellRow * 4 + o.cellCol) / 20;
  const tpPrior = o.tenpaiP;
  const otherRiichi = o.otherRiichi ? 1 : 0;
  const dealer = o.dealer ? 1 : 0;
  const selfDealer = st.selfDealer ? 1 : 0;
  const honba = st.honba;
  const kyotaku = st.kyotaku;
  const scoreRel = (o.score - st.score) / 30;
  const kyoku = st.kyoku / 7;
  const hs = o.honitsu;

  for (let ty = 0; ty < 34; ty++) {
    const b = o.base[ty];
    const at = ty * DEALIN_F;
    const honor = ty >= 27;
    const suit = ty < 9 ? 1 : ty < 18 ? 2 : ty < 27 ? 3 : 0;
    const rank = honor ? 0 : (ty % 9) + 1;

    // スジ, classified exactly as `shapeBasesFlat` books it: which of the two
    // リャンメン holdings that can reach this tile are refuted by a genbutsu
    // three ranks away. Recomputed from the genbutsu SET rather than read off
    // the booked mass, because a mass is also zero when the bridge tiles are
    // simply gone (壁) — which is a different fact, and has its own columns.
    let sujiHalf = 0;
    let sujiFull = 0;
    let kabeUp = 0;
    let kabeDown = 0;
    if (!honor) {
      const up = rank <= 6;
      const dn = rank >= 4;
      const upRef = up && gb[ty + 3] === 1;
      const dnRef = dn && gb[ty - 3] === 1;
      if (up || dn) {
        const full = (!up || upRef) && (!dn || dnRef);
        if (full) sujiFull = 1;
        else if (upRef || dnRef) sujiHalf = 1;
      }
      // 壁: the bridge tile two ranks away is exhausted, so no リャンメン nor
      // カンチャン holding on that side can exist at all.
      if (rank <= 7 && un[ty + 2] === 0) kabeUp = 1;
      if (rank >= 3 && un[ty - 2] === 0) kabeDown = 1;
    }

    out[at + 0] = un[ty] / 4;
    out[at + 1] = st.own[ty] / 4;
    out[at + 2] = un[ty] / total;
    out[at + 3] = gb[ty];
    out[at + 4] = sujiHalf;
    out[at + 5] = sujiFull;
    out[at + 6] = kabeUp;
    out[at + 7] = kabeDown;
    out[at + 8] = b.ryanmen / 32;
    out[at + 9] = b.ryanmenDora / 32;
    out[at + 10] = b.ryanmenHalf / 32;
    out[at + 11] = b.ryanmenFull / 32;
    out[at + 12] = b.kanchan / 32;
    out[at + 13] = b.penchan / 32;
    out[at + 14] = b.shanpon / 32;
    out[at + 15] = b.tanki / 32;
    out[at + 16] = st.dora[ty];
    out[at + 17] = isAkaType(ty) ? 1 : 0;
    out[at + 18] = o.valueHonors[ty];
    out[at + 19] = honor ? 1 : 0;
    out[at + 20] = !honor && (rank === 1 || rank === 9) ? 1 : 0;
    for (let r = 1; r <= 9; r++) out[at + 20 + r] = rank === r ? 1 : 0;
    out[at + 30] = suit === 1 ? 1 : 0;
    out[at + 31] = suit === 2 ? 1 : 0;
    out[at + 32] = suit === 3 ? 1 : 0;
    out[at + 33] = suit === 0 ? 1 : 0;
    out[at + 34] = riichi;
    out[at + 35] = rj;
    out[at + 36] = junme;
    out[at + 37] = wall;
    out[at + 38] = open;
    out[at + 39] = melds;
    // 染め手模様 splits the types in two: the read suit and the honors are hot
    // (`combineShapes`'s `honitsuHot`), everything else is cold.
    out[at + 40] = hs !== 0 && (suit === hs || suit === 0) ? 1 : 0;
    out[at + 41] = hs !== 0 && !(suit === hs || suit === 0) ? 1 : 0;
    out[at + 42] = toitoi;
    out[at + 43] = yakuhai;
    out[at + 44] = meldDora;
    out[at + 45] = tpCell;
    out[at + 46] = tpPrior;
    out[at + 47] = otherRiichi;
    out[at + 48] = dealer;
    out[at + 49] = selfDealer;
    out[at + 50] = honba;
    out[at + 51] = kyotaku;
    out[at + 52] = scoreRel;
    out[at + 53] = kyoku;
  }
}

/**
 * One opponent's tenpai feature row into `out` (length ≥ `TENPAI_F`).
 *
 * The river enters as a bag and only as a bag — see `TENPAI_FEATURES`.
 */
export function tenpaiFeatures(st: DealinState, opp: number, out: Float32Array): void {
  const o = st.opps[opp];
  if (out.length < TENPAI_F) {
    throw new Error(`tenpaiFeatures: 出力長 ${out.length} < ${TENPAI_F}`);
  }
  let len = 0;
  let honors = 0;
  let terminals = 0;
  const suits = [0, 0, 0];
  for (let ty = 0; ty < 34; ty++) {
    const n = o.river[ty];
    if (n === 0) continue;
    len += n;
    if (ty >= 27) honors += n;
    else {
      const rank = (ty % 9) + 1;
      if (rank === 1 || rank === 9) terminals += n;
      suits[ty < 9 ? 0 : ty < 18 ? 1 : 2] += n;
    }
  }
  out[0] = len / 18;
  out[1] = honors / 6;
  out[2] = terminals / 8;
  out[3] = suits[0] / 6;
  out[4] = suits[1] / 6;
  out[5] = suits[2] / 6;
  out[6] = st.junme / 18;
  out[7] = st.wall / 70;
  out[8] = o.melds / 4;
  out[9] = o.openMelds / 4;
  out[10] = o.yakuhaiMelds;
  out[11] = o.honitsu !== 0 ? 1 : 0;
  out[12] = o.toitoi ? 1 : 0;
  out[13] = o.meldDora;
  out[14] = o.otherRiichi ? 1 : 0;
  out[15] = o.dealer ? 1 : 0;
  out[16] = st.selfDealer ? 1 : 0;
  out[17] = o.tenpaiP;
  out[18] = (o.cellRow * 4 + o.cellCol) / 20;
  out[19] = (o.score - st.score) / 30;
  out[20] = st.kyoku / 7;
  out[21] = st.honba;
}

// ---------------------------------------------------------------------------
// the weights block
// ---------------------------------------------------------------------------

/**
 * The `dealin` section of a `--ktune` file: two heads that ride inline, exactly
 * as M13's `fold` block does (plan D2 — `loadKtune` resolves no paths and a
 * `freeze` dump must be self-contained).
 *
 * There is NO identity here and `{}` is therefore an error, not a switch: a
 * deal-in head has no weight setting that reproduces the closed-form model, so
 * "an empty dealin block" can only ever be a mistake. ABSENT is the switch —
 * no `dealin` section means `learnedReads` is never built and the seat plays
 * computed's own numbers, bit for bit.
 */
export interface DealinWeights {
  fv: number;
  /** in = `DEALIN_F`, out = 1: the deal-in LOGIT (sigmoid applied in TS). */
  dealin: MlpSpec;
  /** in = `TENPAI_F`, out = 1: the tenpai LOGIT. */
  tenpai: MlpSpec;
}

/** Validate an untrusted `dealin` section. Throws — the CLI turns it into a die. */
export function mergeDealin(w?: Partial<DealinWeights>): DealinWeights {
  if (w === undefined || w === null || typeof w !== "object" || Array.isArray(w)) {
    throw new Error("dealin ブロックには重みが要ります");
  }
  const raw = w as Record<string, unknown>;
  if (raw.dealin === undefined || raw.tenpai === undefined) {
    throw new Error(
      "dealin ブロックには重みが要ります (dealin と tenpai の両方: " +
        "空の {} に恒等はありません — 学習ヘッドを外すならブロックごと消してください)",
    );
  }
  if (typeof raw.fv !== "number" || !Number.isFinite(raw.fv)) {
    throw new Error(`dealin: fv が数値ではありません (${String(raw.fv)})`);
  }
  if (raw.fv !== DEALIN_FV) {
    throw new Error(`dealin: fv=${raw.fv} は特徴量版 ${DEALIN_FV} と違います (古い重みです)`);
  }
  return {
    fv: DEALIN_FV,
    dealin: validateMlp(
      raw.dealin,
      { inputs: DEALIN_F, outputs: 1, fv: DEALIN_FV },
      "dealin.dealin",
    ),
    tenpai: validateMlp(
      raw.tenpai,
      { inputs: TENPAI_F, outputs: 1, fv: DEALIN_FV },
      "dealin.tenpai",
    ),
  };
}

// ---------------------------------------------------------------------------
// serving
// ---------------------------------------------------------------------------

/** The logistic link. One place, so the fit and the serve agree on the shape. */
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** A provider with a `close()`; whether it FREES anything depends on who built
 *  the heads — see `learnedReads`. Calling it is always safe. */
export type DealinProvider = ReadsProvider & { close(): void };

/** The two built heads. May hold native contexts, so they are freed explicitly. */
export interface DealinHeads {
  dealin: Mlp;
  tenpai: Mlp;
}

/**
 * Build both heads once. THE HARNESS MUST USE THIS: `withReads` rebuilds the
 * provider chain on every `reset` (i.e. every hanchan), and a head built inside
 * that closure would allocate a fresh native context per game and free none of
 * them. Heads are constants of the seat, exactly like `foldHead`.
 */
export function buildDealinHeads(w: DealinWeights): DealinHeads {
  return { dealin: buildMlp(w.dealin), tenpai: buildMlp(w.tenpai) };
}

/** Free both. Idempotent, like `closeMlp`. */
export function closeDealinHeads(h: DealinHeads): void {
  closeMlp(h.dealin);
  closeMlp(h.tenpai);
}

/**
 * The M14 reader: computed's answer with its `dealinP` and `tenpaiP` replaced by
 * the learned heads, and everything else passed straight through.
 *
 * WHAT MOVES
 *   dealinP     34 rows per opponent from the deal-in head, sigmoid in TS.
 *   tenpaiP     the tenpai head, gated by `tenpaiFloor` exactly as computed
 *               gates its own prior — and answered 1 without asking the head
 *               for a DECLARED riichi, because that is tenpai by the rules of
 *               the game and the head is never trained on those rows.
 *
 * WHAT DOES NOT (plan D5). `dealinValue` is rebuilt for ALL 34 types through
 * computed's `valueOnType` — the value model is not being learned, and computed
 * leaves a zero wherever its own `q ≤ 0`, which `riskOf`'s `?? expLoss` fallback
 * would not rescue (it only fires when the whole field is missing). `expLoss`,
 * `wallComposition`, `oppConcealed` and the `planner` flag are `inner`'s own
 * objects, untouched.
 *
 * PURITY. Same Observation ⇒ same Reads, no rng, no memory between calls. The
 * feature buffer and the logit buffer are provider-scoped scratch, filled and
 * drained inside one call; the rows that ESCAPE (`dealinP`, `dealinValue`) are
 * freshly allocated per decision, exactly as `computedReads` allocates its own.
 *
 * `inner` must be a `computedReads(w, traceRef)` built with the SAME `traceRef`
 * — that is how the counts arrive. A missing trace is a wiring bug and says so,
 * the same way `calibrationReads` does.
 */
export function learnedReads(
  w: DealinWeights | DealinHeads,
  cw: ComputedWeights,
  inner: ReadsProvider,
  traceRef: ComputedTraceRef,
): DealinProvider {
  // Given WEIGHTS this provider builds the heads and owns them (`close()` frees
  // them) — which is what a test or a one-shot script wants. Given HEADS it
  // borrows them and `close()` is a no-op, so the harness can build once per
  // seat and rebuild the chain per hanchan without leaking a native context.
  const owned = "fv" in w;
  const heads: DealinHeads = owned ? buildDealinHeads(w as DealinWeights) : (w as DealinHeads);
  const dealinHead = heads.dealin;
  const tenpaiHead = heads.tenpai;
  const xs = new Float32Array(34 * DEALIN_F);
  const logits = new Float32Array(34);
  const tx = new Float32Array(TENPAI_F);
  const tOut = new Float32Array(1);

  const provider = (obs: Observation): Reads | null => {
    traceRef.t = null;
    const reads = inner(obs);
    const trace = traceRef.t;
    traceRef.t = null;
    if (reads === null) return null;
    if (trace === null) {
      throw new Error("learnedReads: computedReads に同じ traceRef が渡されていません");
    }

    const st = dealinStateOf(obs, trace);
    const tenpaiP: number[] = [];
    const dealinP: Float32Array[] = [];
    const dealinValue: Float32Array[] = [];

    for (let i = 0; i < 3; i++) {
      const o = st.opps[i];
      dealinFeatures(st, i, xs);
      mlpForwardBatch(dealinHead, 34, xs, logits);
      const p = new Float32Array(34);
      const v = new Float32Array(34);
      for (let ty = 0; ty < 34; ty++) {
        p[ty] = sigmoid(logits[ty]);
        // D5: the value model answers for EVERY type, not only the ones
        // computed's own row happened to leave positive.
        v[ty] = valueOnType(cw, o.value, st.dora[ty], st.honba);
      }
      dealinP.push(p);
      dealinValue.push(v);

      // A declared riichi is a public fact and outranks any estimate.
      let pT: number;
      if (o.riichi) pT = 1;
      else {
        tenpaiFeatures(st, i, tx);
        mlpForwardBatch(tenpaiHead, 1, tx, tOut);
        pT = sigmoid(tOut[0]);
      }
      // The same gate computed applies to its own prior: under the floor the
      // read is reported as nothing, at or over it as the estimate itself.
      tenpaiP.push(pT >= cw.tenpaiFloor ? (pT < 0 ? 0 : pT > 1 ? 1 : pT) : 0);
    }

    const out: Reads = {
      tenpaiP,
      dealinP,
      dealinValue,
      expLoss: reads.expLoss,
      wallComposition: reads.wallComposition,
      oppConcealed: reads.oppConcealed,
    };
    if (reads.planner !== undefined) out.planner = reads.planner;
    return out;
  };

  provider.close = (): void => {
    if (owned) closeDealinHeads(heads);
  };
  return provider;
}

// ---------------------------------------------------------------------------
// the record path
// ---------------------------------------------------------------------------

/**
 * The `dealin` argument `buildCalibRecord` takes: a state plus a per-opponent
 * feature row, so a v3 record can carry `fh` — the digest of the rows the seat
 * WAS SERVED at that instant.
 *
 * WHY A CALLBACK rather than the rows themselves: only three of the 34×F rows
 * are ever wanted (one per opponent) and the digest consumes each immediately,
 * so a single scratch buffer serves all three. The record never stores the rows.
 */
export interface DealinRecordExtras {
  state: DealinState;
  feats: (opp: number) => Float32Array;
}

/**
 * The other direction of the one contract: the `DealinState` a v3 record
 * encodes, rebuilt exactly — same numbers, so `dealinFeatures` over it produces
 * the row the seat was served and `fh` verifies it.
 *
 * Lives HERE, beside `dealinStateOf`, rather than in the export script: the two
 * directions of a serialisation are one decision, and splitting them across
 * files is how they drift. (The import of `calibration.ts` is one-way at
 * runtime — that module takes only a TYPE from this one, and `import type` is
 * erased.)
 *
 * A v2 record, or a v3 record with a field missing, is REFUSED by name. There is
 * no zero-filled degradation: a lane the export cannot rebuild is a head that
 * would train on columns it will never be served.
 */
export function dealinStateFromRecord(rec: CalibRecord): DealinState {
  const need = (v: unknown, what: string): void => {
    if (v === undefined || v === null) {
      throw new Error(`dealin: この記録には ${what} がありません (v3 の較正レーンが要ります)`);
    }
  };
  need(rec.un, "un");
  need(rec.oh, "oh");
  need(rec.ak, "ak");
  need(rec.sc, "sc");
  need(rec.ri, "ri");
  need(rec.rj, "rj");
  const unseen = decode34(rec.un!);
  const own = decode34(rec.oh!);
  const dora = decode34(rec.dr);
  const sc = rec.sc!;
  const rj = rec.rj!;
  const ri = rec.ri!;
  let unseenTotal = 0;
  for (let ty = 0; ty < 34; ty++) unseenTotal += unseen[ty];

  const opps: DealinOppState[] = [];
  for (let i = 0; i < 3; i++) {
    const o = rec.o[i];
    const rel = i + 1;
    need(o.gb, `o[${i}].gb`);
    need(o.rb, `o[${i}].rb`);
    if (o.gb!.length !== 34) throw new Error(`dealin: gb は34文字です ("${o.gb}")`);
    const gb = new Uint8Array(34);
    for (let ty = 0; ty < 34; ty++) gb[ty] = o.gb![ty] === "1" ? 1 : 0;
    const river = u8from(decode34(o.rb!));
    const vh = new Uint8Array(34);
    for (let h = 0; h < 7; h++) if ((o.vh & (1 << h)) !== 0) vh[27 + h] = 1;
    const declared = (ri & (1 << rel)) !== 0;
    opps.push({
      rel,
      riichi: declared,
      riichiJunme: declared ? rj[rel] : -1,
      genbutsu: gb,
      river,
      base: basesFromRecord(o),
      tenpaiP: o.tp,
      cellRow: o.tr,
      cellCol: o.tc,
      melds: o.ml,
      openMelds: o.om,
      cls: o.yc,
      yakuhaiMelds: o.yh,
      honitsu: o.hs,
      toitoi: o.to === 1,
      meldDora: o.md,
      dealer: o.dl === 1,
      valueHonors: vh,
      otherRiichi: o.or === 1,
      value: o.vb,
      score: sc[rel],
    });
  }

  return {
    junme: rec.j,
    wall: rec.w,
    honba: rec.b,
    kyotaku: rec.c,
    kyoku: rec.k,
    selfDealer: rec.sw === 27,
    dora: u8from(dora),
    unseen: u8from(unseen),
    unseenTotal,
    own: u8from(own),
    aka: rec.ak!,
    score: sc[0],
    opps,
  };
}

/**
 * Build the extras for one decision. Handed to `buildCalibRecord` as its
 * optional fourth argument by the `--calibrate` sink (see `runs/dealin/WIRING.md`
 * — a one-line change in `calibrationReads`).
 *
 * WITHOUT it the record still carries every v3 field this module's features are
 * built from (`un`, `oh`, `ak`, `sc`, `ri`, `rj`, `gb`, `rb`): those need no
 * model, so a lane recorded on the plain computed champion — which is how the
 * FIRST lane must be recorded, since a head cannot be trained on its own
 * outputs — already exports. `fh` is then absent and the reproduction check is
 * skipped with a printed note; `scripts/dealin_export.ts` says so out loud.
 */
export function dealinRecordExtras(
  obs: Observation,
  trace: ComputedTrace,
): DealinRecordExtras {
  const state = dealinStateOf(obs, trace);
  const scratch = new Float32Array(34 * DEALIN_F);
  return {
    state,
    feats: (opp: number) => {
      dealinFeatures(state, opp, scratch);
      return scratch;
    },
  };
}
