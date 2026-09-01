// M15 — the wire layout between `src/ai/ev.ts` and `native/mjev.cc`.
//
// ONE contract, two readers. Every offset below is mirrored by name in
// `native/mjev.cc` (`enum I_*`, `D_*`, `O_*`, `S_*`, `P_*`); `mjev_abi()` returns
// `EV_ABI`, and the wrapper refuses any other number. A change to any offset is
// an ABI bump on BOTH sides — never edit one file alone.
//
// The engine has no TypeScript twin (owner decision, 2026-08-30): what crosses
// here IS the specification of what the DP sees. Nothing hidden rides along —
// no tile ids (types only), no rivers (only the OWN river as a bag for furiten),
// and the hidden-information channels are flagged so absence is distinguishable
// from a uniform vector.

/** Bumped together with `mjev_abi()` in `native/mjev.cc`. */
export const EV_ABI = 1;

// ---------------------------------------------------------------------------
// `ints` — the integer facts of one evaluation (Int32Array, INTS_LEN)
// ---------------------------------------------------------------------------

/** hand[34]: type counts of the ROOT hand — 14 tiles for a discard root (mode 0), 13 for a rest root (mode 1); melds excluded. */
export const I_HAND = 0;
export const I_NMELDS = 34;
/** 4 × {kind, type, concealed}: kind 0 = run (type = lowest), 1 = triplet, 2 = kan; concealed 1 only for 暗槓. */
export const I_MELDS = 35;
export const I_SEAT_WIND = 47; // tile type 27..30
export const I_ROUND_WIND = 48; // tile type 27..30
export const I_DEALER = 49;
export const I_HONBA = 50;
export const I_KYOTAKU = 51;
export const I_OWN_RIICHI = 52;
export const I_FURITEN_PERM = 53;
export const I_FURITEN_TEMP = 54;
export const I_JUNME = 55;
/** own draws remaining (floor(wallRemaining/4), clamped to 20). */
export const I_T = 56;
/** red 5p copies held (0..2) / still unseen (0..2). */
export const I_AKA_HELD = 57;
export const I_AKA_UNSEEN = 58;
/** 1 when every meld is 暗槓 (the hand is menzen). */
export const I_CLOSED = 59;
/** candMask[34]: 1 for each type the root may discard (discard roots only). */
export const I_CAND = 60;
/** unseen[34]: `publicUnseen` — the draw pool unless `hasPool`. */
export const I_UNSEEN = 94;
/** doraCount[34]: dora multiplicity per type from the indicators (aka excluded). */
export const I_DORA = 128;
/** ownRiverBag[34]: types in the seat's OWN river (furiten proof). */
export const I_RIVER = 162;
/** 1 when the root is a post-暗槓 rest and a kan-dora reveal is pending. */
export const I_KANDORA_ON = 196;
export const I_HAS_DRAW = 197;
/** number of `drawDist` rows supplied (≤ KMAX). */
export const I_K = 198;
export const I_HAS_POOL = 199;
export const I_HAS_URA = 200;
export const I_HAS_NEXTDORA = 201;
/** 0 = discard root (`mjev_eval_discard`), 1 = rest root (`mjev_eval_rest`). */
export const I_MODE = 202;
export const INTS_LEN = 208;

// ---------------------------------------------------------------------------
// `dbls` — the real-valued facts (Float64Array, DBLS_LEN)
// ---------------------------------------------------------------------------

export const D_TENPAI = 0; // [3] per opponent, relative order
export const D_EXPLOSS = 3; // [3] points
/** pIn[34]: Σ_i P(opponent i rons this type NOW) for the root discard. */
export const D_PIN = 6;
/** costIn[34]: what discarding this type now costs in POINTS (the `riskOf` arithmetic, doctrine included). */
export const D_COSTIN = 40;
/** 順位効用 multipliers (1 when the layer is off). */
export const D_GAIN = 74;
export const D_RISK = 75;
/** Hidden-information block — read only when the matching flag is set. */
export const KMAX = 4;
export const D_DRAW = 76; // [KMAX][34]
export const D_POOL = 212; // [34]
export const D_URA = 246; // [34] indicator-type distribution
export const D_NEXTDORA = 280; // [34] indicator-type distribution
export const DBLS_LEN = 320;

// ---------------------------------------------------------------------------
// `out` of `mjev_eval_discard` (Float64Array, OUT_LEN)
// ---------------------------------------------------------------------------

/** per type: out[ty*4 + O_TOTAL | O_DAMA | O_RIICHI | O_FOLDLINE]; −Infinity where candMask is 0. */
export const O_STRIDE = 4;
export const O_TOTAL = 0;
export const O_DAMA = 1;
export const O_RIICHI = 2;
export const O_FOLDLINE = 3;
export const O_NODES = 136;
export const O_TRUNC = 137;
export const O_BEST_PUSH = 138;
export const O_BEST_FOLD = 139;
export const OUT_LEN = 140;

/**
 * `mjev_eval_rest` meta (Float64Array, REST_META_LEN): the value, what the
 * search cost, and — from slot 3 on — what the value is MADE OF, under the same
 * policy that produced it. The breakdown is diagnostic: no decision reads it,
 * and it exists so a hand can be audited in the terms it is played in rather
 * than as one number nobody can check.
 */
export const R_VALUE = 0;
export const R_NODES = 1;
export const R_TRUNC = 2;
/** P(the hand is ever tenpai before it ends). */
export const R_PTENPAI = 3;
/** P(this hand is the one that wins). */
export const R_PWIN = 4;
/** Points collected when it does — 0 when `R_PWIN` is 0. */
export const R_EVALUE = 5;
/** Points expected to be paid to the table on the way. */
export const R_ECOST = 6;
export const REST_META_LEN = 8;

// ---------------------------------------------------------------------------
// `mjev_score` — the stateless scorer (Int32Array in / Int32Array out)
// ---------------------------------------------------------------------------

/** counts[34] INCLUDING the winning tile, EXCLUDING meld tiles. */
export const S_COUNTS = 0;
export const S_NMELDS = 34;
export const S_MELDS = 35; // 4 × {kind, type, concealed} as in I_MELDS
export const S_WINTYPE = 47;
export const S_TSUMO = 48;
export const S_RIICHI = 49;
export const S_DOUBLE = 50;
export const S_IPPATSU = 51;
export const S_RINSHAN = 52;
export const S_CHANKAN = 53;
export const S_HAITEI = 54;
export const S_HOUTEI = 55;
export const S_TENHOU = 56;
export const S_CHIIHOU = 57;
export const S_SEAT_WIND = 58;
export const S_ROUND_WIND = 59;
export const S_DORA = 60; // [34] dora multiplicity per type
export const S_URA = 94; // [34] ura multiplicity per type
export const S_AKA = 128;
export const S_KUITAN = 129;
export const S_KAZOE = 130;
export const S_KIRIAGE = 131;
export const S_DWFU = 132; // doubleWindFu 2|4
export const S_IPPATSU_CFG = 133;
export const SCORE_IN_LEN = 136;

/** out: ok(0/1), han, fu, base, yakumanCount, limit(0..5), ronPayment, tsumoTotal. */
export const SO_OK = 0;
export const SO_HAN = 1;
export const SO_FU = 2;
export const SO_BASE = 3;
export const SO_YAKUMAN = 4;
export const SO_LIMIT = 5;
export const SO_RON = 6;
export const SO_TSUMO_TOTAL = 7;
export const SCORE_OUT_LEN = 8;

// ---------------------------------------------------------------------------
// `mjev_create` params (Float64Array, EV_PARAMS_LEN) — order fixed here,
// packed by `packEvParams` in evparams.ts, read by `enum P_*` in mjev.cc.
// ---------------------------------------------------------------------------

export const EV_PARAM_ORDER = [
  "meanUkeire0",
  "meanUkeire1",
  "meanUkeire2",
  "meanUkeire3",
  "ronFactor",
  "oppHazard",
  "oppGrowth",
  "valueRiichi",
  "valueDamaten",
  "valueOpen",
  "valueHonitsu",
  "valuePerDora",
  "valueYakuhai",
  "valueDealer",
  "valueCap",
  "dealinRate",
  "tsumoShare",
  "foldHazard",
  "riichiDealinMult",
  "ippatsuP",
  "stickAtDraw",
  "dealerRenchan",
  "callMargin",
  "riichiMargin",
  "pointsPerScore",
  "exactShanten",
  "sameShantenRungs",
  "maxNodes",
  "discard",
  "riichi",
  "calls",
  "kuitan",
  "kazoeYakuman",
  "kiriageMangan",
  "doubleWindFu",
  "notenPenaltyTotal",
] as const;
export const EV_PARAMS_LEN = EV_PARAM_ORDER.length; // 36
