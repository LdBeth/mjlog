#!/usr/bin/env -S deno run --allow-read --allow-write
// 較正当てはめ (M10c) — 最尤で 計算 の放銃模型の名前つき定数を実測に合わせる。
//
//   deno run --allow-read --allow-write scripts/calibrate_fit.ts \
//     runs/calib/m10b-700000.jsonl --out weights/computed-calibrated.json \
//     --holdout runs/calib/m10b-750000-holdout.jsonl
//
// WHAT THIS IS. `scripts/calibrate_report.ts` says WHERE the model is losing to
// the oracle; this says WHAT TO SET. The record format was designed for exactly
// this (see the header of `src/ai/calibration.ts`): every line carries the
// parameter-FREE counts a prediction was built from, so the model is a closed
// form in `ComputedWeights` over cached features and a fit is an optimisation
// rather than a tournament of selfplay runs.
//
// THE HONESTY RULE. Nothing here re-implements the model's arithmetic as a
// second opinion. Components (a) and (c) call the module's own functions
// (`tenpaiPriorOf`, `baseValueOf`, `valueOnType`) on the packed features, which
// is what `tenpaiFromRecord` / `valueFromRecord` do with the same arguments.
// Component (b) needs a whole 34-type row per gradient evaluation and the
// closed-form path allocates ~40 objects to produce one, so it has an INLINE
// twin (`condRowInline`) — and `test/calibrate_fit_test.ts` pins that twin to
// `condRowFromRecord` ELEMENT BY ELEMENT, bit for bit, over real recorded
// boards at several random weight vectors. A twin that is tested equal is not a
// second implementation; a twin that is merely believed equal would be.
//
// ===========================================================================
// THE OBJECTIVE
// ===========================================================================
//
// The model factors, and so does the loss — the same three components the
// report ranks:
//
//   (a) 聴牌: BCE of `tenpaiPriorOf` against `tt`, over SILENT opponents only.
//       A declared riichi is tenpai by the rules of the game; scoring the model
//       on it would credit it with knowing what the table announced. Exactly
//       the report's exclusion.
//   (b) 待ち: BCE of P(rons ty | tenpai) — `condRowFromRecord`, which is the
//       normalized wait row times the class yaku factor — against the oracle's
//       `R`, over the 34 types of every TRUE-TENPAI opponent row.
//   (c) 打点: Huber on log(predicted payment / actual payment) over the cells
//       the oracle says are live ron tiles.
//
// WHY (b) IS CONDITIONED ON TRUE TENPAI, which is the one modelling choice the
// brief left open. The alternative is to score the PRODUCT
// P(tenpai)×P(ron|tenpai) over every row and let one joint gradient flow. Both
// are defensible; conditioning wins here for three reasons, in order of weight:
//
//   1. IT IS WHAT THE REPORT GRADES. `calibrate_report.ts` scores its (b)
//      component on `o.tt === 1` rows and says why in a comment ("scoring it
//      twice would hide a wait model behind a good base rate"). A fit whose
//      objective is not the published metric is optimising something else.
//   2. THE PARAMETER SETS BECOME DISJOINT. Under the joint form the tenpai
//      table appears in both (a) and (b), and the two disagree: (a) wants the
//      calibrated base rate, (b) wants whatever level makes the product fit,
//      and the wait parameters absorb the difference. Conditioning gives each
//      factor exactly the data that identifies it. (A consequence worth stating
//      plainly: with disjoint parameters the component weights --wa/--wb/--wc
//      change only the REPORTED total, never the fitted vector.)
//   3. IT IS ~8× CHEAPER. Only 13% of opponent rows are truly tenpai, and only
//      those rows need their 34×8 shape counts held in memory.
//
// The joint variant is not thereby unmeasured: the holdout report at the end
// prints BOTH objectives (conditional and joint) at default and at fitted
// weights, so the choice is auditable rather than asserted.
//
// ===========================================================================
// PARAMETERISATION
// ===========================================================================
//
// Fitted in a transformed space so every constraint holds BY CONSTRUCTION and
// the optimiser never has to be clipped: probabilities, table cells and the two
// スジ残存率 through a logit; positive multipliers and point figures through a
// log. Adam then runs unconstrained.
//
// FOUR PARAMETERS ARE PINNED, not because they do not matter but because the
// normalization makes them GAUGE — moving them and rescaling their partner
// leaves every prediction bit-identical, so a fit would wander along that
// direction reporting nothing:
//   yakuFactor.riichi = 1   a declared riichi can always call ron. It is the
//                           reference class, and `expWaitMass` carries the
//                           level: the riichi row sums to `expWaitMass` types.
//   shapePrior.リャンメン    the reference shape; the other four are ratios to it.
//   honitsuHot              only 熱/冷 の比 survives the row normalization.
//   toitoiPair              likewise, only 対子形/順子形 の比 survives.
//
// NOT FITTED, deliberately:
//   valuePerHonba  exactly 300 by the rules (`score.ts#ronValue`), not a model.
//   dealinScale    the un-normalized path's calibration constant; `waitNormalize`
//                  is ON in the fitted vector, so it is not consumed at all.
//   tenpaiFloor    consumption-side. It gates whether the POLICY listens to a
//                  read; it does not change the read. Fitting it against
//                  prediction loss would be fitting the wrong thing.
//   planner, junmeBuckets  structure, not judgment.
//
// SMOOTHING. A light L2 pull toward the starting vector, in the transformed
// space (`--l2`). The 副露3・序盤 cell is hit by a few hundred rows out of a
// million; without a prior its logit is free to run to an edge on noise. With
// one, a cell moves in proportion to how much data argues for moving it, which
// is the Laplace reading of the same idea.
//
// SPLIT. Train/val BY GAME SEED (`s`), never by row: two decisions of one
// hanchan share almost everything, so a row-wise split would leak. Model
// selection (which epoch's vector to keep) is on the val half; the file passed
// to --holdout is never read until the final report.

import type { CalibRecord } from "../src/ai/calibration.ts";
import {
  baseValueFromRecord,
  condRowFromRecord,
  dealinRowFrom,
  decode34,
  scanCalibration,
  tenpaiFromRecord,
} from "../src/ai/calibration.ts";
import type { ComputedWeights, WaitShape } from "../src/ai/computed.ts";
import {
  baseValueOf,
  DEFAULT_COMPUTED,
  mergeComputed,
  tenpaiPriorOf,
  valueOnType,
} from "../src/ai/computed.ts";

function die(msg: string): never {
  console.error(msg);
  Deno.exit(2);
}

// ---------------------------------------------------------------------------
// the packed dataset
// ---------------------------------------------------------------------------

/**
 * Component (a)'s rows: one per SILENT opponent of one decision. Everything
 * `tenpaiPriorOf` reads and nothing else — six small integers and the truth.
 */
export interface PackedA {
  n: number;
  ml: Uint8Array;
  j: Uint8Array;
  yh: Uint8Array;
  or: Uint8Array;
  md: Uint8Array;
  tt: Uint8Array;
  val: Uint8Array;
}

/**
 * Component (b)'s rows: one per TRULY TENPAI opponent, with the eight
 * parameter-free mass rows the record encodes, the decision's dora row, the
 * three class facts and the oracle's ron set as a 34-wide 0/1 row.
 */
export interface PackedB {
  n: number;
  gy: Uint8Array;
  gd: Uint8Array;
  gh: Uint8Array;
  gf: Uint8Array;
  gk: Uint8Array;
  gp: Uint8Array;
  gs: Uint8Array;
  gt: Uint8Array;
  dora: Uint8Array;
  truth: Uint8Array;
  vh: Uint8Array;
  hs: Uint8Array;
  to: Uint8Array;
  yc: Uint8Array;
  val: Uint8Array;
}

/** Component (c)'s rows: one per (opponent, live ron type) cell. */
export interface PackedC {
  n: number;
  cls: Uint8Array;
  honitsu: Uint8Array;
  md: Uint8Array;
  yh: Uint8Array;
  dl: Uint8Array;
  doraTy: Uint8Array;
  honba: Uint8Array;
  actual: Float32Array;
  val: Uint8Array;
}

/** How many rows/cells actually inform each parameter — the diff table's 支持. */
export interface Support {
  cell: number[][];
  aRows: number;
  aYakuhai: number;
  aOtherRiichi: number;
  aMeldDora: number;
  bRows: number;
  bHonitsu: number;
  bToitoi: number;
  bRiichi: number;
  bOpen: number;
  bDamaten: number;
  bRyanmen: number;
  bKanchan: number;
  bPenchan: number;
  bShanpon: number;
  bTanki: number;
  bYakuhaiPair: number;
  bDoraType: number;
  bDoraBridge: number;
  bSujiHalf: number;
  bSujiFull: number;
  cCells: number;
  cRiichi: number;
  cDamaten: number;
  cOpen: number;
  cHonitsu: number;
  cMeldDora: number;
  cYakuhai: number;
  cDealer: number;
  cDoraType: number;
  cCap: number;
}

export interface Packed {
  a: PackedA;
  b: PackedB;
  c: PackedC;
  support: Support;
  games: number;
  valGames: number;
  records: number;
}

interface Counts {
  a: number;
  b: number;
  c: number;
}

function newSupport(): Support {
  return {
    cell: Array.from({ length: 5 }, () => [0, 0, 0, 0]),
    aRows: 0,
    aYakuhai: 0,
    aOtherRiichi: 0,
    aMeldDora: 0,
    bRows: 0,
    bHonitsu: 0,
    bToitoi: 0,
    bRiichi: 0,
    bOpen: 0,
    bDamaten: 0,
    bRyanmen: 0,
    bKanchan: 0,
    bPenchan: 0,
    bShanpon: 0,
    bTanki: 0,
    bYakuhaiPair: 0,
    bDoraType: 0,
    bDoraBridge: 0,
    bSujiHalf: 0,
    bSujiFull: 0,
    cCells: 0,
    cRiichi: 0,
    cDamaten: 0,
    cOpen: 0,
    cHonitsu: 0,
    cMeldDora: 0,
    cYakuhai: 0,
    cDealer: 0,
    cDoraType: 0,
    cCap: 0,
  };
}

function countRecord(rec: CalibRecord, c: Counts): void {
  for (const o of rec.o) {
    if (o.yc !== 0) c.a++;
    if (o.tt === 1) c.b++;
    c.c += o.R.length;
  }
}

function allocPacked(c: Counts): Packed {
  const u8 = (n: number) => new Uint8Array(n);
  return {
    a: {
      n: c.a,
      ml: u8(c.a),
      j: u8(c.a),
      yh: u8(c.a),
      or: u8(c.a),
      md: u8(c.a),
      tt: u8(c.a),
      val: u8(c.a),
    },
    b: {
      n: c.b,
      gy: u8(c.b * 34),
      gd: u8(c.b * 34),
      gh: u8(c.b * 34),
      gf: u8(c.b * 34),
      gk: u8(c.b * 34),
      gp: u8(c.b * 34),
      gs: u8(c.b * 34),
      gt: u8(c.b * 34),
      dora: u8(c.b * 34),
      truth: u8(c.b * 34),
      vh: u8(c.b),
      hs: u8(c.b),
      to: u8(c.b),
      yc: u8(c.b),
      val: u8(c.b),
    },
    c: {
      n: c.c,
      cls: u8(c.c),
      honitsu: u8(c.c),
      md: u8(c.c),
      yh: u8(c.c),
      dl: u8(c.c),
      doraTy: u8(c.c),
      honba: u8(c.c),
      actual: new Float32Array(c.c),
      val: u8(c.c),
    },
    support: newSupport(),
    games: 0,
    valGames: 0,
    records: 0,
  };
}

/**
 * The train/val split, BY GAME SEED. FNV-1a over the seed's decimal digits so
 * that consecutive seeds (a lane is `--seed` … `--seed+N`) do not land in a
 * pattern; two tenths of the games are val. Deterministic and stateless: the
 * same seed is on the same side of the split in every run and in every file.
 */
export function isVal(seed: number): boolean {
  const s = `${seed}`;
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193) >>> 0;
  return h % 10 < 2;
}

const DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz";

/** `decode34` straight into a slice of a packed array — no intermediate array. */
function decodeInto(s: string, dst: Uint8Array, off: number): void {
  if (s.length !== 34) throw new RangeError(`decode34: 34文字が必要: "${s}"`);
  for (let i = 0; i < 34; i++) {
    const v = DIGITS.indexOf(s[i]);
    if (v < 0) throw new RangeError(`decode34: 不正な文字 "${s[i]}"`);
    dst[off + i] = v;
  }
}

function fillRecord(rec: CalibRecord, p: Packed, cur: Counts): void {
  const sp = p.support;
  const v = isVal(rec.s ?? 0) ? 1 : 0;
  const dora = decode34(rec.dr);
  p.records++;
  for (const o of rec.o) {
    if (o.yc !== 0) {
      const i = cur.a++;
      p.a.ml[i] = o.ml;
      p.a.j[i] = Math.min(255, rec.j);
      p.a.yh[i] = o.yh;
      p.a.or[i] = o.or;
      p.a.md[i] = o.md;
      p.a.tt[i] = o.tt;
      p.a.val[i] = v;
      if (v === 0) {
        sp.aRows++;
        const row = Math.min(4, o.ml);
        sp.cell[row][o.tc]++;
        // The 副露4 row is read twice: as its own base rate, and as the CEILING
        // `tenpaiPriorOf` clamps the 役牌 bump to. A 役牌 row therefore supports
        // the top cell of its column as well as its own, and saying otherwise
        // would print "支持 0" beside a cell the fit legitimately moved.
        if (o.yh > 0 && row !== 4) sp.cell[4][o.tc]++;
        if (o.yh > 0) sp.aYakuhai++;
        if (o.or === 1) sp.aOtherRiichi++;
        if (o.md > 0) sp.aMeldDora++;
      }
    }
    if (o.tt === 1) {
      const i = cur.b++;
      const off = i * 34;
      decodeInto(o.gy, p.b.gy, off);
      decodeInto(o.gd, p.b.gd, off);
      decodeInto(o.gh, p.b.gh, off);
      decodeInto(o.gf, p.b.gf, off);
      decodeInto(o.gk, p.b.gk, off);
      decodeInto(o.gp, p.b.gp, off);
      decodeInto(o.gs, p.b.gs, off);
      decodeInto(o.gt, p.b.gt, off);
      for (let ty = 0; ty < 34; ty++) p.b.dora[off + ty] = Math.min(255, dora[ty]);
      for (const ty of o.R) p.b.truth[off + ty] = 1;
      p.b.vh[i] = o.vh;
      p.b.hs[i] = o.hs;
      p.b.to[i] = o.to;
      p.b.yc[i] = o.yc;
      p.b.val[i] = v;
      if (v === 0) {
        sp.bRows++;
        if (o.hs !== 0) sp.bHonitsu++;
        if (o.to === 1) sp.bToitoi++;
        if (o.yc === 0) sp.bRiichi++;
        else if (o.yc === 1) sp.bOpen++;
        else sp.bDamaten++;
        for (let ty = 0; ty < 34; ty++) {
          if (p.b.gy[off + ty] > 0) sp.bRyanmen++;
          if (p.b.gk[off + ty] > 0) sp.bKanchan++;
          if (p.b.gp[off + ty] > 0) sp.bPenchan++;
          if (p.b.gs[off + ty] > 0) sp.bShanpon++;
          if (p.b.gt[off + ty] > 0) sp.bTanki++;
          if (p.b.gd[off + ty] > 0) sp.bDoraBridge++;
          if (p.b.gh[off + ty] > 0) sp.bSujiHalf++;
          if (p.b.gf[off + ty] > 0) sp.bSujiFull++;
          if (dora[ty] > 0 && (p.b.gs[off + ty] > 0 || p.b.gt[off + ty] > 0)) sp.bDoraType++;
          if (ty >= 27 && (o.vh & (1 << (ty - 27))) !== 0 && p.b.gs[off + ty] > 0) {
            sp.bYakuhaiPair++;
          }
        }
      }
    }
    for (let k = 0; k < o.R.length; k++) {
      const i = cur.c++;
      p.c.cls[i] = o.yc;
      p.c.honitsu[i] = o.hs !== 0 ? 1 : 0;
      p.c.md[i] = o.md;
      p.c.yh[i] = o.yh;
      p.c.dl[i] = o.dl;
      p.c.doraTy[i] = Math.min(255, dora[o.R[k]]);
      p.c.honba[i] = Math.min(255, rec.b);
      p.c.actual[i] = o.V[k];
      p.c.val[i] = v;
      if (v === 0) {
        sp.cCells++;
        if (o.yc === 0) sp.cRiichi++;
        else if (o.yc === 1) sp.cOpen++;
        else sp.cDamaten++;
        if (o.hs !== 0 && o.yc === 1) sp.cHonitsu++;
        if (o.md > 0) sp.cMeldDora++;
        if (o.yh > 0) sp.cYakuhai++;
        if (o.dl === 1) sp.cDealer++;
        if (dora[o.R[k]] > 0) sp.cDoraType++;
        if (o.vb >= DEFAULT_COMPUTED.valueCap) sp.cCap++;
      }
    }
  }
}

/** Pack an in-memory record list (fixtures, the recovery test). Two passes. */
export function packRecords(records: readonly CalibRecord[]): Packed {
  const c: Counts = { a: 0, b: 0, c: 0 };
  for (const r of records) countRecord(r, c);
  const p = allocPacked(c);
  const cur: Counts = { a: 0, b: 0, c: 0 };
  const games = new Set<number>();
  for (const r of records) {
    fillRecord(r, p, cur);
    games.add(r.s ?? 0);
  }
  p.games = games.size;
  for (const g of games) if (isVal(g)) p.valGames++;
  return p;
}

/**
 * Pack a lane off disk. Two streaming passes — one to size the arrays, one to
 * fill them — because a 571MB lane parses to ~60MB of typed arrays and the
 * doubling-array alternative would peak at twice that for no benefit. A pass is
 * a couple of seconds; `scanCalibration` retains nothing.
 */
export async function loadPacked(path: string, max?: number): Promise<Packed> {
  const c: Counts = { a: 0, b: 0, c: 0 };
  let n = 0;
  await scanCalibration(path, (rec) => {
    if (max !== undefined && n >= max) return;
    n++;
    countRecord(rec, c);
  });
  const p = allocPacked(c);
  const cur: Counts = { a: 0, b: 0, c: 0 };
  const games = new Set<number>();
  n = 0;
  await scanCalibration(path, (rec) => {
    if (max !== undefined && n >= max) return;
    n++;
    fillRecord(rec, p, cur);
    games.add(rec.s ?? 0);
  });
  p.games = games.size;
  for (const g of games) if (isVal(g)) p.valGames++;
  return p;
}

// ---------------------------------------------------------------------------
// the parameter vector
// ---------------------------------------------------------------------------

type Comp = "a" | "b" | "c";

/** The scalar `ComputedWeights` fields this fit may write. */
type ScalarKey =
  | "yakuhaiTenpai"
  | "tenpaiOtherRiichi"
  | "tenpaiMeldDora"
  | "yakuhaiShanpon"
  | "honitsuHot"
  | "honitsuCold"
  | "toitoiPair"
  | "toitoiRun"
  | "sujiHalfSurvive"
  | "sujiFullSurvive"
  | "doraPair"
  | "doraBridge"
  | "expWaitMass"
  | "valueRiichi"
  | "valueDamaten"
  | "valueOpen"
  | "valueHonitsu"
  | "valueYakuhai"
  | "valuePerDora"
  | "valueDealer"
  | "valueCap";

const SCALAR_KEYS: ScalarKey[] = [
  "yakuhaiTenpai",
  "tenpaiOtherRiichi",
  "tenpaiMeldDora",
  "yakuhaiShanpon",
  "honitsuHot",
  "honitsuCold",
  "toitoiPair",
  "toitoiRun",
  "sujiHalfSurvive",
  "sujiFullSurvive",
  "doraPair",
  "doraBridge",
  "expWaitMass",
  "valueRiichi",
  "valueDamaten",
  "valueOpen",
  "valueHonitsu",
  "valueYakuhai",
  "valuePerDora",
  "valueDealer",
  "valueCap",
];

interface Draft {
  tenpaiPrior: number[][];
  shapePrior: Record<WaitShape, number>;
  yakuFactor: { riichi: number; open: number; damaten: number };
  scal: Record<ScalarKey, number>;
}

function newDraft(): Draft {
  const scal = {} as Record<ScalarKey, number>;
  for (const k of SCALAR_KEYS) scal[k] = DEFAULT_COMPUTED[k];
  return {
    tenpaiPrior: DEFAULT_COMPUTED.tenpaiPrior.map((r) => [...r]),
    shapePrior: { ...DEFAULT_COMPUTED.shapePrior },
    yakuFactor: { ...DEFAULT_COMPUTED.yakuFactor },
    scal,
  };
}

/**
 * The fitted vector as a `ComputedWeights`. `waitNormalize` is ON by
 * construction: the whole point of the M10b row form is that a tenpai hand rons
 * a fixed small number of types, and `expWaitMass` is only consumed on that
 * path. `dealinScale` therefore never enters, and is left at its default.
 */
function toWeights(d: Draft): ComputedWeights {
  return {
    ...DEFAULT_COMPUTED,
    ...d.scal,
    tenpaiPrior: d.tenpaiPrior,
    shapePrior: d.shapePrior,
    yakuFactor: d.yakuFactor,
    waitNormalize: true,
  };
}

interface ParamSpec {
  /** Display name, and the ktune path it writes. */
  name: string;
  comp: Comp;
  kind: "log" | "logit";
  /** Natural-space value the fit starts from (and the L2 prior centres on). */
  init: number;
  /** `DEFAULT_COMPUTED`'s value — what the diff table compares against. */
  def: number;
  write(d: Draft, v: number): void;
  /** The same slot, read back out of a finished vector (the diff table). */
  get(w: ComputedWeights): number;
  /** Rows or cells in the training half that can move this parameter. */
  support(s: Support): number;
  /** One line of mahjong, given the direction the fit moved it. */
  read(def: number, fit: number): string;
}

const logit = (p: number) => Math.log(p / (1 - p));
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

function toTheta(p: ParamSpec, v: number): number {
  return p.kind === "log" ? Math.log(v) : logit(v);
}

function fromTheta(p: ParamSpec, x: number): number {
  return p.kind === "log" ? Math.exp(x) : sigmoid(x);
}

/**
 * 0 is not a point in logit space, and both スジ残存率 default to it (the M10a
 * binary kill). The fit starts them a hair above instead — small enough to be
 * the same model for every practical purpose, large enough that the gradient
 * exists. The L2 prior centres here too, so "no evidence" still means "the
 * refuted shape is dead".
 */
const SUJI_INIT = 0.02;

const JUNME_LABELS = ["序盤(≤6巡)", "中盤(7-9巡)", "終盤(10-12巡)", "大詰(13巡-)"];

function dir(def: number, fit: number, up: string, down: string, tol = 0.02): string {
  const rel = def === 0 ? (fit > 0 ? 1 : 0) : (fit - def) / Math.abs(def);
  if (Math.abs(rel) < tol) return "ほぼ据え置き";
  return rel > 0 ? up : down;
}

/** Every parameter this fit moves, in report order. */
export function paramSpecs(): ParamSpec[] {
  const out: ParamSpec[] = [];
  // ---- (a) 聴牌 ---------------------------------------------------------
  for (let r = 0; r < DEFAULT_COMPUTED.tenpaiPrior.length; r++) {
    for (let c = 0; c < DEFAULT_COMPUTED.tenpaiPrior[r].length; c++) {
      const def = DEFAULT_COMPUTED.tenpaiPrior[r][c];
      out.push({
        name: `tenpaiPrior[副露${r}][${JUNME_LABELS[c]}]`,
        comp: "a",
        kind: "logit",
        init: def,
        def,
        write: (d, v) => {
          d.tenpaiPrior[r][c] = v;
        },
        get: (w) => w.tenpaiPrior[r][c],
        support: (s) => s.cell[r][c],
        read: (a, b) =>
          dir(a, b, "この公開状態はもっと聴牌している", "この公開状態は思ったほど聴牌していない"),
      });
    }
  }
  out.push({
    name: "yakuhaiTenpai",
    comp: "a",
    kind: "logit",
    init: DEFAULT_COMPUTED.yakuhaiTenpai,
    def: DEFAULT_COMPUTED.yakuhaiTenpai,
    write: (d, v) => {
      d.scal.yakuhaiTenpai = v;
    },
    get: (w) => w.yakuhaiTenpai,
    support: (s) => s.aYakuhai,
    read: (a, b) => dir(a, b, "役牌の刻子はより強い聴牌の証拠", "役牌の刻子は聴牌の証拠として弱い"),
  });
  out.push({
    name: "tenpaiOtherRiichi",
    comp: "a",
    kind: "log",
    init: DEFAULT_COMPUTED.tenpaiOtherRiichi,
    def: DEFAULT_COMPUTED.tenpaiOtherRiichi,
    write: (d, v) => {
      d.scal.tenpaiOtherRiichi = v;
    },
    get: (w) => w.tenpaiOtherRiichi,
    support: (s) => s.aOtherRiichi,
    read: (a, b) =>
      dir(a, b, "他家の立直後も静かな手は押している", "他家の立直で静かな手はおりている"),
  });
  out.push({
    name: "tenpaiMeldDora",
    comp: "a",
    kind: "log",
    init: DEFAULT_COMPUTED.tenpaiMeldDora,
    def: DEFAULT_COMPUTED.tenpaiMeldDora,
    write: (d, v) => {
      d.scal.tenpaiMeldDora = v;
    },
    get: (w) => w.tenpaiMeldDora,
    support: (s) => s.aMeldDora,
    read: (a, b) =>
      dir(a, b, "副露にドラを抱えた手は聴牌が早い", "副露のドラは聴牌の証拠にならない"),
  });

  // ---- (b) 待ち ---------------------------------------------------------
  const shapes: [WaitShape, (s: Support) => number, string, string][] = [
    ["カンチャン", (s) => s.bKanchan, "嵌張待ちは思ったより多い", "嵌張待ちは思ったより少ない"],
    ["ペンチャン", (s) => s.bPenchan, "辺張待ちは思ったより多い", "辺張待ちは思ったより少ない"],
    ["シャンポン", (s) => s.bShanpon, "双碰待ちは思ったより多い", "双碰待ちは思ったより少ない"],
    ["タンキ", (s) => s.bTanki, "単騎待ちは思ったより多い", "単騎待ちは思ったより少ない"],
  ];
  for (const [sh, sup, up, down] of shapes) {
    const def = DEFAULT_COMPUTED.shapePrior[sh];
    out.push({
      name: `shapePrior.${sh}`,
      comp: "b",
      kind: "log",
      init: def,
      def,
      write: (d, v) => {
        d.shapePrior[sh] = v;
      },
      get: (w) => w.shapePrior[sh],
      support: sup,
      read: (a, b) => dir(a, b, up, down),
    });
  }
  out.push({
    name: "yakuhaiShanpon",
    comp: "b",
    kind: "log",
    init: DEFAULT_COMPUTED.yakuhaiShanpon,
    def: DEFAULT_COMPUTED.yakuhaiShanpon,
    write: (d, v) => {
      d.scal.yakuhaiShanpon = v;
    },
    get: (w) => w.yakuhaiShanpon,
    support: (s) => s.bYakuhaiPair,
    read: (a, b) => dir(a, b, "役牌の対子はより強く持たれている", "役牌の対子は言うほど持たれない"),
  });
  out.push({
    name: "honitsuCold",
    comp: "b",
    kind: "log",
    init: DEFAULT_COMPUTED.honitsuCold,
    def: DEFAULT_COMPUTED.honitsuCold,
    write: (d, v) => {
      d.scal.honitsuCold = v;
    },
    get: (w) => w.honitsuCold,
    support: (s) => s.bHonitsu,
    read: (a, b) =>
      dir(a, b, "染め手の外側は言うほど安全ではない", "染め手の外側はさらに安全 (最大の割引)"),
  });
  out.push({
    name: "toitoiRun",
    comp: "b",
    kind: "log",
    init: DEFAULT_COMPUTED.toitoiRun,
    def: DEFAULT_COMPUTED.toitoiRun,
    write: (d, v) => {
      d.scal.toitoiRun = v;
    },
    get: (w) => w.toitoiRun,
    support: (s) => s.bToitoi,
    read: (a, b) => dir(a, b, "対々模様でも順子形の待ちは残る", "対々模様の順子形はさらに薄い"),
  });
  out.push({
    name: "sujiHalfSurvive",
    comp: "b",
    kind: "logit",
    init: SUJI_INIT,
    def: DEFAULT_COMPUTED.sujiHalfSurvive,
    write: (d, v) => {
      d.scal.sujiHalfSurvive = v;
    },
    get: (w) => w.sujiHalfSurvive,
    support: (s) => s.bSujiHalf,
    read: (a, b) =>
      b > 0.05
        ? "半スジは殺しきれない — 残存率として値がつく"
        : "半スジはほぼ死んでいる (M10a と同じ)",
  });
  out.push({
    name: "sujiFullSurvive",
    comp: "b",
    kind: "logit",
    init: SUJI_INIT,
    def: DEFAULT_COMPUTED.sujiFullSurvive,
    write: (d, v) => {
      d.scal.sujiFullSurvive = v;
    },
    get: (w) => w.sujiFullSurvive,
    support: (s) => s.bSujiFull,
    read: (a, b) =>
      b > 0.05 ? "全スジにも両面の残り香がある" : "全スジの両面は死んでいる (M10a と同じ)",
  });
  out.push({
    name: "doraPair",
    comp: "b",
    kind: "log",
    init: DEFAULT_COMPUTED.doraPair,
    def: DEFAULT_COMPUTED.doraPair,
    write: (d, v) => {
      d.scal.doraPair = v;
    },
    get: (w) => w.doraPair,
    support: (s) => s.bDoraType,
    read: (a, b) => dir(a, b, "ドラは対子で持たれている", "ドラの対子持ちは過大評価だった"),
  });
  out.push({
    name: "doraBridge",
    comp: "b",
    kind: "log",
    init: DEFAULT_COMPUTED.doraBridge,
    def: DEFAULT_COMPUTED.doraBridge,
    write: (d, v) => {
      d.scal.doraBridge = v;
    },
    get: (w) => w.doraBridge,
    support: (s) => s.bDoraBridge,
    read: (a, b) => dir(a, b, "ドラを含む両面は壊されにくい", "ドラを含む両面は特別ではない"),
  });
  out.push({
    name: "expWaitMass",
    comp: "b",
    kind: "log",
    init: DEFAULT_COMPUTED.expWaitMass,
    def: DEFAULT_COMPUTED.expWaitMass,
    write: (d, v) => {
      d.scal.expWaitMass = v;
    },
    get: (w) => w.expWaitMass,
    support: (s) => s.bRows,
    read: (a, b) => `立直の手が実際にロンする牌種数 ≒ ${b.toFixed(2)}`,
  });
  out.push({
    name: "yakuFactor.open",
    comp: "b",
    kind: "logit",
    init: DEFAULT_COMPUTED.yakuFactor.open,
    def: DEFAULT_COMPUTED.yakuFactor.open,
    write: (d, v) => {
      d.yakuFactor.open = v;
    },
    get: (w) => w.yakuFactor.open,
    support: (s) => s.bOpen,
    read: (a, b) => dir(a, b, "副露手はほぼ必ず役がある", "副露手でもロンできないことがある"),
  });
  out.push({
    name: "yakuFactor.damaten",
    comp: "b",
    kind: "logit",
    init: DEFAULT_COMPUTED.yakuFactor.damaten,
    def: DEFAULT_COMPUTED.yakuFactor.damaten,
    write: (d, v) => {
      d.yakuFactor.damaten = v;
    },
    get: (w) => w.yakuFactor.damaten,
    support: (s) => s.bDamaten,
    read: (a, b) => dir(a, b, "黙聴はもっとロンしてくる", "黙聴は役がなくてロンできないことが多い"),
  });

  // ---- (c) 打点 ---------------------------------------------------------
  const values: [ScalarKey, (s: Support) => number, string, string][] = [
    ["valueRiichi", (s) => s.cRiichi, "立直の放銃は高い", "立直の放銃は安い"],
    ["valueDamaten", (s) => s.cDamaten, "黙聴の放銃は高い", "黙聴の放銃は安い"],
    ["valueOpen", (s) => s.cOpen, "副露手の放銃は高い", "副露手の放銃は安い"],
    ["valueHonitsu", (s) => s.cHonitsu, "染め手の放銃はさらに高い", "染め手の値付けは高すぎた"],
    ["valueYakuhai", (s) => s.cYakuhai, "役牌の刻子は加点が大きい", "役牌の刻子の加点は小さい"],
    [
      "valuePerDora",
      (s) => s.cDoraType + s.cMeldDora,
      "ドラ1枚の重みは大きい",
      "ドラ1枚の重みは小さい",
    ],
    ["valueDealer", (s) => s.cDealer, "親の放銃はさらに重い", "親の割増は 1.5倍 ほどではない"],
    ["valueCap", (s) => s.cCap, "頭打ちはもっと上", "頭打ちはもっと下"],
  ];
  for (const [key, sup, up, down] of values) {
    const def = DEFAULT_COMPUTED[key];
    out.push({
      name: key,
      comp: "c",
      kind: "log",
      init: def,
      def,
      write: (d, v) => {
        d.scal[key] = v;
      },
      get: (w) => w[key],
      support: sup,
      read: (a, b) => dir(a, b, up, down),
    });
  }
  return out;
}

const PARAMS = paramSpecs();

export function weightsOf(theta: Float64Array, params: ParamSpec[] = PARAMS): ComputedWeights {
  const d = newDraft();
  for (let i = 0; i < params.length; i++) params[i].write(d, fromTheta(params[i], theta[i]));
  return toWeights(d);
}

function initTheta(params: ParamSpec[] = PARAMS): Float64Array {
  const t = new Float64Array(params.length);
  for (let i = 0; i < params.length; i++) t[i] = toTheta(params[i], params[i].init);
  return t;
}

// ---------------------------------------------------------------------------
// the forward: (b)'s inline twin
// ---------------------------------------------------------------------------

/**
 * `condRowFromRecord` over packed arrays: P(rons `ty` | tenpai) for one row.
 *
 * BIT-EXACT WITH THE CLOSED FORM, and that is a tested claim rather than a
 * hopeful one — `test/calibrate_fit_test.ts` compares this against
 * `condRowFromRecord` element by element on real recorded boards under several
 * random weight vectors. Every association below is copied from
 * `combineShapes` / `waitRowFrom` deliberately: `prK * kanchan / 16` really is
 * `(prK * kanchan) / 16` there while リャンメン really is `prR * (mass / 32)`,
 * and float multiplication is not associative. The reason to have a twin at all
 * is cost: the closed form allocates 34 `ShapeBase` objects, 34 shape records
 * and three `Float64Array`s per call, and a numeric gradient calls it 29 times
 * per row.
 *
 * BOTH paths are mirrored. The fit always sets `waitNormalize`, but the report
 * blocks compare against the SHIPPED vector, which does not — and a comparison
 * that silently scored the incumbent through the wrong branch would be worse
 * than no comparison.
 */
export function condRowInline(
  b: PackedB,
  i: number,
  w: ComputedWeights,
  out: Float64Array,
): Float64Array {
  const off = i * 34;
  const prR = w.shapePrior["リャンメン"];
  const prK = w.shapePrior["カンチャン"];
  const prP = w.shapePrior["ペンチャン"];
  const prS = w.shapePrior["シャンポン"];
  const prT = w.shapePrior["タンキ"];
  const dB = w.doraBridge - 1;
  const sH = w.sujiHalfSurvive;
  const sF = w.sujiFullSurvive;
  const yakuhaiShanpon = w.yakuhaiShanpon;
  const doraPairW = w.doraPair;
  const hot = w.honitsuHot;
  const cold = w.honitsuCold;
  const tPair = w.toitoiPair;
  const tRun = w.toitoiRun;
  const hs = b.hs[i];
  const toi = b.to[i] === 1;
  const vh = b.vh[i];
  const yaku = b.yc[i] === 0
    ? w.yakuFactor.riichi
    : b.yc[i] === 1
    ? w.yakuFactor.open
    : w.yakuFactor.damaten;

  let total = 0;
  for (let ty = 0; ty < 34; ty++) {
    const k = off + ty;
    const ryanmen = b.gy[k] + dB * b.gd[k] + sH * b.gh[k] + sF * b.gf[k];
    const doraPair = b.dora[k] > 0 ? doraPairW : 1;
    const valueHonor = ty >= 27 && (vh & (1 << (ty - 27))) !== 0;
    let sRyan = prR * (ryanmen / 32);
    let sKan = prK * b.gk[k] / 16;
    let sPen = prP * b.gp[k] / 16;
    let sSha = prS * (b.gs[k] / 6) * (valueHonor ? yakuhaiShanpon : 1) * doraPair;
    let sTan = prT * (b.gt[k] / 4) * doraPair;
    // `applyMeldRead`, shape by shape and in its own order.
    const suit = ty < 9 ? 1 : ty < 18 ? 2 : ty < 27 ? 3 : 0;
    const flush = hs === 0 ? 1 : (suit === hs || suit === 0 ? hot : cold);
    const run = toi ? tRun : 1;
    const pair = toi ? tPair : 1;
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
    for (let ty = 0; ty < 34; ty++) out[ty] = yaku * Math.min(1, out[ty] * sc);
    return out;
  }
  if (total <= 0) {
    out.fill(0);
    return out;
  }
  const exp = w.expWaitMass;
  for (let ty = 0; ty < 34; ty++) {
    const m = out[ty];
    out[ty] = m <= 0 ? 0 : yaku * Math.min(1, exp * (m / total));
  }
  return out;
}

// ---------------------------------------------------------------------------
// the losses
// ---------------------------------------------------------------------------

function bce(p: number, y: number): number {
  const q = Math.min(1 - 1e-9, Math.max(1e-9, p));
  return -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
}

export interface LossOpts {
  posWeight: number;
  huber: number;
  wa: number;
  wb: number;
  wc: number;
  l2: number;
}

export const DEFAULT_LOSS: LossOpts = {
  posWeight: 1,
  huber: 0.5,
  wa: 1,
  wb: 1,
  wc: 1,
  l2: 1e-3,
};

/**
 * (a) — BCE of the tenpai prior against truth, over silent opponents.
 *
 * `tenpaiPriorOf` is the module's own function and these are exactly the
 * arguments `tenpaiFromRecord` hands it (`riichi` is false by construction: a
 * declared riichi never enters this array).
 */
export function lossA(a: PackedA, w: ComputedWeights, idx: Int32Array, from: number, to: number) {
  let s = 0;
  for (let k = from; k < to; k++) {
    const r = idx[k];
    const p = tenpaiPriorOf(w, a.ml[r], a.j[r], false, a.yh[r] > 0, a.or[r] === 1, a.md[r]);
    s += bce(p, a.tt[r]);
  }
  return to > from ? s / (to - from) : 0;
}

const COND = new Float64Array(34);

/** (b) — BCE of P(rons ty | tenpai) against the oracle's ron set, 34 per row. */
export function lossB(
  b: PackedB,
  w: ComputedWeights,
  idx: Int32Array,
  from: number,
  to: number,
  posWeight: number,
) {
  let s = 0;
  for (let k = from; k < to; k++) {
    const r = idx[k];
    condRowInline(b, r, w, COND);
    const off = r * 34;
    for (let ty = 0; ty < 34; ty++) {
      const y = b.truth[off + ty];
      s += (y === 1 ? posWeight : 1) * bce(Math.min(1, COND[ty]), y);
    }
  }
  return to > from ? s / ((to - from) * 34) : 0;
}

/** Huber on the log ratio: symmetric in "twice as much" and "half as much". */
function huber(r: number, d: number): number {
  const a = Math.abs(r);
  return a <= d ? 0.5 * r * r : d * (a - 0.5 * d);
}

/**
 * (c) — Huber on log(予測 / 真値) over the live ron cells.
 *
 * `baseValueOf` + `valueOnType` are the model's own two functions and this is
 * the composition `valueFromRecord` performs; the packed columns are the
 * `ValueFacts` a record carries plus the winning type's dora and the 本場.
 */
export function lossC(
  c: PackedC,
  w: ComputedWeights,
  idx: Int32Array,
  from: number,
  to: number,
  delta: number,
) {
  let s = 0;
  for (let k = from; k < to; k++) {
    const r = idx[k];
    const vb = baseValueOf(w, {
      cls: c.cls[r] as 0 | 1 | 2,
      honitsu: c.honitsu[r] === 1,
      meldDora: c.md[r],
      yakuhai: c.yh[r],
      dealer: c.dl[r] === 1,
    });
    const pred = valueOnType(w, vb, c.doraTy[r], c.honba[r]);
    s += huber(Math.log(Math.max(1, pred) / Math.max(1, c.actual[r])), delta);
  }
  return to > from ? s / (to - from) : 0;
}

// ---------------------------------------------------------------------------
// metrics — the report's own definitions, as numbers
// ---------------------------------------------------------------------------

export interface Metrics {
  aN: number;
  aBce: number;
  aBrier: number;
  aBaseBce: number;
  aBaseBrier: number;
  aMeanP: number;
  aRate: number;
  bN: number;
  bBce: number;
  bBrier: number;
  bBaseBce: number;
  bBaseBrier: number;
  bMeanP: number;
  bRate: number;
  cN: number;
  cMae: number;
  cBaseMae: number;
  cHuber: number;
  cBias: number;
}

/**
 * A streaming proper-scoring accumulator with its own no-skill twin — the same
 * closed forms `calibrate_report.ts#Score` uses (BCE = H(p̄), Brier = p̄(1−p̄)),
 * so a number printed here is comparable with a number printed there.
 */
class Score {
  n = 0;
  bce = 0;
  brier = 0;
  sumP = 0;
  sumY = 0;
  push(p: number, y: number): void {
    this.n++;
    this.sumP += p;
    this.sumY += y;
    this.bce += bce(p, y);
    const d = p - y;
    this.brier += d * d;
  }
  base(): { bce: number; brier: number } {
    if (this.n === 0) return { bce: 0, brier: 0 };
    const p = this.sumY / this.n;
    return { bce: p * bce(p, 1) + (1 - p) * bce(p, 0), brier: p * (1 - p) };
  }
}

export function metricsOf(
  p: Packed,
  w: ComputedWeights,
  want: 0 | 1,
  o: LossOpts,
): Metrics {
  const ta = new Score();
  for (let r = 0; r < p.a.n; r++) {
    if (p.a.val[r] !== want) continue;
    ta.push(
      tenpaiPriorOf(w, p.a.ml[r], p.a.j[r], false, p.a.yh[r] > 0, p.a.or[r] === 1, p.a.md[r]),
      p.a.tt[r],
    );
  }
  const tb = new Score();
  for (let r = 0; r < p.b.n; r++) {
    if (p.b.val[r] !== want) continue;
    condRowInline(p.b, r, w, COND);
    const off = r * 34;
    for (let ty = 0; ty < 34; ty++) tb.push(Math.min(1, COND[ty]), p.b.truth[off + ty]);
  }
  let cN = 0, cAbs = 0, cHub = 0, cPred = 0, cTruth = 0;
  const truths: number[] = [];
  for (let r = 0; r < p.c.n; r++) {
    if (p.c.val[r] !== want) continue;
    const vb = baseValueOf(w, {
      cls: p.c.cls[r] as 0 | 1 | 2,
      honitsu: p.c.honitsu[r] === 1,
      meldDora: p.c.md[r],
      yakuhai: p.c.yh[r],
      dealer: p.c.dl[r] === 1,
    });
    const pred = valueOnType(w, vb, p.c.doraTy[r], p.c.honba[r]);
    const truth = p.c.actual[r];
    cN++;
    cAbs += Math.abs(pred - truth);
    cHub += huber(Math.log(Math.max(1, pred) / Math.max(1, truth)), o.huber);
    cPred += pred;
    cTruth += truth;
    truths.push(truth);
  }
  const mean = cN === 0 ? 0 : cTruth / cN;
  let cBase = 0;
  for (const t of truths) cBase += Math.abs(t - mean);
  const ab = ta.base(), bb = tb.base();
  return {
    aN: ta.n,
    aBce: ta.n === 0 ? 0 : ta.bce / ta.n,
    aBrier: ta.n === 0 ? 0 : ta.brier / ta.n,
    aBaseBce: ab.bce,
    aBaseBrier: ab.brier,
    aMeanP: ta.n === 0 ? 0 : ta.sumP / ta.n,
    aRate: ta.n === 0 ? 0 : ta.sumY / ta.n,
    bN: tb.n,
    bBce: tb.n === 0 ? 0 : tb.bce / tb.n,
    bBrier: tb.n === 0 ? 0 : tb.brier / tb.n,
    bBaseBce: bb.bce,
    bBaseBrier: bb.brier,
    bMeanP: tb.n === 0 ? 0 : tb.sumP / tb.n,
    bRate: tb.n === 0 ? 0 : tb.sumY / tb.n,
    cN,
    cMae: cN === 0 ? 0 : cAbs / cN,
    cBaseMae: cN === 0 ? 0 : cBase / cN,
    cHuber: cN === 0 ? 0 : cHub / cN,
    cBias: cN === 0 ? 0 : (cPred - cTruth) / cN,
  };
}

/** The scalar the optimiser is actually minimising, for a split. */
export function totalOf(m: Metrics, o: LossOpts): number {
  return o.wa * m.aBce + o.wb * m.bBce + o.wc * m.cHuber;
}

// ---------------------------------------------------------------------------
// the optimiser
// ---------------------------------------------------------------------------

/** mulberry32 — small, seeded, and the only source of randomness in the fit. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(idx: Int32Array, r: () => number): void {
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    const t = idx[i];
    idx[i] = idx[j];
    idx[j] = t;
  }
}

function splitIdx(val: Uint8Array, n: number, want: 0 | 1): Int32Array {
  let c = 0;
  for (let i = 0; i < n; i++) if (val[i] === want) c++;
  const out = new Int32Array(c);
  let k = 0;
  for (let i = 0; i < n; i++) if (val[i] === want) out[k++] = i;
  return out;
}

export interface FitOpts extends LossOpts {
  epochs: number;
  lr: number;
  batch: number;
  seed: number;
  /** Print per-epoch progress. */
  verbose: boolean;
}

export const DEFAULT_FIT: FitOpts = {
  ...DEFAULT_LOSS,
  epochs: 30,
  lr: 0.05,
  batch: 8192,
  seed: 12345,
  verbose: true,
};

export interface FitResult {
  theta: Float64Array;
  weights: ComputedWeights;
  /** The epoch each component's val loss bottomed out at. */
  bestEpoch: Record<Comp, number>;
  history: { epoch: number; train: number; val: number }[];
}

/**
 * Adam over the transformed vector, with CENTRAL-DIFFERENCE gradients.
 *
 * Numeric rather than analytic, and that is a considered choice: the model is
 * ~45 parameters and the expensive component's forward is one 34-wide row, so a
 * gradient costs 2 forwards per parameter — cheap enough for minutes-long fits
 * — while an analytic gradient would mean hand-differentiating the row
 * normalization, the `min(1, ·)` clips and the two `Math.min` caps of the value
 * model, i.e. writing the model a second time in derivative form. The +h and −h
 * evaluations use the SAME minibatch, so the estimate is the exact gradient of
 * that minibatch's loss and the sampling noise cancels.
 *
 * The three components have DISJOINT parameter sets (see the header), so each
 * parameter is perturbed against its own component's loss alone — a 3×
 * saving that changes no number.
 */
export function fit(p: Packed, opts: Partial<FitOpts> = {}): FitResult {
  const o: FitOpts = { ...DEFAULT_FIT, ...opts };
  const params = PARAMS;
  const P = params.length;
  const theta = initTheta(params);
  const theta0 = Float64Array.from(theta);
  const best = Float64Array.from(theta);
  const m = new Float64Array(P);
  const v = new Float64Array(P);
  const g = new Float64Array(P);

  const trA = splitIdx(p.a.val, p.a.n, 0);
  const trB = splitIdx(p.b.val, p.b.n, 0);
  const trC = splitIdx(p.c.val, p.c.n, 0);
  const r = rng(o.seed);
  const steps = Math.max(1, Math.ceil(trA.length / o.batch));
  const batchB = Math.max(1, Math.ceil(trB.length / steps));
  const batchC = Math.max(1, Math.ceil(trC.length / steps));
  const h = 1e-3;
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  let t = 0;

  const history: FitResult["history"] = [];
  // MODEL SELECTION IS PER COMPONENT, and that is not a liberty: the three
  // components own disjoint parameters, so the val loss of each depends only on
  // its own block and the argmin over epochs factorises exactly. One global
  // stopping epoch would freeze a component that was still improving because
  // another had started to overfit.
  const bestVal: Record<Comp, number> = { a: Infinity, b: Infinity, c: Infinity };
  const bestEpoch: Record<Comp, number> = { a: -1, b: -1, c: -1 };

  for (let epoch = 1; epoch <= o.epochs; epoch++) {
    shuffle(trA, r);
    shuffle(trB, r);
    shuffle(trC, r);
    // Linear decay to a tenth of the rate: Adam with a constant step leaves the
    // last epochs rattling around the optimum instead of settling into it.
    const lr = o.lr * (1 - 0.9 * (epoch - 1) / Math.max(1, o.epochs - 1));
    for (let s = 0; s < steps; s++) {
      const a0 = s * o.batch, a1 = Math.min(trA.length, a0 + o.batch);
      const b0 = s * batchB, bb1 = Math.min(trB.length, b0 + batchB);
      const c0 = s * batchC, c1 = Math.min(trC.length, c0 + batchC);
      for (let i = 0; i < P; i++) {
        const spec = params[i];
        const x = theta[i];
        theta[i] = x + h;
        const wp = weightsOf(theta, params);
        theta[i] = x - h;
        const wm = weightsOf(theta, params);
        theta[i] = x;
        let lp = 0, lm = 0;
        if (spec.comp === "a") {
          lp = o.wa * lossA(p.a, wp, trA, a0, a1);
          lm = o.wa * lossA(p.a, wm, trA, a0, a1);
        } else if (spec.comp === "b") {
          lp = o.wb * lossB(p.b, wp, trB, b0, bb1, o.posWeight);
          lm = o.wb * lossB(p.b, wm, trB, b0, bb1, o.posWeight);
        } else {
          lp = o.wc * lossC(p.c, wp, trC, c0, c1, o.huber);
          lm = o.wc * lossC(p.c, wm, trC, c0, c1, o.huber);
        }
        g[i] = (lp - lm) / (2 * h) + 2 * o.l2 * (x - theta0[i]);
      }
      t++;
      const c1b = 1 - Math.pow(b1, t), c2b = 1 - Math.pow(b2, t);
      for (let i = 0; i < P; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        theta[i] -= lr * (m[i] / c1b) / (Math.sqrt(v[i] / c2b) + eps);
      }
    }
    const w = weightsOf(theta, params);
    const mt = metricsOf(p, w, 0, o);
    const mv = metricsOf(p, w, 1, o);
    const trainL = totalOf(mt, o), valL = totalOf(mv, o);
    history.push({ epoch, train: trainL, val: valL });
    const per: Record<Comp, number> = { a: o.wa * mv.aBce, b: o.wb * mv.bBce, c: o.wc * mv.cHuber };
    const kept: Comp[] = [];
    for (const comp of ["a", "b", "c"] as Comp[]) {
      if (per[comp] >= bestVal[comp]) continue;
      bestVal[comp] = per[comp];
      bestEpoch[comp] = epoch;
      kept.push(comp);
      for (let i = 0; i < P; i++) if (params[i].comp === comp) best[i] = theta[i];
    }
    if (o.verbose) {
      console.log(
        `  epoch ${String(epoch).padStart(3)}  学習 ${trainL.toFixed(5)}  ` +
          `検証 ${valL.toFixed(5)}  (a ${per.a.toFixed(5)} b ${per.b.toFixed(5)} ` +
          `c ${per.c.toFixed(5)})${kept.length > 0 ? "  採用 " + kept.join("") : ""}`,
      );
    }
  }
  return { theta: best, weights: weightsOf(best, params), bestEpoch, history };
}

// ---------------------------------------------------------------------------
// the ktune file
// ---------------------------------------------------------------------------

/**
 * The fitted vector as a `--ktune` partial. Only what was fitted, plus the two
 * fields that make it MEAN something on load: `waitNormalize` (the fit's whole
 * wait model is the normalized row) and the four pinned gauge constants, which
 * are written at their pinned values so the ratios the fit found survive a
 * future change of the defaults.
 */
export function ktuneOf(w: ComputedWeights): { computed: Partial<ComputedWeights> } {
  return {
    computed: {
      tenpaiPrior: w.tenpaiPrior.map((r) => [...r]),
      yakuhaiTenpai: w.yakuhaiTenpai,
      tenpaiOtherRiichi: w.tenpaiOtherRiichi,
      tenpaiMeldDora: w.tenpaiMeldDora,
      shapePrior: { ...w.shapePrior },
      yakuhaiShanpon: w.yakuhaiShanpon,
      honitsuHot: w.honitsuHot,
      honitsuCold: w.honitsuCold,
      toitoiPair: w.toitoiPair,
      toitoiRun: w.toitoiRun,
      sujiHalfSurvive: w.sujiHalfSurvive,
      sujiFullSurvive: w.sujiFullSurvive,
      doraPair: w.doraPair,
      doraBridge: w.doraBridge,
      waitNormalize: true,
      expWaitMass: w.expWaitMass,
      yakuFactor: { ...w.yakuFactor },
      valueRiichi: w.valueRiichi,
      valueDamaten: w.valueDamaten,
      valueOpen: w.valueOpen,
      valueHonitsu: w.valueHonitsu,
      valueYakuhai: w.valueYakuhai,
      valuePerDora: w.valuePerDora,
      valueDealer: w.valueDealer,
      valueCap: w.valueCap,
    },
  };
}

// ---------------------------------------------------------------------------
// the holdout pass — through the closed forms, never the twin
// ---------------------------------------------------------------------------

export interface HoldoutRow extends Metrics {
  /** The JOINT objective: P(tenpai) × P(rons ty | tenpai) over EVERY row. */
  jN: number;
  jBce: number;
  jBrier: number;
  jBaseBce: number;
  jBaseBrier: number;
}

/**
 * Score a lane under two weight vectors in one streaming pass, using
 * `calibration.ts`'s own closed forms — `tenpaiFromRecord`, `condRowFromRecord`,
 * `baseValueFromRecord`. The inline twin is a fitting-time optimisation and it
 * has no business producing the number the milestone is judged on.
 */
export async function holdoutMetrics(
  path: string,
  vectors: ComputedWeights[],
  o: LossOpts,
): Promise<HoldoutRow[]> {
  const A = vectors.map(() => new Score());
  const B = vectors.map(() => new Score());
  const J = vectors.map(() => new Score());
  const C = vectors.map(() => ({ n: 0, abs: 0, hub: 0, pred: 0 }));
  const truths: number[] = [];
  await scanCalibration(path, (rec) => {
    const dora = decode34(rec.dr);
    for (let i = 0; i < rec.o.length; i++) {
      const o0 = rec.o[i];
      for (let k = 0; k < o0.R.length; k++) truths.push(o0.V[k]);
      for (let q = 0; q < vectors.length; q++) {
        const w = vectors[q];
        const pT = tenpaiFromRecord(rec, i, w);
        if (o0.yc !== 0) A[q].push(pT, o0.tt);
        const cond = condRowFromRecord(rec, i, w);
        const row = dealinRowFrom(pT, cond);
        const truth = new Set(o0.R);
        for (let ty = 0; ty < 34; ty++) J[q].push(Math.min(1, row[ty]), truth.has(ty) ? 1 : 0);
        if (o0.tt === 1) {
          for (let ty = 0; ty < 34; ty++) {
            B[q].push(Math.min(1, cond[ty]), truth.has(ty) ? 1 : 0);
          }
        }
        const vb = baseValueFromRecord(rec, i, w);
        for (let k = 0; k < o0.R.length; k++) {
          const pred = valueOnType(w, vb, dora[o0.R[k]], rec.b);
          const c = C[q];
          c.n++;
          c.abs += Math.abs(pred - o0.V[k]);
          c.hub += huber(Math.log(Math.max(1, pred) / Math.max(1, o0.V[k])), o.huber);
          c.pred += pred;
        }
      }
    }
  });
  let sum = 0;
  for (const t of truths) sum += t;
  const mean = truths.length === 0 ? 0 : sum / truths.length;
  let baseMae = 0;
  for (const t of truths) baseMae += Math.abs(t - mean);
  baseMae = truths.length === 0 ? 0 : baseMae / truths.length;

  return vectors.map((_, q) => {
    const ab = A[q].base(), bb = B[q].base(), jb = J[q].base(), c = C[q];
    return {
      aN: A[q].n,
      aBce: A[q].bce / A[q].n,
      aBrier: A[q].brier / A[q].n,
      aBaseBce: ab.bce,
      aBaseBrier: ab.brier,
      aMeanP: A[q].sumP / A[q].n,
      aRate: A[q].sumY / A[q].n,
      bN: B[q].n,
      bBce: B[q].bce / B[q].n,
      bBrier: B[q].brier / B[q].n,
      bBaseBce: bb.bce,
      bBaseBrier: bb.brier,
      bMeanP: B[q].sumP / B[q].n,
      bRate: B[q].sumY / B[q].n,
      jN: J[q].n,
      jBce: J[q].bce / J[q].n,
      jBrier: J[q].brier / J[q].n,
      jBaseBce: jb.bce,
      jBaseBrier: jb.brier,
      cN: c.n,
      cMae: c.abs / c.n,
      cBaseMae: baseMae,
      cHuber: c.hub / c.n,
      cBias: (c.pred - sum) / c.n,
    };
  });
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

const skill = (m: number, b: number) => (b <= 0 ? 0 : 1 - m / b);
const pct = (x: number) => `${(100 * x).toFixed(2)}%`;

function fmt(v: number): string {
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(2);
  return v.toFixed(4);
}

/**
 * The interpretability deliverable: every fitted parameter, where it started,
 * where it ended, how much data argued for the move, and one line of mahjong
 * saying what the move MEANS. A number that moved a long way on eleven rows is
 * a different fact from one that moved a little on a million.
 */
export function diffTable(w: ComputedWeights, s: Support, params: ParamSpec[] = PARAMS): string {
  const L: string[] = [];
  L.push("■ 既定 → 当てはめ (感性ベクトルの差分)");
  L.push(
    `${"パラメータ".padEnd(30)}${"既定".padStart(10)}${"当てはめ".padStart(12)}` +
      `${"倍率".padStart(9)}${"支持".padStart(11)}  読み`,
  );
  let comp: Comp | null = null;
  for (const p of params) {
    if (p.comp !== comp) {
      comp = p.comp;
      L.push(
        comp === "a"
          ? "-- (a) 聴牌読み --"
          : comp === "b"
          ? "-- (b) 待ち読み --"
          : "-- (c) 打点 --",
      );
    }
    const f = p.get(w);
    const ratio = p.def === 0 ? "-" : (f / p.def).toFixed(2) + "×";
    L.push(
      `${p.name.padEnd(30)}${fmt(p.def).padStart(10)}${fmt(f).padStart(12)}` +
        `${ratio.padStart(9)}${String(p.support(s)).padStart(11)}  ${p.read(p.def, f)}`,
    );
  }
  L.push("固定: yakuFactor.riichi=1 / shapePrior.リャンメン / honitsuHot / toitoiPair");
  L.push("      (行の正規化に対してゲージ — 相方との比だけが予測を動かす)");
  L.push("不変: valuePerHonba=300 (規則), dealinScale・tenpaiFloor (消費側), planner");
  return L.join("\n");
}

function metricBlock(title: string, m: Metrics): string {
  const L: string[] = [];
  L.push(title);
  L.push("成分                        件数     モデル      基準     改善率");
  const row = (name: string, n: number, mm: number, b: number, dp = 4) =>
    L.push(
      `${name.padEnd(24)}${String(n).padStart(10)}${mm.toFixed(dp).padStart(11)}` +
        `${b.toFixed(dp).padStart(10)}${pct(skill(mm, b)).padStart(11)}`,
    );
  row("(a) 聴牌 対数損失", m.aN, m.aBce, m.aBaseBce);
  row("(a) 聴牌 Brier", m.aN, m.aBrier, m.aBaseBrier);
  row("(b) 待ち 対数損失", m.bN, m.bBce, m.bBaseBce);
  row("(b) 待ち Brier", m.bN, m.bBrier, m.bBaseBrier);
  row("(c) 打点 平均絶対誤差", m.cN, m.cMae, m.cBaseMae, 0);
  L.push(
    `(a) 予測平均 ${pct(m.aMeanP)} / 実測 ${pct(m.aRate)}   ` +
      `(b) 予測平均 ${pct(m.bMeanP)} / 実測 ${pct(m.bRate)}   ` +
      `(c) 偏り ${m.cBias >= 0 ? "+" : ""}${m.cBias.toFixed(0)}点`,
  );
  return L.join("\n");
}

// ---------------------------------------------------------------------------
// entry
// ---------------------------------------------------------------------------

async function main(argv: string[]): Promise<void> {
  const paths: string[] = [];
  let out = "weights/computed-calibrated.json";
  let holdout: string | undefined;
  let max: number | undefined;
  const o: FitOpts = { ...DEFAULT_FIT };
  const num = (arg: string, k: number) => {
    const v = Number(arg.slice(k));
    if (!Number.isFinite(v)) die(`数値が必要です: ${arg}`);
    return v;
  };
  for (const arg of argv) {
    if (arg.startsWith("--out=")) out = arg.slice(6);
    else if (arg.startsWith("--holdout=")) holdout = arg.slice(10);
    else if (arg.startsWith("--epochs=")) o.epochs = num(arg, 9);
    else if (arg.startsWith("--lr=")) o.lr = num(arg, 5);
    else if (arg.startsWith("--batch=")) o.batch = num(arg, 8);
    else if (arg.startsWith("--seed=")) o.seed = num(arg, 7);
    else if (arg.startsWith("--l2=")) o.l2 = num(arg, 5);
    else if (arg.startsWith("--pos-weight=")) o.posWeight = num(arg, 13);
    else if (arg.startsWith("--huber=")) o.huber = num(arg, 8);
    else if (arg.startsWith("--wa=")) o.wa = num(arg, 5);
    else if (arg.startsWith("--wb=")) o.wb = num(arg, 5);
    else if (arg.startsWith("--wc=")) o.wc = num(arg, 5);
    else if (arg.startsWith("--max=")) max = num(arg, 6);
    else if (arg === "--quiet") o.verbose = false;
    else if (arg.startsWith("-")) {
      die(
        `不明なオプション: ${arg}\n` +
          "使い方: calibrate_fit.ts FILE --out=PATH [--holdout=PATH] [--epochs=N] [--lr=X]\n" +
          "        [--batch=N] [--seed=N] [--l2=X] [--pos-weight=X] [--huber=X]\n" +
          "        [--wa=X --wb=X --wc=X] [--max=N] [--quiet]",
      );
    } else paths.push(arg);
  }
  if (paths.length !== 1) die("学習用の較正ファイルを1つ指定してください");

  console.log("=== 較正当てはめ (M10c) ===");
  console.log(`学習 ${paths[0]}`);
  const t0 = performance.now();
  const p = await loadPacked(paths[0], max);
  console.log(
    `読み込み ${((performance.now() - t0) / 1000).toFixed(1)}秒  ` +
      `判断 ${p.records}  半荘 ${p.games} (うち検証 ${p.valGames})`,
  );
  console.log(
    `(a) 静かな他家 ${p.a.n}行  (b) 真に聴牌 ${p.b.n}行 × 34牌種  (c) ロン牌 ${p.c.n}マス`,
  );
  console.log(
    `目的: (b)は真の聴牌に条件づけ / 正例重み ${o.posWeight} / Huber δ=${o.huber} / ` +
      `L2 ${o.l2} / 重み a${o.wa} b${o.wb} c${o.wc}`,
  );
  console.log(
    `最適化: Adam 中心差分  epochs ${o.epochs}  lr ${o.lr}  batch ${o.batch}  seed ${o.seed}`,
  );
  console.log();

  const t1 = performance.now();
  const res = fit(p, o);
  console.log(
    `当てはめ ${((performance.now() - t1) / 1000).toFixed(1)}秒  ` +
      `採用 epoch (a)${res.bestEpoch.a} (b)${res.bestEpoch.b} (c)${res.bestEpoch.c} ` +
      "— 成分ごとに検証損失が最小の回",
  );
  console.log();

  const def = mergeComputed({});
  console.log(diffTable(res.weights, p.support));
  console.log();
  for (const [name, want] of [["学習", 0], ["検証", 1]] as [string, 0 | 1][]) {
    console.log(metricBlock(`■ ${name}分割 — 既定`, metricsOf(p, def, want, o)));
    console.log(metricBlock(`■ ${name}分割 — 当てはめ`, metricsOf(p, res.weights, want, o)));
    console.log();
  }

  const outDir = out.slice(0, out.lastIndexOf("/"));
  if (outDir) Deno.mkdirSync(outDir, { recursive: true });
  Deno.writeTextFileSync(out, JSON.stringify(ktuneOf(res.weights), null, 2) + "\n");
  console.log(`書き出し ${out}`);
  console.log();

  if (holdout) {
    console.log(`■ 手つかずの評価用 ${holdout} — 閉形式で再計算 (内部の高速版は使わない)`);
    const [md, mf] = await holdoutMetrics(holdout, [def, res.weights], o);
    console.log(metricBlock("— 既定", md));
    console.log(metricBlock("— 当てはめ", mf));
    console.log();
    console.log("■ 目的関数の2つの読み方 (対数損失、評価用ファイル)");
    console.log("変種                          件数       既定   当てはめ     基準");
    const jrow = (name: string, n: number, a: number, b: number, base: number) =>
      console.log(
        `${name.padEnd(26)}${String(n).padStart(10)}${a.toFixed(4).padStart(11)}` +
          `${b.toFixed(4).padStart(11)}${base.toFixed(4).padStart(9)}`,
      );
    jrow("(b) 条件つき (当てはめ対象)", md.bN, md.bBce, mf.bBce, mf.bBaseBce);
    jrow("(b) 同時 P(聴牌)×P(待ち)", md.jN, md.jBce, mf.jBce, mf.jBaseBce);
    console.log(
      `  同時分布の Brier: 既定 ${md.jBrier.toFixed(5)} → 当てはめ ${mf.jBrier.toFixed(5)}` +
        `  (基準 ${mf.jBaseBrier.toFixed(5)}、改善率 ${pct(skill(mf.jBrier, mf.jBaseBrier))})`,
    );
    console.log();
    console.log("■ 検算コマンド");
    console.log(
      `  MJGAME_NATIVE=1 deno run --allow-read --allow-ffi --allow-env=MJGAME_NATIVE \\\n` +
        `    scripts/calibrate_report.ts ${holdout} --w=${out}`,
    );
  }
}

if (import.meta.main) await main(Deno.args);
