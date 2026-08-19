// 較正記録 — what the 計算 deal-in model predicted, and what was actually true.
//
// WHY (M10a). The oracle arm of the M8-zero ablation is worth about −0.22 rank
// on the C1 channel alone; the 計算 seat that ships converts a fraction of it.
// The difference is a MODELLING error, and it has exactly three places to hide:
//   (a) P(tenpai) — the (副露数 × 巡目) base rate says the wrong thing about who
//       is ready;
//   (b) the wait location — the shape survival count puts the mass on the wrong
//       tile types given that they ARE ready;
//   (c) the value — the static point model misprices what the deal-in costs.
// A rank measurement cannot separate those three; a per-decision record can, and
// `scripts/calibrate_report.ts` does. Nothing here changes how a seat plays: the
// recorder is an out-param on the computed provider, and the Reads the policy
// consumes are the same object with the same numbers.
//
// THE PRINCIPLE: FEATURE EXTRACTION AT RECORD TIME, CLOSED-FORM MODEL AT FIT
// TIME. A record does not merely carry the model's answer — it carries the
// parameter-FREE intermediates the answer was built from (`ShapeBase` per tile
// type, the tenpai-table cell, the meld content class, the visible dora). So an
// offline fit re-evaluates the model under any candidate weight vector by
// calling `combineShapes` / `baseValueOf` / `tenpaiPriorOf` on the cached
// features — no replay, no second implementation of the arithmetic. Those are
// the SAME functions the live seat calls (see the honesty rule in
// `evidence.ts`): the recorder never re-derives anything, it only writes down
// what `computedReads` handed it, and the fit re-enters the one true path.
//
// ===========================================================================
// THE FILE FORMAT — JSONL, one header line then one line per DECISION
// ===========================================================================
//
// Header (line 1):
//   {"v":2,"kind":"mjgame-calib","seats":"khhh","seed":N,"games":N,
//    "w":{…ComputedWeights…}}
//   `w` is the weight vector the predictions were produced with; every
//   re-evaluation below is a function of it, so a file is self-describing.
//
// Record (one per decision of the recorded seat, in play order):
//   s   game seed                          n   decision index within the game
//   k   kyoku (0 = 東1)                     b   本場        c   供託
//   j   巡目                                w   山残り
//   t   1 = turn decision (we drew), 0 = claim decision (someone discarded)
//   rw  round wind TYPE (27..30)            sw  our seat wind TYPE
//   dr  dora count per tile TYPE, 34 base-36 digits (aka excluded — aka live in
//       `md` below, since only melded aka are public)
//   o   the three opponents, RELATIVE order (index 0 = 下家 = seat+1)
//
// Per opponent (`o[i]`), the model's side — all of it public at record time:
//   tp  P(tenpai), the RAW prior before `tenpaiFloor` gates the reported field
//   tr  tenpai-table row (副露数, clamped 0..4)   tc  column (巡目 bucket)
//   ml  melds (ankan included)              om  open melds (non-ankan)
//   yc  yaku class: 0 立直 / 1 副露 / 2 ダマ  yh  役牌 triplet melds showing
//   hs  染め手模様: 0 none, 1 m, 2 p, 3 s     to  1 = トイトイ模様
//   md  dora visible in their melds (aka included)   dl  1 = 親
//   or  1 = some OTHER seat has a live 立直 (this opponent excluded, us included)
//   vh  役牌 mask for this seat: bit i = tile type 27+i (0..6)
//   vb  `baseValueOf` — the modelled payment before the winning tile's own dora
//       AND before the 本場 surcharge (which is `b` above × `valuePerHonba`)
//   gy  live リャンメン mass per type, 34 base-36 digits (0..32)
//   gd  …the part of `gy` whose bridge holds a ドラ (0..32, ≤ gy)
//   gh  リャンメン mass refuted by a 半スジ (0..16)
//   gf  …refuted by a 全スジ (0..32)
//   gk  カンチャン (0..16)   gp  ペンチャン (0..16)
//   gs  シャンポン C(u,2) (0,1,3,6)          gt  タンキ u (0..4)
//   ph  FNV-1a/32 digest of the model's own `dealinP` row as float32 bits
//
// v2 (M10b) added `or`, `gd`, `gh`, `gf` — the features the upgraded model needs
// and v1 did not carry. A v1 file cannot be re-scored under a v2 model (the
// graded スジ residue and the dora conditioning are simply not in it), so it is
// refused rather than read with zeros in those columns; regenerating a lane is a
// ~90s selfplay run. Everything upgrade 4 and 5 need was already recorded: `md`,
// `dl`, `yc`, `vb`, and 本場 as `b`.
//
// Per opponent, the ORACLE's side — ground truth off the live Table:
//   tt  1 = actually tenpai
//   R   tile TYPEs this opponent rons RIGHT NOW (furiten / yakuless / sanctioned
//       already filtered out by `oracleReads`), ascending
//   V   the exact ron payment for each type of `R`, same order
//
// WHY dealinP̂ IS NOT STORED PER TYPE, and this is the one deliberate deviation
// from the M10a brief: it is a closed-form function of the fields above (see
// `dealinRowFromRecord`), so storing ~25 float32 per opponent would triple the
// line for redundancy alone — 2000 games × 70 decisions makes that hundreds of
// megabytes. What is stored instead is `ph`, a digest of the exact float32 row.
// The reproduction claim is therefore CHECKED rather than assumed: the report
// verifies every row it re-derives against `ph` and prints the mismatch count,
// and `test/calibration_test.ts` asserts the reproduction element by element.
// A v2 line comes to roughly 1.6KB (v1 was 1.2KB: the graded スジ and
// dora-bridge rows are three more 34-digit strings), and a drifted model shows up
// as a loud number.
//
// SIZE. About 190 decisions per hanchan (turn AND claim decisions), so ~300KB
// per game: a 2000-game lane is a few hundred megabytes. Nothing reads it whole
// — `scanCalibration` streams — but it is worth pointing `--calibrate` at a
// scratch directory rather than the repo.

import type {
  ComputedOppTrace,
  ComputedTrace,
  ComputedWeights,
  ShapeBase,
  ShapeFlags,
  YakuClass,
} from "./computed.ts";
import {
  baseValueOf,
  DEFAULT_COMPUTED,
  tenpaiPriorOf,
  valueOnType,
  waitRowFrom,
} from "./computed.ts";
import type { Reads } from "./augmented.ts";
import type { Observation } from "../observe.ts";

/** Bumped whenever a field changes meaning. A reader refuses anything else. */
export const CALIB_VERSION = 2;
export const CALIB_KIND = "mjgame-calib";

/** One opponent's row of one decision. Short keys: see the schema above. */
export interface CalibOpp {
  tp: number;
  tr: number;
  tc: number;
  ml: number;
  om: number;
  yc: YakuClass;
  yh: number;
  hs: 0 | 1 | 2 | 3;
  to: 0 | 1;
  md: number;
  dl: 0 | 1;
  or: 0 | 1;
  vh: number;
  vb: number;
  gy: string;
  gd: string;
  gh: string;
  gf: string;
  gk: string;
  gp: string;
  gs: string;
  gt: string;
  ph: string;
  tt: 0 | 1;
  R: number[];
  V: number[];
}

/** One decision. `s`/`n` are stamped by the writer, which owns the game loop. */
export interface CalibRecord {
  s?: number;
  n?: number;
  k: number;
  b: number;
  c: number;
  j: number;
  w: number;
  t: 0 | 1;
  rw: number;
  sw: number;
  dr: string;
  o: CalibOpp[];
}

export interface CalibHeader {
  v: number;
  kind: string;
  seats?: string;
  seed?: number;
  games?: number;
  w: ComputedWeights;
}

// ---------------------------------------------------------------------------
// encoding: small integers as base-36 digits
// ---------------------------------------------------------------------------

const DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz";

/**
 * 34 small non-negative integers as 34 characters. Every quantity encoded this
 * way is a COUNT with a hard ceiling below 36 (a リャンメン mass is at most
 * 4·4 + 4·4 = 32), so the encoding is total and exact — no rounding anywhere in
 * a calibration record's feature half.
 */
export function encode34(a: readonly number[]): string {
  let s = "";
  for (let i = 0; i < 34; i++) {
    const v = a[i] ?? 0;
    if (!Number.isInteger(v) || v < 0 || v > 35) {
      throw new RangeError(`encode34: 0..35 の整数のみ (index ${i}: ${v})`);
    }
    s += DIGITS[v];
  }
  return s;
}

export function decode34(s: string): number[] {
  if (s.length !== 34) throw new RangeError(`decode34: 34文字が必要: "${s}"`);
  const out = new Array<number>(34);
  for (let i = 0; i < 34; i++) {
    const v = DIGITS.indexOf(s[i]);
    if (v < 0) throw new RangeError(`decode34: 不正な文字 "${s[i]}"`);
    out[i] = v;
  }
  return out;
}

/**
 * FNV-1a over the row's float32 bit patterns, little-endian and explicit so the
 * digest does not depend on the machine that wrote it. Its whole job is to fail
 * loudly if the closed form ever stops reproducing the live model.
 */
export function digestRow(row: Float32Array): string {
  const dv = new DataView(new ArrayBuffer(4));
  let h = 0x811c9dc5;
  for (let i = 0; i < row.length; i++) {
    dv.setFloat32(0, row[i], true);
    let x = dv.getUint32(0, true);
    for (let b = 0; b < 4; b++) {
      h = Math.imul(h ^ (x & 0xff), 0x01000193) >>> 0;
      x >>>= 8;
    }
  }
  return h.toString(16).padStart(8, "0");
}

// ---------------------------------------------------------------------------
// building a record
// ---------------------------------------------------------------------------

const SUIT_CODE: Record<string, 0 | 1 | 2 | 3> = { m: 1, p: 2, s: 3 };

function oppRecord(tr: ComputedOppTrace, oracle: Reads, i: number): CalibOpp {
  const gy: number[] = [];
  const gd: number[] = [];
  const gh: number[] = [];
  const gf: number[] = [];
  const gk: number[] = [];
  const gp: number[] = [];
  const gs: number[] = [];
  const gt: number[] = [];
  for (const b of tr.base) {
    gy.push(b.ryanmen);
    gd.push(b.ryanmenDora);
    gh.push(b.ryanmenHalf);
    gf.push(b.ryanmenFull);
    gk.push(b.kanchan);
    gp.push(b.penchan);
    gs.push(b.shanpon);
    gt.push(b.tanki);
  }
  let vh = 0;
  for (const ty of tr.valueHonors) if (ty >= 27 && ty <= 33) vh |= 1 << (ty - 27);

  // The oracle's row: `oracleReads` has already applied every mute (furiten,
  // no yaku, sanctioned), so a type present here is one this opponent rons now.
  const R: number[] = [];
  const V: number[] = [];
  const truth = oracle.dealinP?.[i];
  const value = oracle.dealinValue?.[i];
  if (truth) {
    for (let ty = 0; ty < 34; ty++) {
      if (truth[ty] <= 0) continue;
      R.push(ty);
      V.push(value ? value[ty] : 0);
    }
  }

  return {
    tp: tr.tenpaiP,
    tr: tr.cell.row,
    tc: tr.cell.col,
    ml: tr.melds,
    om: tr.openMelds,
    yc: tr.cls,
    yh: tr.read.yakuhai.size,
    hs: tr.read.honitsuSuit === null ? 0 : SUIT_CODE[tr.read.honitsuSuit],
    to: tr.read.toitoi ? 1 : 0,
    md: tr.meldDora,
    dl: tr.dealer ? 1 : 0,
    or: tr.otherRiichi ? 1 : 0,
    vh,
    vb: tr.value,
    gy: encode34(gy),
    gd: encode34(gd),
    gh: encode34(gh),
    gf: encode34(gf),
    gk: encode34(gk),
    gp: encode34(gp),
    gs: encode34(gs),
    gt: encode34(gt),
    ph: digestRow(tr.dealinP),
    tt: (oracle.tenpaiP?.[i] ?? 0) > 0 ? 1 : 0,
    R,
    V,
  };
}

/**
 * Pair one decision's model trace with the engine's truth for the same instant.
 *
 * Nothing is computed here that the model did not already compute: every model
 * number comes out of `trace` (which `computedReads` filled on its way to the
 * answer) and every truth number out of `oracle` (which `oracleReads` read off
 * the live Table). The function's only work is transcription.
 */
export function buildCalibRecord(
  obs: Observation,
  trace: ComputedTrace,
  oracle: Reads,
): CalibRecord {
  const dora = new Array<number>(34);
  for (let ty = 0; ty < 34; ty++) dora[ty] = trace.dora[ty] ?? 0;
  return {
    k: obs.kyoku,
    b: obs.honba,
    c: obs.kyotaku,
    j: obs.junme,
    w: obs.wallRemaining,
    t: obs.drawn !== null ? 1 : 0,
    rw: obs.roundWind,
    sw: obs.seatWind,
    dr: encode34(dora),
    o: trace.opps.map((t, i) => oppRecord(t, oracle, i)),
  };
}

// ---------------------------------------------------------------------------
// the fit-time model: closed form over the cached features
// ---------------------------------------------------------------------------

/** The per-type `ShapeBase` array a record's eight mass strings encode. */
export function basesFromRecord(o: CalibOpp): ShapeBase[] {
  const gy = decode34(o.gy);
  const gd = decode34(o.gd);
  const gh = decode34(o.gh);
  const gf = decode34(o.gf);
  const gk = decode34(o.gk);
  const gp = decode34(o.gp);
  const gs = decode34(o.gs);
  const gt = decode34(o.gt);
  const out: ShapeBase[] = [];
  for (let ty = 0; ty < 34; ty++) {
    out.push({
      ryanmen: gy[ty],
      ryanmenDora: gd[ty],
      ryanmenHalf: gh[ty],
      ryanmenFull: gf[ty],
      kanchan: gk[ty],
      penchan: gp[ty],
      shanpon: gs[ty],
      tanki: gt[ty],
    });
  }
  return out;
}

/**
 * The `ShapeFlags` the record's class fields imply for one tile type.
 *
 * `dora` is the record's own `dr` row, DECODED BY THE CALLER: the field is per
 * decision rather than per opponent (the indicators are one set of tiles for the
 * whole table), and decoding 34 digits inside a per-type loop would be the one
 * expensive thing in an otherwise linear pass. Omitting it reads as "no dora on
 * the table", which is what a fixture without indicators actually means.
 */
export function flagsFromRecord(o: CalibOpp, ty: number, dora?: readonly number[]): ShapeFlags {
  return {
    valueHonor: ty >= 27 && ty <= 33 ? (o.vh & (1 << (ty - 27))) !== 0 : false,
    honitsuSuit: o.hs === 0 ? null : (["m", "p", "s"][o.hs - 1] as "m" | "p" | "s"),
    toitoi: o.to === 1,
    doraType: (dora?.[ty] ?? 0) > 0,
  };
}

/**
 * P(tenpai) re-derived from the cached public state — NOT read back from `tp`.
 * That is the point of the exercise: with the recording weights it reproduces
 * `tp` exactly, and with a candidate table it is that candidate's prediction for
 * the same board.
 */
export function tenpaiFromRecord(
  rec: CalibRecord,
  i: number,
  w: ComputedWeights = DEFAULT_COMPUTED,
): number {
  const o = rec.o[i];
  return tenpaiPriorOf(w, o.ml, rec.j, o.yc === 0, o.yh > 0, o.or === 1, o.md);
}

/** The modelled payment before the winning tile's own dora, re-derived. */
export function baseValueFromRecord(
  rec: CalibRecord,
  i: number,
  w: ComputedWeights = DEFAULT_COMPUTED,
): number {
  const o = rec.o[i];
  return baseValueOf(w, {
    cls: o.yc,
    honitsu: o.hs !== 0,
    meldDora: o.md,
    yakuhai: o.yh,
    dealer: o.dl === 1,
  });
}

/**
 * P(waiting on each type | tenpai) × the yaku factor, re-derived per type.
 *
 * A ROW rather than 34 independent cells, because with `waitNormalize` on the
 * model is a row: each type's share is divided by the row's own total. That is
 * also why the whole thing goes through `waitRowFrom` — the live seat's own
 * function, not a copy of it (see the honesty rule in `evidence.ts`).
 */
export function condRowFromRecord(
  rec: CalibRecord,
  i: number,
  w: ComputedWeights = DEFAULT_COMPUTED,
): Float64Array {
  const o = rec.o[i];
  const bases = basesFromRecord(o);
  const dora = decode34(rec.dr);
  const yaku = o.yc === 0
    ? w.yakuFactor.riichi
    : o.yc === 1
    ? w.yakuFactor.open
    : w.yakuFactor.damaten;
  const wait = waitRowFrom(bases, (ty) => flagsFromRecord(o, ty, dora), w);
  const out = new Float64Array(34);
  for (let ty = 0; ty < 34; ty++) out[ty] = yaku * wait[ty];
  return out;
}

/**
 * The model's `dealinP` row for one opponent, rebuilt from the record alone.
 *
 * With the file's own weights this is BIT-IDENTICAL to what the seat used —
 * `digestRow(dealinRowFromRecord(…)) === rec.o[i].ph` — because the arithmetic
 * is literally the same code path (`combineShapes` via `waitRowFrom`),
 * fed the same integers, and the assignment into a Float32Array rounds the same
 * way the provider's did. With any other weights it is that vector's prediction.
 */
export function dealinRowFromRecord(
  rec: CalibRecord,
  i: number,
  w: ComputedWeights = DEFAULT_COMPUTED,
): Float32Array {
  return dealinRowFrom(tenpaiFromRecord(rec, i, w), condRowFromRecord(rec, i, w));
}

/**
 * The last multiplication, split out so a caller that already holds the
 * conditional row (the report scores it separately) does not pay for it twice —
 * and, more to the point, does not reproduce this line by hand. The `q > 0`
 * guard and the Float32Array store are what round the answer, so they belong to
 * exactly one place.
 */
export function dealinRowFrom(tenpaiP: number, cond: Float64Array): Float32Array {
  const row = new Float32Array(34);
  for (let ty = 0; ty < 34; ty++) {
    const q = tenpaiP * cond[ty];
    if (q > 0) row[ty] = q;
  }
  return row;
}

/**
 * The modelled payment of a ron on `ty`, re-derived: the per-type dora and the
 * decision's own 本場 surcharge included, through `valueOnType` — the same
 * assembly the live seat's `dealinValue` goes through.
 */
export function valueFromRecord(
  rec: CalibRecord,
  i: number,
  ty: number,
  w: ComputedWeights = DEFAULT_COMPUTED,
): number {
  const dora = decode34(rec.dr);
  return valueOnType(w, baseValueFromRecord(rec, i, w), dora[ty], rec.b);
}

// ---------------------------------------------------------------------------
// the writer
// ---------------------------------------------------------------------------

/**
 * Streams calibration JSONL. Opens (and TRUNCATES) the path: a new run is a new
 * dataset, exactly as `TrajectoryWriter` treats a trajectory file.
 *
 * The writer owns the two fields a policy has no business knowing — the game
 * seed and the decision index within the game — so the provider stays a pure
 * function of the Observation it was handed.
 */
export class CalibrationWriter {
  readonly path: string;
  private file: Deno.FsFile;
  private enc = new TextEncoder();
  private seed = 0;
  private n = 0;
  private rows = 0;
  private games = 0;

  constructor(path: string, header: Omit<CalibHeader, "v" | "kind">) {
    this.path = path;
    const dir = path.slice(0, path.lastIndexOf("/"));
    if (dir) Deno.mkdirSync(dir, { recursive: true });
    this.file = Deno.openSync(path, { create: true, write: true, truncate: true });
    this.writeLine({ v: CALIB_VERSION, kind: CALIB_KIND, ...header });
  }

  /** A new game: stamp its seed on everything that follows, restart the index. */
  beginGame(seed: number): void {
    this.seed = seed;
    this.n = 0;
    this.games++;
  }

  /** The sink handed to `calibrationReads`. Bound, so it can be passed around. */
  readonly record = (rec: CalibRecord): void => {
    rec.s = this.seed;
    rec.n = this.n++;
    this.writeLine(rec);
    this.rows++;
  };

  private writeLine(obj: unknown): void {
    const bytes = this.enc.encode(JSON.stringify(obj) + "\n");
    let n = 0;
    while (n < bytes.length) n += this.file.writeSync(bytes.subarray(n));
  }

  stats(): { games: number; rows: number } {
    return { games: this.games, rows: this.rows };
  }

  close(): void {
    this.file.close();
  }
}

// ---------------------------------------------------------------------------
// reading one back
// ---------------------------------------------------------------------------

/**
 * A file from an older schema is REFUSED, never read with the missing columns
 * filled in. v1 carried no graded スジ residue (`gh`/`gf`), no dora-bearing
 * リャンメン share (`gd`) and no other-seat-riichi flag (`or`), so re-scoring one
 * under an M10b model would silently measure a model with three of its inputs
 * pinned to zero — a wrong number that looks like a right one. Regenerating a
 * lane is a selfplay run of a couple of minutes.
 */
function versionError(path: string, v: number): Error {
  return new Error(
    `${path}: 版が違います (v${v}, 期待 v${CALIB_VERSION})。` +
      "M10b で素性が増えたため v1 の記録は再利用できません — " +
      "selfplay --calibrate で取り直してください",
  );
}

export interface CalibFile {
  header: CalibHeader;
  records: CalibRecord[];
}

/** Parse a JSONL calibration file. A wrong version is fatal, never coerced. */
export function parseCalibration(text: string, path = "<memory>"): CalibFile {
  const lines = text.split("\n").filter((l) => l.trim() !== "");
  if (lines.length === 0) throw new Error(`${path}: 空のファイルです`);
  const header = JSON.parse(lines[0]) as CalibHeader;
  if (header.kind !== CALIB_KIND) throw new Error(`${path}: 較正記録ではありません`);
  if (header.v !== CALIB_VERSION) {
    throw versionError(path, header.v);
  }
  const records = lines.slice(1).map((l) => JSON.parse(l) as CalibRecord);
  return { header, records };
}

export function readCalibration(path: string): CalibFile {
  return parseCalibration(Deno.readTextFileSync(path), path);
}

/**
 * The same file, one record at a time and nothing retained.
 *
 * A lane is ~300KB per hanchan, so the 2000-game run this format was designed
 * for is a few hundred megabytes — reading that into an array of parsed objects
 * would cost gigabytes of heap for no reason. Anything that consumes a whole
 * lane (the report) goes through here; `readCalibration` stays for fixtures and
 * tests, where holding the records is the point.
 *
 * Returns the header, which is parsed from the first line before any record is
 * delivered — so a version mismatch fails before the caller has accumulated
 * anything.
 */
export async function scanCalibration(
  path: string,
  onRecord: (rec: CalibRecord, header: CalibHeader) => void,
): Promise<CalibHeader> {
  const file = await Deno.open(path, { read: true });
  const dec = new TextDecoder();
  let header: CalibHeader | null = null;
  let rest = "";
  const take = (line: string) => {
    if (line.trim() === "") return;
    if (header === null) {
      header = JSON.parse(line) as CalibHeader;
      if (header.kind !== CALIB_KIND) throw new Error(`${path}: 較正記録ではありません`);
      if (header.v !== CALIB_VERSION) {
        throw versionError(path, header.v);
      }
      return;
    }
    onRecord(JSON.parse(line) as CalibRecord, header);
  };
  try {
    const buf = new Uint8Array(1 << 20);
    for (;;) {
      const n = await file.read(buf);
      if (n === null) break;
      rest += dec.decode(buf.subarray(0, n), { stream: true });
      let i = rest.indexOf("\n");
      while (i >= 0) {
        take(rest.slice(0, i));
        rest = rest.slice(i + 1);
        i = rest.indexOf("\n");
      }
    }
    take(rest);
  } finally {
    file.close();
  }
  if (header === null) throw new Error(`${path}: 空のファイルです`);
  return header;
}
