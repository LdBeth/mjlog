#!/usr/bin/env -S deno run --allow-read --allow-write --allow-ffi --allow-env=MJGAME_NATIVE
// EV核の母集団スカラーの当てはめ (M15b) — 自分の手について「核が信じたこと」を、
// 実際の結末に合わせる。
//
//   MJGAME_NATIVE=1 deno run --allow-read --allow-write --allow-ffi \
//     --allow-env=MJGAME_NATIVE scripts/ev_fit.ts \
//     --in=runs/ev/lane.jsonl --out=weights/ev-0830.json [--sample=N] [--iters=N]
//
// WHAT THIS IS, AND WHAT IT IS NOT. M15's first grade found the engine
// internally consistent and the MODEL wrong: `ronFactor` 0.5 turns a ryanmen
// tenpai with 12 turns left into P(win) 0.89, where the champion's realised
// 和了率 is ~21%. A 18-cell paired screen over `ronFactor × oppGrowth ×
// dealinRate` moved nothing, because a paired screen grades PLACEMENT — four
// noisy layers away from the probability the scalar actually distorts. So this
// does not screen. It reads the probability off the engine (`R_PWIN`,
// `R_PTENPAI`), reads the answer off the 局, and minimises the log loss between
// them.
//
// THE HONESTY RULE, as `hand_fit.ts` states it: nothing here re-implements the
// model. Every prediction comes out of `mjev_eval_rest` on the wire the seat
// packed, under a real `EvCore` built from the candidate vector — the SAME
// engine, byte for byte, that would play the hand.
//
// THE REPRODUCTION CHECK, and the one case where it is a NOTICE rather than a
// refusal. When the lane's `engineHash` names the `native/mjev.cc` on disk, every
// loaded record is re-evaluated under the header's own parameters and must
// return the stored `pT`/`pW`/`eV`/`eCost` EXACTLY; a single mismatch means the
// lane and the engine have forked in a way the hash could not see (a stale
// dylib against a fresh source, a changed default, a `dbls` the writer did not
// round) and the fit STOPS rather than calibrating against a ghost.
//
// When the hashes DIFFER — or the lane is a v1 with no hash — the check is
// skipped with a printed notice and the fit continues. That is not a weakening.
// A record carries two independent things: the WIRE and the LABELS, which are
// properties of the game and survive any amount of work on the DP, and the four
// stored predictions, which are one engine's answers and are SUPPOSED to change
// when that engine is corrected. Refusing the lane would throw away the half
// that is still true. What the notice buys is that the "before" column of every
// table below is understood as the CURRENT engine's answer at the header's
// parameters, not as the numbers the seat actually played under.
//
// ===========================================================================
// THE OBJECTIVE
// ===========================================================================
//
//   L = BCE(pW, won) + BCE(pT, tenpaiEnd)
//
// over the TRAIN half. Two terms, both probabilities the engine reports for the
// resting shape the seat actually chose, both labelled by the 局 that followed.
// Nothing else is in the objective and nothing is weighted: the two are the two
// halves of "how does this hand end", and a hand-picked weight between them
// would be one more unmeasured scalar of exactly the kind this exercise exists
// to remove.
//
// THE THIRD TERM THAT IS NOT ONE. `R_ECOST` — points expected to be paid to the
// table on the way — has no per-record Bernoulli label: the 局 pays what it
// pays, and the engine's number is an expectation over a continuation that did
// not happen. There is no probability in the meta for "this hand ends by an
// opponent" either. So `eCost` is REPORTED, never fitted: the holdout tables
// print mean `eCost` against mean realised 放銃失点 by 巡目 bucket, which is
// the audit an owner can act on, and no gradient is taken from it.
//
// WHAT IS FITTED: the six POPULATION scalars — the ones M15's design already
// calls population averages rather than counting facts.
//
//   ronFactor   per-opponent ron hazard relative to the tsumo hazard
//   oppHazard   per own-turn P(a tenpai opponent ends the hand)
//   oppGrowth   growth of Σ tenpaiP per future turn
//   dealinRate  deal-in rate of a FUTURE (unknown) discard
//   tsumoShare  share of an opponent's win we pay when it is not off us
//   foldHazard  residual per-turn deal-in hazard of a folded hand
//
// The last three do not enter `pW`/`pT` directly; they move the DP's own
// push/fold and riichi choices inside the search, and `R_PWIN`/`R_PTENPAI` are
// reported UNDER that policy (see `evlayout.ts`). So they are in the vector and
// their gradient is allowed to be flat — a scalar that turns out not to matter
// for these two labels is a finding, not an omission.
//
// WHAT IS NOT FITTED: `meanUkeire` and the eight `value*` scalars. The first is
// the tail's shape model (M15 recalibrated `TAIL_CAL` against the exact DP in
// C++, and re-deriving it from four numbers here would fight that measurement);
// the second predicts 打点, which `handvalue.ts` already fits from its own lane
// and which is REPORTED here (eV vs realised 和了打点 by 向聴) rather than
// moved. Both are copied through untouched.
//
// THE PARAMETRISATION is a bounded logit, not a log: three of the six have a
// lower bound of exactly 0 (`oppGrowth`, `dealinRate`, `foldHazard`), where a
// log cannot go, and all six have an upper bound past which the quantity stops
// meaning anything. `x = log((v-lo)/(hi-v))` maps the open interval onto the
// whole line, so the optimiser runs unconstrained and no step can ever propose
// a negative hazard or a `tsumoShare` above 1.
//
// THE OPTIMISER is Nelder-Mead — deterministic, derivative-free, and 6-dimensional.
// Derivative-free because the loss is not differentiable in any usable sense:
// the DP's fold and riichi choices are argmaxes, so the surface is piecewise
// smooth with steps at every decision boundary, and a finite-difference
// gradient would mostly measure where those steps are.
//
// THE SPLIT is by SEED PARITY (odd = holdout), `hand_fit.ts`'s rule and for its
// reason: a 半荘 is the unit of correlation (every decision of a 局 shares a
// label, every 局 of a game shares a wall), so splitting by record would leak.

import type { EvCalibHeader, EvRecord } from "../src/ai/evcalib.ts";
import { evaluateWire, evEngineHash, scanEvCalibration } from "../src/ai/evcalib.ts";
import type { EvCore } from "../src/ai/ev.ts";
import { buildEv, closeEv } from "../src/ai/ev.ts";
import { DBLS_LEN, INTS_LEN } from "../src/ai/evlayout.ts";
import type { EvParams } from "../src/ai/evparams.ts";
import { DEFAULT_EV, mergeEv } from "../src/ai/evparams.ts";

function die(msg: string): never {
  console.error(msg);
  Deno.exit(2);
}

// ---------------------------------------------------------------------------
// the data
// ---------------------------------------------------------------------------

/**
 * One record, reduced to what the fit and the report read. The wire is kept as
 * the two typed arrays it will be copied from a few hundred times; everything
 * else is a scalar.
 */
export interface Row {
  ints: Int32Array;
  dbls: Float64Array;
  /** The stored prediction — the reproduction check's right-hand side. */
  pT0: number;
  pW0: number;
  eV0: number;
  eCost0: number;
  sh: number;
  T: number;
  junme: number;
  won: 0 | 1;
  tenpaiEnd: 0 | 1;
  winPoints: number;
  dealtIn: 0 | 1;
  dealtInPoints: number;
  oppWon: 0 | 1;
}

function rowOf(r: EvRecord): Row {
  const ints = new Int32Array(INTS_LEN);
  for (let i = 0; i < INTS_LEN; i++) ints[i] = r.ints[i];
  const dbls = new Float64Array(DBLS_LEN);
  for (let i = 0; i < DBLS_LEN; i++) dbls[i] = r.dbls[i];
  return {
    ints,
    dbls,
    pT0: r.pT,
    pW0: r.pW,
    eV0: r.eV,
    eCost0: r.eCost,
    sh: r.sh,
    T: r.T,
    junme: r.junme,
    won: r.won,
    tenpaiEnd: r.tenpaiEnd,
    winPoints: r.winPoints,
    dealtIn: r.dealtIn,
    dealtInPoints: r.dealtInPoints,
    oppWon: r.oppWon,
  };
}

/** Held out by SEED PARITY — see the header. Deterministic and stateless. */
export function isHeld(seed: number): boolean {
  return (seed & 1) === 1;
}

/** mulberry32 — the only source of randomness here, and it is seeded. */
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

/**
 * Reservoir sampling, one reservoir per half.
 *
 * A lane is hundreds of megabytes and 528 numbers a line; holding it all would
 * cost a gigabyte to answer a question 4,000 rows answer. A reservoir keeps the
 * sample UNIFORM over the whole file (a `head -n` would sample the first few
 * hundred 半荘 and nothing else) and, seeded, keeps it reproducible.
 */
class Reservoir {
  readonly rows: Row[] = [];
  seen = 0;
  constructor(readonly cap: number, private r: () => number) {}
  offer(rec: EvRecord): void {
    this.seen++;
    if (this.rows.length < this.cap) {
      this.rows.push(rowOf(rec));
      return;
    }
    const j = Math.floor(this.r() * this.seen);
    if (j < this.cap) this.rows[j] = rowOf(rec);
  }
}

export interface Lane {
  header: EvCalibHeader;
  params: EvParams;
  train: Row[];
  hold: Row[];
  seenTrain: number;
  seenHold: number;
  games: number;
}

export async function loadLane(
  path: string,
  sample: number,
  holdout: number,
  seed: number,
): Promise<Lane> {
  const r = rng(seed);
  const train = new Reservoir(sample, r);
  const hold = new Reservoir(holdout, r);
  const seeds = new Set<number>();
  const header = await scanEvCalibration(path, (rec) => {
    seeds.add(rec.s);
    (isHeld(rec.s) ? hold : train).offer(rec);
  });
  if (train.seen + hold.seen === 0) die(`${path}: 記録が1行もありません`);
  if (train.rows.length === 0) die(`${path}: 学習側 (偶数シード) の記録がありません`);
  if (hold.rows.length === 0) die(`${path}: 評価側 (奇数シード) の記録がありません`);
  return {
    header,
    params: mergeEv(header.ev),
    train: train.rows,
    hold: hold.rows,
    seenTrain: train.seen,
    seenHold: hold.seen,
    games: seeds.size,
  };
}

// ---------------------------------------------------------------------------
// evaluation
// ---------------------------------------------------------------------------

/** One core's answers for a batch of rows: the two probabilities and the two point figures. */
export interface Preds {
  pT: Float64Array;
  pW: Float64Array;
  eV: Float64Array;
  eCost: Float64Array;
}

export function predict(core: EvCore, rows: readonly Row[]): Preds {
  const n = rows.length;
  const out: Preds = {
    pT: new Float64Array(n),
    pW: new Float64Array(n),
    eV: new Float64Array(n),
    eCost: new Float64Array(n),
  };
  for (let i = 0; i < n; i++) {
    const p = evaluateWire(core, rows[i].ints, rows[i].dbls);
    out.pT[i] = p.pT;
    out.pW[i] = p.pW;
    out.eV[i] = p.eV;
    out.eCost[i] = p.eCost;
  }
  return out;
}

/** Binary cross-entropy in nats, clamped so a confident miss is finite. */
export function bce(p: number, y: number): number {
  const q = Math.min(1 - 1e-9, Math.max(1e-9, p));
  return y === 1 ? -Math.log(q) : -Math.log(1 - q);
}

export function lossOf(preds: Preds, rows: readonly Row[]): number {
  let s = 0;
  for (let i = 0; i < rows.length; i++) {
    s += bce(preds.pW[i], rows[i].won) + bce(preds.pT[i], rows[i].tenpaiEnd);
  }
  return s / Math.max(1, rows.length);
}

// ---------------------------------------------------------------------------
// the parameter vector
// ---------------------------------------------------------------------------

export interface Slot {
  key: "ronFactor" | "oppHazard" | "oppGrowth" | "dealinRate" | "tsumoShare" | "foldHazard";
  lo: number;
  hi: number;
  label: string;
}

/**
 * The six, with the bounds M15b was briefed with. `lo`/`hi` are OPEN — the
 * logit sends them to ∓∞ — so a fitted value that wants to sit on a bound
 * arrives near it and says so, which is more informative than a clamp.
 */
export const SLOTS: Slot[] = [
  { key: "ronFactor", lo: 0.05, hi: 2, label: "ronFactor (ロン/ツモ 比)" },
  { key: "oppHazard", lo: 0.01, hi: 0.5, label: "oppHazard (他家終局/巡)" },
  { key: "oppGrowth", lo: 0, hi: 0.15, label: "oppGrowth (聴牌成長/巡)" },
  { key: "dealinRate", lo: 0, hi: 0.2, label: "dealinRate (未知打牌の放銃率)" },
  { key: "tsumoShare", lo: 0, hi: 1, label: "tsumoShare (他家和了の負担率)" },
  { key: "foldHazard", lo: 0, hi: 0.1, label: "foldHazard (降り手の残存放銃率)" },
];

const EPS = 1e-6;

/** value → unconstrained. `lo`/`hi` are open, so a value ON a bound is nudged inside. */
export function toX(v: number, s: Slot): number {
  const span = s.hi - s.lo;
  const u = Math.min(1 - EPS, Math.max(EPS, (v - s.lo) / span));
  return Math.log(u / (1 - u));
}

/** unconstrained → value. Total: every real maps to a legal parameter. */
export function fromX(x: number, s: Slot): number {
  const u = 1 / (1 + Math.exp(-x));
  return s.lo + (s.hi - s.lo) * u;
}

export function vecOf(p: EvParams): Float64Array {
  return Float64Array.from(SLOTS.map((s) => toX(p[s.key], s)));
}

export function paramsOf(x: ArrayLike<number>, base: EvParams): EvParams {
  const over: Partial<EvParams> = {};
  SLOTS.forEach((s, i) => {
    (over as Record<string, number>)[s.key] = fromX(x[i], s);
  });
  return mergeEv({ ...base, ...over });
}

// ---------------------------------------------------------------------------
// Nelder-Mead
// ---------------------------------------------------------------------------

export interface NmResult {
  x: Float64Array;
  f: number;
  evals: number;
  trace: Array<{ eval: number; f: number }>;
}

/**
 * Nelder-Mead with the textbook coefficients (α 1, γ 2, ρ 1/2, σ 1/2), a fixed
 * initial simplex (`x0` plus `step` along each axis, in that order) and a hard
 * evaluation budget. No randomness anywhere, so two runs on one lane return the
 * same vector to the last bit.
 */
export function nelderMead(
  f: (x: Float64Array) => number,
  x0: Float64Array,
  step: number,
  maxEvals: number,
  onEval?: (n: number, f: number, x: Float64Array) => void,
): NmResult {
  const n = x0.length;
  let evals = 0;
  const trace: Array<{ eval: number; f: number }> = [];
  let best = Infinity;
  const ev = (x: Float64Array): number => {
    const v = f(x);
    evals++;
    if (v < best) {
      best = v;
      trace.push({ eval: evals, f: v });
    }
    onEval?.(evals, v, x);
    return v;
  };

  const pts: Array<{ x: Float64Array; f: number }> = [];
  pts.push({ x: Float64Array.from(x0), f: ev(x0) });
  for (let i = 0; i < n; i++) {
    const x = Float64Array.from(x0);
    x[i] += step;
    pts.push({ x, f: ev(x) });
  }
  const sort = () => pts.sort((a, b) => a.f - b.f || cmp(a.x, b.x));

  while (evals + 2 <= maxEvals) {
    sort();
    const centroid = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) centroid[j] += pts[i].x[j] / n;
    }
    const worst = pts[n];
    const refl = new Float64Array(n);
    for (let j = 0; j < n; j++) refl[j] = centroid[j] + (centroid[j] - worst.x[j]);
    const fr = ev(refl);
    if (fr < pts[0].f) {
      const exp = new Float64Array(n);
      for (let j = 0; j < n; j++) exp[j] = centroid[j] + 2 * (centroid[j] - worst.x[j]);
      const fe = ev(exp);
      pts[n] = fe < fr ? { x: exp, f: fe } : { x: refl, f: fr };
      continue;
    }
    if (fr < pts[n - 1].f) {
      pts[n] = { x: refl, f: fr };
      continue;
    }
    const con = new Float64Array(n);
    const useRefl = fr < worst.f;
    const src = useRefl ? refl : worst.x;
    for (let j = 0; j < n; j++) con[j] = centroid[j] + 0.5 * (src[j] - centroid[j]);
    const fc = ev(con);
    if (fc < Math.min(fr, worst.f)) {
      pts[n] = { x: con, f: fc };
      continue;
    }
    // Shrink toward the best point. Costs n evaluations, so only when nothing
    // else worked — which is the standard rule and the reason the budget is
    // stated in EVALUATIONS rather than in iterations.
    if (evals + n > maxEvals) break;
    const b = pts[0].x;
    for (let i = 1; i <= n; i++) {
      const x = new Float64Array(n);
      for (let j = 0; j < n; j++) x[j] = b[j] + 0.5 * (pts[i].x[j] - b[j]);
      pts[i] = { x, f: ev(x) };
    }
  }
  sort();
  return { x: pts[0].x, f: pts[0].f, evals, trace };
}

function cmp(a: Float64Array, b: Float64Array): number {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i];
  return 0;
}

// ---------------------------------------------------------------------------
// metrics and tables
// ---------------------------------------------------------------------------

export interface Metric {
  n: number;
  bce: number;
  brier: number;
  mean: number;
  rate: number;
  baseBce: number;
  baseBrier: number;
}

export function metricOf(p: Float64Array, ys: readonly (0 | 1)[]): Metric {
  let b = 0, br = 0, sp = 0, sy = 0;
  for (let i = 0; i < ys.length; i++) {
    b += bce(p[i], ys[i]);
    br += (p[i] - ys[i]) * (p[i] - ys[i]);
    sp += p[i];
    sy += ys[i];
  }
  const n = Math.max(1, ys.length);
  const rate = sy / n;
  return {
    n: ys.length,
    bce: b / n,
    brier: br / n,
    mean: sp / n,
    rate,
    baseBce: rate * bce(rate, 1) + (1 - rate) * bce(rate, 0),
    baseBrier: rate * (1 - rate),
  };
}

const pct = (x: number) => `${(100 * x).toFixed(1)}%`;

export function metricLine(name: string, m: Metric): string {
  return `${name.padEnd(16)}${String(m.n).padStart(8)}${m.bce.toFixed(4).padStart(10)}` +
    `${m.brier.toFixed(4).padStart(10)}${pct(m.mean).padStart(10)}${pct(m.rate).padStart(10)}` +
    `${m.baseBce.toFixed(4).padStart(10)}`;
}

export const METRIC_HEAD = `${"成分".padEnd(16)}${"件数".padStart(8)}${"対数損失".padStart(10)}` +
  `${"Brier".padStart(10)}${"予測平均".padStart(10)}${"実測".padStart(10)}${"無情報".padStart(10)}`;

/** Ten equal-width probability bands: predicted mean vs realised rate. */
export function reliability(
  p: Float64Array,
  ys: readonly (0 | 1)[],
  title: string,
): string {
  const n = 10;
  const cnt = new Array(n).fill(0);
  const sp = new Array(n).fill(0);
  const sy = new Array(n).fill(0);
  for (let i = 0; i < ys.length; i++) {
    const b = Math.min(n - 1, Math.max(0, Math.floor(p[i] * n)));
    cnt[b]++;
    sp[b] += p[i];
    sy[b] += ys[i];
  }
  const L = [
    title,
    `${"帯".padEnd(12)}${"件数".padStart(8)}${"予測".padStart(9)}${"実測".padStart(9)}${
      "差".padStart(9)
    }`,
  ];
  for (let b = 0; b < n; b++) {
    if (cnt[b] === 0) continue;
    const pm = sp[b] / cnt[b], ym = sy[b] / cnt[b];
    L.push(
      `${`${(b / n).toFixed(1)}-${((b + 1) / n).toFixed(1)}`.padEnd(12)}` +
        `${String(cnt[b]).padStart(8)}${pct(pm).padStart(9)}${pct(ym).padStart(9)}` +
        `${((pm - ym) >= 0 ? "+" : "") + pct(pm - ym)}`.padStart(10),
    );
  }
  return L.join("\n");
}

const SH_LABEL = ["聴牌", "1向聴", "2向聴", "3向聴+"];
const T_LABEL = ["T≤5", "T6-10", "T11+"];
const shBucket = (sh: number) => Math.min(3, Math.max(0, sh));
const tBucket = (T: number) => (T <= 5 ? 0 : T <= 10 ? 1 : 2);

/**
 * THE AUDIT the M15 entry asks for: does the exact path's P(win) from tenpai
 * match the champion's realised rate? One cell per (向聴 × 残り自摸), predicted
 * mean against realised rate, for both labels at once.
 */
export function bucketTable(
  rows: readonly Row[],
  pW: Float64Array,
  pT: Float64Array,
  title: string,
): string {
  const K = 4 * 3;
  const cnt = new Array(K).fill(0);
  const sw = new Array(K).fill(0), yw = new Array(K).fill(0);
  const st = new Array(K).fill(0), yt = new Array(K).fill(0);
  for (let i = 0; i < rows.length; i++) {
    const k = shBucket(rows[i].sh) * 3 + tBucket(rows[i].T);
    cnt[k]++;
    sw[k] += pW[i];
    yw[k] += rows[i].won;
    st[k] += pT[i];
    yt[k] += rows[i].tenpaiEnd;
  }
  const L = [
    title,
    `${"向聴".padEnd(8)}${"残自摸".padEnd(8)}${"件数".padStart(7)}` +
    `${"P(和了)予".padStart(11)}${"実測".padStart(8)}${"P(聴牌)予".padStart(11)}${
      "実測".padStart(8)
    }`,
  ];
  for (let s = 0; s < 4; s++) {
    for (let t = 0; t < 3; t++) {
      const k = s * 3 + t;
      if (cnt[k] === 0) continue;
      L.push(
        `${SH_LABEL[s].padEnd(8)}${T_LABEL[t].padEnd(8)}${String(cnt[k]).padStart(7)}` +
          `${pct(sw[k] / cnt[k]).padStart(11)}${pct(yw[k] / cnt[k]).padStart(8)}` +
          `${pct(st[k] / cnt[k]).padStart(11)}${pct(yt[k] / cnt[k]).padStart(8)}`,
      );
    }
  }
  return L.join("\n");
}

/**
 * E[打点|和了] against what the seat actually collected, by 向聴. `eV` is the
 * engine's "points collected when this hand is the one that wins", so the
 * honest comparison is against `winPoints` on the records that DID win —
 * a mean ratio near 1 is a realistic value model, and it is REPORTED rather
 * than fitted (the value scalars are `handvalue.ts`'s lane, not this one).
 */
export function valueTable(rows: readonly Row[], eV: Float64Array, title: string): string {
  const cnt = new Array(4).fill(0);
  const sp = new Array(4).fill(0), sy = new Array(4).fill(0);
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].won !== 1) continue;
    const k = shBucket(rows[i].sh);
    cnt[k]++;
    sp[k] += eV[i];
    sy[k] += rows[i].winPoints;
  }
  const L = [
    title,
    `${"向聴".padEnd(8)}${"和了件数".padStart(9)}${"E[打点]".padStart(10)}${"実収支".padStart(10)}${
      "比".padStart(8)
    }`,
  ];
  for (let k = 0; k < 4; k++) {
    if (cnt[k] === 0) continue;
    const pm = sp[k] / cnt[k], ym = sy[k] / cnt[k];
    L.push(
      `${SH_LABEL[k].padEnd(8)}${String(cnt[k]).padStart(9)}${pm.toFixed(0).padStart(10)}` +
        `${ym.toFixed(0).padStart(10)}${(ym === 0 ? "-" : (pm / ym).toFixed(2)).padStart(8)}`,
    );
  }
  return L.join("\n");
}

/**
 * `eCost` against realised 放銃失点, by 巡目. Not an objective term (see the
 * header) — the engine's cost is an expectation over a continuation, and the
 * only ground truth is what the 局 actually took off us.
 */
export function costTable(rows: readonly Row[], eCost: Float64Array, title: string): string {
  const lab = ["巡≤5", "巡6-10", "巡11+"];
  const cnt = new Array(3).fill(0);
  const sc = new Array(3).fill(0), sy = new Array(3).fill(0), nd = new Array(3).fill(0);
  for (let i = 0; i < rows.length; i++) {
    const j = rows[i].junme <= 5 ? 0 : rows[i].junme <= 10 ? 1 : 2;
    cnt[j]++;
    sc[j] += eCost[i];
    sy[j] += rows[i].dealtInPoints;
    nd[j] += rows[i].dealtIn;
  }
  const L = [
    title,
    `${"巡目".padEnd(9)}${"件数".padStart(8)}${"E[放銃コスト]".padStart(14)}${
      "実測失点".padStart(10)
    }${"放銃率".padStart(9)}`,
  ];
  for (let j = 0; j < 3; j++) {
    if (cnt[j] === 0) continue;
    L.push(
      `${lab[j].padEnd(9)}${String(cnt[j]).padStart(8)}${
        (sc[j] / cnt[j]).toFixed(0).padStart(14)
      }` +
        `${(sy[j] / cnt[j]).toFixed(0).padStart(10)}${pct(nd[j] / cnt[j]).padStart(9)}`,
    );
  }
  return L.join("\n");
}

export function diffTable(from: EvParams, to: EvParams): string {
  const L = [
    "■ 初期値 → 当てはめ",
    `${"パラメータ".padEnd(30)}${"下限".padStart(8)}${"初期".padStart(10)}${
      "当てはめ".padStart(12)
    }${"上限".padStart(8)}${"倍率".padStart(9)}`,
  ];
  for (const s of SLOTS) {
    const a = from[s.key], b = to[s.key];
    L.push(
      `${s.label.padEnd(30)}${String(s.lo).padStart(8)}${a.toFixed(4).padStart(10)}` +
        `${b.toFixed(4).padStart(12)}${String(s.hi).padStart(8)}` +
        `${(a === 0 ? "-" : `${(b / a).toFixed(2)}×`).padStart(9)}`,
    );
  }
  L.push("不変: meanUkeire / value* / margins / 探索予算 (尾部形状と打点は別レーンの担当)");
  return L.join("\n");
}

/**
 * Does the stored prediction still belong to the engine on disk?
 *
 * Three answers, and only the first one is checkable. The hash covers
 * `native/mjev.cc` alone — the DP's source — because that is what decides the
 * numbers; the ABI has already been checked by the reader, and a parameter
 * change is caught by the check itself rather than by the hash.
 */
export function reproVerdict(
  laneHash: string | undefined,
  nowHash: string | undefined,
): { check: boolean; why: string } {
  if (!laneHash) {
    return { check: false, why: "レーンにエンジンの指紋がありません (v1 のレーン)" };
  }
  if (!nowHash) return { check: false, why: "native/mjev.cc が読めません" };
  if (laneHash !== nowHash) {
    return {
      check: false,
      why: `エンジンが変わっています (レーン ${laneHash.slice(0, 12)}… / ` +
        `現在 ${nowHash.slice(0, 12)}…)`,
    };
  }
  return { check: true, why: "" };
}

// ---------------------------------------------------------------------------
// entry
// ---------------------------------------------------------------------------

const HELP = "使い方: ev_fit.ts --in=PATH --out=PATH [--sample=N] [--holdout=N]\n" +
  "        [--seed=N] [--iters=N] [--step=X] [--base=header|default]";

function parse(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) die(`余分な引数: ${arg}\n${HELP}`);
    const eq = arg.indexOf("=");
    if (eq >= 0) out[arg.slice(2, eq)] = arg.slice(eq + 1);
    else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) out[arg.slice(2)] = argv[++i];
    else out[arg.slice(2)] = "1";
  }
  return out;
}

async function main(argv: string[]): Promise<void> {
  const f = parse(argv);
  const known = new Set(["in", "out", "sample", "holdout", "seed", "iters", "step", "base", "help"]);
  for (const k of Object.keys(f)) if (!known.has(k)) die(`不明なオプション: --${k}\n${HELP}`);
  if (f.help !== undefined) die(HELP);
  const inPath = f.in;
  if (!inPath) die(`--in が要ります\n${HELP}`);
  const outPath = f.out ?? "weights/ev-fit.json";
  const num = (k: string, d: number) => {
    if (f[k] === undefined) return d;
    const v = Number(f[k]);
    if (!Number.isFinite(v)) die(`--${k} は数値: ${f[k]}`);
    return v;
  };
  const sample = num("sample", 4000);
  const holdout = num("holdout", sample);
  const seed = num("seed", 12345);
  const iters = num("iters", 260);
  const step = num("step", 0.8);

  console.log("=== EV核 母集団スカラーの当てはめ (M15b) ===");
  const t0 = performance.now();
  const lane = await loadLane(inPath, sample, holdout, seed);
  console.log(
    `読み込み ${inPath}  ${((performance.now() - t0) / 1000).toFixed(1)}秒  ` +
      `半荘 ${lane.games}  自摸番 ${lane.seenTrain + lane.seenHold}行 ` +
      `(学習 ${lane.seenTrain} → 標本 ${lane.train.length} / ` +
      `評価 ${lane.seenHold} → 標本 ${lane.hold.length})`,
  );
  // `--base=header` (default) fits on top of the lane's own recorded params;
  // `--base=default` fits on top of the CURRENT `DEFAULT_EV` — what a seat
  // built today would actually run (exactShanten / budget / rung switches
  // included). The two differ whenever the engine defaults moved after the
  // lane was recorded; the fitted scalars only mean something for the base
  // they were fitted on, so the base is written into the meta file.
  const baseArg = f.base ?? "header";
  if (baseArg !== "header" && baseArg !== "default") {
    throw new Error(`--base は header か default: ${baseArg}`);
  }
  const base = baseArg === "default" ? DEFAULT_EV : lane.params;
  console.log(
    `母数の土台 ${baseArg} ${
      JSON.stringify(base) === JSON.stringify(DEFAULT_EV) ? "= DEFAULT_EV" : "(既定と異なる)"
    }`,
  );

  // ---- reproduction, on every loaded row, before anything else --------------
  const core0 = buildEv(base);
  const t1 = performance.now();
  const trainBase = predict(core0, lane.train);
  const holdBase = predict(core0, lane.hold);
  const per = (performance.now() - t1) / (lane.train.length + lane.hold.length);
  const total = lane.train.length + lane.hold.length;
  const now = evEngineHash();
  const verdict = reproVerdict(lane.header.engineHash, now);
  if (verdict.check) {
    let bad = 0;
    let worst = "";
    const check = (rows: readonly Row[], p: Preds) => {
      for (let i = 0; i < rows.length; i++) {
        const r = rows[i];
        if (
          p.pT[i] === r.pT0 && p.pW[i] === r.pW0 && p.eV[i] === r.eV0 && p.eCost[i] === r.eCost0
        ) {
          continue;
        }
        if (bad === 0) {
          worst = `pT ${p.pT[i]} vs ${r.pT0} / pW ${p.pW[i]} vs ${r.pW0} / ` +
            `eV ${p.eV[i]} vs ${r.eV0} / eCost ${p.eCost[i]} vs ${r.eCost0}`;
        }
        bad++;
      }
    };
    check(lane.train, trainBase);
    check(lane.hold, holdBase);
    if (bad > 0) {
      closeEv(core0);
      die(
        `再現に失敗しました: ${bad}/${total} 行が一致しません。\n` +
          `  最初の不一致: ${worst}\n` +
          "エンジンのソースは一致しているのに答えが違います — dylib が古いか " +
          "(deno task build-ev)、既定値が動いています。",
      );
    }
    console.log(`再現 ${total}/${total} 行 ビット単位で一致  (1評価 ${per.toFixed(3)}ms)`);
  } else {
    console.log(`再現検査は飛ばします — ${verdict.why}  (1評価 ${per.toFixed(3)}ms)`);
    console.log(
      "  記録された pT/pW/eV/eCost は当時のエンジンの答えです。wire と札 (勝敗・聴牌・" +
        "収支) はゲームの事実なのでそのまま使えますが、以下の「初期値」列は " +
        "『今のエンジンがヘッダの母数で出す答え』であって、席が実際に打った数字では" +
        "ありません。",
    );
  }
  closeEv(core0);

  // ---- the fit -------------------------------------------------------------
  const x0 = vecOf(base);
  const budgetSec = (iters * lane.train.length * per) / 1000;
  console.log(
    `最適化: Nelder-Mead 6次元  評価上限 ${iters}  初期辺 ${step}  ` +
      `見込み ${(budgetSec / 60).toFixed(1)}分`,
  );
  console.log();
  const t2 = performance.now();
  let shown = 0;
  const objective = (x: Float64Array): number => {
    const p = paramsOf(x, base);
    const core = buildEv(p);
    try {
      return lossOf(predict(core, lane.train), lane.train);
    } finally {
      closeEv(core);
    }
  };
  const res = nelderMead(objective, x0, step, iters, (n, v) => {
    if (n - shown >= 20 || n === 1) {
      shown = n;
      console.log(`  評価 ${String(n).padStart(4)}  損失 ${v.toFixed(6)}`);
    }
  });
  const fitted = paramsOf(res.x, base);
  console.log(
    `当てはめ ${((performance.now() - t2) / 1000).toFixed(1)}秒  評価 ${res.evals}回  ` +
      `学習損失 ${res.f.toFixed(6)} (初期 ${lossOf(trainBase, lane.train).toFixed(6)})`,
  );
  console.log();
  console.log(diffTable(base, fitted));
  console.log();

  // ---- the holdout report --------------------------------------------------
  const coreF = buildEv(fitted);
  const holdFit = predict(coreF, lane.hold);
  closeEv(coreF);

  const wonY = lane.hold.map((r) => r.won);
  const tenY = lane.hold.map((r) => r.tenpaiEnd);
  console.log("■ 評価 (奇数シード) — 初期値 vs 当てはめ");
  console.log(METRIC_HEAD);
  console.log(metricLine("P(和了) 初期", metricOf(holdBase.pW, wonY)));
  console.log(metricLine("P(和了) 当て", metricOf(holdFit.pW, wonY)));
  console.log(metricLine("P(聴牌) 初期", metricOf(holdBase.pT, tenY)));
  console.log(metricLine("P(聴牌) 当て", metricOf(holdFit.pT, tenY)));
  console.log(
    `評価損失 初期 ${lossOf(holdBase, lane.hold).toFixed(6)} → ` +
      `当てはめ ${lossOf(holdFit, lane.hold).toFixed(6)}`,
  );
  console.log();
  console.log(reliability(holdBase.pW, wonY, "■ 信頼度 P(和了) — 初期値"));
  console.log();
  console.log(reliability(holdFit.pW, wonY, "■ 信頼度 P(和了) — 当てはめ"));
  console.log();
  console.log(reliability(holdBase.pT, tenY, "■ 信頼度 P(聴牌) — 初期値"));
  console.log();
  console.log(reliability(holdFit.pT, tenY, "■ 信頼度 P(聴牌) — 当てはめ"));
  console.log();
  console.log(bucketTable(lane.hold, holdBase.pW, holdBase.pT, "■ 向聴 × 残り自摸 — 初期値"));
  console.log();
  console.log(bucketTable(lane.hold, holdFit.pW, holdFit.pT, "■ 向聴 × 残り自摸 — 当てはめ"));
  console.log();
  console.log(valueTable(lane.hold, holdBase.eV, "■ E[打点|和了] vs 実収支 — 初期値"));
  console.log();
  console.log(valueTable(lane.hold, holdFit.eV, "■ E[打点|和了] vs 実収支 — 当てはめ"));
  console.log();
  console.log(costTable(lane.hold, holdBase.eCost, "■ E[放銃コスト] vs 実測 — 初期値"));
  console.log();
  console.log(costTable(lane.hold, holdFit.eCost, "■ E[放銃コスト] vs 実測 — 当てはめ"));
  console.log();

  // ---- output --------------------------------------------------------------
  const block: Record<string, number> = {};
  for (const s of SLOTS) block[s.key] = fitted[s.key];
  const dir = outPath.slice(0, outPath.lastIndexOf("/"));
  if (dir) Deno.mkdirSync(dir, { recursive: true });
  Deno.writeTextFileSync(outPath, JSON.stringify({ ev: block }, null, 2) + "\n");
  const meta = {
    lane: inPath,
    engineHash: { lane: lane.header.engineHash ?? null, now: evEngineHash() ?? null },
    reproduced: verdict.check,
    games: lane.games,
    rows: { train: lane.seenTrain, hold: lane.seenHold },
    sample: { train: lane.train.length, hold: lane.hold.length, seed },
    evals: res.evals,
    trace: res.trace,
    baseFrom: baseArg,
    base: Object.fromEntries(SLOTS.map((s) => [s.key, base[s.key]])),
    baseFull: base,
    fitted: block,
    holdout: {
      pW: { before: metricOf(holdBase.pW, wonY), after: metricOf(holdFit.pW, wonY) },
      pT: { before: metricOf(holdBase.pT, tenY), after: metricOf(holdFit.pT, tenY) },
      loss: { before: lossOf(holdBase, lane.hold), after: lossOf(holdFit, lane.hold) },
    },
  };
  Deno.writeTextFileSync(`${outPath}.meta.json`, JSON.stringify(meta, null, 2) + "\n");
  console.log(`書き出し ${outPath}  (+ ${outPath}.meta.json)`);
}

if (import.meta.main) await main(Deno.args);
