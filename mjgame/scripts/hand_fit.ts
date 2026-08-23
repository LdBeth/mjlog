#!/usr/bin/env -S deno run --allow-read --allow-write
// 手牌価値の当てはめ (M11) — 自分の手について信じたことを、実際の結末に合わせる。
//
//   deno run --allow-read --allow-write scripts/hand_fit.ts \
//     --in runs/hand/lane.jsonl --out weights/hand-calibrated.json \
//     [--init weights/ktune.json] [--steps 300]
//
// WHAT THIS IS. `scripts/hand_report.ts` says WHERE `handOutlook` is wrong;
// this says WHAT TO SET. The record format was designed for exactly this (see
// the header of `src/ai/handcalib.ts`): every line carries the parameter-FREE
// `HandFacts` a prediction was built from, so the model is a closed form in
// `HandWeights` over cached features and a fit is an optimisation rather than a
// tournament of selfplay runs.
//
// THE HONESTY RULE. Nothing here re-implements the model. Every prediction goes
// through `handOutlook` — the same function the seat calls, on the same facts —
// so a fitted vector cannot mean one thing to the fit and another at the table.
// That costs a little speed (the chain recomputes `value` when only `pwin` is
// wanted, and vice versa) and buys the only guarantee that matters.
//
// ===========================================================================
// THE OBJECTIVE
// ===========================================================================
//
// The model factors into two independent halves, and so does the loss. They
// share no parameter, so each is optimised against its own block alone:
//
//   (p) 和了率: BCE of `handOutlook(facts,w).pwin` against `won`, over EVERY
//       recorded turn decision. The label is the round's result, so all of one
//       round's decisions carry the same 0/1 and the fit learns how the chain's
//       forecast should decay with 巡目 and shanten.
//       Parameters: meanUkeire[0..3], ronFactor, oppHazard, oppGrowth.
//
//   (v) 打点: Huber (δ=2000) of `.value` against `winPoints`, over WON records
//       only. On a lost round the seat collected nothing, and a zero there is
//       not evidence that the hand was worth zero — it is evidence about (p),
//       which is already scored.
//       Parameters: the eight value scalars.
//
// Huber rather than squares because a 役満 is a real observation and a rare one:
// least squares would let a handful of them drag `valueRiichi` upward for every
// ordinary hand. δ=2000 is about half a 満貫 — inside it the loss is quadratic
// and an ordinary 3900/8000 miss is priced properly; outside it linear.
//
// WHAT IS NOT FITTED: `pushScale` and `evWeight`. They are how the policy SPENDS
// the outlook, not what the outlook predicts, and no recorded ground truth
// exists for "how hard should this seat push" — that is what `paired` is for.
// They are copied through untouched so the written file is a complete vector.
//
// EVERY PARAMETER IS LOG-PARAMETRIZED. All fifteen are positive quantities
// (counts of tiles, hazard rates, points), Adam then runs unconstrained, and a
// step can never propose a negative ukeire or a negative 打点.
//
// THE SPLIT is by SEED PARITY: odd seeds are held out and never enter a
// gradient. A hanchan is the unit of correlation here (every decision in a round
// shares a label, and every round in a game shares a wall), so splitting by
// record would leak the answer across the boundary.

import type { HandRecord } from "../src/ai/handcalib.ts";
import { scanHandCalibration } from "../src/ai/handcalib.ts";
import type { HandWeights } from "../src/ai/handvalue.ts";
import { DEFAULT_HAND, handOutlook, mergeHand } from "../src/ai/handvalue.ts";

function die(msg: string): never {
  console.error(msg);
  Deno.exit(2);
}

// ---------------------------------------------------------------------------
// the data
// ---------------------------------------------------------------------------

/** One lane, split into the two halves the two components consume. */
export interface Lane {
  records: HandRecord[];
  /** Indices of training (even-seed) records; every one feeds component (p). */
  trainP: Int32Array;
  heldP: Int32Array;
  /** …and of the WON ones, which are the only evidence component (v) has. */
  trainV: Int32Array;
  heldV: Int32Array;
  header: HandWeights;
  games: number;
  heldGames: number;
}

/** Held out by SEED PARITY — see the header. Deterministic and stateless. */
export function isHeld(seed: number): boolean {
  return (seed & 1) === 1;
}

function indicesOf(recs: readonly HandRecord[], held: boolean, wonOnly: boolean): Int32Array {
  const out: number[] = [];
  for (let i = 0; i < recs.length; i++) {
    if (isHeld(recs[i].s) !== held) continue;
    if (wonOnly && recs[i].won !== 1) continue;
    out.push(i);
  }
  return Int32Array.from(out);
}

export async function loadLane(path: string, max?: number): Promise<Lane> {
  const records: HandRecord[] = [];
  const seeds = new Set<number>();
  const heldSeeds = new Set<number>();
  const header = await scanHandCalibration(path, (rec) => {
    if (max !== undefined && records.length >= max) return;
    records.push(rec);
    seeds.add(rec.s);
    if (isHeld(rec.s)) heldSeeds.add(rec.s);
  });
  if (records.length === 0) die(`${path}: 記録が1行もありません`);
  return {
    records,
    trainP: indicesOf(records, false, false),
    heldP: indicesOf(records, true, false),
    trainV: indicesOf(records, false, true),
    heldV: indicesOf(records, true, true),
    header: mergeHand(header.w),
    games: seeds.size,
    heldGames: heldSeeds.size,
  };
}

// ---------------------------------------------------------------------------
// the parameter vector
// ---------------------------------------------------------------------------

type Comp = "p" | "v";

interface Slot {
  name: string;
  comp: Comp;
  get(w: HandWeights): number;
  set(w: Mutable, v: number): void;
}

/** A `HandWeights` whose tuple is a plain writable array while the fit runs. */
interface Mutable extends Omit<HandWeights, "meanUkeire"> {
  meanUkeire: [number, number, number, number];
}

const VALUE_KEYS = [
  "valueRiichi",
  "valueDamaten",
  "valueOpen",
  "valueHonitsu",
  "valuePerDora",
  "valueYakuhai",
  "valueDealer",
  "valueCap",
] as const;

const CHAIN_KEYS = ["ronFactor", "oppHazard", "oppGrowth"] as const;

const UKEIRE_LABELS = ["向聴≥3", "2向聴", "1向聴", "聴牌"];

/** Every parameter this fit moves, in report order. */
export const SLOTS: Slot[] = [
  ...UKEIRE_LABELS.map((lab, i): Slot => ({
    name: `meanUkeire[${lab}]`,
    comp: "p",
    get: (w) => w.meanUkeire[i],
    set: (w, v) => {
      w.meanUkeire[i] = v;
    },
  })),
  ...CHAIN_KEYS.map((k): Slot => ({
    name: k,
    comp: "p",
    get: (w) => w[k],
    set: (w, v) => {
      w[k] = v;
    },
  })),
  ...VALUE_KEYS.map((k): Slot => ({
    name: k,
    comp: "v",
    get: (w) => w[k],
    set: (w, v) => {
      w[k] = v;
    },
  })),
];

/** The vector as weights. `pushScale`/`evWeight` ride along untouched. */
export function weightsOf(theta: Float64Array, base: HandWeights): HandWeights {
  const w: Mutable = { ...base, meanUkeire: [...base.meanUkeire] };
  for (let i = 0; i < SLOTS.length; i++) SLOTS[i].set(w, Math.exp(theta[i]));
  return { ...w, meanUkeire: w.meanUkeire };
}

export function thetaOf(w: HandWeights): Float64Array {
  const t = new Float64Array(SLOTS.length);
  for (let i = 0; i < SLOTS.length; i++) t[i] = Math.log(Math.max(1e-12, SLOTS[i].get(w)));
  return t;
}

// ---------------------------------------------------------------------------
// the losses
// ---------------------------------------------------------------------------

function bce(p: number, y: number): number {
  const q = Math.min(1 - 1e-9, Math.max(1e-9, p));
  return -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
}

export function huber(r: number, d: number): number {
  const a = Math.abs(r);
  return a <= d ? 0.5 * r * r : d * (a - 0.5 * d);
}

export function lossP(
  recs: readonly HandRecord[],
  w: HandWeights,
  idx: Int32Array,
  from: number,
  to: number,
): number {
  let s = 0;
  for (let i = from; i < to; i++) {
    const r = recs[idx[i]];
    s += bce(handOutlook(r.facts, w).pwin, r.won);
  }
  return to > from ? s / (to - from) : 0;
}

export function lossV(
  recs: readonly HandRecord[],
  w: HandWeights,
  idx: Int32Array,
  from: number,
  to: number,
  delta: number,
): number {
  let s = 0;
  for (let i = from; i < to; i++) {
    const r = recs[idx[i]];
    s += huber(handOutlook(r.facts, w).value - r.winPoints, delta);
  }
  return to > from ? s / (to - from) : 0;
}

// ---------------------------------------------------------------------------
// metrics
// ---------------------------------------------------------------------------

export interface Metrics {
  pN: number;
  pBce: number;
  pBrier: number;
  pBaseBce: number;
  pBaseBrier: number;
  pMean: number;
  pRate: number;
  vN: number;
  vHuber: number;
  vMae: number;
  vBias: number;
  vBaseMae: number;
}

export function metricsOf(
  lane: Lane,
  w: HandWeights,
  held: boolean,
  delta: number,
): Metrics {
  const idxP = held ? lane.heldP : lane.trainP;
  const idxV = held ? lane.heldV : lane.trainV;
  let bceSum = 0, brier = 0, sumP = 0, sumY = 0;
  for (let i = 0; i < idxP.length; i++) {
    const r = lane.records[idxP[i]];
    const p = handOutlook(r.facts, w).pwin;
    bceSum += bce(p, r.won);
    brier += (p - r.won) * (p - r.won);
    sumP += p;
    sumY += r.won;
  }
  const n = Math.max(1, idxP.length);
  // The honest no-skill reference: the constant predictor at the empirical rate.
  // Both losses are closed forms there, because the labels are 0/1.
  const rate = sumY / n;
  let hub = 0, mae = 0, bias = 0, sumT = 0;
  for (let i = 0; i < idxV.length; i++) {
    const r = lane.records[idxV[i]];
    const v = handOutlook(r.facts, w).value;
    hub += huber(v - r.winPoints, delta);
    mae += Math.abs(v - r.winPoints);
    bias += v - r.winPoints;
    sumT += r.winPoints;
  }
  const m = Math.max(1, idxV.length);
  const meanT = sumT / m;
  let baseMae = 0;
  for (let i = 0; i < idxV.length; i++) {
    baseMae += Math.abs(meanT - lane.records[idxV[i]].winPoints);
  }
  return {
    pN: idxP.length,
    pBce: bceSum / n,
    pBrier: brier / n,
    pBaseBce: rate * bce(rate, 1) + (1 - rate) * bce(rate, 0),
    pBaseBrier: rate * (1 - rate),
    pMean: sumP / n,
    pRate: rate,
    vN: idxV.length,
    vHuber: hub / m,
    vMae: mae / m,
    vBias: bias / m,
    vBaseMae: baseMae / m,
  };
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

export interface FitOpts {
  steps: number;
  lr: number;
  batch: number;
  seed: number;
  delta: number;
  /** L2 pull back towards the starting vector, in log space. */
  l2: number;
  verbose: boolean;
}

export const DEFAULT_FIT: FitOpts = {
  steps: 300,
  lr: 0.05,
  batch: 2048,
  seed: 12345,
  delta: 2000,
  l2: 1e-4,
  verbose: true,
};

export interface FitResult {
  weights: HandWeights;
  /** The step at which each component's held-out loss bottomed out. */
  bestStep: Record<Comp, number>;
  history: { step: number; p: number; v: number }[];
}

/**
 * Adam over the log-space vector, with CENTRAL-DIFFERENCE gradients.
 *
 * Numeric rather than analytic, for the same reason `calibrate_fit.ts` is: the
 * model is fifteen parameters and one forward is a ≤20×4 sweep, so a gradient
 * costs two forwards per parameter — cheap — while an analytic gradient would
 * mean hand-differentiating the DP, the `min(1,·)` clips and the value cap, i.e.
 * writing the model a second time in derivative form. The +h and −h evaluations
 * use the SAME minibatch, so the estimate is the exact gradient of that
 * minibatch's loss and the sampling noise cancels.
 *
 * The two components own DISJOINT parameters, so each is perturbed against its
 * own loss alone and model selection factorises over them exactly.
 */
export function fit(lane: Lane, init: HandWeights, opts: Partial<FitOpts> = {}): FitResult {
  const o: FitOpts = { ...DEFAULT_FIT, ...opts };
  const P = SLOTS.length;
  const theta = thetaOf(init);
  const theta0 = Float64Array.from(theta);
  const best = Float64Array.from(theta);
  const m = new Float64Array(P);
  const v = new Float64Array(P);
  const g = new Float64Array(P);
  const trP = Int32Array.from(lane.trainP);
  const trV = Int32Array.from(lane.trainV);
  const r = rng(o.seed);
  const h = 1e-3;
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;

  const history: FitResult["history"] = [];
  const bestLoss: Record<Comp, number> = { p: Infinity, v: Infinity };
  const bestStep: Record<Comp, number> = { p: -1, v: -1 };
  // Held-out loss is checked on a coarse grid, not every step: it is a full pass
  // over the held half and would otherwise dominate the fit's runtime.
  const every = Math.max(1, Math.floor(o.steps / 20));

  let cursorP = trP.length, cursorV = trV.length;
  const window = (idx: Int32Array, cursor: number, batch: number): [number, number, number] => {
    if (cursor + batch > idx.length) {
      shuffle(idx, r);
      cursor = 0;
    }
    const to = Math.min(idx.length, cursor + batch);
    return [cursor, to, to];
  };

  for (let step = 1; step <= o.steps; step++) {
    const [p0, p1, np] = window(trP, cursorP, o.batch);
    cursorP = np;
    const [v0, v1, nv] = window(trV, cursorV, Math.min(o.batch, Math.max(1, trV.length)));
    cursorV = nv;
    // Linear decay to a tenth of the rate: Adam with a constant step leaves the
    // last steps rattling around the optimum instead of settling into it.
    const lr = o.lr * (1 - 0.9 * (step - 1) / Math.max(1, o.steps - 1));
    for (let i = 0; i < P; i++) {
      const x = theta[i];
      let lp = 0, lm = 0;
      theta[i] = x + h;
      const wp = weightsOf(theta, init);
      theta[i] = x - h;
      const wm = weightsOf(theta, init);
      theta[i] = x;
      if (SLOTS[i].comp === "p") {
        lp = lossP(lane.records, wp, trP, p0, p1);
        lm = lossP(lane.records, wm, trP, p0, p1);
      } else {
        lp = lossV(lane.records, wp, trV, v0, v1, o.delta) / o.delta;
        lm = lossV(lane.records, wm, trV, v0, v1, o.delta) / o.delta;
      }
      g[i] = (lp - lm) / (2 * h) + 2 * o.l2 * (x - theta0[i]);
    }
    const c1 = 1 - Math.pow(b1, step), c2 = 1 - Math.pow(b2, step);
    for (let i = 0; i < P; i++) {
      m[i] = b1 * m[i] + (1 - b1) * g[i];
      v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
      theta[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
    }
    if (step % every !== 0 && step !== o.steps) continue;
    const mv = metricsOf(lane, weightsOf(theta, init), true, o.delta);
    history.push({ step, p: mv.pBce, v: mv.vHuber });
    const per: Record<Comp, number> = { p: mv.pBce, v: mv.vHuber };
    const kept: Comp[] = [];
    for (const comp of ["p", "v"] as Comp[]) {
      if (per[comp] >= bestLoss[comp]) continue;
      bestLoss[comp] = per[comp];
      bestStep[comp] = step;
      kept.push(comp);
      for (let i = 0; i < P; i++) if (SLOTS[i].comp === comp) best[i] = theta[i];
    }
    if (o.verbose) {
      console.log(
        `  step ${String(step).padStart(4)}  検証 和了 ${mv.pBce.toFixed(5)}  ` +
          `打点 ${mv.vHuber.toFixed(0)}${kept.length > 0 ? "  採用 " + kept.join("") : ""}`,
      );
    }
  }
  return { weights: weightsOf(best, init), bestStep, history };
}

// ---------------------------------------------------------------------------
// the ktune file
// ---------------------------------------------------------------------------

/**
 * The fitted vector as a `--ktune` partial. The WHOLE `HandWeights` is written,
 * the two consumption scalars included at whatever they came in as: a `hand`
 * block is what switches M11 on at all (an absent one leaves every policy
 * bit-identical), so a half-written block would be a different model, not a
 * smaller diff.
 */
export function ktuneOf(w: HandWeights): { hand: HandWeights } {
  return { hand: { ...w, meanUkeire: [...w.meanUkeire] as [number, number, number, number] } };
}

/** Read a `--init` ktune file's `hand` block, merged onto the defaults. */
export function loadInit(path: string): HandWeights {
  let json: unknown;
  try {
    json = JSON.parse(Deno.readTextFileSync(path));
  } catch (e) {
    die(`--init が読めません: ${path}\n${e instanceof Error ? e.message : e}`);
  }
  const obj = json as { hand?: Partial<HandWeights> };
  return mergeHand(obj.hand ?? (json as Partial<HandWeights>));
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

const pct = (x: number) => `${(100 * x).toFixed(2)}%`;
const skill = (m: number, b: number) => (b <= 0 ? 0 : 1 - m / b);

function fmt(v: number): string {
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(2);
  return v.toFixed(4);
}

export function diffTable(from: HandWeights, to: HandWeights): string {
  const L: string[] = [];
  L.push("■ 初期値 → 当てはめ");
  L.push(
    `${"パラメータ".padEnd(24)}${"初期".padStart(11)}${"当てはめ".padStart(13)}${
      "倍率".padStart(9)
    }`,
  );
  let comp: Comp | null = null;
  for (const s of SLOTS) {
    if (s.comp !== comp) {
      comp = s.comp;
      L.push(comp === "p" ? "-- (p) 和了率の連鎖 --" : "-- (v) 打点 --");
    }
    const a = s.get(from), b = s.get(to);
    L.push(
      `${s.name.padEnd(24)}${fmt(a).padStart(11)}${fmt(b).padStart(13)}` +
        `${(a === 0 ? "-" : (b / a).toFixed(2) + "×").padStart(9)}`,
    );
  }
  L.push("不変: pushScale / evWeight (消費側 — 実測の真値が無い。paired で決める)");
  // Not a bug and worth saying out loud: rung 0 is only ever the CURRENT level,
  // and the current level's acceptance count is read off the hand (`f.ukeire`),
  // never from the table. So `meanUkeire[向聴≥3]` has an identically zero
  // gradient and stays where it started, by construction rather than by luck.
  L.push("恒等: meanUkeire[向聴≥3] は現在地の段でしか使われず、現在地は実測の受入枚数を");
  L.push("      読むため、この段の勾配は恒等的に0 (動かないのが正しい)");
  return L.join("\n");
}

export function metricBlock(title: string, m: Metrics): string {
  const L: string[] = [];
  L.push(title);
  L.push("成分                        件数     モデル      基準     改善率");
  const row = (name: string, n: number, mm: number, b: number, dp = 4) =>
    L.push(
      `${name.padEnd(24)}${String(n).padStart(10)}${mm.toFixed(dp).padStart(11)}` +
        `${b.toFixed(dp).padStart(10)}${pct(skill(mm, b)).padStart(11)}`,
    );
  row("(p) 和了 対数損失", m.pN, m.pBce, m.pBaseBce);
  row("(p) 和了 Brier", m.pN, m.pBrier, m.pBaseBrier);
  row("(v) 打点 平均絶対誤差", m.vN, m.vMae, m.vBaseMae, 0);
  L.push(
    `(p) 予測平均 ${pct(m.pMean)} / 実測 ${pct(m.pRate)}   ` +
      `(v) 偏り ${m.vBias >= 0 ? "+" : ""}${m.vBias.toFixed(0)}点  ` +
      `Huber ${m.vHuber.toFixed(0)}`,
  );
  return L.join("\n");
}

// ---------------------------------------------------------------------------
// entry
// ---------------------------------------------------------------------------

const HELP = "使い方: hand_fit.ts --in=PATH --out=PATH [--init=PATH] [--steps=N]\n" +
  "        [--lr=X] [--batch=N] [--seed=N] [--huber=X] [--l2=X] [--max=N] [--quiet]";

/** `--flag=v` and `--flag v` both work; the spec writes the latter. */
function parse(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) die(`余分な引数: ${arg}\n${HELP}`);
    const eq = arg.indexOf("=");
    if (eq >= 0) out[arg.slice(2, eq)] = arg.slice(eq + 1);
    else if (arg === "--quiet") out.quiet = "1";
    else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) out[arg.slice(2)] = argv[++i];
    else die(`値が要ります: ${arg}\n${HELP}`);
  }
  return out;
}

async function main(argv: string[]): Promise<void> {
  const f = parse(argv);
  const known = new Set([
    "in",
    "out",
    "init",
    "steps",
    "lr",
    "batch",
    "seed",
    "huber",
    "l2",
    "max",
    "quiet",
    "help",
  ]);
  for (const k of Object.keys(f)) if (!known.has(k)) die(`不明なオプション: --${k}\n${HELP}`);
  if (f.help !== undefined) die(HELP);
  const inPath = f.in;
  if (!inPath) die(`--in が要ります\n${HELP}`);
  const out = f.out ?? "weights/hand-calibrated.json";
  const num = (k: string, d: number) => {
    if (f[k] === undefined) return d;
    const v = Number(f[k]);
    if (!Number.isFinite(v)) die(`--${k} は数値: ${f[k]}`);
    return v;
  };
  const o: FitOpts = {
    steps: num("steps", DEFAULT_FIT.steps),
    lr: num("lr", DEFAULT_FIT.lr),
    batch: num("batch", DEFAULT_FIT.batch),
    seed: num("seed", DEFAULT_FIT.seed),
    delta: num("huber", DEFAULT_FIT.delta),
    l2: num("l2", DEFAULT_FIT.l2),
    verbose: f.quiet === undefined,
  };

  console.log("=== 手牌価値の当てはめ (M11) ===");
  const t0 = performance.now();
  const lane = await loadLane(inPath, f.max === undefined ? undefined : num("max", 0));
  console.log(
    `読み込み ${inPath}  ${((performance.now() - t0) / 1000).toFixed(1)}秒  ` +
      `自摸番 ${lane.records.length}行  半荘 ${lane.games} (うち評価用 ${lane.heldGames})`,
  );
  console.log(
    `(p) 学習 ${lane.trainP.length}行 / 評価 ${lane.heldP.length}行  ` +
      `(v) 和了のみ 学習 ${lane.trainV.length}行 / 評価 ${lane.heldV.length}行`,
  );
  if (lane.trainV.length === 0) die("和了した記録が1行もありません — (v) を当てはめられません");

  // The initial vector: the `--init` file's `hand` block, or the LANE'S OWN
  // header, which is what the seat actually played under. Starting anywhere else
  // would make the "before" column describe a player who never existed.
  const init = f.init ? loadInit(f.init) : lane.header;
  console.log(
    `初期値 ${f.init ?? `${inPath} のヘッダ`}` +
      (JSON.stringify(init) === JSON.stringify(DEFAULT_HAND) ? " (= DEFAULT_HAND)" : ""),
  );
  console.log(
    `最適化: Adam 中心差分  steps ${o.steps}  lr ${o.lr}  batch ${o.batch}  ` +
      `seed ${o.seed}  Huber δ=${o.delta}  L2 ${o.l2}`,
  );
  console.log();

  const t1 = performance.now();
  const res = fit(lane, init, o);
  console.log(
    `当てはめ ${((performance.now() - t1) / 1000).toFixed(1)}秒  ` +
      `採用 step (p)${res.bestStep.p} (v)${res.bestStep.v} — 成分ごとに検証損失が最小の回`,
  );
  console.log();
  console.log(diffTable(init, res.weights));
  console.log();
  for (
    const [name, held] of [["学習", false], ["評価 (奇数シード)", true]] as [string, boolean][]
  ) {
    console.log(metricBlock(`■ ${name} — 初期値`, metricsOf(lane, init, held, o.delta)));
    console.log(metricBlock(`■ ${name} — 当てはめ`, metricsOf(lane, res.weights, held, o.delta)));
    console.log();
  }

  const dir = out.slice(0, out.lastIndexOf("/"));
  if (dir) Deno.mkdirSync(dir, { recursive: true });
  Deno.writeTextFileSync(out, JSON.stringify(ktuneOf(res.weights), null, 2) + "\n");
  console.log(`書き出し ${out}`);
  console.log("■ 検算コマンド");
  console.log(`  deno run --allow-read scripts/hand_report.ts --in ${inPath} --ktune ${out}`);
}

if (import.meta.main) await main(Deno.args);
