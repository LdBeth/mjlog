#!/usr/bin/env -S deno run --allow-read
// M14 の検算 — does the shipped head answer what the trainer thought it fitted?
//
//   deno run --allow-read scripts/dealin_report.ts --lane=runs/dealin/700000 \
//       --weights=weights/dealin-0829.json [--max=2000000]
//
// TWO QUESTIONS, in this order.
//
// 1. REPRODUCTION, and it is exact. `train/dealin_fit.py` writes the first 10k
//    rows' LOGITS through `mlp_forward_np` — the numpy mirror of the same
//    double-accumulate loop `src/ai/mlp.ts` runs — and this script runs the
//    TypeScript head over the same rows of `lane.X.f32` and demands
//    BIT-IDENTICAL float32. Not "close": a head's number reaches `riskOf`,
//    which decides discards, and a whole-hanchan fingerprint moves on one
//    changed decision. A mismatch means the Python and the TypeScript are not
//    the same function, and nothing below it is worth reading.
//
// 2. IS IT BETTER THAN THE MODEL IT REPLACES? `lane.pc.f32` carries the
//    closed-form 計算 `dealinP` for the very same cells, so the reliability
//    tables print learned and computed side by side over the HOLDOUT half
//    (odd hanchan seeds — the same split the fit used, re-derived here rather
//    than trusted). The tenpai head gets the same treatment against computed's
//    own (副露数 × 巡目) prior.
//
// Everything streams: a full lane is gigabytes and nothing here retains a row.

import { mergeDealin, TENPAI_F } from "../src/ai/dealin.ts";
import type { Mlp } from "../src/ai/mlp.ts";
import { buildMlp, closeMlp, mlpForwardBatch } from "../src/ai/mlp.ts";

function die(msg: string): never {
  console.error(msg);
  Deno.exit(2);
}

// ---------------------------------------------------------------------------
// lane files
// ---------------------------------------------------------------------------

interface LaneHead {
  fv: number;
  features: string[];
  tenpaiFeatures: string[];
  negKeep: number;
  seats?: string;
  seed?: number;
}

/** The meta sidecar: its header line, then one small int array per row. */
function readMeta(path: string, cols: number): { head: LaneHead; rows: Int32Array; n: number } {
  const text = Deno.readTextFileSync(path);
  const lines = text.split("\n");
  const head = JSON.parse(lines[0]) as LaneHead;
  const n = lines.length - 1 - (lines[lines.length - 1] === "" ? 1 : 0);
  const rows = new Int32Array(n * cols);
  for (let i = 0; i < n; i++) {
    const a = JSON.parse(lines[i + 1]) as number[];
    for (let c = 0; c < cols; c++) rows[i * cols + c] = a[c];
  }
  return { head, rows, n };
}

/** A float32/uint8 lane file read in row blocks, so nothing is held whole. */
class Blocks {
  private readonly file: Deno.FsFile;
  constructor(path: string) {
    try {
      this.file = Deno.openSync(path, { read: true });
    } catch {
      die(`読めません: ${path}`);
    }
  }
  /** Reads `n` rows into `into` (a view of the right type). */
  read(into: Uint8Array): void {
    let off = 0;
    while (off < into.length) {
      const got = this.file.readSync(into.subarray(off));
      if (got === null) break;
      off += got;
    }
  }
  close(): void {
    this.file.close();
  }
}

// ---------------------------------------------------------------------------
// scoring
// ---------------------------------------------------------------------------

const BANDS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 1.0];

class Table {
  n = new Float64Array(BANDS.length);
  pred = new Float64Array(BANDS.length);
  truth = new Float64Array(BANDS.length);
  bce = 0;
  brier = 0;
  w = 0;
  ny = 0;

  add(p: number, y: number, w: number): void {
    let b = 0;
    while (b < BANDS.length - 1 && p >= BANDS[b]) b++;
    this.n[b] += w;
    this.pred[b] += w * p;
    this.truth[b] += w * y;
    const q = Math.min(1 - 1e-7, Math.max(1e-7, p));
    this.bce += w * -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
    this.brier += w * (p - y) * (p - y);
    this.w += w;
    this.ny += w * y;
  }

  print(name: string): void {
    console.log(
      `  ${name}: BCE ${(this.bce / this.w).toFixed(6)}  ` +
        `Brier ${(this.brier / this.w).toFixed(8)}  ` +
        `実際 ${((100 * this.ny) / this.w).toFixed(4)}%`,
    );
    let lo = 0;
    for (let b = 0; b < BANDS.length; b++) {
      if (this.n[b] === 0) {
        lo = BANDS[b];
        continue;
      }
      console.log(
        `    [${lo.toFixed(4)},${BANDS[b].toFixed(4)})  ` +
          `${String(Math.round(this.n[b])).padStart(12)}  ` +
          `予測 ${(100 * this.pred[b] / this.n[b]).toFixed(3).padStart(8)}%  ` +
          `実際 ${(100 * this.truth[b] / this.n[b]).toFixed(3).padStart(8)}%`,
      );
      lo = BANDS[b];
    }
  }
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

function main(argv: string[]): void {
  let lane = "";
  let weights = "";
  let max = Infinity;
  for (const a of argv) {
    if (a.startsWith("--lane=")) lane = a.slice(7);
    else if (a.startsWith("--weights=")) weights = a.slice(10);
    else if (a.startsWith("--max=")) max = Number(a.slice(6));
    else die(`知らない引数: ${a}`);
  }
  if (lane === "" || weights === "") {
    die("使い方: dealin_report.ts --lane=PREFIX --weights=FILE [--max=N]");
  }

  const raw = JSON.parse(Deno.readTextFileSync(weights)) as Record<string, unknown>;
  const block = (raw.dealin ?? raw) as Record<string, unknown>;
  const w = mergeDealin(block);
  const dealin: Mlp = buildMlp(w.dealin);
  const tenpai: Mlp = buildMlp(w.tenpai);

  try {
    const { head, rows: meta, n } = readMeta(`${lane}.meta.jsonl`, 5);
    const F = head.features.length;
    if (F !== dealin.inputs) {
      die(`レーンの列 ${F} と重みの入力 ${dealin.inputs} が違う`);
    }
    console.log(`レーン ${lane}: ${n} 行 × ${F} 列 (fv=${head.fv}, 負例採用率 ${head.negKeep})`);

    // ---- 1. reproduction ---------------------------------------------------
    const predBytes = Deno.readFileSync(`${lane}.pred.f32`);
    const pred = new Float32Array(predBytes.buffer, predBytes.byteOffset, predBytes.length / 4);
    const nPred = Math.min(pred.length, n);
    {
      const X = new Blocks(`${lane}.X.f32`);
      const chunk = new Float32Array(nPred * F);
      X.read(new Uint8Array(chunk.buffer));
      X.close();
      const out = new Float32Array(nPred);
      mlpForwardBatch(dealin, nPred, chunk, out);
      let bad = 0;
      let worst = 0;
      for (let i = 0; i < nPred; i++) {
        if (out[i] !== pred[i]) {
          bad++;
          worst = Math.max(worst, Math.abs(out[i] - pred[i]));
        }
      }
      console.log(`\n再現 (Python の参照ループ vs TypeScript): ${nPred} 行中 不一致 ${bad}`);
      if (bad > 0) {
        console.log(`  最大差 ${worst}`);
        die("ビット単位で一致しない — 学習した関数と供する関数が違う");
      }
    }

    // ---- 2. learned vs computed, on the holdout ---------------------------
    const learned = new Table();
    const computed = new Table();
    const byBand = new Map<string, [Table, Table]>();
    const bandOf = (junme: number, cls: number): string => {
      const j = junme <= 6
        ? "巡目 1–6"
        : junme <= 9
        ? "巡目 7–9"
        : junme <= 12
        ? "巡目 10–12"
        : "巡目 13+";
      const c = cls === 0 ? "立直" : cls === 1 ? "副露" : "静か";
      return `${j} / ${c}`;
    };

    const STEP = 1 << 16;
    const X = new Blocks(`${lane}.X.f32`);
    const Yb = Deno.readFileSync(`${lane}.y.u8`);
    const PCb = Deno.readFileSync(`${lane}.pc.f32`);
    const pc = new Float32Array(PCb.buffer, PCb.byteOffset, PCb.length / 4);
    const chunk = new Float32Array(STEP * F);
    const out = new Float32Array(STEP);
    const limit = Math.min(n, max);
    for (let base = 0; base < limit; base += STEP) {
      const rowsHere = Math.min(STEP, limit - base);
      X.read(new Uint8Array(chunk.buffer, 0, rowsHere * F * 4));
      mlpForwardBatch(dealin, rowsHere, chunk, out);
      for (let r = 0; r < rowsHere; r++) {
        const i = base + r;
        const seed = meta[i * 5];
        if (seed % 2 === 0) continue; // holdout = odd seeds, as the fit split
        const y = Yb[i];
        const weight = y === 1 ? 1 : 1 / head.negKeep;
        const p = sigmoid(out[r]);
        learned.add(p, y, weight);
        computed.add(pc[i], y, weight);
        const key = bandOf(meta[i * 5 + 1], meta[i * 5 + 2]);
        let pair = byBand.get(key);
        if (!pair) byBand.set(key, pair = [new Table(), new Table()]);
        pair[0].add(p, y, weight);
        pair[1].add(pc[i], y, weight);
      }
    }
    X.close();

    console.log("\n放銃読み — 検証 (奇数の種), 1/keep で重み戻し済み");
    learned.print("学習ヘッド");
    computed.print("計算 (基準)");
    console.log("\n  層別 (BCE 学習 / 計算):");
    for (const [key, [a, b]] of [...byBand].sort()) {
      const flag = a.bce / a.w < b.bce / b.w ? "◎" : "×";
      console.log(
        `    ${key.padEnd(20)} ${(a.bce / a.w).toFixed(6)} / ${(b.bce / b.w).toFixed(6)}  ${flag}`,
      );
    }

    // ---- the tenpai head ---------------------------------------------------
    const tm = readMeta(`${lane}.tmeta.jsonl`, 3);
    const Tb = Deno.readFileSync(`${lane}.T.f32`);
    const T = new Float32Array(Tb.buffer, Tb.byteOffset, Tb.length / 4);
    const ttb = Deno.readFileSync(`${lane}.tt.u8`);
    const tLearned = new Table();
    const tPrior = new Table();
    const priorCol = head.tenpaiFeatures.indexOf("tpPrior");
    const tOut = new Float32Array(tm.n);
    mlpForwardBatch(tenpai, tm.n, T, tOut);
    for (let i = 0; i < tm.n; i++) {
      if (tm.rows[i * 3] % 2 === 0) continue;
      const y = ttb[i];
      tLearned.add(sigmoid(tOut[i]), y, 1);
      tPrior.add(T[i * TENPAI_F + priorCol], y, 1);
    }
    console.log("\n聴牌読み — 検証 (立直の行はレーンに入っていない)");
    tLearned.print("学習ヘッド");
    tPrior.print("計算の事前確率");
  } finally {
    closeMlp(dealin);
    closeMlp(tenpai);
  }
}

if (import.meta.main) main(Deno.args);
