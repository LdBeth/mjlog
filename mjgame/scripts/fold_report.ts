#!/usr/bin/env -S deno run --allow-read
// 押し引きレポート (M13) — what the fold lane actually contains, and what a
// candidate head would have done with it.
//
//   deno run --allow-read scripts/fold_report.ts runs/fold/lane.jsonl \
//     [--fold=weights/fold-MMDD.json] [--max=N]
//
// THE REPRODUCTION CHECK COMES FIRST, and everything below it is worthless if
// it fails. On a lane recorded with `head:"gate"` the seat's verdict WAS the
// incumbent comparison, and the incumbent comparison is `margin < 0` — feature
// 0 of the very row the record carries. So `(x[0] < 0) === verdict` must hold on
// every single row: if it does not, the recorder and the policy have forked and
// the columns do not mean what the header says they mean. (It is the exact
// analogue of `hand_report`'s 再現不一致 line and of `digestRow` on the deal-in
// side — the honesty rule `calibration.ts` states.)
//
// THEN THE LANE'S SHAPE: how often the gate folds, how often the flip fired,
// and the mean settlement broken out by (verdict, taken) — the four cells a
// bandit lane exists to fill. The two off-diagonal cells are the whole point:
// they are the only rows in the corpus that say what happens when the seat does
// the OPPOSITE of what it believes.
//
// WITH `--fold=PATH` the block's head is built and run on every row through the
// SEAT'S OWN `decideFold` — not a re-implementation — giving an agreement
// matrix against the recorded verdict and a self-normalised off-policy value
// (SNIPS) of the candidate on the HOLDOUT (odd seeds, the split
// `train/fold_fit.py` uses). That number is the one to compare with the fit's
// own "SNIPS (holdout)" line: the two must agree to ~1e-6, which is what proves
// the TS forward pass and the MLX-trained block are the same function.
//
// Why SNIPS and not the DR value here: DR needs the fitted q̂(x,a) model, which
// lives in the trainer. This script has the lane and the head and nothing else,
// so it computes the estimator that needs nothing else — and the fit prints the
// same one beside its DR figures for exactly this comparison.

import type { FoldRecord } from "../src/ai/foldcalib.ts";
import { scanFoldCalibration } from "../src/ai/foldcalib.ts";
import { decideFold, FOLD_FEATURES, FOLD_INPUTS, mergeFold } from "../src/ai/fold.ts";
import type { FoldFacts } from "../src/ai/fold.ts";
import { buildMlp, closeMlp } from "../src/ai/mlp.ts";

const HELP = `使い方: deno run --allow-read scripts/fold_report.ts LANE.jsonl \\
  [--fold=weights/fold-MMDD.json] [--max=N]`;

function die(msg: string): never {
  console.error(msg);
  Deno.exit(2);
}

function pct(a: number, b: number): string {
  return b === 0 ? "  —  " : `${(100 * a / b).toFixed(1)}%`;
}

function mean(sum: number, n: number): string {
  return n === 0 ? "    —" : (sum / n).toFixed(0);
}

/**
 * `FoldFacts` back from a recorded row. The names come from `FOLD_FEATURES`, so
 * a reordering of the feature list cannot silently mis-label a column here —
 * and the row's own length is checked against the head's width.
 */
function factsOf(x: number[]): FoldFacts {
  const f: Record<string, number> = {};
  for (let i = 0; i < FOLD_FEATURES.length; i++) f[FOLD_FEATURES[i]] = x[i];
  return f as unknown as FoldFacts;
}

interface Cell {
  n: number;
  sum: number;
}

async function main(argv: string[]): Promise<void> {
  let lane = "";
  let foldPath = "";
  let max = Infinity;
  for (const a of argv) {
    if (a.startsWith("--fold=")) foldPath = a.slice(7);
    else if (a.startsWith("--max=")) {
      max = Number(a.slice(6));
      if (!Number.isInteger(max) || max < 1) die(`--max は1以上の整数: ${a.slice(6)}`);
    } else if (a.startsWith("-")) die(`不明なオプション: ${a}\n${HELP}`);
    else if (!lane) lane = a;
    else die(`余分な引数: ${a}\n${HELP}`);
  }
  if (!lane) die(`記録ファイルが要ります\n${HELP}`);

  // The candidate head, if one was named. Built through `mergeFold`, so a
  // malformed or stale block is refused here exactly as the seat would refuse
  // it — the report never runs a head the game would not.
  let head = null;
  if (foldPath) {
    let json: unknown;
    try {
      json = JSON.parse(Deno.readTextFileSync(foldPath));
    } catch (e) {
      die(`--fold が読めません: ${foldPath}\n${e instanceof Error ? e.message : e}`);
    }
    const obj = json as { fold?: Parameters<typeof mergeFold>[0] };
    try {
      head = buildMlp(mergeFold(obj.fold ?? (json as Parameters<typeof mergeFold>[0])));
    } catch (e) {
      die(`--fold の重みが不正です: ${foldPath}\n${e instanceof Error ? e.message : e}`);
    }
  }

  let rows = 0, mismatch = 0, badWidth = 0;
  let folds = 0, flips = 0;
  // (verdict, taken) — [0]=push/push [1]=push/fold [2]=fold/push [3]=fold/fold
  const quad: Cell[] = [0, 1, 2, 3].map(() => ({ n: 0, sum: 0 }));
  // Agreement of the candidate head with the recorded verdict.
  const agree = [0, 0, 0, 0]; // rec push/head push, push/fold, fold/push, fold/fold
  // SNIPS on the holdout (odd seeds), for the candidate head.
  let wSum = 0, wsqSum = 0, wrSum = 0, holdout = 0;
  const scratch = new Float32Array(FOLD_INPUTS);

  const header = await scanFoldCalibration(lane, (r: FoldRecord, h) => {
    if (rows >= max) return;
    rows++;
    if (r.x.length !== FOLD_INPUTS) {
      badWidth++;
      return;
    }
    // THE CHECK. On a gate-recorded lane the verdict IS `margin < 0`, and
    // `margin` is column 0 of the row beside it.
    if (h.head === "gate" && (r.x[0] < 0) !== r.verdict) mismatch++;
    if (r.verdict) folds++;
    if (r.flipped) flips++;
    quad[(r.verdict ? 2 : 0) + (r.taken ? 1 : 0)].n++;
    quad[(r.verdict ? 2 : 0) + (r.taken ? 1 : 0)].sum += r.delta;
    if (head === null) return;
    const want = decideFold(head, factsOf(r.x), scratch);
    agree[(r.verdict ? 2 : 0) + (want ? 1 : 0)]++;
    // The split `fold_fit.py` uses: odd seeds are the holdout, so this number is
    // an estimate on games the head was NOT fitted on.
    if (r.s % 2 === 1) {
      holdout++;
      // Self-normalised IPS: the candidate is deterministic, so its probability
      // of the logged action is 1 when they agree and 0 when they do not — the
      // weight is `1[a = π(x)] / p`. Rows the candidate would not have played
      // contribute a zero to both sums, which is what makes SNIPS a ratio and
      // not a sum of nothings.
      const w = want === r.taken ? 1 / r.p : 0;
      wSum += w;
      wsqSum += w * w;
      // The reward is `delta/1000` — the same units `train/fold_fit.py` fits in
      // (D7), so the two SNIPS figures are directly comparable.
      wrSum += w * (r.delta / 1000);
    }
  });

  const out: string[] = [];
  out.push(`■ 押し引き記録 ${lane}`);
  out.push(
    `  席 ${header.seats ?? "?"}  種 ${header.seed ?? "?"}  半荘 ${header.games ?? "?"}  ` +
      `ε=${header.eps}  fv=${header.fv}  判定=${
        header.head === "gate" ? "旧ゲート" : "学習ヘッド"
      }`,
  );
  out.push(`  行数 ${rows}${badWidth > 0 ? `  (特徴量長が不正 ${badWidth}行)` : ""}`);
  if (header.features.length !== FOLD_FEATURES.length) {
    out.push(
      `  ⚠ 特徴量の数が違います (記録 ${header.features.length} / 現行 ${FOLD_FEATURES.length})`,
    );
  }
  out.push("");

  out.push("■ 再現検査 (旧ゲートの判定は margin<0 と一致するはず)");
  out.push(
    header.head === "gate"
      ? `  不一致 ${mismatch}行 / ${rows}行` +
        (mismatch === 0 ? "  ✓" : "  ✗ 記録と方策が食い違っています")
      : "  — (学習ヘッドで打った記録なので margin の符号とは一致しません)",
  );
  out.push("");

  out.push("■ 手の内訳");
  out.push(
    `  ベタ降り判定 ${folds}行 (${pct(folds, rows)})   反転 ${flips}行 (${pct(flips, rows)})`,
  );
  out.push("");
  out.push("  判定＼実際      押し                降り");
  const cell = (i: number) =>
    `${String(quad[i].n).padStart(6)}行 ${mean(quad[i].sum, quad[i].n).padStart(7)}点`;
  out.push(`  押し        ${cell(0)}   ${cell(1)}`);
  out.push(`  降り        ${cell(2)}   ${cell(3)}`);
  out.push("  (点は局の収支 deltas[0] の平均 — 対角が方策どおり、非対角が反転行)");

  if (head !== null) {
    out.push("");
    out.push(`■ 候補ヘッド ${foldPath}`);
    const n = agree[0] + agree[1] + agree[2] + agree[3];
    out.push("  記録＼候補        押し        降り");
    out.push(`  押し        ${String(agree[0]).padStart(8)}  ${String(agree[1]).padStart(10)}`);
    out.push(`  降り        ${String(agree[2]).padStart(8)}  ${String(agree[3]).padStart(10)}`);
    out.push(`  一致 ${pct(agree[0] + agree[3], n)}   降り率 ${pct(agree[1] + agree[3], n)}`);
    out.push(
      `  SNIPS (holdout=奇数種, ${holdout}行): ` +
        (wSum > 0 ? (wrSum / wSum).toFixed(4) : "—") +
        `   有効標本 ESS ${wsqSum > 0 ? (wSum * wSum / wsqSum).toFixed(1) : "0.0"}`,
    );
    out.push("  (train/fold_fit.py の SNIPS (holdout) と 1e-6 まで一致するはず)");
    closeMlp(head);
  }
  console.log(out.join("\n"));
  if (mismatch > 0 || badWidth > 0) Deno.exit(1);
}

if (import.meta.main) await main(Deno.args);
