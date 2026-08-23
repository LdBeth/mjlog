#!/usr/bin/env -S deno run --allow-read
// 手牌価値レポート (M11) — where `handOutlook` is wrong about our own hand.
//
//   deno run --allow-read scripts/hand_report.ts --in runs/hand/lane.jsonl \
//     [--ktune weights/hand-calibrated.json] [--max N]
//
// WHAT THIS ANSWERS. The model factors into a forecast and a price,
//
//   EV = P(this hand wins the round) × (what it collects when it does)
//
// and the two are wrong in different ways for different reasons, so they are
// reported SEPARATELY with a no-skill baseline beside each. The forecast is
// broken out by 向聴 × 巡目 because that is the surface the chain is supposed to
// model: a hand two away with fifteen draws left and a tenpai hand on the last
// go-around are the same formula evaluated at opposite ends, and a table that
// pools them says nothing about either.
//
// WHAT IT DOES NOT DO: it never replays a game and it never re-implements the
// model. Every predicted number comes from `handOutlook`, the seat's own
// function, applied to the facts cached in the record — and under the file's own
// header weights that has to reproduce the `pwin`/`value` the seat computed
// exactly. The 再現不一致 line at the top is that check; a nonzero count
// invalidates everything below it.
//
// With `--ktune` the same boards are scored a second time under a candidate
// `hand` block, so the two blocks below answer "would this vector have been
// better on the games the incumbent actually played".

import type { HandRecord } from "../src/ai/handcalib.ts";
import { scanHandCalibration } from "../src/ai/handcalib.ts";
import type { HandWeights } from "../src/ai/handvalue.ts";
import { handOutlook, mergeHand } from "../src/ai/handvalue.ts";

function die(msg: string): never {
  console.error(msg);
  Deno.exit(2);
}

// ---------------------------------------------------------------------------
// accumulators
// ---------------------------------------------------------------------------

/** Upper edges of the 巡目 buckets, ascending; anything above is its own row. */
const JUNME_EDGES = [6, 9, 12];
const JUNME_LABELS = ["序盤(≤6巡)", "中盤(7-9巡)", "終盤(10-12巡)", "大詰(13巡-)"];
/** Rows of the forecast table: how far the resting shape still has to go. */
const SHANTEN_LABELS = ["向聴≥3", "2向聴", "1向聴", "聴牌"];

function junmeBucket(j: number): number {
  for (let i = 0; i < JUNME_EDGES.length; i++) if (j <= JUNME_EDGES[i]) return i;
  return JUNME_EDGES.length;
}

/** ≥3 / 2 / 1 / 0, in the order `meanUkeire`'s four rungs are indexed. */
function shantenBucket(sh: number): number {
  return sh >= 3 ? 0 : sh === 2 ? 1 : sh === 1 ? 2 : 3;
}

interface Cell {
  n: number;
  pred: number;
  truth: number;
  brier: number;
}

const cell = (): Cell => ({ n: 0, pred: 0, truth: 0, brier: 0 });

function addCell(c: Cell, p: number, y: number): void {
  c.n++;
  c.pred += p;
  c.truth += y;
  c.brier += (p - y) * (p - y);
}

/** Everything one weight vector's half of the report is counted from. */
interface Totals {
  grid: Cell[][];
  all: Cell;
  /** Value error, on won records only — see the fit's header for why. */
  vN: number;
  vAbs: number;
  vBias: number;
  vTruth: number;
  vTruths: number[];
  /** Predictions that did not reproduce the recorded answer (own weights only). */
  bad: number;
}

function newTotals(): Totals {
  return {
    grid: SHANTEN_LABELS.map(() => JUNME_LABELS.map(cell)),
    all: cell(),
    vN: 0,
    vAbs: 0,
    vBias: 0,
    vTruth: 0,
    vTruths: [],
    bad: 0,
  };
}

function scan(t: Totals, rec: HandRecord, w: HandWeights, verify: boolean): void {
  const { pwin, value } = handOutlook(rec.facts, w);
  if (verify && (pwin !== rec.pwin || value !== rec.value)) t.bad++;
  addCell(t.grid[shantenBucket(rec.facts.shanten)][junmeBucket(rec.facts.junme)], pwin, rec.won);
  addCell(t.all, pwin, rec.won);
  if (rec.won === 1) {
    t.vN++;
    t.vAbs += Math.abs(value - rec.winPoints);
    t.vBias += value - rec.winPoints;
    t.vTruth += rec.winPoints;
    t.vTruths.push(rec.winPoints);
  }
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

const pct = (x: number) => `${(100 * x).toFixed(2)}%`;
const signed = (x: number) => (x >= 0 ? "+" : "") + (100 * x).toFixed(2) + "pt";

function block(title: string, t: Totals, verify: boolean): string {
  const L: string[] = [];
  L.push(title);
  if (verify) {
    L.push(
      `再現不一致 ${t.bad}件` +
        (t.bad === 0 ? "  (記録した読みを閉形式で完全に再現)" : "  ← 0でなければ以下の数字は無効"),
    );
  }
  L.push("■ 和了率の読み — 向聴 × 巡目 (実測=その局を実際に和了した率)");
  L.push(
    `${"向聴".padEnd(8)}${"巡目帯".padEnd(14)}${"件数".padStart(9)}${"予測".padStart(10)}${
      "実測".padStart(10)
    }${"差".padStart(10)}${"Brier".padStart(10)}${"基準".padStart(10)}`,
  );
  for (let r = 0; r < t.grid.length; r++) {
    for (let c = 0; c < t.grid[r].length; c++) {
      const b = t.grid[r][c];
      if (b.n === 0) continue;
      const pred = b.pred / b.n, truth = b.truth / b.n;
      L.push(
        `${SHANTEN_LABELS[r].padEnd(8)}${JUNME_LABELS[c].padEnd(14)}${String(b.n).padStart(9)}` +
          `${pct(pred).padStart(10)}${pct(truth).padStart(10)}${
            signed(truth - pred).padStart(10)
          }` +
          `${(b.brier / b.n).toFixed(4).padStart(10)}${
            (truth * (1 - truth)).toFixed(4).padStart(10)
          }`,
      );
    }
  }
  const a = t.all;
  if (a.n > 0) {
    const pred = a.pred / a.n, truth = a.truth / a.n;
    L.push(
      `${"合計".padEnd(8)}${"".padEnd(14)}${String(a.n).padStart(9)}${pct(pred).padStart(10)}` +
        `${pct(truth).padStart(10)}${signed(truth - pred).padStart(10)}` +
        `${(a.brier / a.n).toFixed(4).padStart(10)}${
          (truth * (1 - truth)).toFixed(4).padStart(10)
        }`,
    );
    // 基準 is the Brier of the constant predictor at the row's own rate — the
    // best a model with no discrimination could do THERE. A cell whose Brier is
    // at its 基準 is not reading the hand, it is quoting the base rate.
    L.push("  基準 = そのセルの実測率をそのまま返す定数予測の Brier (識別力ゼロの下限)");
  }
  L.push("");
  L.push("■ 打点の読み — 和了した局のみ (実測=その局の実収支、本場・供託込み)");
  if (t.vN === 0) {
    L.push("  和了した記録がありません");
  } else {
    const meanT = t.vTruth / t.vN;
    let baseMae = 0;
    for (const y of t.vTruths) baseMae += Math.abs(meanT - y);
    baseMae /= t.vN;
    const mae = t.vAbs / t.vN;
    L.push(
      `件数 ${t.vN}  平均絶対誤差 ${mae.toFixed(0)}点  ` +
        `基準 ${baseMae.toFixed(0)}点  改善率 ${pct(baseMae <= 0 ? 0 : 1 - mae / baseMae)}`,
    );
    L.push(
      `偏り ${t.vBias / t.vN >= 0 ? "+" : ""}${(t.vBias / t.vN).toFixed(0)}点 (予測−実測)  ` +
        `実測平均 ${meanT.toFixed(0)}点`,
    );
  }
  return L.join("\n");
}

// ---------------------------------------------------------------------------
// entry
// ---------------------------------------------------------------------------

/** Scan one lane under the header's weights and, optionally, a candidate. */
export async function report(
  path: string,
  opts: { ktune?: HandWeights; ktuneNote?: string; max?: number } = {},
): Promise<string> {
  const own = newTotals();
  const cand = opts.ktune ? newTotals() : null;
  let weights: HandWeights | null = null;
  let n = 0;
  const games = new Set<number>();
  const header = await scanHandCalibration(path, (rec, h) => {
    if (opts.max !== undefined && n >= opts.max) return;
    n++;
    games.add(rec.s);
    weights ??= mergeHand(h.w);
    scan(own, rec, weights, true);
    if (cand) scan(cand, rec, opts.ktune!, false);
  });
  if (n === 0) die(`${path}: 記録が1行もありません`);
  const L: string[] = [];
  L.push("=== 手牌価値レポート (M11) ===");
  L.push(`ファイル ${path}  半荘 ${games.size}  自摸番 ${n}  席 ${header.seats ?? "?"}`);
  // Said out loud because it is the first thing a surprising table means: the
  // 和了率 of a bot population is not the 和了率 of a human one, and the model is
  // being graded against the opponents it was recorded against.
  L.push("※ 実測はすべて対戦相手の母集団 (--seats) 依存。人間相手の率とは別物として読むこと");
  L.push("");
  L.push(block("── 記録時の重み ──", own, true));
  if (cand) {
    L.push("");
    L.push(block(`── 候補 ${opts.ktuneNote ?? "--ktune"} ──`, cand, false));
  }
  return L.join("\n");
}

const HELP = "使い方: hand_report.ts --in=PATH [--ktune=PATH] [--max=N]";

/** `--flag=v` and `--flag v` both work; the spec writes the latter. */
function parse(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) die(`余分な引数: ${arg}\n${HELP}`);
    const eq = arg.indexOf("=");
    if (eq >= 0) out[arg.slice(2, eq)] = arg.slice(eq + 1);
    else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) out[arg.slice(2)] = argv[++i];
    else die(`値が要ります: ${arg}\n${HELP}`);
  }
  return out;
}

async function main(argv: string[]): Promise<void> {
  const f = parse(argv);
  for (const k of Object.keys(f)) {
    if (k !== "in" && k !== "ktune" && k !== "max") die(`不明なオプション: --${k}\n${HELP}`);
  }
  if (!f.in) die(`--in が要ります\n${HELP}`);
  let ktune: HandWeights | undefined;
  if (f.ktune) {
    let json: unknown;
    try {
      json = JSON.parse(Deno.readTextFileSync(f.ktune));
    } catch (e) {
      die(`--ktune が読めません: ${f.ktune}\n${e instanceof Error ? e.message : e}`);
    }
    const obj = json as { hand?: Partial<HandWeights> };
    ktune = mergeHand(obj.hand ?? (json as Partial<HandWeights>));
  }
  let max: number | undefined;
  if (f.max !== undefined) {
    max = Number(f.max);
    if (!Number.isInteger(max) || max < 1) die(`--max は1以上の整数: ${f.max}`);
  }
  console.log(await report(f.in, { ktune, ktuneNote: f.ktune, max }));
}

if (import.meta.main) await main(Deno.args);
