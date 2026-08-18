#!/usr/bin/env -S deno run --allow-read
// 較正レポート — where the 計算 deal-in model is actually losing to the oracle.
//
//   deno run --allow-read scripts/calibrate_report.ts runs/calib/*.jsonl
//
// WHAT THIS ANSWERS. The C1 oracle arm is worth roughly −0.22 rank; the counting
// seat that ships converts part of it. The shortfall is a modelling error and it
// factorises exactly the way the model does:
//
//   P(deal in to o on ty) = P(o is tenpai) × P(o rons ty | tenpai) × value
//
// so the loss decomposes the same way, and the three components are reported
// SEPARATELY with a no-skill baseline beside each. Whichever component shows the
// least skill over its baseline is where the next milestone's work belongs —
// that ranking is this report's entire job, and the reliability tables above it
// are there to say WHICH WAY a component is wrong (over- or under-confident,
// and in which region) once the ranking has named it.
//
// WHAT IT DOES NOT DO: it never replays a game and it never re-implements the
// model. Every predicted number is produced by the model's own functions
// (`combineShapes`, `tenpaiPriorOf`, `baseValueOf`) applied to the features
// cached in the record, and every one of them is checked against the digest the
// recorder wrote — the 再現不一致 line at the top is that check. A nonzero count
// there invalidates everything below it.
//
// Options:
//   --w=PATH   re-evaluate under a different 感性 vector (a --ktune file, or a
//              bare ComputedWeights JSON) instead of the file's own header. The
//              reliability tables then describe the CANDIDATE, over exactly the
//              boards the incumbent was recorded on.
//   --max=N    stop after N records (a quick look at a huge lane).

import type { CalibRecord } from "../src/ai/calibration.ts";
import {
  baseValueFromRecord,
  condRowFromRecord,
  dealinRowFrom,
  decode34,
  digestRow,
  scanCalibration,
  tenpaiFromRecord,
} from "../src/ai/calibration.ts";
import type { ComputedWeights } from "../src/ai/computed.ts";
import { mergeComputed, valueOnType } from "../src/ai/computed.ts";

function die(msg: string): never {
  console.error(msg);
  Deno.exit(2);
}

// ---------------------------------------------------------------------------
// accumulators
// ---------------------------------------------------------------------------

/** One reliability bin: how many, what was predicted, what happened. */
interface Bin {
  n: number;
  pred: number;
  truth: number;
}

const bin = (): Bin => ({ n: 0, pred: 0, truth: 0 });

function add(b: Bin, pred: number, truth: number): void {
  b.n++;
  b.pred += pred;
  b.truth += truth;
}

/**
 * A proper-scoring accumulator with its own no-skill twin.
 *
 * STREAMING on purpose: a full lane is tens of millions of (opponent, type)
 * cells, so nothing is retained. The baseline needs no sample either — for the
 * constant predictor at the empirical rate p̄ the two losses are closed forms,
 * BCE = H(p̄) and Brier = p̄(1−p̄), because the labels are 0/1 and their mean IS
 * p̄.
 */
class Score {
  n = 0;
  bce = 0;
  brier = 0;
  sumP = 0;
  sumY = 0;
  sumP2 = 0;
  sumPY = 0;

  push(p: number, y: number): void {
    this.n++;
    this.sumY += y;
    this.sumP += p;
    this.sumP2 += p * p;
    this.sumPY += p * y;
    this.bce += bce(p, y);
    const d = p - y;
    this.brier += d * d;
  }

  /**
   * The best this prediction could do if its only fault were LEVEL: the Brier
   * score after multiplying every prediction by the single constant that
   * minimises it, plus that constant.
   *
   * This is the report's sharpest instrument. A component whose raw Brier is at
   * the no-skill baseline but whose rescaled Brier is well below it is RANKING
   * the cells correctly and merely shouting or whispering — a one-parameter fix
   * (`dealinScale`, `tenpaiFloor`). One whose rescaled score is also at the
   * baseline is not discriminating at all, and no amount of tuning will save it.
   * Σp², Σpy and Σy are all the closed form needs, so this costs one pass and no
   * memory. (`min(1, k·p)` clipping is ignored: k·p exceeds 1 on a vanishing
   * fraction of cells, and pretending otherwise would need the sample back.)
   */
  scaled(): { k: number; brier: number } {
    if (this.n === 0 || this.sumP2 <= 0) return { k: 1, brier: 0 };
    const k = this.sumPY / this.sumP2;
    return { k, brier: (this.sumY - (this.sumPY * this.sumPY) / this.sumP2) / this.n };
  }

  /**
   * The honest "no skill" reference. Anything the model cannot beat here it is
   * not modelling, it is guessing, and a component whose skill is near zero is
   * exactly the one worth rebuilding.
   */
  baseline(): { bce: number; brier: number } {
    if (this.n === 0) return { bce: 0, brier: 0 };
    const p = this.sumY / this.n;
    return { bce: p * bce(p, 1) + (1 - p) * bce(p, 0), brier: p * (1 - p) };
  }

  /** Mean predicted probability — against `sumY/n` this is the calibration bias. */
  meanPred(): number {
    return this.n === 0 ? 0 : this.sumP / this.n;
  }
}

function bce(p: number, y: number): number {
  const q = Math.min(1 - 1e-9, Math.max(1e-9, p));
  return -(y * Math.log(q) + (1 - y) * Math.log(1 - q));
}

/** Mean absolute error against a running twin at the empirical mean. */
class Mae {
  n = 0;
  sum = 0;
  private truths: number[] = [];
  private preds: number[] = [];

  push(pred: number, truth: number): void {
    this.n++;
    this.sum += Math.abs(pred - truth);
    this.truths.push(truth);
    this.preds.push(pred);
  }

  mean(): number {
    return this.n === 0 ? 0 : this.sum / this.n;
  }

  baseline(): number {
    if (this.n === 0) return 0;
    const m = this.truths.reduce((a, b) => a + b, 0) / this.n;
    return this.truths.reduce((a, t) => a + Math.abs(m - t), 0) / this.n;
  }

  bias(): number {
    if (this.n === 0) return 0;
    const mp = this.preds.reduce((a, b) => a + b, 0) / this.n;
    const mt = this.truths.reduce((a, b) => a + b, 0) / this.n;
    return mp - mt;
  }
}

// ---------------------------------------------------------------------------
// the pass
// ---------------------------------------------------------------------------

/** Upper edges of the deal-in probability bands, ascending. Zero is its own. */
const P_EDGES = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.2, 1.0];

interface Totals {
  files: number;
  records: number;
  games: Set<string>;
  cells: number;
  /** False when re-scoring under `--w=`: there is nothing to reproduce. */
  verified: boolean;
  digestBad: number;
  valueBad: number;
  /** Tenpai reliability by (副露 row, 巡目 column), riichi excluded. */
  cell: Bin[][];
  /** …and by predicted decile. */
  dec: Bin[];
  riichiN: number;
  riichiTenpai: number;
  /** Deal-in reliability by predicted band, plus a zero bin at index 0. */
  band: Bin[];
  tenpaiScore: Score;
  waitScore: Score;
  /** Ron types and rows behind (b), for the "how many waits" sanity line. */
  waitTypes: number;
  waitRows: number;
  value: Mae;
}

function newTotals(w: ComputedWeights): Totals {
  const rows = w.tenpaiPrior.length;
  const cols = w.junmeBuckets.length + 1;
  return {
    files: 0,
    records: 0,
    games: new Set(),
    cells: 0,
    verified: true,
    digestBad: 0,
    valueBad: 0,
    cell: Array.from({ length: rows }, () => Array.from({ length: cols }, bin)),
    dec: Array.from({ length: 10 }, bin),
    riichiN: 0,
    riichiTenpai: 0,
    band: Array.from({ length: P_EDGES.length + 1 }, bin),
    tenpaiScore: new Score(),
    waitScore: new Score(),
    waitTypes: 0,
    waitRows: 0,
    value: new Mae(),
  };
}

function bandOf(p: number): number {
  if (p <= 0) return 0;
  for (let i = 0; i < P_EDGES.length; i++) if (p <= P_EDGES[i]) return i + 1;
  return P_EDGES.length;
}

function scan(t: Totals, rec: CalibRecord, w: ComputedWeights, verifyDigest: boolean): void {
  t.records++;
  t.games.add(`${rec.s}`);
  const dora = decode34(rec.dr);

  for (let i = 0; i < rec.o.length; i++) {
    const o = rec.o[i];
    t.cells++;

    // ---- (a) 聴牌 -------------------------------------------------------
    const pT = tenpaiFromRecord(rec, i, w);
    if (o.yc === 0) {
      // A declared riichi is tenpai by the rules of the game. Scoring the
      // model on it would credit it with knowing what the table announced.
      t.riichiN++;
      t.riichiTenpai += o.tt;
    } else {
      t.tenpaiScore.push(pT, o.tt);
      add(t.cell[o.tr][o.tc], pT, o.tt);
      add(t.dec[Math.min(9, Math.floor(pT * 10))], pT, o.tt);
    }

    // ---- (b) 待ち ------------------------------------------------------
    const cond = condRowFromRecord(rec, i, w);
    const row = dealinRowFrom(pT, cond);
    if (verifyDigest && digestRow(row) !== o.ph) t.digestBad++;
    const truth = new Set(o.R);
    for (let ty = 0; ty < 34; ty++) add(t.band[bandOf(row[ty])], row[ty], truth.has(ty) ? 1 : 0);
    if (o.tt === 1) {
      // Conditional on being tenpai, which is what this factor claims to know:
      // the tenpai component above is already scored, and scoring it twice
      // would hide a wait model behind a good base rate.
      t.waitTypes += o.R.length;
      t.waitRows++;
      for (let ty = 0; ty < 34; ty++) {
        t.waitScore.push(Math.min(1, cond[ty]), truth.has(ty) ? 1 : 0);
      }
    }

    // ---- (c) 打点 ------------------------------------------------------
    const vb = baseValueFromRecord(rec, i, w);
    // Same gate as the digest above, and for the same reason: `o.vb` is what
    // the RECORDING vector priced this hand at, so re-deriving it under a
    // candidate vector is supposed to give a different number. Counting that as
    // a reproduction failure made every `--w=` run — the whole point of the
    // option — print "以下の数字は無効" over a table that was perfectly valid.
    if (verifyDigest && vb !== o.vb) t.valueBad++;
    for (let k = 0; k < o.R.length; k++) {
      // `o.V` is `ronValue`, which charges the 本場 — so the prediction must be
      // assembled the way the seat assembles it, surcharge included, or the
      // error column measures a bookkeeping gap instead of a modelling one.
      t.value.push(valueOnType(w, vb, dora[o.R[k]], rec.b), o.V[k]);
    }
  }
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

const pct = (x: number) => `${(100 * x).toFixed(2)}%`;
const signed = (x: number) => (x >= 0 ? "+" : "") + (100 * x).toFixed(2) + "pt";

function junmeLabels(w: ComputedWeights): string[] {
  const out: string[] = [];
  let lo = 1;
  for (const b of w.junmeBuckets) {
    out.push(lo === b ? `${b}巡` : `${lo}-${b}巡`);
    lo = b + 1;
  }
  out.push(`${lo}巡-`);
  return out;
}

function bandLabel(i: number): string {
  if (i === 0) return "0 (予測なし)";
  const lo = i === 1 ? 0 : P_EDGES[i - 2];
  const hi = P_EDGES[i - 1];
  return `${(100 * lo).toFixed(2)}–${(100 * hi).toFixed(2)}%`;
}

/** `1 − loss/baseline`: the fraction of the no-skill loss the model removes. */
function skill(model: number, base: number): number {
  return base <= 0 ? 0 : 1 - model / base;
}

export function render(t: Totals, w: ComputedWeights, weightsNote: string): string {
  const L: string[] = [];
  const p = (s = "") => L.push(s);

  p("=== 較正レポート (M10b) ===");
  p(
    `ファイル ${t.files}  半荘 ${t.games.size}  判断 ${t.records}  ` +
      `局面×他家 ${t.cells}  重み ${weightsNote}`,
  );
  if (t.verified) {
    p(
      `再現不一致 ${t.digestBad}件  打点模型不一致 ${t.valueBad}件` +
        (t.digestBad + t.valueBad === 0
          ? "  (記録した読みを閉形式で完全に再現)"
          : "  ← 0でなければ以下の数字は無効"),
    );
  } else {
    p("再現検査は省略 (候補ベクトルでの再評価 — 記録時の読みと一致しないのが正しい)");
  }
  // Said out loud because it is the first thing a surprising table means: the
  // 聴牌率 of a bot population is not the 聴牌率 of a human one. A seat that
  // riichi-declares the instant it is tenpai leaves almost no silent tenpai
  // behind, so a base rate lifted from human data will read as wildly
  // over-confident here — a fact about the opponents, not about the counting.
  p("※ 実測はすべて対戦相手の母集団 (--seats) 依存。人間相手の率とは別物として読むこと");
  p();

  // ---- tenpai reliability, by table cell ---------------------------------
  p("■ 聴牌読み — 卓表セル別 (立直は除外: ルール上の真値であって予測ではない)");
  p("副露  巡目帯      件数      予測      実測        差");
  const labels = junmeLabels(w);
  for (let r = 0; r < t.cell.length; r++) {
    for (let c = 0; c < t.cell[r].length; c++) {
      const b = t.cell[r][c];
      if (b.n === 0) continue;
      const pred = b.pred / b.n;
      const truth = b.truth / b.n;
      p(
        `${String(r).padStart(2)}   ${labels[c].padEnd(8)}${String(b.n).padStart(9)}` +
          `${pct(pred).padStart(10)}${pct(truth).padStart(10)}${signed(truth - pred).padStart(10)}`,
      );
    }
  }
  p(
    `立直 ${t.riichiN}件 (うち真に聴牌 ${t.riichiTenpai}件、` +
      `${t.riichiN === 0 ? "-" : pct(t.riichiTenpai / t.riichiN)})`,
  );
  p();

  // ---- tenpai reliability, by decile -------------------------------------
  p("■ 聴牌読み — 予測十分位別");
  p("予測帯          件数      予測      実測        差");
  for (let i = 0; i < t.dec.length; i++) {
    const b = t.dec[i];
    if (b.n === 0) continue;
    const lab = `${(10 * i).toFixed(0)}–${(10 * (i + 1)).toFixed(0)}%`;
    p(
      `${lab.padEnd(12)}${String(b.n).padStart(9)}${pct(b.pred / b.n).padStart(10)}` +
        `${pct(b.truth / b.n).padStart(10)}${signed(b.truth / b.n - b.pred / b.n).padStart(10)}`,
    );
  }
  p();

  // ---- deal-in reliability ------------------------------------------------
  p("■ 放銃読み — 予測確率帯別 (全34牌種 × 他家3人。実測=いまロンできる牌である率)");
  p("予測帯              件数      予測      実測        差       倍率");
  for (let i = 0; i < t.band.length; i++) {
    const b = t.band[i];
    if (b.n === 0) continue;
    const pred = b.pred / b.n;
    const truth = b.truth / b.n;
    const ratio = pred <= 0 ? "-" : (truth / pred).toFixed(2);
    p(
      `${bandLabel(i).padEnd(16)}${String(b.n).padStart(10)}${pct(pred).padStart(10)}` +
        `${pct(truth).padStart(10)}${signed(truth - pred).padStart(10)}${ratio.padStart(10)}`,
    );
  }
  p();

  // ---- the decomposition --------------------------------------------------
  const tb = t.tenpaiScore.baseline();
  const wb = t.waitScore.baseline();
  const tBce = t.tenpaiScore.n === 0 ? 0 : t.tenpaiScore.bce / t.tenpaiScore.n;
  const tBri = t.tenpaiScore.n === 0 ? 0 : t.tenpaiScore.brier / t.tenpaiScore.n;
  const wBce = t.waitScore.n === 0 ? 0 : t.waitScore.bce / t.waitScore.n;
  const wBri = t.waitScore.n === 0 ? 0 : t.waitScore.brier / t.waitScore.n;

  p("■ 損失分解 — 基準は「その部分集合の実測率をそのまま返す定数予測」");
  p("成分                        件数     モデル      基準     改善率");
  const row = (name: string, n: number, m: number, b: number, dp = 4) =>
    p(
      `${name.padEnd(24)}${String(n).padStart(10)}${m.toFixed(dp).padStart(11)}` +
        `${b.toFixed(dp).padStart(10)}${pct(skill(m, b)).padStart(11)}`,
    );
  const tScaled = t.tenpaiScore.scaled();
  const wScaled = t.waitScore.scaled();
  row("(a) 聴牌 対数損失", t.tenpaiScore.n, tBce, tb.bce);
  row("(a) 聴牌 Brier", t.tenpaiScore.n, tBri, tb.brier);
  row(`(a) 聴牌 Brier ×${tScaled.k.toFixed(2)}`, t.tenpaiScore.n, tScaled.brier, tb.brier);
  row("(b) 待ち 対数損失", t.waitScore.n, wBce, wb.bce);
  row("(b) 待ち Brier", t.waitScore.n, wBri, wb.brier);
  row(`(b) 待ち Brier ×${wScaled.k.toFixed(2)}`, t.waitScore.n, wScaled.brier, wb.brier);
  row("(c) 打点 平均絶対誤差", t.value.n, t.value.mean(), t.value.baseline(), 0);
  p("  ×k の行は「予測を定数倍しただけで届く最良値」= 水準のずれを除いた識別力");
  p();
  p(
    `(a) 予測平均 ${pct(t.tenpaiScore.meanPred())} / 実測 ` +
      `${pct(t.tenpaiScore.n === 0 ? 0 : t.tenpaiScore.sumY / t.tenpaiScore.n)}`,
  );
  p(
    `(b) 予測平均 ${pct(t.waitScore.meanPred())} / 実測 ` +
      `${pct(t.waitScore.n === 0 ? 0 : t.waitScore.sumY / t.waitScore.n)}  ` +
      "(聴牌が真の局面のみ、34牌種すべて)",
  );
  p(
    `    真の待ち ${t.waitRows === 0 ? "-" : (t.waitTypes / t.waitRows).toFixed(2)}種/局面` +
      `  モデルの合計質量 ${(34 * t.waitScore.meanPred()).toFixed(2)}種/局面`,
  );
  p(`(c) 打点の偏り ${t.value.bias() >= 0 ? "+" : ""}${t.value.bias().toFixed(0)}点 (予測−真値)`);
  p();

  // ---- the ranking, which is the point ------------------------------------
  const ranked = [
    { name: "(a) 聴牌", s: skill(tBri, tb.brier), n: t.tenpaiScore.n },
    { name: "(b) 待ち", s: skill(wBri, wb.brier), n: t.waitScore.n },
    { name: "(c) 打点", s: skill(t.value.mean(), t.value.baseline()), n: t.value.n },
  ].filter((x) => x.n > 0).sort((a, b) => a.s - b.s);
  p("■ 改善余地の順 (無情報基準に対する改善率が低いものから)");
  for (const r of ranked) p(`  ${r.name}  改善率 ${pct(r.s)}`);
  if (ranked.length > 0) {
    p(`オラクル比の取りこぼしが最も大きいのは ${ranked[0].name} の成分。`);
  }
  // Level or location: the one distinction that decides whether the next step
  // is a tuning constant or a new model.
  for (
    const [name, raw, sc, base] of [
      ["(a) 聴牌", tBri, tScaled.brier, tb.brier],
      ["(b) 待ち", wBri, wScaled.brier, wb.brier],
    ] as [string, number, number, number][]
  ) {
    const rawS = skill(raw, base);
    const scS = skill(sc, base);
    p(
      `  ${name}: 定数倍補正で ${pct(rawS)} → ${pct(scS)}  ` +
        (scS - rawS > 0.02
          ? "⇒ 順序は当たっている。水準 (較正定数) のずれが主因"
          : "⇒ 定数倍では届かない。形の読みそのものを作り直す必要がある"),
    );
  }
  return L.join("\n");
}

// ---------------------------------------------------------------------------
// entry
// ---------------------------------------------------------------------------

/** Read, scan and render one or more calibration files. Exported for tests. */
export async function report(
  paths: string[],
  opts: { weights?: ComputedWeights; weightsNote?: string; max?: number } = {},
): Promise<string> {
  if (paths.length === 0) die("較正ファイルを1つ以上指定してください");
  // Streamed, never accumulated: see `scanCalibration`. The accumulators are
  // shaped by the first header seen, because the tenpai table's dimensions come
  // from the weights, and a file recorded under a different vector is refused
  // rather than silently pooled with the rest.
  let own: ComputedWeights | null = null;
  let t: Totals | null = null;
  let w: ComputedWeights | null = null;
  let verify = false;
  let n = 0;
  const start = (h: ComputedWeights) => {
    own = h;
    w = opts.weights ?? h;
    // Re-evaluating under a candidate vector is legitimate and is what a fit
    // does; the digest check only means anything against the file's OWN
    // weights, so it is switched off whenever the two differ.
    verify = w === h;
    t = newTotals(w);
    t.files = paths.length;
    t.verified = verify;
  };
  for (const path of paths) {
    const header = await scanCalibration(path, (rec, h) => {
      if (t === null) start(h.w);
      if (opts.max !== undefined && n >= opts.max) return;
      n++;
      scan(t!, rec, w!, verify);
    });
    if (own === null) start(header.w);
    else if (JSON.stringify(header.w) !== JSON.stringify(own)) {
      die("重みの異なる較正ファイルは混ぜられません (別々にレポートしてください)");
    }
  }
  const note = opts.weightsNote ?? (verify ? "記録時のまま" : "指定ファイル");
  return render(t!, w!, note);
}

async function main(argv: string[]): Promise<void> {
  const paths: string[] = [];
  let weights: ComputedWeights | undefined;
  let note: string | undefined;
  let max: number | undefined;
  for (const arg of argv) {
    if (arg.startsWith("--w=")) {
      const path = arg.slice(4);
      let json: unknown;
      try {
        json = JSON.parse(Deno.readTextFileSync(path));
      } catch (e) {
        die(`--w が読めません: ${path}\n${e instanceof Error ? e.message : e}`);
      }
      const obj = json as { computed?: Partial<ComputedWeights> };
      weights = mergeComputed(obj.computed ?? (json as Partial<ComputedWeights>));
      note = path;
    } else if (arg.startsWith("--max=")) {
      max = Number(arg.slice(6));
      if (!Number.isFinite(max) || max < 1) die(`--max は1以上の整数: ${arg.slice(6)}`);
    } else if (arg.startsWith("-")) {
      die(`不明なオプション: ${arg}\n使い方: calibrate_report.ts [--w=PATH] [--max=N] FILE...`);
    } else paths.push(arg);
  }
  if (paths.length === 0) die("使い方: calibrate_report.ts [--w=PATH] [--max=N] FILE...");
  console.log(await report(paths, { weights, weightsNote: note, max }));
}

if (import.meta.main) await main(Deno.args);
