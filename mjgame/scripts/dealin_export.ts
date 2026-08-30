#!/usr/bin/env -S deno run --allow-read --allow-write
// M14 の学習レーン書き出し — a v3 calibration lane into flat arrays the MLX
// trainer can `np.fromfile`.
//
//   deno run --allow-read --allow-write scripts/dealin_export.ts \
//       runs/calib/v3-700000.jsonl --out=runs/dealin/700000 [--neg-keep=0.1]
//
// WHAT COMES OUT (`--out P`):
//   P.X.f32       N × DEALIN_F float32 — one row per kept (decision, opponent,
//                 tile type), through `dealinFeatures` and nothing else
//   P.y.u8        N labels: 1 iff that type is in the oracle's ron set `R`
//   P.pc.f32      N float32 — the CLOSED-FORM model's own `dealinP` for the same
//                 cell, re-derived from the record under the lane's own weights.
//                 The baseline every holdout table is read against: a head that
//                 cannot beat this column is not worth shipping.
//   P.meta.jsonl  a header line, then `[seed, junme, class, opp, type]` per row
//   P.T.f32       M × TENPAI_F — one row per (decision, opponent) NOT under a
//                 declared riichi (a declaration is tenpai by rule, so those
//                 rows are neither trained nor served: see `dealin.ts`)
//   P.tt.u8       M labels: the oracle's 聴牌 truth
//   P.tmeta.jsonl the same header, then `[seed, junme, class]` per tenpai row
//
// NEGATIVE SUBSAMPLING. Almost every cell is a 0 — three opponents × 34 types,
// of which a handful are ever ronnable — so a full lane is 99.9% negatives and
// mostly disk. Negatives are kept with probability `--neg-keep` (0.1 by
// default) on a seeded rng, and the fit re-weights the survivors by `1/keep`,
// which is unbiased for the log-likelihood. Positives are ALWAYS kept. The rng
// is seeded and consumed in stream order, so the same lane exports byte-identical
// files.
//
// THE REPRODUCTION CHECK. A record written by a seat that was RUNNING the heads
// carries `fh`, the digest of the 34 × F rows it was served. Every such row is
// re-derived here and checked against it; a single mismatch means the offline
// features and the served features have drifted, which would make the whole fit
// a measurement of a player nobody ships, so it exits 1. A lane recorded on the
// plain computed champion — which is how the FIRST lane must be recorded — has
// no `fh`, and that is normal: the check is skipped with a printed note.

import type { CalibRecord } from "../src/ai/calibration.ts";
import { dealinRowFromRecord, digestRow, scanCalibration } from "../src/ai/calibration.ts";
import type { ComputedWeights } from "../src/ai/computed.ts";
import { mergeComputed } from "../src/ai/computed.ts";
import {
  DEALIN_F,
  DEALIN_FEATURES,
  DEALIN_FV,
  dealinFeatures,
  dealinStateFromRecord,
  TENPAI_F,
  TENPAI_FEATURES,
  tenpaiFeatures,
} from "../src/ai/dealin.ts";
import { sfc32 } from "../src/rng.ts";

function die(msg: string): never {
  console.error(msg);
  Deno.exit(2);
}

// ---------------------------------------------------------------------------
// buffered binary/text sinks — a full lane is gigabytes, so nothing is retained
// ---------------------------------------------------------------------------

class Sink {
  private file: Deno.FsFile;
  private buf: Uint8Array;
  private used = 0;
  bytes = 0;

  constructor(readonly path: string, cap = 1 << 20) {
    const dir = path.slice(0, path.lastIndexOf("/"));
    if (dir) Deno.mkdirSync(dir, { recursive: true });
    this.file = Deno.openSync(path, { create: true, write: true, truncate: true });
    this.buf = new Uint8Array(cap);
  }

  write(bytes: Uint8Array): void {
    if (bytes.length > this.buf.length - this.used) this.flush();
    if (bytes.length > this.buf.length) {
      let n = 0;
      while (n < bytes.length) n += this.file.writeSync(bytes.subarray(n));
      this.bytes += bytes.length;
      return;
    }
    this.buf.set(bytes, this.used);
    this.used += bytes.length;
    this.bytes += bytes.length;
  }

  private flush(): void {
    let n = 0;
    while (n < this.used) n += this.file.writeSync(this.buf.subarray(n, this.used));
    this.used = 0;
  }

  close(): void {
    this.flush();
    this.file.close();
  }
}

const ENC = new TextEncoder();

// ---------------------------------------------------------------------------
// arguments
// ---------------------------------------------------------------------------

interface Args {
  lane: string;
  out: string;
  negKeep: number;
  seed: number;
  max: number;
}

function parseArgs(argv: string[]): Args {
  let lane = "";
  let out = "";
  let negKeep = 0.1;
  let seed = 20260829;
  let max = Infinity;
  for (const a of argv) {
    if (a.startsWith("--out=")) out = a.slice(6);
    else if (a.startsWith("--neg-keep=")) negKeep = Number(a.slice(11));
    else if (a.startsWith("--seed=")) seed = Number(a.slice(7));
    else if (a.startsWith("--max=")) max = Number(a.slice(6));
    else if (a.startsWith("-")) die(`知らない引数: ${a}`);
    else if (lane === "") lane = a;
    else die(`レーンは1つだけです: ${a}`);
  }
  if (lane === "") die("使い方: dealin_export.ts <lane.jsonl> --out=PREFIX [--neg-keep=0.1]");
  if (out === "") die("--out=PREFIX が要ります");
  if (!(negKeep > 0 && negKeep <= 1)) die(`--neg-keep は 0 < x ≤ 1: ${negKeep}`);
  if (!Number.isFinite(seed)) die(`--seed が数値ではありません: ${seed}`);
  return { lane, out, negKeep, seed, max };
}

// ---------------------------------------------------------------------------
// the pass
// ---------------------------------------------------------------------------

/** 0 立直 / 1 副露 / 2 静か — the strata every holdout table is cut by. */
function classOf(rec: CalibRecord, i: number): 0 | 1 | 2 {
  const declared = ((rec.ri ?? 0) & (1 << (i + 1))) !== 0;
  if (declared) return 0;
  return rec.o[i].om > 0 ? 1 : 2;
}

async function main(argv: string[]): Promise<void> {
  const args = parseArgs(argv);
  const X = new Sink(`${args.out}.X.f32`);
  const Y = new Sink(`${args.out}.y.u8`);
  const PC = new Sink(`${args.out}.pc.f32`);
  const META = new Sink(`${args.out}.meta.jsonl`);
  const T = new Sink(`${args.out}.T.f32`);
  const TT = new Sink(`${args.out}.tt.u8`);
  const TMETA = new Sink(`${args.out}.tmeta.jsonl`);

  // One buffer per artifact, viewed as bytes: `Sink.write` copies, so the same
  // view is handed over again and again. (Everything below is little-endian —
  // arm64/x86 native order, which is what numpy's `<f4` reads.)
  const feats = new Float32Array(34 * DEALIN_F);
  const featBytes = new Uint8Array(feats.buffer);
  const trow = new Float32Array(TENPAI_F);
  const trowBytes = new Uint8Array(trow.buffer);
  const pcOne = new Float32Array(1);
  const pcBytes = new Uint8Array(pcOne.buffer);
  const one = new Uint8Array(1);
  const rng = sfc32(args.seed >>> 0);

  let seen = 0;
  let rows = 0;
  let pos = 0;
  let trows = 0;
  let tpos = 0;
  let checked = 0;
  let mismatch = 0;
  let noDigest = 0;
  let w: ComputedWeights = mergeComputed({});
  let headerLine = "";
  let stop = false;

  const header = await scanCalibration(args.lane, (rec, h) => {
    if (stop) return;
    if (seen === 0) {
      w = mergeComputed(h.w);
      headerLine = JSON.stringify({
        kind: "mjgame-dealin-lane",
        fv: DEALIN_FV,
        lane: args.lane,
        seats: h.seats,
        seed: h.seed,
        games: h.games,
        negKeep: args.negKeep,
        exportSeed: args.seed,
        features: DEALIN_FEATURES,
        tenpaiFeatures: TENPAI_FEATURES,
        columns: ["seed", "junme", "class", "opp", "type"],
        tenpaiColumns: ["seed", "junme", "class"],
      }) + "\n";
      META.write(ENC.encode(headerLine));
      TMETA.write(ENC.encode(headerLine));
    }
    if (seen >= args.max) {
      stop = true;
      return;
    }
    seen++;

    // The state is rebuilt from the record alone — the same object
    // `dealinStateOf` builds live, which is what `fh` verifies below.
    const st = dealinStateFromRecord(rec);
    const s = rec.s ?? 0;

    for (let i = 0; i < 3; i++) {
      const o = rec.o[i];
      dealinFeatures(st, i, feats);
      if (o.fh !== undefined) {
        checked++;
        if (digestRow(feats) !== o.fh) mismatch++;
      } else noDigest++;

      const baseline = dealinRowFromRecord(rec, i, w);
      const ron = new Set(o.R);
      const cls = classOf(rec, i);
      for (let ty = 0; ty < 34; ty++) {
        const y = ron.has(ty) ? 1 : 0;
        if (y === 0 && rng.float() >= args.negKeep) continue;
        X.write(featBytes.subarray(ty * DEALIN_F * 4, (ty + 1) * DEALIN_F * 4));
        one[0] = y;
        Y.write(one);
        pcOne[0] = baseline[ty];
        PC.write(pcBytes);
        META.write(ENC.encode(`[${s},${rec.j},${cls},${i},${ty}]\n`));
        rows++;
        pos += y;
      }

      // The tenpai head: every opponent who has NOT declared. A declaration is
      // tenpai by the rules of the game, so those rows would teach the head a
      // label it will never be asked for.
      if (cls !== 0) {
        tenpaiFeatures(st, i, trow);
        T.write(trowBytes);
        one[0] = o.tt;
        TT.write(one);
        TMETA.write(ENC.encode(`[${s},${rec.j},${cls}]\n`));
        trows++;
        tpos += o.tt;
      }
    }
  });

  for (const s of [X, Y, PC, META, T, TT, TMETA]) s.close();

  const pct = (a: number, b: number) => (b > 0 ? `${((100 * a) / b).toFixed(3)}%` : "—");
  console.log(`レーン: ${args.lane} (v${header.v}, ${header.seats ?? "?"}, 種 ${header.seed})`);
  console.log(`  決定 ${seen} 件 → 放銃行 ${rows} (正例 ${pos}, ${pct(pos, rows)})`);
  console.log(`  聴牌行 ${trows} (聴牌 ${tpos}, ${pct(tpos, trows)}) — 立直の行は除いてある`);
  console.log(`  負例採用率 ${args.negKeep} (重み 1/${args.negKeep} を当てはめで戻す)`);
  console.log(`  列: 放銃 ${DEALIN_F}, 聴牌 ${TENPAI_F} (fv=${DEALIN_FV})`);
  if (checked > 0) {
    console.log(`  fh 照合: ${checked} 件中 不一致 ${mismatch}`);
  }
  if (noDigest > 0) {
    console.log(
      `  註: ${noDigest} 件の相手行に fh がない — ` +
        "計算のみの席で録ったレーンなので、供された特徴量との照合は省略した",
    );
  }
  if (mismatch > 0) {
    die(
      `再現不一致 ${mismatch} 件: 記録から作り直した特徴量が、席が供されたものと違う。` +
        "この当てはめは誰も指さない打ち手を測ることになるので中止する",
    );
  }
  if (rows === 0) die("行が1つも出なかった");
}

if (import.meta.main) await main(Deno.args);
