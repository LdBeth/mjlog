// 較正当てはめ (M10c) — three claims, and the tests are grouped by them:
//
//   1. THE FITTER'S FORWARD IS THE MODEL'S FORWARD. Components (a) and (c) call
//      `computed.ts`'s own functions; component (b) has an inline twin, and the
//      twin is pinned to `condRowFromRecord` ELEMENT BY ELEMENT, bit for bit,
//      over real recorded boards at several random weight vectors. A twin that
//      is tested equal is not a second implementation of the model.
//   2. THE PACKING IS LOSSLESS AND THE SPLIT IS HONEST — a packed row scores
//      what the record it came from scores, and train/val cut by game seed.
//   3. THE FIT RECOVERS WHAT PUT THE DATA THERE. Synthetic records are sampled
//      from a KNOWN, non-default vector; the fitter is pointed at them; the
//      well-identified parameters come back.

import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import type { CalibOpp, CalibRecord } from "../src/ai/calibration.ts";
import {
  baseValueFromRecord,
  CalibrationWriter,
  condRowFromRecord,
  encode34,
  readCalibration,
  tenpaiFromRecord,
  valueFromRecord,
} from "../src/ai/calibration.ts";
import type { ComputedWeights, ShapeBase, WaitContext } from "../src/ai/computed.ts";
import {
  baseValueOf,
  DEFAULT_COMPUTED,
  mergeComputed,
  shapeBaseMasses,
  tenpaiCellOf,
  tenpaiPriorOf,
  valueOnType,
} from "../src/ai/computed.ts";
import {
  condRowInline,
  diffTable,
  fit,
  isVal,
  ktuneOf,
  metricsOf,
  type Packed,
  packRecords,
  paramSpecs,
} from "../scripts/calibrate_fit.ts";
import { pairedRun } from "../src/paired.ts";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/** mulberry32, mirrored from the fitter so a fixture is reproducible. */
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
 * Real play, recorded — the boards the twin must reproduce on. Memoized: three
 * hanchan of `pairedRun` cost a few seconds and every test below wants the same
 * corpus, not a different one.
 */
let CORPUS: Promise<{ recs: CalibRecord[]; w: ComputedWeights }> | null = null;
function realRecords(): Promise<{ recs: CalibRecord[]; w: ComputedWeights }> {
  if (CORPUS === null) CORPUS = recordGames(7171, 3);
  return CORPUS;
}

async function recordGames(
  seed: number,
  games: number,
): Promise<{ recs: CalibRecord[]; w: ComputedWeights }> {
  const path = await Deno.makeTempFile({ suffix: ".jsonl" });
  try {
    const writer = new CalibrationWriter(path, {
      seats: "khhh",
      seed,
      games,
      w: DEFAULT_COMPUTED,
    });
    try {
      pairedRun(games, seed, "khhh", { calibrate: writer });
    } finally {
      writer.close();
    }
    const f = readCalibration(path);
    return { recs: f.records, w: f.header.w };
  } finally {
    await Deno.remove(path);
  }
}

/**
 * A random vector over EVERY constant the wait row consumes — including the
 * four the fit pins as gauge, because the twin has to be right about those too.
 */
function randomWaitWeights(r: () => number, normalize = true): ComputedWeights {
  return mergeComputed({
    waitNormalize: normalize,
    dealinScale: 0.01 + 0.2 * r(),
    expWaitMass: 0.4 + 2.6 * r(),
    shapePrior: {
      "リャンメン": 0.1 + 0.8 * r(),
      "カンチャン": 0.03 + 0.5 * r(),
      "ペンチャン": 0.01 + 0.3 * r(),
      "シャンポン": 0.03 + 0.5 * r(),
      "タンキ": 0.02 + 0.4 * r(),
    },
    yakuhaiShanpon: 0.4 + 2.2 * r(),
    honitsuHot: 0.4 + 2.6 * r(),
    honitsuCold: 0.05 + 1.5 * r(),
    toitoiPair: 0.4 + 2.2 * r(),
    toitoiRun: 0.05 + 1.5 * r(),
    sujiHalfSurvive: 0.5 * r(),
    sujiFullSurvive: 0.35 * r(),
    doraPair: 0.4 + 2.2 * r(),
    doraBridge: 0.4 + 2.2 * r(),
    yakuFactor: { riichi: 0.5 + 0.5 * r(), open: 0.2 + 0.8 * r(), damaten: 0.1 + 0.9 * r() },
  });
}

// ---------------------------------------------------------------------------
// 1. the fitter's forward is the model's forward
// ---------------------------------------------------------------------------

Deno.test("当てはめ: (b)の高速版は閉形式とビット単位で一致する (実戦の記録・複数の重み)", async () => {
  const { recs } = await realRecords();
  const packed = packRecords(recs);
  assert(packed.b.n > 120, `聴牌の行が薄すぎる: ${packed.b.n}`);
  const r = rng(20260815);
  let checked = 0;
  // Six vectors on the normalized path (what the fit uses) and two on the
  // un-normalized one (what the shipped vector uses, and what every "既定" row
  // of the report blocks is scored through).
  for (let trial = 0; trial < 8; trial++) {
    const w = trial < 6 ? randomWaitWeights(r) : randomWaitWeights(r, false);
    const out = new Float64Array(34);
    // The packing walks records in order and takes every `tt === 1` opponent,
    // so the same walk names which record each packed row came from.
    let row = 0;
    for (const rec of recs) {
      for (let i = 0; i < rec.o.length; i++) {
        if (rec.o[i].tt !== 1) continue;
        const want = condRowFromRecord(rec, i, w);
        condRowInline(packed.b, row, w, out);
        for (let ty = 0; ty < 34; ty++) {
          assertEquals(out[ty], want[ty], `trial=${trial} row=${row} ty=${ty}`);
        }
        row++;
        checked++;
      }
    }
    assertEquals(row, packed.b.n);
  }
  assert(checked > 900, `検査した組み合わせが少なすぎる: ${checked}`);
});

Deno.test("当てはめ: (a)と(c)の前向き計算は閉形式そのもの", async () => {
  const { recs } = await realRecords();
  const packed = packRecords(recs);
  const r = rng(99);
  for (let trial = 0; trial < 5; trial++) {
    const w = mergeComputed({
      tenpaiPrior: DEFAULT_COMPUTED.tenpaiPrior.map((row) => row.map(() => 0.02 + 0.9 * r())),
      yakuhaiTenpai: 0.4 * r(),
      tenpaiOtherRiichi: 0.3 + 1.4 * r(),
      tenpaiMeldDora: 0.6 + 1.0 * r(),
      valueRiichi: 2000 + 9000 * r(),
      valueDamaten: 1000 + 6000 * r(),
      valueOpen: 1000 + 6000 * r(),
      valueHonitsu: 3000 + 9000 * r(),
      valueYakuhai: 3000 * r(),
      valuePerDora: 3000 * r(),
      valueDealer: 1 + r(),
      valueCap: 8000 + 16000 * r(),
    });
    let ai = 0, ci = 0;
    for (const rec of recs) {
      for (let i = 0; i < rec.o.length; i++) {
        const o = rec.o[i];
        if (o.yc !== 0) {
          const got = tenpaiPriorOf(
            w,
            packed.a.ml[ai],
            packed.a.j[ai],
            false,
            packed.a.yh[ai] > 0,
            packed.a.or[ai] === 1,
            packed.a.md[ai],
          );
          assertEquals(got, tenpaiFromRecord(rec, i, w), `聴牌 row=${ai}`);
          ai++;
        }
        for (let k = 0; k < o.R.length; k++) {
          const vb = baseValueOf(w, {
            cls: packed.c.cls[ci] as 0 | 1 | 2,
            honitsu: packed.c.honitsu[ci] === 1,
            meldDora: packed.c.md[ci],
            yakuhai: packed.c.yh[ci],
            dealer: packed.c.dl[ci] === 1,
          });
          assertEquals(vb, baseValueFromRecord(rec, i, w), `打点基礎 cell=${ci}`);
          assertEquals(
            valueOnType(w, vb, packed.c.doraTy[ci], packed.c.honba[ci]),
            valueFromRecord(rec, i, o.R[k], w),
            `打点 cell=${ci}`,
          );
          ci++;
        }
      }
    }
    assertEquals(ai, packed.a.n);
    assertEquals(ci, packed.c.n);
  }
});

// ---------------------------------------------------------------------------
// 2. the packing and the split
// ---------------------------------------------------------------------------

Deno.test("当てはめ: 分割は半荘の種で切られ、行では切られない", async () => {
  const { recs } = await realRecords();
  const packed = packRecords(recs);
  // Every row of one game lands on one side — that is the whole point of
  // splitting by seed: two decisions of a hanchan share almost everything.
  const side = new Map<number, number>();
  let ai = 0;
  for (const rec of recs) {
    for (const o of rec.o) {
      if (o.yc === 0) continue;
      const s = rec.s ?? 0;
      const v = packed.a.val[ai++];
      const seen = side.get(s);
      if (seen === undefined) side.set(s, v);
      else assertEquals(seen, v, `半荘 ${s} の行が両側に散っている`);
    }
  }
  assertEquals(isVal(700003), isVal(700003), "同じ種は必ず同じ側");
  let val = 0;
  for (let s = 0; s < 4000; s++) if (isVal(s)) val++;
  assert(val > 600 && val < 1000, `検証側の割合が2割から離れすぎ: ${val}/4000`);
});

Deno.test("当てはめ: 立直の行は(a)に入らない (ルール上の真値であって予測ではない)", async () => {
  const { recs } = await realRecords();
  const packed = packRecords(recs);
  let silent = 0, riichi = 0, tenpai = 0, cells = 0;
  for (const rec of recs) {
    for (const o of rec.o) {
      if (o.yc === 0) riichi++;
      else silent++;
      if (o.tt === 1) tenpai++;
      cells += o.R.length;
    }
  }
  assert(riichi > 0, "この記録に立直が1件もない");
  assertEquals(packed.a.n, silent);
  assertEquals(packed.b.n, tenpai);
  assertEquals(packed.c.n, cells);
});

// ---------------------------------------------------------------------------
// 3. the fit recovers what put the data there
// ---------------------------------------------------------------------------

/** The vector the synthetic truths are sampled FROM. Nothing default about it. */
const TRUTH: ComputedWeights = mergeComputed({
  waitNormalize: true,
  tenpaiPrior: [
    [0.08, 0.20, 0.34, 0.44],
    [0.14, 0.28, 0.44, 0.56],
    [0.20, 0.36, 0.52, 0.64],
    [0.26, 0.44, 0.60, 0.70],
    [0.30, 0.50, 0.66, 0.76],
  ],
  yakuhaiTenpai: 0.14,
  tenpaiOtherRiichi: 0.75,
  tenpaiMeldDora: 1.20,
  shapePrior: {
    "リャンメン": 0.45, // pinned gauge — the fit must NOT need to move it
    "カンチャン": 0.30,
    "ペンチャン": 0.05,
    "シャンポン": 0.24,
    "タンキ": 0.20,
  },
  yakuhaiShanpon: 2.2,
  honitsuHot: 1.6, // pinned gauge
  honitsuCold: 0.30,
  toitoiPair: 1.5, // pinned gauge
  toitoiRun: 0.35,
  sujiHalfSurvive: 0.16,
  sujiFullSurvive: 0.06,
  doraPair: 1.6,
  doraBridge: 1.4,
  expWaitMass: 1.90,
  yakuFactor: { riichi: 1, open: 0.70, damaten: 0.45 },
  valueRiichi: 6000,
  valueDamaten: 3400,
  valueOpen: 4600,
  valueHonitsu: 8400,
  valueYakuhai: 1300,
  valuePerDora: 1300,
  valueDealer: 1.55,
  valueCap: 16000,
});

/** One synthetic decision: plausible public state, masses from the real counter. */
function synthRecord(r: () => number, seed: number, n: number): CalibRecord {
  const unseen = Array.from({ length: 34 }, () => Math.floor(r() * 5));
  const dora = new Array<number>(34).fill(0);
  for (let k = 0; k < 1 + Math.floor(r() * 2); k++) dora[Math.floor(r() * 34)] = 1;
  const rec: CalibRecord = {
    s: seed,
    n,
    k: Math.floor(r() * 8),
    b: Math.floor(r() * 3),
    c: 0,
    j: 1 + Math.floor(r() * 17),
    w: 70 - Math.floor(r() * 60),
    t: r() < 0.5 ? 1 : 0,
    rw: 27,
    sw: 27,
    dr: encode34(dora),
    o: [],
  };
  for (let i = 0; i < 3; i++) {
    const genbutsu = new Set<number>();
    for (let ty = 0; ty < 34; ty++) if (r() < 0.14) genbutsu.add(ty);
    const vh = 0b1110001; // 東 (場風/自風) と三元牌
    const valueHonors = new Set<number>([27, 31, 32, 33]);
    const ctx: WaitContext = { unseen, genbutsu, valueHonors, dora };
    const bases: ShapeBase[] = [];
    for (let ty = 0; ty < 34; ty++) bases.push(shapeBaseMasses(ty, ctx));
    const col = <K extends keyof ShapeBase>(k: K) => encode34(bases.map((b) => b[k]));
    const ml = Math.floor(r() * 5);
    const cell = tenpaiCellOf(DEFAULT_COMPUTED, ml, rec.j);
    // 30% of rows carry one of the conditioning facts; the rest are plain, so
    // the table cells are identified before the multipliers get to argue.
    const yc: 0 | 1 | 2 = ml > 0 ? 1 : (r() < 0.12 ? 0 : 2);
    const yh = ml > 0 && r() < 0.3 ? 1 : 0;
    const hs = (ml >= 2 && r() < 0.3 ? 1 + Math.floor(r() * 3) : 0) as 0 | 1 | 2 | 3;
    const to = (ml >= 2 && r() < 0.25 ? 1 : 0) as 0 | 1;
    const md = r() < 0.25 ? 1 + Math.floor(r() * 3) : 0;
    const dl = (r() < 0.25 ? 1 : 0) as 0 | 1;
    const or = (r() < 0.2 ? 1 : 0) as 0 | 1;
    const o: CalibOpp = {
      tp: 0,
      tr: cell.row,
      tc: cell.col,
      ml,
      om: ml,
      yc,
      yh,
      hs,
      to,
      md,
      dl,
      or,
      vh,
      vb: 0,
      gy: col("ryanmen"),
      gd: col("ryanmenDora"),
      gh: col("ryanmenHalf"),
      gf: col("ryanmenFull"),
      gk: col("kanchan"),
      gp: col("penchan"),
      gs: col("shanpon"),
      gt: col("tanki"),
      ph: "00000000",
      tt: 0,
      R: [],
      V: [],
    };
    rec.o.push(o);
  }
  return rec;
}

/** Sample the oracle half of a record from `TRUTH` — the generative story. */
function sampleTruth(rec: CalibRecord, r: () => number): void {
  for (let i = 0; i < rec.o.length; i++) {
    const o = rec.o[i];
    const pT = tenpaiFromRecord(rec, i, TRUTH);
    o.tp = pT;
    o.vb = baseValueFromRecord(rec, i, TRUTH);
    o.tt = r() < pT ? 1 : 0;
    if (o.tt !== 1) continue;
    const cond = condRowFromRecord(rec, i, TRUTH);
    for (let ty = 0; ty < 34; ty++) {
      if (r() < Math.min(1, cond[ty])) {
        o.R.push(ty);
        o.V.push(valueFromRecord(rec, i, ty, TRUTH));
      }
    }
  }
}

function synthPacked(records: number, seed: number): Packed {
  const r = rng(seed);
  const recs: CalibRecord[] = [];
  for (let n = 0; n < records; n++) {
    // Six decisions per "hanchan", so the seed split has something to cut.
    const rec = synthRecord(r, 900000 + Math.floor(n / 6), n % 6);
    sampleTruth(rec, r);
    recs.push(rec);
  }
  return packRecords(recs);
}

Deno.test("当てはめ: 既知のベクトルから作った記録で、そのベクトルを取り戻す", () => {
  const p = synthPacked(6000, 424242);
  assert(p.b.n > 3000, `合成データの聴牌行が薄い: ${p.b.n}`);
  assert(p.c.n > 3000, `合成データのロン牌が薄い: ${p.c.n}`);
  const res = fit(p, {
    epochs: 45,
    lr: 0.15,
    batch: 2048,
    seed: 7,
    l2: 1e-5, // the recovery claim is about the DATA, not about the prior
    verbose: false,
  });
  const w = res.weights;
  const specs = paramSpecs();
  const support = new Map(specs.map((s) => [s.name, s.support(p.support)]));

  // ---- (a) the table cells that have support ----------------------------
  let checkedCells = 0;
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 4; col++) {
      const n = p.support.cell[row][col];
      if (n < 300) continue;
      assertAlmostEquals(
        w.tenpaiPrior[row][col],
        TRUTH.tenpaiPrior[row][col],
        0.07,
        `聴牌表 [${row}][${col}] (支持 ${n}行)`,
      );
      checkedCells++;
    }
  }
  assert(checkedCells >= 12, `支持のあるセルが少なすぎる: ${checkedCells}`);

  // ---- (b) the level, which is the best-identified thing in the wait row --
  assertAlmostEquals(w.expWaitMass, TRUTH.expWaitMass, 0.30, "期待待ち牌種数");
  assertAlmostEquals(w.yakuFactor.damaten, TRUTH.yakuFactor.damaten, 0.15, "黙聴の役率");
  assertAlmostEquals(w.honitsuCold, TRUTH.honitsuCold, 0.35, "染め手の外側");

  // ---- (c) the value bases, which the truths pin exactly ----------------
  for (
    const [key, tol] of [
      ["valueRiichi", 0.10],
      ["valueDamaten", 0.14],
      ["valueOpen", 0.14],
      ["valuePerDora", 0.30],
    ] as [keyof ComputedWeights & string, number][]
  ) {
    const got = w[key] as number;
    const want = TRUTH[key] as number;
    assert(
      Math.abs(got - want) <= tol * want,
      `${key}: ${got.toFixed(0)} が ${want} から ${(100 * tol).toFixed(0)}% 以上ずれている ` +
        `(支持 ${support.get(key)})`,
    );
  }

  // …and the fit is an IMPROVEMENT on the val half by the objective it claims.
  const mDef = metricsOf(p, mergeComputed({}), 1, {
    posWeight: 1,
    huber: 0.5,
    wa: 1,
    wb: 1,
    wc: 1,
    l2: 0,
  });
  const mFit = metricsOf(p, w, 1, { posWeight: 1, huber: 0.5, wa: 1, wb: 1, wc: 1, l2: 0 });
  assert(mFit.aBce < mDef.aBce, `(a) が既定より悪い: ${mFit.aBce} vs ${mDef.aBce}`);
  assert(mFit.bBce < mDef.bBce, `(b) が既定より悪い: ${mFit.bBce} vs ${mDef.bBce}`);
  assert(mFit.cHuber < mDef.cHuber, `(c) が既定より悪い: ${mFit.cHuber} vs ${mDef.cHuber}`);
});

// ---------------------------------------------------------------------------
// the artefacts
// ---------------------------------------------------------------------------

Deno.test("当てはめ: 書き出したベクトルは --ktune の合流路で読める形をしている", () => {
  const p = synthPacked(400, 5);
  const res = fit(p, { epochs: 2, lr: 0.1, batch: 512, seed: 3, verbose: false });
  const k = ktuneOf(res.weights);
  // Round-tripped through JSON, exactly as `loadKtune` would see it.
  const round = JSON.parse(JSON.stringify(k)) as { computed: Partial<ComputedWeights> };
  const merged = mergeComputed(round.computed);
  assertEquals(merged.waitNormalize, true, "当てはめは正規化された行の模型");
  assertEquals(merged.valuePerHonba, 300, "本場は規則であって当てはめ対象ではない");
  assertEquals(merged.tenpaiFloor, DEFAULT_COMPUTED.tenpaiFloor, "消費側は触らない");
  assertEquals(merged.planner, DEFAULT_COMPUTED.planner);
  assertEquals(merged.tenpaiPrior.length, 5);
  for (const row of merged.tenpaiPrior) {
    assertEquals(row.length, 4);
    for (const v of row) assert(v > 0 && v < 1, `表のセルが確率になっていない: ${v}`);
  }
  assertEquals(merged.yakuFactor.riichi, 1, "立直は必ずロンできる — 基準クラス");
  for (const v of [merged.sujiHalfSurvive, merged.sujiFullSurvive]) {
    assert(v >= 0 && v <= 1, `残存率が [0,1] を外れた: ${v}`);
  }
  for (const v of Object.values(merged.shapePrior)) assert(v > 0, "形の事前分布が非正");
  // Every fitted value is finite and the diff table names all of them.
  const table = diffTable(res.weights, p.support);
  for (const s of paramSpecs()) {
    assert(Number.isFinite(s.get(res.weights)), `${s.name} が有限でない`);
    assert(table.includes(s.name), `差分表に ${s.name} がない`);
  }
});

Deno.test("当てはめ: 同じ --seed なら同じベクトルが出る", () => {
  const p = synthPacked(300, 11);
  const a = fit(p, { epochs: 3, lr: 0.1, batch: 256, seed: 4242, verbose: false });
  const b = fit(p, { epochs: 3, lr: 0.1, batch: 256, seed: 4242, verbose: false });
  assertEquals(JSON.stringify(ktuneOf(a.weights)), JSON.stringify(ktuneOf(b.weights)));
  const c = fit(p, { epochs: 3, lr: 0.1, batch: 256, seed: 99, verbose: false });
  assert(
    JSON.stringify(ktuneOf(c.weights)) !== JSON.stringify(ktuneOf(a.weights)),
    "種を変えても一字一句同じ — 混ぜ方が種に依存していない",
  );
});
