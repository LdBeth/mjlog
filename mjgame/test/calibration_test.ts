// 較正記録 — the per-decision prediction recorder (M10a).
//
// Three claims, and the tests are grouped by them:
//   1. the refactor that split 計算's wait model into COUNT and JUDGMENT is
//      arithmetic-preserving — the closed form reproduces the expressions the
//      module shipped with, bit for bit, over an exhaustive sweep;
//   2. the cached features are SUFFICIENT — the closed form rebuilds the exact
//      float32 `dealinP` row from a record alone, with no game to replay;
//   3. the recorder is INVISIBLE — a run with `--calibrate` plays the identical
//      game, seed for seed, to one without.

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import type { Reads } from "../src/ai/augmented.ts";
import { calibrationReads } from "../src/ai/augmented.ts";
import type { CalibRecord } from "../src/ai/calibration.ts";
import {
  baseValueFromRecord,
  buildCalibRecord,
  CALIB_KIND,
  CALIB_VERSION,
  CalibrationWriter,
  dealinRowFromRecord,
  decode34,
  digestRow,
  encode34,
  parseCalibration,
  readCalibration,
  tenpaiFromRecord,
  valueFromRecord,
} from "../src/ai/calibration.ts";
import type {
  ComputedTraceRef,
  ComputedWeights,
  WaitContext,
  WaitShape,
} from "../src/ai/computed.ts";
import {
  computedReads,
  DEFAULT_COMPUTED,
  mergeComputed,
  shapeBaseMasses,
  shapeFlagsOf,
  WAIT_SHAPES,
  waitShapeWeights,
} from "../src/ai/computed.ts";
import { pairedRun } from "../src/main.ts";
import type { Observation } from "../src/observe.ts";
import { AKA_5P } from "../src/tiles.ts";
import { report } from "../scripts/calibrate_report.ts";
import { tiles } from "./helpers.ts";

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

function obsOf(o: Partial<Observation> = {}): Observation {
  return {
    seat: 0,
    kyoku: 0,
    honba: 0,
    kyotaku: 0,
    junme: 1,
    wallRemaining: 70,
    hand: [],
    drawn: null,
    melds: [[], [], [], []],
    rivers: [[], [], [], []],
    scores: [25000, 25000, 25000, 25000],
    riichi: [false, false, false, false],
    riichiJunme: [-1, -1, -1, -1],
    doraIndicators: [],
    seatWind: 27,
    roundWind: 27,
    akaIds: new Set<Tile>(AKA_5P),
    shanten: 3,
    waits: [],
    ronnable: [],
    katagari: false,
    ukeire: [],
    doraCount: 0,
    furiten: { permanent: false, temporary: false, riichi: false },
    discardInfo: new Map(),
    tsumogiriLock: false,
    danger: new Map(),
    violations: [0, 0, 0, 0],
    legal: [],
    ...o,
  };
}

const river = (spec: string, junme = 1): RiverEntry[] =>
  tiles(spec).map((tile) => ({ tile, junme, tsumogiri: false, riichiDeclare: false }));

const meldOf = (kind: Meld["kind"], who: number, spec: string): Meld => {
  const t = tiles(spec);
  return { kind, who, fromWho: kind === "ankan" ? who : 0, tiles: t, calledTile: t[0] };
};

/** A synthetic oracle row: opponent `i` rons `types` for `value` each. */
function fakeOracle(tenpai: number[], rons: number[][], value = 5200): Reads {
  return {
    tenpaiP: tenpai,
    dealinP: rons.map((tys) => {
      const a = new Float32Array(34);
      for (const ty of tys) a[ty] = 1;
      return a;
    }),
    dealinValue: rons.map((tys) => {
      const a = new Float32Array(34);
      for (const ty of tys) a[ty] = value;
      return a;
    }),
  };
}

/** Run the provider with a trace attached and build the record it implies. */
function recordOf(obs: Observation, oracle: Reads, w?: Partial<ComputedWeights>) {
  const ref: ComputedTraceRef = { t: null };
  const reads = computedReads(w, ref)(obs);
  assert(ref.t !== null, "trace was not filled");
  assert(reads !== null);
  return { reads, rec: buildCalibRecord(obs, ref.t, oracle) };
}

// ---------------------------------------------------------------------------
// 1. the split is arithmetic-preserving
// ---------------------------------------------------------------------------

/**
 * The wait model exactly as `computed.ts` expressed it BEFORE the M10a split
 * into `shapeBaseMasses` + `combineShapes`. Reproduced here — the one place in
 * the project where a re-implementation is the point — so that the refactor's
 * only claim ("the arithmetic did not move") is tested rather than asserted.
 */
function legacyShapeWeights(
  ty: number,
  ctx: WaitContext,
  w: ComputedWeights,
): Record<WaitShape, number> {
  const out: Record<WaitShape, number> = {
    "リャンメン": 0,
    "カンチャン": 0,
    "ペンチャン": 0,
    "シャンポン": 0,
    "タンキ": 0,
  };
  if (ctx.genbutsu.has(ty)) return out;
  const u = (t: number) => (t < 0 || t > 33 ? 0 : (ctx.unseen[t] ?? 0));
  const choose2 = (n: number) => (n < 2 ? 0 : (n * (n - 1)) / 2);
  const suitOf = (t: number) => (t >= 27 ? "z" : "mps"[Math.floor(t / 9)]);
  const rankOf = (t: number) => (t % 9) + 1;

  out["シャンポン"] = w.shapePrior["シャンポン"] * (choose2(u(ty)) / 6) *
    (ctx.valueHonors.has(ty) ? w.yakuhaiShanpon : 1);
  out["タンキ"] = w.shapePrior["タンキ"] * (u(ty) / 4);
  if (suitOf(ty) !== "z") {
    const r = rankOf(ty);
    const upper = r <= 6 && !ctx.genbutsu.has(ty + 3) ? u(ty + 1) * u(ty + 2) : 0;
    const lower = r >= 4 && !ctx.genbutsu.has(ty - 3) ? u(ty - 1) * u(ty - 2) : 0;
    out["リャンメン"] = w.shapePrior["リャンメン"] * ((upper + lower) / 32);
    if (r >= 2 && r <= 8) {
      out["カンチャン"] = w.shapePrior["カンチャン"] * (u(ty - 1) * u(ty + 1)) / 16;
    }
    if (r === 3) out["ペンチャン"] = w.shapePrior["ペンチャン"] * (u(ty - 1) * u(ty - 2)) / 16;
    if (r === 7) out["ペンチャン"] = w.shapePrior["ペンチャン"] * (u(ty + 1) * u(ty + 2)) / 16;
  }
  if (ctx.read) {
    const suit = suitOf(ty);
    const flush = ctx.read.honitsuSuit === null
      ? 1
      : (suit === ctx.read.honitsuSuit || suit === "z" ? w.honitsuHot : w.honitsuCold);
    for (const s of WAIT_SHAPES) {
      if (out[s] === 0) continue;
      const pairShape = s === "シャンポン" || s === "タンキ";
      const toitoi = !ctx.read.toitoi ? 1 : (pairShape ? w.toitoiPair : w.toitoiRun);
      out[s] *= flush * toitoi;
    }
  }
  return out;
}

Deno.test("較正: 計算の待ち計算は分割前の式とビット単位で一致する", () => {
  const w = DEFAULT_COMPUTED;
  // A pseudo-random but fixed sweep over pools, genbutsu sets and meld reads:
  // every tile type against every combination of the facts that can move it.
  let checked = 0;
  for (let seed = 0; seed < 24; seed++) {
    const unseen = new Array<number>(34);
    for (let i = 0; i < 34; i++) unseen[i] = (seed * 7 + i * 5) % 5;
    const genbutsu = new Set<number>();
    for (let i = 0; i < 34; i++) if ((seed * 3 + i) % 7 === 0) genbutsu.add(i);
    const valueHonors = new Set([27, 31 + (seed % 3)]);
    // The dora row is new in M10b and the legacy expression knows nothing about
    // it: including it in the sweep is what makes "the arithmetic did not move"
    // cover the new counting too, and not merely the code paths it replaced.
    const dora = new Array<number>(34).fill(0);
    for (let i = 0; i < 34; i++) if ((seed * 5 + i * 3) % 11 === 0) dora[i] = 1;
    const read = {
      honitsuSuit: (seed % 4 === 0 ? null : (["m", "p", "s"][seed % 3])) as
        | "m"
        | "p"
        | "s"
        | null,
      toitoi: seed % 2 === 0,
      yakuhai: new Set<number>(seed % 3 === 0 ? [31] : []),
      open: seed % 3,
    };
    for (
      const ctx of [
        { unseen, genbutsu, valueHonors } as WaitContext,
        { unseen, genbutsu, valueHonors, read } as WaitContext,
        { unseen, genbutsu, valueHonors, read, dora } as WaitContext,
      ]
    ) {
      for (let ty = 0; ty < 34; ty++) {
        const got = waitShapeWeights(ty, ctx, w);
        const want = legacyShapeWeights(ty, ctx, w);
        for (const s of WAIT_SHAPES) {
          assertEquals(got[s], want[s], `ty=${ty} ${s} seed=${seed}`);
        }
        checked++;
      }
    }
  }
  assertEquals(checked, 24 * 3 * 34);
});

Deno.test("較正: 素の枚数はパラメータに依存しない整数で、現物は全形を殺す", () => {
  const ctx: WaitContext = {
    unseen: new Array<number>(34).fill(4),
    genbutsu: new Set<number>([7]), // 八萬 passed
    valueHonors: new Set<number>(),
  };
  const live = shapeBaseMasses(4, ctx); // 五萬
  assertEquals(live.shanpon, 6);
  assertEquals(live.tanki, 4);
  assertEquals(live.kanchan, 16);
  assertEquals(live.penchan, 0, "ペンチャンは3/7の牌にしか届かない");
  // リャンメン loses the upper (67萬) holding to the スジ proof and keeps the
  // lower (34萬) one: half the mass, not all of it. M10b books the refuted half
  // in `ryanmenHalf` instead of discarding it — a count, weighted by nothing.
  assertEquals(live.ryanmen, 16);
  assertEquals([live.ryanmenHalf, live.ryanmenFull, live.ryanmenDora], [16, 0, 0]);
  const dead = shapeBaseMasses(7, ctx); // 現物 itself
  assertEquals(dead, {
    ryanmen: 0,
    ryanmenDora: 0,
    ryanmenHalf: 0,
    ryanmenFull: 0,
    kanchan: 0,
    penchan: 0,
    shanpon: 0,
    tanki: 0,
  });
  // A ペンチャン does exist where it can: 三萬 off 一二萬.
  assertEquals(shapeBaseMasses(2, ctx).penchan, 16);
  // An honor has no run shapes at all, whatever the pool says.
  const honor = shapeBaseMasses(31, ctx);
  assertEquals(honor.ryanmen + honor.kanchan + honor.penchan, 0);
  assertEquals(honor.shanpon, 6);
  // …and the flags a context implies are read straight off it.
  assertEquals(shapeFlagsOf(31, ctx), {
    valueHonor: false,
    honitsuSuit: null,
    toitoi: false,
    doraType: false,
  });
});

// ---------------------------------------------------------------------------
// 2. the cached features are sufficient
// ---------------------------------------------------------------------------

/** Every reconstruction claim, checked against one live provider call. */
function assertReproduces(
  obs: Observation,
  oracle: Reads,
  w?: Partial<ComputedWeights>,
): CalibRecord {
  const { reads, rec } = recordOf(obs, oracle, w);
  const merged = mergeComputed(w);
  const dora = decode34(rec.dr);
  for (let i = 0; i < 3; i++) {
    const row = dealinRowFromRecord(rec, i, merged);
    const live = reads.dealinP![i];
    for (let ty = 0; ty < 34; ty++) {
      assertEquals(row[ty], live[ty], `放銃確率が再現できない (他家${i} 牌種${ty})`);
    }
    assertEquals(digestRow(row), rec.o[i].ph, "ダイジェストが一致しない");
    // The value model, same test: base figure and per-type figure both.
    assertEquals(baseValueFromRecord(rec, i, merged), rec.o[i].vb);
    for (let ty = 0; ty < 34; ty++) {
      if (live[ty] <= 0) continue;
      assertEquals(valueFromRecord(rec, i, ty, merged), reads.dealinValue![i][ty]);
      assertEquals(
        valueFromRecord(rec, i, ty, merged),
        Math.min(merged.valueCap, rec.o[i].vb + merged.valuePerDora * dora[ty]) +
          merged.valuePerHonba * rec.b,
        "本場の割増は頭打ちの上に乗る (score.ts と同じ順序)",
      );
    }
    // …and the tenpai prior, re-derived from the public state rather than read
    // back out of the field the recorder wrote.
    assertEquals(tenpaiFromRecord(rec, i, merged), rec.o[i].tp);
  }
  return rec;
}

Deno.test("較正: 記録した素性だけで放銃確率を完全に再現できる (門前・静かな卓)", () => {
  const rec = assertReproduces(obsOf({ junme: 8 }), fakeOracle([0, 0, 0], [[], [], []]));
  assertEquals(rec.o.length, 3);
  assertEquals(rec.o[0].yc, 2, "副露なし・立直なしはダマ");
  assertEquals(rec.o[0].tr, 0);
  assertEquals(rec.o[0].tc, 1, "8巡目は 7-9 の帯");
  assertEquals(rec.o[0].tt, 0);
  assertEquals(rec.o[0].R, []);
});

Deno.test("較正: 立直・現物・スジのある卓でも再現できる", () => {
  const obs = obsOf({
    junme: 11,
    riichi: [false, true, false, false],
    riichiJunme: [-1, 6, -1, -1],
    rivers: [river("123m"), river("5m9p"), river("1s"), river("南")],
    doraIndicators: tiles("3m"),
  });
  const rec = assertReproduces(obs, fakeOracle([1, 0, 0], [[12, 15], [], []], 7700));
  assertEquals(rec.o[0].yc, 0, "立直は yc=0");
  assertEquals(rec.o[0].tp, 1, "立直は聴牌が真値");
  assertEquals(rec.o[0].tt, 1);
  assertEquals(rec.o[0].R, [12, 15]);
  assertEquals(rec.o[0].V, [7700, 7700]);
  assertEquals(rec.j, 11);
  assertEquals(rec.dr[3], "1", "3萬の指標 ⇒ 4萬がドラ1枚");
});

Deno.test("較正: 副露の内容読み (染め手・トイトイ・役牌) も素性に入る", () => {
  const honitsu = obsOf({
    junme: 9,
    melds: [[], [meldOf("pon", 1, "111m"), meldOf("chi", 1, "456m")], [], []],
  });
  const rec = assertReproduces(honitsu, fakeOracle([0, 0, 0], [[], [], []]));
  assertEquals(rec.o[0].hs, 1, "萬子の染め手模様");
  assertEquals(rec.o[0].om, 2);
  assertEquals(rec.o[0].yc, 1, "副露あり");

  const toitoi = obsOf({
    junme: 9,
    melds: [[], [meldOf("pon", 1, "東東東"), meldOf("pon", 1, "555p")], [], []],
  });
  const rec2 = assertReproduces(toitoi, fakeOracle([0, 0, 0], [[], [], []]));
  assertEquals(rec2.o[0].to, 1, "全部ポン ⇒ トイトイ模様");
  assertEquals(rec2.o[0].yh, 1, "東は場風なので役牌の刻子");
});

Deno.test("較正: 感性ベクトルを変えても、その重みでの予測を閉形式で再現する", () => {
  // The same boards under a DIFFERENT vector: the record does not privilege the
  // weights it was written with, which is what makes an offline refit possible.
  const w: Partial<ComputedWeights> = {
    dealinScale: 0.2,
    honitsuHot: 3,
    tenpaiPrior: [
      [0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5, 0.5],
    ],
    valueOpen: 9999,
  };
  const obs = obsOf({
    junme: 9,
    melds: [[], [meldOf("pon", 1, "111m"), meldOf("chi", 1, "456m")], [], []],
  });
  assertReproduces(obs, fakeOracle([0, 0, 0], [[], [], []]), w);
});

// ---------------------------------------------------------------------------
// 2b. …and sufficient for the M10b model, not merely for the M10a one
// ---------------------------------------------------------------------------

/** Every M10b parameter away from its no-op value, all at once. */
const M10B_ON: Partial<ComputedWeights> = {
  waitNormalize: true,
  expWaitMass: 1.38,
  sujiHalfSurvive: 0.25,
  sujiFullSurvive: 0.1,
  doraPair: 1.5,
  doraBridge: 1.3,
  tenpaiOtherRiichi: 0.6,
  tenpaiMeldDora: 1.15,
  valuePerHonba: 300,
};

Deno.test("較正(v2): 新しい構造をすべて有効にしても記録だけで再現できる", () => {
  // Everything the upgraded model conditions on, on one board: a 本場, a
  // declared riichi at another seat (so the two silent opponents carry the
  // suppression), a 現物 that makes one tile 半スジ and another 全スジ, dora
  // indicators that put a dora on a bridge tile AND on a pair type, and a
  // 副露 with dora in it for the density bump.
  const obs = obsOf({
    junme: 10,
    honba: 3,
    riichi: [false, false, true, false],
    riichiJunme: [-1, -1, 7, -1],
    rivers: [river("2p8p"), river("8p5m"), river("1s"), river("南")],
    doraIndicators: tiles("5p6m"), // ⇒ ⑥筒 と 7萬 がドラ
    melds: [[], [], [], [meldOf("pon", 3, "666p")]],
  });
  const oracle = fakeOracle([0, 0, 1], [[], [], [12, 15]], 8000);
  const rec = assertReproduces(obs, oracle, M10B_ON);
  assertEquals(rec.b, 3);
  // The features the new structure needs are actually in the line…
  assertEquals(rec.o[0].or, 1, "他家 (対面) が立直している");
  assertEquals(rec.o[1].or, 0, "立直した本人から見れば「他家の立直」はない");
  assert(rec.o[0].gh.split("").some((c) => c !== "0"), "半スジの記帳が1つもない");
  assert(rec.o[0].gd.split("").some((c) => c !== "0"), "ドラ橋の記帳が1つもない");
  assertEquals(rec.o[2].md, 3, "副露の中のドラ3枚");
  // …and the same record re-scores correctly under the SHIPPED vector too: a
  // v2 line does not privilege the weights it was written with.
  assertReproduces(obs, oracle);
});

Deno.test("較正(v2): 半スジ/全スジ/ドラ橋は別々の列として記録される", () => {
  const obs = obsOf({
    junme: 9,
    // 下家 passed ⑧筒 (⇒ ⑤筒 は半スジ) and 4萬・1萬 (⇒ 1萬 は現物、4萬も現物、
    // 7萬 は 4萬 で全スジ: 7萬に届く両面は 56萬 だけ).
    rivers: [[], river("8p4m1m"), [], []],
    doraIndicators: tiles("5p"), // ⇒ ⑥筒 がドラ (⑤筒 の上側の橋の中)
  });
  const { rec } = recordOf(obs, fakeOracle([0, 0, 0], [[], [], []]));
  const o = rec.o[0];
  const at = (s: string, ty: number) => parseInt(s[ty], 36);
  assertEquals(at(o.gy, 13), 16, "⑤筒: 生きているのは 34筒 の持ち方だけ");
  assertEquals(at(o.gh, 13), 16, "…殺された 67筒 の持ち方は半スジ列へ");
  assertEquals(at(o.gf, 13), 0);
  assertEquals(at(o.gd, 13), 0, "生きている 34筒 の橋にドラはない");
  assertEquals(at(o.gf, 6), 16, "7萬: 届く両面は 56萬 だけで、それが4萬で死んでいる");
  assertEquals(at(o.gh, 6), 0);
  // ④筒 keeps both holdings, and its upper bridge (⑤⑥筒) holds the dora. The
  // indicator ⑤筒 is itself a visible copy, so that holding counts 3×4 not 4×4.
  assertEquals(at(o.gy, 12), 12 + 16);
  assertEquals(at(o.gd, 12), 12);
});

Deno.test("較正(v2): v1 の記録は素性が足りないので黙って読まず、理由を言う", () => {
  const v1 = JSON.stringify({ v: 1, kind: CALIB_KIND, w: DEFAULT_COMPUTED });
  assertThrows(() => parseCalibration(v1), Error, "v1");
  assertThrows(() => parseCalibration(v1), Error, "M10b");
  assertThrows(() => parseCalibration(v1), Error, "取り直して");
});

Deno.test("較正: 34桁の符号化は全射で、範囲外は静かに丸めず落ちる", () => {
  const a = Array.from({ length: 34 }, (_, i) => i % 33);
  assertEquals(decode34(encode34(a)), a);
  assertEquals(encode34(new Array(34).fill(35)).endsWith("z"), true);
  assertThrows(() => encode34(new Array(34).fill(36)), RangeError);
  assertThrows(() => encode34(new Array(34).fill(-1)), RangeError);
  assertThrows(() => decode34("abc"), RangeError);
});

// ---------------------------------------------------------------------------
// 3. the recorder is invisible
// ---------------------------------------------------------------------------

/** `PairedStats` minus the wall-clock field, which is never reproducible. */
function pinned(st: ReturnType<typeof pairedRun>): string {
  const { ms: _ms, vioA0, vioB0, ...rest } = st;
  return JSON.stringify({
    ...rest,
    vioA0: Object.fromEntries(vioA0),
    vioB0: Object.fromEntries(vioB0),
  });
}

Deno.test("較正: --calibrate をつけても全局ビット単位で同一に進行する", async () => {
  const path = await Deno.makeTempFile({ suffix: ".jsonl" });
  try {
    const plain = pairedRun(2, 4242, "khhh");
    const writer = new CalibrationWriter(path, {
      seats: "khhh",
      seed: 4242,
      games: 2,
      w: DEFAULT_COMPUTED,
    });
    let watched: ReturnType<typeof pairedRun>;
    try {
      watched = pairedRun(2, 4242, "khhh", { calibrate: writer });
    } finally {
      writer.close();
    }
    assertEquals(pinned(watched), pinned(plain), "記録の有無で対局が変わってはならない");

    // …and it did record: the file is a well-formed dataset for those seeds.
    const f = readCalibration(path);
    assertEquals(f.header.kind, CALIB_KIND);
    assertEquals(f.header.v, CALIB_VERSION);
    assertEquals(f.header.seats, "khhh");
    assert(f.records.length > 100, `記録が薄すぎる: ${f.records.length}行`);
    assertEquals(writer.stats().games, 2, "A腕だけを記録する (対照腕は記録しない)");
    assertEquals(writer.stats().rows, f.records.length);
    const seeds = new Set(f.records.map((r) => r.s));
    assertEquals([...seeds].sort(), [4242, 4243]);

    // Every row reproduces, over real boards this time rather than fixtures.
    let bad = 0;
    for (const r of f.records) {
      assertEquals(r.o.length, 3);
      for (let i = 0; i < 3; i++) {
        if (digestRow(dealinRowFromRecord(r, i, f.header.w)) !== r.o[i].ph) bad++;
        assertEquals(baseValueFromRecord(r, i, f.header.w), r.o[i].vb);
      }
    }
    assertEquals(bad, 0, "実戦の記録が閉形式で再現できない");
    // A recorded lane must contain something to calibrate AGAINST.
    assert(f.records.some((r) => r.o.some((o) => o.tt === 1)), "聴牌の真値が1件もない");
    assert(f.records.some((r) => r.o.some((o) => o.R.length > 0)), "ロン牌の真値が1件もない");
    // Decision indices restart per game and are dense.
    const first = f.records.filter((r) => r.s === 4242);
    assertEquals(first.map((r) => r.n), first.map((_, i) => i));
  } finally {
    await Deno.remove(path);
  }
});

Deno.test("較正: 記録は席0の k席のものだけで、対照腕のものは混ざらない", async () => {
  const path = await Deno.makeTempFile({ suffix: ".jsonl" });
  try {
    const writer = new CalibrationWriter(path, {
      seats: "khhh",
      seed: 909,
      games: 1,
      w: DEFAULT_COMPUTED,
    });
    try {
      pairedRun(1, 909, "khhh", { calibrate: writer });
    } finally {
      writer.close();
    }
    const f = readCalibration(path);
    // Seat 0 is 東 in East-1 and its own wind never appears as someone else's:
    // every record is written from ONE seat's point of view, so the observing
    // seat's wind is constant within a kyoku.
    for (const r of f.records.filter((x) => x.k === 0)) assertEquals(r.sw, 27);
    // The B arm is `hhhh` — no "k" seat exists there to record at all.
    assertEquals(new Set(f.records.map((r) => r.s)).size, 1);
  } finally {
    await Deno.remove(path);
  }
});

Deno.test("較正: 版が違う/種別が違うファイルは黙って読まない", () => {
  assertThrows(() => parseCalibration('{"v":99,"kind":"mjgame-calib","w":{}}'), Error, "版");
  assertThrows(() => parseCalibration('{"v":1,"kind":"other","w":{}}'), Error, "較正記録");
  assertThrows(() => parseCalibration(""), Error, "空");
});

// ---------------------------------------------------------------------------
// the report
// ---------------------------------------------------------------------------

Deno.test("較正レポート: 小さな記録から3成分の分解を出す", async () => {
  const path = await Deno.makeTempFile({ suffix: ".jsonl" });
  try {
    const writer = new CalibrationWriter(path, {
      seats: "khhh",
      seed: 4242,
      games: 1,
      w: DEFAULT_COMPUTED,
    });
    try {
      pairedRun(1, 4242, "khhh", { calibrate: writer });
    } finally {
      writer.close();
    }
    const out = await report([path]);
    assert(out.includes("=== 較正レポート (M10b) ==="));
    assert(out.includes("再現不一致 0件"), out.split("\n").slice(0, 4).join("\n"));
    assert(out.includes("打点模型不一致 0件"));
    assert(out.includes("■ 聴牌読み — 卓表セル別"));
    assert(out.includes("■ 聴牌読み — 予測十分位別"));
    assert(out.includes("■ 放銃読み — 予測確率帯別"));
    assert(out.includes("(a) 聴牌 対数損失"));
    assert(out.includes("(b) 待ち 対数損失"));
    assert(out.includes("(c) 打点 平均絶対誤差"));
    assert(out.includes("■ 改善余地の順"));

    // A candidate vector re-scores the SAME boards, and says something else.
    const other = await report([path], {
      weights: mergeComputed({ dealinScale: 0.3, tenpaiFloor: 0 }),
      weightsNote: "候補",
    });
    assert(other.includes("重み 候補"));
    assertNotEquals(other, out, "重みを変えたのに数字が動かない");
  } finally {
    await Deno.remove(path);
  }
});

Deno.test("較正: traceRef を渡し忘れた配線は静かに空ファイルを書かず落ちる", () => {
  const plain = computedReads();
  const reads = calibrationReads(plain, { t: null }, () => ({}), () => {});
  assertThrows(() => reads(obsOf()), Error, "traceRef");
});
