// M14 — the learned deal-in read.
//
// Four claims, tested in this order:
//   1. the FEATURES are public-current-state and nothing else — the 河読み ban,
//      pinned by a river-permutation test rather than by a promise;
//   2. the RECORD reproduces them — a v3 calibration line rebuilds the exact
//      state the seat was served, which is what makes an offline fit honest;
//   3. `learnedReads` is a pure `ReadsProvider` with computed's own key set, its
//      value model (D5) and its tenpai gate;
//   4. the CONSUMER is unchanged — 安全 is still a proof no head may price, and
//      the rule ladder still floors an estimate that says zero.
//
// The wiring tests (a `dealin` block absent ≡ identical play; `--calibrate`
// refused beside a block) belong to the harness pass and are listed in
// `runs/dealin/WIRING.md`.

import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import type { DangerAssessment, DangerLevel } from "mjrender/danger.ts";
import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import type { Reads } from "../src/ai/augmented.ts";
import { AugmentedHeuristic } from "../src/ai/augmented.ts";
import {
  buildCalibRecord,
  CALIB_ACCEPTED,
  CALIB_KIND,
  CALIB_VERSION,
  digestRow,
  parseCalibration,
} from "../src/ai/calibration.ts";
import type { ComputedTrace, ComputedTraceRef } from "../src/ai/computed.ts";
import { computedReads, DEFAULT_COMPUTED, mergeComputed } from "../src/ai/computed.ts";
import type { DealinState, DealinWeights } from "../src/ai/dealin.ts";
import {
  buildDealinHeads,
  closeDealinHeads,
  DEALIN_F,
  DEALIN_FEATURES,
  DEALIN_FV,
  dealinFeatures,
  dealinRecordExtras,
  dealinStateFromRecord,
  dealinStateOf,
  learnedReads,
  mergeDealin,
  TENPAI_F,
  TENPAI_FEATURES,
  tenpaiFeatures,
} from "../src/ai/dealin.ts";
import type { Ctx } from "../src/ai/heuristic.ts";
import type { MlpSpec } from "../src/ai/mlp.ts";
import type { Observation } from "../src/observe.ts";
import { sfc32 } from "../src/rng.ts";
import { AKA_5P } from "../src/tiles.ts";
import { tiles } from "./helpers.ts";

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

/** A river whose entries carry ascending junme, the way a real one does. */
const walk = (spec: string): RiverEntry[] =>
  tiles(spec).map((tile, i) => ({
    tile,
    junme: i + 1,
    tsumogiri: false,
    riichiDeclare: false,
  }));

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

/** A meld in front of the shimocha (Observation index 1). */
const meldOf = (kind: Meld["kind"], spec: string): Meld => {
  const t = tiles(spec);
  return { kind, who: 1, fromWho: kind === "ankan" ? 1 : 0, tiles: t, calledTile: t[0] };
};

/** A board with something to read: melds, discards, dora, a live declaration. */
function busyObs(over: Partial<Observation> = {}): Observation {
  return obsOf({
    junme: 8,
    wallRemaining: 40,
    honba: 1,
    kyotaku: 1,
    kyoku: 3,
    hand: tiles("123m456p789s東東南南"),
    melds: [[], [meldOf("pon", "555p")], [], [meldOf("pon", "南南南")]],
    rivers: [walk("1m9p"), walk("東9m3s"), walk("白5m2p6s"), walk("3p")],
    riichi: [false, false, true, false],
    riichiJunme: [-1, -1, 5, -1],
    doraIndicators: tiles("4s"),
    scores: [31000, 22000, 25000, 22000],
    ...over,
  });
}

/** The trace `computedReads` fills on its way to its own answer. */
function traceOf(obs: Observation): { trace: ComputedTrace; reads: Reads } {
  const ref: ComputedTraceRef = { t: null };
  const reads = computedReads({}, ref)(obs)!;
  assert(ref.t !== null, "trace が埋まっていない");
  return { trace: ref.t, reads };
}

const stateOf = (obs: Observation): DealinState => dealinStateOf(obs, traceOf(obs).trace);

function featuresOf(st: DealinState, opp: number): Float32Array {
  const out = new Float32Array(34 * DEALIN_F);
  dealinFeatures(st, opp, out);
  return out;
}

function tenpaiRow(st: DealinState, opp: number): Float32Array {
  const out = new Float32Array(TENPAI_F);
  tenpaiFeatures(st, opp, out);
  return out;
}

const col = (x: Float32Array, ty: number, name: string): number => {
  const f = DEALIN_FEATURES.indexOf(name);
  assert(f >= 0, `${name} という列はない`);
  return x[ty * DEALIN_F + f];
};

// --- heads -----------------------------------------------------------------

/** A constant head: every input maps to `bias`. ±1000 saturates the sigmoid. */
function constHead(inputs: number, bias: number): MlpSpec {
  return {
    fv: DEALIN_FV,
    layers: [{ in: inputs, out: 1, act: "none", w: new Array(inputs).fill(0), b: [bias] }],
  };
}

/** A small two-layer head with reproducible weights. */
function randHead(inputs: number, seed: number): MlpSpec {
  const r = sfc32(seed);
  const w0 = Array.from({ length: inputs * 4 }, () => r.float() - 0.5);
  const w1 = Array.from({ length: 4 }, () => r.float() - 0.5);
  return {
    fv: DEALIN_FV,
    layers: [
      { in: inputs, out: 4, act: "relu", w: w0, b: [0.1, -0.2, 0.3, 0] },
      { in: 4, out: 1, act: "none", w: w1, b: [-2] },
    ],
  };
}

const weightsOf = (dealin: MlpSpec, tenpai: MlpSpec): DealinWeights =>
  mergeDealin({ fv: DEALIN_FV, dealin, tenpai });

/** The chain the harness builds: computed with a trace, then the heads. */
function chain(w: DealinWeights, cw = mergeComputed({})) {
  const ref: ComputedTraceRef = { t: null };
  const inner = computedReads({}, ref);
  return learnedReads(w, cw, inner, ref);
}

// ---------------------------------------------------------------------------
// 1. the features are public current state
// ---------------------------------------------------------------------------

Deno.test("M14: 特徴量の並びは凍結された契約で、名前は重複しない", () => {
  assertEquals(DEALIN_F, DEALIN_FEATURES.length);
  assertEquals(TENPAI_F, TENPAI_FEATURES.length);
  assertEquals(new Set(DEALIN_FEATURES).size, DEALIN_F, "列名が重複している");
  assertEquals(new Set(TENPAI_FEATURES).size, TENPAI_F);
  // The frozen widths: a change to either is a DEALIN_FV bump and a new lane.
  assertEquals(DEALIN_F, 54);
  assertEquals(TENPAI_F, 22);
  // 河読み: no column names a river position, a tedashi, or a timing.
  for (const n of [...DEALIN_FEATURES, ...TENPAI_FEATURES]) {
    for (const banned of ["tedashi", "tsumogiri", "order", "recent", "last"]) {
      assert(!n.toLowerCase().includes(banned), `${n} は河読みの匂いがする`);
    }
  }
});

Deno.test("M14: 河を並べ替えても特徴量は1ビットも動かない (河読み禁止)", () => {
  // No declaration anywhere: 現物 is then each seat's own discard SET, and the
  // bag is the same multiset however it was thrown. Both the deal-in rows and
  // the tenpai row must be blind to the order.
  const spec = "1m9p5s東3m";
  const shuffled = "東5s3m1m9p";
  const base = obsOf({
    junme: 6,
    wallRemaining: 48,
    hand: tiles("123m456p789s東東"),
    melds: [[], [meldOf("chi", "234p")], [], []],
    rivers: [walk("5m"), walk(spec), walk("2p6p"), walk("9s")],
    doraIndicators: tiles("西"),
  });
  const perm = obsOf({
    ...base,
    rivers: [walk("5m"), walk(shuffled), walk("2p6p"), walk("9s")],
  });
  const a = stateOf(base);
  const b = stateOf(perm);
  for (let i = 0; i < 3; i++) {
    assertEquals(featuresOf(a, i), featuresOf(b, i), `放銃読み ${i} が河の順序で動いた`);
    assertEquals(tenpaiRow(a, i), tenpaiRow(b, i), `聴牌読み ${i} が河の順序で動いた`);
  }
  // …and the permutation really did change the Observation.
  assert(
    base.rivers[1].map((e) => e.tile).join() !== perm.rivers[1].map((e) => e.tile).join(),
    "並べ替えになっていない",
  );
});

Deno.test("M14: 現物・スジ・壁の列は主張どおりのものを数えている", () => {
  // 下家 (relative 1) has thrown 4萬, ②筒 and 8萬. The module books スジ exactly
  // as `shapeBasesFlat` does: over the リャンメン holdings that can REACH the
  // tile — the (ty+1, ty+2) bridge for ranks 1..6 and the (ty−1, ty−2) bridge
  // for ranks 4..9 — 全スジ when every one of them is refuted, 半スジ when some
  // but not all are.
  const obs = obsOf({
    junme: 5,
    hand: tiles("東東東南南南西西西北北白"),
    rivers: [[], walk("4m2p8m"), [], []],
  });
  const st = stateOf(obs);
  const x = featuresOf(st, 0);
  const M1 = 0, M4 = 3, M5 = 4, M7 = 6, P2 = 10;
  assertEquals(col(x, M4, "genbutsu"), 1, "4萬は現物");
  assertEquals(col(x, P2, "genbutsu"), 1, "②筒も現物");
  assertEquals(col(x, M1, "genbutsu"), 0);
  // 1萬: the only リャンメン that reaches it is 23萬 (89萬 is a ペンチャン, not a
  // リャンメン), and 4萬 refutes it ⇒ 全スジ.
  assertEquals(col(x, M1, "sujiFull"), 1);
  assertEquals(col(x, M1, "sujiHalf"), 0);
  // 7萬: same shape from the other end — 56萬 is its only リャンメン, dead ⇒ 全スジ.
  assertEquals(col(x, M7, "sujiFull"), 1);
  assertEquals(col(x, M7, "sujiHalf"), 0);
  // 5萬 has two: 67萬 (refuted by 8萬) and 34萬 (2萬 is still live) ⇒ 半スジ.
  assertEquals(col(x, M5, "sujiHalf"), 1);
  assertEquals(col(x, M5, "sujiFull"), 0);
  // 壁: nothing is exhausted on this board.
  assertEquals(col(x, M1, "kabeUp"), 0);

  // Now exhaust 3萬 (four copies visible) and 1萬 loses its 3萬 bridge.
  const walled = obsOf({
    junme: 5,
    hand: tiles("3333m東東南南西西西"),
    rivers: [[], [], [], []],
  });
  const y = featuresOf(stateOf(walled), 0);
  assertEquals(col(y, M1, "kabeUp"), 1, "1萬の上の橋 (3萬) が枯れている");
  assertEquals(col(y, 4, "kabeDown"), 1, "5萬の下の橋 (3萬) が枯れている");
  assertEquals(col(y, M1, "kabeDown"), 0, "1萬に下の橋はない");
  assertEquals(col(y, 30, "kabeUp"), 0, "字牌に橋はない");
});

Deno.test("M14: 聴牌読みは河を袋としてしか見ない", () => {
  const obs = obsOf({
    junme: 7,
    rivers: [[], walk("東南白1m9p5s"), [], []],
  });
  const t = tenpaiRow(stateOf(obs), 0);
  const at = (name: string) => t[TENPAI_FEATURES.indexOf(name)];
  assertAlmostEquals(at("riverLen/18"), 6 / 18, 1e-7);
  assertAlmostEquals(at("riverHonors/6"), 3 / 6, 1e-7);
  assertAlmostEquals(at("riverTerminals/8"), 2 / 8, 1e-7);
  assertAlmostEquals(at("riverM/6"), 1 / 6, 1e-7);
  assertAlmostEquals(at("riverP/6"), 1 / 6, 1e-7);
  assertAlmostEquals(at("riverS/6"), 1 / 6, 1e-7);
});

// ---------------------------------------------------------------------------
// 2. the record reproduces the state
// ---------------------------------------------------------------------------

/** A minimal oracle answer — `buildCalibRecord` only transcribes it. */
function fakeOracle(): Reads {
  const p = [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
  const v = [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
  p[2][5] = 1;
  v[2][5] = 5800;
  return { tenpaiP: [0, 0, 1], dealinP: p, dealinValue: v };
}

Deno.test("M14: v3 の記録は供された特徴量をそのまま復元する (fh 照合)", () => {
  const obs = busyObs();
  const { trace } = traceOf(obs);
  const extras = dealinRecordExtras(obs, trace);
  const rec = buildCalibRecord(obs, trace, fakeOracle(), extras);

  for (let i = 0; i < 3; i++) {
    assert(rec.o[i].fh !== undefined, "ヘッドを載せた席の記録には fh が要る");
  }
  const back = dealinStateFromRecord(rec);
  const live = dealinStateOf(obs, trace);
  for (let i = 0; i < 3; i++) {
    const a = featuresOf(live, i);
    const b = featuresOf(back, i);
    assertEquals(b, a, `記録から作り直した特徴量が供されたものと違う (相手 ${i})`);
    assertEquals(digestRow(b), rec.o[i].fh, "fh が再現しない");
    assertEquals(tenpaiRow(back, i), tenpaiRow(live, i));
  }
});

Deno.test("M14: ヘッドなしで録った v3 レーンも特徴量を完全に復元できる", () => {
  // The FIRST lane must be recorded on the plain computed champion — a head
  // cannot be trained on its own outputs — so the model-free half of the record
  // has to be enough on its own. `fh` is what it loses, and only that.
  const obs = busyObs({ junme: 12, wallRemaining: 20, honba: 3 });
  const { trace } = traceOf(obs);
  const rec = buildCalibRecord(obs, trace, fakeOracle());
  assertEquals(rec.o[0].fh, undefined, "ヘッドがないのに fh がある");
  assert(rec.un !== undefined && rec.oh !== undefined && rec.sc !== undefined);
  assert(rec.o[0].gb !== undefined && rec.o[0].rb !== undefined);

  const back = dealinStateFromRecord(rec);
  const live = dealinStateOf(obs, trace);
  for (let i = 0; i < 3; i++) {
    assertEquals(featuresOf(back, i), featuresOf(live, i));
    assertEquals(tenpaiRow(back, i), tenpaiRow(live, i));
  }
});

Deno.test("M14: v2 の記録には無い列があるので、名指しで断る", () => {
  const obs = busyObs();
  const { trace } = traceOf(obs);
  const rec = buildCalibRecord(obs, trace, fakeOracle());
  const v2 = { ...rec, un: undefined, oh: undefined };
  assertThrows(() => dealinStateFromRecord(v2), Error, "un");
  const noGb = { ...rec, o: rec.o.map((o) => ({ ...o, gb: undefined })) };
  assertThrows(() => dealinStateFromRecord(noGb), Error, "gb");
});

Deno.test("較正: v3 を書き、v2 も読み、v1 だけ断る", () => {
  assertEquals(CALIB_VERSION, 3);
  assertEquals([...CALIB_ACCEPTED].sort(), [2, 3]);
  const head = (v: number) => JSON.stringify({ v, kind: CALIB_KIND, w: DEFAULT_COMPUTED });
  // v2 lanes are hundreds of megabytes and `calibrate_fit`/`calibrate_report`
  // still re-score them: v3 is a SUPERSET, so both are read.
  assertEquals(parseCalibration(head(2)).header.v, 2);
  assertEquals(parseCalibration(head(3)).header.v, 3);
  assertThrows(() => parseCalibration(head(1)), Error, "v1");
  assertThrows(() => parseCalibration(head(1)), Error, "M10b");
  assertThrows(() => parseCalibration(head(4)), Error, "v2/v3");
});

// ---------------------------------------------------------------------------
// 3. learnedReads
// ---------------------------------------------------------------------------

Deno.test("M14: learnedReads は純粋で、計算の鍵集合をそのまま返す", () => {
  const w = weightsOf(randHead(DEALIN_F, 7), randHead(TENPAI_F, 9));
  const p = chain(w);
  try {
    const a = p(busyObs())!;
    const b = p(busyObs())!;
    assertEquals(a, b, "同じ観測 ⇒ 同じ読み");
    const computed = computedReads({})(busyObs())!;
    assertEquals(Object.keys(a).sort(), Object.keys(computed).sort());
    // The escaping rows are fresh per call — a consumer may hold one.
    assert(a.dealinP![0] !== b.dealinP![0], "行を使い回すと読みが後から変わる");
  } finally {
    p.close();
  }
});

Deno.test("M14: 放銃価値は34種すべてに入る (D5)", () => {
  const w = weightsOf(randHead(DEALIN_F, 11), randHead(TENPAI_F, 13));
  const p = chain(w);
  try {
    const obs = busyObs();
    const learned = p(obs)!;
    const computed = computedReads({})(obs)!;
    let computedZeros = 0;
    for (let i = 0; i < 3; i++) {
      for (let ty = 0; ty < 34; ty++) {
        assert(
          learned.dealinValue![i][ty] > 0,
          `相手 ${i} の ${ty} に価値がない — riskOf の ?? expLoss は救ってくれない`,
        );
        if (computed.dealinValue![i][ty] === 0) computedZeros++;
      }
    }
    assert(computedZeros > 0, "計算側に 0 が無い盤面では D5 を測れない");
  } finally {
    p.close();
  }
});

Deno.test("M14: 聴牌ヘッドは tenpaiFloor で門を通る", () => {
  // A constant head at p ≈ 0.119, which is under the shipped floor of 0.25.
  const logit = -2;
  const p = 1 / (1 + Math.exp(-logit));
  const w = weightsOf(constHead(DEALIN_F, -1000), constHead(TENPAI_F, logit));
  const obs = busyObs({ riichi: [false, false, false, false], riichiJunme: [-1, -1, -1, -1] });

  const gated = chain(w, mergeComputed({}));
  try {
    assertEquals(gated(obs)!.tenpaiP, [0, 0, 0], "床下の読みは「何も言わない」");
  } finally {
    gated.close();
  }

  const open = chain(w, mergeComputed({ tenpaiFloor: 0 }));
  try {
    const t = open(obs)!.tenpaiP!;
    for (let i = 0; i < 3; i++) assertAlmostEquals(t[i], p, 1e-6);
  } finally {
    open.close();
  }
});

Deno.test("M14: 宣言された立直はヘッドに聞かない — 規則で聴牌である", () => {
  const w = weightsOf(constHead(DEALIN_F, -1000), constHead(TENPAI_F, -1000));
  const p = chain(w);
  try {
    // seat 2 (relative index 1 of the Reads) has declared.
    const reads = p(busyObs())!;
    assertEquals(reads.tenpaiP![1], 1, "立直はヘッドの答えより強い");
    assertEquals(reads.tenpaiP![0], 0);
  } finally {
    p.close();
  }
});

Deno.test("M14: 飽和したヘッドは 0 と 1 をきっちり出す", () => {
  const hot = chain(weightsOf(constHead(DEALIN_F, 1000), constHead(TENPAI_F, 1000)));
  const cold = chain(weightsOf(constHead(DEALIN_F, -1000), constHead(TENPAI_F, -1000)));
  try {
    const h = hot(busyObs())!;
    const c = cold(busyObs())!;
    for (let ty = 0; ty < 34; ty++) {
      assertEquals(h.dealinP![0][ty], 1);
      assertEquals(c.dealinP![0][ty], 0);
    }
  } finally {
    hot.close();
    cold.close();
  }
});

Deno.test("M14: 席が持つヘッドは局ごとに作り直されない (借り物は close で解放しない)", () => {
  // `withReads` rebuilds the provider chain on every reset, so the harness
  // builds the heads ONCE and hands them in; a provider that borrowed them must
  // not free them when its own chain is dropped.
  const w = weightsOf(constHead(DEALIN_F, 1000), constHead(TENPAI_F, 1000));
  const heads = buildDealinHeads(w);
  try {
    const ref: ComputedTraceRef = { t: null };
    const first = learnedReads(heads, mergeComputed({}), computedReads({}, ref), ref);
    assertEquals(first(busyObs())!.dealinP![0][0], 1);
    first.close(); // the hanchan ended — the heads are the SEAT's, not the chain's
    const ref2: ComputedTraceRef = { t: null };
    const second = learnedReads(heads, mergeComputed({}), computedReads({}, ref2), ref2);
    assertEquals(second(busyObs())!.dealinP![0][0], 1, "借りたヘッドが解放されている");
    second.close();
  } finally {
    closeDealinHeads(heads);
  }
});

Deno.test("M14: mergeDealin は空も古い版も形違いも断る", () => {
  assertThrows(() => mergeDealin(), Error, "重みが要ります");
  assertThrows(() => mergeDealin({}), Error, "重みが要ります");
  assertThrows(
    () => mergeDealin({ fv: DEALIN_FV, dealin: constHead(DEALIN_F, 0) } as Partial<DealinWeights>),
    Error,
    "重みが要ります",
  );
  assertThrows(
    () => mergeDealin({ fv: 99, dealin: constHead(DEALIN_F, 0), tenpai: constHead(TENPAI_F, 0) }),
    Error,
    "特徴量版",
  );
  assertThrows(
    () =>
      mergeDealin({
        fv: DEALIN_FV,
        dealin: constHead(DEALIN_F - 1, 0),
        tenpai: constHead(TENPAI_F, 0),
      }),
    Error,
    "入力次元",
  );
});

// ---------------------------------------------------------------------------
// 4. the consumer is unchanged
// ---------------------------------------------------------------------------

/** Reaches the protected risk hook the way `scoreDiscard` does. */
class Probe extends AugmentedHeuristic {
  riskWith(ctx: Ctx, tile: Tile, reads: Reads | null): number {
    this.reads = reads;
    try {
      return this.riskOf(ctx, tile);
    } finally {
      this.reads = null;
    }
  }
}

function danger(level: DangerLevel, seat = 1): DangerAssessment {
  return { level, seats: [seat], details: [{ seat, level, kind: "riichi", notes: [] }] };
}

function riskCtx(map: Map<number, DangerAssessment>): Ctx {
  const obs = { danger: map, riichi: [false, true, false, false] } as unknown as Observation;
  return {
    obs,
    open: 0,
    closed: true,
    doraTypes: new Set(),
    valueHonors: new Set(),
    unseen: new Array<number>(34).fill(4),
    folding: false,
    canRiichi: false,
    eff: 1,
    def: 1,
  };
}

/** A head that is certain about everything — the worst case for the proof. */
function certain(value: number): Reads {
  const p = [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
  const v = [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
  for (let i = 0; i < 3; i++) {
    p[i].fill(1);
    v[i].fill(value);
  }
  return { dealinP: p, dealinValue: v };
}

Deno.test("M14: 安全は証明 — p=1 のヘッドでも現物は無料のまま", () => {
  const safe = 5; // 6萬, genbutsu
  const hot = 12; // ④筒, 無スジ
  const probe = new Probe("probe", 1, () => null);
  const ctx = riskCtx(new Map([[safe, danger("安全")], [hot, danger("危険度高")]]));
  const reads = certain(12000);
  // The 感性 surcharges are zero without a `sense` vector, so the proof's price
  // is exactly zero — and a learned head, however confident, may not raise it.
  assertEquals(probe.riskWith(ctx, safe * 4, reads), 0);
  // The same certainty on a live tile does get through, scaled by λ.
  assertEquals(probe.riskWith(ctx, hot * 4, reads), 0.25 * 3 * 12000);
});

Deno.test("M14: 梯子は残る — floor>0 なら 0 を言うヘッドの下でも値がつく", () => {
  const low = 12; // ④筒 at 危険度低 = 30 on the rule ladder
  const probe = new Probe("probe", 1, () => null);
  const ctx = riskCtx(new Map([[low, danger("危険度低")]]));
  assertEquals(probe.riskWith(ctx, low * 4, null), 30, "素の梯子");

  // A head that says "nobody can ron this" — exactly what an M14 head trained
  // to 0 on a quiet tile produces. The rule floor still charges half.
  const zero: Reads = {
    dealinP: [new Float32Array(34), new Float32Array(34), new Float32Array(34)],
    dealinValue: [new Float32Array(34), new Float32Array(34), new Float32Array(34)],
  };
  assertEquals(probe.riskWith(ctx, low * 4, zero), 0.5 * 30);

  // …and with the floor knob at 0 (the owner's edit once the head is graded)
  // only the 安全 proof is left, so the price is the head's own zero.
  const noFloor = new Probe("probe", 1, () => null, { augment: { floor: 0 } });
  assertEquals(noFloor.riskWith(ctx, low * 4, zero), 0);
  assertEquals(noFloor.riskWith(ctx, low * 4, null), 30, "読みが無ければ梯子そのもの");
});
