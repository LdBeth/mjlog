// M9a — the evidence vector and its monotone consumer.
//
// Four claims, one group each:
//   1. the curve is what it says it is — piecewise linear between its knots,
//      clamped or linearly extended outside them, and NONDECREASING under any θ
//      whatsoever (the property that makes fitting it safe);
//   2. a curve set survives the round trip to JSON, and a broken file is
//      refused rather than half-loaded;
//   3. the evidence vector names the right numbers, and an augmented policy's
//      overrides reach it — the same assembly, a different `risk`;
//   4. INIT-EQUIVALENCE: `initFromWeights` reproduces the hand-written discard
//      score EXACTLY — same float, same argmax, same hanchan — for the base
//      policy and for the augmented one. This is M9a's acceptance criterion,
//      and everything else in the milestone rests on it.

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { DangerAssessment, DangerLevel } from "mjrender/danger.ts";
import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import { AugmentedHeuristic } from "../src/ai/augmented.ts";
import type { Reads } from "../src/ai/augmented.ts";
import {
  ALL_SPECS,
  constTheta,
  curvePoints,
  curveValues,
  evalCurve,
  initFromWeights,
  linearTheta,
  parseConsumerParams,
  scoreDiscard,
  serializeConsumer,
  SPEC_BY_KEY,
} from "../src/ai/consumer.ts";
import type { ConsumerParams, CurveParams, CurveSpec } from "../src/ai/consumer.ts";
import { assembleCandidate, assembleContext } from "../src/ai/evidence.ts";
import type { EvidenceHooks, EvidenceVector } from "../src/ai/evidence.ts";
import { DEFAULT_WEIGHTS, HeuristicPolicy } from "../src/ai/heuristic.ts";
import type { Ctx, HeuristicWeights } from "../src/ai/heuristic.ts";
import type { Observation } from "../src/observe.ts";
import { sfc32 } from "../src/rng.ts";
import { JANKI } from "../src/rules.ts";
import type { Action } from "../src/types.ts";
import { playHanchan, tiles } from "./helpers.ts";

// ---------------------------------------------------------------------------
// fixtures (the pattern comes from heuristic_test.ts: hand-built, one claim
// each, nothing shared with a live table)
// ---------------------------------------------------------------------------

function discardsOf(hand: Tile[], drawn: Tile | null): Action[] {
  return hand.map((tile) => ({
    t: "discard",
    tile,
    riichi: false,
    tsumogiri: drawn !== null && tile === drawn,
  }));
}

function openingRivers(): RiverEntry[][] {
  const pool = tiles("北北北北西西西西");
  return [0, 1, 2, 3].map((s) =>
    [pool[s], pool[4 + s]].map((tile, i): RiverEntry => ({
      tile,
      junme: i + 1,
      tsumogiri: false,
      riichiDeclare: false,
    }))
  );
}

function baseObs(over: Partial<Observation> = {}): Observation {
  const hand = over.hand ?? tiles("123456789m1122p東");
  const drawn = "drawn" in over ? over.drawn! : hand[hand.length - 1];
  const obs: Observation = {
    seat: 0,
    kyoku: 0,
    honba: 0,
    kyotaku: 0,
    junme: 3,
    wallRemaining: 58,
    hand,
    drawn,
    melds: [[], [], [], []],
    rivers: openingRivers(),
    scores: [25000, 25000, 25000, 25000],
    riichi: [false, false, false, false],
    riichiJunme: [-1, -1, -1, -1],
    doraIndicators: tiles("9s"),
    seatWind: 27,
    roundWind: 27,
    akaIds: JANKI.akaIds,
    shanten: 0,
    waits: [],
    ronnable: [],
    katagari: false,
    discardInfo: new Map(),
    tsumogiriLock: false,
    ukeire: [],
    doraCount: 0,
    furiten: { permanent: false, temporary: false, riichi: false },
    danger: new Map(),
    violations: [0, 0, 0, 0],
    legal: [],
    ...over,
  };
  if (!over.legal) obs.legal = discardsOf(obs.hand, obs.drawn);
  return obs;
}

function threat(level: DangerLevel, seat = 1): DangerAssessment {
  return { level, seats: [seat], details: [{ seat, level, kind: "riichi", notes: [] }] };
}

/** A danger map over a whole hand: every type in `spec` at `level`. */
function dangerMap(spec: [DangerLevel, string][]): Map<number, DangerAssessment> {
  const m = new Map<number, DangerAssessment>();
  for (const [level, s] of spec) {
    for (const t of tiles(s)) m.set(Math.floor(t / 4), threat(level));
  }
  return m;
}

const PON_TON: Meld = {
  kind: "pon",
  who: 0,
  fromWho: 1,
  tiles: tiles("東東東"),
  calledTile: tiles("東")[0],
};

/**
 * The situations the equivalence claim is tested over. Deliberately spread
 * across the regimes the score's own branches care about: pushing and folding
 * (which is what `ctx.eff`/`ctx.def` switch), open and closed, with and without
 * a danger map, dora held and dora cut, an armed ツモ切り lock, and a hand far
 * enough out for 不聴時ドラ切り to price.
 */
function situations(): { name: string; obs: Observation }[] {
  const dora9s = tiles("9s"); // ⇒ dora is 1s
  const out: { name: string; obs: Observation }[] = [];

  out.push({ name: "門前・平場・完成形", obs: baseObs() });

  {
    const hand = tiles("19m19p19s東南西北白發中1m");
    out.push({
      name: "国士模様・字牌だらけ",
      obs: baseObs({ hand, drawn: hand[hand.length - 1], shanten: 4, junme: 8 }),
    });
  }

  {
    const hand = tiles("1234s567m11p發發西2m");
    out.push({
      name: "ベタ降り (リーチ1件・遠い手)",
      obs: baseObs({
        hand,
        drawn: hand[hand.length - 1],
        shanten: 3,
        junme: 9,
        riichi: [false, true, false, false],
        riichiJunme: [-1, 6, -1, -1],
        danger: dangerMap([["安全", "1s"], ["危険度低", "2s"], ["危険度中", "3s4s"], [
          "危険度高",
          "5m6m7m",
        ]]),
        doraIndicators: dora9s,
        doraCount: 1,
      }),
    });
  }

  {
    const hand = tiles("123456m11p123s0p");
    out.push({
      name: "赤ドラ入り聴牌・危険度あり",
      obs: baseObs({
        hand,
        drawn: hand[hand.length - 1],
        shanten: 0,
        junme: 11,
        riichi: [false, false, true, false],
        danger: dangerMap([["危険度高", "1p"], ["危険度中", "1m2m3m"], ["安全", "1s"]]),
        doraCount: 1,
      }),
    });
  }

  {
    const hand = tiles("456m789p12s白白2s");
    out.push({
      name: "副露あり (ポン東) ・中盤",
      obs: baseObs({
        hand,
        drawn: hand[hand.length - 1],
        melds: [[PON_TON], [], [], []],
        shanten: 1,
        junme: 7,
      }),
    });
  }

  {
    const hand = tiles("1s1s2334m567p999s1z".replace("1z", "東"));
    out.push({
      name: "不聴時ドラ切り圏 (3向聴以上)",
      obs: baseObs({
        hand,
        drawn: hand[hand.length - 1],
        shanten: 3,
        junme: 4,
        doraIndicators: dora9s,
        doraCount: 2,
      }),
    });
  }

  {
    const hand = tiles("123m456p789s11z22z".replace("11z", "東東").replace("22z", "南南"));
    out.push({
      name: "ツモ切り縛り中",
      obs: baseObs({
        hand,
        drawn: hand[0],
        shanten: 0,
        junme: 12,
        tsumogiriLock: true,
      }),
    });
  }

  {
    const hand = tiles("111m999p11s東東南南西");
    out.push({
      name: "字牌過多・終盤・オーラス点差あり",
      obs: baseObs({
        hand,
        drawn: hand[hand.length - 1],
        shanten: 2,
        junme: 14,
        kyoku: 7,
        scores: [13000, 38000, 36000, 33000],
        wallRemaining: 8,
        riichi: [false, false, false, true],
        danger: dangerMap([["危険度高", "西南"], ["危険度低", "1m9p"]]),
      }),
    });
  }

  return out;
}

// ---------------------------------------------------------------------------
// probes — the equivalence test is white-box on purpose: comparing DECISIONS
// would pass on a consumer that is merely close, and the claim is exactness.
// ---------------------------------------------------------------------------

interface Innards {
  context(obs: Observation): Ctx;
  shantenWithout(ctx: Ctx, tile: Tile): number;
  scoreDiscard(ctx: Ctx, tile: Tile, sh: number, wide: boolean, run: unknown): number;
  evidenceHooks(): EvidenceHooks;
  consumer: ConsumerParams | null;
}

/** Every legal discard's score, computed the way `chooseDiscard` computes it. */
function scoreMap(p: HeuristicPolicy, obs: Observation): Map<Tile, number> {
  const inner = p as unknown as Innards;
  const ctx = inner.context(obs);
  const candidates = [
    ...new Set(
      obs.legal.filter((a) => a.t === "discard").map((a) => (a as { tile: Tile }).tile),
    ),
  ];
  const shantenAfter = new Map<Tile, number>();
  let best = Infinity;
  for (const tile of candidates) {
    const s = inner.shantenWithout(ctx, tile);
    shantenAfter.set(tile, s);
    if (s < best) best = s;
  }
  let run: unknown = null;
  if (inner.consumer) {
    const hooks = inner.evidenceHooks();
    run = { hooks, context: assembleContext(hooks, ctx) };
  }
  const out = new Map<Tile, number>();
  for (const tile of candidates) {
    const sh = shantenAfter.get(tile)!;
    out.set(tile, inner.scoreDiscard(ctx, tile, sh, sh === best, run));
  }
  return out;
}

/** The evidence one candidate produces, through the instance's own hooks. */
function evidenceOf(p: HeuristicPolicy, obs: Observation, tile: Tile): EvidenceVector {
  const inner = p as unknown as Innards;
  const ctx = inner.context(obs);
  const hooks = inner.evidenceHooks();
  const sh = inner.shantenWithout(ctx, tile);
  return {
    context: assembleContext(hooks, ctx),
    candidate: assembleCandidate(hooks, ctx, tile, sh, false),
  };
}

/** Bit-for-bit, with −0 and +0 counted equal (no comparison can tell them apart). */
function assertExact(a: number, b: number, msg: string): void {
  assert(a === b, `${msg}: ${a} ≠ ${b} (差 ${a - b})`);
}

class AugProbe extends AugmentedHeuristic {
  setReads(r: Reads | null): void {
    this.reads = r;
  }
}

const INIT = initFromWeights({ ...DEFAULT_WEIGHTS, danger: { ...DEFAULT_WEIGHTS.danger } });

// ---------------------------------------------------------------------------
// 1. the curve
// ---------------------------------------------------------------------------

const SPEC: CurveSpec = {
  key: "risk",
  field: "risk",
  knots: [0, 10, 20, 40],
  sign: 1,
  mode: "linear",
  about: "test",
};
const CLAMPED: CurveSpec = { ...SPEC, mode: "clamp" };

Deno.test("consumer: 曲線の節点値は θ の絶対値の累積", () => {
  assertEquals(curveValues([5, -3, 2, -1]), [5, 8, 10, 11]);
  assertEquals(curveValues([0, 0, 0, 0]), [0, 0, 0, 0]);
});

Deno.test("consumer: 節点上では節点値そのもの", () => {
  const th: CurveParams = [1, 2, 3, 4];
  const y = curveValues(th);
  for (let i = 0; i < 4; i++) {
    assertExact(evalCurve(SPEC, th, SPEC.knots[i]), y[i], `節点${i}`);
  }
});

Deno.test("consumer: 節点間は線形補間", () => {
  const th: CurveParams = [0, 10, 10, 20]; // y = 0,10,20,40 over x = 0,10,20,40
  assertExact(evalCurve(SPEC, th, 5), 5, "前半");
  assertExact(evalCurve(SPEC, th, 15), 15, "中盤");
  assertExact(evalCurve(SPEC, th, 30), 30, "後半");
});

Deno.test("consumer: clamp は端の値で止まり linear は端の傾きで伸びる", () => {
  const th: CurveParams = [0, 10, 10, 20];
  assertExact(evalCurve(CLAMPED, th, -100), 0, "下側 clamp");
  assertExact(evalCurve(CLAMPED, th, 1000), 40, "上側 clamp");
  assertExact(evalCurve(SPEC, th, -5), -5, "下側 linear");
  assertExact(evalCurve(SPEC, th, 100), 100, "上側 linear");
});

Deno.test("consumer: どんな θ でも単調非減少 (構造的保証)", () => {
  const rng = sfc32(0x9E3779B9);
  for (let trial = 0; trial < 500; trial++) {
    const th: CurveParams = [
      (rng.float() - 0.5) * 2000,
      (rng.float() - 0.5) * 2000,
      (rng.float() - 0.5) * 2000,
      (rng.float() - 0.5) * 2000,
    ];
    const spec = trial % 2 === 0 ? SPEC : CLAMPED;
    let prev = -Infinity;
    for (let x = -60; x <= 100; x += 0.5) {
      const y = evalCurve(spec, th, x);
      assert(y >= prev, `θ=${th} x=${x} で減少した (${y} < ${prev})`);
      prev = y;
    }
  }
});

Deno.test("consumer: linearTheta は直線を、constTheta は定数を張る", () => {
  const th = linearTheta(SPEC.knots, 3);
  for (const x of [-7, 0, 4, 10, 25, 40, 90]) {
    assertExact(evalCurve(SPEC, th, x), 3 * x, `x=${x}`);
  }
  const c = constTheta(1);
  for (const x of [-7, 0, 4, 40, 90]) assertExact(evalCurve(SPEC, c, x), 1, `定数 x=${x}`);
});

Deno.test("consumer: 減少する曲線は作れない — 負の傾きは拒否", () => {
  assertThrows(() => linearTheta(SPEC.knots, -1), RangeError);
  assertThrows(() => linearTheta(SPEC.knots, NaN), RangeError);
});

Deno.test("consumer: 曲線は17本・パラメータは68個", () => {
  assertEquals(ALL_SPECS.length, 17);
  assertEquals(ALL_SPECS.length * 4, 68);
  assertEquals(new Set(ALL_SPECS.map((s) => s.key)).size, 17, "鍵は一意");
  assertEquals(SPEC_BY_KEY.size, 17);
});

// ---------------------------------------------------------------------------
// 2. serialization
// ---------------------------------------------------------------------------

Deno.test("consumer: JSON 往復で θ が一致する", () => {
  const back = parseConsumerParams(JSON.parse(serializeConsumer(INIT)));
  for (const spec of ALL_SPECS) {
    assertEquals(back.curves[spec.key], INIT.curves[spec.key], spec.key);
  }
});

Deno.test("consumer: 素の θ 配列だけの形式も読める", () => {
  const bare: Record<string, CurveParams> = {};
  for (const spec of ALL_SPECS) bare[spec.key] = INIT.curves[spec.key];
  const back = parseConsumerParams({ version: 1, curves: bare });
  assertEquals(back.curves.shanten, INIT.curves.shanten);
});

Deno.test("consumer: 壊れた入力は読み込まない", () => {
  const good = JSON.parse(serializeConsumer(INIT));
  assertThrows(() => parseConsumerParams(null), Error);
  assertThrows(() => parseConsumerParams([1, 2, 3]), Error);
  assertThrows(() => parseConsumerParams({ curves: good.curves }), Error, "version");
  assertThrows(() => parseConsumerParams({ version: 2, curves: good.curves }), Error, "version");
  assertThrows(() => parseConsumerParams({ version: 1 }), Error, "curves");

  const missing = JSON.parse(serializeConsumer(INIT));
  delete missing.curves.dora;
  assertThrows(() => parseConsumerParams(missing), Error, "dora");

  const short = JSON.parse(serializeConsumer(INIT));
  short.curves.risk.theta = [1, 2, 3];
  assertThrows(() => parseConsumerParams(short), Error, "risk");

  const nan = JSON.parse(serializeConsumer(INIT));
  nan.curves.ukeire.theta = [0, "x", 0, 0];
  assertThrows(() => parseConsumerParams(nan), Error, "ukeire");
});

Deno.test("consumer: 曲線は図に起こせる (節点と値の対)", () => {
  const pts = curvePoints("shanten", INIT);
  assertEquals(pts.map((p) => p.x), [-1, 1, 3, 6]);
  // sign −1: 向聴が増えるほど下がる。
  assertEquals(pts.map((p) => p.y), [1000, -1000, -3000, -6000]);
});

// ---------------------------------------------------------------------------
// 3. the evidence vector
// ---------------------------------------------------------------------------

Deno.test("evidence: 名前つきの値が観測と一致する", () => {
  const hand = tiles("123456789m1122p東");
  const obs = baseObs({ hand, junme: 5 });
  const p = new HeuristicPolicy("cpu", 1);
  const ton = hand[hand.length - 1];
  const ev = evidenceOf(p, obs, ton);

  assertEquals(ev.candidate.shantenAfter, 0, "東を切れば聴牌");
  assertEquals(ev.candidate.doraKept, 0, "ドラは 1s、手にない");
  assertEquals(ev.candidate.yakuhaiPairs, 0);
  assertEquals(ev.candidate.isolatedHonorLate, 0, "東を切れば孤立字牌は残らない");
  assertEquals(ev.candidate.risk, 0, "危険度の記載なし ⇒ 安全扱い (基本方策)");
  assertEquals(ev.candidate.dangerLevel, 0);
  assertEquals(ev.candidate.explicitSafe, 0, "明示された現物ではない");
  assertEquals(ev.candidate.tsumogiri, 1, "ツモ牌そのもの");
  assertEquals(ev.context.eff, 1, "押している");
  assertEquals(ev.context.def, 1);
  assertEquals(ev.context.folding, 0);
  assertEquals(ev.context.junme, 5);
  assertEquals(ev.context.standingsGain, 1, "順位効用は既定で切れている");
  assertEquals(ev.context.standingsRisk, 1);

  // The 1m: cutting it costs a shanten step and strands the honor.
  const evm = evidenceOf(p, obs, hand[0]);
  assertEquals(evm.candidate.shantenAfter, 1);
  assertEquals(evm.candidate.isolatedHonorLate, 5, "孤立東 × min(巡目5,12)");
});

Deno.test("evidence: 危険度と現物が段位と証明の別として入る", () => {
  const hand = tiles("123456789m1122p東");
  const obs = baseObs({
    hand,
    riichi: [false, true, false, false],
    danger: dangerMap([["危険度高", "東"], ["安全", "1m"]]),
  });
  const p = new HeuristicPolicy("cpu", 1);
  const ton = evidenceOf(p, obs, hand[hand.length - 1]);
  assertEquals(ton.candidate.dangerLevel, 3);
  assertEquals(ton.candidate.explicitSafe, 0);
  assertEquals(ton.candidate.risk, DEFAULT_WEIGHTS.danger["危険度高"]);
  const man = evidenceOf(p, obs, hand[0]);
  assertEquals(man.candidate.dangerLevel, 0);
  assertEquals(man.candidate.explicitSafe, 1, "明示された安全");
  assertEquals(man.candidate.risk, 0);
});

Deno.test("evidence: 増補方策の上書きが自動的に流れ込む", () => {
  const hand = tiles("123456789m1122p東");
  const dealinP = [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
  const dealinValue = [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
  dealinP[0][27] = 1; // 下家が東でロンできる
  dealinValue[0][27] = 8000;
  const reads: Reads = { dealinP, dealinValue, tenpaiP: [1, 0, 0], expLoss: [8000, 0, 0] };
  const obs = baseObs({
    hand,
    riichi: [false, true, false, false],
    danger: dangerMap([["危険度中", "東"]]),
  });

  const plain = new HeuristicPolicy("h", 1);
  const aug = new AugProbe("k", 1, () => null);
  aug.setReads(reads);
  const ton = hand[hand.length - 1];
  const base = evidenceOf(plain, obs, ton);
  const rich = evidenceOf(aug, obs, ton);

  assertEquals(base.candidate.risk, DEFAULT_WEIGHTS.danger["危険度中"], "規則の梯子");
  assertEquals(rich.candidate.risk, 0.25 * 8000, "λ × P×失点 が梯子を上回る");
  assert(rich.context.pressure > 0, "聴牌真値から圧が立つ");
  assertEquals(base.candidate.dangerLevel, rich.candidate.dangerLevel, "証拠の段位は同じ");
});

// ---------------------------------------------------------------------------
// 4. init-equivalence — M9a's acceptance criterion
// ---------------------------------------------------------------------------

Deno.test("consumer: 初期値は手書き評価とビット単位で一致する (基本方策)", () => {
  for (const { name, obs } of situations()) {
    const plain = new HeuristicPolicy("h", 7);
    const fitted = new HeuristicPolicy("h", 7, { consumer: INIT });
    const a = scoreMap(plain, obs);
    const b = scoreMap(fitted, obs);
    assertEquals([...b.keys()], [...a.keys()], `${name}: 候補集合`);
    for (const [tile, x] of a) assertExact(b.get(tile)!, x, `${name}: 牌 ${tile}`);
    // ...and therefore the same choice.
    assertEquals(plain.decide(obs), fitted.decide(obs), `${name}: 打牌`);
  }
});

Deno.test("consumer: 初期値は手書き評価と一致する (増補方策・オラクル読みあり)", () => {
  const dealinP = [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
  const dealinValue = [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
  for (let ty = 0; ty < 34; ty += 3) {
    dealinP[0][ty] = 0.4;
    dealinValue[0][ty] = 5200;
    dealinP[1][ty] = 0.1;
    dealinValue[1][ty] = 12000;
  }
  const reads: Reads = {
    dealinP,
    dealinValue,
    tenpaiP: [1, 0.5, 0],
    expLoss: [5200, 12000, 0],
    nextDora: 4,
    ownNextDraw: 8,
  };
  for (const { name, obs } of situations()) {
    const plain = new AugProbe("k", 7, () => null);
    const fitted = new AugProbe("k", 7, () => null, { consumer: INIT });
    plain.setReads(reads);
    fitted.setReads(reads);
    const a = scoreMap(plain, obs);
    const b = scoreMap(fitted, obs);
    for (const [tile, x] of a) assertExact(b.get(tile)!, x, `${name}: 牌 ${tile}`);
  }
});

Deno.test("consumer: 初期値は --ktune 済みの重みからでも一致する", () => {
  const tuned: Partial<HeuristicWeights> = {
    ukeire: 9.776926332250419,
    ukeireType: 3.892804436766673,
    dora: 64.43682755090288,
    foldDanger: 9.509882109687206,
    foldEfficiency: 0.05423917210149726,
    danger: { "安全": 0, "危険度低": 36.57, "危険度中": 56.52, "危険度高": 215.93 },
  };
  const merged: HeuristicWeights = {
    ...DEFAULT_WEIGHTS,
    ...tuned,
    danger: { ...DEFAULT_WEIGHTS.danger, ...tuned.danger },
  };
  const init = initFromWeights(merged);
  for (const { name, obs } of situations()) {
    const plain = new HeuristicPolicy("h", 7, { weights: tuned });
    const fitted = new HeuristicPolicy("h", 7, { weights: tuned, consumer: init });
    const a = scoreMap(plain, obs);
    const b = scoreMap(fitted, obs);
    // Non-integer weights make the last bit a matter of summation order; the
    // decision, which is all the engine consumes, must be untouched.
    for (const [tile, x] of a) {
      const y = b.get(tile)!;
      assert(
        Math.abs(y - x) <= 1e-9 * Math.max(1, Math.abs(x)),
        `${name}: 牌 ${tile} で ${y} ≠ ${x}`,
      );
    }
    assertEquals(plain.decide(obs), fitted.decide(obs), `${name}: 打牌`);
  }
});

Deno.test("consumer: 初期値の消費は打牌ではなく「消費」だけを置き換える", () => {
  // A curve set that is NOT the init must be able to move a decision — the
  // test above would pass on a consumer that was ignored entirely.
  const bent: ConsumerParams = {
    version: 1,
    curves: {
      ...INIT.curves,
      isolatedHonor: linearTheta(SPEC_BY_KEY.get("isolatedHonor")!.knots, 0),
    },
  };
  const obs = baseObs(); // 123456789m 1122p + lone 東: the honor goes on init
  const plain = new HeuristicPolicy("h", 7, { consumer: INIT });
  const other = new HeuristicPolicy("h", 7, { consumer: bent });
  const a = scoreMap(plain, obs);
  const b = scoreMap(other, obs);
  let moved = false;
  for (const [tile, x] of a) if (b.get(tile) !== x) moved = true;
  assert(moved, "孤立字牌の項を潰しても点数が動かないなら曲線は読まれていない");
});

Deno.test("consumer: 初期値の席0は半荘まるごと同一の対局になる", () => {
  for (const seed of [320001]) {
    const plainRun = playHanchan(
      seed,
      (seat) => new HeuristicPolicy(`H${seat}`, seed * 4 + seat),
    );
    const fittedRun = playHanchan(
      seed,
      (seat) =>
        new HeuristicPolicy(`H${seat}`, seed * 4 + seat, seat === 0 ? { consumer: INIT } : {}),
    );
    assertEquals(fittedRun.scores, plainRun.scores, `seed ${seed}: 終局点`);
    assertEquals(fittedRun.rounds.length, plainRun.rounds.length, `seed ${seed}: 局数`);
    assertEquals(fittedRun.ledger.length, plainRun.ledger.length, `seed ${seed}: 違反件数`);
  }
});

Deno.test("consumer: scoreDiscard の構造は M_atk·Σf − M_def·g + Σh", () => {
  // Read the pieces straight off the evidence: with the init curves, the whole
  // score is the textbook formula and every factor is identifiable.
  const obs = baseObs({
    hand: tiles("1234s567m11p發發西2m"),
    shanten: 3,
    junme: 9,
    riichi: [false, true, false, false],
    danger: dangerMap([["危険度高", "2m"]]),
  });
  const p = new HeuristicPolicy("h", 7, { consumer: INIT });
  const tile = tiles("2m")[0];
  const ev = evidenceOf(p, obs, tile);
  assertEquals(ev.context.folding, 1, "この場面は降りている");
  assertEquals(ev.context.eff, DEFAULT_WEIGHTS.foldEfficiency);
  assertEquals(ev.context.def, DEFAULT_WEIGHTS.foldDanger);

  const c = ev.candidate;
  const w = DEFAULT_WEIGHTS;
  const atk = -c.shantenAfter * w.shanten + c.doraKept * w.dora +
    c.yakuhaiPairs * w.yakuhaiPair - c.isolatedHonorLate * w.isolatedHonor;
  const expected = ev.context.eff * atk - ev.context.def * c.risk +
    c.drawBonus - c.keepBonus;
  assertExact(scoreDiscard(ev, INIT), expected, "構造");
});
