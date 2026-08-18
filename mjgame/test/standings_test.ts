// 順位効用 — the rank-utility layer.
//
// Four claims, and one test group each:
//   1. the normal-CDF machinery is a correct standard normal;
//   2. the standings model is a PROBABILITY DISTRIBUTION over placements — it
//      sums to one, and a flat table is the exact symmetric case;
//   3. the two scales are a statement about score POSITION and nothing else: a
//      flat table consumes as 1 × 1 at every stage, the same gap says more at
//      All Last than at East-1, the 8000 floor is never discounted, and the
//      雀鬼流 rabbit keeps a won game worth winning;
//   4. the layer is OFF unless asked for, and when asked for it moves the one
//      decision it claims to move — push/fold — in the direction claimed.

import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import type { Tile } from "mjrender/model.ts";
import { DEFAULT_WEIGHTS, HeuristicPolicy } from "../src/ai/heuristic.ts";
import {
  DEFAULT_STANDINGS_WEIGHTS,
  kyokuRemaining,
  phi,
  rankStats,
  standingsScales,
} from "../src/ai/standings.ts";
import type { StandingsWeights } from "../src/ai/standings.ts";
import type { Observation } from "../src/observe.ts";
import { JANKI } from "../src/rules.ts";
import { AKA_5P } from "../src/tiles.ts";
import { playHanchan } from "./helpers.ts";

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

const W = DEFAULT_STANDINGS_WEIGHTS;

/** Every table below adds up to 4 × 配給原点 (30000持ち), as a real one does. */
const FLAT = [30000, 30000, 30000, 30000];
/** A 20000 lead. Nothing short of a disaster loses this from here. */
const LEADER = [50000, 30000, 23000, 17000];
/** Last by 8000, and still 14000 clear of the 8000 line. */
const TRAILER = [22000, 35000, 33000, 30000];
/** Last, and one 6000 deal-in from 持ち点8000未満. */
const FLOOR = [13000, 38000, 36000, 33000];
/** The same three GAPS as `FLOOR`, lifted 10000 clear of the line. */
const FLOOR_CLEAR = [23000, 48000, 46000, 43000];

/**
 * A hand-built Observation. 順位効用 reads four fields — scores, kyoku, honba,
 * kyotaku — and the rest are filled with the empty/neutral value so the object
 * is a real `Observation` and not a cast.
 */
function obsOf(o: Partial<Observation> = {}): Observation {
  return {
    seat: 0,
    kyoku: 0,
    honba: 0,
    kyotaku: 0,
    junme: 6,
    wallRemaining: 70,
    hand: [],
    drawn: null,
    melds: [[], [], [], []],
    rivers: [[], [], [], []],
    scores: [...FLAT],
    riichi: [false, false, false, false],
    riichiJunme: [-1, -1, -1, -1],
    doraIndicators: [],
    seatWind: 28,
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

const scales = (scores: number[], kyoku: number, over: Partial<StandingsWeights> = {}) =>
  standingsScales(obsOf({ scores, kyoku }), { ...W, ...over });

// ---------------------------------------------------------------------------
// the normal machinery
// ---------------------------------------------------------------------------

Deno.test("順位効用: phi is a standard normal CDF", () => {
  // A&S 7.1.26 is a rational approximation, so 0 lands within its error bound
  // rather than exactly on 0.5 — well inside anything a 5000点 prior deserves.
  assertAlmostEquals(phi(0), 0.5, 1e-9);
  assertAlmostEquals(phi(1.96), 0.975, 1e-3);
  assertAlmostEquals(phi(-1.96), 0.025, 1e-3);
  assertAlmostEquals(phi(1), 0.8413447, 1e-6);
  for (const x of [0.1, 0.5, 1, 1.7, 2.5, 4]) {
    assertAlmostEquals(phi(-x), 1 - phi(x), 1e-9, `対称性 ${x}`);
  }
  // Monotone, and it saturates at the tails rather than running off.
  let prev = -1;
  for (let x = -6; x <= 6; x += 0.25) {
    const p = phi(x);
    assert(p >= prev, `単調 ${x}`);
    assert(p >= 0 && p <= 1, `確率 ${x}`);
    prev = p;
  }
});

Deno.test("順位効用: the current kyoku counts as remaining, and 連荘 pins at 1", () => {
  assertEquals(kyokuRemaining(0, W), 8, "東1では半荘まるごと残っている");
  assertEquals(kyokuRemaining(7, W), 1, "オーラスの結果こそが今決まるもの");
  // A 連荘 repeats the kyoku number and may run past the nominal length; one
  // more hand is one more hand, and it might be the last.
  assertEquals(kyokuRemaining(9, W), 1);
});

// ---------------------------------------------------------------------------
// the standings model
// ---------------------------------------------------------------------------

Deno.test("順位効用: a flat table is the exact symmetric case", () => {
  const r = rankStats(FLAT, 3, W);
  for (const p of r.pBeat) assertAlmostEquals(p, 0.5, 1e-9, "同点は素直に五分");
  assertAlmostEquals(r.rankDist.reduce((a, b) => a + b, 0), 1, 1e-9, "分布は1に和する");
  // 1e-8, not 1e-9: A&S 7.1.26 puts phi(0) about 5e-10 off exactly 0.5, and the
  // three pairwise comparisons carry that error into the mean. Tightening this
  // would be a statement about the erf approximation, not about the model.
  assertAlmostEquals(r.expRank, 2.5, 1e-8);
  // 1着と4着は片側3勝/3敗の1通りずつ、2着と3着は3通りずつ。
  assertAlmostEquals(r.rankDist[0], 0.125, 1e-9);
  assertAlmostEquals(r.rankDist[1], 0.375, 1e-9);
});

Deno.test("順位効用: expRank moves with the score, and stays inside 1..4", () => {
  for (const k of [0, 3, 7]) {
    const lead = rankStats(LEADER, k, W).expRank;
    const flat = rankStats(FLAT, k, W).expRank;
    const last = rankStats(TRAILER, k, W).expRank;
    assert(lead < flat && flat < last, `k=${k}: 点棒の順に期待順位が並ぶ`);
    for (const x of [lead, flat, last]) assert(x >= 1 && x <= 4);
    // Every distribution is a distribution, whatever the table.
    for (const s of [LEADER, FLAT, TRAILER]) {
      const d = rankStats(s, k, W).rankDist;
      assertAlmostEquals(d.reduce((a, b) => a + b, 0), 1, 1e-9);
    }
  }
  // Late in a decided game the leader's placement is nearly settled…
  assert(rankStats(LEADER, 7, W).expRank < 1.05);
  // …and early it is not, because there is a whole hanchan left to lose it in.
  assert(rankStats(LEADER, 0, W).expRank > 1.2);
});

// ---------------------------------------------------------------------------
// the two scales
// ---------------------------------------------------------------------------

Deno.test("順位効用: a flat table consumes as 1 × 1, at every stage and any pot", () => {
  for (const k of [0, 1, 4, 7, 12]) {
    const s = scales(FLAT, k);
    assertAlmostEquals(s.gain, 1, 1e-9, `k=${k} gain`);
    assertAlmostEquals(s.risk, 1, 1e-9, `k=${k} risk`);
    assertAlmostEquals(s.decisiveness, 1, 1e-9, `k=${k} 決着度`);
  }
  // 供託 and 本場 raise what a win is worth — but they raise it identically in
  // the numerator and in the flat reference, so a flat table is still 1 × 1.
  const pot = standingsScales(obsOf({ scores: FLAT, kyoku: 7, honba: 3, kyotaku: 2 }), W);
  assertAlmostEquals(pot.gain, 1, 1e-9);
  assertAlmostEquals(pot.risk, 1, 1e-9);
});

Deno.test("順位効用: a lead raises the price of danger and lowers the value of a win", () => {
  // Mid-game, where the lead is real but a hanchan can still take it away: the
  // asymmetry is wide and unmistakable.
  const mid = scales(LEADER, 3);
  assert(mid.risk > 1.2, `放銃が高くつく: ${mid.risk}`);
  assert(mid.gain < 0.85, `和了の限界価値は下がる: ${mid.gain}`);
  assert(mid.gain / mid.risk < 0.7, "押し引きは引き寄り");

  // At All Last the same lead is nearly untouchable, so the model has almost
  // nothing left to say about the REAL opponents — the direction still holds…
  const last = scales(LEADER, 7);
  assert(last.risk > 1, `オーラスでも放銃は高い: ${last.risk}`);
  assert(last.gain < 1, `…和了の限界価値は平場以下: ${last.gain}`);
  // …but the margin is thin ON PURPOSE: the rabbit (下記) is what holds `gain`
  // up, and without it this seat would be told to coast. Measured 0.976/1.024.
  assert(last.gain > 0.5, `兎がいる限り勝ちに行く値打ちは残る: ${last.gain}`);
});

Deno.test("順位効用: a deficit buys variance — the win is worth more, danger less", () => {
  for (const k of [3, 7]) {
    const s = scales(TRAILER, k);
    assert(s.gain > 1, `k=${k}: 和了の値打ちが上がる ${s.gain}`);
    assert(s.risk < 1, `k=${k}: 放銃の代償は下がる ${s.risk}`);
    assert(s.gain / s.risk > 1.2, `k=${k}: 押し寄り`);
  }
  const last = scales(TRAILER, 7);
  assert(last.gain > 1.4, `オーラスのラス目は大きく押す: ${last.gain}`);
  assert(last.risk < 0.8, `${last.risk}`);
});

Deno.test("順位効用: the 持ち点8000 line is never discounted", () => {
  // 13000持ち: one 6000 deal-in crosses the line, and crossing it is a ledger
  // entry — a binary demotion the rank model has no standing to price away.
  for (const k of [0, 3, 7]) {
    const guarded = scales(FLOOR, k);
    assertEquals(guarded.risk, 1, `k=${k}: 床の手前では割り引かせない`);
    // The deficit still buys offence; only the discount on danger is refused.
    assert(guarded.gain > 1, `k=${k}: 攻める理由は残る ${guarded.gain}`);
  }
  // The same three gaps, lifted clear of the line: now the discount applies,
  // which is what shows the guard was doing the work above.
  for (const k of [0, 3, 7]) {
    const free = scales(FLOOR_CLEAR, k);
    assert(free.risk < 1, `k=${k}: 床から離れれば素の割引 ${free.risk}`);
    assertEquals(free.gain, scales(FLOOR, k).gain, `k=${k}: 攻め側は点差だけの関数`);
  }
});

Deno.test("順位効用: the same gap says more the later it is said", () => {
  // Stage sharpening. Deliberately measured on a table that is still LIVE: a
  // runaway leader at All Last has almost no real placement left to move, and
  // there the rabbit (below) pulls both scales back towards 1 — which is the
  // 雀鬼流 correction working, not the sharpening failing.
  for (const s of [TRAILER, FLOOR_CLEAR]) {
    const early = scales(s, 0);
    const late = scales(s, 7);
    assert(
      Math.abs(late.gain - 1) > Math.abs(early.gain - 1),
      `gain: ${early.gain} → ${late.gain}`,
    );
    assert(
      Math.abs(late.risk - 1) > Math.abs(early.risk - 1),
      `risk: ${early.risk} → ${late.risk}`,
    );
  }
  // The leader's asymmetry sharpens too, over the stretch where it is live.
  assert(scales(LEADER, 3).gain < scales(LEADER, 0).gain);
  assert(scales(LEADER, 3).risk > scales(LEADER, 0).risk);
});

Deno.test("順位効用: the rabbit keeps a won game worth winning", () => {
  // 雀鬼流: keep WINNING, even from first — so the utility carries a virtual
  // player one placement above me, permanently 8000点 ahead. Without it, rank
  // utility saturates at 1着 and a secured leader is told to coast: `gain`
  // collapses to the floor and the layer recommends win-by-not-losing.
  const withRabbit = scales(LEADER, 7);
  const pureRank = scales(LEADER, 7, { phantomWeight: 0 });
  assert(pureRank.gain < withRabbit.gain, "兎なしでは和了の値打ちが潰れる");
  assertEquals(pureRank.gain, W.scaleMin, "純粋順位効用は独走で下限に張り付く");
  assert(withRabbit.gain > 3 * pureRank.gain, `${pureRank.gain} → ${withRabbit.gain}`);

  // The switch is exact: weight 0 removes the term, it does not merely shrink it.
  const flatRabbit = scales(FLAT, 7);
  const flatPure = scales(FLAT, 7, { phantomWeight: 0 });
  assertEquals(flatRabbit.gain, flatPure.gain, "平場ではどちらでも1 — 分母子で相殺する");
  assertAlmostEquals(flatPure.gain, 1, 1e-9);

  // A rabbit that ran away from every gain would be no rabbit: it is anchored
  // on the pre-branch score, so chasing it actually closes the distance and the
  // leader's `gain` stays a real number rather than a constant.
  assert(scales(LEADER, 7, { phantomLead: 20000 }).gain !== withRabbit.gain);
});

// ---------------------------------------------------------------------------
// what the policy does with them
// ---------------------------------------------------------------------------

/** The push/fold gate is `protected`; a test subclass is the only honest way in. */
class FoldProbe extends HeuristicPolicy {
  folds(obs: Observation): boolean {
    return this.shouldFold(obs);
  }
}

/** One riichi opponent and nothing else: `pressureOf` = 1, so the bar is 0.5. */
const underRiichi = (scores: number[], kyoku: number, doraCount: number): Observation =>
  obsOf({ scores, kyoku, doraCount, shanten: 1, riichi: [false, true, false, false] });

Deno.test("順位効用: off by default, and off means bit-identical", () => {
  assertEquals(DEFAULT_WEIGHTS.standings, undefined, "既定の重みは順位効用を持たない");
  const plain = new FoldProbe("plain", 1);
  const rank = new FoldProbe("rank", 1, { weights: { standings: W } });
  // On a flat table the layer is 1 × 1 by construction, so the two agree on
  // every position — including the ones where the gate is nowhere near its bar.
  for (const k of [0, 3, 7]) {
    for (const dora of [0, 1, 2]) {
      const obs = underRiichi(FLAT, k, dora);
      assertEquals(rank.folds(obs), plain.folds(obs), `平場 k=${k} ドラ${dora}`);
    }
  }
  // And with no weights at all it agrees everywhere, flat table or not.
  for (const s of [LEADER, TRAILER, FLOOR]) {
    const obs = underRiichi(s, 7, 1);
    assertEquals(new FoldProbe("b", 1).folds(obs), plain.folds(obs));
  }
});

Deno.test("順位効用: a protected lead folds where a point-EV agent pushes", () => {
  const plain = new FoldProbe("plain", 1);
  const rank = new FoldProbe("rank", 1, { weights: { standings: W } });
  // 1向聴 + ドラ1 under one riichi: push 0.57 against a bar of 0.5 — the
  // point-EV agent pushes, by a margin too small to survive a 1.28× price on
  // danger and a 0.78× discount on the win.
  const lead = underRiichi(LEADER, 3, 1);
  assertEquals(plain.folds(lead), false, "素の評価関数は押す");
  assertEquals(rank.folds(lead), true, "順位効用は同じ手を降ろす");
  // …and it is the SCORES that did it, not the kyoku: the same stage, flat.
  assertEquals(rank.folds(underRiichi(FLAT, 3, 1)), false);
});

Deno.test("順位効用: a deficit pushes tiles a point-EV agent would not", () => {
  const plain = new FoldProbe("plain", 1);
  const rank = new FoldProbe("rank", 1, { weights: { standings: W } });
  // 1向聴 with no dora: push 0.45 against a bar of 0.5 — a fold by a hair, and
  // last place at All Last has no use for a hair.
  const behind = underRiichi(TRAILER, 7, 0);
  assertEquals(plain.folds(behind), true, "素の評価関数は降りる");
  assertEquals(rank.folds(behind), false, "ラス目のオーラスは押す");
  assertEquals(rank.folds(underRiichi(FLAT, 7, 0)), true, "平場なら降りたまま");
});

Deno.test("順位効用: a seat carrying the layer plays a legal hanchan, deterministically", () => {
  const make = (rank: boolean) => (s: number) =>
    s === 0 && rank
      ? new HeuristicPolicy(`R${s}`, 909 * 4 + s, { weights: { standings: W } })
      : new HeuristicPolicy(`H${s}`, 909 * 4 + s);
  const a = playHanchan(909, make(true));
  const b = playHanchan(909, make(true));
  assert(a.rounds.length > 0);
  assertEquals(a.scores, b.scores, "same seed ⇒ same match");
  assertEquals(a.scores.reduce((x, y) => x + y, 0), 4 * JANKI.startScore, "点棒は保存される");
  // And it is a different player from the baseline, or it measures nothing.
  const plain = playHanchan(909, make(false));
  assert(
    a.scores[0] !== plain.scores[0] || a.ledger.length !== plain.ledger.length,
    "順位効用席が素の評価関数と同一 — 尺度が効いていない",
  );
});
