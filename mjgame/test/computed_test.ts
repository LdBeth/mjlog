// 計算 — the combinatorial ReadsProvider.
//
// Three layers of test, matching the three claims the module makes:
//   1. the wait-shape survival count is EXACT — a shape public facts refute
//      scores zero, and nothing else does;
//   2. the provider is PURE and PUBLIC — same Observation ⇒ same Reads, and the
//      Reads never contain a statement about the order of unseen tiles;
//   3. a seat driven by it plays legal, reproducible mahjong.

import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import { tileType } from "mjrender/tiles.ts";
import { AugmentedHeuristic } from "../src/ai/augmented.ts";
import type { Reads } from "../src/ai/augmented.ts";
import {
  combineShapes,
  computedReads,
  DEFAULT_COMPUTED,
  genbutsuSets,
  meldReadOf,
  mergeComputed,
  SHAPE_ROW_LEN,
  shapeBaseMasses,
  shapeFlagsOf,
  shapeRowTS,
  tenpaiPriorOf,
  valueOnType,
  WAIT_SHAPES,
  waitLikelihood,
  waitRowFrom,
  waitShapeWeights,
} from "../src/ai/computed.ts";
import type {
  ComputedWeights,
  MeldRead,
  ShapeBase,
  WaitContext,
  WaitShape,
} from "../src/ai/computed.ts";
import { HeuristicPolicy } from "../src/ai/heuristic.ts";
import { publicUnseen } from "../src/ai/planner.ts";
import { makeDojoHooks } from "../src/dojo.ts";
import { runMatchSync } from "../src/match.ts";
import type { Observation } from "../src/observe.ts";
import { AKA_5P, zeros34 } from "../src/tiles.ts";
import { DOJO_DEFAULT, JANKI } from "../src/rules.ts";
import { ronValue, scorer } from "../src/score.ts";
import { Table } from "../src/table.ts";
import { SEATS } from "../src/types.ts";
import { Wall } from "../src/wall.ts";
import { playHanchan, tiles } from "./helpers.ts";

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

const M3 = 2; // 三萬 — rank 3, so every one of the five shapes can reach it
const P5 = 13; // ⑤筒 — rank 5, the interior case
const S6 = 23; // ６索
const HAKU = 31; // 白

/** A full unseen pool: nothing at all has been seen. */
const ALL_LIVE: readonly number[] = new Array<number>(34).fill(4);

function ctxOf(o: Partial<WaitContext> = {}): WaitContext {
  return {
    unseen: o.unseen ?? ALL_LIVE,
    genbutsu: o.genbutsu ?? new Set<number>(),
    valueHonors: o.valueHonors ?? new Set<number>(),
    ...(o.read ? { read: o.read } : {}),
    ...(o.dora ? { dora: o.dora } : {}),
  };
}

/** `unseen` with the listed types knocked down to the given counts. */
function pool(...set: [number, number][]): number[] {
  const u = new Array<number>(34).fill(4);
  for (const [ty, n] of set) u[ty] = n;
  return u;
}

const river = (spec: string, junme = 1): RiverEntry[] =>
  tiles(spec).map((tile) => ({ tile, junme, tsumogiri: false, riichiDeclare: false }));

/**
 * A hand-built Observation. Only the fields `computedReads` actually counts are
 * interesting; the rest are filled with the empty/neutral value so the object is
 * a real `Observation` and not a cast.
 */
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

const shapes = (ty: number, c: Partial<WaitContext> = {}): Record<WaitShape, number> =>
  waitShapeWeights(ty, ctxOf(c));

/** A meld in front of the shimocha (Observation index 1). */
const meldOf = (kind: Meld["kind"], spec: string): Meld => {
  const t = tiles(spec);
  return { kind, who: 1, fromWho: kind === "ankan" ? 1 : 0, tiles: t, calledTile: t[0] };
};

/** The same weights with the meld-content multipliers switched off. */
const NO_CONTENT = { honitsuHot: 1, honitsuCold: 1, toitoiPair: 1, toitoiRun: 1 };

const alive = (r: Record<WaitShape, number>): WaitShape[] => WAIT_SHAPES.filter((s) => r[s] > 0);

// ---------------------------------------------------------------------------
// wait-shape survival
// ---------------------------------------------------------------------------

Deno.test("計算: an untouched rank-3 tile leaves all five shapes alive", () => {
  assertEquals(alive(shapes(M3)), [...WAIT_SHAPES]);
  // …and the interior 5筒 has every shape but ペンチャン, which cannot reach it.
  assertEquals(alive(shapes(P5)), ["リャンメン", "カンチャン", "シャンポン", "タンキ"]);
  // An honor has only the two shapes that wait on copies of itself.
  assertEquals(alive(shapes(HAKU)), ["シャンポン", "タンキ"]);
});

Deno.test("計算: four copies visible kills シャンポン and タンキ, and only those", () => {
  const dead = shapes(P5, { unseen: pool([P5, 0]) });
  assertEquals(dead["シャンポン"], 0, "they cannot hold a pair of a type nobody has left");
  assertEquals(dead["タンキ"], 0, "…nor the single copy a タンキ waits with");
  // A 両面 waits on the tile with its BRIDGING tiles, which are untouched here:
  // the last copy in our own hand can still deal into 46筒 or 34筒.
  assertEquals(alive(dead), ["リャンメン", "カンチャン"]);

  // Three visible leaves タンキ alive and シャンポン dead: C(1,2) = 0 holdings.
  const three = shapes(P5, { unseen: pool([P5, 1]) });
  assertEquals(three["シャンポン"], 0);
  assert(three["タンキ"] > 0);
});

Deno.test("計算: カベ kills the run shapes that need the dead bridging tile", () => {
  // No 4筒 anywhere: the 46筒 カンチャン and the 45筒 両面 are both impossible.
  const kabe = shapes(P5, { unseen: pool([P5 - 1, 0]) });
  assertEquals(kabe["カンチャン"], 0);
  // The upper 両面 (67筒 waiting 5/8) survives — only the lower holding died.
  assertAlmostEquals(kabe["リャンメン"], shapes(P5)["リャンメン"] / 2, 1e-9);
});

Deno.test("計算: 現物 is a proof — a passed type deals in to nobody", () => {
  const all = shapes(P5, { genbutsu: new Set([P5]) });
  assertEquals(alive(all), [], "furiten kills every shape at once");
  assertEquals(waitLikelihood(P5, ctxOf({ genbutsu: new Set([P5]) })), 0);
});

Deno.test("計算: スジ kills the リャンメン holding it refutes, and nothing else", () => {
  const full = shapes(P5);
  // Half suji: 8筒 passed ⇒ the 67筒 holding is furiten. The 34筒 holding lives.
  const half = shapes(P5, { genbutsu: new Set([P5 + 3]) });
  assertAlmostEquals(half["リャンメン"], full["リャンメン"] / 2, 1e-9);
  // Full suji: both ends passed ⇒ no 両面 can be waiting on it at all.
  const suji = shapes(P5, { genbutsu: new Set([P5 - 3, P5 + 3]) });
  assertEquals(suji["リャンメン"], 0);
  for (const s of ["カンチャン", "シャンポン", "タンキ"] as WaitShape[]) {
    assertEquals(suji[s], full[s], `スジ says nothing about ${s}`);
  }
  assert(waitLikelihood(P5, ctxOf({ genbutsu: new Set([P5 - 3, P5 + 3]) })) > 0);
});

Deno.test("計算: a value honor is held as a pair — its シャンポン weight is raised", () => {
  const guest = shapes(HAKU);
  const value = shapes(HAKU, { valueHonors: new Set([HAKU]) });
  assertEquals(value["シャンポン"], guest["シャンポン"] * DEFAULT_COMPUTED.yakuhaiShanpon);
  assertEquals(value["タンキ"], guest["タンキ"], "a タンキ is one copy either way");
});

Deno.test("計算: revealing a needed copy never raises the estimate", () => {
  for (const ty of [0, 2, 4, P5, 8, 26, HAKU]) {
    for (const t of [ty - 2, ty - 1, ty, ty + 1, ty + 2]) {
      if (t < 0 || t > 33) continue;
      let prev = Infinity;
      for (let n = 4; n >= 0; n--) {
        const p = waitLikelihood(ty, ctxOf({ unseen: pool([t, n]) }));
        assert(p <= prev + 1e-12, `type ${ty}: ${t} が ${n} 枚残りで確率が上がった`);
        prev = p;
      }
    }
  }
});

Deno.test("計算: the tenpai prior is a rate by (副露数, 巡目) — and riichi is 1", () => {
  const w = DEFAULT_COMPUTED;
  assertEquals(tenpaiPriorOf(w, 0, 3, true), 1, "a declared riichi IS tenpai");
  assertEquals(tenpaiPriorOf(w, 0, 3, false), w.tenpaiPrior[0][0]);
  assertEquals(tenpaiPriorOf(w, 0, 8, false), w.tenpaiPrior[0][1]);
  assertEquals(tenpaiPriorOf(w, 2, 11, false), w.tenpaiPrior[2][2]);
  assertEquals(tenpaiPriorOf(w, 9, 30, false), w.tenpaiPrior[4][3], "both indices clamp");
  // Monotone in both arguments: more melds, later turn ⇒ never less likely.
  for (let m = 0; m < 4; m++) {
    for (const j of [1, 7, 10, 14]) {
      assert(tenpaiPriorOf(w, m + 1, j, false) >= tenpaiPriorOf(w, m, j, false));
    }
  }
});

// ---------------------------------------------------------------------------
// what the melds say — public CONTENT, not a history
// ---------------------------------------------------------------------------

Deno.test("計算: 染め手模様 — その色と字牌が熱く、他の二色は冷える", () => {
  const w = DEFAULT_COMPUTED;
  const flush = [meldOf("pon", "222m"), meldOf("chi", "678m")];
  const obs = obsOf({ junme: 9, melds: [[], flush, [], []] });
  const M4 = 3; // ４萬 — untouched by either meld
  const read = computedReads()(obs)!.dealinP![0];
  const flat = computedReads(NO_CONTENT)(obs)!.dealinP![0];
  assertAlmostEquals(read[M4] / flat[M4], w.honitsuHot, 1e-5, "萬子は熱い");
  assertAlmostEquals(read[HAKU] / flat[HAKU], w.honitsuHot, 1e-5, "字牌も一緒に熱い");
  assertAlmostEquals(read[P5] / flat[P5], w.honitsuCold, 1e-5, "筒子は冷える");
  assertAlmostEquals(read[S6] / flat[S6], w.honitsuCold, 1e-5, "索子も冷える");
  // …and the same melds price the hand at the 染め手 base, not the flat open one.
  assertEquals(computedReads()(obs)!.expLoss![0], w.valueHonitsu);

  // Two melds spanning two suits say nothing: no modifier at all, either way.
  const mixed = obsOf({
    junme: 9,
    melds: [[], [meldOf("pon", "222m"), meldOf("chi", "678p")], [], []],
  });
  const mr = computedReads()(mixed)!;
  const mf = computedReads(NO_CONTENT)(mixed)!;
  for (const ty of [M4, HAKU, P5, S6]) {
    assertEquals(mr.dealinP![0][ty], mf.dealinP![0][ty], `type ${ty}: 混色の副露は読みではない`);
  }
  assertEquals(mr.expLoss![0], w.valueOpen, "素の副露相場のまま");
});

Deno.test("計算: トイトイ模様 — 対子形が上がり、順子形が下がる", () => {
  const w = DEFAULT_COMPUTED;
  const pons = [meldOf("pon", "222m"), meldOf("pon", "555s")];
  const read: MeldRead = meldReadOf(pons, new Set<number>());
  assertEquals(read.toitoi, true);
  assertEquals(read.honitsuSuit, null, "二色に跨がる副露は染め手ではない");
  const on = shapes(P5, { read });
  const off = shapes(P5);
  for (const s of ["シャンポン", "タンキ"] as WaitShape[]) {
    assertAlmostEquals(on[s], off[s] * w.toitoiPair, 1e-12);
  }
  for (const s of ["リャンメン", "カンチャン"] as WaitShape[]) {
    assertAlmostEquals(on[s], off[s] * w.toitoiRun, 1e-12);
  }
  const pairShare = (r: Record<WaitShape, number>) => {
    let pair = 0, all = 0;
    for (const s of WAIT_SHAPES) {
      all += r[s];
      if (s === "シャンポン" || s === "タンキ") pair += r[s];
    }
    return pair / all;
  };
  assert(pairShare(on) > pairShare(off), "対子形の取り分が増える");

  // One chi and the read is off — a hand that ate a run is not a 対々和.
  const withChi = meldReadOf([...pons, meldOf("chi", "678p")], new Set<number>());
  assertEquals(withChi.toitoi, false);
  assertEquals(withChi.honitsuSuit, null);
  assertEquals(shapes(P5, { read: withChi }), off, "読みが消えれば重みは素のまま");
});

Deno.test("計算: 役牌ポンは聴牌率も打点も素の副露1より上", () => {
  const w = DEFAULT_COMPUTED;
  // The prior bump is additive over the (副露数, 巡目) cell, and clamped there.
  assert(tenpaiPriorOf(w, 1, 9, false, true) > w.tenpaiPrior[1][1], "素の副露1のセルより上");
  assertEquals(tenpaiPriorOf(w, 1, 9, false, true), w.tenpaiPrior[1][1] + w.yakuhaiTenpai);
  assertEquals(tenpaiPriorOf(w, 4, 20, false, true), w.tenpaiPrior[4][3], "副露4の天井で止まる");
  assertEquals(tenpaiPriorOf(w, 0, 3, true, true), 1, "リーチはリーチ");

  const haku = computedReads()(
    obsOf({ junme: 9, melds: [[], [meldOf("pon", "白白白")], [], []] }),
  )!;
  const tan = computedReads()(obsOf({ junme: 9, melds: [[], [meldOf("chi", "678p")], [], []] }))!;
  assertEquals(tan.expLoss![0], w.valueOpen, "タンヤオのチーは素の副露相場");
  assertEquals(haku.expLoss![0], w.valueOpen + w.valueYakuhai, "役牌ポンは一翻ぶん高い");
  // The tenpai bump carries straight into the deal-in row (M3 is untouched by
  // either meld, so the ratio is the prior ratio and nothing else).
  assertAlmostEquals(
    haku.dealinP![0][M3] / tan.dealinP![0][M3],
    (w.tenpaiPrior[1][1] + w.yakuhaiTenpai) / w.tenpaiPrior[1][1],
    1e-5,
  );
});

Deno.test("計算: 副露の中の赤ドラもドラとして数える", () => {
  const w = DEFAULT_COMPUTED;
  // `tiles` hands out copies in ascending id order and ids 52/53 ARE the two
  // 赤⑤筒, so a pon of three ⑤筒 built here contains both of them.
  const aka = obsOf({ junme: 9, melds: [[], [meldOf("pon", "555p")], [], []] });
  const plain = obsOf({ junme: 9, melds: [[], [meldOf("pon", "222p")], [], []] });
  assertEquals(computedReads()(plain)!.expLoss![0], w.valueOpen, "指標ドラなしの素の副露");
  assertEquals(
    computedReads()(aka)!.expLoss![0],
    w.valueOpen + 2 * w.valuePerDora,
    "赤は表示牌と無関係に副露の中でもドラ",
  );
});

Deno.test("計算: tenpaiP は閾値で切るだけ — 上は生の事前確率のまま", () => {
  const w = DEFAULT_COMPUTED;
  const at = (junme: number) => computedReads()(obsOf({ junme }))!.tenpaiP![0];
  // Below the floor: a silent table says nothing the base heuristic lacks.
  assertEquals(at(8), 0, `${w.tenpaiPrior[0][1]} < ${w.tenpaiFloor} ⇒ 0`);
  // At the floor: reported as itself. This is the regression the rescale caused —
  // it used to arrive as 0 here, and 0.38 used to arrive as ≈0.17.
  assertEquals(at(11), 0.25);
  assertEquals(at(11), w.tenpaiPrior[0][2]);
  assertEquals(at(13), 0.38, "大詰の門前は満額の圧力で届く");
});

// ---------------------------------------------------------------------------
// the provider over an Observation
// ---------------------------------------------------------------------------

Deno.test("計算: genbutsu from the rivers zeroes that opponent's deal-in", () => {
  const obs = obsOf({
    junme: 8,
    rivers: [[], river("5p"), [], []],
  });
  const r = computedReads()(obs)!;
  assertEquals(r.dealinP![0][P5], 0, "shimocha passed ⑤筒 — they are furiten on it");
  assert(r.dealinP![0][P5 + 1] > 0, "…and its neighbour is untouched");
  assert(r.dealinP![1][P5] > 0, "the other two never saw it");
});

Deno.test("計算: a riichi opponent is priced at P(tenpai)=1 and everything since is 現物", () => {
  const obs = obsOf({
    junme: 6,
    riichi: [false, false, true, false],
    riichiJunme: [-1, -1, 3, -1],
    // toimen (relative 2) declared on 3巡目; kamicha discarded 6索 on 5巡目,
    // which passed the riichi and can never win it now.
    rivers: [[], [], river("1m", 3), river("6s", 5)],
  });
  const r = computedReads()(obs)!;
  const riichiSeat = 1; // Reads index 1 = Observation index 2
  assertEquals(r.tenpaiP![riichiSeat], 1);
  assertEquals(r.dealinP![riichiSeat][S6], 0, "passed after the declaration ⇒ 現物");
  assert(r.dealinP![riichiSeat][S6 + 1] > 0);
  // The quiet seats are read by the base rate alone, so they price far lower.
  assert(r.dealinP![riichiSeat][P5] > 4 * r.dealinP![0][P5]);
  // …and the value model prices a riichi above a silent closed hand.
  assert(r.expLoss![riichiSeat] > r.expLoss![0]);
});

Deno.test("計算: the value model reads melded dora and the dealer's seat", () => {
  const pon: Meld = {
    kind: "pon",
    who: 1,
    fromWho: 0,
    tiles: tiles("222p"),
    calledTile: tiles("222p")[0],
  };
  const obs = obsOf({
    junme: 9,
    // We are 南 (seatWind 28), so the dealer sits at relative index 3 (kamicha).
    seatWind: 28,
    doraIndicators: tiles("1p"), // ⇒ ②筒 is dora, and they melded three of them
    melds: [[], [pon], [], []],
  });
  const w = DEFAULT_COMPUTED;
  const r = computedReads()(obs)!;
  const plain = computedReads()(obsOf({ junme: 9, seatWind: 28 }))!;
  // An open hand is cheaper by default and dearer by its three visible dora.
  assertEquals(r.expLoss![0], w.valueOpen + 3 * w.valuePerDora);
  assertEquals(plain.expLoss![0], w.valueDamaten, "no melds, no riichi ⇒ ダマ聴の相場");
  // The winning tile's own dora count is public too, so it is priced per type.
  assertEquals(
    r.dealinValue![0][10] - r.dealinValue![0][11],
    w.valuePerDora,
    "②筒 で和了られる方が高い",
  );
  // The dealer is at relative 3 = Reads index 2.
  assertEquals(plain.expLoss![2], plain.expLoss![0] * w.valueDealer);
});

Deno.test("計算: wallComposition is the unseen pool, scaled onto the live wall", () => {
  const hand = tiles("123m456p0p789s東東白");
  const obs = obsOf({
    junme: 5,
    wallRemaining: 42,
    hand,
    doraIndicators: tiles("9m"),
    rivers: [river("南"), river("西"), river("北"), river("發")],
    melds: [
      [],
      [{ kind: "pon", who: 1, fromWho: 0, tiles: tiles("555s"), calledTile: tiles("555s")[0] }],
      [],
      [],
    ],
  });
  const unseen = publicUnseen(obs);
  const visible = 13 + 1 + 4 + 3; // hand + indicator + four discards + one meld
  let total = 0;
  for (const n of unseen) {
    assert(n >= 0, "an unseen count is never negative");
    total += n;
  }
  assertEquals(total, 136 - visible, "unseen = 136 − 見えている枚数");
  // Our hand holds ⑤筒 and 赤⑤筒: the red five is a COPY of type ⑤筒 and is
  // counted as one, exactly like every other tile here.
  assertEquals(unseen[13], 2, "赤⑤筒 も ⑤筒 の1枚として数える");
  assertEquals(unseen[22], 1, "three ５索 in a meld leaves one");

  const r = computedReads()(obs)!;
  let live = 0;
  for (const n of r.wallComposition!) live += n;
  assertAlmostEquals(live, obs.wallRemaining, 1e-3, "山に残る枚数の分だけ配分される");
  // Each opponent's share is their real concealed hand size: 13 − 3×副露.
  const sizes = [10, 13, 13];
  for (let i = 0; i < 3; i++) {
    let n = 0;
    for (const x of r.oppConcealed![i]) n += x;
    assertAlmostEquals(n, sizes[i], 1e-3);
  }
  // The ratio the planner actually consumes is P(next draw is this type).
  assertAlmostEquals(r.wallComposition![0] / live, unseen[0] / total, 1e-6);
});

Deno.test("計算: the planner flag is an option, and only the flag", () => {
  const obs = obsOf({ junme: 4 });
  const off = computedReads()(obs)!;
  const on = computedReads({ planner: true })(obs)!;
  assertEquals(off.planner, undefined);
  assertEquals(on.planner, true);
  // Availability is public either way; engaging the planner adds no information.
  assertEquals(off.wallComposition, on.wallComposition);
  assertEquals(off.oppConcealed, on.oppConcealed);
  assertEquals(off.dealinP, on.dealinP);
});

Deno.test("計算: no field of the Reads shape states an ORDER of unseen tiles", () => {
  const forbidden: (keyof Reads)[] = ["nextDraw", "ownNextDraw", "nextDora", "riichiNextDraw"];
  const obs = obsOf({
    junme: 11,
    riichi: [false, true, false, true],
    riichiJunme: [-1, 9, -1, 10],
    rivers: [river("1m"), river("2m"), river("3m"), river("4m")],
  });
  const r = computedReads({ planner: true })(obs)!;
  for (const f of forbidden) {
    assert(!(f in r), `${String(f)} は牌の順序の主張 — 数えても分からない`);
  }
  assertEquals(Object.keys(r).sort(), [
    "dealinP",
    "dealinValue",
    "expLoss",
    "oppConcealed",
    "planner",
    "tenpaiP",
    "wallComposition",
  ]);
});

Deno.test("計算: the provider is pure — same Observation, same Reads", () => {
  const mk = () =>
    obsOf({
      junme: 7,
      wallRemaining: 33,
      hand: tiles("123m456p789s東東2m"),
      riichi: [false, false, true, false],
      riichiJunme: [-1, -1, 5, -1],
      rivers: [river("東"), river("9m"), river("5s", 5), river("3p")],
      doraIndicators: tiles("4s"),
    });
  const p = computedReads({ planner: true });
  assertEquals(p(mk()), p(mk()), "同じ観測 ⇒ 同じ読み");
  // A second provider with the same weights is the same function, too.
  assertEquals(computedReads()(mk()), computedReads()(mk()));
});

Deno.test("計算: genbutsuSets orders same-go-around discards by turn order", () => {
  // Seat 0 is 北 (seatWind 30) ⇒ the dealer is at relative index 1, and the turn
  // order within a go-around is 1 → 2 → 3 → 0. The riichi is at relative 2, so
  // in its own declaring go-around only relative 3 and 0 come after it.
  const obs = obsOf({
    seatWind: 30,
    junme: 4,
    riichi: [false, false, true, false],
    riichiJunme: [-1, -1, 4, -1],
    rivers: [river("1m", 4), river("2m", 4), river("3m", 4), river("4m", 4)],
  });
  const g = genbutsuSets(obs)[2];
  assertEquals(g.has(tileType(tiles("3m")[0])), true, "their own discard");
  assertEquals(g.has(tileType(tiles("4m")[0])), true, "relative 3 acts after them");
  assertEquals(g.has(tileType(tiles("1m")[0])), true, "…and so do we");
  assertEquals(g.has(tileType(tiles("2m")[0])), false, "the dealer had already acted");
});

// ---------------------------------------------------------------------------
// end to end
// ---------------------------------------------------------------------------

function playComputed(seed: number, seats: string, plan = false) {
  return playHanchan(
    seed,
    (s) =>
      seats[s] === "k"
        ? new AugmentedHeuristic(`K${s}`, seed * 4 + s, computedReads({ planner: plan }))
        : new HeuristicPolicy(`H${s}`, seed * 4 + s),
  );
}

Deno.test("計算: a k seat plays a legal hanchan, deterministically", () => {
  const a = playComputed(101, "khhh");
  const b = playComputed(101, "khhh");
  assert(a.rounds.length > 0);
  assertEquals(a.scores, b.scores, "same seed ⇒ same match");
  assertEquals(a.scores.reduce((x, y) => x + y, 0), 4 * JANKI.startScore, "点棒は保存される");
  // It is a different policy from the plain heuristic, or it measures nothing.
  const plain = playComputed(101, "hhhh");
  assert(
    a.scores[0] !== plain.scores[0] || a.ledger.length !== plain.ledger.length,
    "計算席が素の評価関数と同一の打牌 — 読みが効いていない",
  );
});

Deno.test("計算: a whole table of k seats is legal too, planner on and off", () => {
  for (const plan of [false, true]) {
    const a = playComputed(202, "kkkk", plan);
    const b = playComputed(202, "kkkk", plan);
    assert(a.rounds.length > 0, `立案${plan}: 半荘が成立した`);
    assertEquals(a.scores, b.scores, `立案${plan}: same seed ⇒ same match`);
    assertEquals(a.scores.reduce((x, y) => x + y, 0), 4 * JANKI.startScore);
  }
});

Deno.test("計算: a k seat needs no tap on the Table — so `play` may seat one", () => {
  // THE difference from an "o" seat. `cmdPlay` hands its CPUs no `tableRef`
  // (there is nothing to tap in an interactive game — the human's hand is not
  // on offer) and runs under DOJO_DEFAULT. A seat that reads only its own
  // Observation does not notice.
  const policies = SEATS.map((s) =>
    s === 1
      ? new AugmentedHeuristic(`K${s}`, 707 * 4 + s, computedReads())
      : new HeuristicPolicy(`H${s}`, 707 * 4 + s)
  );
  const r = runMatchSync(policies, {
    seed: 707,
    cfg: JANKI,
    dojo: DOJO_DEFAULT,
    scorer,
    ...makeDojoHooks(DOJO_DEFAULT),
  });
  assert(r.rounds.length > 0, "卓に触らずとも半荘が成立する");
  assertEquals(r.scores.reduce((x, y) => x + y, 0), 4 * JANKI.startScore);
});

Deno.test("計算: nothing it reports depends on hidden state", () => {
  // The provider is handed an Observation and nothing else — no Table tap, no
  // closure over the round. The strongest cheap statement of that: two
  // Observations that differ ONLY in hidden state cannot exist, so instead show
  // the counted quantities move with the counts. Adding a copy of ⑤筒 to the
  // river lowers every opponent's ⑤筒 deal-in, monotonically.
  let prev = Infinity;
  for (const spec of ["", "5p", "5p5p", "5p5p5p"]) {
    const obs = obsOf({ junme: 8, rivers: [spec === "" ? [] : river(spec), [], [], []] });
    const p = computedReads()(obs)!.dealinP![0][P5];
    assert(p <= prev + 1e-12, `${spec}: 見えている枚数が増えて確率が上がった`);
    prev = p;
  }
  assert(prev >= 0);
  // And the pool the estimate counts against is exactly the visible bookkeeping.
  const empty = zeros34();
  assertEquals(publicUnseen(obsOf()).length, empty.length);
});

// ---------------------------------------------------------------------------
// M10b — the structural upgrades
//
// Five changes to the model's SHAPE (per-hand wait normalization, graded スジ,
// dora conditioning, tenpai multipliers, the 本場 surcharge), and one rule that
// governs all of them: every new parameter defaults to the value that leaves the
// shipped seat exactly where M10a left it, with the single deliberate exception
// of `valuePerHonba` — which is not a model but an arithmetic correction, and is
// pinned against `score.ts` below.
// ---------------------------------------------------------------------------

/** Every M10b knob turned on, well away from its no-op value. */
const M10B_ON: Partial<ComputedWeights> = {
  waitNormalize: true,
  expWaitMass: 1.38,
  sujiHalfSurvive: 0.25,
  sujiFullSurvive: 0.1,
  doraPair: 1.5,
  doraBridge: 1.3,
  tenpaiOtherRiichi: 0.6,
  tenpaiMeldDora: 1.15,
};

/** The whole 34-type wait row one context implies, under one vector. */
function waitRow(ctx: WaitContext, w: ComputedWeights = DEFAULT_COMPUTED): Float64Array {
  const bases: ShapeBase[] = [];
  for (let ty = 0; ty < 34; ty++) bases.push(shapeBaseMasses(ty, ctx));
  return waitRowFrom(bases, (ty) => shapeFlagsOf(ty, ctx), w);
}

const sum = (a: Float64Array): number => {
  let s = 0;
  for (const x of a) s += x;
  return s;
};

// ---- 1. the defaults are strict no-ops ------------------------------------

Deno.test("計算(M10b): 既定値では新しい素性が式にひとつも現れない", () => {
  const w = DEFAULT_COMPUTED;
  // An exhaustive sweep over bases whose NEW rows are all non-zero — if any of
  // them leaked into the arithmetic at its default weight, one of these would
  // move. The reference is the M10a expression, written out.
  let checked = 0;
  for (const ryanmen of [0, 7, 32]) {
    for (const ryanmenDora of [0, 3, ryanmen]) {
      for (const ryanmenHalf of [0, 16]) {
        for (const ryanmenFull of [0, 32]) {
          const base: ShapeBase = {
            ryanmen,
            ryanmenDora,
            ryanmenHalf,
            ryanmenFull,
            kanchan: 9,
            penchan: 4,
            shanpon: 3,
            tanki: 2,
          };
          for (const doraType of [false, true]) {
            for (const valueHonor of [false, true]) {
              const f = { valueHonor, honitsuSuit: null, toitoi: false, doraType };
              const got = combineShapes(13, base, f, w);
              assertEquals(got["リャンメン"], w.shapePrior["リャンメン"] * (ryanmen / 32));
              assertEquals(got["カンチャン"], w.shapePrior["カンチャン"] * 9 / 16);
              assertEquals(got["ペンチャン"], w.shapePrior["ペンチャン"] * 4 / 16);
              assertEquals(
                got["シャンポン"],
                w.shapePrior["シャンポン"] * (3 / 6) * (valueHonor ? w.yakuhaiShanpon : 1),
              );
              assertEquals(got["タンキ"], w.shapePrior["タンキ"] * (2 / 4));
              checked++;
            }
          }
        }
      }
    }
  }
  assertEquals(checked, 3 * 3 * 2 * 2 * 2 * 2);
});

Deno.test("計算(M10b): 既定値では卓のドラを渡しても読みが1ビットも動かない", () => {
  // The provider now hands `dora` into the wait context, so the new counting
  // runs on every real board. At the shipped weights it must change nothing.
  const dora = zeros34();
  dora[13] = 1; // ⑤筒
  dora[14] = 2; // ⑥筒 — a bridge tile of both ④筒 and ⑤筒
  const g = new Set<number>([16, 10]);
  for (const genbutsu of [new Set<number>(), g]) {
    const withDora = ctxOf({ genbutsu, dora });
    const without = ctxOf({ genbutsu });
    for (let ty = 0; ty < 34; ty++) {
      assertEquals(
        waitShapeWeights(ty, withDora),
        waitShapeWeights(ty, without),
        `牌種${ty}: 既定値でドラ条件付けが効いてしまっている`,
      );
    }
    assertEquals([...waitRow(withDora)], [...waitRow(without)]);
  }
});

// ---- 2. per-hand wait normalization ---------------------------------------

Deno.test("計算(M10b): 正規化すると1局面の待ち質量の合計が expWaitMass になる", () => {
  const w = mergeComputed(M10B_ON);
  const ctx = ctxOf({
    unseen: pool([4, 2], [13, 1], [26, 0]),
    genbutsu: new Set([0, 9, 31]),
  });
  const row = waitRow(ctx, w);
  assertAlmostEquals(sum(row), w.expWaitMass, 1e-9, "1手あたりの当たり牌種数がそのまま出る");
  // …and the level is per HAND: a board where far fewer shapes survive still
  // carries the same total, which is exactly what the un-normalized form
  // could not do (its total moved with the number of live types).
  const narrow = ctxOf({
    unseen: pool([4, 1], [5, 1], [6, 1]),
    genbutsu: new Set([0, 1, 2, 3, 9, 10, 11, 18, 19, 20, 27, 28]),
  });
  assertAlmostEquals(sum(waitRow(narrow, w)), w.expWaitMass, 1e-9);
  assert(
    Math.abs(sum(waitRow(narrow)) - sum(waitRow(ctx))) > 0.05,
    "正規化なしの合計は局面ごとに動く — それが直したかったもの",
  );
});

Deno.test("計算(M10b): 正規化は順位を保ち、証明で殺された牌は0のまま", () => {
  const w = mergeComputed(M10B_ON);
  const ctx = ctxOf({ unseen: pool([4, 2], [13, 3]), genbutsu: new Set([13, 16]) });
  const norm = waitRow(ctx, w);
  const raw = waitRow(ctx);
  assertEquals(norm[13], 0, "現物は正規化しても0 — 証明は取り分ではない");
  const order = (a: Float64Array) =>
    [...a.keys()].filter((ty) => a[ty] > 0).sort((x, y) => a[y] - a[x]);
  assertEquals(order(norm), order(raw), "正規化は水準だけを直し、順位には触れない");
  // Every type refuted ⇒ nothing to normalize, and no division by zero.
  const dead = ctxOf({ genbutsu: new Set(Array.from({ length: 34 }, (_, i) => i)) });
  assertEquals(sum(waitRow(dead, w)), 0);
});

Deno.test("計算(M10b): 正規化は席の読みにも通り、既定では通らない", () => {
  // A declared riichi is P(tenpai)=1 with yakuFactor 1, so that opponent's row
  // IS the wait distribution and its total is directly readable.
  const at = (rivers: RiverEntry[][], w?: Partial<ComputedWeights>) => {
    const row = computedReads(w)(obsOf({
      junme: 9,
      riichi: [false, true, false, false],
      riichiJunme: [-1, 8, -1, -1],
      rivers,
    }))!.dealinP![0];
    let s = 0;
    for (let ty = 0; ty < 34; ty++) s += row[ty];
    return s;
  };
  const quiet: RiverEntry[][] = [[], [], [], []];
  const busy: RiverEntry[][] = [
    river("1112223334m"),
    river("19m19p19s東南西北白"), // the riichi seat's own river: ten 現物
    river("111222333p"),
    river("111222333s"),
  ];
  const w = mergeComputed(M10B_ON);
  // Normalized: the same total on both boards, because a tenpai hand waits on
  // about `expWaitMass` types whatever else is on the table.
  assertAlmostEquals(at(quiet, M10B_ON), w.expWaitMass, 1e-4);
  assertAlmostEquals(at(busy, M10B_ON), w.expWaitMass, 1e-4);
  // Un-normalized: the total is a function of how much of the board is still
  // live — the level error the M10a report measured as 0.38 against a true 1.38.
  assert(
    at(quiet) / at(busy) > 1.8,
    `正規化なしの合計は盤面で動く: 静かな卓 ${at(quiet)} / 荒れた卓 ${at(busy)}`,
  );
});

// ---- 3. graded スジ --------------------------------------------------------

Deno.test("計算(M10b): スジで殺した両面の質量は捨てずに半スジ/全スジへ記帳される", () => {
  const all = ctxOf();
  // 半スジ: ⑤筒 with only 8筒 passed. The 67筒 holding is refuted, the 34筒 one
  // is untouched — the tile is still a リャンメン tile from the other side.
  const half = shapeBaseMasses(P5, ctxOf({ genbutsu: new Set([P5 + 3]) }));
  assertEquals(half.ryanmen, 16, "生きている方の持ち方はそのまま");
  assertEquals(half.ryanmenHalf, 16);
  assertEquals(half.ryanmenFull, 0);
  // 全スジ: both ends passed ⇒ no リャンメン reaches ⑤筒 at all.
  const full = shapeBaseMasses(P5, ctxOf({ genbutsu: new Set([P5 - 3, P5 + 3]) }));
  assertEquals([full.ryanmen, full.ryanmenHalf, full.ryanmenFull], [0, 0, 32]);
  // A 3萬 has ONE リャンメン that reaches it (12萬 does not — 45萬 does), so a
  // single passed 6萬 is already a full suji for it, not a half one.
  const term = shapeBaseMasses(M3, ctxOf({ genbutsu: new Set([M3 + 3]) }));
  assertEquals([term.ryanmen, term.ryanmenHalf, term.ryanmenFull], [0, 0, 16]);
  assertEquals(shapeBaseMasses(P5, all).ryanmenHalf, 0, "殺されていなければ記帳もない");

  // The two residues are priced apart, and neither is priced at all by default.
  const w = mergeComputed({ sujiHalfSurvive: 0.25, sujiFullSurvive: 0.1 });
  const prior = w.shapePrior["リャンメン"];
  assertAlmostEquals(
    combineShapes(P5, half, shapeFlagsOf(P5, all), w)["リャンメン"],
    prior * ((16 + 0.25 * 16) / 32),
    1e-12,
  );
  assertAlmostEquals(
    combineShapes(P5, full, shapeFlagsOf(P5, all), w)["リャンメン"],
    prior * ((0.1 * 32) / 32),
    1e-12,
  );
  assertEquals(combineShapes(P5, full, shapeFlagsOf(P5, all))["リャンメン"], 0, "既定は完全な殺し");
});

// ---- 4. dora conditioning --------------------------------------------------

Deno.test("計算(M10b): ドラの対子読み — シャンポン/タンキだけが、ドラ牌種だけで上がる", () => {
  const dora = zeros34();
  dora[P5] = 1;
  const w = mergeComputed({ doraPair: 1.5 });
  const hot = ctxOf({ dora });
  const cold = ctxOf();
  const on = waitShapeWeights(P5, hot, w);
  const off = waitShapeWeights(P5, cold, w);
  for (const s of ["シャンポン", "タンキ"] as WaitShape[]) {
    assertAlmostEquals(on[s], off[s] * 1.5, 1e-12, `${s} はドラで持たれやすい`);
  }
  for (const s of ["リャンメン", "カンチャン"] as WaitShape[]) {
    assertEquals(on[s], off[s], `${s} は対子読みと無関係`);
  }
  // The neighbour is not a dora, so nothing there moves.
  assertEquals(waitShapeWeights(P5 + 1, hot, w), waitShapeWeights(P5 + 1, cold, w));
  assertEquals(waitShapeWeights(P5, hot), waitShapeWeights(P5, cold), "既定は無操作");
});

Deno.test("計算(M10b): ドラを含む両面の橋だけが上がる", () => {
  const dora = zeros34();
  dora[P5 + 1] = 1; // ⑥筒 — in the (⑥⑦) bridge of ⑤筒, not in the (③④) one
  const w = mergeComputed({ doraBridge: 1.3 });
  const ctx = ctxOf({ dora });
  const base = shapeBaseMasses(P5, ctx);
  assertEquals(base.ryanmen, 32);
  assertEquals(base.ryanmenDora, 16, "ドラを含むのは上側の持ち方だけ");
  const prior = w.shapePrior["リャンメン"];
  assertAlmostEquals(
    combineShapes(P5, base, shapeFlagsOf(P5, ctx), w)["リャンメン"],
    prior * ((32 + 0.3 * 16) / 32),
    1e-12,
  );
  // …and a bridge with no dora in it is untouched, however loud the weight.
  const plain = shapeBaseMasses(8, ctxOf({ dora })); // 9萬
  assertEquals(plain.ryanmenDora, 0);
  assertEquals(
    combineShapes(8, plain, shapeFlagsOf(8, ctxOf({ dora })), w)["リャンメン"],
    combineShapes(8, plain, shapeFlagsOf(8, ctxOf({ dora })))["リャンメン"],
  );
});

// ---- 5. the tenpai multipliers ---------------------------------------------

Deno.test("計算(M10b): 他家の立直と副露ドラは聴牌率の倍率で、既定では1倍", () => {
  const w = DEFAULT_COMPUTED;
  const cell = w.tenpaiPrior[1][1]; // 副露1・7-9巡
  assertEquals(tenpaiPriorOf(w, 1, 9, false, false, true, 3), cell, "既定はどちらも無操作");

  const sup = mergeComputed({ tenpaiOtherRiichi: 0.6 });
  assertAlmostEquals(tenpaiPriorOf(sup, 1, 9, false, false, true), cell * 0.6, 1e-12);
  assertEquals(
    tenpaiPriorOf(sup, 1, 9, false, false, false),
    cell,
    "誰も立直していなければ素のまま",
  );
  assertEquals(tenpaiPriorOf(sup, 0, 3, true, false, true), 1, "立直した本人はルール上の1のまま");

  const dens = mergeComputed({ tenpaiMeldDora: 1.2 });
  assertAlmostEquals(tenpaiPriorOf(dens, 1, 9, false, false, false, 2), cell * 1.44, 1e-12);
  assertEquals(tenpaiPriorOf(dens, 1, 9, false, false, false, 0), cell, "ドラ0枚なら倍率も1");
  // A multiplier can never carry a prior out of [0,1].
  const loud = mergeComputed({ tenpaiMeldDora: 9 });
  assertEquals(tenpaiPriorOf(loud, 4, 20, false, false, false, 8), 1);
});

Deno.test("計算(M10b): 卓の立直は静かな他家の行に通り、立直した席には通らない", () => {
  const obs = obsOf({
    junme: 9,
    riichi: [false, true, false, false], // 下家 (Reads index 0) が立直
    riichiJunme: [-1, 6, -1, -1],
  });
  const w = mergeComputed({ tenpaiOtherRiichi: 0.5 });
  const on = computedReads({ tenpaiOtherRiichi: 0.5 })(obs)!;
  const off = computedReads()(obs)!;
  // The declarer is tenpai by rule: the multiplier must not touch their row.
  assertAlmostEquals(on.dealinP![0][M3], off.dealinP![0][M3], 1e-9);
  // The two silent seats see the other seat's declaration and are damped.
  for (const i of [1, 2]) {
    assertAlmostEquals(on.dealinP![i][M3], off.dealinP![i][M3] * 0.5, 1e-9);
  }
  assertEquals(w.tenpaiOtherRiichi, 0.5);
});

// ---- 6. the 本場 surcharge -------------------------------------------------

Deno.test("計算(M10b): 本場は模型ではなく計算 — score.ts と同じ 300点/本", () => {
  const w = DEFAULT_COMPUTED;
  assertEquals(w.valuePerHonba, 300, "ルールの数字であって感性の数字ではない");
  // The one place the figure is assembled: the surcharge rides ON TOP of the
  // cap, exactly as `ronValue` adds it on top of the hand's own limit.
  assertEquals(valueOnType(w, 5200, 0, 0), 5200);
  assertEquals(valueOnType(w, 5200, 0, 3), 5200 + 900);
  assertEquals(valueOnType(w, w.valueCap, 5, 2), w.valueCap + 600, "頭打ちの上に乗る");

  const plain = obsOf({ junme: 9 });
  const honba = obsOf({ junme: 9, honba: 4 });
  const a = computedReads()(plain)!;
  const b = computedReads()(honba)!;
  assertEquals(b.expLoss![0] - a.expLoss![0], 4 * 300);
  assertEquals(b.dealinValue![0][M3] - a.dealinValue![0][M3], 4 * 300);
  // …and switching the term off restores the pre-M10b figure exactly.
  assertEquals(computedReads({ valuePerHonba: 0 })(honba)!.expLoss![0], a.expLoss![0]);
});

Deno.test("計算(M10b): 本場の額は score.ts の ronValue が実際に請求する額と一致する", () => {
  // The claim is arithmetic, so it is checked against the scorer itself rather
  // than against a constant copied out of it.
  const mk = (honba: number) => {
    const taken = new Set<Tile>();
    const d = (spec: string): Tile[] =>
      tiles(spec).map((want) => {
        const ty = tileType(want);
        let id = ty * 4;
        while (taken.has(id)) id++;
        taken.add(id);
        return id;
      });
    const junk = () => d("149m149p149s東南西北");
    const hands = [junk(), d("234m567m234p55p78s"), junk(), junk()];
    const wallTiles: Tile[] = new Array(136).fill(-1);
    let n = 0;
    const put = (id: Tile) => {
      wallTiles[135 - n++] = id;
    };
    for (let block = 0; block < 3; block++) {
      for (let k = 0; k < 4; k++) for (let i = 0; i < 4; i++) put(hands[k][block * 4 + i]);
    }
    for (let k = 0; k < 4; k++) put(hands[k][12]);
    wallTiles[5] = d("北")[0];
    const used = new Set(wallTiles.filter((id) => id >= 0));
    const spare: Tile[] = [];
    for (let id = 0; id < 136; id++) if (!used.has(id)) spare.push(id);
    for (let i = 0; i < 136; i++) if (wallTiles[i] < 0) wallTiles[i] = spare.pop()!;
    const t = new Table(
      {
        kyoku: 0,
        honba,
        kyotaku: 0,
        dealer: 0,
        scores: [25000, 25000, 25000, 25000],
        wall: new Wall(wallTiles),
        dice: [0, 0],
      },
      JANKI,
      SEATS.map((seat) => ({ seat, name: `P${seat}` })),
    );
    t.riichi[1] = true;
    return t;
  };
  const zero = ronValue(mk(0), 1, 0, 23 * 4); // ６索 で出上がり
  const three = ronValue(mk(3), 1, 0, 23 * 4);
  assert(zero !== null && three !== null);
  assertEquals(three - zero, 3 * DEFAULT_COMPUTED.valuePerHonba, "3本場ぶんの差がまさに 900点");
});

// ---- 7. nothing else moved -------------------------------------------------

/**
 * The 計算 seat's decision stream with the 本場 term OFF, seed by seed:
 * `scores#FNV(全局の結果)`.
 *
 * WHAT IT PINS. That `valuePerHonba` is the only weight in M10b's batch with a
 * behavioural effect at the shipped defaults — the four other upgrades
 * (`sujiHalfSurvive`, `sujiFullSurvive`, `doraBridge`, `waitNormalize`) are
 * exact no-ops there — and, from that, that nothing ELSE moves the seat either.
 * Seed 606 diverges with the term on (asserted below), which is what keeps the
 * pinning meaningful rather than vacuous.
 *
 * RE-CAPTURED at the unification of the unseen-tile count. These were originally
 * the M10a stream, but M10a priced ukeire liveness through two different
 * formulas — `Observation.ukeire[].live` for a type the resting hand happened to
 * accept and `4 − own copies` (blind to rivers, melds and indicators) for the
 * rest — so the seat it describes no longer exists. The claim the strings carry
 * is unchanged; only the code they were read off is newer.
 */
// Re-captured 2026-08-27: the owner-directed dojo rules correction (under-8000
// judged at game end, buffer 南入以降, call-gate 対々和/バック tightening)
// moved every default seat. The claim is unchanged; only the world is newer.
// Re-captured 2026-08-28 (owner-directed, shared-code bug fix): the danger
// assessor was fed the discarder's own tiles twice (`observe.ts` passed the
// own-inclusive `Table.visibleCounts` as the public count AND `ownCounts`),
// so a held pair rated 安全 through the killed-shapes cap; fixed to public
// counts + hand. Found from the ranked arena wire log. Claim unchanged.
const HONBA_OFF: Record<number, string> = {
  101: "60600/15100/22200/22100#6f4c9d6b",
  404: "15800/37200/39500/27500#fcb4e8e4",
  505: "10200/14300/60900/34600#d4b33c43",
  606: "24800/19900/23000/52300#eedf9a8c",
  707: "3400/29900/47300/39400#5078adbe",
};

function fingerprint(seed: number, w?: Partial<ComputedWeights>): string {
  const r = playHanchan(
    seed,
    (s) =>
      s === 0
        ? new AugmentedHeuristic(`K${s}`, seed * 4 + s, computedReads(w))
        : new HeuristicPolicy(`H${s}`, seed * 4 + s),
  );
  const body = JSON.stringify({
    scores: r.scores,
    outcomes: r.outcomes,
    ledger: r.ledger.map((v) => `${v.seat}:${v.label}`),
    riichis: r.riichis,
    furo: r.furoRounds,
  });
  let h = 0x811c9dc5;
  for (let i = 0; i < body.length; i++) h = Math.imul(h ^ body.charCodeAt(i), 0x01000193) >>> 0;
  return `${r.scores.join("/")}#${h.toString(16).padStart(8, "0")}`;
}

Deno.test("計算(M10b): 本場項を切れば対局はビット単位で固定されている", () => {
  let diverged = 0;
  for (const [seed, want] of Object.entries(HONBA_OFF)) {
    assertEquals(
      fingerprint(Number(seed), { valuePerHonba: 0 }),
      want,
      `種${seed}: 本場項以外に挙動が変わったものがある`,
    );
    if (fingerprint(Number(seed), {}) !== want) diverged++;
  }
  assert(diverged > 0, "本場を課しても何も変わらないなら、この訂正は測れていない");
});

// ---------------------------------------------------------------------------
// the flat hot path is the same arithmetic
// ---------------------------------------------------------------------------
//
// `shapeRowTS` writes the counts and the row into one Float64Array instead of
// allocating a `ShapeBase` per tile type and a `Record<WaitShape, number>` per
// cell — three quarters of what the 計算 seat spends its time on. It exists
// ONLY as a faster spelling of `shapeBaseMasses` + `waitRowFrom`, so the thing
// worth testing is that it is exactly that: the same double, bit for bit, on
// every slot and under every weight vector. (The whole-hanchan fingerprints
// above would catch a drift too, but not tell anyone where it was.)

/** A board's worth of public facts, arbitrary. */
function fuzzCtx(rng: () => number): WaitContext {
  const pick = (n: number) => Math.floor(rng() * n);
  const unseen: number[] = [];
  for (let t = 0; t < 34; t++) unseen.push(pick(5));
  const genbutsu = new Set<number>();
  for (let i = 0, n = pick(14); i < n; i++) genbutsu.add(pick(34));
  const dora: number[] = new Array(34).fill(0);
  for (let i = 0, n = pick(4); i < n; i++) dora[pick(34)]++;
  const suits = [null, "m", "p", "s"] as const;
  const read: MeldRead = {
    honitsuSuit: suits[pick(4)],
    toitoi: pick(2) === 0,
    yakuhai: new Set<number>(),
    open: pick(5),
  };
  return {
    unseen,
    genbutsu,
    valueHonors: new Set([31, 32, 33, 27 + pick(4), 27 + pick(4)]),
    read: pick(5) === 0 ? undefined : read,
    dora: pick(6) === 0 ? undefined : dora,
  };
}

Deno.test("計算: 平坦化した経路は定義そのものと1ビットも違わない", () => {
  // mulberry32 — a self-contained stream, so this test owns its own randomness.
  let s = 0x9e3779b9;
  const rng = () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const f = () => Math.round(rng() * 1000) / 1000;

  const vectors: ComputedWeights[] = [DEFAULT_COMPUTED, mergeComputed(M10B_ON)];
  for (let i = 0; i < 16; i++) {
    vectors.push(mergeComputed({
      shapePrior: {
        "リャンメン": f(),
        "カンチャン": f(),
        "ペンチャン": f(),
        "シャンポン": f(),
        "タンキ": f(),
      },
      yakuhaiShanpon: f() * 2,
      honitsuHot: f() * 2,
      honitsuCold: f() * 2,
      toitoiPair: f() * 2,
      toitoiRun: f() * 2,
      sujiHalfSurvive: f(),
      sujiFullSurvive: f(),
      doraPair: f() * 2,
      doraBridge: f() * 2,
      dealinScale: f() * 0.3,
      expWaitMass: f() * 3,
      waitNormalize: rng() < 0.5,
    }));
  }

  const flat = new Float64Array(SHAPE_ROW_LEN);
  let cells = 0;
  for (const w of vectors) {
    for (let n = 0; n < 120; n++) {
      const ctx = fuzzCtx(rng);
      shapeRowTS(ctx, w, flat);
      const want = waitRow(ctx, w);
      for (let ty = 0; ty < 34; ty++) {
        assertEquals(flat[ty], want[ty], `重み${vectors.indexOf(w)} 牌${ty}: 待ち確率が違う`);
        // and the counts the row was built from, field by field
        const b = shapeBaseMasses(ty, ctx);
        const o = 34 + ty * 8;
        assertEquals(
          [
            flat[o],
            flat[o + 1],
            flat[o + 2],
            flat[o + 3],
            flat[o + 4],
            flat[o + 5],
            flat[o + 6],
            flat[o + 7],
          ],
          [
            b.ryanmen,
            b.ryanmenDora,
            b.ryanmenHalf,
            b.ryanmenFull,
            b.kanchan,
            b.penchan,
            b.shanpon,
            b.tanki,
          ],
          `牌${ty}: 素の数え上げが違う`,
        );
        cells++;
      }
    }
  }
  assert(cells > 70000, `検査したセルが少なすぎる (${cells})`);
});
