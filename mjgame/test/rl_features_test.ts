// Feature encoding v3. The layout is a FROZEN contract shared with the trainer,
// so this file pins it three times over: once index by index against a single
// hand-built Observation (the `baseObs` pattern from heuristic_test.ts), once
// as an FNV-1a digest of the complete buffers, which catches any reorder the
// named assertions happen not to look at, and once as a PREFIX GUARD — the v2
// digests recomputed over the first 36 planes / 39 scalars of the v3 output,
// which is what the trainer's layer-surgery tool assumes when it copies the old
// first-layer columns across and zero-fills the new ones.

import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import type { Meld, Tile } from "mjrender/model.ts";
import type { DangerAssessment, DangerLevel } from "mjrender/danger.ts";
import type { RiverEntry } from "mjrender/state.ts";
import { doraFromIndicatorType, tileType } from "mjrender/tiles.ts";
import { countsFromTiles, shanten } from "mjrender/shanten.ts";
import { HeuristicPolicy } from "../src/ai/heuristic.ts";
import { RandomPolicy } from "../src/ai/random.ts";
import { makeDojoHooks } from "../src/main.ts";
import { runMatchSync } from "../src/match.ts";
import type { Observation } from "../src/observe.ts";
import type { SyncPolicy } from "../src/policy.ts";
import type { Encoded } from "../src/rl/features.ts";
import {
  encode,
  encodeOracle,
  FEATURES,
  flatten,
  INPUT_LEN,
  ORACLE_LEN,
  ORACLE_PLANES,
  PLANE_LEN,
  TYPES,
} from "../src/rl/features.ts";
import { DOJO_HEADLESS, JANKI } from "../src/rules.ts";
import { scorer } from "../src/score.ts";
import type { Table } from "../src/table.ts";
import type { Action, Seat } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { DRAWS_PER_ROUND } from "../src/wall.ts";
import { tiles } from "./helpers.ts";

// --------------------------------------------------------------------------
// the fixture
// --------------------------------------------------------------------------

/** The 34-long stretch plane `p` occupies. */
function plane(e: Encoded, p: number): Int8Array {
  return e.planes.slice(p * TYPES, (p + 1) * TYPES);
}

/** The tile types a plane lights up, ascending. */
function lit(e: Encoded, p: number): number[] {
  const out: number[] = [];
  const s = plane(e, p);
  for (let ty = 0; ty < TYPES; ty++) if (s[ty]) out.push(ty);
  return out;
}

/** The v2 shape, kept here on purpose: it is the prefix v3 must not disturb. */
const V2_PLANES = 36;
const V2_SCALARS = 39;

/** The cells of one plane as plain numbers — for the v3 counting planes. */
function cells(e: Encoded, p: number): number[] {
  return [...plane(e, p)];
}

/** The non-zero cells of a plane as `[type, value]` pairs, ascending. */
function nonzero(e: Encoded, p: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  const s = plane(e, p);
  for (let ty = 0; ty < TYPES; ty++) if (s[ty]) out.push([ty, s[ty]]);
  return out;
}

function river(ids: Tile[]): RiverEntry[] {
  return ids.map((tile, i) => ({ tile, junme: i + 1, tsumogiri: false, riichiDeclare: false }));
}

/** A river whose entries carry per-entry 手出し / リーチ marks. */
function markedRiver(
  specs: Array<{ tile: Tile; tsumogiri?: boolean; riichiDeclare?: boolean }>,
): RiverEntry[] {
  return specs.map((sp, i) => ({
    tile: sp.tile,
    junme: i + 1,
    tsumogiri: sp.tsumogiri ?? false,
    riichiDeclare: sp.riichiDeclare ?? false,
  }));
}

/** The bare assessment the encoder reads: only `level` reaches the planes. */
function assessment(level: DangerLevel): DangerAssessment {
  return { level, seats: [1], details: [{ seat: 1, level, kind: "riichi", notes: [] }] };
}

/**
 * An assessment with one detail per threat seat, `seat` being the ABSOLUTE seat
 * `Table.threats` reports. The summary `level` is the worst of them, exactly as
 * `assessDanger` builds it.
 */
function assessSeats(per: Array<[number, DangerLevel]>): DangerAssessment {
  const RANK: Record<DangerLevel, number> = {
    "安全": 0,
    "危険度低": 1,
    "危険度中": 2,
    "危険度高": 3,
  };
  const worst = per.reduce((a, [, l]) => (RANK[l] > RANK[a] ? l : a), "安全" as DangerLevel);
  return {
    level: worst,
    seats: per.map(([seat]) => seat),
    details: per.map(([seat, level]) => ({ seat, level, kind: "riichi" as const, notes: [] })),
  };
}

// Every zone draws from its own `tiles()` call, and the specs are chosen so no
// tile id is handed out twice — the visible-count planes dedupe by id, so a
// collision between zones would silently change p20/p21.
const HAND = [...tiles("1111m222m33m4m"), ...tiles("0p")]; // 4×1m, 3×2m, 2×3m, 1×4m, 赤5筒
const ANKAN = tiles("東東東東");
const PON = tiles("白白白");
const R1 = tiles("6s6s7s"); // shimocha's river: 6s twice, 7s once
const R3 = tiles("8s"); // kamicha's river
const INDICATOR = tiles("9s"); // ⇒ dora is 1s

const TY = {
  m1: 0,
  m2: 1,
  m3: 2,
  m4: 3,
  p5: 13,
  s1: 18,
  s6: 23,
  s7: 24,
  s8: 25,
  s9: 26,
  ton: 27,
  haku: 31,
} as const;

/**
 * One complete, internally consistent Observation. Everything the encoder reads
 * is set to a value that is distinguishable from its neighbours: relative seats
 * 1 and 3 have different rivers, seat 2 has the only called meld, the seat wind
 * is 西 (so the one-hot lands in the middle of s18–21), and the indicator's type
 * differs from the dora it names.
 */
function baseObs(over: Partial<Observation> = {}): Observation {
  const ankan: Meld = {
    kind: "ankan",
    who: 0,
    fromWho: 0,
    tiles: ANKAN,
    calledTile: ANKAN[0],
  };
  const pon: Meld = { kind: "pon", who: 2, fromWho: 1, tiles: PON, calledTile: PON[2] };
  return {
    seat: 0,
    kyoku: 1,
    honba: 2,
    kyotaku: 1,
    junme: 6,
    wallRemaining: 42,
    hand: HAND,
    drawn: null,
    melds: [[ankan], [], [pon], []],
    rivers: [[], river(R1), [], river(R3)],
    scores: [25000, 32000, 18000, 25000],
    riichi: [false, true, false, false],
    riichiJunme: [-1, 4, -1, -1],
    doraIndicators: INDICATOR,
    seatWind: 29, // 西 ⇒ s20
    roundWind: 27, // 東場
    akaIds: JANKI.akaIds,
    shanten: 2,
    waits: [],
    ronnable: [],
    katagari: false,
    ukeire: [],
    doraCount: 0,
    furiten: { permanent: false, temporary: true, riichi: false },
    discardInfo: new Map(),
    tsumogiriLock: true,
    danger: new Map(),
    violations: [1, 0, 2, 5],
    legal: [],
    ...over,
  };
}

// --------------------------------------------------------------------------
// dimensions
// --------------------------------------------------------------------------

Deno.test("features: the advertised dimensions are the frozen ones", () => {
  assertEquals(FEATURES.version, 3);
  assertEquals(FEATURES.planes, 48);
  assertEquals(FEATURES.scalars, 42);
  assertEquals(FEATURES.actions, 78);
  assertEquals(TYPES, 34);
  assertEquals(PLANE_LEN, 1632);
  assertEquals(INPUT_LEN, 1674);
  // The v2 prefix the trainer's surgery tool copies, spelled out: the new
  // first-layer columns it zero-fills are [1224..1631] and [1671..1673].
  assertEquals(V2_PLANES * TYPES, 1224);
  assertEquals(V2_PLANES * TYPES + V2_SCALARS, 1263);
});

Deno.test("features: encode returns 1632 plane cells and 42 scalars", () => {
  const e = encode(baseObs());
  assertEquals(e.planes.length, 1632);
  assertEquals(e.scalars.length, 42);
  assert(e.planes instanceof Int8Array);
  assert(e.scalars instanceof Float32Array);
  // The v2 prefix is strictly indicator bits…
  for (let i = 0; i < V2_PLANES * TYPES; i++) {
    assert(e.planes[i] === 0 || e.planes[i] === 1, `plane cell ${i} = ${e.planes[i]}`);
  }
  // …while p36–p44 are small counts/levels, and p45–p47 are bits again.
  for (let i = V2_PLANES * TYPES; i < 45 * TYPES; i++) {
    assert(e.planes[i] >= 0 && e.planes[i] <= 4, `plane cell ${i} = ${e.planes[i]}`);
  }
  for (let i = 45 * TYPES; i < PLANE_LEN; i++) {
    assert(e.planes[i] === 0 || e.planes[i] === 1, `plane cell ${i} = ${e.planes[i]}`);
  }
  for (let i = 0; i < e.scalars.length; i++) {
    assert(Number.isFinite(e.scalars[i]), `scalar ${i} = ${e.scalars[i]}`);
  }
});

// --------------------------------------------------------------------------
// planes, index by index
// --------------------------------------------------------------------------

Deno.test("features: p0–p3 are the own-hand count thresholds", () => {
  const e = encode(baseObs());
  assertEquals(lit(e, 0), [TY.m1, TY.m2, TY.m3, TY.m4, TY.p5], "≥1: every type held");
  assertEquals(lit(e, 1), [TY.m1, TY.m2, TY.m3], "≥2");
  assertEquals(lit(e, 2), [TY.m1, TY.m2], "≥3");
  assertEquals(lit(e, 3), [TY.m1], "≥4");
  // The melded ankan is NOT part of the hand planes.
  assertEquals(plane(e, 0)[TY.ton], 0, "a melded type is not in hand");
});

Deno.test("features: p4 is the drawn tile, and is empty when there is no draw", () => {
  const none = encode(baseObs());
  assertEquals(lit(none, 4), [], "drawn === null ⇒ the whole plane is zero");

  const [drawn] = tiles("5s");
  const drew = encode(baseObs({ drawn }));
  assertEquals(lit(drew, 4), [tileType(drawn)]);
  assertEquals(plane(drew, 4)[tileType(drawn)], 1);
});

Deno.test("features: p5 is redness, and only for the aka actually held", () => {
  const e = encode(baseObs());
  assertEquals(lit(e, 5), [TY.p5], "the 赤5筒 in hand");
  // A plain five sits in p0 exactly where the red one did, but leaves p5 dark.
  const [, , plain] = tiles("0p0p5p");
  const plainHand = encode(baseObs({ hand: [...HAND.slice(0, 10), plain] }));
  assertEquals(lit(plainHand, 5), []);
  assertEquals(plane(plainHand, 0)[TY.p5], 1, "a plain five still counts as a five");
});

Deno.test("features: p6–p13 place rivers by RELATIVE seat, two planes each", () => {
  const e = encode(baseObs());
  assertEquals(lit(e, 6), [], "self: ≥1");
  assertEquals(lit(e, 7), [], "self: ≥2");
  assertEquals(lit(e, 8), [TY.s6, TY.s7], "shimocha: ≥1");
  assertEquals(lit(e, 9), [TY.s6], "shimocha: ≥2 — the doubled 6s");
  assertEquals(lit(e, 10), [], "toimen: ≥1");
  assertEquals(lit(e, 11), []);
  assertEquals(lit(e, 12), [TY.s8], "kamicha: ≥1");
  assertEquals(lit(e, 13), [], "kamicha: ≥2");

  // Rotating the rivers one seat rotates the plane pairs one step.
  const rot = encode(baseObs({ rivers: [river(R1), [], river(R3), []] }));
  assertEquals(lit(rot, 6), [TY.s6, TY.s7]);
  assertEquals(lit(rot, 7), [TY.s6]);
  assertEquals(lit(rot, 10), [TY.s8]);
  assertEquals(lit(rot, 8), []);
});

Deno.test("features: p14–p17 are one meld plane per relative seat", () => {
  const e = encode(baseObs());
  assertEquals(lit(e, 14), [TY.ton], "our own ankan");
  assertEquals(lit(e, 15), []);
  assertEquals(lit(e, 16), [TY.haku], "toimen's pon");
  assertEquals(lit(e, 17), []);
});

Deno.test("features: p18 is the dora, p19 the indicator that named it", () => {
  const e = encode(baseObs());
  assertEquals(lit(e, 19), [TY.s9], "the indicator is 9索");
  assertEquals(lit(e, 18), [TY.s1], "…so the dora is 1索, not 9索");
  assertEquals(doraFromIndicatorType(TY.s9), TY.s1);
  assertEquals(plane(e, 18)[TY.s9], 0, "the indicator's own type is not dora");

  // Honors wrap: 中 indicates 白.
  const chun = tiles("中");
  const w = encode(baseObs({ doraIndicators: chun }));
  assertEquals(lit(w, 19), [33]);
  assertEquals(lit(w, 18), [31]);
});

Deno.test("features: p20/p21 are the seat's own visible-count view", () => {
  const e = encode(baseObs());
  // 1m ×4 (hand), 2m ×3 (hand), 東 ×4 (own ankan), 白 ×3 (toimen's pon).
  assertEquals(lit(e, 20), [TY.m1, TY.m2, TY.ton, TY.haku], "≥3 visible");
  assertEquals(lit(e, 21), [TY.m1, TY.ton], "all four accounted for");
});

// --------------------------------------------------------------------------
// v2 planes
// --------------------------------------------------------------------------

/** A claim decision: 白 is on the table, we may pon it or pass. */
function claimObs(over: Partial<Observation> = {}): Observation {
  const [called] = tiles("白");
  const pair = tiles("白白");
  return baseObs({
    claimTile: called,
    legal: [{ t: "pon", tiles: [pair[0], pair[1]], called }, { t: "pass" }],
    ...over,
  });
}

Deno.test("features: p22 is the claimed tile — from the field, or from the call", () => {
  const turn = encode(baseObs());
  assertEquals(lit(turn, 22), [], "a turn decision offers no tile");

  const claim = encode(claimObs());
  assertEquals(lit(claim, 22), [TY.haku]);

  // An Observation built without the field still encodes: the pon names it.
  const legacy = claimObs();
  delete legacy.claimTile;
  assertEquals(lit(encode(legacy), 22), [TY.haku], "recovered from the pon's `called`");

  // ...but ron/pass carry no tile, so there is nothing to recover from.
  const ronOnly = baseObs({ legal: [{ t: "ron" }, { t: "pass" }] });
  assertEquals(lit(encode(ronOnly), 22), []);
});

Deno.test("features: p23–p25 are the other seats' last discards", () => {
  const e = encode(baseObs());
  assertEquals(lit(e, 23), [TY.s7], "shimocha discarded 6s,6s,7s — the LAST one");
  assertEquals(lit(e, 24), [], "toimen's river is empty");
  assertEquals(lit(e, 25), [TY.s8], "kamicha's only discard");
  // Our own last discard is deliberately absent: p6/p7 already carry our river.
  const self = encode(baseObs({ rivers: [river(R1), [], [], []] }));
  assertEquals(lit(self, 23), [], "nothing shifts our river into a neighbour's plane");
});

Deno.test("features: p26–p28 bucket the danger levels, 安全 lighting none", () => {
  const danger = new Map([
    [TY.m1, assessment("危険度低")],
    [TY.m2, assessment("危険度中")],
    [TY.m3, assessment("危険度高")],
    [TY.m4, assessment("安全")],
  ]);
  const e = encode(baseObs({ danger }));
  assertEquals(lit(e, 26), [TY.m1], "低");
  assertEquals(lit(e, 27), [TY.m2], "中");
  assertEquals(lit(e, 28), [TY.m3], "高");
  for (const p of [26, 27, 28]) {
    assertEquals(plane(e, p)[TY.m4], 0, `安全 must not light p${p}`);
  }
  const quiet = encode(baseObs());
  assertEquals([lit(quiet, 26), lit(quiet, 27), lit(quiet, 28)], [[], [], []]);
});

Deno.test("features: p29 is the legal discard set", () => {
  const [a, b] = tiles("1m2m");
  const e = encode(baseObs({
    legal: [
      { t: "discard", tile: a, riichi: false, tsumogiri: false },
      { t: "discard", tile: b, riichi: true, tsumogiri: false },
      { t: "tsumo" },
    ],
  }));
  assertEquals(lit(e, 29), [TY.m1, TY.m2], "riichi/tsumogiri variants collapse onto the type");
  assertEquals(lit(encode(baseObs()), 29), [], "no legal discards ⇒ dark");
});

Deno.test("features: p30–p32 are what each discard leaves behind", () => {
  const [m1, m2, m3, m4] = tiles("1m2m3m4m");
  const discardInfo = new Map([
    [m1, { shanten: 0, katagari: true, yakuless: false }],
    [m2, { shanten: 0, katagari: false, yakuless: true }],
    [m3, { shanten: 1, katagari: false, yakuless: false }],
    [m4, { shanten: 2, katagari: false, yakuless: false }],
  ]);
  const e = encode(baseObs({ discardInfo }));
  assertEquals(lit(e, 30), [TY.m1, TY.m2], "both tenpai discards tie for best");
  assertEquals(lit(e, 31), [TY.m1], "片和了り");
  assertEquals(lit(e, 32), [TY.m2], "役なし聴牌");

  // "Best" is relative: with no tenpai available, the 1-shanten discard wins.
  const worse = new Map([
    [m3, { shanten: 1, katagari: false, yakuless: false }],
    [m4, { shanten: 2, katagari: false, yakuless: false }],
  ]);
  assertEquals(lit(encode(baseObs({ discardInfo: worse })), 30), [TY.m3]);
  assertEquals(lit(encode(baseObs()), 30), [], "an empty discardInfo lights nothing");
});

Deno.test("features: p33/p34 are the waits and the ukeire", () => {
  const e = encode(baseObs({
    waits: [TY.s1, TY.s6],
    ukeire: [{ type: TY.m1, live: 0 }, { type: TY.s6, live: 3 }],
  }));
  assertEquals(lit(e, 33), [TY.s1, TY.s6]);
  assertEquals(lit(e, 34), [TY.m1, TY.s6], "ukeire is a separate plane, live count and all");
});

Deno.test("features: p35 is the aka seen OUTSIDE our concealed hand", () => {
  const base = encode(baseObs());
  assertEquals(lit(base, 35), [], "our own 赤5筒 is in hand, so p5 has it and p35 does not");

  // 雀鬼流 has exactly two aka, both 5筒; the fixture holds one, so the other is
  // the only red that can show up anywhere else.
  const [, other, plain, plain2] = tiles("0p0p5p5p"); // ids 52,53 are the reds
  const inRiver = encode(baseObs({ rivers: [[], river([other]), [], []] }));
  assertEquals(lit(inRiver, 35), [TY.p5], "shimocha cut the other 赤5筒");
  assertEquals(lit(inRiver, 5), [TY.p5], "…and p5 still means only what WE hold");

  const pon: Meld = {
    kind: "pon",
    who: 2,
    fromWho: 1,
    tiles: [other, plain, plain2],
    calledTile: other,
  };
  const inMeld = encode(baseObs({ melds: [[], [], [pon], []] }));
  assertEquals(lit(inMeld, 35), [TY.p5], "a red inside a meld is public too");

  // A plain five outside the hand is not red.
  const plainOut = encode(baseObs({ rivers: [[], river([plain]), [], []] }));
  assertEquals(lit(plainOut, 35), []);
});

// --------------------------------------------------------------------------
// v3 planes — the per-opponent block (relative seats 1 / 2 / 3)
// --------------------------------------------------------------------------

const M5 = 4; // 五萬
const P1 = 9; // 一筒

/** A pon of three copies of one type, called from `fromWho`. */
function ponOf(spec: string, who: number, fromWho: number): Meld {
  const t = tiles(spec);
  return { kind: "pon", who, fromWho, tiles: t, calledTile: t[2] };
}

Deno.test("features: p36–p38 count what the opponents' melds are MADE OF", () => {
  const e = encode(baseObs());
  assertEquals(nonzero(e, 36), [], "shimocha has no melds");
  assertEquals(nonzero(e, 37), [[TY.haku, 3]], "toimen's pon of 白 is three copies");
  assertEquals(nonzero(e, 38), [], "kamicha has no melds");
  // Our OWN ankan is p14's business; the opponent block must not see it.
  for (const p of [36, 37, 38]) assertEquals(plane(e, p)[TY.ton], 0, `own ankan leaked into p${p}`);

  // A pon of 5m by shimocha ⇒ 3 at 五萬 in p36, and nowhere else.
  const pon5 = encode(baseObs({ melds: [[], [ponOf("555m", 1, 0)], [], []] }));
  assertEquals(nonzero(pon5, 36), [[M5, 3]]);
  assertEquals([nonzero(pon5, 37), nonzero(pon5, 38)], [[], []]);

  // Relative order: the same meld one seat further round moves one plane on.
  const pon5b = encode(baseObs({ melds: [[], [], [], [ponOf("555m", 3, 2)]] }));
  assertEquals(nonzero(pon5b, 38), [[M5, 3]]);
  assertEquals(nonzero(pon5b, 36), []);

  // A kan is four, and an ankan counts too: its composition is announced even
  // though the hand it sits in stays closed.
  const kanTiles = tiles("5555m");
  const ankan: Meld = {
    kind: "ankan",
    who: 1,
    fromWho: 1,
    tiles: kanTiles,
    calledTile: kanTiles[0],
  };
  assertEquals(nonzero(encode(baseObs({ melds: [[], [ankan], [], []] })), 36), [[M5, 4]]);

  // Two melds sharing a type add up, and the cell is clamped at four.
  const chi = tiles("345m");
  const chiMeld: Meld = { kind: "chi", who: 1, fromWho: 0, tiles: chi, calledTile: chi[0] };
  const both = encode(baseObs({ melds: [[], [ankan, chiMeld], [], []] }));
  assertEquals(plane(both, 36)[M5], 4, "4 + 1 clamps, never overflows the Int8 bucket");
  assertEquals(plane(both, 36)[TY.m3], 1, "the chi's other tiles are counted once each");
});

Deno.test("features: p39–p41 count only the opponents' 手出し", () => {
  const [a, b, c, d] = tiles("1p1p2p3p");
  const rivers = [
    [],
    markedRiver([
      { tile: a }, // tedashi 1p
      { tile: b, tsumogiri: true }, // tsumogiri 1p — not counted
      { tile: c }, // tedashi 2p
    ]),
    [],
    markedRiver([{ tile: d, tsumogiri: true }]),
  ];
  const e = encode(baseObs({ rivers }));
  assertEquals(nonzero(e, 39), [[P1, 1], [P1 + 1, 1]], "shimocha: the two hand-cuts");
  assertEquals(nonzero(e, 40), [], "toimen's river is empty");
  assertEquals(nonzero(e, 41), [], "kamicha's only discard was a tsumogiri");
  // The river planes p8–p13 still see every entry, tsumogiri or not.
  assertEquals(lit(e, 8), [P1, P1 + 1], "p8 is unchanged by the 手出し split");

  // The fixture's rivers are all tedashi, so they count in full.
  const base = encode(baseObs());
  assertEquals(nonzero(base, 39), [[TY.s6, 2], [TY.s7, 1]], "6s twice, 7s once");
  assertEquals(nonzero(base, 41), [[TY.s8, 1]]);

  // Five hand-cuts of one type cannot happen, but the clamp is still the rule.
  const many = tiles("1p1p1p1p").map((tile) => ({ tile }));
  const capped = encode(baseObs({ rivers: [[], markedRiver([...many, { tile: a }]), [], []] }));
  assertEquals(plane(capped, 39)[P1], 4, "capped at 4");
});

Deno.test("features: p42–p44 are the danger EACH opponent poses, 0–3", () => {
  // Absolute seats 1/2/3 with obs.seat = 0 ⇒ relative 1/2/3 ⇒ p42/p43/p44.
  const danger = new Map([
    [TY.m1, assessSeats([[1, "危険度高"], [3, "危険度低"]])],
    [TY.m2, assessSeats([[2, "危険度中"]])],
  ]);
  const e = encode(baseObs({ danger }));
  assertEquals(nonzero(e, 42), [[TY.m1, 3]], "shimocha: 高");
  assertEquals(nonzero(e, 43), [[TY.m2, 2]], "toimen: 中");
  assertEquals(nonzero(e, 44), [[TY.m1, 1]], "kamicha: 低");
  // A seat the tile is genbutsu against simply has no detail, which reads as
  // 安全 = 0 — the same zero an unassessed type gets.
  assertEquals(plane(e, 43)[TY.m1], 0, "no detail for toimen ⇒ 安全");
  assertEquals(cells(encode(baseObs()), 42).reduce((x, y) => x + y, 0), 0, "no threats ⇒ dark");

  // 安全 carried EXPLICITLY is still 0, matching p26–p28's "安全 lights none".
  const safe = encode(baseObs({ danger: new Map([[TY.m1, assessSeats([[1, "安全"]])]]) }));
  assertEquals(nonzero(safe, 42), []);

  // The detail's `seat` is ABSOLUTE, so it folds into relative order: seat 2
  // observing a threat from absolute seat 1 sees it as kamicha (relative 3).
  const rel = encode(
    baseObs({ seat: 2, danger: new Map([[TY.m1, assessSeats([[1, "危険度高"]])]]) }),
  );
  assertEquals(nonzero(rel, 44), [[TY.m1, 3]], "abs 1 seen from seat 2 is relative 3");
  assertEquals([nonzero(rel, 42), nonzero(rel, 43)], [[], []]);

  // The v2 summary planes are untouched by the split.
  assertEquals(lit(e, 28), [TY.m1], "p28 still carries the WORST level");
  assertEquals(lit(e, 27), [TY.m2]);
});

Deno.test("features: p45–p47 are each opponent's riichi declaration tile", () => {
  const base = encode(baseObs());
  for (const p of [45, 46, 47]) {
    assertEquals(lit(base, p), [], `p${p}: the fixture marks no declaration tile`);
  }

  const [x, y, z] = tiles("1p2p3p");
  const declared = encode(baseObs({
    rivers: [
      [],
      markedRiver([{ tile: x }, { tile: y, riichiDeclare: true }, { tile: z }]),
      [],
      [],
    ],
  }));
  assertEquals(lit(declared, 45), [P1 + 1], "the sideways tile, not the first or the last");
  assertEquals(plane(declared, 45)[P1 + 1], 1, "one-hot, value 1");
  assertEquals([lit(declared, 46), lit(declared, 47)], [[], []], "only the declaring seat");

  // Our own declaration is never in the opponent block.
  const own = encode(
    baseObs({ rivers: [markedRiver([{ tile: y, riichiDeclare: true }]), [], [], []] }),
  );
  assertEquals([lit(own, 45), lit(own, 46), lit(own, 47)], [[], [], []]);

  // Relative order again: kamicha's declaration lands in p47.
  const kami = encode(baseObs({
    rivers: [[], [], [], markedRiver([{ tile: z, riichiDeclare: true }])],
  }));
  assertEquals(lit(kami, 47), [P1 + 2]);
});

// --------------------------------------------------------------------------
// scalars, index by index
// --------------------------------------------------------------------------

Deno.test("features: s0–s11 are the per-seat score / riichi / riichi-junme block", () => {
  const { scalars: s } = encode(baseObs());
  assertAlmostEquals(s[0], 1, 1e-6);
  assertAlmostEquals(s[1], 32000 / 25000, 1e-6);
  assertAlmostEquals(s[2], 18000 / 25000, 1e-6);
  assertAlmostEquals(s[3], 1, 1e-6);
  assertEquals([s[4], s[5], s[6], s[7]], [0, 1, 0, 0], "only shimocha is in riichi");
  assertEquals(s[8], 0, "-1 (no riichi) encodes as 0, not as a negative");
  assertAlmostEquals(s[9], 4 / 18, 1e-6);
  assertEquals([s[10], s[11]], [0, 0]);
});

Deno.test("features: s12/s13 are junme and wall remaining, both normalised", () => {
  const { scalars: s } = encode(baseObs());
  assertAlmostEquals(s[12], 6 / 18, 1e-6);
  assertAlmostEquals(s[13], 42 / 70, 1e-6);
  const late = encode(baseObs({ junme: 18, wallRemaining: 0 })).scalars;
  assertEquals([late[12], late[13]], [1, 0]);
});

Deno.test("features: s14–s16 are kyoku, honba and kyotaku, the latter two clamped", () => {
  const { scalars: s } = encode(baseObs());
  assertAlmostEquals(s[14], 1 / 8, 1e-6);
  assertAlmostEquals(s[15], 2 / 8, 1e-6);
  assertAlmostEquals(s[16], 1 / 4, 1e-6);
  const big = encode(baseObs({ honba: 40, kyotaku: 9 })).scalars;
  assertEquals([big[15], big[16]], [1, 1], "clamped, never above 1");
});

Deno.test("features: s17 is the round wind, s18–21 the seat wind one-hot", () => {
  const east = encode(baseObs()).scalars;
  assertEquals(east[17], 1, "東場");
  const south = encode(baseObs({ roundWind: 28 })).scalars;
  assertEquals(south[17], 0, "南場");

  // 西家 in the fixture ⇒ the third of the four slots.
  assertEquals([east[18], east[19], east[20], east[21]], [0, 0, 1, 0]);
  for (let w = 0; w < 4; w++) {
    const s = encode(baseObs({ seatWind: 27 + w })).scalars;
    assertEquals(
      [s[18], s[19], s[20], s[21]].reduce((a, b) => a + b, 0),
      1,
      `seat wind ${27 + w} must set exactly one slot`,
    );
    assertEquals(s[18 + w], 1, `seat wind ${27 + w} ⇒ s${18 + w}`);
  }
});

Deno.test("features: s22 is shanten, floored at 0 for a complete hand", () => {
  assertAlmostEquals(encode(baseObs()).scalars[22], 2 / 8, 1e-6);
  assertEquals(encode(baseObs({ shanten: 0 })).scalars[22], 0, "tenpai");
  assertEquals(encode(baseObs({ shanten: -1 })).scalars[22], 0, "和了形 does not go negative");
});

Deno.test("features: s23–s25 are the three furiten flags", () => {
  const { scalars: s } = encode(baseObs());
  assertEquals([s[23], s[24], s[25]], [0, 1, 0], "temporary only");
  const all = encode(
    baseObs({ furiten: { permanent: true, temporary: true, riichi: true } }),
  ).scalars;
  assertEquals([all[23], all[24], all[25]], [1, 1, 1]);
});

Deno.test("features: s26 is the ドラ切り tsumogiri lock", () => {
  assertEquals(encode(baseObs()).scalars[26], 1);
  assertEquals(encode(baseObs({ tsumogiriLock: false })).scalars[26], 0);
});

Deno.test("features: s27–s30 are the relative violation counts, clamped at four", () => {
  const { scalars: s } = encode(baseObs());
  assertAlmostEquals(s[27], 1 / 4, 1e-6);
  assertEquals(s[28], 0);
  assertAlmostEquals(s[29], 2 / 4, 1e-6);
  assertEquals(s[30], 1, "five violations still reads as a full bar");
});

Deno.test("features: s31/s32 are 門前 and the open-meld count", () => {
  const { scalars: s } = encode(baseObs());
  assertEquals(s[31], 1, "a hand whose only meld is an ankan is still 門前");
  assertAlmostEquals(s[32], 1 / 4, 1e-6);

  const bare = encode(baseObs({ melds: [[], [], [], []] })).scalars;
  assertEquals([bare[31], bare[32]], [1, 0], "no melds at all is vacuously 門前");

  const chiTiles = tiles("123m");
  const chi: Meld = {
    kind: "chi",
    who: 0,
    fromWho: 3,
    tiles: chiTiles,
    calledTile: chiTiles[0],
  };
  const open = encode(baseObs({ melds: [[chi], [], [], []] })).scalars;
  assertEquals(open[31], 0, "a chi opens the hand");
  assertAlmostEquals(open[32], 1 / 4, 1e-6);
});

Deno.test("features: s33 is the dora count, clamped at eight", () => {
  assertEquals(encode(baseObs()).scalars[33], 0);
  assertAlmostEquals(encode(baseObs({ doraCount: 3 })).scalars[33], 3 / 8, 1e-6);
  assertEquals(encode(baseObs({ doraCount: 12 })).scalars[33], 1, "clamped");
});

Deno.test("features: s34–s37 place the dealer by RELATIVE seat", () => {
  // 東二局 with the fixture's seat 0 ⇒ the dealer is absolute seat 1, which from
  // seat 0 is shimocha: relative 1.
  const s = encode(baseObs()).scalars;
  assertEquals([s[34], s[35], s[36], s[37]], [0, 1, 0, 0]);

  // Every (kyoku, seat) pair lights exactly one slot, and it is the right one.
  for (let kyoku = 0; kyoku < 8; kyoku++) {
    for (const seat of [0, 1, 2, 3] as const) {
      const v = encode(baseObs({ kyoku, seat })).scalars;
      const slots = [v[34], v[35], v[36], v[37]];
      assertEquals(slots.reduce((a, b) => a + b, 0), 1, `kyoku ${kyoku} seat ${seat}`);
      const rel = ((kyoku % 4) - seat + 4) % 4;
      assertEquals(slots[rel], 1, `kyoku ${kyoku} seat ${seat} ⇒ relative dealer ${rel}`);
    }
  }
  // 親 itself reads as relative 0.
  assertEquals(encode(baseObs({ kyoku: 5, seat: 1 })).scalars[34], 1);
});

Deno.test("features: s38 marks a claim decision, and p22 comes with it", () => {
  assertEquals(encode(baseObs()).scalars[38], 0, "a turn decision has no pass");
  const claim = encode(claimObs());
  assertEquals(claim.scalars[38], 1);
  assertEquals(lit(claim, 22), [TY.haku], "the two v2 claim features travel together");
  // A ron/pass claim is still a claim, even though no tile can be recovered.
  const ron = encode(baseObs({ legal: [{ t: "ron" }, { t: "pass" }] }));
  assertEquals(ron.scalars[38], 1);
});

Deno.test("features: s39–s41 are WHEN each opponent declared riichi", () => {
  const { scalars: s } = encode(baseObs());
  assertAlmostEquals(s[39], 4 / 18, 1e-6, "shimocha declared on 巡目 4");
  assertEquals([s[40], s[41]], [0, 0], "nobody else declared");
  // Deliberately the same numbers as s9–s11: the v3 block is appended, never a
  // reorder of the v2 prefix, so the pair with p45–p47 is spelled out again.
  assertEquals([s[39], s[40], s[41]], [s[9], s[10], s[11]]);
  // Our OWN declaration (relative 0) has no slot here — it is s8's.
  const self = encode(
    baseObs({ riichi: [true, false, false, false], riichiJunme: [7, -1, -1, -1] }),
  );
  assertEquals([self.scalars[39], self.scalars[40], self.scalars[41]], [0, 0, 0]);
  assertAlmostEquals(self.scalars[8], 7 / 18, 1e-6);

  const all = encode(baseObs({ riichiJunme: [-1, 1, 9, 18] })).scalars;
  assertAlmostEquals(all[39], 1 / 18, 1e-6);
  assertAlmostEquals(all[40], 9 / 18, 1e-6);
  assertEquals(all[41], 1, "a 18巡目 declaration is a full bar");
});

// --------------------------------------------------------------------------
// flatten
// --------------------------------------------------------------------------

Deno.test("features: flatten is planes ++ scalars, 1674 wide", () => {
  const e = encode(baseObs());
  const f = flatten(e);
  assertEquals(f.length, INPUT_LEN);
  assertEquals(f.length, 1674);
  assert(f instanceof Float32Array);
  for (let i = 0; i < PLANE_LEN; i++) assertEquals(f[i], e.planes[i], `plane cell ${i}`);
  for (let j = 0; j < FEATURES.scalars; j++) {
    assertEquals(f[PLANE_LEN + j], e.scalars[j], `scalar ${j}`);
  }
});

// --------------------------------------------------------------------------
// whole-buffer digest
// --------------------------------------------------------------------------

/**
 * What v2 emitted for `baseObs()` — KEPT, and now checked against the v3
 * output's PREFIX. See the prefix-guard test below.
 */
const PLANE_DIGEST = 816561413;
const SCALAR_DIGEST = 479078765;

/** What v3 emits for `baseObs()` over the FULL buffers. */
const V3_PLANE_DIGEST = 96848848;
const V3_SCALAR_DIGEST = 4175675587;

function fnv1a(bytes: Uint8Array): number {
  let h = 0x811c9dc5;
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** Float32 values as their exact little-endian bytes — no rounding anywhere. */
function floatBytes(f: Float32Array): Uint8Array {
  const out = new Uint8Array(f.length * 4);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < f.length; i++) dv.setFloat32(i * 4, f[i], true);
  return out;
}

/**
 * A change detector over the FULL buffers, not just the cells named above.
 *
 * These two numbers are not derived from anything — they are what v3 emits for
 * the fixture. REGENERATE THEM DELIBERATELY: any edit that moves them means the
 * wire format the trainer reshapes has changed, so `FEATURES.version` must be
 * bumped on both sides in the same commit.
 */
Deno.test("features: the v3 encoding of the fixture digests to the frozen values", () => {
  const e = encode(baseObs());
  const planeBytes = new Uint8Array(e.planes.buffer, e.planes.byteOffset, e.planes.byteLength);
  assertEquals(fnv1a(planeBytes), V3_PLANE_DIGEST, "plane digest");
  assertEquals(fnv1a(floatBytes(e.scalars)), V3_SCALAR_DIGEST, "scalar digest");
});

/**
 * THE PREFIX GUARD. v3 appends; it does not reorder. The Python surgery tool
 * that widens a trained first layer copies columns [0..1262] straight across
 * and zero-fills [1224..1631] and [1671..1673], so every v2 cell must still be
 * exactly where — and what — it was. The two constants below are the ORIGINAL
 * v2 digests, unchanged; recomputing them over the first 36 planes / first 39
 * scalars of the v3 buffers is what proves the prefix survived.
 *
 * If this test fails, the surgery mapping is wrong and no amount of retraining
 * will fix the resulting network: fix the encoder instead.
 */
Deno.test("features: the v2 layout survives untouched as a PREFIX of v3", () => {
  const e = encode(baseObs());
  const prefix = e.planes.slice(0, V2_PLANES * TYPES);
  assertEquals(prefix.length, 1224);
  assertEquals(
    fnv1a(new Uint8Array(prefix.buffer, prefix.byteOffset, prefix.byteLength)),
    PLANE_DIGEST,
    "the first 36 planes still digest to the frozen v2 value",
  );
  const sPrefix = e.scalars.slice(0, V2_SCALARS);
  assertEquals(sPrefix.length, 39);
  assertEquals(
    fnv1a(floatBytes(sPrefix)),
    SCALAR_DIGEST,
    "the first 39 scalars still digest to the frozen v2 value",
  );

  // …and the same holds for the flattened vector the network actually eats:
  // its first 1263 entries are the v2 input, cell for cell.
  const f = flatten(e);
  for (let i = 0; i < V2_PLANES * TYPES; i++) assertEquals(f[i], prefix[i], `plane cell ${i}`);
  for (let j = 0; j < V2_SCALARS; j++) {
    assertEquals(f[PLANE_LEN + j], sPrefix[j], `scalar ${j}`);
  }
});

// --------------------------------------------------------------------------
// the oracle planes (hidden state — trainer-side only)
// --------------------------------------------------------------------------

/**
 * `encodeOracle` reads a live Table, so its fixture has to be a real round.
 * Every decision of one hanchan is checked mid-flight through the driver's
 * `tableRef` seam — which is also what the recorder uses, so a break in the
 * seam shows up here rather than as silently missing oracle data.
 */
function driveOracle(
  seed: number,
  check: (t: Table, obs: Observation) => void,
  kind: "r" | "h" = "r",
): number {
  const ref: { t: Table | null } = { t: null };
  let n = 0;
  const policies: SyncPolicy[] = SEATS.map((s) => {
    const inner = kind === "h"
      ? new HeuristicPolicy(`H${s}`, seed * 4 + s)
      : new RandomPolicy(`R${s}`, seed * 4 + s);
    return {
      name: inner.name,
      sync: true,
      decide(obs: Observation): Action {
        assert(ref.t !== null, "tableRef is null during a decision");
        check(ref.t, obs);
        n++;
        return inner.decide(obs);
      },
    };
  });
  runMatchSync(policies, {
    seed,
    cfg: JANKI,
    dojo: DOJO_HEADLESS,
    scorer,
    tableRef: ref,
    ...makeDojoHooks(DOJO_HEADLESS),
  });
  assertEquals(ref.t, null, "the match ended without releasing the table");
  assert(n > 0, "no decisions were made");
  return n;
}

/** The 34-long stretch oracle plane `p` occupies. */
function oplane(o: Int8Array, p: number): Int8Array {
  return o.slice(p * TYPES, (p + 1) * TYPES);
}

Deno.test("oracle: 5 planes of 34 counts, and the shape the trainer reshapes", () => {
  assertEquals(ORACLE_PLANES, 5);
  assertEquals(ORACLE_LEN, 170);
  driveOracle(11, (t, obs) => {
    const { oplanes, oppShanten } = encodeOracle(t, obs.seat);
    assert(oplanes instanceof Int8Array);
    assertEquals(oplanes.length, ORACLE_LEN);
    assertEquals(oppShanten.length, 3);
    for (const v of oplanes) assert(v >= 0 && v <= 4, `oracle cell out of range: ${v}`);
  });
});

Deno.test("oracle: o0–o2 are the opponents' CONCEALED hands, in relative order", () => {
  driveOracle(12, (t, obs) => {
    const { oplanes } = encodeOracle(t, obs.seat);
    for (let r = 1; r < 4; r++) {
      const s = ((obs.seat + r) % 4) as Seat;
      // Straight from the table, not through the encoder's helper.
      const want = new Array<number>(TYPES).fill(0);
      for (const id of t.hands[s]) want[tileType(id)]++;
      assertEquals([...oplane(oplanes, r - 1)], want, `relative seat ${r}`);
      // Melds are public (the policy already has them in p14–p17), so a called
      // tile must not show up here a second time: a resting hand plus its melds
      // is 13 tiles, one more for each kan's fourth tile.
      const kans = t.melds[s].filter((m) => m.tiles.length === 4).length;
      const meldTiles = t.melds[s].reduce((n, m) => n + m.tiles.length, 0);
      assertEquals(
        t.hands[s].length + meldTiles,
        13 + kans,
        `relative seat ${r}: concealed+meld tile count`,
      );
    }
  });
});

Deno.test("oracle: o3 is the hidden remainder — every type sums to exactly four", () => {
  driveOracle(13, (t, obs) => {
    const { oplanes } = encodeOracle(t, obs.seat);
    const own = new Array<number>(TYPES).fill(0);
    for (const id of t.hands[obs.seat]) own[tileType(id)]++;
    const table = new Array<number>(TYPES).fill(0);
    for (const s of SEATS) {
      // A called river entry lives in the meld that took it; counting both
      // would put five copies of a type on the table.
      for (const e of t.board.rivers[s]) if (e.calledBy === undefined) table[tileType(e.tile)]++;
      for (const m of t.melds[s]) for (const id of m.tiles) table[tileType(id)]++;
    }
    for (const ind of t.indicators) table[tileType(ind)]++;

    for (let ty = 0; ty < TYPES; ty++) {
      const opp = oplanes[ty] + oplanes[TYPES + ty] + oplanes[2 * TYPES + ty];
      const hidden = oplanes[3 * TYPES + ty];
      assertEquals(
        opp + own[ty] + table[ty] + hidden,
        4,
        `type ${ty}: ${opp}+${own[ty]}+${table[ty]}+${hidden}`,
      );
    }
  });
});

/**
 * The same remainder, recomputed from the WALL rather than by subtraction: the
 * tiles the wall has not yet handed out are indices 0..135-liveTaken, minus the
 * dora indicators it revealed and the rinshan slots it already served. The two
 * routes share nothing but the tile ids, so a bug in either one shows up as a
 * mismatch instead of as two agreeing copies.
 */
function wallHidden(t: Table): number[] {
  const RINSHAN_ORD = [1, 0, 3, 2]; // wall.ts's private order, pinned here on purpose
  const kans = t.wall.kanCount;
  const drawn = DRAWS_PER_ROUND - t.wall.remaining; // live + rinshan, post-deal
  const liveTaken = 52 + drawn - kans; // the 52 dealt, plus every live draw
  const gone = new Set<number>();
  for (let k = 0; k < t.wall.indicatorCount; k++) gone.add(5 + 2 * k);
  for (let k = 0; k < kans; k++) gone.add(RINSHAN_ORD[k]);
  const c = new Array<number>(TYPES).fill(0);
  for (let i = 0; i <= 135 - liveTaken; i++) {
    if (!gone.has(i)) c[tileType(t.wall.tiles[i])]++;
  }
  return c;
}

Deno.test("oracle: o3 agrees with the wall's own untouched tiles", () => {
  driveOracle(14, (t, obs) => {
    const { oplanes } = encodeOracle(t, obs.seat);
    assertEquals([...oplane(oplanes, 3)], wallHidden(t));
  });
});

Deno.test("oracle: o4 is the ura indicators, one count per type", () => {
  let sawUra = false;
  driveOracle(15, (t, obs) => {
    const { oplanes } = encodeOracle(t, obs.seat);
    const want = new Array<number>(TYPES).fill(0);
    for (const id of t.wall.uraIndicators()) want[tileType(id)]++;
    assertEquals([...oplane(oplanes, 4)], want);
    const total = want.reduce((a, b) => a + b, 0);
    assertEquals(total, t.wall.indicatorCount, "one ura per revealed indicator");
    if (total > 0) sawUra = true;
    // Ura tiles are dead-wall tiles nobody has seen, so they are part of the
    // hidden remainder too — never subtracted from it.
    for (let ty = 0; ty < TYPES; ty++) {
      if (want[ty] > 0) assert(oplanes[3 * TYPES + ty] >= want[ty], `ura type ${ty} not hidden`);
    }
  });
  assert(sawUra, "the fixture never revealed an indicator");
});

// Heuristic seats, not random ones: the labels are only interesting if hands
// actually advance, and random play almost never reaches tenpai at all.
Deno.test("oracle: sh is each opponent's shanten, raw (−1 = complete hand)", () => {
  let sawTenpai = false;
  driveOracle(16, (t, obs) => {
    const { oplanes, oppShanten } = encodeOracle(t, obs.seat);
    for (let r = 1; r < 4; r++) {
      const s = ((obs.seat + r) % 4) as Seat;
      // observe.ts's convention: called melds fill set slots, ankan stays 門前.
      const want = shanten(countsFromTiles(t.hands[s]), t.melds[s].length, t.isMenzen(s));
      assertEquals(oppShanten[r - 1], want, `relative seat ${r}`);
      // …and the label must describe the hand the planes actually carry.
      assertEquals(
        shanten([...oplane(oplanes, r - 1)], t.melds[s].length, t.isMenzen(s)),
        want,
        `relative seat ${r}: label vs planes`,
      );
      assert(Number.isInteger(want) && want >= -1 && want <= 8, `shanten ${want}`);
      if (want <= 0) sawTenpai = true;
    }
  }, "h");
  assert(sawTenpai, "the fixture never had an opponent at tenpai");
});

Deno.test("oracle: the policy encoder is untouched by it", () => {
  // Belt and braces against the one thing that must not happen: encodeOracle
  // must not reach into (or perturb) the frozen policy features.
  driveOracle(17, (t, obs) => {
    const before = encode(obs);
    encodeOracle(t, obs.seat);
    const after = encode(obs);
    assertEquals([...after.planes], [...before.planes]);
    assertEquals([...after.scalars], [...before.scalars]);
  });
});

// --------------------------------------------------------------------------
// the v3 block against a real hanchan
// --------------------------------------------------------------------------

/**
 * The hand-built fixture pins the mapping; this pins the INVARIANTS, over every
 * decision of a full match. `driveOracle` is reused only for its driver seam —
 * nothing here reads the Table, so these are properties of the encoding alone.
 */
Deno.test("features: the v3 opponent planes conserve what the Observation says", () => {
  let sawMeld = false, sawTedashi = false, sawRiichi = false, sawDanger = false;
  driveOracle(21, (_t, obs) => {
    const e = encode(obs);

    for (let r = 1; r < 4; r++) {
      // p36–p38: every meld tile of that opponent, counted exactly once.
      const meldTiles = (obs.melds[r] ?? []).reduce((n, m) => n + m.tiles.length, 0);
      const sum = cells(e, 35 + r).reduce((a, b) => a + b, 0);
      assertEquals(sum, meldTiles, `p${35 + r}: meld tile conservation`);
      if (meldTiles > 0) sawMeld = true;

      // p39–p41: the 手出し subset of the river, never more than the river.
      const river = obs.rivers[r] ?? [];
      const tedashi = river.filter((x) => !x.tsumogiri).length;
      const tsum = cells(e, 38 + r).reduce((a, b) => a + b, 0);
      assertEquals(tsum, tedashi, `p${38 + r}: 手出し conservation`);
      assert(tsum <= river.length, `p${38 + r}: ${tsum} > river ${river.length}`);
      if (tedashi > 0) sawTedashi = true;

      // p42–p44: a level, never anything else.
      for (const v of cells(e, 41 + r)) assert(v >= 0 && v <= 3, `p${41 + r}: level ${v}`);

      // p45–p47: at most one declaration tile, and it is a 1.
      const hot = nonzero(e, 44 + r);
      assert(hot.length <= 1, `p${44 + r}: ${hot.length} declaration tiles`);
      if (hot.length === 1) {
        assertEquals(hot[0][1], 1, `p${44 + r}: one-hot value`);
        assert(obs.riichi[r], `p${44 + r}: declared without being in riichi`);
        const mark = (obs.rivers[r] ?? []).find((x) => x.riichiDeclare);
        assert(mark !== undefined, `p${44 + r}: no marked river entry`);
        assertEquals(hot[0][0], tileType(mark.tile), `p${44 + r}: the marked tile's type`);
        sawRiichi = true;
      }
    }

    // The summary bucket p26–p28 is the WORST of the three per-seat levels.
    for (const [ty, d] of obs.danger) {
      const per = [42, 43, 44].map((p) => plane(e, p)[ty]);
      const worst = Math.max(...per);
      const summary = { "安全": 0, "危険度低": 1, "危険度中": 2, "危険度高": 3 }[d.level];
      assertEquals(worst, summary, `type ${ty}: per-seat max vs the summary level`);
      if (worst > 0) sawDanger = true;
    }
  }, "h");
  assert(sawMeld, "the fixture never saw an opponent meld");
  assert(sawTedashi, "the fixture never saw an opponent 手出し");
  assert(sawRiichi, "the fixture never saw an opponent riichi");
  assert(sawDanger, "the fixture never assessed a dangerous tile");
});
