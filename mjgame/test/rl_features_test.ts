// Feature encoding v2. The layout is a FROZEN contract shared with the trainer,
// so this file pins it twice over: once index by index against a single
// hand-built Observation (the `baseObs` pattern from heuristic_test.ts), and
// once as an FNV-1a digest of the complete buffers, which catches any reorder
// the named assertions happen not to look at.

import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import type { Meld, Tile } from "mjrender/model.ts";
import type { DangerAssessment, DangerLevel } from "mjrender/danger.ts";
import type { RiverEntry } from "mjrender/state.ts";
import { doraFromIndicatorType, tileType } from "mjrender/tiles.ts";
import type { Observation } from "../src/observe.ts";
import type { Encoded } from "../src/rl/features.ts";
import { encode, FEATURES, flatten, INPUT_LEN, PLANE_LEN, TYPES } from "../src/rl/features.ts";
import { JANKI } from "../src/rules.ts";
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

function river(ids: Tile[]): RiverEntry[] {
  return ids.map((tile, i) => ({ tile, junme: i + 1, tsumogiri: false, riichiDeclare: false }));
}

/** The bare assessment the encoder reads: only `level` reaches the planes. */
function assessment(level: DangerLevel): DangerAssessment {
  return { level, seats: [1], details: [{ seat: 1, level, kind: "riichi", notes: [] }] };
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
  assertEquals(FEATURES.version, 2);
  assertEquals(FEATURES.planes, 36);
  assertEquals(FEATURES.scalars, 39);
  assertEquals(FEATURES.actions, 78);
  assertEquals(TYPES, 34);
  assertEquals(PLANE_LEN, 1224);
  assertEquals(INPUT_LEN, 1263);
});

Deno.test("features: encode returns 1224 plane cells and 39 scalars", () => {
  const e = encode(baseObs());
  assertEquals(e.planes.length, 1224);
  assertEquals(e.scalars.length, 39);
  assert(e.planes instanceof Int8Array);
  assert(e.scalars instanceof Float32Array);
  // Planes are strictly indicator bits.
  for (let i = 0; i < e.planes.length; i++) {
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

// --------------------------------------------------------------------------
// flatten
// --------------------------------------------------------------------------

Deno.test("features: flatten is planes ++ scalars, 1263 wide", () => {
  const e = encode(baseObs());
  const f = flatten(e);
  assertEquals(f.length, INPUT_LEN);
  assertEquals(f.length, 1263);
  assert(f instanceof Float32Array);
  for (let i = 0; i < PLANE_LEN; i++) assertEquals(f[i], e.planes[i], `plane cell ${i}`);
  for (let j = 0; j < FEATURES.scalars; j++) {
    assertEquals(f[PLANE_LEN + j], e.scalars[j], `scalar ${j}`);
  }
});

// --------------------------------------------------------------------------
// whole-buffer digest
// --------------------------------------------------------------------------

/** What v2 emits for `baseObs()`. See the note on the digest test below. */
const PLANE_DIGEST = 816561413;
const SCALAR_DIGEST = 479078765;

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
 * These two numbers are not derived from anything — they are what v2 emits for
 * the fixture. REGENERATE THEM DELIBERATELY: any edit that moves them means the
 * wire format the trainer reshapes has changed, so `FEATURES.version` must be
 * bumped on both sides in the same commit.
 */
Deno.test("features: the v2 encoding of the fixture digests to the frozen values", () => {
  const e = encode(baseObs());
  const planeBytes = new Uint8Array(e.planes.buffer, e.planes.byteOffset, e.planes.byteLength);
  assertEquals(fnv1a(planeBytes), PLANE_DIGEST, "plane digest");
  assertEquals(fnv1a(floatBytes(e.scalars)), SCALAR_DIGEST, "scalar digest");
});
