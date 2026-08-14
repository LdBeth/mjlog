// The baseline CPU. Every observation here is hand-built (the pattern comes
// from timer_test.ts) so each test states exactly one situation and varies
// nothing else — no table, no wall, no round driver.

import { assert, assertEquals } from "@std/assert";
import type { DangerAssessment, DangerLevel } from "mjrender/danger.ts";
import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import { tileType } from "mjrender/tiles.ts";
import { HeuristicPolicy } from "../src/ai/heuristic.ts";
import type { Observation } from "../src/observe.ts";
import { JANKI } from "../src/rules.ts";
import type { Action } from "../src/types.ts";
import { tiles } from "./helpers.ts";

/** One discard action per tile in hand, riichi never offered. */
function discardsOf(hand: Tile[], drawn: Tile | null): Action[] {
  return hand.map((tile) => ({
    t: "discard",
    tile,
    riichi: false,
    tsumogiri: drawn !== null && tile === drawn,
  }));
}

/**
 * Two 巡 already played, one honor cut per seat. The rivers must not be empty:
 * the policy prices a 第一打字牌切り at 4000, which would otherwise swamp every
 * other term whenever an honor is the discard under test.
 */
function openingRivers(): RiverEntry[][] {
  const pool = tiles("北北北北西西西西"); // one distinct copy per seat, twice
  return [0, 1, 2, 3].map((s) =>
    [pool[s], pool[4 + s]].map((tile, i): RiverEntry => ({
      tile,
      junme: i + 1,
      tsumogiri: false,
      riichiDeclare: false,
    }))
  );
}

/**
 * A complete, internally consistent Observation. `legal` defaults to "discard
 * anything"; pass it explicitly for claim situations.
 */
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
    doraIndicators: tiles("9s"), // ⇒ dora is 1s, which no hand below holds
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

/** A single-seat threat entry, shaped the way `assessDanger` returns them. */
function threat(
  level: DangerLevel,
  seat = 1,
  kind: "riichi" | "furo" = "riichi",
): DangerAssessment {
  return { level, seats: [seat], details: [{ seat, level, kind, notes: [] }] };
}

const TON = 27; // 東
const HAKU = 31; // 白

Deno.test("heuristic: a win is taken over any discard", () => {
  const hand = tiles("123456789m11122p"); // 123m 456m 789m 111p 22p
  const obs = baseObs({
    hand,
    drawn: hand[hand.length - 1],
    shanten: -1,
    legal: [...discardsOf(hand, hand[hand.length - 1]), { t: "tsumo" }],
  });
  const p = new HeuristicPolicy("cpu", 1);
  assertEquals(p.decide(obs).t, "tsumo");
});

Deno.test("heuristic: cuts the isolated honor out of a finished shape", () => {
  const obs = baseObs(); // 123456789m 1122p + lone 東
  const p = new HeuristicPolicy("cpu", 1);
  const a = p.decide(obs);
  assertEquals(a.t, "discard");
  assert(a.t === "discard");
  assertEquals(tileType(a.tile), TON, "the lone honor is the only free tile");
});

Deno.test("heuristic: folds a far hand against a riichi", () => {
  const hand = tiles("1346m2479p2466s東白");
  const danger = new Map<number, DangerAssessment>([[HAKU, threat("危険度高")]]);
  const obs = baseObs({
    hand,
    drawn: hand[hand.length - 1], // the 白 was just drawn
    shanten: 3,
    riichi: [false, true, false, false],
    riichiJunme: [-1, 2, -1, -1],
    danger,
  });
  const p = new HeuristicPolicy("cpu", 1);
  const a = p.decide(obs);
  assertEquals(a.t, "discard");
  assert(a.t === "discard");
  assert(tileType(a.tile) !== HAKU, "must not fire the 危険度高 tile while folding");
  assertEquals(tileType(a.tile), TON, "the genbutsu honor is the fold");
});

Deno.test("heuristic: pushes the dangerous tile when tenpai with value", () => {
  const hand = tiles("123456789m1122p東");
  const danger = new Map<number, DangerAssessment>([[TON, threat("危険度高")]]);
  const obs = baseObs({
    hand,
    shanten: 0,
    waits: [tileType(tiles("1p")[0]), tileType(tiles("2p")[0])],
    doraIndicators: tiles("9p"), // ⇒ dora 1p, of which the hand holds two
    doraCount: 2,
    riichi: [false, true, false, false],
    riichiJunme: [-1, 2, -1, -1],
    danger,
  });
  const p = new HeuristicPolicy("cpu", 1);
  const a = p.decide(obs);
  assertEquals(a.t, "discard");
  assert(a.t === "discard");
  assertEquals(tileType(a.tile), TON, "tenpai with two dora pushes through the threat");
});

Deno.test("heuristic: never calls 明槓", () => {
  const hand = tiles("111m2345678p234s");
  const called = tiles("1111m")[3];
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 1,
    legal: [{ t: "daiminkan", called }, { t: "pass" }],
  });
  const p = new HeuristicPolicy("cpu", 1);
  assertEquals(p.decide(obs).t, "pass");
});

Deno.test("heuristic: refuses a chi with no route to any yaku", () => {
  // 111m 77m 45p 23s 56s 89s — the chi improves the shape but the prospect
  // screen finds no yaku left to aim at: no 役牌 melded or paired, three 1m
  // put 断幺九 out of reach, the concealed part spans m+s against a p meld so
  // 混一色 is gone, and the chi itself rules out 対々和.
  const hand = tiles("11177m45p235689s");
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 2,
    legal: [
      { t: "pass" },
      { t: "chi", tiles: [hand[5], hand[6]], called: tiles("6p")[0] },
    ],
  });
  const p = new HeuristicPolicy("cpu", 1);
  assertEquals(p.decide(obs).t, "pass");
});

Deno.test("heuristic: calls toward a 混一色 build", () => {
  // 12s 456s 789s 33s 白白 ＋ 一枚の 5m. The old confirmed-yaku filter refused
  // this chi (a 白 pair with nothing "confirmed" read as バック); the screen
  // lets it through, and so does the ledger — the crime would be finishing on
  // a yakuless wait, which this build never does.
  const all = tiles("12s456s789s33s白白5m3s");
  const hand = all.slice(0, 13);
  const called = all[13]; // a third 3s, distinct from the pair in hand
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 1, // 456s + 789s + 33s/白白/12s の三ブロック
    legal: [
      { t: "pass" },
      { t: "chi", tiles: [hand[0], hand[1]], called },
    ],
  });
  const p = new HeuristicPolicy("cpu", 1);
  assertEquals(p.decide(obs).t, "chi");
});

Deno.test("heuristic: avoids the discard that fixes a yakuless open wait", () => {
  // An open hand (one chi), two discards that both reach tenpai — one of them
  // on a wait nothing scores on, which is a Tier A 後付け the moment it lands.
  const meldTiles = tiles("123m");
  const meld: Meld = {
    kind: "chi",
    who: 0,
    fromWho: 3,
    tiles: meldTiles,
    calledTile: meldTiles[0],
  };
  const hand = tiles("456m678p55s78s5m");
  const [dirty, clean] = [hand[hand.length - 1], hand[0]];
  const obs = baseObs({
    hand,
    drawn: null,
    melds: [[meld], [], [], []],
    discardInfo: new Map([
      [dirty, { shanten: 0, katagari: false, yakuless: true }],
      [clean, { shanten: 0, katagari: false, yakuless: false }],
    ]),
    legal: [
      { t: "discard", tile: dirty, riichi: false, tsumogiri: false },
      { t: "discard", tile: clean, riichi: false, tsumogiri: false },
    ],
  });
  const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
  assertEquals(chosen.t, "discard");
  if (chosen.t === "discard") assertEquals(chosen.tile, clean);
});

Deno.test("heuristic: takes a yakuhai pon that gains a shanten", () => {
  const hand = tiles("11199m22p4578s白白");
  const called = tiles("白白白")[2];
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 2,
    legal: [
      { t: "pass" },
      { t: "pon", tiles: [hand[11], hand[12]], called },
    ],
  });
  const p = new HeuristicPolicy("cpu", 1);
  assertEquals(p.decide(obs).t, "pon");
});

Deno.test("heuristic: the same seed and observation give the same action", () => {
  const obs = baseObs();
  const a = new HeuristicPolicy("a", 7);
  const b = new HeuristicPolicy("b", 7);
  assertEquals(a.decide(obs), b.decide(obs), "two policies, one seed");
  assertEquals(a.decide(obs), a.decide(obs), "and no hidden per-call state");
});

Deno.test("heuristic: dojo:false lifts the 禁じ手 filters", () => {
  // 明槓: deliberately loose. `chooseKan` only ever *returns* an ankan, so the
  // dojo flag cannot change this one either way — all that is asserted is that
  // lifting the filter still yields a legal action.
  const kanHand = tiles("111m2345678p234s");
  const kanObs = baseObs({
    hand: kanHand,
    drawn: null,
    shanten: 1,
    legal: [{ t: "daiminkan", called: tiles("1111m")[3] }, { t: "pass" }],
  });
  const kan = new HeuristicPolicy("cpu", 1, { dojo: false }).decide(kanObs);
  assert(kanObs.legal.includes(kan), "whatever it picks must be a legal action");

  // The yaku-prospect screen, where the flag does bite: the very chi refused
  // above is now taken.
  const hand = tiles("11177m45p235689s");
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 2,
    legal: [
      { t: "pass" },
      { t: "chi", tiles: [hand[5], hand[6]], called: tiles("6p")[0] },
    ],
  });
  assertEquals(new HeuristicPolicy("cpu", 1, { dojo: false }).decide(obs).t, "chi");
});

// --------------------------------------------------------------------------
// 片和了り: prevention first, abandonment only when there is nothing to shift to
// --------------------------------------------------------------------------

Deno.test("heuristic: prefers the clean tenpai over the 片和了り one", () => {
  const hand = tiles("123456789m1122p東");
  const [a, b] = [hand[9], hand[11]]; // one tile from each pair of 1p/2p
  const obs = baseObs({
    hand,
    drawn: hand[hand.length - 1],
    // Both discards reach tenpai; only one of them splits the wait.
    discardInfo: new Map([
      [hand[hand.length - 1], { shanten: 0, katagari: true, yakuless: false }],
      [a, { shanten: 0, katagari: false, yakuless: false }],
      [b, { shanten: 0, katagari: true, yakuless: false }],
    ]),
  });
  const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
  assertEquals(chosen.t, "discard");
  if (chosen.t === "discard") assertEquals(chosen.tile, a);
});

Deno.test("heuristic: gives the hand up when every tenpai is 片和了り and it cannot riichi", () => {
  const hand = tiles("123456789m1122p東");
  const drawn = hand[hand.length - 1];
  // An open hand: no riichi is on offer, so the split wait has no cure.
  const discardInfo = new Map(
    hand.map((t) =>
      [t, { shanten: t === drawn ? 1 : 0, katagari: t !== drawn, yakuless: false }] as const
    ),
  );
  const obs = baseObs({ hand, drawn, discardInfo });
  const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
  assertEquals(chosen.t, "discard");
  // The one tile that breaks tenpai is taken over any of the split-wait shapes.
  if (chosen.t === "discard") assertEquals(chosen.tile, drawn);
});

Deno.test("heuristic: riichi cures 片和了り rather than abandoning the hand", () => {
  const hand = tiles("123456789m1122p東");
  const drawn = hand[hand.length - 1];
  const keep = hand[9];
  const obs = baseObs({
    hand,
    drawn,
    waits: [tileType(tiles("1p")[0])],
    ukeire: [{ type: tileType(tiles("1p")[0]), live: 3 }],
    discardInfo: new Map([[keep, { shanten: 0, katagari: true, yakuless: false }]]),
    legal: [
      { t: "discard", tile: keep, riichi: false, tsumogiri: false },
      { t: "discard", tile: keep, riichi: true, tsumogiri: false },
    ],
  });
  const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
  assertEquals(chosen.t, "discard");
  if (chosen.t === "discard") assertEquals(chosen.riichi, true);
});

Deno.test("heuristic: a short stack folds where a healthy one would push", () => {
  const hand = tiles("123456789m1122p東");
  const drawn = hand[hand.length - 1];
  const shared = {
    hand,
    drawn,
    shanten: 1,
    riichi: [false, true, false, false],
    danger: new Map([[TON, threat("危険度高")]]),
  };
  const pick = (score: number) => {
    const obs = baseObs({ ...shared, scores: [score, 25000, 25000, 25000] });
    const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
    return chosen.t === "discard" ? chosen.tile : -1;
  };
  // 1-shanten with a dangerous 東: comfortable stack pushes it, short one does not.
  assertEquals(pick(25000), drawn);
  assert(pick(9000) !== drawn, "a 9000-point stack should not fire the 危険度高 tile");
});

Deno.test("heuristic: tedashi stops once ドラ切り has been ponned off us", () => {
  const hand = tiles("123456789m1122p東");
  const drawn = hand[hand.length - 1]; // 東, the tile a free hand would cut
  const pick = (tsumogiriLock: boolean) => {
    const obs = baseObs({ hand, drawn, tsumogiriLock });
    const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
    return chosen.t === "discard" ? chosen.tile : -1;
  };
  // Unlocked, the lone honor goes. Locked, every tedashi is a violation, so the
  // drawn tile goes back out instead — which here happens to be the same tile.
  assertEquals(pick(false), drawn);
  assertEquals(pick(true), drawn, "a locked hand must tsumogiri");

  // With a drawn tile that efficiency would rather keep, the lock still wins.
  const keeper = tiles("123456789m1122p1p")[15];
  const locked = baseObs({
    hand: [...hand.slice(0, 13), keeper],
    drawn: keeper,
    tsumogiriLock: true,
  });
  const chosen = new HeuristicPolicy("cpu", 1).decide(locked);
  assertEquals(chosen.t, "discard");
  if (chosen.t === "discard") assertEquals(chosen.tile, keeper);
});
