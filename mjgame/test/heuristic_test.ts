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
import type { ActionPreview } from "../src/penalty/preview.ts";
import { JANKI } from "../src/rules.ts";
import type { Action, Violation } from "../src/types.ts";
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

// --------------------------------------------------------------------------
// The compliance filter. These Observations carry a `preview` — the referee,
// asked hypothetically — which is what every real driver supplies and no
// hand-built Observation above does. That asymmetry is the point: without a
// preview the policy keeps the older priced behaviour every test above pins,
// and with one it stops CHOOSING the charged action at all.
// --------------------------------------------------------------------------

/** One Tier A entry, enough for the policy to see the action is charged. */
function charge(rule: string): Violation[] {
  return [{
    rule,
    label: rule,
    seat: 0,
    kyoku: 0,
    junme: 0,
    points: 3,
    tier: "A",
    confidence: 1,
    detail: "stub",
  }];
}

/**
 * A referee stub. Each hook says whether it would charge; anything unset stays
 * silent, so a test states exactly the one veto it is about.
 */
function stubPreview(o: {
  discard?: (a: Extract<Action, { t: "discard" }>) => string | null;
  call?: (a: Action) => string | null;
  kan?: (a: Action) => string | null;
  skipKan?: boolean;
}): ActionPreview {
  const vs = (id: string | null | undefined) => (id ? charge(id) : []);
  return {
    discard: (a) => vs(o.discard?.(a)),
    call: (a) => vs(o.call?.(a)),
    kan: (a) => vs(o.kan?.(a)),
    skipKan: () => (o.skipKan ? charge("riichi-kan-skip") : []),
  };
}

Deno.test("filter: a discard the referee would charge is not chosen", () => {
  const hand = tiles("123456789m1122p東");
  const free = hand[hand.length - 1]; // the lone 東 — what an unfiltered policy cuts
  const obs = baseObs({
    hand,
    preview: stubPreview({ discard: (a) => (a.tile === free ? "first-honor" : null) }),
  });
  const a = new HeuristicPolicy("cpu", 1).decide(obs);
  assertEquals(a.t, "discard");
  assert(a.t === "discard");
  assert(a.tile !== free, "the charged tile must be out of the choice set entirely");
});

Deno.test("filter: when every discard is charged the prices decide again", () => {
  // The fallthrough. Something has to be thrown, so the filter stands aside and
  // `dojoCost` ranks the damage — which here is the same tile the unfiltered
  // policy picks, because nothing else has changed.
  const hand = tiles("123456789m1122p東");
  const free = hand[hand.length - 1];
  const obs = baseObs({ hand, preview: stubPreview({ discard: () => "noten-dora" }) });
  const a = new HeuristicPolicy("cpu", 1).decide(obs);
  assertEquals(a.t, "discard");
  assert(a.t === "discard");
  assertEquals(a.tile, free);
});

Deno.test("filter: a riichi declaration the referee would charge is declined", () => {
  const hand = tiles("123456789m1122p東");
  const keep = hand[9];
  const build = (preview?: ActionPreview) =>
    baseObs({
      hand,
      drawn: hand[hand.length - 1],
      waits: [tileType(tiles("1p")[0])],
      ukeire: [{ type: tileType(tiles("1p")[0]), live: 3 }],
      discardInfo: new Map([[keep, { shanten: 0, katagari: true, yakuless: false }]]),
      legal: [
        { t: "discard", tile: keep, riichi: false, tsumogiri: false },
        { t: "discard", tile: keep, riichi: true, tsumogiri: false },
      ],
      preview,
    });
  // Control: with no referee to ask, the split wait is cured by declaring.
  const cure = new HeuristicPolicy("cpu", 1).decide(build());
  assert(cure.t === "discard" && cure.riichi, "control: 片和了り is cured by riichi");

  // The declaration is 地獄単騎 — a 禁じ手 outranks the cure.
  const banned = stubPreview({ discard: (a) => (a.riichi ? "jigoku-tanki" : null) });
  const chosen = new HeuristicPolicy("cpu", 1).decide(build(banned));
  assert(chosen.t === "discard" && !chosen.riichi, "a charged declaration is not made");
});

Deno.test("filter: a call the referee would charge is not taken", () => {
  const hand = tiles("11199m22p4578s白白");
  const called = tiles("白白白")[2];
  const legal: Action[] = [
    { t: "pass" },
    { t: "pon", tiles: [hand[11], hand[12]], called },
  ];
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 2,
    legal,
    preview: stubPreview({ call: () => "hadaka-tanki" }),
  });
  assertEquals(new HeuristicPolicy("cpu", 1).decide(obs).t, "pass");
});

Deno.test("filter: 立直後カン見送り forces the kan the policy would have skipped", () => {
  // `chooseKan` declines this one on its own terms (the concealed four are not
  // all in hand, so it cannot prove the wait survives). Declining is the
  // violation here, so the referee's verdict on the OMISSION overrides.
  const hand = tiles("111m456p78p111s99s");
  const drawn = hand[0];
  const legal: Action[] = [
    { t: "discard", tile: drawn, riichi: false, tsumogiri: true },
    { t: "ankan", type: 0 },
  ];
  const shared = { hand, drawn, shanten: 0, riichi: [true, false, false, false], legal };
  assertEquals(
    new HeuristicPolicy("cpu", 1).decide(baseObs(shared)).t,
    "discard",
    "control: with nobody asking, the kan is passed up",
  );
  const obs = baseObs({ ...shared, preview: stubPreview({ skipKan: true }) });
  assertEquals(new HeuristicPolicy("cpu", 1).decide(obs).t, "ankan");
});

Deno.test("filter: it is off with dojo:false, prices and all", () => {
  const hand = tiles("123456789m1122p東");
  const free = hand[hand.length - 1];
  const obs = baseObs({
    hand,
    preview: stubPreview({ discard: (a) => (a.tile === free ? "first-honor" : null) }),
  });
  const a = new HeuristicPolicy("cpu", 1, { dojo: false }).decide(obs);
  assert(a.t === "discard" && a.tile === free, "an unleashed policy ignores the referee");
});

// --------------------------------------------------------------------------
// 片和了り / 後付け are vetoed too, though no preview can see them: they are
// charged at WIN time (declining a win is 見逃し, so there is no action to
// veto there) while the only prevention is the discard that builds the wait.
// Left as prices they lose — a shanten step is 1000 and the katagari price is
// 1500, so a two-step gap already buys the violation, and the C7 planner's
// planKeep malus (5000) buys it outright.
// --------------------------------------------------------------------------

/** All-but-one tile flagged, with the flagged one the only route to tenpai. */
function splitWaitObs(flag: "katagari" | "yakuless", over: Partial<Observation> = {}) {
  const hand = tiles("123456789m1122p東");
  const split = hand[9]; // a 1p: cutting it reaches the split tenpai
  const discardInfo = new Map(
    hand.map((t) =>
      [t, {
        shanten: t === split ? 0 : 2,
        katagari: flag === "katagari" && t === split,
        yakuless: flag === "yakuless" && t === split,
      }] as const
    ),
  );
  return { split, obs: baseObs({ hand, discardInfo, preview: stubPreview({}), ...over }) };
}

Deno.test("filter: a 片和了り discard is vetoed, not merely outbid", () => {
  const { split, obs } = splitWaitObs("katagari");
  // Control: the price alone loses. Tenpai vs 2向聴 is worth 2000 and 片和了り
  // costs 1500, so an unrefereed policy takes the split wait.
  const priced = new HeuristicPolicy("cpu", 1).decide(baseObs({ ...obs, preview: undefined }));
  assert(priced.t === "discard" && priced.tile === split, "control: the price is outbid");

  const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
  assert(chosen.t === "discard" && chosen.tile !== split, "the split wait is out of the set");
});

Deno.test("filter: a 後付け discard is vetoed for an open hand", () => {
  const meld: Meld[] = [{
    kind: "pon",
    who: 0,
    fromWho: 1,
    tiles: tiles("白白白"),
    calledTile: tiles("白白白")[0],
  }];
  const open = { melds: [meld, [], [], []] as Meld[][] };
  const { split, obs } = splitWaitObs("yakuless", open);
  const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
  assert(chosen.t === "discard" && chosen.tile !== split, "a yakuless open tenpai is refused");

  // 門前 is exempt: the same shape can still be cured by declaring riichi, so
  // the rule cannot fire and the veto must not either.
  const closed = splitWaitObs("yakuless");
  const kept = new HeuristicPolicy("cpu", 1).decide(closed.obs);
  assert(kept.t === "discard" && kept.tile === closed.split, "門前 keeps the tenpai");
});

Deno.test("filter: when every tenpai is 片和了り the prices decide again", () => {
  // The fallthrough contract, shared with the previewable rules: something has
  // to be thrown, so the veto stands aside and `dojoCost` ranks the damage —
  // which is exactly what the unrefereed policy already does.
  const hand = tiles("123456789m1122p東");
  const drawn = hand[hand.length - 1];
  const discardInfo = new Map(
    hand.map((t) => [t, { shanten: 0, katagari: true, yakuless: false }] as const),
  );
  const build = (preview?: ActionPreview) => baseObs({ hand, drawn, discardInfo, preview });
  const a = new HeuristicPolicy("cpu", 1).decide(build(stubPreview({})));
  const b = new HeuristicPolicy("cpu", 1).decide(build());
  assert(a.t === "discard" && b.t === "discard");
  assertEquals(a.tile, b.tile, "an empty candidate set falls back to the priced ranking");
});

Deno.test("filter: the 片和了り veto lifts when riichi is on offer", () => {
  const hand = tiles("123456789m1122p東");
  const keep = hand[9]; // a 1p — the split tenpai
  const alt = hand[hand.length - 1]; // the 東 — 1向聴, but clean
  const build = (canRiichi: boolean) =>
    baseObs({
      hand,
      drawn: alt,
      waits: [tileType(tiles("1p")[0])],
      ukeire: [{ type: tileType(tiles("1p")[0]), live: 3 }],
      discardInfo: new Map([
        [keep, { shanten: 0, katagari: true, yakuless: false }],
        [alt, { shanten: 1, katagari: false, yakuless: false }],
      ]),
      legal: [
        { t: "discard", tile: keep, riichi: false, tsumogiri: false },
        ...(canRiichi
          ? [{ t: "discard", tile: keep, riichi: true, tsumogiri: false } as Action]
          : []),
        { t: "discard", tile: alt, riichi: false, tsumogiri: true },
      ],
      preview: stubPreview({}),
    });
  // 立直 is itself a yaku: it makes every wait scoring, so the shape stops being
  // split and there is nothing to veto — the same clause `dojoCost` prices on.
  const cure = new HeuristicPolicy("cpu", 1).decide(build(true));
  assert(cure.t === "discard" && cure.tile === keep && cure.riichi, "riichi cures it instead");

  // With no declaration available the exemption lapses and the veto bites.
  const folded = new HeuristicPolicy("cpu", 1).decide(build(false));
  assert(folded.t === "discard" && folded.tile === alt, "no cure ⇒ the split wait is refused");
});
