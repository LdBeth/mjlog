// The C7 max-profit lock-on planner.
//
// Three layers, tested separately: the ENUMERATION (does a hand's futures come
// out with the right missing tiles, the right call routes and no 後付け traps),
// the ARITHMETIC (pComplete's monotonicity, which is all the ranking rests on),
// and the LOCK (hysteresis, and the round boundary that clears it). Every
// observation is hand-built, in the style of heuristic_test.ts, so each test
// states one situation and varies nothing else.

import { assert, assertEquals, assertGreater, assertLessOrEqual } from "@std/assert";
import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import { countsFromTiles, shanten } from "mjrender/shanten.ts";
import { tileType } from "mjrender/tiles.ts";
import type { Availability, TargetPlan } from "../src/ai/planner.ts";
import {
  availabilityFrom,
  enumerateTargets,
  ownDraws,
  pCompleteOf,
  publicUnseen,
  relock,
} from "../src/ai/planner.ts";
import { AugmentedHeuristic } from "../src/ai/augmented.ts";
import type { Reads } from "../src/ai/augmented.ts";
import type { Observation } from "../src/observe.ts";
import { JANKI } from "../src/rules.ts";
import type { Action } from "../src/types.ts";
import { tiles } from "./helpers.ts";

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

/** Two 巡 of honor discards, so no hand under test is a 第一打 situation. */
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

interface ObsOver {
  hand: Tile[];
  melds?: Meld[];
  indicator?: string;
  legal?: Action[];
  claimTile?: Tile | null;
  kyoku?: number;
  honba?: number;
  wallRemaining?: number;
}

/** A complete Observation whose `shanten` is derived, not asserted by hand. */
function obsOf(o: ObsOver): Observation {
  const melds = o.melds ?? [];
  const closed = melds.every((m) => m.kind === "ankan");
  return {
    seat: 0,
    kyoku: o.kyoku ?? 0,
    honba: o.honba ?? 0,
    kyotaku: 0,
    junme: 4,
    wallRemaining: o.wallRemaining ?? 60,
    hand: o.hand,
    drawn: null,
    claimTile: o.claimTile ?? null,
    melds: [melds, [], [], []],
    rivers: openingRivers(),
    scores: [25000, 25000, 25000, 25000],
    riichi: [false, false, false, false],
    riichiJunme: [-1, -1, -1, -1],
    doraIndicators: tiles(o.indicator ?? "9s"), // ⇒ 1索, held by nothing below
    seatWind: 28, // 南: never the dealer, so the payment table is the plain one
    roundWind: 27,
    akaIds: JANKI.akaIds,
    shanten: shanten(countsFromTiles(o.hand), melds.length, closed),
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
    legal: o.legal ?? [],
  };
}

const chi = (spec: string): Meld => {
  const ts = tiles(spec);
  return { kind: "chi", who: 0, fromWho: 3, tiles: ts, calledTile: ts[0] };
};

/** Every plan of a hand, under the public (C7P) availability model. */
function plansOf(o: ObsOver): TargetPlan[] {
  const obs = obsOf(o);
  return enumerateTargets(obs, availabilityFrom(obs, null));
}

// ---------------------------------------------------------------------------
// enumeration
// ---------------------------------------------------------------------------

const THREE_P = 11;

Deno.test("planner: a tenpai hand's best future needs exactly its winning tile", () => {
  // 123m456m789m + 12p + 55s — 一通 already assembled, waiting on 3p.
  const plans = plansOf({ hand: tiles("123m456m789m12p55s") });
  assert(plans.length > 0);

  const best = plans[0];
  assertEquals(best.required, [THREE_P], "one tile short, and it is the 3p");
  assertEquals(best.callable, [], "the winning tile is ronned, never called");
  assertEquals(best.open, false);
  assertEquals(best.keep.size, 13, "a 一通 tenpai keeps every tile it holds");
  // 一通 + 立直 + the closed bonus is already mangan at this seat.
  assertEquals(best.value, 8000);
  assertEquals(best.profit, best.pComplete * best.value);

  // Every plan is internally consistent, not just the winner.
  for (const p of plans) {
    assertEquals(p.open, p.callable.length > 0);
    assert(p.callable.every((ty) => p.required.includes(ty)), "callable ⊆ required");
    assert(p.required.length <= obsOf({ hand: tiles("123m456m789m12p55s") }).shanten + 2);
  }
});

Deno.test("planner: dora in the target outranks the same shape without it", () => {
  const hand = tiles("234m567m234p88s34s"); // tenpai on 2s/5s, 断幺九
  const plain = plansOf({ hand, indicator: "9m" }); // dora 1m — held by nobody
  const rich = plansOf({ hand, indicator: "7s" }); // dora 8s — the head, twice

  const a = plain[0];
  const b = rich.find((p) => p.key === a.key);
  assert(b !== undefined, "the same final shape is available under both dora");

  // The waits are untouched, so availability — and therefore pComplete — is
  // identical; only the finished hand's price moved.
  assertEquals(b.pComplete, a.pComplete);
  assertGreater(b.value, a.value);
  assertGreater(b.profit, a.profit);
});

Deno.test("planner: an open future with no yaku is not a future (後付け screen)", () => {
  // 456p called. Everything left completes to 456p 234m 567m 123s 99s and its
  // relatives — no 役牌, no 断幺九 (1索/9索), no 全帯幺 (234m/567m). Nothing to
  // win on, so nothing to plan for.
  const trap = plansOf({ hand: tiles("234m567m99s12s"), melds: [chi("456p")] });
  assertEquals(trap, [], "an open hand with no route to a yaku yields no plans");

  // The control: swap the 12索 for a 中 pair and the same shape is a 役牌 hand.
  const good = plansOf({ hand: tiles("234m567m99s中中"), melds: [chi("456p")] });
  assert(good.length > 0);
  assert(
    good.every((p) => p.required.includes(33)),
    "every surviving future goes through the 中 triplet — the only yaku on offer",
  );
});

Deno.test("planner: callable marks pon/chi routes and never the head or the win", () => {
  // 123m456m789m + 中中 + 23p: 一通 with a 中 pair and a 23p 両面.
  const plans = plansOf({ hand: tiles("123m456m789m中中2p3p") });
  assert(plans.length > 0);

  for (const p of plans) {
    for (const ty of p.callable) {
      assert(p.required.includes(ty), "a callable type is a required type");
      // 単騎 is the only way a head is still one tile short, and no call makes a
      // head — so a plan whose head is unfinished must not offer it as callable.
      assert(p.open, "callable implies the plan is an open one");
    }
  }

  // A plan that pons the 中 exists, and it is an open one.
  const ponPlan = plans.find((p) => p.callable.includes(33));
  assert(ponPlan !== undefined);
  assertEquals(ponPlan.open, true);

  // One tile short, nothing is callable: the tile left IS the win, and claiming
  // it is a ron, not a completion.
  const tenpai = plansOf({ hand: tiles("123m456m789m12p55s") });
  for (const p of tenpai.filter((x) => x.required.length === 1)) assertEquals(p.callable, []);

  // A 単騎 head is never callable — no call makes a head.
  for (const p of plansOf({ hand: tiles("234m567m99s12s"), melds: [chi("456p")] })) {
    const tanki = p.required.filter((ty) => !p.callable.includes(ty));
    assert(tanki.length > 0 || p.callable.length === 0);
  }
});

Deno.test("planner: beyond 3向聴 there is nothing to plan", () => {
  const junk = tiles("139m258p47s東南西北");
  const obs = obsOf({ hand: junk });
  assertGreater(obs.shanten, 3);
  assertEquals(enumerateTargets(obs, availabilityFrom(obs, null)), []);
});

Deno.test("planner: the oracle and the public model see different availability", () => {
  const obs = obsOf({ hand: tiles("123m456m789m12p55s") });
  const pub = availabilityFrom(obs, null);
  assertEquals(pub.draw(THREE_P), 4, "no 3p is visible anywhere");

  // C7O: the wall holds a single 3p, and one opponent is sitting on another.
  const wall = new Float32Array(34);
  wall[THREE_P] = 1;
  wall[0] = 60;
  const opp = [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
  opp[1][THREE_P] = 1;
  const reads: Reads = { planner: true, wallComposition: wall, oppConcealed: opp };
  const ora = availabilityFrom(obs, reads);
  assertEquals(ora.draw(THREE_P), 1, "one copy left in the wall — that is the truth");
  assertEquals(ora.call(THREE_P), 0.5, "half of every held copy, the documented crudeness");
  assertEquals(ora.poolSize(), 61);

  // And that truth reprices the future: a nearly dead wait is a worse plan.
  const oraclePlans = enumerateTargets(obs, ora);
  const publicPlans = enumerateTargets(obs, pub);
  assertGreater(publicPlans[0].pComplete, oraclePlans[0].pComplete);
});

Deno.test("planner: publicUnseen counts what the seat cannot see", () => {
  const obs = obsOf({ hand: tiles("123m456m789m12p55s"), indicator: "3p" });
  const u = publicUnseen(obs);
  assertEquals(u[0], 3, "one 1m in hand");
  assertEquals(u[THREE_P], 3, "the 3p indicator is face up");
  assertEquals(u[22], 2, "two 5索 in hand");
  assertEquals(u[30], 0, "all four 北 are in the rivers");
});

// ---------------------------------------------------------------------------
// the arithmetic
// ---------------------------------------------------------------------------

/** A stub pool: `n` copies of every type, in a pool of `size`. */
function flatAvail(n: number, size = 70, callable = 0): Availability {
  return { draw: () => n, call: () => callable, poolSize: () => size };
}

Deno.test("planner: pComplete rises with copies and with draws, never the reverse", () => {
  const req = [5, 12];
  const few = pCompleteOf(req, [], flatAvail(1), 10);
  const many = pCompleteOf(req, [], flatAvail(3), 10);
  assertGreater(many, few, "more copies reachable ⇒ at least as likely");

  const early = pCompleteOf(req, [], flatAvail(2), 14);
  const late = pCompleteOf(req, [], flatAvail(2), 4);
  assertLessOrEqual(late, early, "fewer draws left ⇒ no more likely");

  // A callable type adds the claim route on top of the draw route.
  const drawOnly = pCompleteOf(req, [], flatAvail(2, 70, 2), 10);
  const withCall = pCompleteOf(req, [5], flatAvail(2, 70, 2), 10);
  assertGreater(withCall, drawOnly);

  // Degenerate ends: a dead type is unreachable, and a finished round has no
  // draws left to reach anything in.
  assertEquals(pCompleteOf(req, [], flatAvail(0), 10), 0);
  assertEquals(pCompleteOf(req, [], flatAvail(4), 0), 0);
  assertEquals(pCompleteOf([], [], flatAvail(4), 10), 1);

  // The multi-tile discount: two tiles are worth less than the product alone.
  const one = pCompleteOf([5], [], flatAvail(2), 10);
  const two = pCompleteOf([5, 5], [], flatAvail(2), 10);
  assertEquals(two, one * one * 0.85);
});

Deno.test("planner: own draws are a quarter of what the wall has left", () => {
  assertEquals(ownDraws(obsOf({ hand: tiles("123m456m789m12p55s"), wallRemaining: 60 })), 15);
  assertEquals(ownDraws(obsOf({ hand: tiles("123m456m789m12p55s"), wallRemaining: 3 })), 0);
});

// ---------------------------------------------------------------------------
// the lock
// ---------------------------------------------------------------------------

function fakePlan(key: string, profit: number): TargetPlan {
  return {
    key,
    keep: new Set<Tile>(),
    required: [],
    callable: [],
    open: false,
    value: profit,
    pComplete: 1,
    profit,
    label: key,
  };
}

Deno.test("planner: the lock holds against a small challenger and yields to a big one", () => {
  const locked = fakePlan("A", 1000);

  // +10%: not enough. The refreshed A — not the object handed in — is returned,
  // because the current plan is re-priced every turn.
  const near = [fakePlan("B", 1100), fakePlan("A", 1000)];
  assertEquals(relock(locked, near)!.key, "A");

  // +25%: enough.
  const far = [fakePlan("B", 1250), fakePlan("A", 1000)];
  assertEquals(relock(locked, far)!.key, "B");

  // Exactly at the margin is not "clearly better", so the lock holds.
  assertEquals(relock(locked, [fakePlan("B", 1180), fakePlan("A", 1000)])!.key, "A");

  // The lock decays: A is re-priced downward, and B takes it without moving.
  const decayed = [fakePlan("B", 1100), fakePlan("A", 400)];
  assertEquals(relock(locked, decayed)!.key, "B");

  // A shape that is no longer reachable loses the lock outright.
  assertEquals(relock(locked, [fakePlan("C", 10)])!.key, "C");
  // Nothing to lock on to.
  assertEquals(relock(locked, []), null);
  // Nothing locked yet ⇒ the maximum, whatever order it arrives in.
  assertEquals(relock(null, [fakePlan("X", 1), fakePlan("Y", 9)])!.key, "Y");
});

// ---------------------------------------------------------------------------
// the policy that consumes it
// ---------------------------------------------------------------------------

/** A provider that switches the planner on and feeds it nothing else (C7P). */
const publicPlanner = () => ({ planner: true }) as Reads;

function discardsOf(hand: Tile[]): Action[] {
  return hand.map((tile) => ({ t: "discard", tile, riichi: false, tsumogiri: false }));
}

Deno.test("planner: the policy locks on, and the round boundary clears the lock", () => {
  const p = new AugmentedHeuristic("plan", 1, publicPlanner);
  assertEquals(p.plan, null, "nothing is planned before the first decision");

  const hand = tiles("123m456m789m12p55s");
  p.decide(obsOf({ hand, legal: discardsOf(hand) }));
  const first = p.plan;
  assert(first !== null, "a tenpai hand always has a future to lock on to");
  assertEquals(first.required, [THREE_P]);

  // Same 局, same hand: the lock is held (re-priced, same shape).
  p.decide(obsOf({ hand, legal: discardsOf(hand) }));
  assertEquals(p.plan!.key, first.key);

  // A different 局 with a hand that has no standard future clears it outright.
  const junk = tiles("139m258p47s東南西北");
  p.decide(obsOf({ hand: junk, kyoku: 1, legal: discardsOf(junk) }));
  assertEquals(p.plan, null, "a new 局 does not inherit the last one's shape");
});

Deno.test("planner: with no planner channel the policy never plans", () => {
  const p = new AugmentedHeuristic("blind", 1, () => null);
  const hand = tiles("123m456m789m12p55s");
  p.decide(obsOf({ hand, legal: discardsOf(hand) }));
  assertEquals(p.plan, null);
});

Deno.test("planner: the keep-set decides which tile goes", () => {
  // 一通 one tile short, plus a stray 東. The plan keeps thirteen tiles; the
  // fourteenth is the one that leaves.
  const hand = tiles("123m456m789m12p55s東");
  const p = new AugmentedHeuristic("plan", 1, publicPlanner);
  const act = p.decide(obsOf({ hand, legal: discardsOf(hand) }));
  assert(act.t === "discard");
  assertEquals(tileType(act.tile), 27, "the 東 is the only tile no future wants");
  assert(!p.plan!.keep.has(act.tile));
});

Deno.test("planner: plan discipline declines a call the plan never asked for", () => {
  // 234m 567m 89p 北北 55s + 3索. Pon-ing the 北 buys a shanten step — the base
  // policy takes it, because its 対々和 clause sees a route to a yaku that the
  // finished hand does not actually have. Every future the planner prices keeps
  // the 北 as the HEAD, so the call is simply not part of any of them.
  const hand = tiles("234m567m89p北北55s3s");
  const kita = tiles("北北北")[2];
  const pon: Action = { t: "pon", tiles: [hand[8], hand[9]], called: kita };
  const legal: Action[] = [pon, { t: "pass" }];

  const blind = new AugmentedHeuristic("blind", 1, () => null);
  assertEquals(
    blind.decide(obsOf({ hand, legal, claimTile: kita })).t,
    "pon",
    "the base policy takes the shanten step",
  );

  const planner = new AugmentedHeuristic("plan", 1, publicPlanner);
  const declined = planner.decide(obsOf({ hand, legal, claimTile: kita }));
  assert(planner.plan !== null);
  assertEquals(planner.plan!.callable, [], "the locked future is a closed one");
  assertEquals(declined.t, "pass", "so the call is off-plan, however fast it is");
});

Deno.test("planner: a call the plan DID ask for is taken", () => {
  // 456p already called and a 中 pair in hand: with the hand open, the 中 triplet
  // is the only yaku on offer, so the futures that survive the screen all want
  // that pon — and the 中 sorts last, which is exactly the case the winning-slot
  // rule has to keep callable.
  const hand = tiles("234m56m99s中中1p");
  const chun = tiles("中中中")[2];
  const pon: Action = { t: "pon", tiles: [hand[7], hand[8]], called: chun };
  const legal: Action[] = [pon, { t: "pass" }];

  const p = new AugmentedHeuristic("plan", 1, publicPlanner);
  const act = p.decide(obsOf({ hand, melds: [chi("456p")], legal, claimTile: chun }));
  assert(p.plan !== null);
  assert(p.plan!.callable.includes(33), "the plan asked for the 中");
  assertEquals(act.t, "pon");
});

// ---------------------------------------------------------------------------
// cost
// ---------------------------------------------------------------------------

Deno.test("planner: a decision stays well inside the per-move budget", () => {
  const hands = [
    tiles("123m456m789m12p55s"),
    tiles("2345m34567p345s東"),
    tiles("23456m3457p2345s"),
    tiles("1234m3456p78s東東南"),
    tiles("123m456m789m中中2p3p"),
  ].map((hand) => obsOf({ hand }));

  // Warm the JIT, then measure.
  for (const obs of hands) enumerateTargets(obs, availabilityFrom(obs, null));

  const reps = 200;
  const t0 = performance.now();
  for (let k = 0; k < reps; k++) {
    for (const obs of hands) enumerateTargets(obs, availabilityFrom(obs, null));
  }
  const us = (performance.now() - t0) * 1000 / (reps * hands.length);
  console.log(`    planner: ${us.toFixed(0)}µs / enumeration`);
  assertLessOrEqual(us, 5000, "a decision must stay well under 5ms");
});
