// Regression fixtures for two legal-enumeration bugs:
//   1. a chi/pon whose 同巡内食い替え ban would leave the caller with no legal
//      discard must not be offered (an empty turn action list soft-locks the
//      round loop),
//   2. when the drawn tile's type is also held, BOTH the tsumogiri and the
//      tedashi discard must be enumerated,
//   3. chi/pon options are distinct by tile *kind* (type + aka), not id: two
//      plain copies of a tile must not double the offered calls.

import { assert, assertEquals } from "@std/assert";
import type { Tile } from "mjrender/model.ts";
import { tileType } from "mjrender/tiles.ts";
import { buildMeld, claimActions, kuikaeTypes, turnActions } from "../src/legal.ts";
import { JANKI } from "../src/rules.ts";
import { sfc32 } from "../src/rng.ts";
import { Table } from "../src/table.ts";
import type { Action, Seat } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { Wall } from "../src/wall.ts";
import { tiles } from "./helpers.ts";

function makeTable(): Table {
  const rng = sfc32(7);
  return new Table(
    {
      kyoku: 0,
      honba: 0,
      kyotaku: 0,
      dealer: 0,
      scores: [30000, 30000, 30000, 30000],
      wall: Wall.shuffled(rng),
      dice: [0, 0],
    },
    JANKI,
    SEATS.map((seat) => ({ seat, name: `P${seat}` })),
  );
}

function setHand(t: Table, seat: Seat, ids: Tile[]): void {
  t.hands[seat].length = 0;
  t.hands[seat].push(...ids);
}

/** Give `seat` `n` open chi melds out of tiles nobody else in the fixture uses. */
function giveMelds(t: Table, seat: Seat, specs: string[]): void {
  for (const spec of specs) {
    const ids = tiles(spec);
    const meld = buildMeld(seat, ((seat + 3) % 4) as Seat, {
      t: "chi",
      tiles: [ids[0], ids[1]],
      called: ids[2],
    }, t);
    t.emit({ t: "call", meld }, { e: "call", meld });
  }
}

/** Plain (non-riichi) discard actions. */
const discards = (as: Action[]) => as.filter((a) => a.t === "discard" && !a.riichi);

Deno.test("claim: chi that leaves no legal discard is not offered", () => {
  const t = makeTable();
  // Seat 1 with three melds and 5m6m7m8m concealed. Kamicha (seat 0) cuts a 5m:
  // chi 6m7m bans 5m and 8m (the 4m8m suji swap), leaving nothing discardable.
  giveMelds(t, 1, ["123p", "456p", "789p"]);
  setHand(t, 1, tiles("5678m"));
  const called = tiles("5m5m")[1]; // a different 5m id than the held one

  const legal = claimActions(t, 1, called, 0);
  const chis = legal.filter((a) => a.t === "chi");
  assertEquals(chis.length, 0, "the self-locking chi must not be offered");

  // Sanity: the ban really is total, i.e. the fixture reproduces the bug.
  const ban = kuikaeTypes({ t: "chi", tiles: [tiles("5678m")[1], tiles("5678m")[2]], called });
  assertEquals([...ban].sort((a, b) => a - b), [
    tileType(tiles("5m")[0]),
    tileType(tiles("8m")[0]),
  ]);
});

Deno.test("claim: a chi that still leaves a discard stays available", () => {
  const t = makeTable();
  giveMelds(t, 1, ["123p", "456p", "789p"]);
  setHand(t, 1, tiles("678m9s")); // 9s survives the 5m/8m ban
  const called = tiles("5m")[0];

  const chis = claimActions(t, 1, called, 0).filter((a) => a.t === "chi");
  assertEquals(chis.length, 1);
});

Deno.test("turn: the withheld chi is exactly the one that would empty the turn", () => {
  const t = makeTable();
  giveMelds(t, 1, ["123p", "456p", "789p"]);
  const hand = tiles("5678m");
  setHand(t, 1, hand);
  const called = tiles("5m5m")[1];
  const chi: Action = { t: "chi", tiles: [hand[1], hand[2]], called };

  // Nothing but pass is on offer here…
  for (const a of claimActions(t, 1, called, 0)) {
    if (a.t === "chi" || a.t === "pon") {
      assert(false, `offered a call with no follow-up discard: ${a.t}`);
    }
  }

  // …because forcing that call produces the soft-lock: no discard, no kan,
  // nothing for the TUI to select or a policy to return.
  const meld = buildMeld(1, 0, chi, t);
  t.emit({ t: "call", meld }, { e: "call", meld });
  t.kuikaeBan = { seat: 1, types: kuikaeTypes(chi) };
  assertEquals(turnActions(t, 1, null).length, 0);
});

Deno.test("turn: drawn type also held yields both tsumogiri and tedashi", () => {
  const t = makeTable();
  const hand = tiles("1112345678999m"); // 14 tiles, the drawn one is a 1m copy
  setHand(t, 0, hand);
  const drawn = hand[2]; // third 1m

  const ds = discards(turnActions(t, 0, drawn));
  const ones = ds.filter((a) => a.t === "discard" && tileType(a.tile) === tileType(drawn));
  assertEquals(ones.length, 2, "one tsumogiri and one tedashi candidate for the drawn type");
  assert(ones.some((a) => a.t === "discard" && a.tile === drawn && a.tsumogiri));
  assert(ones.some((a) => a.t === "discard" && a.tile !== drawn && !a.tsumogiri));
  // Held representative first, drawn id second (ordering is load-bearing).
  assert(ones[0].t === "discard" && !ones[0].tsumogiri);
  assert(ones[1].t === "discard" && ones[1].tsumogiri);
  // Exactly one held representative, as before: 1m2m…9m plus the drawn copy.
  assertEquals(ds.length, 10);
});

Deno.test("claim: duplicate copies in hand do not duplicate chi options", () => {
  const t = makeTable();
  // Two 6m and two 7m used to enumerate 2×2 identical 6m7m chis.
  setHand(t, 1, tiles("66778m9s"));
  const called = tiles("5m")[0];

  const chis = claimActions(t, 1, called, 0).filter((a) => a.t === "chi");
  assertEquals(chis.length, 1, "one chi per distinct shape, not per id combination");
});

Deno.test("claim: aka and plain five stay distinct chi options", () => {
  const t = makeTable();
  // tiles() hands out 5p copies in ascending id order and ids 52/53 are aka,
  // so "555p" is aka, aka, plain: two aka must collapse to one option while
  // the plain copy stays separately offered.
  setHand(t, 1, tiles("555p6p9s"));
  const called = tiles("4p")[0];

  const chis = claimActions(t, 1, called, 0).filter((a) => a.t === "chi");
  assertEquals(chis.length, 2, "spend-the-aka and keep-the-aka, nothing else");
});

Deno.test("claim: pon pairs are distinct by aka composition, not id", () => {
  const t = makeTable();

  // Three plain copies (5s has no aka): one pon, not three.
  setHand(t, 1, tiles("555s9m9p"));
  const calledS = tiles("5555s")[3];
  const ponsS = claimActions(t, 1, calledS, 0).filter((a) => a.t === "pon");
  assertEquals(ponsS.length, 1);

  // aka, aka, plain 5p: pon with two aka and pon with aka+plain — the two
  // aka-aka id pairs collapse, the aka-plain choice survives.
  setHand(t, 2, tiles("555p9m9p"));
  const calledP = tiles("5555p")[3]; // id 55, a plain copy
  const ponsP = claimActions(t, 2, calledP, 0).filter((a) => a.t === "pon");
  assertEquals(ponsP.length, 2);
});

Deno.test("turn: drawn type held only once yields a single tsumogiri candidate", () => {
  const t = makeTable();
  const hand = tiles("123456789m1234p"); // 14 tiles, all distinct types
  setHand(t, 0, hand);
  const drawn = hand[hand.length - 1]; // 4p, sole copy

  const ds = discards(turnActions(t, 0, drawn));
  assertEquals(ds.length, 13);
  const fours = ds.filter((a) => a.t === "discard" && tileType(a.tile) === tileType(drawn));
  assertEquals(fours.length, 1);
  assert(fours[0].t === "discard" && fours[0].tsumogiri);
});
