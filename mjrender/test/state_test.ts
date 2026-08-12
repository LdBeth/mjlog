// BoardState replay-engine tests: river reconstruction (tedashi / tsumogiri /
// riichi tile / called-away), live scores across riichi, wall countdown, and
// positional replay (replayTo by eventIndex / junme).

import { BoardState, replayTo } from "../src/state.ts";
import type { Game, Meld, Round } from "../src/model.ts";

function eq<T>(a: T, b: T, msg: string): void {
  const as = JSON.stringify(a), bs = JSON.stringify(b);
  if (as !== bs) throw new Error(`${msg}: got ${as}, want ${bs}`);
}

// Minimal 4p game scaffold around a single crafted round.
function gameWith(round: Partial<Round>): Game {
  return {
    version: "2.3",
    rules: { raw: 0, aka: true, kuitan: true, sanma: false, hanchan: true },
    players: [0, 1, 2, 3].map((seat) => ({ seat, name: `p${seat}` })),
    rounds: [{
      kyoku: 0,
      honba: 0,
      kyotaku: 0,
      dealer: 0,
      dice: [1, 1],
      startScores: [250, 250, 250, 250],
      startHands: [
        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
        [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49],
        [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50],
        [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51],
      ],
      firstDora: 104,
      events: [],
      results: [],
      ...round,
    }],
  };
}

Deno.test("state: river records junme, tsumogiri/tedashi, riichi tile, calledBy", () => {
  const g = gameWith({
    events: [
      { t: "draw", who: 0, tile: 100, rinshan: false }, // junme → 1
      { t: "discard", who: 0, tile: 100, tsumogiri: true, riichi: false },
      { t: "draw", who: 1, tile: 101, rinshan: false },
      { t: "discard", who: 1, tile: 48, tsumogiri: false, riichi: false }, // tedashi 4s(48>>2=12? no: 48>>2=12 → 4p)
      // P2 pons P1's discard
      { t: "call", meld: { kind: "pon", who: 2, fromWho: 1, tiles: [48, 49, 50], calledTile: 48 } },
      { t: "discard", who: 2, tile: 50, tsumogiri: false, riichi: false },
      { t: "draw", who: 3, tile: 102, rinshan: false },
      { t: "reach", who: 3, step: 1 },
      { t: "discard", who: 3, tile: 102, tsumogiri: true, riichi: true },
      { t: "reach", who: 3, step: 2, scores: [250, 250, 250, 240] },
    ],
  });
  const st = new BoardState(g, g.rounds[0]);
  for (const e of g.rounds[0].events) st.applyEvent(e);

  eq(st.rivers[0].map((r) => [r.tile, r.junme, r.tsumogiri]), [[100, 1, true]], "P0 tsumogiri");
  eq(st.rivers[1][0].tsumogiri, false, "P1 discard is tedashi");
  eq(st.rivers[1][0].calledBy, 2, "P1's tile marked called by P2");
  eq(st.rivers[3][0].riichiDeclare, true, "P3 riichi tile marked");
  eq(st.riichiActive[3], true, "P3 riichi active");
  eq(st.riichiJunme[3], 1, "P3 riichi in junme 1");
  eq(st.scores, [250, 250, 250, 240], "riichi stick debited from live scores");
  eq(st.wallRemaining, 70 - 3, "three wall draws consumed");
});

Deno.test("state: reach step-2 without ten attr falls back to -1000", () => {
  const g = gameWith({
    events: [
      { t: "draw", who: 1, tile: 101, rinshan: false },
      { t: "reach", who: 1, step: 1 },
      { t: "discard", who: 1, tile: 101, tsumogiri: true, riichi: true },
      { t: "reach", who: 1, step: 2 },
    ],
  });
  const st = new BoardState(g, g.rounds[0]);
  for (const e of g.rounds[0].events) st.applyEvent(e);
  eq(st.scores, [250, 240, 250, 250], "fallback stick debit");
});

// Meld helpers: tile ids are type*4+copy, so `ids(type, n)` names n copies.
const ids = (type: number, n: number) => Array.from({ length: n }, (_, i) => type * 4 + i);
const pon = (who: number, type: number, fromWho = (who + 3) % 4): Meld => ({
  kind: "pon",
  who,
  fromWho,
  tiles: ids(type, 3),
  calledTile: type * 4,
});
const chi = (who: number, start: number, fromWho = (who + 3) % 4): Meld => ({
  kind: "chi",
  who,
  fromWho,
  tiles: [start * 4, (start + 1) * 4, (start + 2) * 4],
  calledTile: start * 4,
});

Deno.test("state: furoThreats activate on a yakuhai meld or a second meld", () => {
  const g = gameWith({
    events: [
      // P0: two chi AND a riichi — riichi seats are owned by threats(), so they
      // must never be double-reported as a furo threat.
      { t: "call", meld: chi(0, 0) },
      { t: "call", meld: chi(0, 3) },
      { t: "draw", who: 0, tile: 100, rinshan: false },
      { t: "reach", who: 0, step: 1 },
      { t: "discard", who: 0, tile: 100, tsumogiri: true, riichi: true },
      { t: "reach", who: 0, step: 2 },
      { t: "call", meld: pon(1, 31) }, // 白 = yakuhai for everyone ⇒ one meld is enough
      { t: "call", meld: chi(2, 6) }, // a lone chi commits to nothing
      { t: "call", meld: chi(3, 0) },
      { t: "call", meld: chi(3, 3) }, // two melds ⇒ threat
    ],
  });
  const st = new BoardState(g, g.rounds[0]);
  for (const e of g.rounds[0].events) st.applyEvent(e);

  eq(st.furoThreats(2).map((f) => [f.seat, f.openMeldCount]), [[1, 1], [3, 2]], "seats + counts");
  eq(st.furoThreats(1).map((f) => f.seat), [3], "a single chi is not a threat");
  eq(st.furoThreats(3).map((f) => f.seat), [1], "riichi seat excluded, own seat excluded");
  eq([...st.furoThreats(0)[0].yakuhaiMelds], [31], "白 pon recorded as a yakuhai meld");
});

Deno.test("state: furoThreats read honitsu, toitoi, and dora inside melds", () => {
  const base = (events: Round["events"]) => {
    const g = gameWith({ events });
    const st = new BoardState(g, g.rounds[0]);
    for (const e of g.rounds[0].events) st.applyEvent(e);
    return st;
  };
  // firstDora 104 = 9s indicator ⇒ dora type 18 (1s).
  const soup = base([
    { t: "call", meld: pon(1, 32) }, // 發 (honor meld — does not break a flush)
    { t: "call", meld: chi(1, 18) }, // 1s2s3s, and 1s is dora ×1
    { t: "draw", who: 1, tile: 100, rinshan: false },
    { t: "discard", who: 1, tile: 100, tsumogiri: true, riichi: false }, // id100 = type 25 = 8s
  ]);
  eq(soup.furoThreats(0)[0].honitsuSuit, "s", "melds are honors + one suit, river almost clean");
  eq(soup.furoThreats(0)[0].meldDora, 1, "the 1s in the chi is dora");

  const dirty = base([
    { t: "call", meld: pon(1, 32) },
    { t: "call", meld: chi(1, 18) },
    { t: "draw", who: 1, tile: 100, rinshan: false },
    { t: "discard", who: 1, tile: 100, tsumogiri: true, riichi: false },
    { t: "draw", who: 1, tile: 101, rinshan: false },
    { t: "discard", who: 1, tile: 101, tsumogiri: true, riichi: false },
  ]);
  eq(dirty.furoThreats(0)[0].honitsuSuit, null, "two sou tiles in their own river kills the read");

  const toitoi = base([
    { t: "call", meld: pon(1, 32) },
    { t: "call", meld: pon(1, 18) },
  ]);
  eq([toitoi.furoThreats(0)[0].toitoi, toitoi.furoThreats(0)[0].meldDora], [true, 3], "two pons");

  // An ankan keeps the hand closed: it neither counts as 副露 nor activates.
  const ankanOnly = base([
    {
      t: "call",
      meld: { kind: "ankan", who: 1, fromWho: 1, tiles: ids(18, 4), calledTile: 72 },
    },
  ]);
  eq(ankanOnly.furoThreats(0).length, 0, "ankan alone is not an open hand");
  const ankanPlus = base([
    { t: "call", meld: { kind: "ankan", who: 1, fromWho: 1, tiles: ids(18, 4), calledTile: 72 } },
    { t: "call", meld: pon(1, 32) },
    { t: "call", meld: chi(1, 6) },
  ]);
  eq(
    [ankanPlus.furoThreats(0)[0].openMeldCount, ankanPlus.furoThreats(0)[0].meldDora],
    [2, 4],
    "ankan excluded from 副露N but its four dora still count",
  );
});

Deno.test("state: replayTo by eventIndex and by junme", () => {
  const g = gameWith({
    events: [
      { t: "draw", who: 0, tile: 100, rinshan: false }, // junme 1
      { t: "discard", who: 0, tile: 100, tsumogiri: true, riichi: false },
      { t: "draw", who: 1, tile: 101, rinshan: false },
      { t: "discard", who: 1, tile: 101, tsumogiri: true, riichi: false },
      { t: "draw", who: 2, tile: 102, rinshan: false },
      { t: "discard", who: 2, tile: 102, tsumogiri: true, riichi: false },
      { t: "draw", who: 3, tile: 103, rinshan: false },
      { t: "discard", who: 3, tile: 103, tsumogiri: true, riichi: false },
      { t: "draw", who: 0, tile: 104, rinshan: false }, // junme 2 begins
      { t: "discard", who: 0, tile: 104, tsumogiri: true, riichi: false },
    ],
  });
  const r = g.rounds[0];

  const deal = replayTo(g, r, { eventIndex: -1 });
  eq(deal.rivers.map((rv) => rv.length), [0, 0, 0, 0], "eventIndex -1 = deal only");
  eq(deal.junme, 0, "no draws yet");

  const afterFirst = replayTo(g, r, { eventIndex: 1 });
  eq(afterFirst.rivers[0].length, 1, "P0 discard applied at eventIndex 1");
  eq(afterFirst.hands[0].length, 13, "hand back to 13");

  const endJ1 = replayTo(g, r, { junme: 1 });
  eq(endJ1.junme, 1, "stopped inside junme 1");
  eq(endJ1.rivers.map((rv) => rv.length), [1, 1, 1, 1], "all four junme-1 discards applied");
  eq(endJ1.wallRemaining, 66, "four draws consumed");

  const all = replayTo(g, r, { junme: 99 });
  eq(all.rivers[0].length, 2, "out-of-range junme clamps to full round");
});
