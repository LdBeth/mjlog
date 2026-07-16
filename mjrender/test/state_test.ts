// BoardState replay-engine tests: river reconstruction (tedashi / tsumogiri /
// riichi tile / called-away), live scores across riichi, wall countdown, and
// positional replay (replayTo by eventIndex / junme).

import { BoardState, replayTo } from "../src/state.ts";
import type { Game, Round } from "../src/model.ts";

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
