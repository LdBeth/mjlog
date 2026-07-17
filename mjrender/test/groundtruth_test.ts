// End-of-hand ground truth: 見逃し detection (with the winning-discard
// exclusion) and the furiten flag on the result block.

import { renderGame } from "../src/render.ts";
import type { Game, GameEvent, RoundResult } from "../src/model.ts";

// P1: 123m 456m 789m 99p 13p — tenpai from the deal, waiting on 2p (type 10).
const P1_TENPAI = [0, 4, 8, 12, 16, 20, 24, 28, 32, 68, 69, 36, 44];
// The other seats: honor-heavy junk that is nowhere near tenpai.
const JUNK = [
  [52, 56, 60, 76, 80, 84, 100, 108, 112, 116, 120, 124, 128],
  [53, 57, 61, 77, 81, 85, 101, 109, 113, 117, 121, 125, 129],
  [54, 58, 62, 78, 82, 86, 102, 110, 114, 118, 122, 126, 130],
];

function gameWith(events: GameEvent[], results: RoundResult[]): Game {
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
      startHands: [JUNK[0], P1_TENPAI, JUNK[1], JUNK[2]],
      firstDora: 132,
      events,
      results,
    }],
  };
}

function renderQuiet(g: Game, snapshots: "none" | "inline" = "none"): string {
  const orig = console.error;
  console.error = () => {};
  try {
    return renderGame(g, { hands: "key", snapshots });
  } finally {
    console.error = orig;
  }
}

Deno.test("ground truth: passed win tile is a 見逃し; own-wait discard flags 振聴", () => {
  const text = renderQuiet(gameWith(
    [
      // P0 draws and throws 2p — P1 could ron it: a miss.
      { t: "draw", who: 0, tile: 40, rinshan: false },
      { t: "discard", who: 0, tile: 40, tsumogiri: true, riichi: false },
      // P1 draws the other 2p and throws it back: now furiten.
      { t: "draw", who: 1, tile: 41, rinshan: false },
      { t: "discard", who: 1, tile: 41, tsumogiri: true, riichi: false },
    ],
    // abortive type skips the exhaustive-draw tenpai cross-check (JUNK hands
    // never appear in a real log's tenpai list)
    [{ kind: "ryuukyoku", type: "yao9", sc: [], tenpaiHands: [] }],
  ));
  if (!text.includes("◇見逃し: P1が1巡目 P0の②をロンせず（待ち②）")) {
    throw new Error(
      `missing 見逃し line:\n${text.split("\n").filter((l) => l.includes("◇")).join("\n")}`,
    );
  }
  // (id 16 in the crafted hand is the red 5m, hence ドラ1)
  if (!/P1: .*〔聴牌 待ち② 残2枚 ドラ1〕（振聴）/.test(text)) {
    throw new Error("missing 振聴 flag on P1's result-time hand");
  }
});

Deno.test("inline render folds 結果時点 into the final snapshot, 振聴 mark included", () => {
  const text = renderQuiet(
    gameWith(
      [
        { t: "draw", who: 0, tile: 40, rinshan: false },
        { t: "discard", who: 0, tile: 40, tsumogiri: true, riichi: false },
        { t: "draw", who: 1, tile: 41, rinshan: false },
        { t: "discard", who: 1, tile: 41, tsumogiri: true, riichi: false },
      ],
      [{ kind: "ryuukyoku", type: "yao9", sc: [], tenpaiHands: [] }],
    ),
    "inline",
  );
  // the legend still *describes* the block (「◇結果時点の各家手牌」=…); only the
  // block line itself (with the colon) must be gone
  if (text.includes("◇結果時点の各家手牌:")) {
    throw new Error("inline render must omit the 結果時点 block (the snapshot carries it)");
  }
  if (!/│手牌 P1: .*（振聴）/.test(text)) {
    throw new Error("snapshot 手牌 line missing the 振聴 mark");
  }
});

Deno.test("ground truth: the actually-ronned discard is not a 見逃し", () => {
  const machi = 40;
  const text = renderQuiet(gameWith(
    [
      { t: "draw", who: 0, tile: machi, rinshan: false },
      { t: "discard", who: 0, tile: machi, tsumogiri: true, riichi: false },
    ],
    [{
      kind: "agari",
      who: 1,
      fromWho: 0,
      machi,
      hand: [...P1_TENPAI, machi],
      melds: [],
      fu: 40,
      points: 2600,
      limit: 0,
      yaku: [{ id: 8, han: 1 }],
      yakuman: [],
      doraHai: [132],
      uraDoraHai: [],
      sc: [250, -26, 250, 26, 250, 0, 250, 0],
    }],
  ));
  // match an actual record line, not the format-preamble bullet mentioning ◇見逃し
  if (/^◇見逃し:/m.test(text)) throw new Error("winning ron flagged as 見逃し");
  if (!text.includes("P1(和了):")) throw new Error("missing winner mark in ground truth");
});
