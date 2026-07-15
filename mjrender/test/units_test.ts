// Unit sanity checks for the trickiest pieces: shanten/ukeire, meld decode,
// tile notation, and a full end-to-end render of the bundled sample.

import { countsFromTiles, shanten, ukeireTypes } from "../src/shanten.ts";
import { decodeMeld } from "../src/meld.ts";
import { doraFromIndicatorType, isAka, tileType, typeGlyph } from "../src/tiles.ts";
import { render } from "../src/core.ts";
import { renderGame } from "../src/render.ts";
import type { Game } from "../src/model.ts";

function eq<T>(a: T, b: T, msg: string): void {
  const as = JSON.stringify(a), bs = JSON.stringify(b);
  if (as !== bs) throw new Error(`${msg}: got ${as}, want ${bs}`);
}

// "123m456p" → tile ids (first copy of each type)
function tiles(s: string): number[] {
  const out: number[] = [];
  let digits = "";
  for (const ch of s) {
    if (ch >= "0" && ch <= "9") digits += ch;
    else if ("mpsz".includes(ch)) {
      const base = ch === "m" ? 0 : ch === "p" ? 9 : ch === "s" ? 18 : 27;
      for (const d of digits) out.push((base + (Number(d) - 1)) * 4);
      digits = "";
    }
  }
  return out;
}
const sh = (s: string) => shanten(countsFromTiles(tiles(s)));

Deno.test("shanten: winning / tenpai / n-shanten", () => {
  eq(sh("123m456m789m11122p"), -1, "complete hand");
  eq(sh("123m456m789m99p13p"), 0, "kanchan tenpai");
  eq(sh("123m456m789m9p13p5s"), 1, "one-shanten");
});

Deno.test("shanten: chiitoitsu and kokushi", () => {
  eq(sh("1122334455667m8m"), 0, "chiitoitsu tenpai");
  eq(sh("19m19p19s1234567z"), 0, "kokushi 13-sided tenpai");
  eq(sh("119m19p19s1234567z"), -1, "kokushi complete");
});

Deno.test("ukeire: kanchan waits on 2p only", () => {
  eq(ukeireTypes(countsFromTiles(tiles("123m456m789m99p13p"))), [10], "waits = type 10 (2p)");
});

Deno.test("meld decode from 1.xml samples", () => {
  const pon = decodeMeld(1, 50249);
  eq([pon.kind, pon.calledTile, pon.fromWho], ["pon", 131, 2], "50249 = pon 發 from P2");
  const pon2 = decodeMeld(0, 47658);
  eq([pon2.kind, pon2.fromWho], ["pon", 2], "47658 = pon 白 from P2");
  const chi = decodeMeld(0, 13343);
  eq([chi.kind, chi.calledTile, chi.fromWho], ["chi", 20, 3], "13343 = chi 567m called 6m from P3");
});

Deno.test("tiles: aka, glyphs, dora successor", () => {
  eq([isAka(16), isAka(52), isAka(88), isAka(20)], [true, true, true, false], "aka ids");
  eq([typeGlyph(0), typeGlyph(9), typeGlyph(27), typeGlyph(33)], ["一", "①", "東", "中"], "glyphs");
  eq(doraFromIndicatorType(tileType(53)), 14, "5p indicator → 6p dora"); // type13→14
  eq(doraFromIndicatorType(8), 0, "9m indicator → 1m dora (wrap)");
  eq(doraFromIndicatorType(30), 27, "North indicator → East dora (wrap)");
  eq(doraFromIndicatorType(33), 31, "中 indicator → 白 dora (wrap)");
});

Deno.test("kan turn integrates into one line; ankan dora before rinshan draw", () => {
  // P0 holds four 東 (108-111). Sequence: draw 4m, ankan 東, new-dora reveal,
  // rinshan draw 5m, tsumogiri 5m. For ankan the dora is revealed BEFORE the
  // rinshan draw, so it must appear left of "嶺上ツモ" in the integrated line.
  const game: Game = {
    version: "2.3",
    rules: { raw: 0, aka: true, kuitan: true, sanma: false, hanchan: true },
    players: [0, 1, 2, 3].map((seat) => ({ seat, name: `P${seat}` })),
    rounds: [{
      kyoku: 0,
      honba: 0,
      kyotaku: 0,
      dealer: 0,
      dice: [1, 1],
      startScores: [250, 250, 250, 250],
      startHands: [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 108, 109, 110, 111],
        [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
        [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53],
        [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
      ],
      firstDora: 104,
      events: [
        { t: "draw", who: 0, tile: 12, rinshan: false },
        { t: "call", meld: { kind: "ankan", who: 0, fromWho: 0, tiles: [108, 109, 110, 111], calledTile: 108 } },
        { t: "dora", indicator: 20 },
        { t: "draw", who: 0, tile: 17, rinshan: true },
        { t: "discard", who: 0, tile: 17, tsumogiri: true, riichi: false },
      ],
      results: [],
    }],
  };

  const warns: string[] = [];
  const orig = console.error;
  console.error = (...a: unknown[]) => warns.push(a.join(" "));
  let text: string;
  try {
    text = renderGame(game, { hands: "key" });
  } finally {
    console.error = orig;
  }
  eq(warns, [], "no replay inconsistencies");
  const line = text.split("\n").find((l) => l.includes("暗槓")) ?? "";
  if (!line.includes("P0 暗槓東東東東  ＋新ドラ六(→ドラ七)  嶺上ツモ 五  → 打 五(ツモ切り)")) {
    throw new Error(`unexpected kan line: ${line}`);
  }
  if (line.indexOf("＋新ドラ") > line.indexOf("嶺上ツモ")) {
    throw new Error(`ankan dora must precede the rinshan draw: ${line}`);
  }
});

Deno.test("end-to-end: sample renders, no inconsistency warnings", async () => {
  const warns: string[] = [];
  const orig = console.error;
  console.error = (...a: unknown[]) => warns.push(a.join(" "));
  let text: string;
  try {
    text = await render(new URL("../../1.xml", import.meta.url).pathname, { hands: "key" });
  } finally {
    console.error = orig;
  }
  if (!text.includes("◆終局")) throw new Error("missing 終局 block");
  if (!text.includes("〔解説ポイント:")) throw new Error("missing commentary anchors");
  // seat winds (自風) label every 配牌 line; the dealer is 東家・親
  if (!text.includes("(東家・親):")) throw new Error("missing dealer seat wind");
  for (const w of ["(南家):", "(西家):", "(北家):"]) {
    if (!text.includes(w)) throw new Error(`missing seat wind ${w}`);
  }
  // 巡目: the go-around opener carries "1巡 "; riichi keeps its explicit timing
  if (!text.includes("1巡 ")) throw new Error("missing 巡目 marker");
  if (!/\d+巡目リーチ宣言/.test(text)) throw new Error("missing junme on riichi");
  eq(warns, [], "no [warn] inconsistencies during replay");
});
