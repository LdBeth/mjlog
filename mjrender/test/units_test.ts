// Unit sanity checks for the trickiest pieces: shanten/ukeire, meld decode,
// tile notation, and a full end-to-end render of the bundled sample.

import { countsFromTiles, shanten, ukeireTypes } from "../src/shanten.ts";
import { decodeMeld } from "../src/meld.ts";
import { doraFromIndicatorType, isAka, tileType, typeGlyph } from "../src/tiles.ts";
import { render } from "../src/core.ts";

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
  eq(warns, [], "no [warn] inconsistencies during replay");
});
