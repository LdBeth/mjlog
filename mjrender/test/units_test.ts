// Unit sanity checks for the trickiest pieces: shanten/ukeire, meld decode,
// tile notation, and a full end-to-end render of the bundled sample.

import { countsFromTiles, shanten, ukeireTypes } from "../src/shanten.ts";
import { assessDanger } from "../src/danger.ts";
import { decodeMeld } from "../src/meld.ts";
import { doraFromIndicatorType, isAka, tileType, typeGlyph } from "../src/tiles.ts";
import { render } from "../src/core.ts";
import { renderGame } from "../src/render.ts";
import { parseGame } from "../src/parse.ts";
import type { AgariResult, Game } from "../src/model.ts";

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

Deno.test("danger: yakuhai honors read hotter than guest honors", () => {
  const value = new Set<number>([27, 31, 32, 33]); // round-east wind + 3 dragons
  const threat = (safe: number[] = []) => [
    { seat: 1, safeTypes: new Set<number>(safe), valueHonors: value },
  ];
  const seen = (t: number, n: number) => {
    const v = new Array<number>(34).fill(0);
    v[t] = n;
    return v;
  };
  const level = (type: number, visible: number[], safe: number[] = []) =>
    assessDanger(type, threat(safe), visible)!.level;

  // 白 (type 31) is a dragon = yakuhai: fully live ⇒ 危険度高
  eq(level(31, seen(31, 0)), "危険度高", "live dragon = high");
  // guest wind (南=28, not a value honor here) fully live ⇒ 危険度中
  eq(level(28, seen(28, 0)), "危険度中", "guest wind = mid");
  // two copies already public ⇒ shanpon unlikely ⇒ 危険度中 even for a dragon
  eq(level(31, seen(31, 2)), "危険度中", "dragon w/ 2 out = mid");
  // three out ⇒ at most a tanki ⇒ 危険度低
  eq(level(31, seen(31, 3)), "危険度低", "dragon w/ 3 out = low");
  // genbutsu (in safeTypes) ⇒ 安全
  eq(level(31, seen(31, 0), [31]), "安全", "genbutsu = safe");
});

Deno.test("danger: evidence notes carry suji, kabe/chance, and live counts", () => {
  const threat = (safe: number[] = []) => [
    { seat: 1, safeTypes: new Set<number>(safe), valueHonors: new Set<number>([31]) },
  ];
  const notes = (type: number, visible: number[], safe: number[] = [], own?: number[]) =>
    assessDanger(type, threat(safe), visible, own)!.details[0].notes;
  const vis = (pairs: Array<[number, number]>) => {
    const v = new Array<number>(34).fill(0);
    for (const [t, n] of pairs) v[t] = n;
    return v;
  };

  // 5m (type 4) with 2m and 8m safe = double suji
  eq(notes(4, vis([]), [1, 7]), ["スジ", "生牌"], "full suji + fresh tile");
  // 5m, all four 4m visible and three 6m visible: lower ryanmen dead, upper has
  // one bridging copy left ⇒ ワンチャンス
  eq(notes(4, vis([[3, 4], [5, 3]])), ["無スジ", "ワンチャンス", "生牌"], "one-chance via kabe");
  // both bridging ranks dead on both sides ⇒ ノーチャンス
  eq(notes(4, vis([[3, 4], [6, 4]])), ["無スジ", "ノーチャンス", "生牌"], "no-chance");
  // the discarder's own tiles count toward the wall they can see
  eq(
    notes(4, vis([[3, 2], [6, 4]]), [], vis([[3, 2]])),
    ["無スジ", "ノーチャンス", "生牌"],
    "own hand completes the kabe",
  );
  // yakuhai honor: kind + live-count evidence
  eq(notes(31, vis([[31, 1]])), ["役牌", "場に1枚"], "dragon with one out");
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
  if (!line.includes("P0 暗槓東東東東  ＋新ドラ六(→ドラ七)  嶺上ツモ 五  → 五 ▽")) {
    throw new Error(`unexpected kan line: ${line}`);
  }
  if (line.indexOf("＋新ドラ") > line.indexOf("嶺上ツモ")) {
    throw new Error(`ankan dora must precede the rinshan draw: ${line}`);
  }
});

// --- parse.ts XML edge cases (previously zero coverage) ---
// These parse a minimal but valid <mjloggm> string through the *real* parser
// rather than hand-building a Game object, so the parser itself is exercised.

// 13 sequential tile ids from a base, comma-joined: a valid-shape (if arbitrary)
// starting hand — parse.ts does not validate deal legality.
const hai13 = (base: number) => Array.from({ length: 13 }, (_, i) => base + i).join(",");

Deno.test("parse: double ron yields multiple agari results with per-sc scoring", () => {
  // P0 discards 64; P1 and P2 both ron it — two <AGARI> in one round, the last
  // carrying `owari`. The TS parser has no endRound (the C++ "only the first
  // triggers endRound" note doesn't apply here); both AGARI must be captured.
  const xml = `<mjloggm ver="2.3">
<GO type="169"/>
<UN n0="alice" n1="bob" n2="carol" n3="dave" dan="10,11,12,13" rate="1500,1600,1700,1800"/>
<INIT seed="0,0,0,0,0,4" ten="250,250,250,250" oya="0" hai0="${hai13(0)}" hai1="${hai13(16)}" hai2="${
    hai13(32)
  }" hai3="${hai13(48)}"/>
<T64/><D64/>
<AGARI ba="0,0" hai="16,17,18" machi="64" ten="30,3900,0" who="1" fromWho="0" sc="250,-39,250,39,250,0,250,0"/>
<AGARI ba="0,0" hai="32,33,34" machi="64" ten="40,5200,0" who="2" fromWho="0" sc="250,0,250,0,250,52,250,0" owari="211,-45.0,289,15.0,250,0.0,250,30.0"/>
</mjloggm>`;
  const g = parseGame(xml);
  eq(g.rounds.length, 1, "one round");
  eq(g.rounds[0].results.length, 2, "both AGARI captured (parser doesn't stop at the first)");
  const res = g.rounds[0].results as AgariResult[];
  eq(
    res.map((r) => [r.kind, r.who, r.fromWho]),
    [["agari", 1, 0], ["agari", 2, 0]],
    "two rons, both off P0",
  );
  eq(res[0].sc, [250, -39, 250, 39, 250, 0, 250, 0], "first ron sc preserved verbatim");
  eq(res[1].points, 5200, "second ron points");
  eq(g.owari, [211, -45, 289, 15, 250, 0, 250, 30], "owari taken from the AGARI that carries it");

  // Rendered scores follow the raw sc (delta*100): +3900 to P1, +5200 to P2,
  // and two separate 和了 blocks. (Synthetic hands mismatch the reconstruction,
  // so the consistency guard warns — expected here; we capture and ignore it.)
  const orig = console.error;
  console.error = () => {};
  let text: string;
  try {
    text = renderGame(g, { hands: "key" });
  } finally {
    console.error = orig;
  }
  eq((text.match(/◆和了/g) ?? []).length, 2, "two 和了 blocks rendered");
  if (!text.includes("P1 +3900")) throw new Error("missing P1 +3900 score delta");
  if (!text.includes("P2 +5200")) throw new Error("missing P2 +5200 score delta");
});

Deno.test("parse: UN reconnect with missing n{i} preserves omitted seats", () => {
  // A mid-game <UN> carrying only n2 (a seat 2 reconnect) must update seat 2 and
  // leave the other seats' names — and their dan/rate the reconnect omits — intact.
  const xml = `<mjloggm ver="2.3">
<GO type="169"/>
<UN n0="alice" n1="bob" n2="carol" n3="dave" dan="10,11,12,13" rate="1500,1600,1700,1800" sx="M,M,F,M"/>
<INIT seed="0,0,0,0,0,4" ten="250,250,250,250" oya="0" hai0="${hai13(0)}" hai1="${hai13(16)}" hai2="${
    hai13(32)
  }" hai3="${hai13(48)}"/>
<T64/><D64/>
<UN n2="carolX"/>
<U80/><E80/>
</mjloggm>`;
  const g = parseGame(xml);
  eq(g.players.map((p) => p.name), ["alice", "bob", "carolX", "dave"], "only seat 2 renamed");
  eq(g.players.map((p) => p.dan), ["10", "11", "12", "13"], "dan not clobbered by dan-less reconnect UN");
  eq(g.players[2].rate, 1700, "rate not clobbered for the reconnecting seat");
});

Deno.test("parse: BYE (disconnect) is ignored and does not corrupt replay", () => {
  // BYE has no case in the parser's switch, so it's silently skipped. This only
  // guards that an unknown mid-round element doesn't disturb the surrounding
  // draw/discard replay — essentially confirming the no-op default path.
  const xml = `<mjloggm ver="2.3">
<GO type="169"/>
<UN n0="alice" n1="bob" n2="carol" n3="dave" dan="10,11,12,13"/>
<INIT seed="0,0,0,0,0,4" ten="250,250,250,250" oya="0" hai0="${hai13(0)}" hai1="${hai13(16)}" hai2="${
    hai13(32)
  }" hai3="${hai13(48)}"/>
<T64/><BYE who="1"/><D64/><U80/><E80/>
</mjloggm>`;
  const g = parseGame(xml);
  eq(
    g.rounds[0].events.map((e) => e.t),
    ["draw", "discard", "draw", "discard"],
    "BYE skipped; draws/discards intact",
  );
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
  if (!/〔解説ポイント#1: 配牌評価｜/.test(text)) throw new Error("missing commentary anchors");
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
