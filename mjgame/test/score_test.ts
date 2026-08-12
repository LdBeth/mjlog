// 点数計算・精算のテスト: the han/fu grid, ron/tsumo splits, 本場/供託, 罰符,
// 流し満貫, and the dojo's violation-first final standings.

import { assertEquals } from "@std/assert";
import { JANKI } from "../src/rules.ts";
import { sfc32 } from "../src/rng.ts";
import { basePoints, finalStandings, ronPayment, scorer, tsumoPayment } from "../src/score.ts";
import { Table } from "../src/table.ts";
import type { Seat, Violation, WinInfo } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { Wall } from "../src/wall.ts";
import { tiles } from "./helpers.ts";

// ---------------------------------------------------------------------------
// 基本点
// ---------------------------------------------------------------------------

Deno.test("score: 基本点の階段", () => {
  const b = (han: number, fu: number) => basePoints(han, fu, JANKI);
  assertEquals(b(1, 30).base, 240);
  assertEquals(b(2, 30).base, 480);
  assertEquals(b(3, 30).base, 960);
  assertEquals(b(4, 30), { base: 1920, limit: 0, name: "" }); // 満貫未満のまま
  assertEquals(b(4, 40), { base: 2000, limit: 1, name: "満貫" }); // 2560 を頭打ち
  assertEquals(b(5, 20), { base: 2000, limit: 1, name: "満貫" });
  assertEquals(b(6, 30), { base: 3000, limit: 2, name: "跳満" });
  assertEquals(b(7, 30), { base: 3000, limit: 2, name: "跳満" });
  assertEquals(b(8, 30), { base: 4000, limit: 3, name: "倍満" });
  assertEquals(b(10, 30), { base: 4000, limit: 3, name: "倍満" });
  assertEquals(b(11, 30), { base: 6000, limit: 4, name: "三倍満" });
});

Deno.test("score: 数え役満なし ⇒ 13飜は三倍満", () => {
  assertEquals(basePoints(13, 30, JANKI), { base: 6000, limit: 4, name: "三倍満" });
  assertEquals(basePoints(20, 30, JANKI), { base: 6000, limit: 4, name: "三倍満" });
  assertEquals(
    basePoints(13, 30, { ...JANKI, kazoeYakuman: true }),
    { base: 8000, limit: 5, name: "役満" },
  );
});

Deno.test("score: 切り上げ満貫はオプション", () => {
  assertEquals(basePoints(4, 30, JANKI).base, 1920);
  assertEquals(basePoints(4, 30, { ...JANKI, kiriageMangan: true }), {
    base: 2000,
    limit: 1,
    name: "満貫",
  });
});

Deno.test("score: ロン払いの代表値", () => {
  const ron = (han: number, fu: number, dealer: boolean) =>
    ronPayment(dealer, basePoints(han, fu, JANKI).base);
  assertEquals([ron(1, 30, false), ron(1, 30, true)], [1000, 1500]);
  assertEquals([ron(2, 30, false), ron(2, 30, true)], [2000, 2900]);
  assertEquals([ron(3, 30, false), ron(3, 30, true)], [3900, 5800]);
  assertEquals([ron(3, 40, false), ron(3, 40, true)], [5200, 7700]);
  assertEquals([ron(4, 30, false), ron(4, 30, true)], [7700, 11600]);
  assertEquals([ron(4, 40, false), ron(4, 40, true)], [8000, 12000]); // 満貫頭打ち
  assertEquals([ron(5, 30, false), ron(5, 30, true)], [8000, 12000]); // 満貫
  assertEquals([ron(6, 30, false), ron(6, 30, true)], [12000, 18000]); // 跳満
  assertEquals([ron(8, 30, false), ron(8, 30, true)], [16000, 24000]); // 倍満
  assertEquals([ron(11, 30, false), ron(11, 30, true)], [24000, 36000]); // 三倍満
  assertEquals([ronPayment(false, 8000), ronPayment(true, 8000)], [32000, 48000]); // 役満
});

Deno.test("score: ツモ払いの分割", () => {
  const ts = (han: number, fu: number, dealer: boolean) =>
    tsumoPayment(dealer, basePoints(han, fu, JANKI).base);
  assertEquals(ts(3, 30, false), { fromDealer: 2000, fromOther: 1000 }); // 1000-2000
  assertEquals(ts(3, 30, true).fromOther, 2000); // 2000オール
  assertEquals(ts(5, 30, false), { fromDealer: 4000, fromOther: 2000 }); // 満貫 2000-4000
  assertEquals(ts(5, 30, true).fromOther, 4000); // 満貫 4000オール
  assertEquals(tsumoPayment(false, 8000), { fromDealer: 16000, fromOther: 8000 }); // 役満
  assertEquals(tsumoPayment(true, 8000).fromOther, 16000);
});

// ---------------------------------------------------------------------------
// Table-driven settlement
// ---------------------------------------------------------------------------

function table(honba = 0, kyotaku = 0, dealer: Seat = 0): Table {
  const rng = sfc32(7);
  const t = new Table(
    {
      kyoku: 0,
      honba,
      kyotaku,
      dealer,
      scores: [25000, 25000, 25000, 25000],
      wall: Wall.shuffled(rng),
      dice: [1, 2],
    },
    JANKI,
    SEATS.map((seat) => ({ seat, name: `P${seat}` })),
  );
  return t;
}

function win(over: Partial<WinInfo>): WinInfo {
  return {
    who: 0,
    fromWho: 0,
    winTile: 0,
    han: 3,
    fu: 30,
    points: 0,
    limit: 0,
    yaku: [],
    yakuman: [],
    doraIndicators: [],
    uraIndicators: [],
    hand: [],
    melds: [],
    ...over,
  };
}

Deno.test("score: 子のロン (3飜30符)", () => {
  const t = table();
  assertEquals(scorer.winDeltas(t, [win({ who: 1, fromWho: 2 })]), [0, 3900, -3900, 0]);
});

Deno.test("score: 本場と供託", () => {
  const t = table(2, 1);
  // 3900 + 2本場600 + 供託1000
  assertEquals(scorer.winDeltas(t, [win({ who: 1, fromWho: 2 })]), [0, 5500, -4500, 0]);
});

Deno.test("score: 親のツモ満貫は4000オール", () => {
  const t = table();
  assertEquals(
    scorer.winDeltas(t, [win({ who: 0, fromWho: 0, han: 5, fu: 30 })]),
    [12000, -4000, -4000, -4000],
  );
});

Deno.test("score: 子のツモは親から倍取り", () => {
  const t = table();
  assertEquals(
    scorer.winDeltas(t, [win({ who: 1, fromWho: 1 })]),
    [-2000, 4000, -1000, -1000],
  );
});

Deno.test("score: ツモの本場は各家100点ずつ", () => {
  const t = table(3, 0);
  assertEquals(
    scorer.winDeltas(t, [win({ who: 1, fromWho: 1 })]),
    [-2300, 4900, -1300, -1300],
  );
});

Deno.test("score: 役満のロンは子32000 / 親48000", () => {
  const t = table();
  assertEquals(
    scorer.winDeltas(t, [win({ who: 1, fromWho: 3, han: 13, yakuman: [39] })]),
    [0, 32000, 0, -32000],
  );
  assertEquals(
    scorer.winDeltas(t, [win({ who: 0, fromWho: 3, han: 13, yakuman: [39] })]),
    [48000, 0, 0, -48000],
  );
});

// ---------------------------------------------------------------------------
// 流局
// ---------------------------------------------------------------------------

const T = true, F = false;

Deno.test("score: 罰符 (聴牌0〜4人)", () => {
  const t = table();
  const d = (tenpai: boolean[]) => scorer.drawDeltas(t, tenpai, "exhaustive", []);
  assertEquals(d([F, F, F, F]), [0, 0, 0, 0]);
  assertEquals(d([T, F, F, F]), [3000, -1000, -1000, -1000]);
  assertEquals(d([T, T, F, F]), [1500, 1500, -1500, -1500]);
  assertEquals(d([T, T, T, F]), [1000, 1000, 1000, -3000]);
  assertEquals(d([T, T, T, T]), [0, 0, 0, 0]);
});

Deno.test("score: 途中流局は点棒移動なし", () => {
  const t = table();
  for (const kind of ["sanchahou", "suukaikan", "suucha-riichi"] as const) {
    assertEquals(scorer.drawDeltas(t, [T, F, F, F], kind, []), [0, 0, 0, 0]);
  }
});

Deno.test("score: 流し満貫はツモ払いで罰符に置き換わる", () => {
  const t = table();
  // 親の流し満貫: 4000オール
  assertEquals(
    scorer.drawDeltas(t, [T, F, F, F], "nagashi", [0]),
    [12000, -4000, -4000, -4000],
  );
  // 子の流し満貫: 親4000 / 子2000
  assertEquals(
    scorer.drawDeltas(t, [F, T, T, T], "nagashi", [1]),
    [-4000, 8000, -2000, -2000],
  );
});

// ---------------------------------------------------------------------------
// Scorer wired to a Table
// ---------------------------------------------------------------------------

Deno.test("score: scorer.hasYaku / scoreWin over a real Table", () => {
  const t = table();
  t.board.indicators.length = 0; // pin the dora count at 0 for a deterministic total
  t.hands[0].length = 0;
  t.hands[0].push(...tiles("234m567m234p678p55s"));
  const winTile = tiles("6p")[0];

  assertEquals(scorer.hasYaku(t, 0, winTile, true), true);

  const flags = {
    tsumo: true,
    riichi: false,
    doubleRiichi: false,
    ippatsu: false,
    rinshan: false,
    chankan: false,
    haitei: false,
    houtei: false,
    tenhou: false,
    chiihou: false,
  };
  const w = scorer.scoreWin(t, 0, 0, winTile, flags);
  if (!w) throw new Error("expected a win");
  // 門前清自摸和 + 平和 + 断幺九 = 3飜20符, 親ツモ 1300オール.
  assertEquals([w.han, w.fu], [3, 20]);
  assertEquals(w.points, 3900);
  assertEquals(w.uraIndicators, []); // not in riichi
});

// ---------------------------------------------------------------------------
// 順位
// ---------------------------------------------------------------------------

function violation(seat: Seat): Violation {
  return {
    rule: "test",
    label: "テスト違反",
    seat,
    kyoku: 0,
    junme: 1,
    points: 3,
    tier: "A",
    confidence: 1,
    detail: "",
  };
}

Deno.test("score: finalStandings — 素点・ウマ・返し点", () => {
  const s = finalStandings([40000, 30000, 20000, 10000], 0, [], JANKI);
  assertEquals(s.map((r) => r.seat), [0, 1, 2, 3]);
  assertEquals(s.map((r) => r.place), [1, 2, 3, 4]);
  assertEquals(s.map((r) => r.points), [30, 10, -10, -20]);
  assertEquals(s.every((r) => r.clean), true);
});

Deno.test("score: finalStandings — 反則者はトップでも清廉な打ち手の下", () => {
  const s = finalStandings([40000, 30000, 20000, 10000], 0, [violation(0)], JANKI);
  assertEquals(s.map((r) => r.seat), [1, 2, 3, 0]);
  assertEquals(s[3], {
    place: 4,
    seat: 0,
    score: 40000,
    points: 10, // 素点は残るが ウマ は4位のもの
    clean: false,
    violations: 1,
  });
  assertEquals(s[0].points, 20); // 30000点ちょうどの1位 = 素点0 + ウマ20
});

Deno.test("score: finalStandings — 反則者同士は点数順", () => {
  const s = finalStandings(
    [40000, 30000, 20000, 10000],
    0,
    [violation(0), violation(2), violation(2)],
    JANKI,
  );
  assertEquals(s.map((r) => r.seat), [1, 3, 0, 2]);
  assertEquals(s.map((r) => r.violations), [0, 0, 1, 2]);
});

Deno.test("score: finalStandings — 1000点未満切り捨てと起家順の同点処理", () => {
  const s = finalStandings([32400, 27600, 30000, 30000], 0, [], JANKI);
  assertEquals(s.map((r) => r.seat), [0, 2, 3, 1]);
  assertEquals(s.map((r) => r.points), [22, 10, 0, -2]);

  const raw = finalStandings([32400, 27600, 30000, 30000], 0, [], {
    ...JANKI,
    truncateSub1000: false,
  });
  assertEquals(raw.map((r) => r.points), [22.4, 10, 0, -2.4]);
});

Deno.test("score: finalStandings — 同点は起家に近い順", () => {
  const s = finalStandings([30000, 30000, 30000, 30000], 2, [], JANKI);
  assertEquals(s.map((r) => r.seat), [2, 3, 0, 1]);
});
