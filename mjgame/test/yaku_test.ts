// 役判定 / 符計算 table tests. Every case names the reading it expects, so a
// regression in the max-over-decompositions selection shows up as a diff in the
// yaku list rather than only in the total.

import { assertEquals } from "@std/assert";
import type { Meld } from "mjrender/model.ts";
import { JANKI } from "../src/rules.ts";
import type { RuleConfig } from "../src/rules.ts";
import type { Seat } from "../src/types.ts";
import { hasAnyYaku, scoreWin, type WinContext } from "../src/yaku.ts";
import { tiles } from "./helpers.ts";

function meld(kind: Meld["kind"], spec: string, fromWho: Seat = 1): Meld {
  const ts = tiles(spec);
  return {
    kind,
    who: 0,
    fromWho: kind === "ankan" ? 0 : fromWho,
    tiles: ts,
    calledTile: ts[0],
  };
}

const chi = (s: string) => meld("chi", s, 3);
const pon = (s: string) => meld("pon", s);
const ankan = (s: string) => meld("ankan", s);
const minkan = (s: string) => meld("daiminkan", s);

function mk(hand: string, win: string, over: Partial<WinContext> = {}): WinContext {
  return {
    seat: 0,
    hand: tiles(hand),
    melds: [],
    winTile: tiles(win)[0],
    tsumo: false,
    riichi: false,
    doubleRiichi: false,
    ippatsu: false,
    rinshan: false,
    chankan: false,
    haitei: false,
    houtei: false,
    tenhou: false,
    chiihou: false,
    seatWind: 28, // 南家
    roundWind: 27, // 東場
    doraTypes: [],
    uraTypes: [],
    akaCount: 0,
    cfg: JANKI,
    ...over,
  };
}

const PINFU = "234m567m234p678p55s"; // 平和 + 断幺九, ryanmen on 6p

interface Case {
  n: string;
  hand: string;
  win: string;
  melds?: Meld[];
  over?: Partial<WinContext>;
  cfg?: RuleConfig;
  yaku?: Array<[number, number]>;
  yakuman?: number[];
  han?: number;
  fu?: number;
  limit?: number;
  name?: string;
  none?: true;
}

const CASES: Case[] = [
  // --- 平和 / 符の基本 ---
  {
    n: "平和ツモ 20符",
    hand: PINFU,
    win: "6p",
    over: { tsumo: true },
    yaku: [[0, 1], [7, 1], [8, 1]],
    han: 3,
    fu: 20,
  },
  { n: "平和ロン 30符", hand: PINFU, win: "6p", yaku: [[7, 1], [8, 1]], han: 2, fu: 30 },

  // --- 喰い下がり ---
  {
    n: "三色同順(門前) 2飜",
    hand: "234m567m234p234s99p",
    win: "7m",
    yaku: [[7, 1], [25, 2]],
    han: 3,
    fu: 30,
  },
  {
    n: "三色同順(副露) 1飜 + 喰い平和30符",
    hand: "234m567m234s99p",
    win: "7m",
    melds: [chi("234p")],
    yaku: [[25, 1]],
    han: 1,
    fu: 30,
  },
  {
    n: "清一色+一気通貫(門前) 6+2飜",
    hand: "123m456m789m11m234m",
    win: "4m",
    yaku: [[7, 1], [24, 2], [35, 6]],
    han: 9,
    fu: 30,
    limit: 3,
    name: "倍満",
  },
  {
    n: "清一色+一気通貫(副露) 5+1飜",
    hand: "456789m11m234m",
    win: "4m",
    melds: [chi("123m")],
    yaku: [[24, 1], [35, 5]],
    han: 6,
    fu: 30,
    limit: 2,
    name: "跳満",
  },
  {
    n: "混全帯幺九(門前) 2飜",
    hand: "123m123p789s西西西99m",
    win: "3m",
    yaku: [[23, 2]],
    han: 2,
    fu: 40,
  },
  {
    n: "混全帯幺九(副露) 1飜",
    hand: "123m123p789s99m",
    win: "3m",
    melds: [pon("西西西")],
    yaku: [[23, 1]],
    han: 1,
    fu: 30,
  },
  {
    n: "純全帯幺九(門前) 3飜",
    hand: "123m789m123p789s99p",
    win: "3m",
    yaku: [[33, 3]],
    han: 3,
    fu: 40,
  },
  {
    n: "純全帯幺九(副露) 2飜",
    hand: "123m789m789s99p",
    win: "3m",
    melds: [chi("123p")],
    yaku: [[33, 2]],
    han: 2,
    fu: 30,
  },
  {
    n: "混一色+一通+役牌(門前)",
    hand: "123p456p789p白白白西西",
    win: "3p",
    yaku: [[18, 1], [24, 2], [34, 3]],
    han: 6,
    fu: 40,
  },
  {
    n: "混一色+一通+役牌(副露)",
    hand: "123p456p789p西西",
    win: "3p",
    melds: [pon("白白白")],
    yaku: [[18, 1], [24, 1], [34, 2]],
    han: 4,
    fu: 30,
  },

  // --- 一盃口 / 二盃口 / 七対子 の取り合い ---
  {
    n: "一盃口 + 平和",
    hand: "112233m456p789s99p",
    win: "1m",
    yaku: [[7, 1], [9, 1]],
    han: 2,
    fu: 30,
  },
  {
    n: "二盃口を七対子より優先",
    hand: "112233m445566s99p",
    win: "1m",
    yaku: [[7, 1], [32, 3]],
    han: 4,
    fu: 30,
  },
  { n: "七対子", hand: "1122m3344p5566s東東", win: "東", yaku: [[22, 2]], han: 2, fu: 25 },
  {
    n: "七対子 + 断幺九",
    hand: "2233m4455p6677s88s",
    win: "8s",
    yaku: [[8, 1], [22, 2]],
    han: 3,
    fu: 25,
  },

  // --- 刻子系 ---
  {
    n: "対々和 + 三暗刻",
    hand: "111m333p555s99m",
    win: "9m",
    melds: [pon("東東東")],
    over: { seatWind: 29, roundWind: 30 },
    yaku: [[28, 2], [29, 2]],
    han: 4,
    fu: 50,
    limit: 1,
    name: "満貫",
  },
  {
    n: "三暗刻を一盃口より優先",
    hand: "111222333m456p99s",
    win: "9s",
    yaku: [[29, 2]],
    han: 2,
    fu: 50,
  },
  {
    n: "三色同刻 + 三暗刻",
    hand: "111m111p111s234m99s",
    win: "4m",
    yaku: [[26, 2], [29, 2]],
    han: 4,
    fu: 60,
  },
  {
    n: "三槓子 (明槓入り)",
    hand: "456m99p",
    win: "6m",
    melds: [ankan("1111m"), ankan("2222p"), minkan("3333s")],
    yaku: [[27, 2]],
    han: 2,
    fu: 80,
  },
  {
    n: "小三元 + 役牌2",
    hand: "白白白發發發中中123m456p",
    win: "6p",
    yaku: [[18, 1], [19, 1], [30, 2]],
    han: 4,
    fu: 50,
  },
  {
    n: "混老頭 + 対々和 + 三暗刻 + 役牌",
    hand: "111m999m西西西99p",
    win: "9p",
    melds: [pon("白白白")],
    yaku: [[18, 1], [28, 2], [29, 2], [31, 2]],
    han: 7,
    fu: 50,
    limit: 2,
    name: "跳満",
  },

  // --- 断幺九と喰いタン ---
  {
    n: "喰いタン (kuitan on)",
    hand: "567m234p678p55s",
    win: "6p",
    melds: [chi("234m")],
    yaku: [[8, 1]],
    han: 1,
    fu: 30,
  },
  {
    n: "喰いタン禁止なら役なし",
    hand: "567m234p678p55s",
    win: "6p",
    melds: [chi("234m")],
    cfg: { ...JANKI, kuitan: false },
    none: true,
  },

  // --- 役牌 (場風 / 自風 / 連風) ---
  {
    n: "場風 南",
    hand: "南南南123m456m123p11p",
    win: "4m",
    over: { seatWind: 29, roundWind: 28 },
    yaku: [[15, 1]],
    han: 1,
    fu: 40,
  },
  {
    n: "自風 南",
    hand: "南南南123m456m123p11p",
    win: "4m",
    over: { seatWind: 28, roundWind: 27 },
    yaku: [[11, 1]],
    han: 1,
    fu: 40,
  },
  {
    n: "連風牌 (自風+場風で2飜)",
    hand: "南南南123m456m123p11p",
    win: "4m",
    over: { seatWind: 28, roundWind: 28 },
    yaku: [[11, 1], [15, 1]],
    han: 2,
    fu: 40,
  },
  {
    n: "連風牌の雀頭 +2符",
    hand: "南南111m456m789m123p",
    win: "4m",
    over: { seatWind: 28, roundWind: 28, riichi: true },
    yaku: [[1, 1]],
    han: 1,
    fu: 40,
  },
  {
    n: "連風牌の雀頭 +4符 (doubleWindFu=4)",
    hand: "南南111m456m789m123p",
    win: "4m",
    over: { seatWind: 28, roundWind: 28, riichi: true },
    cfg: { ...JANKI, doubleWindFu: 4 },
    han: 1,
    fu: 50,
  },

  // --- 状況役 ---
  {
    n: "海底摸月",
    hand: PINFU,
    win: "6p",
    over: { tsumo: true, haitei: true },
    yaku: [[0, 1], [5, 1], [7, 1], [8, 1]],
    han: 4,
    fu: 20,
  },
  {
    n: "河底撈魚",
    hand: PINFU,
    win: "6p",
    over: { houtei: true },
    yaku: [[6, 1], [7, 1], [8, 1]],
    han: 3,
    fu: 30,
  },
  {
    n: "嶺上開花",
    hand: PINFU,
    win: "6p",
    over: { tsumo: true, rinshan: true },
    yaku: [[0, 1], [4, 1], [7, 1], [8, 1]],
    han: 4,
    fu: 20,
  },
  {
    n: "槍槓",
    hand: PINFU,
    win: "6p",
    over: { chankan: true },
    yaku: [[3, 1], [7, 1], [8, 1]],
    han: 3,
    fu: 30,
  },
  {
    n: "立直 + 一発 + 門前ツモ",
    hand: PINFU,
    win: "6p",
    over: { tsumo: true, riichi: true, ippatsu: true },
    yaku: [[0, 1], [1, 1], [2, 1], [7, 1], [8, 1]],
    han: 5,
    fu: 20,
    limit: 1,
    name: "満貫",
  },
  {
    n: "両立直は立直を置き換える",
    hand: PINFU,
    win: "6p",
    over: { riichi: true, doubleRiichi: true },
    yaku: [[7, 1], [8, 1], [21, 2]],
    han: 4,
    fu: 30,
  },

  // --- ドラ ---
  {
    n: "ドラ・裏ドラ・赤ドラは独立の id",
    hand: PINFU,
    win: "6p",
    over: { riichi: true, doraTypes: [4], uraTypes: [22], akaCount: 2 },
    yaku: [[1, 1], [7, 1], [8, 1], [52, 1], [53, 2], [54, 2]],
    han: 8,
    fu: 30,
    limit: 3,
    name: "倍満",
  },
  {
    n: "ドラだけの手は和了れない",
    hand: "567m456p789s99s",
    win: "9s",
    melds: [chi("234m")],
    over: { doraTypes: [26, 26] },
    none: true,
  },

  // --- 待ちの読み替え (シャンポン/単騎/両面) を最大で解決 ---
  {
    n: "シャンポン・単騎・両面の三通りから最大を選ぶ",
    hand: "111222333m44m789p",
    win: "1m",
    over: { riichi: true },
    yaku: [[1, 1], [7, 1], [9, 1]],
    han: 3,
    fu: 30,
  },

  // --- 役満 ---
  {
    n: "天和",
    hand: PINFU,
    win: "6p",
    over: { tsumo: true, tenhou: true },
    yakuman: [37],
    han: 13,
  },
  {
    n: "地和",
    hand: PINFU,
    win: "6p",
    over: { tsumo: true, chiihou: true },
    yakuman: [38],
    han: 13,
  },
  { n: "大三元", hand: "白白白發發發中中中123m99p", win: "1m", yakuman: [39] },
  {
    n: "四暗刻 (シャンポンをツモ)",
    hand: "111m333p555s777s99m",
    win: "7s",
    over: { tsumo: true },
    yakuman: [40],
  },
  { n: "四暗刻単騎 (単騎をロン)", hand: "111m333p555s777s99m", win: "9m", yakuman: [41] },
  {
    n: "字一色",
    hand: "東東東南南南白白白發發",
    win: "發",
    melds: [pon("中中中")],
    yakuman: [42],
  },
  { n: "緑一色", hand: "222s333s444s88s", win: "8s", melds: [pon("666s")], yakuman: [43] },
  { n: "清老頭", hand: "111m999m111p11s", win: "1s", melds: [pon("999p")], yakuman: [44] },
  { n: "九蓮宝燈", hand: "11123455678999m", win: "1m", yakuman: [45] },
  { n: "純正九蓮宝燈", hand: "11123455678999m", win: "5m", yakuman: [46] },
  { n: "国士無双", hand: "19m19p19s東南西北白發中中", win: "1m", yakuman: [47] },
  { n: "国士無双十三面", hand: "19m19p19s東南西北白發中中", win: "中", yakuman: [48] },
  {
    n: "大四喜",
    hand: "東東東南南南西西西11m",
    win: "1m",
    melds: [pon("北北北")],
    yakuman: [49],
  },
  { n: "小四喜", hand: "東東東南南南西西西北北123m", win: "3m", yakuman: [50] },
  {
    n: "四槓子",
    hand: "99p",
    win: "9p",
    melds: [ankan("1111m"), minkan("2222p"), minkan("3333s"), minkan("4444m")],
    yakuman: [51],
  },
  {
    n: "役満は通常役とドラを吸収する",
    hand: "白白白發發發中中中123m99p",
    win: "1m",
    over: { riichi: true, doraTypes: [31, 31, 31] },
    yakuman: [39],
    yaku: [],
    han: 13,
  },

  // --- 雀鬼会の否定形 ---
  {
    n: "13飜は数え役満ではなく三倍満",
    hand: "123m456m789m11m234m",
    win: "4m",
    over: { doraTypes: [3, 3] },
    yaku: [[7, 1], [24, 2], [35, 6], [52, 4]],
    han: 13,
    yakuman: [],
    limit: 4,
    name: "三倍満",
  },
  {
    n: "kazoeYakuman を有効にすれば13飜は役満扱い",
    hand: "123m456m789m11m234m",
    win: "4m",
    over: { doraTypes: [3, 3] },
    cfg: { ...JANKI, kazoeYakuman: true },
    han: 13,
    limit: 5,
    name: "役満",
  },
];

const BASE_BY_LIMIT = [0, 2000, 3000, 4000, 6000, 8000];

for (const c of CASES) {
  Deno.test(`yaku: ${c.n}`, () => {
    const ctx = mk(c.hand, c.win, {
      melds: c.melds ?? [],
      ...(c.cfg ? { cfg: c.cfg } : {}),
      ...c.over,
    });
    const r = scoreWin(ctx);
    if (c.none) {
      assertEquals(r, null);
      assertEquals(hasAnyYaku(ctx), false);
      return;
    }
    if (r === null) throw new Error(`expected a win, got 役なし`);
    if (c.yaku) assertEquals(r.yaku.map((x) => [x.id, x.han]), c.yaku);
    if (c.yakuman) assertEquals(r.yakuman, c.yakuman);
    if (c.han !== undefined) assertEquals(r.han, c.han);
    if (c.fu !== undefined) assertEquals(r.fu, c.fu);
    if (c.limit !== undefined) {
      assertEquals(r.limit, c.limit);
      assertEquals(r.base, BASE_BY_LIMIT[c.limit] * (r.yakuman.length || 1));
    }
    if (c.name !== undefined) assertEquals(r.name, c.name);
    assertEquals(hasAnyYaku(ctx), true);
  });
}

Deno.test("yaku: 人和 (id 36) is never awarded — the dojo does not adopt it", () => {
  for (const c of CASES) {
    const ctx = mk(c.hand, c.win, {
      melds: c.melds ?? [],
      ...(c.cfg ? { cfg: c.cfg } : {}),
      ...c.over,
    });
    const r = scoreWin(ctx);
    if (!r) continue;
    assertEquals(r.yaku.some((x) => x.id === 36), false);
    assertEquals(r.yakuman.includes(36), false);
  }
});

Deno.test("yaku: 役満 stacks (字一色 + 四暗刻単騎)", () => {
  const r = scoreWin(mk("東東東南南南白白白中中中發發", "發"));
  assertEquals(r?.yakuman, [41, 42]);
  assertEquals(r?.base, 16000);
  assertEquals(r?.name, "役満");
});

Deno.test("yaku: incomplete hands score nothing", () => {
  assertEquals(scoreWin(mk("123m456m789m11m235m", "5m")), null);
});
