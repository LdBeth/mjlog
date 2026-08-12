// 符計算. Also the home of the two structural predicates that both fu and yaku
// need (門前 and 暗刻 counting), so that the runtime import graph stays acyclic:
// this module imports only *types* from `yaku.ts`, never values.

import type { Block, Decomposition } from "./decompose.ts";
import { isSimple } from "./tiles.ts";
import type { WinContext } from "./yaku.ts";

/** 門前: an ankan does not open the hand, any other meld does. */
export function menzenOf(ctx: WinContext): boolean {
  return ctx.melds.every((m) => m.kind === "ankan");
}

function isTripletish(b: Block): boolean {
  return b.kind === "triplet" || b.kind === "kan";
}

/**
 * Whether `blocks[i]` counts as a *concealed* triplet/kan for 三暗刻/四暗刻 and
 * for fu. A shanpon ron is the one exception: the winning tile came from another
 * player, so the set it completed is a 明刻 even though it sits in a closed hand.
 */
export function isConcealedSet(d: Decomposition, i: number, ctx: WinContext): boolean {
  const b = d.blocks[i];
  if (!isTripletish(b) || !b.concealed) return false;
  return ctx.tsumo || i !== d.winBlock || d.wait !== "shanpon";
}

/** 暗刻の数 (kans included). */
export function ankouCount(d: Decomposition, ctx: WinContext): number {
  let n = 0;
  for (let i = 0; i < d.blocks.length; i++) if (isConcealedSet(d, i, ctx)) n++;
  return n;
}

/** 雀頭が役牌か: dragons, the seat wind, or the round wind. */
export function isYakuhaiPair(type: number, ctx: WinContext): boolean {
  return type >= 31 || type === ctx.seatWind || type === ctx.roundWind;
}

export function isPinfu(d: Decomposition, ctx: WinContext): boolean {
  if (d.form !== "standard") return false;
  if (!menzenOf(ctx)) return false;
  if (!d.blocks.every((b) => b.kind === "run")) return false;
  if (d.wait !== "ryanmen") return false;
  return !isYakuhaiPair(d.pair.type, ctx);
}

/** Fu for one triplet/kan block, given whether it reads as concealed. */
function setFu(b: Block, concealed: boolean): number {
  const cheap = isSimple(b.type);
  if (b.kind === "kan") return concealed ? (cheap ? 16 : 32) : cheap ? 8 : 16;
  return concealed ? (cheap ? 4 : 8) : cheap ? 2 : 4;
}

const ceil10 = (n: number): number => Math.ceil(n / 10) * 10;

/**
 * 符 for one reading of the hand, with a human-readable breakdown.
 *
 * 七対子 is a flat 25 (never rounded); 国士無双 is a yakuman, so its fu is never
 * used for payment — it reports the 副底 alone rather than inventing a value.
 */
export function countFu(d: Decomposition, ctx: WinContext): { fu: number; parts: string[] } {
  if (d.form === "chiitoi") return { fu: 25, parts: ["七対子25"] };
  if (d.form === "kokushi") return { fu: 20, parts: ["副底20"] };

  const menzen = menzenOf(ctx);
  const pinfu = isPinfu(d, ctx);
  const parts = ["副底20"];
  let fu = 20;

  if (menzen && !ctx.tsumo) {
    fu += 10;
    parts.push("門前加符10");
  }
  // 平和自摸 is the one hand that forgoes the 自摸符 — that is what pins it to 20.
  if (ctx.tsumo && !pinfu) {
    fu += 2;
    parts.push("自摸2");
  }

  for (let i = 0; i < d.blocks.length; i++) {
    const b = d.blocks[i];
    if (!isTripletish(b)) continue;
    const concealed = isConcealedSet(d, i, ctx);
    const v = setFu(b, concealed);
    fu += v;
    parts.push(`${concealed ? "暗" : "明"}${b.kind === "kan" ? "槓" : "刻"}${v}`);
  }

  const p = d.pair.type;
  if (p === ctx.seatWind && p === ctx.roundWind) {
    fu += ctx.cfg.doubleWindFu;
    parts.push(`連風牌${ctx.cfg.doubleWindFu}`);
  } else if (isYakuhaiPair(p, ctx)) {
    fu += 2;
    parts.push("役牌雀頭2");
  }

  if (d.wait === "kanchan" || d.wait === "penchan" || d.wait === "tanki") {
    fu += 2;
    parts.push("待ち2");
  }

  // 喰い平和形: an open hand with no fu-bearing part would score 20, which no
  // ruleset pays — it is settled as a flat 30.
  if (!menzen && fu === 20) {
    parts.push("喰い平和30");
    return { fu: 30, parts };
  }
  return { fu: ceil10(fu), parts };
}
