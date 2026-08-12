// 役判定. Yaku are identified by Tenhou's numeric ids (see mjrender/src/yaku.ts
// for the name table) so that both the XML export and the mjrender transcript
// renderer need zero mapping.
//
// The evaluator runs over *every* reading `decomposeWin` produces and keeps the
// best one: 三暗刻 vs 一盃口, 二盃口 vs 七対子, and shanpon vs ryanmen fu are all
// decided by which reading pays more, not by a heuristic.

import type { Meld, Tile } from "mjrender/model.ts";
import { suitOfType, tileType } from "mjrender/tiles.ts";
import type { Block, Decomposition } from "./decompose.ts";
import { decomposeWin } from "./decompose.ts";
import { ankouCount, countFu, isPinfu, menzenOf } from "./fu.ts";
import { basePoints } from "./score.ts";
import type { RuleConfig } from "./rules.ts";
import { GREEN_TYPES, isHonor, isSimple, isTerminal, isYaochu, zeros34 } from "./tiles.ts";
import type { Seat } from "./types.ts";

export interface WinContext {
  seat: Seat;
  /** Concealed tiles INCLUDING the winning tile. */
  hand: Tile[];
  melds: Meld[];
  winTile: Tile;
  tsumo: boolean;
  riichi: boolean;
  doubleRiichi: boolean;
  ippatsu: boolean;
  rinshan: boolean;
  chankan: boolean;
  haitei: boolean;
  houtei: boolean;
  tenhou: boolean;
  chiihou: boolean;
  /** Tile types 27..30. */
  seatWind: number;
  roundWind: number;
  /** Dora TYPES (already mapped through `doraFromIndicatorType`), with multiplicity. */
  doraTypes: number[];
  uraTypes: number[];
  akaCount: number;
  cfg: RuleConfig;
}

export interface YakuResult {
  yaku: Array<{ id: number; han: number }>;
  yakuman: number[];
  han: number;
  fu: number;
  /** Base points, before the ×4 / ×6 ron multiplier. */
  base: number;
  limit: 0 | 1 | 2 | 3 | 4 | 5;
  name: string;
  decomposition: Decomposition;
}

/** Dora / 裏ドラ / 赤ドラ ids — han-bearing but not yaku: they cannot open a win. */
export const DORA_IDS = { dora: 52, ura: 53, aka: 54 } as const;

const GREEN = new Set(GREEN_TYPES);

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

const isRun = (b: Block): boolean => b.kind === "run";
const isTripletish = (b: Block): boolean => b.kind === "triplet" || b.kind === "kan";

/** Tile types a block occupies (a run spans three). */
function blockTypes(b: Block): number[] {
  return b.kind === "run" ? [b.type, b.type + 1, b.type + 2] : [b.type];
}

/**
 * The blocks that partition the hand. 七対子 lists its own head inside `blocks`,
 * so adding `pair` there would double-count it.
 */
function allBlocks(d: Decomposition): Block[] {
  return d.form === "chiitoi" ? d.blocks : [...d.blocks, d.pair];
}

/** 34-vector over hand + meld tiles (a kan therefore contributes 4). */
function fullCounts(ctx: WinContext): number[] {
  const c = zeros34();
  for (const t of ctx.hand) c[tileType(t)]++;
  for (const m of ctx.melds) for (const t of m.tiles) c[tileType(t)]++;
  return c;
}

function everyType(counts: readonly number[], pred: (t: number) => boolean): boolean {
  for (let t = 0; t < 34; t++) if (counts[t] > 0 && !pred(t)) return false;
  return true;
}

function anyType(counts: readonly number[], pred: (t: number) => boolean): boolean {
  for (let t = 0; t < 34; t++) if (counts[t] > 0 && pred(t)) return true;
  return false;
}

/** Number of identical-run pairs: 1 ⇒ 一盃口, 2 ⇒ 二盃口. */
function peikoPairs(d: Decomposition): number {
  const runs = new Map<number, number>();
  for (const b of d.blocks) if (isRun(b)) runs.set(b.type, (runs.get(b.type) ?? 0) + 1);
  let n = 0;
  for (const c of runs.values()) n += Math.floor(c / 2);
  return n;
}

function hasIttsu(d: Decomposition): boolean {
  const runs = new Set(d.blocks.filter(isRun).map((b) => b.type));
  for (const base of [0, 9, 18]) {
    if (runs.has(base) && runs.has(base + 3) && runs.has(base + 6)) return true;
  }
  return false;
}

function hasSanshokuDoujun(d: Decomposition): boolean {
  const runs = d.blocks.filter(isRun).map((b) => b.type);
  for (const t of runs) {
    if (t >= 9) continue; // only need to try man-suit starts
    const r = t % 9;
    if (runs.includes(r + 9) && runs.includes(r + 18)) return true;
  }
  return false;
}

function hasSanshokuDoukou(d: Decomposition): boolean {
  const trips = d.blocks.filter(isTripletish).map((b) => b.type);
  for (const t of trips) {
    if (t >= 9) continue;
    if (trips.includes(t + 9) && trips.includes(t + 18)) return true;
  }
  return false;
}

/** 45 九蓮宝燈 / 46 純正九蓮宝燈 / 0 for neither. `counts` is the concealed hand. */
function chuurenId(ctx: WinContext, counts: readonly number[]): number {
  if (ctx.melds.length > 0) return 0;
  const first = tileType(ctx.hand[0]);
  const suit = suitOfType(first);
  if (suit === "z") return 0;
  const base = suit === "m" ? 0 : suit === "p" ? 9 : 18;
  for (let t = 0; t < 34; t++) {
    if (counts[t] > 0 && (t < base || t >= base + 9)) return 0;
  }
  const need = [3, 1, 1, 1, 1, 1, 1, 1, 3];
  for (let r = 0; r < 9; r++) if (counts[base + r] < need[r]) return 0;
  // 純正 = the 13 tiles before the win were exactly 1112345678999, i.e. the hand
  // was a nine-sided wait.
  const w = tileType(ctx.winTile);
  for (let r = 0; r < 9; r++) {
    const c = counts[base + r] - (base + r === w ? 1 : 0);
    if (c !== need[r]) return 45;
  }
  return 46;
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

function detectYakuman(d: Decomposition, ctx: WinContext, counts: readonly number[]): number[] {
  const out: number[] = [];
  if (ctx.tenhou) out.push(37);
  if (ctx.chiihou) out.push(38);

  if (d.form === "kokushi") {
    out.push(d.wait === "kokushi13" ? 48 : 47);
    return out;
  }

  const all = fullCounts(ctx);
  if (everyType(all, isHonor)) out.push(42); // 字一色
  if (everyType(all, (t) => GREEN.has(t))) out.push(43); // 緑一色
  if (everyType(all, isTerminal)) out.push(44); // 清老頭

  if (d.form === "standard") {
    const trips = d.blocks.filter(isTripletish).map((b) => b.type);
    if (trips.filter((t) => t >= 31).length === 3) out.push(39); // 大三元
    const winds = trips.filter((t) => t >= 27 && t <= 30).length;
    if (winds === 4) out.push(49); // 大四喜
    else if (winds === 3 && d.pair.type >= 27 && d.pair.type <= 30) out.push(50); // 小四喜
    if (d.blocks.filter((b) => b.kind === "kan").length === 4) out.push(51); // 四槓子
    if (ankouCount(d, ctx) === 4) out.push(d.wait === "tanki" ? 41 : 40);
  }

  const ch = chuurenId(ctx, counts);
  if (ch) out.push(ch);
  return out.sort((a, b) => a - b);
}

function detectYaku(d: Decomposition, ctx: WinContext): Array<{ id: number; han: number }> {
  const y: Array<{ id: number; han: number }> = [];
  const add = (id: number, han: number) => y.push({ id, han });
  const menzen = menzenOf(ctx);
  /** 喰い下がり: one han less once the hand is open. */
  const ks = (closed: number) => (menzen ? closed : closed - 1);

  // --- situational ---
  if (menzen && ctx.tsumo) add(0, 1); // 門前清自摸和
  if (ctx.doubleRiichi) add(21, 2); // 両立直
  else if (ctx.riichi) add(1, 1); // 立直
  if (ctx.ippatsu && ctx.cfg.ippatsu && (ctx.riichi || ctx.doubleRiichi)) add(2, 1);
  if (ctx.chankan) add(3, 1);
  if (ctx.rinshan) add(4, 1);
  if (ctx.haitei && ctx.tsumo) add(5, 1);
  if (ctx.houtei && !ctx.tsumo) add(6, 1);

  const all = fullCounts(ctx);
  const blocks = allBlocks(d);
  const hasRun = blocks.some(isRun);
  const hasHonor = anyType(all, isHonor);

  if (d.form === "chiitoi") add(22, 2);

  if (d.form === "standard") {
    if (isPinfu(d, ctx)) add(7, 1);

    // 役牌: a wind can score twice when it is both the seat and the round wind.
    for (const b of d.blocks) {
      if (!isTripletish(b)) continue;
      if (b.type === ctx.seatWind) add(10 + (b.type - 27), 1);
      if (b.type === ctx.roundWind) add(14 + (b.type - 27), 1);
      if (b.type >= 31) add(18 + (b.type - 31), 1);
    }

    if (menzen) {
      const peiko = peikoPairs(d);
      if (peiko >= 2) add(32, 3); // 二盃口
      else if (peiko === 1) add(9, 1); // 一盃口
    }
    if (hasIttsu(d)) add(24, ks(2));
    if (hasSanshokuDoujun(d)) add(25, ks(2));
    if (hasSanshokuDoukou(d)) add(26, 2);
    if (d.blocks.filter((b) => b.kind === "kan").length === 3) add(27, 2); // 三槓子
    if (d.blocks.every(isTripletish)) add(28, 2); // 対々和
    if (ankouCount(d, ctx) === 3) add(29, 2); // 三暗刻
    if (d.blocks.filter((b) => isTripletish(b) && b.type >= 31).length === 2 && d.pair.type >= 31) {
      add(30, 2); // 小三元
    }
  }

  // --- 幺九系 ---
  if (everyType(all, isYaochu) && hasHonor && !hasRun) {
    add(31, 2); // 混老頭 (清老頭 is a yakuman and never reaches here)
  } else if (d.form === "standard" && hasRun) {
    const term = (b: Block) => blockTypes(b).some(isTerminal);
    const yao = (b: Block) => blockTypes(b).some(isYaochu);
    if (blocks.every(term)) add(33, ks(3)); // 純全帯幺九 (no honor can satisfy `term`)
    else if (blocks.every(yao)) add(23, ks(2)); // 混全帯幺九
  }

  if (everyType(all, isSimple) && (menzen || ctx.cfg.kuitan)) add(8, 1); // 断幺九

  const suits = new Set<string>();
  for (let t = 0; t < 34; t++) if (all[t] > 0 && !isHonor(t)) suits.add(suitOfType(t));
  if (suits.size === 1) {
    if (hasHonor) add(34, ks(3)); // 混一色
    else add(35, ks(6)); // 清一色
  }

  return y.sort((a, b) => a.id - b.id);
}

function countHits(counts: readonly number[], types: readonly number[]): number {
  let n = 0;
  for (const t of types) n += counts[t];
  return n;
}

// ---------------------------------------------------------------------------

function evaluate(d: Decomposition, ctx: WinContext, counts: readonly number[]): YakuResult | null {
  const fu = countFu(d, ctx).fu;
  const yakuman = detectYakuman(d, ctx, counts);
  if (yakuman.length > 0) {
    // A yakuman suppresses every normal yaku and all dora.
    return {
      yaku: [],
      yakuman,
      han: 13 * yakuman.length,
      fu,
      base: 8000 * yakuman.length,
      limit: 5,
      name: "役満",
      decomposition: d,
    };
  }

  const yaku = detectYaku(d, ctx);
  const yakuHan = yaku.reduce((a, x) => a + x.han, 0);
  if (yakuHan === 0) return null; // 役なし — dora alone never opens a win

  const all = fullCounts(ctx);
  const dora = countHits(all, ctx.doraTypes);
  const ura = countHits(all, ctx.uraTypes);
  if (dora) yaku.push({ id: DORA_IDS.dora, han: dora });
  if (ura) yaku.push({ id: DORA_IDS.ura, han: ura });
  if (ctx.akaCount) yaku.push({ id: DORA_IDS.aka, han: ctx.akaCount });

  const han = yakuHan + dora + ura + ctx.akaCount;
  const { base, limit, name } = basePoints(han, fu, ctx.cfg);
  return {
    yaku,
    yakuman: [],
    han,
    fu,
    base,
    limit: limit as YakuResult["limit"],
    name,
    decomposition: d,
  };
}

/** Rank readings by (役満数, 飜, 符) — the order the scoring rules prescribe. */
function better(a: YakuResult, b: YakuResult): boolean {
  if (a.yakuman.length !== b.yakuman.length) return a.yakuman.length > b.yakuman.length;
  if (a.han !== b.han) return a.han > b.han;
  return a.fu > b.fu;
}

/** The best-scoring reading of a completed hand, or null when it has no yaku. */
export function scoreWin(ctx: WinContext): YakuResult | null {
  const counts = zeros34();
  for (const t of ctx.hand) counts[tileType(t)]++;
  let best: YakuResult | null = null;
  for (const d of decomposeWin(counts.slice(), ctx.melds, ctx.winTile, ctx.tsumo)) {
    const r = evaluate(d, ctx, counts);
    if (r && (!best || better(r, best))) best = r;
  }
  return best;
}

/** The ron/tsumo legality gate: does this hand score at all? */
export function hasAnyYaku(ctx: WinContext): boolean {
  return scoreWin(ctx) !== null;
}
