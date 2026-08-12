// 点数計算と精算: the han/fu ladder, ron/tsumo payment splits, 罰符/流し満貫, the
// final standings, and the `Scorer` the game master runs on.
//
// The arithmetic mirrors mjrender/src/scoring.ts, which keeps it private; it is
// reimplemented here rather than exported from there so mjgame owns its rules
// (the dojo turns 数え役満 off, which mjrender's replay path never needs).
//
// `limit` is an index into LIMIT_NAMES below — note this is one wider than
// mjrender's Tenhou-XML table, which has no 三倍満 slot.

import type { Meld, Tile } from "mjrender/model.ts";
import { countsFromTiles, shanten } from "mjrender/shanten.ts";
import { doraFromIndicatorType, tileType } from "mjrender/tiles.ts";
import type { RuleConfig } from "./rules.ts";
import type { Scorer, WinFlags } from "./round.ts";
import type { Table } from "./table.ts";
import type { DrawKind, Seat, Violation, WinInfo } from "./types.ts";
import { SEATS } from "./types.ts";
import { hasAnyYaku, scoreWin as scoreHand, type WinContext } from "./yaku.ts";

const LIMIT_NAMES = ["", "満貫", "跳満", "倍満", "三倍満", "役満"] as const;

const ceil100 = (n: number): number => Math.ceil(n / 100) * 100;

/**
 * 基本点 for a han/fu pair, capped on the 満貫 ladder.
 * With `cfg.kazoeYakuman` off (the dojo setting), 13+ han settles as 三倍満.
 */
export function basePoints(
  han: number,
  fu: number,
  cfg: RuleConfig,
): { base: number; limit: number; name: string } {
  const at = (limit: number, base: number) => ({ base, limit, name: LIMIT_NAMES[limit] });
  if (han >= 13) return cfg.kazoeYakuman ? at(5, 8000) : at(4, 6000);
  if (han >= 11) return at(4, 6000);
  if (han >= 8) return at(3, 4000);
  if (han >= 6) return at(2, 3000);
  if (han >= 5) return at(1, 2000);
  const base = fu * (1 << (2 + han));
  if (base >= 2000) return at(1, 2000);
  // 切り上げ満貫 promotes exactly the 1920 cell (4飜30符 / 3飜60符).
  if (cfg.kiriageMangan && base >= 1920) return at(1, 2000);
  return { base, limit: 0, name: "" };
}

export function ronPayment(dealerWins: boolean, base: number): number {
  return ceil100(base * (dealerWins ? 6 : 4));
}

/**
 * Tsumo split. When the dealer wins there is no dealer to pay, so both fields
 * carry the same all-pay figure.
 */
export function tsumoPayment(
  dealerWins: boolean,
  base: number,
): { fromDealer: number; fromOther: number } {
  if (dealerWins) {
    const each = ceil100(base * 2);
    return { fromDealer: each, fromOther: each };
  }
  return { fromDealer: ceil100(base * 2), fromOther: ceil100(base) };
}

function totalWon(dealerWins: boolean, tsumo: boolean, base: number): number {
  if (!tsumo) return ronPayment(dealerWins, base);
  const p = tsumoPayment(dealerWins, base);
  return dealerWins ? p.fromOther * 3 : p.fromDealer + p.fromOther * 2;
}

function baseOf(w: WinInfo, cfg: RuleConfig): number {
  if (w.yakuman.length > 0) return 8000 * w.yakuman.length;
  return basePoints(w.han, w.fu, cfg).base;
}

// ---------------------------------------------------------------------------
// Final standings
// ---------------------------------------------------------------------------

export interface Standing {
  place: number;
  seat: Seat;
  score: number;
  /** 素点 + ウマ, in 1000-point units. */
  points: number;
  clean: boolean;
  violations: number;
}

/** 起家 proximity: the tie-break for equal scores. */
function prio(seat: number, initialEast: number): number {
  return (seat - initialEast + 4) % 4;
}

/**
 * The headline dojo rule: a player who broke a rule ranks below every clean
 * player, whatever the scores say. Within each group it is score, then 起家
 * proximity.
 */
export function finalStandings(
  scores: number[],
  initialEast: Seat,
  ledger: Violation[],
  cfg: RuleConfig,
): Standing[] {
  const violations = SEATS.map((s) => ledger.filter((v) => v.seat === s).length);
  const order = [...SEATS].sort((a, b) =>
    Number(violations[a] > 0) - Number(violations[b] > 0) ||
    scores[b] - scores[a] ||
    prio(a, initialEast) - prio(b, initialEast)
  );
  return order.map((seat, i) => {
    const raw = (scores[seat] - cfg.returnScore) / 1000;
    const base = cfg.truncateSub1000 ? Math.trunc(raw) : raw;
    return {
      place: i + 1,
      seat,
      score: scores[seat],
      points: base + cfg.uma[i],
      clean: violations[seat] === 0,
      violations: violations[seat],
    };
  });
}

// ---------------------------------------------------------------------------
// Table → WinContext
// ---------------------------------------------------------------------------

function doraTypesOf(indicators: readonly Tile[]): number[] {
  return indicators.map((t) => doraFromIndicatorType(tileType(t)));
}

function akaCountOf(cfg: RuleConfig, hand: Tile[], melds: Meld[]): number {
  let n = 0;
  for (const t of hand) if (cfg.akaIds.has(t)) n++;
  for (const m of melds) for (const t of m.tiles) if (cfg.akaIds.has(t)) n++;
  return n;
}

function buildContext(
  t: Table,
  who: Seat,
  winTile: Tile,
  flags: WinFlags,
): WinContext {
  // On tsumo the drawn tile is already in the hand; on ron it is not.
  const hand = flags.tsumo ? [...t.hands[who]] : [...t.hands[who], winTile];
  const melds = [...t.melds[who]];

  // 裏ドラ is riichi-only, and 槓裏 is a separate switch: without it only the
  // opening indicator's ura counts.
  let ura: Tile[] = [];
  if (flags.riichi && t.cfg.uradora) {
    const all = t.wall.uraIndicators();
    ura = t.cfg.kanUra ? all : all.slice(0, 1);
  }

  return {
    seat: who,
    hand,
    melds,
    winTile,
    tsumo: flags.tsumo,
    riichi: flags.riichi,
    doubleRiichi: flags.doubleRiichi,
    ippatsu: flags.ippatsu,
    rinshan: flags.rinshan,
    chankan: flags.chankan,
    haitei: flags.haitei,
    houtei: flags.houtei,
    tenhou: flags.tenhou,
    chiihou: flags.chiihou,
    seatWind: t.seatWindType(who),
    roundWind: t.roundWindType,
    doraTypes: doraTypesOf(t.indicators),
    uraTypes: doraTypesOf(ura),
    akaCount: akaCountOf(t.cfg, hand, melds),
    cfg: t.cfg,
  };
}

/**
 * Flags for the legality gate, which is called from `legal.ts` and so has no
 * `WinFlags` to hand. 海底/河底 are recoverable from the wall; 嶺上開花 and 槍槓
 * are not, so a hand whose *only* yaku would be one of those is refused here —
 * accepted, because both are vanishingly rare and the alternative is widening
 * the `WinOracle` signature that `round.ts` owns.
 */
function gateFlags(t: Table, seat: Seat, tsumo: boolean): WinFlags {
  const last = t.wall.remaining === 0;
  return {
    tsumo,
    riichi: t.riichi[seat],
    doubleRiichi: t.doubleRiichi[seat],
    ippatsu: t.ippatsu[seat],
    rinshan: false,
    chankan: false,
    haitei: tsumo && last,
    houtei: !tsumo && last,
    tenhou: t.cfg.tenhouChiihou && tsumo && t.firstTurnIntact && seat === t.dealer &&
      t.turnIndex === 0,
    chiihou: t.cfg.tenhouChiihou && tsumo && t.firstTurnIntact && seat !== t.dealer &&
      t.turnIndex < 4,
  };
}

// ---------------------------------------------------------------------------

export const scorer: Scorer = {
  hasYaku(
    t: Table,
    seat: Seat,
    tile: Tile,
    tsumo: boolean,
    extra?: { rinshan?: boolean; chankan?: boolean },
  ): boolean {
    // 嶺上開花 / 槍槓 are invisible in the table state, so the caller supplies
    // them; without that a hand whose only yaku is one of the two is refused.
    const flags = { ...gateFlags(t, seat, tsumo), ...extra };
    return hasAnyYaku(buildContext(t, seat, tile, flags));
  },

  scoreWin(t: Table, who: Seat, fromWho: Seat, winTile: Tile, flags: WinFlags): WinInfo | null {
    const ctx = buildContext(t, who, winTile, flags);
    const res = scoreHand(ctx);
    if (!res) return null;
    const dealerWins = who === t.dealer;
    return {
      who,
      fromWho,
      winTile,
      han: res.han,
      fu: res.fu,
      points: totalWon(dealerWins, flags.tsumo, res.base),
      limit: res.limit,
      yaku: res.yaku,
      yakuman: res.yakuman,
      doraIndicators: [...t.indicators],
      uraIndicators: uraIndicatorsOf(t, flags),
      hand: ctx.hand,
      melds: ctx.melds,
    };
  },

  tenpaiAtDraw(t: Table, seat: Seat): boolean {
    // 形式テンパイ: no yaku requirement, so this is a pure shape question.
    const open = t.melds[seat].length;
    return shanten(countsFromTiles(t.hands[seat]), open, t.isMenzen(seat)) <= 0;
  },

  winDeltas(t: Table, wins: WinInfo[]): number[] {
    const d = [0, 0, 0, 0];
    for (const w of wins) {
      const base = baseOf(w, t.cfg);
      const dealerWins = w.who === t.dealer;
      if (w.fromWho === w.who) {
        const p = tsumoPayment(dealerWins, base);
        for (const s of SEATS) {
          if (s === w.who) continue;
          const pay = (dealerWins || s !== t.dealer ? p.fromOther : p.fromDealer) +
            t.round.honba * 100;
          d[s] -= pay;
          d[w.who] += pay;
        }
      } else {
        const pay = ronPayment(dealerWins, base) + t.round.honba * 300;
        d[w.fromWho] -= pay;
        d[w.who] += pay;
      }
    }
    // 頭ハネ makes this list length 1 under the dojo rules; if a ruleset ever
    // allows double ron, the sticks go to the seat closest to the discarder,
    // which `wins` is already ordered by.
    if (wins.length) d[wins[0].who] += t.round.kyotaku * 1000;
    return d;
  },

  drawDeltas(t: Table, tenpai: boolean[], kind: DrawKind, nagashi: Seat[]): number[] {
    const d = [0, 0, 0, 0];

    // 流し満貫 pays like a mangan tsumo and REPLACES the 罰符 entirely.
    if (kind === "nagashi") {
      for (const w of nagashi) {
        const p = tsumoPayment(w === t.dealer, 2000);
        for (const s of SEATS) {
          if (s === w) continue;
          const pay = w === t.dealer || s !== t.dealer ? p.fromOther : p.fromDealer;
          d[s] -= pay;
          d[w] += pay;
        }
      }
      return d;
    }

    // 三家和 / 四開槓 / 四人立直 abort with no transfer.
    if (kind !== "exhaustive") return d;

    const yes = SEATS.filter((s) => tenpai[s]);
    if (yes.length === 0 || yes.length === 4) return d;
    const total = t.cfg.notenPenaltyTotal;
    const gain = total / yes.length;
    const loss = total / (4 - yes.length);
    for (const s of SEATS) d[s] += tenpai[s] ? gain : -loss;
    return d;
  },
};

/** The ura indicators actually in play for this win (see `buildContext`). */
function uraIndicatorsOf(t: Table, flags: WinFlags): Tile[] {
  if (!flags.riichi || !t.cfg.uradora) return [];
  const all = t.wall.uraIndicators();
  return t.cfg.kanUra ? all : all.slice(0, 1);
}
