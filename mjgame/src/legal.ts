// Legal-action enumeration and claim resolution.
//
// The dividing line that makes the dojo ruleset work: this module is
// PERMISSIVE about 禁じ手. A 大明槓, a first-turn honor discard, a non-tenpai
// dora cut and a 地獄単騎 riichi are all enumerated here as perfectly legal
// moves — the penalty ledger records them afterwards. What this module DOES
// restrict is only:
//   - win validity: furiten, 見せ牌 ron blocks, 和了放棄 sanctions,
//   - mechanics: 同巡内食い替え, the post-riichi discard lock, having the tiles.

import type { Meld, Tile } from "mjrender/model.ts";
import { rankOfType, suitOfType, tileType } from "mjrender/tiles.ts";
import { countsFromTiles, shanten } from "./kernel.ts";
import type { Table } from "./table.ts";
import { anyFuriten } from "./table.ts";
import type { Action, Seat } from "./types.ts";
import { SEATS } from "./types.ts";

/**
 * Whether a completed shape actually scores. Injected so that `legal.ts` does
 * not depend on the scorer (and so M1 can run before yaku detection exists).
 */
export interface WinOracle {
  /**
   * `extra` carries the context the oracle cannot recover from the table
   * alone. Without it, a hand whose ONLY yaku would be 嶺上開花 or 槍槓 would be
   * refused by this gate and the win silently disallowed.
   */
  hasYaku(
    t: Table,
    seat: Seat,
    tile: Tile,
    tsumo: boolean,
    extra?: { rinshan?: boolean; chankan?: boolean },
  ): boolean;
}

/** Placeholder oracle: every complete shape counts. Replaced by `yaku.ts`. */
export const ANY_WIN: WinOracle = { hasYaku: () => true };

/**
 * One representative id per distinct tile *type* in hand, plus every aka copy —
 * except for the drawn tile's type, which gets two (see below).
 */
function discardCandidates(t: Table, seat: Seat, drawn: Tile | null): Tile[] {
  const seen = new Set<number>();
  const out: Tile[] = [];
  // Copies of one type are interchangeable in the hand but NOT in the river:
  // letting the drawn one go is a tsumogiri, letting a held one go is a tedashi,
  // and 不聴時ドラ切りをポンされた後 only the former is allowed. So when the drawn
  // tile's type is ALSO held, both variants are enumerated — collapsing them to
  // one id would make either the tsumogiri or the tedashi inexpressible, and the
  // river, the export and the 禁じ手 hooks all distinguish the two.
  //
  // Ordering (load-bearing — seeded policies consume candidate order): the held
  // representative keeps the position the type has always occupied and the drawn
  // id is inserted directly after it, so every other candidate keeps its relative
  // order. Held-first also means a by-type lookup (the TUI's slot fallback)
  // resolves a hand slot to the tedashi, while the drawn slot still matches the
  // drawn id exactly.
  const drawnType = drawn !== null && !t.cfg.akaIds.has(drawn) ? tileType(drawn) : -1;
  for (const id of t.hands[seat]) {
    // Aka copies stay individually selectable: cutting 0p is not cutting 5p.
    if (t.cfg.akaIds.has(id)) {
      out.push(id);
      continue;
    }
    const ty = tileType(id);
    if (seen.has(ty)) continue;
    seen.add(ty);
    if (ty !== drawnType) {
      out.push(id);
      continue;
    }
    const held = t.hands[seat].find(
      (x) => x !== drawn && !t.cfg.akaIds.has(x) && tileType(x) === ty,
    );
    if (held !== undefined) out.push(held);
    out.push(drawn!);
  }
  return out;
}

/**
 * Whether `counts` (`total` concealed tiles alongside `openMelds` melds) is a
 * complete winning shape.
 *
 * Shanten −1 IS completeness, and the kernel answers it in one call (native:
 * one FFI crossing) where a hand-rolled DFS costs 34 partition attempts. The
 * tile-count check is what `shanten` alone does not do: it happily reads a
 * complete 14 out of a 15-tile vector, which the callers here must not.
 * Irregular forms line up exactly — `shanten` gates 七対子/国士 on
 * `openMelds === 0`, the same condition a hand-rolled check would use.
 */
function isComplete(counts: number[], openMelds: number, total: number): boolean {
  const need = 4 - openMelds;
  if (need < 0 || total !== need * 3 + 2) return false;
  return shanten(counts, openMelds) === -1;
}

function tenpaiAfter(t: Table, seat: Seat, discard: Tile): boolean {
  const rest = [...t.hands[seat]];
  const i = rest.indexOf(discard);
  if (i < 0) return false;
  rest.splice(i, 1);
  const open = t.melds[seat].length;
  return shanten(countsFromTiles(rest), open, t.isMenzen(seat)) <= 0;
}

function canDeclareRiichi(t: Table, seat: Seat): boolean {
  return (
    !t.riichi[seat] &&
    t.isMenzen(seat) &&
    t.scores[seat] >= 1000 &&
    t.wall.remaining >= 4
  );
}

// ---------------------------------------------------------------------------
// Turn actions (the seat holds 14 tiles, or 13 right after calling)
// ---------------------------------------------------------------------------

export function turnActions(
  t: Table,
  seat: Seat,
  drawn: Tile | null,
  oracle: WinOracle = ANY_WIN,
  rinshan = false,
): Action[] {
  const out: Action[] = [];
  const banned = t.kuikaeBan?.seat === seat ? t.kuikaeBan.types : null;

  // --- discards ---
  if (t.riichi[seat] && drawn !== null) {
    // Post-riichi the hand is locked: the drawn tile is the only discard.
    out.push({ t: "discard", tile: drawn, riichi: false, tsumogiri: true });
  } else {
    for (const tile of discardCandidates(t, seat, drawn)) {
      if (banned?.has(tileType(tile))) continue;
      const tsumogiri = drawn !== null && tile === drawn;
      out.push({ t: "discard", tile, riichi: false, tsumogiri });
      if (canDeclareRiichi(t, seat) && tenpaiAfter(t, seat, tile)) {
        out.push({ t: "discard", tile, riichi: true, tsumogiri });
      }
    }
  }

  // --- tsumo and kans ---
  // 明槓 (kakan) is a dojo 禁じ手 but remains mechanically available; only the
  // fifth kan and an exhausted wall actually block it.
  //
  // Both questions read the same 34-vector, so it is built at most once.
  const hand = t.hands[seat];
  const wantsTsumo = drawn !== null && !t.sanctioned[seat];
  const wantsKan = t.kanTotal < 4 && t.wall.remaining > 0 && !t.sanctioned[seat];
  const counts = wantsTsumo || wantsKan ? countsFromTiles(hand) : null;

  if (
    wantsTsumo && isComplete(counts!, t.melds[seat].length, hand.length) &&
    oracle.hasYaku(t, seat, drawn!, true, { rinshan })
  ) {
    out.push({ t: "tsumo" });
  }

  if (wantsKan) {
    for (let ty = 0; ty < 34; ty++) {
      if (counts![ty] === 4) out.push({ t: "ankan", type: ty });
    }
    for (const m of t.melds[seat]) {
      if (m.kind !== "pon") continue;
      const ty = tileType(m.tiles[0]);
      const held = hand.find((id) => tileType(id) === ty);
      if (held !== undefined) out.push({ t: "kakan", tile: held });
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Claim actions (someone else discarded, or added a kan we may 槍槓)
// ---------------------------------------------------------------------------

/** Chi shapes `seat` can make from `tile`, as the two concealed tiles used. */
function chiShapes(t: Table, seat: Seat, tile: Tile): Array<[Tile, Tile]> {
  const ty = tileType(tile);
  if (ty >= 27) return [];
  const suit = suitOfType(ty);
  const r = rankOfType(ty);
  // One representative per distinct *kind* at each rank: copies of a tile are
  // interchangeable in a chi, but an aka is not a plain 5 — spending it or
  // keeping it are different options, so both representatives are offered.
  const pick = (rank: number): Tile[] => {
    if (rank < 1 || rank > 9) return [];
    let plain: Tile | undefined;
    let aka: Tile | undefined;
    for (const id of t.hands[seat]) {
      const q = tileType(id);
      if (suitOfType(q) !== suit || rankOfType(q) !== rank) continue;
      if (t.cfg.akaIds.has(id)) aka ??= id;
      else plain ??= id;
    }
    return [plain, aka].filter((x): x is Tile => x !== undefined);
  };

  const out: Array<[Tile, Tile]> = [];
  const combos: Array<[number, number]> = [[r - 2, r - 1], [r - 1, r + 1], [r + 1, r + 2]];
  for (const [x, y] of combos) {
    for (const a of pick(x)) {
      for (const b of pick(y)) out.push([a, b]);
    }
  }
  return out;
}

/**
 * Whether the caller would still have something to discard after this call.
 *
 * 同巡内食い替え can ban every tile a short hand has left: with three melds and
 * 5m6m7m8m concealed, chi-ing a 5m with 6m7m bans both 5m and 8m, leaving no
 * legal discard at all. A turn with an empty action list has no way out — the
 * TUI offers nothing selectable and a policy has nothing to return — so such a
 * call is simply not offered. This is a mechanical restriction, not a 禁じ手
 * judgement: the ledger has nothing to say about a move that cannot be played.
 */
function leavesLegalDiscard(t: Table, seat: Seat, action: Action): boolean {
  if (action.t !== "chi" && action.t !== "pon") return true;
  const banned = kuikaeTypes(action);
  const spent = new Set<Tile>(action.tiles);
  return t.hands[seat].some((id) => !spent.has(id) && !banned.has(tileType(id)));
}

/**
 * Whether `tile` completes `seat`'s hand as a shape — yaku, furiten and the
 * ron blocks are all somebody else's business. `round.ts` asks the same
 * question to decide 見逃し furiten, which must be judged on the shape alone.
 */
export function completesHand(t: Table, seat: Seat, tile: Tile): boolean {
  const hand = t.hands[seat];
  return isComplete(countsFromTiles([...hand, tile]), t.melds[seat].length, hand.length + 1);
}

export function claimActions(
  t: Table,
  seat: Seat,
  tile: Tile,
  from: Seat,
  oracle: WinOracle = ANY_WIN,
  chankan = false,
  /**
   * `completesHand(t, seat, tile)`, when the caller already knows it. `round.ts`
   * needs the same answer to judge 見逃し furiten, so it computes it once and
   * hands it in rather than paying for the shape twice per discard.
   */
  completes = completesHand(t, seat, tile),
): Action[] {
  if (seat === from) return [];
  const out: Action[] = [{ t: "pass" }];
  if (t.sanctioned[seat]) return out;

  // --- ron ---
  if (completes) {
    const ty = tileType(tile);
    const blocked = t.ronBlocked[seat].has(ty);
    if (
      !blocked && !anyFuriten(t.furiten[seat]) &&
      oracle.hasYaku(t, seat, tile, false, { chankan })
    ) {
      out.push({ t: "ron" });
    }
  }

  // A 槍槓 window offers nothing but ron (and pass).
  if (chankan) return out;

  // --- calls ---
  // Riichi locks the hand: no calls. A wall with no draws left ends the round
  // before anyone could use the meld.
  if (t.riichi[seat] || t.wall.remaining <= 0) return out;

  const ty = tileType(tile);
  const sameType = t.hands[seat].filter((id) => tileType(id) === ty);

  if (sameType.length >= 2) {
    // Offer each distinct pair, so a player can choose to keep or spend an aka.
    // Distinct means by aka composition, not by id: copies of a plain tile are
    // interchangeable, so e.g. 5p 5p 0p offers exactly 5p5p and 5p0p.
    // A pair is characterised entirely by how many of the two are aka (0, 1 or
    // 2), so the seen-set is three bits rather than a sorted string per pair.
    let combos = 0;
    const aka = sameType.map((id) => (t.cfg.akaIds.has(id) ? 1 : 0));
    for (let i = 0; i < sameType.length; i++) {
      for (let j = i + 1; j < sameType.length; j++) {
        const bit = 1 << (aka[i] + aka[j]);
        if (combos & bit) continue;
        combos |= bit;
        const pair: [Tile, Tile] = [sameType[i], sameType[j]];
        const pon: Action = { t: "pon", tiles: pair, called: tile };
        if (leavesLegalDiscard(t, seat, pon)) out.push(pon);
      }
    }
  }
  if (sameType.length >= 3 && t.kanTotal < 4) {
    out.push({ t: "daiminkan", called: tile });
  }
  // Chi is only from the seat to your left (kamicha).
  if (from === ((seat + 3) % 4)) {
    for (const pair of chiShapes(t, seat, tile)) {
      const chi: Action = { t: "chi", tiles: pair, called: tile };
      if (leavesLegalDiscard(t, seat, chi)) out.push(chi);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Claim resolution: priority, 頭ハネ, 三家和
// ---------------------------------------------------------------------------

type Claim =
  | { kind: "ron"; seats: Seat[] }
  | { kind: "call"; seat: Seat; action: Action }
  | { kind: "sanchahou"; seats: Seat[] }
  | null;

/**
 * Resolve everyone's reply to one discard.
 *
 * Priority is ron > pon/daiminkan > chi. With `cfg.doubleRon === false` the
 * dojo uses 頭ハネ: among multiple rons the seat nearest counter-clockwise from
 * the discarder takes it alone. 三家和 is checked BEFORE head-bumping, because
 * three simultaneous rons abort the hand rather than feeding the nearest seat.
 */
export function resolveClaims(
  t: Table,
  from: Seat,
  replies: Map<Seat, Action>,
): Claim {
  const order = (s: Seat) => (s - from + 4) % 4; // 1 = shimocha … 3 = kamicha

  const rons = SEATS.filter((s) => replies.get(s)?.t === "ron").sort(
    (a, b) => order(a) - order(b),
  );
  if (rons.length >= 3 && t.cfg.sanchahouDraw) return { kind: "sanchahou", seats: rons };
  if (rons.length > 0) {
    return { kind: "ron", seats: t.cfg.doubleRon ? rons : [rons[0]] };
  }

  const rank = (a: Action): number =>
    a.t === "daiminkan" || a.t === "pon" ? 2 : a.t === "chi" ? 1 : 0;
  let best: { seat: Seat; action: Action } | null = null;
  for (const s of SEATS) {
    const a = replies.get(s);
    if (!a || rank(a) === 0) continue;
    if (!best || rank(a) > rank(best.action)) best = { seat: s, action: a };
  }
  return best ? { kind: "call", seat: best.seat, action: best.action } : null;
}

// ---------------------------------------------------------------------------
// Meld construction and 食い替え
// ---------------------------------------------------------------------------

export function buildMeld(seat: Seat, from: Seat, action: Action, t: Table): Meld {
  switch (action.t) {
    case "chi":
      return {
        kind: "chi",
        who: seat,
        fromWho: from,
        tiles: [...action.tiles, action.called].sort((a, b) => a - b),
        calledTile: action.called,
      };
    case "pon":
      return {
        kind: "pon",
        who: seat,
        fromWho: from,
        tiles: [...action.tiles, action.called].sort((a, b) => a - b),
        calledTile: action.called,
      };
    case "daiminkan": {
      const ty = tileType(action.called);
      const held = t.hands[seat].filter((id) => tileType(id) === ty).slice(0, 3);
      return {
        kind: "daiminkan",
        who: seat,
        fromWho: from,
        tiles: [...held, action.called].sort((a, b) => a - b),
        calledTile: action.called,
      };
    }
    case "ankan": {
      const held = t.hands[seat].filter((id) => tileType(id) === action.type);
      return {
        kind: "ankan",
        who: seat,
        fromWho: seat,
        tiles: [...held].sort((a, b) => a - b),
        calledTile: held[0],
      };
    }
    case "kakan": {
      const ty = tileType(action.tile);
      const pon = t.melds[seat].find((m) => m.kind === "pon" && tileType(m.tiles[0]) === ty);
      if (!pon) throw new Error("kakan without a matching pon");
      return {
        kind: "shouminkan",
        who: seat,
        fromWho: pon.fromWho,
        tiles: [...pon.tiles, action.tile].sort((a, b) => a - b),
        calledTile: action.tile,
      };
    }
    default:
      throw new Error(`not a meld action: ${action.t}`);
  }
}

/**
 * Tile types the caller may not immediately discard (同巡内食い替え禁止):
 * the called tile's type, plus — for a two-sided chi — the tile at the other
 * end of the run, which would be a pure swap.
 */
export function kuikaeTypes(action: Action): Set<number> {
  const out = new Set<number>();
  if (action.t !== "chi" && action.t !== "pon") return out;
  out.add(tileType(action.called));
  if (action.t !== "chi") return out;

  const [a, b] = action.tiles.map(tileType).sort((x, y) => x - y);
  const c = tileType(action.called);
  if (b - a !== 1) return out; // kanchan chi has no suji swap
  if (c === a - 1 && rankOfType(b) < 9) out.add(b + 1);
  if (c === b + 1 && rankOfType(a) > 1) out.add(a - 1);
  return out;
}
