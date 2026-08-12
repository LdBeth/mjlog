// Hand-level questions the referee and the dojo rules ask.
//
// Distinct from `decompose.ts` (pure shape math) and `observe.ts` (what a policy
// sees): this module answers questions *about a seat at this table* — is it
// tenpai, on what, can it actually win on each of those, and which yaku are
// already locked in regardless of how the hand finishes.

import type { Meld, Tile } from "mjrender/model.ts";
import { countsFromTiles, shanten, ukeireTypes } from "mjrender/shanten.ts";
import { rankOfType, suitOfType, tileType } from "mjrender/tiles.ts";
import { isHonor, isYaochu } from "./tiles.ts";
import type { WinOracle } from "./legal.ts";
import { ANY_WIN } from "./legal.ts";
import type { Table } from "./table.ts";
import type { Seat } from "./types.ts";

export interface TenpaiInfo {
  tenpai: boolean;
  shanten: number;
  /** Every tile type that completes the hand. */
  waits: number[];
  /** Waits that would actually score (出和了可能形). */
  ronnable: number[];
  /**
   * 片和了り: some waits carry a yaku and some do not. The dojo forbids this
   * unless the yakuless side is 純カラ (no live copies) — the caller supplies
   * the live counts to decide that.
   */
  katagari: boolean;
}

/** Resting (3n+1) hand for a seat: concealed tiles minus the drawn tile. */
export function restingHand(t: Table, seat: Seat, drawn: Tile | null): Tile[] {
  const hand = [...t.hands[seat]];
  if (drawn === null) return hand;
  const i = hand.lastIndexOf(drawn);
  if (i >= 0) hand.splice(i, 1);
  return hand;
}

export function analyze(
  t: Table,
  seat: Seat,
  drawn: Tile | null = null,
  oracle: WinOracle = ANY_WIN,
): TenpaiInfo {
  const rest = restingHand(t, seat, drawn);
  const open = t.melds[seat].length;
  const closed = t.isMenzen(seat);
  const counts = countsFromTiles(rest);
  const sh = shanten(counts, open, closed);
  if (sh > 0) {
    return { tenpai: false, shanten: sh, waits: [], ronnable: [], katagari: false };
  }
  const waits = ukeireTypes(counts, open, closed, sh);
  const ronnable = waits.filter((w) => oracle.hasYaku(t, seat, w * 4, false));
  return {
    tenpai: true,
    shanten: sh,
    waits,
    ronnable,
    katagari: ronnable.length > 0 && ronnable.length < waits.length,
  };
}

/** Copies of a tile type that could still be out there, from `seat`'s view. */
export function liveCopies(t: Table, seat: Seat, type: number): number {
  return Math.max(0, 4 - t.visibleCounts(seat)[type]);
}

/**
 * Would declaring an ankan on `kanType` change what the hand waits on?
 * The dojo forbids it (テンパイが変わるカン不可) and so does standard riichi law.
 */
export function wouldChangeWait(
  t: Table,
  seat: Seat,
  kanType: number,
  drawn: Tile | null,
): boolean {
  const before = analyze(t, seat, drawn).waits;

  const after = [...t.hands[seat]].filter((id) => tileType(id) !== kanType);
  const open = t.melds[seat].length + 1;
  const closed = t.isMenzen(seat); // an ankan keeps the hand closed
  const counts = countsFromTiles(after);
  const sh = shanten(counts, open, closed);
  const now = sh <= 0 ? ukeireTypes(counts, open, closed, sh) : [];

  const a = [...before].sort((x, y) => x - y).join(",");
  const b = [...now].sort((x, y) => x - y).join(",");
  return a !== b;
}

/**
 * 送りカン: kanning a type you were already holding four of, rather than the
 * type you just drew. Legal everywhere else; a 禁じ手 in the dojo.
 */
export function isOkurikan(kanType: number, drawn: Tile | null): boolean {
  return drawn !== null && tileType(drawn) !== kanType;
}

// ---------------------------------------------------------------------------
// 後付け support
// ---------------------------------------------------------------------------

/**
 * Yaku that hold in EVERY completion of this hand, computed structurally
 * rather than by enumerating completions.
 *
 * This is the 後付け test: the dojo allows 先付け (call with a yaku already
 * determined) but forbids calling while the only yaku is one you still hope to
 * arrive. Enumerating completions instead would produce false negatives — a
 * single lucky completion that happens to carry a yaku does not make the call
 * 先付け in the dojo's reading.
 */
export function confirmedYaku(
  hand: readonly Tile[],
  melds: readonly Meld[],
  valueHonors: ReadonlySet<number>,
  kuitan: boolean,
): string[] {
  const out: string[] = [];
  const all = [...hand, ...melds.flatMap((m) => m.tiles)];
  const types = all.map(tileType);

  // 役牌: a melded triplet of a value honor can never go away.
  for (const m of melds) {
    const ty = tileType(m.tiles[0]);
    if (m.kind !== "chi" && valueHonors.has(ty)) {
      out.push("役牌");
      break;
    }
  }

  // 断幺九: no yaochu anywhere, and no partial shape that needs one.
  if (kuitan && types.every((ty) => !isYaochu(ty))) out.push("断幺九");

  // 混一色/清一色: one suit (plus honors) across hand and melds.
  const suits = new Set(types.filter((ty) => !isHonor(ty)).map((ty) => suitOfType(ty)));
  if (suits.size <= 1) {
    out.push(types.some(isHonor) ? "混一色" : "清一色");
  }

  // 対々和: every meld is a triplet/kan and the concealed part is pairs only.
  if (melds.length > 0 && melds.every((m) => m.kind !== "chi")) {
    const counts = new Map<number, number>();
    for (const ty of hand.map(tileType)) counts.set(ty, (counts.get(ty) ?? 0) + 1);
    if ([...counts.values()].every((n) => n >= 2)) out.push("対々和");
  }

  // 混全帯幺九: every block reachable from here still touches a yaochu.
  if (melds.length > 0 && melds.every((m) => m.tiles.map(tileType).some(isYaochu))) {
    const handTypes = new Set(hand.map(tileType));
    const reachable = [...handTypes].every((ty) => {
      if (isYaochu(ty)) return true;
      const r = rankOfType(ty);
      return r <= 3 || r >= 7; // could still be part of a 123 / 789 run
    });
    if (reachable) out.push("混全帯幺九");
  }

  return out;
}

/** Value honors this seat holds concealed as a pair — the classic バック shape. */
export function concealedYakuhaiPairs(
  hand: readonly Tile[],
  valueHonors: ReadonlySet<number>,
): number[] {
  const counts = new Map<number, number>();
  for (const ty of hand.map(tileType)) counts.set(ty, (counts.get(ty) ?? 0) + 1);
  return [...counts.entries()]
    .filter(([ty, n]) => n >= 2 && valueHonors.has(ty))
    .map(([ty]) => ty);
}
