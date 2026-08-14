// Hand-level questions the referee and the dojo rules ask.
//
// Distinct from `decompose.ts` (pure shape math) and `observe.ts` (what a policy
// sees): this module answers questions *about a seat at this table* — is it
// tenpai, on what, and can it actually win on each of those.
//
// Note what is deliberately absent: any structural guess at "the yaku this hand
// will end up with". The dojo's 後付け test is asked of the WAITING hand (does
// every winning tile carry a yaku?) and answered by the real scorer through
// `analyze`, not by pattern-matching a half-built hand.

import type { Tile } from "mjrender/model.ts";
import { countsFromTiles, shanten, ukeireTypes } from "mjrender/shanten.ts";
import { tileType } from "mjrender/tiles.ts";
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
