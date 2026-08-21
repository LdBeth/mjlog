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
import { tileType } from "mjrender/tiles.ts";
import { countsFromTiles, shanten, ukeireTypes } from "./kernel.ts";
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

/**
 * The shape half of `analyze`'s work: the 3n+1 hand as a 34-vector, the meld
 * context it is read against, and its shanten.
 *
 * Split out so a caller that has already paid for it — `observe()` computes
 * exactly this for its own `shanten`/`ukeire` fields, and again per candidate
 * discard — can hand it back instead of making `analyze` redo it.
 */
export interface RestingShape {
  counts: number[];
  open: number;
  closed: boolean;
  shanten: number;
  /** Ukeire types at shanten <= 0; derived on demand when omitted. */
  waits?: number[];
}

/** Resting (3n+1) hand for a seat: concealed tiles minus the drawn tile. */
function restingHand(t: Table, seat: Seat, drawn: Tile | null): Tile[] {
  const hand = [...t.hands[seat]];
  if (drawn === null) return hand;
  const i = hand.lastIndexOf(drawn);
  if (i >= 0) hand.splice(i, 1);
  return hand;
}

function restingShape(t: Table, seat: Seat, drawn: Tile | null): RestingShape {
  const counts = countsFromTiles(restingHand(t, seat, drawn));
  const open = t.melds[seat].length;
  const closed = t.isMenzen(seat);
  return { counts, open, closed, shanten: shanten(counts, open, closed) };
}

export function analyze(
  t: Table,
  seat: Seat,
  drawn: Tile | null = null,
  oracle: WinOracle = ANY_WIN,
  /** The seat's resting shape, when the caller already computed it. */
  pre?: RestingShape,
): TenpaiInfo {
  const s = pre ?? restingShape(t, seat, drawn);
  if (s.shanten > 0) {
    return { tenpai: false, shanten: s.shanten, waits: [], ronnable: [], katagari: false };
  }
  const waits = s.waits ?? ukeireTypes(s.counts, s.open, s.closed, s.shanten);
  const ronnable = waits.filter((w) => oracle.hasYaku(t, seat, w * 4, false));
  return {
    tenpai: true,
    shanten: s.shanten,
    waits,
    ronnable,
    katagari: ronnable.length > 0 && ronnable.length < waits.length,
  };
}

/**
 * Would declaring an ankan on `kanType` change what the hand waits on?
 * The dojo forbids it (テンパイが変わるカン不可) and so does standard riichi law.
 *
 * Both sides are built from the SAME tiles, so the only thing that can differ is
 * where the four copies are counted: in the hand as a concealed triplet plus the
 * drawn fourth (before), or as one more meld slot (after). That symmetry is why
 * the caller must say WHEN it is asking. `committed` means the kan meld is
 * already on the table — which is the case for the `on-kan` hook, since
 * `round.ts` emits the meld before calling `onAction`. Getting that wrong
 * double-counts the kan (a 10-tile hand judged against four melds' worth of
 * slots) and answers "changed" for every ankan there is.
 */
export function wouldChangeWait(
  t: Table,
  seat: Seat,
  kanType: number,
  committed = false,
): boolean {
  // Everything that is not part of the kan. The four copies are added back
  // explicitly below, so it does not matter whether the hand still holds them.
  const rest = countsFromTiles([...t.hands[seat]].filter((id) => tileType(id) !== kanType));
  const open = Math.max(0, t.melds[seat].length - (committed ? 1 : 0));
  const closed = t.isMenzen(seat); // an ankan keeps the hand closed

  // Before: the resting (3n+1) view of the pre-kan hand — the triplet is in the
  // hand and the drawn fourth copy, being the tile under consideration, is not.
  const kept = [...rest];
  kept[kanType] += 3;
  const shBefore = shanten(kept, open, closed);
  const before = shBefore <= 0 ? ukeireTypes(kept, open, closed, shBefore) : [];

  // After: the same tiles minus the triplet, with the kan as an extra meld.
  const shAfter = shanten(rest, open + 1, closed);
  const after = shAfter <= 0 ? ukeireTypes(rest, open + 1, closed, shAfter) : [];

  const a = [...before].sort((x, y) => x - y).join(",");
  const b = [...after].sort((x, y) => x - y).join(",");
  return a !== b;
}

/**
 * 送りカン: kanning a type you were already holding four of, rather than the
 * type you just drew. Legal everywhere else; a 禁じ手 in the dojo.
 */
export function isOkurikan(kanType: number, drawn: Tile | null): boolean {
  return drawn !== null && tileType(drawn) !== kanType;
}
