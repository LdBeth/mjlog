// MJAI wire notation ↔ mjgame's 136-id tile scheme, and the arena ruleset.
//
// Verified against a captured riichi.dev wire log (runs/arena/validate-*.jsonl):
// the arena's RiichiEnv uses the SAME Tenhou 136-id scheme as mjgame/mjrender
// (its decoded observations carry raw ids — e.g. indicator "F" = id 129,
// 129>>2 = 32 = 發), and the red five of each suit is the FIRST physical copy:
// ids 16 (5mr), 52 (5pr), 88 (5sr). That set happens to equal mjrender's
// module-global default, so the board layer needs no `setAkaIds` call; only
// the Table's `cfg.akaIds` must be overridden (JANKI plays 赤5筒×2 = {52,53}).
//
// Strings are only half the story: a wire tile names a TYPE (plus redness),
// not a physical id. Id allocation for masked tiles belongs to the shadow
// state, not here — this module stays pure string↔type↔id arithmetic.

import type { Tile } from "mjrender/model.ts";
import { tileType } from "mjrender/tiles.ts";
import { JANKI, type RuleConfig } from "../rules.ts";

/** The arena's red fives: one per suit, first copy of each 5. */
export const ARENA_AKA_IDS: ReadonlySet<Tile> = new Set([16, 52, 88]);

/**
 * The ruleset the shadow table runs under. Everything that scores or settles
 * is the SERVER's business — this config only has to make the champion's own
 * reading of the board correct: aka identity for dora counting, and the same
 * kuitan/furiten conventions the arena plays.
 */
export const ARENA_CFG: RuleConfig = { ...JANKI, akaIds: ARENA_AKA_IDS };

const HONOR_PAI = ["E", "S", "W", "N", "P", "F", "C"] as const;
const SUIT_BASE: Record<string, number> = { m: 0, p: 9, s: 18 };

/** "3m" → type 2, "5pr" → type 13, "F" → type 32. Throws on garbage. */
export function paiToType(pai: string): number {
  const honor = HONOR_PAI.indexOf(pai as typeof HONOR_PAI[number]);
  if (honor >= 0) return 27 + honor;
  const rank = pai.charCodeAt(0) - 0x30;
  const base = SUIT_BASE[pai[1]];
  if (rank >= 1 && rank <= 9 && base !== undefined) return base + rank - 1;
  throw new Error(`MJAI牌表記が読めない: ${JSON.stringify(pai)}`);
}

/** Red five? ("5mr"/"5pr"/"5sr" — the arena never writes "0m".) */
export function paiIsRed(pai: string): boolean {
  return pai.length === 3 && pai[2] === "r";
}

/** Type (+redness) → wire string. */
export function typeToPai(ty: number, red = false): string {
  if (ty >= 27) return HONOR_PAI[ty - 27];
  const rank = (ty % 9) + 1;
  const suit = ty < 9 ? "m" : ty < 18 ? "p" : "s";
  return `${rank}${suit}${red ? "r" : ""}`;
}

/** Physical id → wire string, red-aware. */
export function idToPai(id: Tile): string {
  return typeToPai(tileType(id), ARENA_AKA_IDS.has(id));
}

/** Does this physical id answer to this wire string? */
export function idMatchesPai(id: Tile, pai: string): boolean {
  return tileType(id) === paiToType(pai) && ARENA_AKA_IDS.has(id) === paiIsRed(pai);
}

/**
 * Pick the id in `pool` that a wire string names, or null. A plain "5m" never
 * takes the red copy and vice versa; among equivalent copies the first wins
 * (they are indistinguishable to every consumer, all of which count by type).
 */
export function matchId(pool: readonly Tile[], pai: string): Tile | null {
  return pool.find((id) => idMatchesPai(id, pai)) ?? null;
}
