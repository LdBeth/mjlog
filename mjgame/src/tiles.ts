// Tile helpers layered on mjrender's encoding.
//
// Encoding recap (mjrender/src/model.ts): a tile is 0..135, its type is `id >> 2`
// (0-8 man, 9-17 pin, 18-26 sou, 27-30 winds, 31-33 dragons), and the four
// physical copies of a type are ids `type*4 .. type*4+3`.

import type { Tile } from "mjrender/model.ts";
import { rankOfType, tileType } from "mjrender/tiles.ts";

export * from "mjrender/tiles.ts";
export type { Tile };

/**
 * 雀鬼会 aka set: 赤5筒 ×2. 5p is type 13, so its four copies are ids 52..55;
 * we make the first two red. This differs from the Tenhou default {16,52,88}
 * (one red 5m/5p/5s), which mjrender hardcodes — see `setAkaIds()` there.
 */
export const AKA_5P: readonly Tile[] = [52, 53];

export function isHonor(type: number): boolean {
  return type >= 27;
}

export function isTerminal(type: number): boolean {
  return type < 27 && (type % 9 === 0 || type % 9 === 8);
}

export function isYaochu(type: number): boolean {
  return isHonor(type) || isTerminal(type);
}

export function isSimple(type: number): boolean {
  return !isYaochu(type);
}

/** Wind types 東南西北 = 27..30. */
export const WIND_TYPES: readonly number[] = [27, 28, 29, 30];
/** Dragon types 白發中 = 31..33. */
export const DRAGON_TYPES: readonly number[] = [31, 32, 33];

export const TERMINAL_TYPES: readonly number[] = [0, 8, 9, 17, 18, 26];
export const YAOCHU_TYPES: readonly number[] = [
  0,
  8,
  9,
  17,
  18,
  26,
  27,
  28,
  29,
  30,
  31,
  32,
  33,
];
/** 緑一色 tiles: 2,3,4,6,8 sou + 發. */
export const GREEN_TYPES: readonly number[] = [19, 20, 21, 23, 25, 32];

/**
 * Suji partners of a tile type: the tiles 3 ranks away in the same suit.
 * Honors have none. Used by the 即引っかけ and 腰 penalty rules.
 */
export function sujiTypes(type: number): number[] {
  if (isHonor(type)) return [];
  const r = rankOfType(type);
  const out: number[] = [];
  if (r > 3) out.push(type - 3);
  if (r < 7) out.push(type + 3);
  return out;
}

/** 34-length zero vector. */
export function zeros34(): number[] {
  return new Array<number>(34).fill(0);
}

export function countsOf(tiles: readonly Tile[]): number[] {
  const c = zeros34();
  for (const t of tiles) c[tileType(t)]++;
  return c;
}
