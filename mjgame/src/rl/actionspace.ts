// The fixed 78-slot action space, and the bridge back to the engine's `Action`.
//
// The engine's action list is variable-length and carries tile IDS (which copy
// of the 5p, which two tiles the chi spends). A network needs a fixed vector,
// so the space here is by tile TYPE plus a handful of call slots — and the lost
// information (which physical copy) is put back by `resolve`, which must return
// an element of `legal` BY IDENTITY: the game master compares by reference and
// refuses anything else.
//
// Layout (FROZEN — shared with the trainer):
//   0–33   plain discard of tile type t
//   34–67  riichi discard of tile type t   (34 + t)
//   68     pon
//   69     chi, called tile lowest of the run
//   70     chi, called tile middle of the run
//   71     chi, called tile highest of the run
//   72     daiminkan   73  ankan   74  kakan
//   75     ron         76  tsumo   77  pass

import type { Tile } from "mjrender/model.ts";
import { tileType } from "mjrender/tiles.ts";
import type { Action } from "../types.ts";

export const ACTIONS = 78;

export const IDX = {
  discard: 0,
  riichi: 34,
  pon: 68,
  chiLow: 69,
  chiMid: 70,
  chiHigh: 71,
  daiminkan: 72,
  ankan: 73,
  kakan: 74,
  ron: 75,
  tsumo: 76,
  pass: 77,
} as const;

const EMPTY: ReadonlySet<Tile> = new Set<Tile>();

/**
 * Extra context `resolve` uses to break ties. Both fields are optional: the
 * frozen signature is `resolve(index, legal)`, and everything here only ever
 * chooses BETWEEN actions that already map to the requested index.
 */
export interface ResolveCtx {
  /** The tile just drawn (`obs.drawn`) — disambiguates multiple ankan/kakan. */
  drawn?: Tile | null;
  /** The ruleset's red fives (`obs.akaIds`) — without it, aka is not preferred. */
  akaIds?: ReadonlySet<Tile>;
}

/**
 * Where the called tile ranks inside the sorted three-tile run: 0, 1 or 2.
 *
 * `a.tiles` is exactly two tiles, so the sorted pair is one comparison — the
 * `map().sort()` this used to spell it with allocated two arrays for every chi
 * candidate `resolve` walks past, and `resolve` is on the decision path.
 */
function chiPosition(a: Action & { t: "chi" }): number {
  const c = tileType(a.called);
  const t0 = tileType(a.tiles[0]), t1 = tileType(a.tiles[1]);
  const x = t0 < t1 ? t0 : t1, y = t0 < t1 ? t1 : t0;
  return c < x ? 0 : c > y ? 2 : 1;
}

/**
 * The slot an engine action occupies.
 *
 * Red fives need no say here: a discard is indexed by tile TYPE and an aka
 * shares its type with the plain copies (plane 5 of the feature encoding is
 * where redness lives). The function took an `akaIds` argument for a while and
 * never once consulted it.
 */
export function actionIndex(a: Action): number {
  switch (a.t) {
    case "discard":
      return (a.riichi ? IDX.riichi : IDX.discard) + tileType(a.tile);
    case "pon":
      return IDX.pon;
    case "chi":
      return IDX.chiLow + chiPosition(a);
    case "daiminkan":
      return IDX.daiminkan;
    case "ankan":
      return IDX.ankan;
    case "kakan":
      return IDX.kakan;
    case "ron":
      return IDX.ron;
    case "tsumo":
      return IDX.tsumo;
    case "pass":
      return IDX.pass;
  }
}

/** 1 for every slot at least one legal action maps to. */
export function maskFor(legal: Action[]): Uint8Array {
  const m = new Uint8Array(ACTIONS);
  for (const a of legal) {
    const i = actionIndex(a);
    if (i >= 0 && i < ACTIONS) m[i] = 1;
  }
  return m;
}

/** The masked slots, ascending — the `"mask"` field of a trajectory line. */
export function maskIndices(legal: Action[]): number[] {
  const out: number[] = [];
  const m = maskFor(legal);
  for (let i = 0; i < ACTIONS; i++) if (m[i]) out.push(i);
  return out;
}

/**
 * Preference key for one candidate; lower sorts first.
 *
 *   discard — tsumogiri before tedashi, then a plain copy before a red five
 *             (spending the aka on a discard throws away a dora). The last key
 *             is the HIGHEST id, not the lowest: every red five in this project
 *             is one of the low copies of its type (雀鬼会 赤5筒 = 52,53;
 *             Tenhou's default = the %4==0 copy), so a caller that omitted
 *             `akaIds` still errs toward keeping the red rather than cutting it.
 *   pon/chi — the shape that spends no aka
 *   ankan/kakan — the drawn tile's type when we know it, else the lowest type
 */
function preference(a: Action, ctx: ResolveCtx): number[] {
  const aka = ctx.akaIds ?? EMPTY;
  const drawnType = ctx.drawn === null || ctx.drawn === undefined ? -1 : tileType(ctx.drawn);
  switch (a.t) {
    case "discard":
      return [a.tsumogiri ? 0 : 1, aka.has(a.tile) ? 1 : 0, -a.tile];
    case "pon":
    case "chi":
      return [a.tiles.filter((t) => aka.has(t)).length, a.tiles[0], a.tiles[1]];
    case "ankan":
      return [a.type === drawnType ? 0 : 1, a.type];
    case "kakan": {
      const ty = tileType(a.tile);
      return [ty === drawnType ? 0 : 1, ty];
    }
    default:
      return [0];
  }
}

function less(x: number[], y: number[]): boolean {
  for (let i = 0; i < Math.max(x.length, y.length); i++) {
    const a = x[i] ?? 0, b = y[i] ?? 0;
    if (a !== b) return a < b;
  }
  return false;
}

/**
 * The legal action occupying `index`, or null when the slot is not legal.
 *
 * The returned action is always one of the objects in `legal` (identity, not a
 * copy) and always satisfies `actionIndex(result) === index`.
 */
export function resolve(index: number, legal: Action[], ctx: ResolveCtx = {}): Action | null {
  let best: Action | null = null;
  let bestKey: number[] | null = null;
  for (const a of legal) {
    if (actionIndex(a) !== index) continue;
    const key = preference(a, ctx);
    if (bestKey === null || less(key, bestKey)) {
      best = a;
      bestKey = key;
    }
  }
  return best;
}
