// Shanten (tiles-away-from-tenpai) and ukeire (tile acceptance) engine.
//
// Works on a 34-length count vector of concealed tiles. `openMelds` is the number
// of already-called melds (each fills one of the four set slots). Standard-form
// shanten is an exhaustive decomposition; chiitoitsu and kokushi are closed-only.

import type { Tile } from "./model.ts";
import { tileType } from "./tiles.ts";

export function countsFromTiles(tiles: Tile[]): number[] {
  const c = new Array<number>(34).fill(0);
  for (const id of tiles) c[tileType(id)]++;
  return c;
}

const YAOCHU = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]; // terminals + honors

/**
 * Standard-form shanten = 2*cap - V, where cap = 4 - openMelds set slots and
 * V = 2*sets + min(partials, free slots) + (1 if a pair head was found) is the
 * best decomposition value. One DFS peels blocks off the leftmost non-empty
 * tile; it also picks the pair head (once) inline, so there is no separate
 * head loop. It prunes once the slots are full and a head is secured.
 */
function standardShanten(counts: number[], openMelds: number): number {
  const cap = 4 - openMelds;
  if (cap < 0) return 8;
  let best = 0;

  const dfs = (i: number, sets: number, partials: number, head: number): void => {
    while (i < 34 && counts[i] === 0) i++;

    const used = Math.min(sets, cap);
    const v = 2 * used + Math.min(partials, cap - used) + head;
    if (v > best) best = v;
    if (i >= 34 || (used + partials >= cap && head)) return;

    const rank = i < 27 ? i % 9 : -1; // 0-based rank within suit, -1 for honors

    // complete triplet
    if (counts[i] >= 3) {
      counts[i] -= 3;
      dfs(i, sets + 1, partials, head);
      counts[i] += 3;
    }
    // complete sequence
    if (rank >= 0 && rank <= 6 && counts[i + 1] > 0 && counts[i + 2] > 0) {
      counts[i]--;
      counts[i + 1]--;
      counts[i + 2]--;
      dfs(i, sets + 1, partials, head);
      counts[i]++;
      counts[i + 1]++;
      counts[i + 2]++;
    }
    // pair reserved as the head (at most once)
    if (!head && counts[i] >= 2) {
      counts[i] -= 2;
      dfs(i, sets, partials, 1);
      counts[i] += 2;
    }
    // partial: pair toward a triplet
    if (counts[i] >= 2) {
      counts[i] -= 2;
      dfs(i, sets, partials + 1, head);
      counts[i] += 2;
    }
    // partial: ryanmen/penchan (i, i+1)
    if (rank >= 0 && rank <= 7 && counts[i + 1] > 0) {
      counts[i]--;
      counts[i + 1]--;
      dfs(i, sets, partials + 1, head);
      counts[i]++;
      counts[i + 1]++;
    }
    // partial: kanchan (i, i+2)
    if (rank >= 0 && rank <= 6 && counts[i + 2] > 0) {
      counts[i]--;
      counts[i + 2]--;
      dfs(i, sets, partials + 1, head);
      counts[i]++;
      counts[i + 2]++;
    }
    // drop a floater and move on
    counts[i]--;
    dfs(i, sets, partials, head);
    counts[i]++;
  };

  dfs(0, 0, 0, 0);
  return 2 * cap - best;
}

function chiitoitsuShanten(counts: number[]): number {
  let pairs = 0;
  let kinds = 0;
  for (let t = 0; t < 34; t++) {
    if (counts[t] >= 1) kinds++;
    if (counts[t] >= 2) pairs++;
  }
  return 6 - pairs + Math.max(0, 7 - kinds);
}

function kokushiShanten(counts: number[]): number {
  let kinds = 0;
  let hasPair = false;
  for (const t of YAOCHU) {
    if (counts[t] >= 1) kinds++;
    if (counts[t] >= 2) hasPair = true;
  }
  return 13 - kinds - (hasPair ? 1 : 0);
}

/**
 * Minimum shanten across standard / chiitoitsu / kokushi.
 * `closed` disables the two closed-only forms (i.e. when melds were called).
 */
export function shanten(counts: number[], openMelds = 0, closed = true): number {
  let s = standardShanten(counts, openMelds);
  if (closed && openMelds === 0) {
    s = Math.min(s, chiitoitsuShanten(counts), kokushiShanten(counts));
  }
  return s;
}

/**
 * Tile types whose addition lowers shanten of a resting (3n+1) hand.
 * At shanten 0 these are exactly the winning (wait) tiles.
 * Callers that already computed the hand's shanten can pass it as `base`.
 */
export function ukeireTypes(
  counts: number[],
  openMelds = 0,
  closed = true,
  base = shanten(counts, openMelds, closed),
): number[] {
  const out: number[] = [];
  for (let t = 0; t < 34; t++) {
    if (counts[t] >= 4) continue;
    counts[t]++;
    const s = shanten(counts, openMelds, closed);
    counts[t]--;
    if (s < base) out.push(t);
  }
  return out;
}
