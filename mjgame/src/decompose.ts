// Winning-hand decomposition: *every* distinct way a complete hand reads as
// 4 sets + a head, plus the two irregular forms (七対子 / 国士無双).
//
// Deliberately not built on mjrender's shanten DFS (mjrender/src/shanten.ts):
// that one prunes and collapses everything to a number, while the scorer needs
// the structure — 一盃口 / 三色同順 / 一気通貫 / 三暗刻 and fu can each prefer a
// different reading of the same 14 tiles, and the scorer maximises over them.
// The two are independently written on purpose; decompose_test.ts cross-checks
// this file against `shanten`/`ukeireTypes` on tens of thousands of hands.

import type { Meld, Tile } from "mjrender/model.ts";
import { tileType } from "mjrender/tiles.ts";
import type { Seat } from "./types.ts";
import { YAOCHU_TYPES } from "./tiles.ts";

export type BlockKind = "run" | "triplet" | "kan" | "pair";

export interface Block {
  kind: BlockKind;
  /** For "run", the LOWEST of the three types. Otherwise the type itself. */
  type: number;
  /** false for called melds (chi/pon/daiminkan/shouminkan); true for ankan and
   *  for everything formed from concealed tiles. */
  concealed: boolean;
  /** Seat the called tile came from; present only for called blocks. */
  fromWho?: Seat;
}

export type WaitShape =
  | "ryanmen"
  | "kanchan"
  | "penchan"
  | "shanpon"
  | "tanki"
  | "kokushi"
  | "kokushi13"
  | "chiitoi";

export interface Decomposition {
  form: "standard" | "chiitoi" | "kokushi";
  /** standard: exactly 4 sets, EXCLUDING the pair (called melds included, in
   *  the order they appear in `melds`, marked concealed:false — except ankan).
   *  chiitoi: the 7 pair-blocks. kokushi: empty. */
  blocks: Block[];
  /** standard: the head. chiitoi: the pair completed by the winning tile.
   *  kokushi: the doubled yaochu block. */
  pair: Block;
  /** Index into `blocks` of the block the winning tile completed, or -1 when
   *  the winning tile completed `pair` instead. */
  winBlock: number;
  wait: WaitShape;
}

// Called melds occupy the front of `blocks`; concealed sets follow in a
// canonical order (type, then run before triplet) so that two enumerations of
// the same reading compare equal.
function cmpBlock(a: Block, b: Block): number {
  return a.type - b.type || (a.kind === b.kind ? 0 : a.kind === "run" ? -1 : 1);
}

function blockKey(b: Block): string {
  return `${b.kind[0]}${b.type}${b.concealed ? "c" : "o"}${b.fromWho ?? ""}`;
}

/** Set-slot blocks for the called melds, in call order. 抜き (sanma) is ignored. */
function meldBlocks(melds: readonly Meld[]): Block[] {
  const out: Block[] = [];
  for (const m of melds) {
    if (m.kind === "nuki") continue;
    const type = Math.min(...m.tiles.map(tileType));
    const fromWho = m.fromWho as Seat;
    switch (m.kind) {
      case "chi":
        out.push({ kind: "run", type, concealed: false, fromWho });
        break;
      case "pon":
        out.push({ kind: "triplet", type, concealed: false, fromWho });
        break;
      case "daiminkan":
      case "shouminkan":
        out.push({ kind: "kan", type, concealed: false, fromWho });
        break;
      case "ankan":
        out.push({ kind: "kan", type, concealed: true });
        break;
    }
  }
  return out;
}

function meldSlots(melds: readonly Meld[]): number {
  let n = 0;
  for (const m of melds) if (m.kind !== "nuki") n++;
  return n;
}

function total(counts: readonly number[]): number {
  let n = 0;
  for (let t = 0; t < 34; t++) n += counts[t];
  return n;
}

/**
 * Peel `need` sets off the lowest non-empty type, backtracking; every complete
 * partition lands in `out`. Always attacking the lowest tile keeps the search
 * tiny (a 14-tile hand explores a handful of nodes) and emits each partition in
 * ascending order, though a type held 4+ times can still yield one partition by
 * two different orders — hence the canonical sort + dedup in `decomposeWin`.
 */
function peelSets(counts: number[], need: number, acc: Block[], out: Block[][]): void {
  let i = 0;
  while (i < 34 && counts[i] === 0) i++;
  if (i === 34) {
    if (need === 0) out.push(acc.slice());
    return;
  }
  if (need === 0) return; // leftover tiles ⇒ not a partition

  if (counts[i] >= 3) {
    counts[i] -= 3;
    acc.push({ kind: "triplet", type: i, concealed: true });
    peelSets(counts, need - 1, acc, out);
    acc.pop();
    counts[i] += 3;
  }
  const rank = i < 27 ? i % 9 : -1; // honors form no runs
  if (rank >= 0 && rank <= 6 && counts[i + 1] > 0 && counts[i + 2] > 0) {
    counts[i]--;
    counts[i + 1]--;
    counts[i + 2]--;
    acc.push({ kind: "run", type: i, concealed: true });
    peelSets(counts, need - 1, acc, out);
    acc.pop();
    counts[i]++;
    counts[i + 1]++;
    counts[i + 2]++;
  }
}

/** Existence-only twin of `peelSets`, for the AI's inner loops. Restores `counts`. */
function canPeel(counts: number[], need: number): boolean {
  let i = 0;
  while (i < 34 && counts[i] === 0) i++;
  if (i === 34) return need === 0;
  if (need === 0) return false;

  if (counts[i] >= 3) {
    counts[i] -= 3;
    const ok = canPeel(counts, need - 1);
    counts[i] += 3;
    if (ok) return true;
  }
  const rank = i < 27 ? i % 9 : -1;
  if (rank >= 0 && rank <= 6 && counts[i + 1] > 0 && counts[i + 2] > 0) {
    counts[i]--;
    counts[i + 1]--;
    counts[i + 2]--;
    const ok = canPeel(counts, need - 1);
    counts[i]++;
    counts[i + 1]++;
    counts[i + 2]++;
    if (ok) return true;
  }
  return false;
}

function isChiitoi(counts: readonly number[]): boolean {
  let pairs = 0;
  for (let t = 0; t < 34; t++) {
    if (counts[t] === 0) continue;
    if (counts[t] !== 2) return false;
    pairs++;
  }
  return pairs === 7;
}

/** The doubled yaochu type of a complete kokushi, or -1. */
function kokushiPairType(counts: readonly number[]): number {
  let doubled = -1;
  for (const t of YAOCHU_TYPES) {
    if (counts[t] === 0) return -1;
    if (counts[t] === 2) {
      if (doubled >= 0) return -1;
      doubled = t;
    } else if (counts[t] !== 1) return -1;
  }
  // 13 kinds + the doubled copy already accounts for all 14 tiles, so any
  // non-yaochu tile would push the total over 14 — checked by the caller.
  return doubled;
}

/**
 * Which wait shape does `winType` complete `b` with? null if `b` does not
 * contain `winType`. A run is penchan only when the missing partner would fall
 * off the 1/9 boundary (3 finishing 123, 7 finishing 789).
 */
function waitFor(b: Block, winType: number): WaitShape | null {
  if (b.kind === "triplet") return b.type === winType ? "shanpon" : null;
  if (b.kind !== "run") return null;
  const lo = b.type;
  if (winType === lo + 1) return "kanchan";
  if (winType === lo) return lo % 9 === 6 ? "penchan" : "ryanmen"; // 7 of 789
  if (winType === lo + 2) return lo % 9 === 0 ? "penchan" : "ryanmen"; // 3 of 123
  return null;
}

/**
 * All distinct winning decompositions of a 14-tile hand (or 14 - 3*melds
 * concealed tiles plus `melds`).
 * `counts34` MUST already include the winning tile and MUST NOT include meld
 * tiles. Returns [] if the hand is not a winning shape.
 */
export function decomposeWin(
  counts34: number[],
  melds: Meld[],
  winTile: Tile,
  tsumo: boolean,
): Decomposition[] {
  // `tsumo` is part of the frozen signature so a future rule can distinguish
  // tsumo-only readings (e.g. a 門前 fu variant); no current rule branches on
  // it, so it is accepted and unused rather than silently dropped.
  void tsumo;

  const slots = meldSlots(melds);
  const need = 4 - slots;
  if (need < 0) return [];
  const winType = tileType(winTile);
  if (counts34[winType] === 0) return []; // the winning tile must be in the hand
  if (total(counts34) !== need * 3 + 2) return [];

  const counts = counts34.slice();
  const out: Decomposition[] = [];
  const seen = new Set<string>();
  const push = (d: Decomposition): void => {
    const winKey = d.winBlock < 0 ? "-" : blockKey(d.blocks[d.winBlock]);
    // Keyed on block *contents*, not the winBlock index: two identical blocks
    // (e.g. 234m 234m both containing the winning 3m) are one reading.
    const key = `${d.form}|${d.blocks.map(blockKey).join(",")}|${
      blockKey(d.pair)
    }|${winKey}|${d.wait}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(d);
  };

  // --- standard form: a head plus 4 sets (melds fill the first `slots`) ---
  const called = meldBlocks(melds);
  for (let p = 0; p < 34; p++) {
    if (counts[p] < 2) continue;
    counts[p] -= 2;
    const parses: Block[][] = [];
    peelSets(counts, need, [], parses);
    counts[p] += 2;
    for (const sets of parses) {
      const blocks = called.concat(sets.sort(cmpBlock));
      const pair: Block = { kind: "pair", type: p, concealed: true };
      if (p === winType) {
        push({ form: "standard", blocks, pair, winBlock: -1, wait: "tanki" });
      }
      // A called meld is never the winning block: it was complete before the win.
      for (let i = called.length; i < blocks.length; i++) {
        const wait = waitFor(blocks[i], winType);
        if (wait) push({ form: "standard", blocks, pair, winBlock: i, wait });
      }
    }
  }

  // --- irregular forms: closed hands only ---
  if (slots === 0) {
    if (isChiitoi(counts)) {
      const blocks: Block[] = [];
      for (let t = 0; t < 34; t++) {
        if (counts[t] === 2) blocks.push({ kind: "pair", type: t, concealed: true });
      }
      // The winning tile always completes the head here, so winBlock stays -1
      // even though that same pair is also listed in `blocks`.
      push({
        form: "chiitoi",
        blocks,
        pair: { kind: "pair", type: winType, concealed: true },
        winBlock: -1,
        wait: "chiitoi",
      });
    }
    const doubled = kokushiPairType(counts);
    if (doubled >= 0) {
      push({
        form: "kokushi",
        blocks: [],
        pair: { kind: "pair", type: doubled, concealed: true },
        winBlock: -1,
        // Drawing the 14th of 13 distinct yaochu ⇒ the hand was a 13-sided wait.
        wait: counts[winType] === 2 ? "kokushi13" : "kokushi",
      });
    }
  }

  return out;
}

/** True iff `counts34` (+ `openMelds` melds) is a complete winning shape. */
export function isComplete(counts34: number[], openMelds: number): boolean {
  const need = 4 - openMelds;
  if (need < 0) return false;
  if (total(counts34) !== need * 3 + 2) return false;
  if (openMelds === 0 && (isChiitoi(counts34) || kokushiPairType(counts34) >= 0)) return true;

  // Mutate-and-restore: `waitTypes` calls this 34 times per hand, so no copy.
  for (let p = 0; p < 34; p++) {
    if (counts34[p] < 2) continue;
    counts34[p] -= 2;
    const ok = canPeel(counts34, need);
    counts34[p] += 2;
    if (ok) return true;
  }
  return false;
}

/** The tile types that would complete this hand (i.e. the waits) at tenpai. */
export function waitTypes(counts34: number[], melds: Meld[]): number[] {
  const openMelds = meldSlots(melds);
  const out: number[] = [];
  for (let t = 0; t < 34; t++) {
    if (counts34[t] >= 4) continue; // all four copies already in hand
    counts34[t]++;
    if (isComplete(counts34, openMelds)) out.push(t);
    counts34[t]--;
  }
  return out;
}
