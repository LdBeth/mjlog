// decompose.ts is cross-checked two ways: property tests against mjrender's
// independently written shanten/ukeire engine (the valuable half — it catches
// anything our enumerator over- or under-counts), and table-driven structural
// cases pinning block layout, wait shapes and meld placement.

import { assertEquals } from "@std/assert";
import type { Meld, MeldKind, Tile } from "mjrender/model.ts";
import { shanten, ukeireTypes } from "mjrender/shanten.ts";
import { type Rng, sfc32 } from "../src/rng.ts";
import {
  type Block,
  decomposeWin,
  type Decomposition,
  isComplete,
  waitTypes,
} from "../src/decompose.ts";
import { countsOf, YAOCHU_TYPES, zeros34 } from "../src/tiles.ts";
import type { Seat } from "../src/types.ts";
import { tiles } from "./helpers.ts";

// ---------------------------------------------------------------------------
// Random hand generation
// ---------------------------------------------------------------------------

/** A hand described by tile *types*; ids are handed out by `realize`. */
interface HandSpec {
  melds: Array<{ kind: MeldKind; types: number[]; fromWho: Seat }>;
  hand: number[];
}

interface Dealt {
  counts: number[];
  melds: Meld[];
  hand: Tile[];
}

function realize(spec: HandSpec): Dealt {
  const next = zeros34();
  const id = (t: number): Tile => {
    if (next[t] >= 4) throw new Error(`generator handed out a 5th copy of type ${t}`);
    return t * 4 + next[t]++;
  };
  const melds: Meld[] = spec.melds.map((m) => {
    const ts = m.types.slice().sort((a, b) => a - b).map(id);
    return { kind: m.kind, who: 0, fromWho: m.fromWho, tiles: ts, calledTile: ts[0] };
  });
  const hand = spec.hand.map(id);
  return { counts: countsOf(hand), melds, hand };
}

/** `left[t]` = copies of type t still available. */
function fullPool(): number[] {
  return new Array<number>(34).fill(4);
}

function pick(rng: Rng, left: number[], n: number): number[] {
  const bag: number[] = [];
  for (let t = 0; t < 34; t++) for (let i = 0; i < left[t]; i++) bag.push(t);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const j = i + rng.int(bag.length - i);
    [bag[i], bag[j]] = [bag[j], bag[i]];
    out.push(bag[i]);
    left[bag[i]]--;
  }
  return out;
}

function randomMeld(rng: Rng, left: number[]): HandSpec["melds"][number] | null {
  for (let attempt = 0; attempt < 40; attempt++) {
    const roll = rng.int(10);
    const fromWho = (1 + rng.int(3)) as Seat;
    if (roll < 4) {
      const lo = rng.int(3) * 9 + rng.int(7);
      if (left[lo] && left[lo + 1] && left[lo + 2]) {
        left[lo]--, left[lo + 1]--, left[lo + 2]--;
        return { kind: "chi", types: [lo, lo + 1, lo + 2], fromWho };
      }
    } else if (roll < 8) {
      const t = rng.int(34);
      if (left[t] >= 3) {
        left[t] -= 3;
        return { kind: "pon", types: [t, t, t], fromWho };
      }
    } else {
      const t = rng.int(34);
      if (left[t] === 4) {
        left[t] -= 4;
        return { kind: "ankan", types: [t, t, t, t], fromWho: 0 };
      }
    }
  }
  return null;
}

/** A pure random deal: 14 - 3k concealed tiles behind k melds. */
function randomSpec(rng: Rng, k: number): HandSpec {
  const left = fullPool();
  const melds: HandSpec["melds"] = [];
  for (let i = 0; i < k; i++) {
    const m = randomMeld(rng, left);
    if (m) melds.push(m);
  }
  return { melds, hand: pick(rng, left, 14 - 3 * melds.length) };
}

/**
 * A *winning* hand (4 sets + pair), then 0-2 tiles swapped out. Pure random
 * deals essentially never complete, so this is what exercises the positive
 * side of the invariant and the tenpai comparison.
 */
function structuredSpec(rng: Rng, k: number): HandSpec {
  for (let attempt = 0; attempt < 20; attempt++) {
    const left = fullPool();
    const melds: HandSpec["melds"] = [];
    for (let i = 0; i < k; i++) {
      const m = randomMeld(rng, left);
      if (m) melds.push(m);
    }
    const hand: number[] = [];
    let ok = true;
    for (let s = melds.length; s < 4 && ok; s++) {
      ok = false;
      for (let tries = 0; tries < 40 && !ok; tries++) {
        if (rng.int(2) === 0) {
          const lo = rng.int(3) * 9 + rng.int(7);
          if (left[lo] && left[lo + 1] && left[lo + 2]) {
            left[lo]--, left[lo + 1]--, left[lo + 2]--;
            hand.push(lo, lo + 1, lo + 2);
            ok = true;
          }
        } else {
          const t = rng.int(34);
          if (left[t] >= 3) {
            left[t] -= 3;
            hand.push(t, t, t);
            ok = true;
          }
        }
      }
    }
    if (!ok) continue;
    let paired = false;
    for (let tries = 0; tries < 40 && !paired; tries++) {
      const t = rng.int(34);
      if (left[t] >= 2) {
        left[t] -= 2;
        hand.push(t, t);
        paired = true;
      }
    }
    if (!paired) continue;

    for (let swaps = rng.int(3); swaps > 0; swaps--) {
      const idx = rng.int(hand.length);
      left[hand[idx]]++;
      hand[idx] = pick(rng, left, 1)[0];
    }
    return { melds, hand };
  }
  return randomSpec(rng, k);
}

/**
 * Closed hands biased toward the irregular forms: neither 七対子 nor 国士無双
 * ever shows up in a uniform deal, so they get their own generator (again with
 * 0-2 tiles swapped out, which lands most of the rest at tenpai).
 */
function irregularSpec(rng: Rng): HandSpec {
  const left = fullPool();
  const hand: number[] = [];
  if (rng.int(2) === 0) {
    const seen = new Set<number>();
    while (seen.size < 7) {
      const t = rng.int(34);
      if (seen.has(t)) continue;
      seen.add(t);
      left[t] -= 2;
      hand.push(t, t);
    }
  } else {
    for (const t of YAOCHU_TYPES) {
      left[t]--;
      hand.push(t);
    }
    const doubled = YAOCHU_TYPES[rng.int(YAOCHU_TYPES.length)];
    left[doubled]--;
    hand.push(doubled);
  }
  for (let swaps = rng.int(3); swaps > 0; swaps--) {
    const idx = rng.int(hand.length);
    left[hand[idx]]++;
    hand[idx] = pick(rng, left, 1)[0];
  }
  return { melds: [], hand };
}

/** Concealed blocks + pair must re-add to exactly the concealed tiles. */
function blocksCover(d: Decomposition, nMelds: number, counts: number[]): boolean {
  const c = zeros34();
  const add = (b: Block) => {
    if (b.kind === "run") c[b.type]++, c[b.type + 1]++, c[b.type + 2]++;
    else if (b.kind === "pair") c[b.type] += 2;
    else c[b.type] += b.kind === "kan" ? 4 : 3;
  };
  if (d.form === "standard") {
    d.blocks.slice(nMelds).forEach(add);
    add(d.pair);
  } else if (d.form === "chiitoi") {
    d.blocks.forEach(add);
  } else {
    return true; // kokushi carries no blocks
  }
  for (let t = 0; t < 34; t++) if (c[t] !== counts[t]) return false;
  return true;
}

const MELD_WEIGHTS = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3];

Deno.test("decomposeWin/waitTypes agree with mjrender's shanten engine", () => {
  const rng = sfc32("decompose-invariants");
  const N = 20000;
  let completeSeen = 0;
  let tenpaiSeen = 0;
  const formsSeen: Record<string, number> = { standard: 0, chiitoi: 0, kokushi: 0 };

  for (let i = 0; i < N; i++) {
    const k = MELD_WEIGHTS[rng.int(MELD_WEIGHTS.length)];
    const spec = i % 4 === 0
      ? randomSpec(rng, k)
      : i % 4 === 3
      ? irregularSpec(rng)
      : structuredSpec(rng, k);
    const { counts, melds, hand } = realize(spec);
    // mjrender's `closed` gates the two closed-only forms; an ankan keeps a
    // hand closed (and its openMelds > 0 already disables them anyway).
    const closed = melds.every((m) => m.kind === "ankan");
    const winTile = hand[rng.int(hand.length)];

    const dec = decomposeWin(counts.slice(), melds, winTile, false);
    const sh = shanten(counts.slice(), melds.length, closed);
    assertEquals(
      dec.length > 0,
      sh === -1,
      `hand #${i} ${JSON.stringify(counts)} melds=${melds.length} shanten=${sh}`,
    );
    assertEquals(isComplete(counts.slice(), melds.length), sh === -1, `isComplete #${i}`);

    if (sh === -1) {
      completeSeen++;
      for (const d of dec) {
        formsSeen[d.form]++;
        if (d.form === "standard") assertEquals(d.blocks.length, 4, `4 sets #${i}`);
        assertEquals(blocksCover(d, melds.length, counts), true, `blocks cover #${i}`);
      }
    }

    // Drop one tile and compare the wait sets whenever that leaves tenpai.
    const drop = rng.int(hand.length);
    const c13 = counts.slice();
    c13[hand[drop] >> 2]--;
    if (shanten(c13.slice(), melds.length, closed) !== 0) continue;
    tenpaiSeen++;
    const mine = waitTypes(c13.slice(), melds).sort((a, b) => a - b);
    const theirs = ukeireTypes(c13.slice(), melds.length, closed, 0).sort((a, b) => a - b);
    assertEquals(mine, theirs, `waits #${i} ${JSON.stringify(c13)} melds=${melds.length}`);
  }

  // Guard against a generator regression silently emptying the property test.
  assertEquals(completeSeen > 1000, true, `too few complete hands: ${completeSeen}`);
  assertEquals(tenpaiSeen > 1000, true, `too few tenpai hands: ${tenpaiSeen}`);
  for (const form of ["standard", "chiitoi", "kokushi"]) {
    assertEquals(formsSeen[form] > 100, true, `too few ${form} wins: ${formsSeen[form]}`);
  }
});

// ---------------------------------------------------------------------------
// Structural cases
// ---------------------------------------------------------------------------

function dec(spec: string, win: string, melds: Meld[] = []): Decomposition[] {
  const hand = tiles(spec);
  const w = tiles(win)[0];
  return decomposeWin(countsOf(hand), melds, w, false);
}

Deno.test("111222333m reads as three triplets and as three runs", () => {
  // Tanki on 9s, so the only ambiguity is how the man tiles split.
  const ds = dec("111222333m456p99s", "9s");
  assertEquals(ds.length, 2);
  assertEquals(ds.map((d) => d.wait), ["tanki", "tanki"]);
  const kinds = ds.map((d) => d.blocks.map((b) => `${b.kind}${b.type}`).join(" "));
  assertEquals(kinds, [
    "triplet0 triplet1 triplet2 run12",
    "run0 run0 run0 run12",
  ]);
});

Deno.test("identical blocks holding the winning tile collapse to one reading", () => {
  // 1m sits in all three of 123m/123m/123m — one ryanmen reading, not three —
  // alongside the 111m triplet reading (shanpon).
  const ds = dec("111222333m456p99s", "1m");
  assertEquals(ds.map((d) => d.wait), ["shanpon", "ryanmen"]);
});

Deno.test("a 七対子 that also parses as 二盃口 returns both forms", () => {
  const ds = dec("223344m556677p88s", "8s");
  assertEquals(ds.map((d) => d.form), ["standard", "chiitoi"]);
  assertEquals(ds[0].blocks.map((b) => `${b.kind}${b.type}`), [
    "run1",
    "run1",
    "run13",
    "run13",
  ]);
  assertEquals(ds[0].wait, "tanki");
  assertEquals(ds[1].blocks.length, 7);
  assertEquals(ds[1].pair.type, 25); // 8s
});

Deno.test("kokushi: 13-sided vs single wait", () => {
  const thirteen = dec("119m19p19s東南西北白發中", "1m");
  assertEquals(thirteen.length, 1);
  assertEquals(thirteen[0].form, "kokushi");
  assertEquals(thirteen[0].wait, "kokushi13");
  assertEquals(thirteen[0].pair.type, 0);
  assertEquals(thirteen[0].blocks, []);

  // Same 14 tiles, but the 中 was the tile that arrived: the hand held 1m1m
  // already, so it was a plain single wait.
  const single = dec("119m19p19s東南西北白發中", "中");
  assertEquals(single.map((d) => d.wait), ["kokushi"]);
  assertEquals(single[0].pair.type, 0);
});

Deno.test("wait shapes", () => {
  const cases: Array<[string, string, string]> = [
    ["123m456m789m345p55s", "5s", "tanki"],
    ["123m456m789m11p222p", "2p", "shanpon"],
    ["123m456m789m11p234s", "3s", "kanchan"], // 3 in the middle of 234
    ["123m456m789m11p234s", "3m", "penchan"], // 3 finishing 123
    ["123m456m789m11p345s", "3s", "ryanmen"], // 3 finishing 345 (from 4-5)
    ["123m456m789m11p345s", "7m", "penchan"], // 7 finishing 789
    ["123m456m789m11p345s", "4m", "ryanmen"], // 4 finishing 456
  ];
  for (const [hand, win, wait] of cases) {
    const ds = dec(hand, win);
    assertEquals(ds.length, 1, `${hand} +${win}`);
    assertEquals(ds[0].wait, wait, `${hand} +${win}`);
    assertEquals(ds[0].winBlock >= 0, wait !== "tanki", `${hand} +${win} winBlock`);
  }
});

Deno.test("called melds fill set slots in call order", () => {
  const chi = tiles("456p");
  const pon = tiles("東東東");
  const melds: Meld[] = [
    { kind: "chi", who: 0, fromWho: 3, tiles: chi, calledTile: chi[0] },
    { kind: "pon", who: 0, fromWho: 1, tiles: pon, calledTile: pon[0] },
  ];
  const ds = dec("123m789m55s", "1m", melds);
  assertEquals(ds.length, 1);
  const d = ds[0];
  assertEquals(d.blocks, [
    { kind: "run", type: 12, concealed: false, fromWho: 3 }, // 4p
    { kind: "triplet", type: 27, concealed: false, fromWho: 1 }, // 東
    { kind: "run", type: 0, concealed: true }, // 123m
    { kind: "run", type: 6, concealed: true }, // 789m
  ]);
  assertEquals(d.pair, { kind: "pair", type: 22, concealed: true }); // 5s
  assertEquals(d.winBlock, 2);
  assertEquals(d.wait, "ryanmen");
});

Deno.test("ankan stays concealed and occupies a slot", () => {
  const kan = tiles("1111m");
  const melds: Meld[] = [
    { kind: "ankan", who: 0, fromWho: 0, tiles: kan, calledTile: kan[0] },
  ];
  const ds = dec("456m789m123p55s", "7m", melds);
  assertEquals(ds.length, 1);
  assertEquals(ds[0].blocks.length, 4);
  assertEquals(ds[0].blocks[0], { kind: "kan", type: 0, concealed: true });
  assertEquals(ds[0].blocks[0].fromWho, undefined);
  assertEquals(ds[0].wait, "penchan"); // 7 finishing 789, waiting from 8-9
});

Deno.test("isComplete and waitTypes on a plain tenpai hand", () => {
  const hand = countsOf(tiles("123m456m789m11p23s"));
  assertEquals(isComplete(hand.slice(), 0), false);
  assertEquals(waitTypes(hand.slice(), []), [18, 21]); // 1s, 4s
  const won = hand.slice();
  won[18]++;
  assertEquals(isComplete(won, 0), true);
});

Deno.test("non-winning hands and mismatched tile counts return nothing", () => {
  assertEquals(dec("123m456m789m11p24s", "4s").length, 0);
  assertEquals(dec("123m456m789m11p", "1p").length, 0); // 11 tiles, no melds
  assertEquals(dec("123m456m789m345p55s", "1s").length, 0); // winning tile absent
});
