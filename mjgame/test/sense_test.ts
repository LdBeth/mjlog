// 色読み (`ai/sense.ts`): the 感性 field sense — トイツ場 and 染め場.
//
// The doctrine (owner, 2026-08-28): 色読み is NOT a river read. It senses what
// kind of 場 the board has become — pairs flowing (トイツ場) or a color taking
// over (染め場) — and prices the FIELD, never a reconstructed hand. The facts
// are fixed arithmetic; all behaviour flows through three consumption weights
// that init to 0, so the load-bearing claim mirrors `riichi_head_test.ts`:
// INIT-EQUIVALENCE — a seat carrying `sense: {}` plays bit-for-bit the hanchan
// the sense-less seat plays, and a hostile vector visibly changes the games.

import { assert, assertEquals } from "@std/assert";
import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import {
  chiitoiShanten,
  fieldSense,
  INIT_SENSE,
  mergeSense,
  senseActive,
} from "../src/ai/sense.ts";
import { headless, loadKtune } from "../src/harness.ts";
import type { KTune } from "../src/harness.ts";
import type { Observation } from "../src/observe.ts";
import { JANKI } from "../src/rules.ts";
import { tiles } from "./helpers.ts";

const SEED = 8191;
const GAMES = 4;

// ---------------------------------------------------------------------------
// hand-built observations (the pattern of heuristic_test.ts, facts-only slice)
// ---------------------------------------------------------------------------

/** Discards in order, junme 1, 2, 3, … — the shape `fieldSense` reads. */
function river(spec: string): RiverEntry[] {
  return tiles(spec).map((tile, i): RiverEntry => ({
    tile,
    junme: i + 1,
    tsumogiri: false,
    riichiDeclare: false,
  }));
}

/** An opponent pon, seat-relative index `who`. */
function pon(spec: string, who: number): Meld {
  const ts = tiles(spec);
  return { kind: "pon", who, fromWho: (who + 1) % 4, tiles: ts, calledTile: ts[0] };
}

/**
 * Only the fields `fieldSense` reads need to be truthful (hand, rivers, melds);
 * the rest is the minimal internally consistent filler.
 */
function obsFor(over: Partial<Observation> = {}): Observation {
  const hand = over.hand ?? tiles("123456789m1122p東");
  return {
    seat: 0,
    kyoku: 0,
    honba: 0,
    kyotaku: 0,
    junme: 3,
    wallRemaining: 58,
    hand,
    drawn: null,
    melds: [[], [], [], []],
    rivers: [[], [], [], []],
    scores: [25000, 25000, 25000, 25000],
    riichi: [false, false, false, false],
    riichiJunme: [-1, -1, -1, -1],
    doraIndicators: tiles("9s"),
    seatWind: 27,
    roundWind: 27,
    akaIds: JANKI.akaIds,
    shanten: 1,
    waits: [],
    ronnable: [],
    katagari: false,
    discardInfo: new Map(),
    tsumogiriLock: false,
    ukeire: [],
    doraCount: 0,
    furiten: { permanent: false, temporary: false, riichi: false },
    danger: new Map(),
    violations: [0, 0, 0, 0],
    legal: [],
    ...over,
  };
}

Deno.test("sense: a balanced early field senses nothing", () => {
  // Two honor cuts per seat — no number discards at all, hand only two pairs.
  const f = fieldSense(obsFor({
    rivers: [river("北西"), river("北西"), river("北西"), river("北西")],
  }));
  assertEquals(f.someba, [0, 0, 0]);
  assertEquals(f.hot, 0);
  assert(f.toitsuba < 0.1, `toitsuba ${f.toitsuba} — 平場のはず`);
});

Deno.test("sense: a river void of one suit is a 染め場", () => {
  // Opponent 1: twelve number discards, not one manzu — the classic dyer.
  const f = fieldSense(obsFor({
    rivers: [[], river("123456789p123s"), [], []],
  }));
  assert(f.someba[0] >= 0.8, `someba[m] ${f.someba[0]} — 染め場のはず`);
  assertEquals(f.hot, f.someba[0]);
  // …and the dye source's own discards are proven out.
  for (const t of tiles("123456789p123s")) {
    const ty = Math.floor(t / 4);
    assert(f.safe[0].has(ty), `type ${ty} は染め手本人の河 — safe のはず`);
  }
});

Deno.test("sense: 字牌保持 corroborates the dye — honor cuts cool it", () => {
  // The same twelve number discards in both rivers, so the void score itself is
  // identical (`nNum` counts number tiles only); the second river merely also
  // shows four honors. A dyer keeps honors for the pair and the 役牌, so the
  // river that has been shedding them makes the weaker claim.
  const kept = fieldSense(obsFor({ rivers: [[], river("123456789p123s"), [], []] }));
  const shed = fieldSense(obsFor({
    rivers: [[], river("123456789p123s北北北北"), [], []],
  }));
  assert(kept.someba[0] > shed.someba[0], `字牌0枚 ${kept.someba[0]} ≦ 字牌4枚 ${shed.someba[0]}`);
  assertEquals(shed.someba[0], kept.someba[0] * 0.5);
  // Two honors is the intermediate rung, not the full discount.
  const some = fieldSense(obsFor({ rivers: [[], river("123456789p123s北北"), [], []] }));
  assertEquals(some.someba[0], kept.someba[0] * 0.8);
  assert(some.someba[0] > shed.someba[0]);
});

Deno.test("sense: a same-suit meld reinforces the dye", () => {
  // A modest void alone (six off-suit discards, evidence still ramping)…
  const rivers: RiverEntry[][] = [[], river("123456p"), [], []];
  const plain = fieldSense(obsFor({ rivers }));
  // …versus the same river with an all-manzu pon showing.
  const melded = fieldSense(obsFor({
    rivers,
    melds: [[], [pon("555m", 1)], [], []],
  }));
  assert(plain.someba[0] > 0, "六枚の河でも証拠は立ち始める");
  assert(
    melded.someba[0] > plain.someba[0],
    `meld ${melded.someba[0]} ≦ river-only ${plain.someba[0]}`,
  );
});

Deno.test("sense: pon + duplicated discards + paired draws is a トイツ場", () => {
  const balanced = fieldSense(obsFor({
    rivers: [river("北西"), river("北西"), river("北西"), river("北西")],
  }));
  const paired = fieldSense(obsFor({
    // Every early type discarded twice across the table.
    rivers: [river("1m2m3m"), river("1m2m3m"), river("1p2p3p"), river("1p2p3p")],
    melds: [[], [pon("777s", 1)], [pon("888s", 2)], []],
    hand: tiles("1122334455m123s"), // five pairs — the draws themselves pair up
  }));
  assert(paired.toitsuba > 0.5, `toitsuba ${paired.toitsuba} — トイツ場のはず`);
  assert(paired.toitsuba > balanced.toitsuba);
});

// ---------------------------------------------------------------------------
// chiitoi shanten
// ---------------------------------------------------------------------------

function countsOf(spec: string): number[] {
  const counts = new Array(34).fill(0);
  for (const t of tiles(spec)) counts[Math.floor(t / 4)]++;
  return counts;
}

Deno.test("sense: chiitoiShanten — 6 − pairs, held back under 7 kinds", () => {
  // Six pairs + a singleton: tenpai.
  assertEquals(chiitoiShanten(countsOf("1122334455667m")), 0);
  // Four pairs + five distinct singles: 2向聴.
  assertEquals(chiitoiShanten(countsOf("11223344m12345p")), 2);
  // Six pairs but only six kinds (a triplet wastes its third): the kinds
  // shortfall holds the count back — 6 − 6 + (7 − 6) = 1.
  assertEquals(chiitoiShanten(countsOf("1122334455m666p")), 1);
});

// ---------------------------------------------------------------------------
// merge / identity
// ---------------------------------------------------------------------------

Deno.test("sense: mergeSense fills partials against the zero identity", () => {
  assertEquals(mergeSense(), INIT_SENSE);
  assertEquals(mergeSense({}), INIT_SENSE);
  const w = mergeSense({ someRisk: 120 });
  assertEquals(w.someRisk, 120);
  assertEquals(w.somePressure, 0);
  assertEquals(w.chiitoiTax, 0);
  assert(!senseActive(INIT_SENSE));
  assert(senseActive(w));
  assert(senseActive(mergeSense({ chiitoiTax: 1 })));
});

// ---------------------------------------------------------------------------
// init-equivalence over real hanchan
// ---------------------------------------------------------------------------

Deno.test("sense: `{}` (⇒ zero weights) plays the identical hanchan — kkkk", () => {
  const plain = headless(GAMES, SEED, "kkkk", {});
  const inited = headless(GAMES, SEED, "kkkk", { ktune: { sense: {} } });
  assertEquals(inited.results, plain.results);
});

Deno.test("sense: `{}` plays the identical hanchan — khhh + champion", () => {
  const CHAMPION: KTune = loadKtune(
    new URL("../weights/champion.json", import.meta.url).pathname,
  );
  const plain = headless(GAMES, SEED, "khhh", { ktune: CHAMPION });
  const inited = headless(GAMES, SEED, "khhh", {
    ktune: { ...CHAMPION, sense: {} },
  });
  assertEquals(inited.results, plain.results);
});

// ---------------------------------------------------------------------------
// 生牌の役牌 — the OTHER 感性 surcharge, composed through the same `surcharge`
// hook as `senseRisk` (see `HeuristicWeights.liveYakuhai`). It lives beside the
// sense identity tests because it makes the same load-bearing claim: DEFAULT 0
// ⇒ bit-for-bit the prior game, a live weight ⇒ visibly different games.
// ---------------------------------------------------------------------------

Deno.test("liveYakuhai: the default 0 plays the identical hanchan — kkkk", () => {
  const plain = headless(GAMES, SEED, "kkkk", {});
  const zeroed = headless(GAMES, SEED, "kkkk", {
    ktune: { heuristic: { liveYakuhai: 0 } },
  });
  assertEquals(zeroed.results, plain.results);
});

Deno.test("liveYakuhai: the default 0 plays the identical hanchan — khhh + champion", () => {
  const CHAMPION: KTune = loadKtune(
    new URL("../weights/champion.json", import.meta.url).pathname,
  );
  const plain = headless(GAMES, SEED, "khhh", { ktune: CHAMPION });
  const zeroed = headless(GAMES, SEED, "khhh", {
    ktune: { ...CHAMPION, heuristic: { ...CHAMPION.heuristic, liveYakuhai: 0 } },
  });
  assertEquals(zeroed.results, plain.results);
});

Deno.test("liveYakuhai: a live weight changes the games", () => {
  const plain = headless(GAMES, SEED, "kkkk", {});
  const charged = headless(GAMES, SEED, "kkkk", {
    ktune: { heuristic: { liveYakuhai: 200 } },
  });
  assert(
    JSON.stringify(charged.results) !== JSON.stringify(plain.results),
    "生牌役牌の課金が一局も動かさない — 消費に届いていない",
  );
});

// ---------------------------------------------------------------------------
// the block is live
// ---------------------------------------------------------------------------

Deno.test("sense: a hostile vector changes the games", () => {
  const plain = headless(GAMES, SEED, "kkkk", {});
  const sensed = headless(GAMES, SEED, "kkkk", {
    ktune: { sense: { someRisk: 5000, somePressure: 3, chiitoiTax: 100000 } },
  });
  assert(
    JSON.stringify(sensed.results) !== JSON.stringify(plain.results),
    "重い sense ベクトルでも一局も動かない — 消費に届いていない",
  );
});
