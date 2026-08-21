// The fixed 78-slot action space: indexing (a pure function of one action) and
// `resolve` (the lossy inverse, which must hand back an object FROM `legal` by
// identity because the game master compares replies by reference).
//
// The unit cases are hand-built one situation at a time; the last test is a
// property sweep over real self-play, where the round-trip law is checked at
// every single decision of a hanchan.

import { assert, assertEquals } from "@std/assert";
import type { Tile } from "mjrender/model.ts";
import { tileType } from "mjrender/tiles.ts";
import { RandomPolicy } from "../src/ai/random.ts";
import { runMatchSync } from "../src/match.ts";
import type { Observation } from "../src/observe.ts";
import type { SyncPolicy } from "../src/policy.ts";
import { actionIndex, ACTIONS, IDX, maskFor, maskIndices, resolve } from "../src/rl/actionspace.ts";
import { DOJO_HEADLESS, JANKI } from "../src/rules.ts";
import { scorer } from "../src/score.ts";
import type { Action, Seat } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { tiles } from "./helpers.ts";

function discard(tile: Tile, over: Partial<Action & { t: "discard" }> = {}): Action {
  return { t: "discard", tile, riichi: false, tsumogiri: false, ...over };
}

// ---------------------------------------------------------------------------
// actionIndex: the frozen layout
// ---------------------------------------------------------------------------

Deno.test("actionspace: a plain discard is indexed by tile type", () => {
  for (const spec of ["1m", "9m", "5p", "9s", "東", "中"]) {
    const [t] = tiles(spec);
    assertEquals(actionIndex(discard(t)), tileType(t), spec);
  }
  // The layout's two ends.
  assertEquals(actionIndex(discard(tiles("1m")[0])), 0);
  assertEquals(actionIndex(discard(tiles("中")[0])), 33);
});

Deno.test("actionspace: a riichi discard is 34 + tile type", () => {
  for (const spec of ["1m", "5p", "中"]) {
    const [t] = tiles(spec);
    assertEquals(actionIndex(discard(t, { riichi: true })), IDX.riichi + tileType(t), spec);
  }
  assertEquals(IDX.riichi, 34);
  assertEquals(actionIndex(discard(tiles("中")[0], { riichi: true })), 67);
});

Deno.test("actionspace: an aka five indexes exactly like a plain five", () => {
  // Redness lives in plane 5 of the feature encoding, never in the action slot:
  // the two must collide or the trainer's mask would name a copy it cannot pick.
  const [aka, , plain] = tiles("0p0p5p"); // 52 (red), 53 (red), 54 (plain)
  assertEquals(actionIndex(discard(aka)), actionIndex(discard(plain)));
  assertEquals(actionIndex(discard(aka)), tileType(plain));
});

Deno.test("actionspace: chi is indexed by where the called tile ranks in the run", () => {
  const [m3, m4, m5] = tiles("345m");
  const low: Action = { t: "chi", tiles: [m4, m5], called: m3 };
  const mid: Action = { t: "chi", tiles: [m3, m5], called: m4 };
  const high: Action = { t: "chi", tiles: [m3, m4], called: m5 };
  assertEquals(actionIndex(low), IDX.chiLow);
  assertEquals(actionIndex(mid), IDX.chiMid);
  assertEquals(actionIndex(high), IDX.chiHigh);
  assertEquals([IDX.chiLow, IDX.chiMid, IDX.chiHigh], [69, 70, 71]);

  // The hand tiles are not required to arrive sorted — position is computed on
  // the sorted run, so swapping them changes nothing.
  assertEquals(actionIndex({ t: "chi", tiles: [m5, m4], called: m3 }), IDX.chiLow);
  assertEquals(actionIndex({ t: "chi", tiles: [m4, m3], called: m5 }), IDX.chiHigh);
});

Deno.test("actionspace: the call and terminal slots sit where the trainer expects", () => {
  const [p5a, p5b, p5c] = tiles("0p0p5p");
  const cases: Array<[Action, number]> = [
    [{ t: "pon", tiles: [p5a, p5b], called: p5c }, 68],
    [{ t: "daiminkan", called: p5c }, 72],
    [{ t: "ankan", type: 13 }, 73],
    [{ t: "kakan", tile: p5c }, 74],
    [{ t: "ron" }, 75],
    [{ t: "tsumo" }, 76],
    [{ t: "pass" }, 77],
  ];
  for (const [a, i] of cases) assertEquals(actionIndex(a), i, a.t);
  assertEquals(
    [IDX.pon, IDX.daiminkan, IDX.ankan, IDX.kakan, IDX.ron, IDX.tsumo, IDX.pass],
    [68, 72, 73, 74, 75, 76, 77],
  );
  assertEquals(ACTIONS, 78);
});

// ---------------------------------------------------------------------------
// resolve: tie-breaking between actions that share one slot
// ---------------------------------------------------------------------------

Deno.test("resolve: tsumogiri wins over tedashi for the same tile type", () => {
  const [a, b] = tiles("1m1m"); // two copies of 一萬
  const tedashi = discard(a);
  const tsumogiri = discard(b, { tsumogiri: true });
  const legal = [tedashi, tsumogiri];
  assertEquals(resolve(0, legal), tsumogiri);
  // Order in `legal` must not matter.
  assertEquals(resolve(0, [tsumogiri, tedashi]), tsumogiri);
});

Deno.test("resolve: a plain copy is preferred over the aka five", () => {
  const [aka, , plain] = tiles("0p0p5p");
  const ctx = { akaIds: JANKI.akaIds };
  const akaAct = discard(aka);
  const plainAct = discard(plain);
  const i = tileType(plain);
  assertEquals(resolve(i, [akaAct, plainAct], ctx), plainAct, "spending the aka drops a dora");
  assertEquals(resolve(i, [plainAct, akaAct], ctx), plainAct, "order in `legal` must not matter");

  // Without `akaIds` there is nothing to test aka-ness against, so the tiebreak
  // falls through to the highest id — which under every ruleset here is a plain
  // copy, because the reds are always low copies of their type. Dropping the
  // context loses the guarantee, not the intent.
  assertEquals(resolve(i, [akaAct, plainAct]), plainAct);
});

Deno.test("resolve: tsumogiri outranks aka-avoidance", () => {
  // The keys are ordered [tsumogiri, aka, id]: a red five that was just drawn
  // still goes back out rather than breaking the hand open.
  const [aka, , plain] = tiles("0p0p5p");
  const legal = [discard(plain), discard(aka, { tsumogiri: true })];
  assertEquals(resolve(tileType(plain), legal, { akaIds: JANKI.akaIds }), legal[1]);
});

Deno.test("resolve: the pon that spends no aka is preferred", () => {
  const [r1, r2, p1, p2] = tiles("0p0p5p5p"); // 52, 53 red; 54, 55 plain
  const ctx = { akaIds: JANKI.akaIds };
  const withTwoAka: Action = { t: "pon", tiles: [r2, p1], called: r1 };
  const clean: Action = { t: "pon", tiles: [p1, p2], called: r1 };
  const legal = [withTwoAka, clean];
  assertEquals(resolve(IDX.pon, legal, ctx), clean);
  assertEquals(resolve(IDX.pon, [clean, withTwoAka], ctx), clean);
});

Deno.test("resolve: the chi that spends no aka is preferred", () => {
  const [aka, , plain] = tiles("0p0p5p");
  const [p6] = tiles("6p");
  const [called] = tiles("4p");
  const dirty: Action = { t: "chi", tiles: [aka, p6], called };
  const clean: Action = { t: "chi", tiles: [plain, p6], called };
  const legal = [dirty, clean];
  // 4p called onto 5p6p ⇒ the called tile is the low end of the run.
  assertEquals(actionIndex(dirty), IDX.chiLow);
  assertEquals(resolve(IDX.chiLow, legal, { akaIds: JANKI.akaIds }), clean);
});

Deno.test("resolve: multiple ankan are broken by the drawn tile, else by lowest type", () => {
  const manzu: Action = { t: "ankan", type: 0 };
  const pinzu: Action = { t: "ankan", type: 13 };
  const legal = [manzu, pinzu];
  const [drawn] = tiles("5p"); // type 13
  assertEquals(resolve(IDX.ankan, legal, { drawn }), pinzu, "the tile just drawn completes it");
  assertEquals(resolve(IDX.ankan, legal), manzu, "with no draw, the lowest type wins");
  assertEquals(resolve(IDX.ankan, legal, { drawn: null }), manzu);
});

Deno.test("resolve: multiple kakan are broken by the drawn tile, else by lowest type", () => {
  const [m1] = tiles("1m");
  const [aka, , plain] = tiles("0p0p5p");
  const manzu: Action = { t: "kakan", tile: m1 };
  const pinzu: Action = { t: "kakan", tile: aka };
  const legal = [manzu, pinzu];
  // A different copy of the same TYPE is what actually arrives from the wall.
  assertEquals(resolve(IDX.kakan, legal, { drawn: plain }), pinzu);
  assertEquals(resolve(IDX.kakan, legal), manzu);
});

Deno.test("resolve: returns an element of `legal` by identity", () => {
  const [a, b] = tiles("1m1m");
  const legal: Action[] = [discard(a), discard(b, { tsumogiri: true }), { t: "pass" }];
  for (const i of maskIndices(legal)) {
    const got = resolve(i, legal);
    assert(got !== null, `slot ${i} is masked but unresolvable`);
    assert(legal.includes(got), `slot ${i}: resolve returned a copy, not the legal object`);
    assert(
      got === legal[0] || got === legal[1] || got === legal[2],
      "identity, not structural equality",
    );
  }
  // A structurally identical twin is NOT what comes back.
  const twin = discard(b, { tsumogiri: true });
  assert(resolve(0, legal) !== twin);
});

Deno.test("resolve: an unmasked slot is null", () => {
  const legal: Action[] = [discard(tiles("1m")[0])];
  assertEquals(resolve(IDX.pass, legal), null);
  assertEquals(resolve(IDX.riichi, legal), null);
  assertEquals(resolve(1, legal), null);
  assertEquals(resolve(-1, legal), null);
  assertEquals(resolve(ACTIONS, legal), null);
  assertEquals(resolve(0, []), null);
});

// ---------------------------------------------------------------------------
// mask
// ---------------------------------------------------------------------------

Deno.test("actionspace: maskFor / maskIndices mark exactly the legal slots", () => {
  const [m1, m1b, m1c] = tiles("1m1m1m");
  const legal: Action[] = [
    discard(m1),
    discard(m1, { riichi: true }),
    { t: "pon", tiles: [m1b, m1c], called: m1 },
    { t: "pass" },
  ];
  const mask = maskFor(legal);
  assertEquals(mask.length, ACTIONS);
  assertEquals(mask instanceof Uint8Array, true);
  assertEquals(maskIndices(legal), [0, 34, 68, 77]);
  assertEquals(mask.reduce((a, b) => a + b, 0), 4);
  for (const i of [0, 34, 68, 77]) assertEquals(mask[i], 1, `slot ${i}`);
  for (const i of [1, 33, 35, 69, 75, 76]) assertEquals(mask[i], 0, `slot ${i}`);

  // Duplicates collapse: two copies of one type are one slot.
  const [a, b] = tiles("5p5p");
  assertEquals(maskIndices([discard(a), discard(b, { tsumogiri: true })]), [tileType(a)]);
  assertEquals(maskIndices([]), []);
});

// ---------------------------------------------------------------------------
// The round-trip law, swept over real games
// ---------------------------------------------------------------------------

/**
 * Delegates every decision to a `RandomPolicy`, but first checks the law the
 * trainer depends on at that exact observation:
 *
 *   ∀ i ∈ maskIndices(legal):  resolve(i, legal, ctx) ∈ legal  ∧
 *                              actionIndex(resolve(i, …)) === i
 *   ∀ a ∈ legal:               mask[actionIndex(a)] === 1
 */
class ProbePolicy implements SyncPolicy {
  readonly sync = true;
  decisions = 0;
  private inner: RandomPolicy;

  constructor(readonly name: string, seed: number) {
    this.inner = new RandomPolicy(name, seed);
  }

  reset(seed: number): void {
    this.inner.reset(seed);
  }

  decide(obs: Observation): Action {
    this.decisions++;
    const ctx = { drawn: obs.drawn, akaIds: obs.akaIds };
    const idx = maskIndices(obs.legal);
    assert(idx.length > 0, "a decision was requested with an empty mask");

    for (const i of idx) {
      const got = resolve(i, obs.legal, ctx);
      assert(got !== null, `slot ${i} masked but resolve returned null`);
      assert(obs.legal.includes(got), `slot ${i}: resolved action is not in legal (identity)`);
      assertEquals(actionIndex(got), i, `slot ${i}: round trip landed elsewhere`);
    }

    const mask = maskFor(obs.legal);
    for (const a of obs.legal) {
      const i = actionIndex(a);
      assert(i >= 0 && i < ACTIONS, `${a.t} indexed out of range: ${i}`);
      assertEquals(mask[i], 1, `${a.t} is legal but its own slot ${i} is unmasked`);
    }
    return this.inner.decide(obs);
  }
}

Deno.test("actionspace: the round trip holds at every decision of real self-play", () => {
  let total = 0;
  for (let seed = 1; seed <= 3; seed++) {
    const probes = SEATS.map((s: Seat) => new ProbePolicy(`probe${s}`, seed * 4 + s));
    const r = runMatchSync(probes, { seed, cfg: JANKI, dojo: DOJO_HEADLESS, scorer });
    assert(r.rounds.length > 0, `seed ${seed}: no rounds played`);
    const decisions = probes.reduce((n, p) => n + p.decisions, 0);
    assert(decisions > 0, `seed ${seed}: the probe never fired`);
    total += decisions;
  }
  assert(total > 1000, `only ${total} decisions swept — the probe is not seeing real play`);
});
