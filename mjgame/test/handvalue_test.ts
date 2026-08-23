// 手牌価値 — the own-hand value model.
//
// Three claims, three groups of test:
//   1. the chain is a PROBABILITY — pwin never leaves [0,1], and it moves the
//      way the counting says it must (wider ukeire ↑, further from tenpai ↓,
//      a readier table ↓, more draws left ↑);
//   2. the value model applies its terms in the ONE stated order — dora, then
//      dealer, then the cap, and 本場/供託 outside the cap;
//   3. the numbers are PINNED. The offline fit and the live seat both call
//      `handOutlook`; if either ever forked, these two vectors would move.

import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import {
  DEFAULT_HAND,
  type HandFacts,
  handOutlook,
  type HandWeights,
  mergeHand,
} from "../src/ai/handvalue.ts";

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

/** A plain mid-game 1-shanten: menzen, no dora, a quiet table. */
const BASE: HandFacts = {
  shanten: 1,
  ukeire: 12,
  ukeireTypes: 4,
  unseenTotal: 90,
  turnsLeft: 10,
  junme: 8,
  dora: 0,
  open: 0,
  closed: true,
  riichi: false,
  yakuhaiTriplets: 0,
  yakuhaiPairs: 0,
  honitsu: false,
  ronnable: true,
  furiten: false,
  dealer: false,
  oppTenpai: [0.2, 0.1, 0.3],
  honba: 0,
  kyotaku: 0,
};

function facts(over: Partial<HandFacts>): HandFacts {
  return { ...BASE, ...over };
}

const W = DEFAULT_HAND;

// ---------------------------------------------------------------------------
// 1. pwin is a probability, and it is monotone in the right directions
// ---------------------------------------------------------------------------

Deno.test("handvalue: pwin stays in [0,1] over an extreme sweep", () => {
  for (const shanten of [0, 1, 2, 3, 6]) {
    for (const ukeire of [0, 4, 20, 136]) {
      for (const turnsLeft of [0, 1, 17, 100]) {
        for (const oppTenpai of [[0, 0, 0], [1, 1, 1], [4, 4, 4]]) {
          for (const unseenTotal of [0, 1, 136]) {
            const o = handOutlook(
              facts({ shanten, ukeire, turnsLeft, oppTenpai, unseenTotal }),
              W,
            );
            assert(o.pwin >= 0 && o.pwin <= 1, `pwin out of range: ${o.pwin}`);
            assert(Number.isFinite(o.value) && Number.isFinite(o.ev));
            assertAlmostEquals(o.ev, o.pwin * o.value, 1e-12);
          }
        }
      }
    }
  }
});

Deno.test("handvalue: no draws left ⇒ no win", () => {
  assertEquals(handOutlook(facts({ turnsLeft: 0, shanten: 0, ukeire: 8 }), W).pwin, 0);
  assertEquals(handOutlook(facts({ turnsLeft: -3 }), W).pwin, 0);
});

Deno.test("handvalue: pwin rises with ukeire, at every level", () => {
  for (const shanten of [0, 1, 2, 3]) {
    let prev = -1;
    for (const ukeire of [0, 2, 4, 8, 16, 24]) {
      const p = handOutlook(facts({ shanten, ukeire }), W).pwin;
      assert(p > prev, `shanten ${shanten}: ukeire ${ukeire} gave ${p} ≤ ${prev}`);
      prev = p;
    }
  }
});

Deno.test("handvalue: pwin falls as the hand gets further from tenpai", () => {
  let prev = 2;
  for (const shanten of [0, 1, 2, 3]) {
    const p = handOutlook(facts({ shanten, ukeire: 12 }), W).pwin;
    assert(p < prev, `shanten ${shanten} gave ${p} ≥ ${prev}`);
    prev = p;
  }
  // Beyond three away the chain saturates: rung 0 is the last rung there is.
  assertEquals(
    handOutlook(facts({ shanten: 3, ukeire: 12 }), W).pwin,
    handOutlook(facts({ shanten: 6, ukeire: 12 }), W).pwin,
  );
});

Deno.test("handvalue: pwin falls as the table gets readier", () => {
  let prev = 2;
  for (const x of [0, 0.25, 0.5, 1, 2]) {
    const p = handOutlook(facts({ oppTenpai: [x, x, x] }), W).pwin;
    assert(p < prev, `Σ oppTenpai ${3 * x} gave ${p} ≥ ${prev}`);
    prev = p;
  }
});

Deno.test("handvalue: pwin rises with the draws left", () => {
  // Never falls: every extra own draw adds a non-negative term.
  let prev = -1;
  for (const turnsLeft of [0, 1, 2, 5, 10, 17]) {
    const p = handOutlook(facts({ turnsLeft }), W).pwin;
    assert(p >= prev, `turnsLeft ${turnsLeft} gave ${p} < ${prev}`);
    prev = p;
  }
  // A 1-shanten needs one draw to reach tenpai and a second to win on it, so
  // the chain is genuinely flat at zero until then and strict afterwards.
  assertEquals(handOutlook(facts({ turnsLeft: 1 }), W).pwin, 0);
  prev = 0;
  for (const turnsLeft of [2, 3, 5, 10, 17]) {
    const p = handOutlook(facts({ turnsLeft }), W).pwin;
    assert(p > prev, `turnsLeft ${turnsLeft} gave ${p} ≤ ${prev}`);
    prev = p;
  }
  // A tenpai hand does collect on its very first draw.
  assert(handOutlook(facts({ shanten: 0, ukeire: 8, turnsLeft: 1 }), W).pwin > 0);
  // The sweep is capped, so a silly wall does not run forever.
  assertEquals(
    handOutlook(facts({ turnsLeft: 20 }), W).pwin,
    handOutlook(facts({ turnsLeft: 1000 }), W).pwin,
  );
});

Deno.test("handvalue: a ronnable, non-furiten tenpai beats a furiten one", () => {
  const live = handOutlook(facts({ shanten: 0, ukeire: 8 }), W).pwin;
  const furiten = handOutlook(facts({ shanten: 0, ukeire: 8, furiten: true }), W).pwin;
  const tsumoOnly = handOutlook(facts({ shanten: 0, ukeire: 8, ronnable: false }), W).pwin;
  assert(live > furiten);
  assertAlmostEquals(furiten, tsumoOnly, 1e-12);
  // Exactly the stated multiple: 1 + 3·ronFactor.
  assert(live < tsumoOnly * (1 + 3 * W.ronFactor) + 1e-12);
  // A declared riichi is ronnable whatever the flag says.
  assertEquals(
    handOutlook(facts({ shanten: 0, ukeire: 8, ronnable: false, riichi: true }), W).pwin,
    live,
  );
});

// ---------------------------------------------------------------------------
// 2. the value model
// ---------------------------------------------------------------------------

Deno.test("handvalue: bases select on menzen / furiten / 染め手", () => {
  const v = (o: Partial<HandFacts>) => handOutlook(facts(o), W).value;
  assertEquals(v({}), W.valueRiichi);
  assertEquals(v({ furiten: true }), W.valueDamaten);
  assertEquals(v({ furiten: true, riichi: true }), W.valueRiichi);
  assertEquals(v({ closed: false }), W.valueOpen);
  assertEquals(v({ closed: false, honitsu: true }), W.valueHonitsu);
  // 染め手 is an OPEN-hand read here; a closed hand already prices as riichi.
  assertEquals(v({ honitsu: true }), W.valueRiichi);
});

Deno.test("handvalue: value rises with dora and 役牌, a pair counting half", () => {
  const v = (o: Partial<HandFacts>) => handOutlook(facts(o), W).value;
  assertEquals(v({ dora: 2 }) - v({}), 2 * W.valuePerDora);
  assertEquals(v({ yakuhaiTriplets: 1 }) - v({}), W.valueYakuhai);
  assertEquals(v({ yakuhaiPairs: 1 }) - v({}), 0.5 * W.valueYakuhai);
});

Deno.test("handvalue: dealer multiplies before the cap", () => {
  const w: HandWeights = { ...W, valueCap: 1e9 };
  const flat = handOutlook(facts({ dora: 1 }), w).value;
  const dealer = handOutlook(facts({ dora: 1, dealer: true }), w).value;
  assertAlmostEquals(dealer, flat * W.valueDealer, 1e-9);
});

Deno.test("handvalue: the cap binds, and 本場/供託 are added on top of it", () => {
  const huge = facts({ dora: 20, dealer: true });
  assertEquals(handOutlook(huge, W).value, W.valueCap);
  assertEquals(
    handOutlook({ ...huge, honba: 3, kyotaku: 2 }, W).value,
    W.valueCap + 3 * 300 + 2 * 1000,
  );
  // …and on a hand nowhere near the cap, the same exact add-ons.
  assertEquals(
    handOutlook(facts({ honba: 1, kyotaku: 1 }), W).value,
    W.valueRiichi + 300 + 1000,
  );
});

// ---------------------------------------------------------------------------
// 3. mergeHand
// ---------------------------------------------------------------------------

Deno.test("handvalue: mergeHand fills from the defaults and copies the tuple", () => {
  assertEquals(mergeHand(), DEFAULT_HAND);
  assertEquals(mergeHand(undefined).meanUkeire, DEFAULT_HAND.meanUkeire);

  const m = mergeHand({ ronFactor: 0.9 });
  assertEquals(m.ronFactor, 0.9);
  assertEquals(m.meanUkeire, DEFAULT_HAND.meanUkeire);
  assert(m.meanUkeire !== DEFAULT_HAND.meanUkeire, "must not alias the defaults");
  assertEquals(m.pushScale, DEFAULT_HAND.pushScale);
  assertEquals(m.evWeight, DEFAULT_HAND.evWeight);

  const full = mergeHand({ meanUkeire: [1, 2, 3, 4] });
  assertEquals(full.meanUkeire, [1, 2, 3, 4]);
  assertEquals(full.valueCap, DEFAULT_HAND.valueCap);

  // A short array out of a hand-written --ktune file keeps four rungs: the
  // missing ones fall back cell by cell instead of arriving as `undefined`.
  const short = mergeHand(
    { meanUkeire: [30] as unknown as readonly [number, number, number, number] },
  );
  assertEquals(short.meanUkeire, [30, 20, 14, 6]);
  assert(short.meanUkeire.every((x) => typeof x === "number"));
});

// ---------------------------------------------------------------------------
// 4. pinned — the fit and the policy must never fork
// ---------------------------------------------------------------------------

Deno.test("handvalue: pinned outlook for a 1-shanten menzen hand", () => {
  const o = handOutlook(BASE, DEFAULT_HAND);
  assertAlmostEquals(o.pwin, 0.28451521457467505, 1e-12);
  assertAlmostEquals(o.value, 7000, 1e-12);
  assertAlmostEquals(o.ev, 1991.6065020227254, 1e-12);
});

Deno.test("handvalue: pinned outlook for a dealer riichi tenpai on a live table", () => {
  const f = facts({
    shanten: 0,
    ukeire: 8,
    ukeireTypes: 2,
    unseenTotal: 70,
    turnsLeft: 8,
    junme: 10,
    dora: 3,
    closed: true,
    riichi: true,
    dealer: true,
    oppTenpai: [1, 0.35, 0.2],
    honba: 2,
    kyotaku: 1,
  });
  const o = handOutlook(f, DEFAULT_HAND);
  assertAlmostEquals(o.pwin, 0.6681791119144961, 1e-12);
  assertAlmostEquals(o.value, 17600, 1e-12);
  assertAlmostEquals(o.ev, 11759.952369695131, 1e-12);
});
