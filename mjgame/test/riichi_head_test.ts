// The M12 riichi head: declare-vs-damaten behind an absent-by-default `--ktune`
// block.
//
// The load-bearing claim is INIT-EQUIVALENCE: a head merged from `{}` is
// `INIT_RIICHI` (bias 1, weights 0), and a seat carrying it plays bit-for-bit
// the hanchan the head-less seat plays — which enforces both the
// always-declare init and the finiteness of every feature (one NaN in the sum
// and the equivalence breaks). The live test then shows the block is a real
// switch: a head that hates declaring produces a run with fewer riichi.

import { assert, assertEquals } from "@std/assert";
import { decideRiichi, INIT_RIICHI, mergeRiichi } from "../src/ai/riichi.ts";
import type { RiichiFeatures } from "../src/ai/riichi.ts";
import { headless, loadKtune } from "../src/harness.ts";
import type { KTune } from "../src/harness.ts";

const SEED = 8191;
const GAMES = 4;

/** An arbitrary but realistic gated-in decision. */
const FEATURES: RiichiFeatures = {
  ev: 2.1,
  pwin: 0.35,
  value: 6.0,
  liveWaits: 6,
  waitTypes: 2,
  junme: 8,
  turnsLeft: 10,
  dora: 2,
  dealer: 0,
  oppRiichi: 1,
  kyotaku: 0,
  improvable: 1,
  tenpaiHeld: 0,
  holdShape: 1,
};

Deno.test("riichi head: INIT declares on any finite features", () => {
  assert(decideRiichi(FEATURES, INIT_RIICHI));
  // The init must not depend on the numbers at all — zero weights see nothing.
  assert(decideRiichi({ ...FEATURES, ev: 0, pwin: 0, liveWaits: 0, oppRiichi: 3 }, INIT_RIICHI));
});

Deno.test("riichi head: mergeRiichi fills partials against the identity", () => {
  assertEquals(mergeRiichi(), INIT_RIICHI);
  assertEquals(mergeRiichi({}), INIT_RIICHI);
  const w = mergeRiichi({ oppRiichi: -2 });
  assertEquals(w.oppRiichi, -2);
  assertEquals(w.bias, 1);
  assertEquals(w.ev, 0);
});

Deno.test("riichi head: a hostile-enough bias declines", () => {
  assert(!decideRiichi(FEATURES, mergeRiichi({ bias: -1000 })));
  // …and a weighted head can be swung by a single feature (追っかけ aversion).
  const cautious = mergeRiichi({ oppRiichi: -1 });
  assert(decideRiichi({ ...FEATURES, oppRiichi: 0 }, cautious));
  assert(!decideRiichi({ ...FEATURES, oppRiichi: 2 }, cautious));
});

// ---------------------------------------------------------------------------
// init-equivalence over real hanchan
// ---------------------------------------------------------------------------

/** The shipped vector — exercises the head alongside the M11 hand block. */
const CHAMPION: KTune = loadKtune(
  new URL("../weights/champion.json", import.meta.url).pathname,
);

Deno.test("riichi head: `{}` (⇒ INIT) plays the identical hanchan — kkkk", () => {
  // "k" seats since the 2026-08-25 epoch: a vector no longer reaches an "h"
  // seat (frozen), and an hhhh version of this test would pass VACUOUSLY —
  // equal because both arms ignore the block, testing nothing.
  const plain = headless(GAMES, SEED, "kkkk", {});
  const inited = headless(GAMES, SEED, "kkkk", { ktune: { riichi: {} } });
  assertEquals(inited.results, plain.results);
});

Deno.test("riichi head: `{}` (⇒ INIT) plays the identical hanchan — khhh + champion", () => {
  // Since 2026-08-27 champion.json SHIPS a doctrine riichi block, so the
  // equivalence base strips it: headless-vs-`riichi:{}` is a claim about the
  // loader and the INIT weights, not about the shipped doctrine.
  const base = { ...CHAMPION, riichi: undefined };
  const plain = headless(GAMES, SEED, "khhh", { ktune: base });
  const inited = headless(GAMES, SEED, "khhh", {
    ktune: { ...base, riichi: {} },
  });
  assertEquals(inited.results, plain.results);
});

// ---------------------------------------------------------------------------
// the block is live
// ---------------------------------------------------------------------------

Deno.test("riichi head: a declare-averse head yields fewer riichi, different games", () => {
  const plain = headless(GAMES, SEED, "kkkk", {});
  const averse = headless(GAMES, SEED, "kkkk", {
    ktune: { riichi: { bias: -1000 } },
  });
  const riichis = (r: typeof plain) =>
    r.results.reduce(
      (n, m) => n + (m.riichis ?? [0, 0, 0, 0]).reduce((a, b) => a + b, 0),
      0,
    );
  const before = riichis(plain);
  const after = riichis(averse);
  // The head cannot forbid what stays outside it: `mustCure` declarations
  // survive, so "fewer", not "zero", is the claim — plus the games diverging
  // at all, which is what proves the block reached the seats.
  assert(before > 0, "baseline lane must contain riichi for the claim to bite");
  assert(after < before, `expected fewer riichi, got ${after} vs ${before}`);
});
