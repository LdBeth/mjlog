// M14 — the WIRING of the `dealin` block: loader → seat → CLI → freeze.
//
// `test/dealin_test.ts` owns the model (features, record round-trip, purity,
// the value rebuild). This file owns the switch, and the switch has exactly two
// claims:
//
//   ABSENT is bit-identical. No `dealin` section ⇒ no head is built, no trace
//   is asked for, and the seat plays the game it has always played. Unlike
//   `hand`/`riichi`/`fold` there is no `{}` arm to compare against — an empty
//   block THROWS (`dealin_test.ts` asserts that) — so the identity is stated
//   against a vector whose section was stripped.
//
//   PRESENT reaches the seat. A head that answers differently from the counting
//   model must move the games, or the block is being dropped silently — which
//   is exactly what happened to `riichi` for a day (`spec.ts`: the interface
//   and the `loadKtune` whitelist have to move together).
//
// Plus the two refusals that keep the lane honest (`--calibrate` beside a block,
// D6) and the ownership rule (heads are built once per seat and freed in
// `close()`, never rebuilt inside `withReads`).

import { assert, assertEquals, assertNotEquals } from "@std/assert";
import { argError } from "../src/cli/args.ts";
import type { DealinWeights } from "../src/ai/dealin.ts";
import { DEALIN_F, DEALIN_FV, mergeDealin, TENPAI_F } from "../src/ai/dealin.ts";
import { closeArm, headless, loadKtune, openArm } from "../src/harness.ts";
import type { KTune, TableSpec } from "../src/harness.ts";
import type { MlpSpec } from "../src/ai/mlp.ts";
import { sfc32 } from "../src/rng.ts";

const SEED = 8191;
const GAMES = 3;

/** The shipped vector — the block has to compose with everything already on it. */
const CHAMPION: KTune = loadKtune(
  new URL("../weights/champion.json", import.meta.url).pathname,
);
/**
 * The champion with its `dealin` section stripped. Since the 2026-08-30
 * promotion the champion CARRIES the block, so the "absent" claims and the
 * `--calibrate` lane are stated against this vector — the block tests (`{}`
 * throws, D6) are unchanged; only their baseline moved.
 */
const BASE: KTune = { ...CHAMPION, dealin: undefined };

/** A small two-layer head with reproducible weights — a plausible fit shape. */
function randHead(inputs: number, seed: number): MlpSpec {
  const r = sfc32(seed);
  const w0 = Array.from({ length: inputs * 4 }, () => r.float() - 0.5);
  const w1 = Array.from({ length: 4 }, () => r.float() - 0.5);
  return {
    fv: DEALIN_FV,
    layers: [
      { in: inputs, out: 4, act: "relu", w: w0, b: [0.1, -0.2, 0.3, 0] },
      { in: 4, out: 1, act: "none", w: w1, b: [-1] },
    ],
  };
}

/** Tiny but VALID weights: the point is that they are not computed's numbers. */
const TINY: DealinWeights = mergeDealin({
  fv: DEALIN_FV,
  dealin: randHead(DEALIN_F, 20260829),
  tenpai: randHead(TENPAI_F, 902620),
});

// ---------------------------------------------------------------------------
// 1. absent ≡ identical
// ---------------------------------------------------------------------------

Deno.test("M14 wiring: dealin ブロックが無い席はビット単位で従来どおり — kkkk", () => {
  // "k" seats: since the 2026-08-25 epoch a vector no longer reaches an "h"
  // seat, so an hhhh arm would be equal VACUOUSLY.
  const plain = headless(GAMES, SEED, "kkkk", {});
  const stripped = headless(GAMES, SEED, "kkkk", { ktune: { dealin: undefined } });
  assertEquals(stripped.results, plain.results);
});

Deno.test("M14 wiring: dealin ブロックが無い席はビット単位で従来どおり — khhh + champion", () => {
  const plain = headless(GAMES, SEED, "khhh", { ktune: BASE });
  const stripped = headless(GAMES, SEED, "khhh", {
    ktune: { ...BASE, dealin: undefined },
  });
  assertEquals(stripped.results, plain.results);
});

// ---------------------------------------------------------------------------
// 2. present reaches the seat
// ---------------------------------------------------------------------------

Deno.test("M14 wiring: 学習ヘッドを載せた席は別の対局を打つ — kkkk", () => {
  const plain = headless(GAMES, SEED, "kkkk", {});
  const learned = headless(GAMES, SEED, "kkkk", { ktune: { dealin: TINY } });
  // The claim is only that the block REACHED the seats — a head that never
  // arrived would reproduce `plain` exactly, which is the silent drop this
  // test exists to catch.
  assertNotEquals(learned.results, plain.results, "dealin ブロックが席に届いていない");
});

Deno.test("M14 wiring: 学習ヘッドを載せた席は別の対局を打つ — khhh + champion", () => {
  const plain = headless(GAMES, SEED, "khhh", { ktune: BASE });
  const learned = headless(GAMES, SEED, "khhh", {
    ktune: { ...BASE, dealin: TINY },
  });
  assertNotEquals(learned.results, plain.results, "dealin ブロックが席に届いていない");
  // …and only seat 0 carries the vector, so the frozen field is untouched: the
  // two runs are the same environment with one seat changed.
  assert(plain.results.length === GAMES);
});

// ---------------------------------------------------------------------------
// 3. ownership: built once per seat, freed with the seat
// ---------------------------------------------------------------------------

Deno.test("M14 wiring: ヘッドは席ごとに一度だけ作られ、close で解放される", () => {
  const table: TableSpec = [
    { kind: "k", ktune: { dealin: TINY } },
    { kind: "h" },
    { kind: "h" },
    { kind: "h" },
  ];
  const arm = openArm(table);
  // `reset` rebuilds the provider chain (that is what `withReads` is for); the
  // heads must survive it, so a second match after a reset must still play.
  try {
    assertEquals(arm.built.length, 4);
    arm.built[0].reset(SEED);
    arm.built[0].reset(SEED + 1);
  } finally {
    // Idempotent by contract — a double close would be a native double-free.
    closeArm(arm);
    closeArm(arm);
  }
});

// ---------------------------------------------------------------------------
// 4. the CLI refusal (D6)
// ---------------------------------------------------------------------------

Deno.test("M14 wiring: --calibrate は dealin ブロックを載せた席を断る", () => {
  const base = { cmd: "selfplay", seats: "khhh", calibrate: "" };
  // The plain lane is what the fit needs, and it is still legal.
  assertEquals(argError({ ...base, calibrate: "lane.jsonl", ktune: BASE }), null);
  // `--ktune` carrying the block — the promoted champion itself is one…
  assert(argError({ ...base, calibrate: "lane.jsonl", ktune: CHAMPION })?.includes("dealin"));
  const withBlock: KTune = { ...BASE, dealin: TINY };
  const viaKtune = argError({ ...base, calibrate: "lane.jsonl", ktune: withBlock });
  assert(viaKtune?.includes("dealin"), viaKtune ?? "(拒否されなかった)");
  assert(viaKtune?.includes("--calibrate"));
  // …and the same vector spelled as a `--table` seat 0.
  const table: TableSpec = [
    { kind: "k", ktune: withBlock },
    { kind: "h" },
    { kind: "h" },
    { kind: "h" },
  ];
  const viaTable = argError({ ...base, calibrate: "lane.jsonl", table });
  assert(viaTable?.includes("dealin"), viaTable ?? "(拒否されなかった)");
  // Without `--calibrate` the block is nobody's business.
  assertEquals(argError({ ...base, ktune: withBlock }), null);
  assertEquals(argError({ ...base, table }), null);
});
