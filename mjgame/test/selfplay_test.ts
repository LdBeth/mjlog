// Headless self-play smoke test: does the game master survive thousands of
// rounds of arbitrary legal play, and do the conservation laws hold?
//
// Drives the real scorer, so the conservation checks cover actual payouts.

import { assert, assertEquals } from "@std/assert";
import { RandomPolicy } from "../src/ai/random.ts";
import { dojoHooks } from "../src/dojo.ts";
import { runMatchSync } from "../src/match.ts";
import { DOJO_DEFAULT, DOJO_HEADLESS, JANKI } from "../src/rules.ts";
import { scorer } from "../src/score.ts";
import type { PublicEvent } from "../src/types.ts";
import { SEATS } from "../src/types.ts";

function policies(seed: number) {
  return SEATS.map((s) => new RandomPolicy(`R${s}`, seed * 4 + s));
}

function play(seed: number, sink?: (e: PublicEvent) => void) {
  return runMatchSync(policies(seed), {
    seed,
    cfg: JANKI,
    dojo: DOJO_HEADLESS,
    scorer,
    sink,
  });
}

Deno.test("self-play: 100 hanchan complete without throwing", () => {
  for (let seed = 1; seed <= 100; seed++) {
    const r = play(seed);
    assert(r.rounds.length > 0, `seed ${seed}: no rounds`);
    assert(r.rounds.length <= 64, `seed ${seed}: ${r.rounds.length} rounds`);
  }
});

Deno.test("self-play: points are conserved every round", () => {
  const TOTAL = JANKI.startScore * 4;
  for (let seed = 1; seed <= 50; seed++) {
    const r = play(seed);
    let kyotaku = 0;
    for (let i = 0; i < r.rounds.length; i++) {
      kyotaku = r.rounds[i].kyotaku;
      const o = r.outcomes[i];
      if (o.kind === "agari") kyotaku = 0;
    }
    const sum = r.scores.reduce((a, b) => a + b, 0);
    assertEquals(
      sum + kyotaku * 1000,
      TOTAL,
      `seed ${seed}: ${sum} + ${kyotaku} sticks != ${TOTAL}`,
    );
  }
});

Deno.test("self-play: hand sizes stay legal throughout", () => {
  for (let seed = 1; seed <= 20; seed++) {
    let hands: number[] = [];
    const melds = [0, 0, 0, 0];
    play(seed, (e) => {
      if (e.e === "deal") {
        hands = [13, 13, 13, 13];
        melds.fill(0);
      } else if (e.e === "draw") hands[e.who]++;
      else if (e.e === "discard") hands[e.who]--;
      else if (e.e === "call") {
        const m = e.meld;
        const consumed = m.kind === "shouminkan"
          ? 1
          : m.tiles.length - (m.fromWho === m.who ? 0 : 1);
        hands[m.who] -= consumed;
        if (m.kind !== "shouminkan") melds[m.who]++;
      }
      for (const s of SEATS) {
        const total = hands[s] + 3 * melds[s];
        assert(
          total === 13 || total === 14,
          `seed ${seed}: seat ${s} holds ${hands[s]} + 3*${melds[s]} = ${total}`,
        );
      }
    });
  }
});

Deno.test("self-play: same seed reproduces the same game exactly", () => {
  const a = play(7);
  const b = play(7);
  assertEquals(a.scores, b.scores);
  assertEquals(
    JSON.stringify(a.rounds.map((r) => r.events)),
    JSON.stringify(b.rounds.map((r) => r.events)),
  );
  const c = play(8);
  assert(
    JSON.stringify(a.rounds[0].events) !== JSON.stringify(c.rounds[0].events),
    "different seeds should diverge",
  );
});

Deno.test("dojo ledger: rules fire in live play without disturbing it", () => {
  const hooks = dojoHooks({ dojo: DOJO_DEFAULT, oracle: scorer });
  const withLedger = (seed: number) =>
    runMatchSync(policies(seed), {
      seed,
      cfg: JANKI,
      dojo: DOJO_DEFAULT,
      scorer,
      ...hooks,
    });

  const seen = new Set<string>();
  let total = 0;
  for (let seed = 1; seed <= 20; seed++) {
    const withRules = withLedger(seed);
    total += withRules.ledger.length;
    for (const v of withRules.ledger) {
      seen.add(v.rule);
      assert(v.points >= 0, `${v.rule}: negative penalty`);
      assert(v.confidence > 0 || v.points === 0, `${v.rule}: scored with no confidence`);
      assert(v.detail.length > 0, `${v.rule}: no evidence recorded`);
    }
    // Ledger rules must be observers only: the game plays out identically.
    const plain = play(seed);
    assertEquals(
      JSON.stringify(withRules.rounds.map((r) => r.events)),
      JSON.stringify(plain.rounds.map((r) => r.events)),
      `seed ${seed}: penalty rules changed the course of play`,
    );
  }
  assert(total > 0, "random play should trip some 禁じ手");
  // Random policies cut honors early and kan freely, so these are the ones we
  // can actually count on seeing.
  for (const id of ["first-honor", "minkan"]) {
    assert(seen.has(id), `expected rule ${id} to fire; saw [${[...seen].join(",")}]`);
  }
});

Deno.test("self-play: every round ends within the 70-draw budget", () => {
  for (let seed = 1; seed <= 30; seed++) {
    const r = play(seed);
    for (const round of r.rounds) {
      const draws = round.events.filter((e) => e.t === "draw").length;
      assert(draws <= 70, `seed ${seed} kyoku ${round.kyoku}: ${draws} draws`);
    }
  }
});
