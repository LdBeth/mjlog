// Headless self-play smoke test: does the game master survive hundreds of
// rounds of arbitrary legal play, and do the conservation laws hold?
//
// Drives the real scorer, so the conservation checks cover actual payouts.

import { assert, assertEquals } from "@std/assert";
import { HeuristicPolicy } from "../src/ai/heuristic.ts";
import { RandomPolicy } from "../src/ai/random.ts";
import { dojoHooks } from "../src/dojo.ts";
import type { MatchResult } from "../src/match.ts";
import { runMatchSync, topFinisher } from "../src/match.ts";
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

Deno.test("self-play: 30 hanchan complete without throwing", () => {
  for (let seed = 1; seed <= 30; seed++) {
    const r = play(seed);
    assert(r.rounds.length > 0, `seed ${seed}: no rounds`);
    assert(r.rounds.length <= 64, `seed ${seed}: ${r.rounds.length} rounds`);
  }
});

/** 供託 still on the table when the last round ended, before `finalize` pays it out. */
function leftoverKyotaku(r: MatchResult): number {
  let k = 0;
  for (let i = 0; i < r.rounds.length; i++) {
    k = r.outcomes[i].kind === "agari" ? 0 : r.rounds[i].kyotaku;
  }
  return k;
}

Deno.test("self-play: points are conserved every round", () => {
  // Nothing is left over any more: `finalize` hands the last round's sticks to
  // the top finisher (the Tenhou convention), so the four scores account for
  // every point that went in.
  const TOTAL = JANKI.startScore * 4;
  for (let seed = 1; seed <= 25; seed++) {
    const r = play(seed);
    const sum = r.scores.reduce((a, b) => a + b, 0);
    assertEquals(sum, TOTAL, `seed ${seed}: ${sum} != ${TOTAL}`);
  }
});

Deno.test("finalize: leftover 供託 goes to the top finisher instead of evaporating", () => {
  // Seed 24 ends in an exhaustive draw with one riichi stick still on the table.
  // If the fixture ever stops leaving one, re-pick a seed with that property
  // rather than dropping the test — a match that ends on a draw is the only way
  // sticks can outlive the last round.
  const r = play(24);
  const sticks = leftoverKyotaku(r);
  assertEquals(r.outcomes[r.outcomes.length - 1].kind, "ryuukyoku");
  assert(sticks > 0, "fixture seed no longer leaves a 供託 stick");

  assertEquals(
    r.scores.reduce((a, b) => a + b, 0),
    JANKI.startScore * 4,
    "the sticks are back in the scores",
  );
  // Strip the payout off again: the seat that took it was already top, and no
  // other seat was touched.
  const top = topFinisher(r.scores);
  const before = r.scores.map((s, i) => (i === top ? s - sticks * 1000 : s));
  assertEquals(topFinisher(before), top, "the sticks landed on the top finisher");
  assertEquals(
    before.reduce((a, b) => a + b, 0),
    JANKI.startScore * 4 - sticks * 1000,
    "exactly the sticks were added, and only to one seat",
  );
});

Deno.test("finalize: the 供託 tiebreak is the settlement tiebreak (lower seat)", () => {
  // `settlement` (src/rl/record.ts) and the TUI's `finalStandings` both sort by
  // `b.s - a.s || a.seat - b.seat`, so the sticks must never land on a seat
  // those two rank second.
  assertEquals(topFinisher([25000, 30000, 30000, 15000]), 1);
  assertEquals(topFinisher([30000, 30000, 30000, 30000]), 0);
  assertEquals(topFinisher([1000, 2000, 3000, 94000]), 3);
});

Deno.test("self-play: hand sizes stay legal throughout", () => {
  for (let seed = 1; seed <= 10; seed++) {
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
  for (let seed = 1; seed <= 10; seed++) {
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
  for (let seed = 1; seed <= 10; seed++) {
    const r = play(seed);
    for (const round of r.rounds) {
      const draws = round.events.filter((e) => e.t === "draw").length;
      assert(draws <= 70, `seed ${seed} kyoku ${round.kyoku}: ${draws} draws`);
    }
  }
});

/**
 * The rules a compliance-filtered seat can still be charged for. Everything
 * else in `RULES` is previewable, and a policy that can preview a violation and
 * still commits it has a bug — this is the test that says so over real play
 * rather than over a fixture. See `penalty/preview.ts` for why each of these
 * cannot be vetoed: 持ち点8000点未満 fires on payments, 片和了り/後付け fire on
 * taking a win (declining is 見逃し), and 立直後カン見送り is the horn of a
 * dilemma the ledger creates for itself (`ankan-form` charges the kan too).
 */
const UNAVOIDABLE = new Set(["under-8000", "katagari", "atozuke", "riichi-kan-skip"]);

Deno.test("dojo filter: heuristic seats file no avoidable violation in 20 hanchan", () => {
  const seen = new Map<string, number>();
  for (let seed = 1; seed <= 20; seed++) {
    const r = runMatchSync(
      SEATS.map((s) => new HeuristicPolicy(`H${s}`, seed * 4 + s)),
      {
        seed,
        cfg: JANKI,
        dojo: DOJO_HEADLESS,
        scorer,
        ...dojoHooks({ dojo: DOJO_HEADLESS, oracle: scorer }),
      },
    );
    for (const v of r.ledger) seen.set(v.rule, (seen.get(v.rule) ?? 0) + 1);
    const stray = r.ledger.filter((v) => !UNAVOIDABLE.has(v.rule));
    assertEquals(
      stray.map((v) => `P${v.seat} ${v.rule}: ${v.detail}`),
      [],
      `seed ${seed}: an avoidable 禁じ手 reached the ledger`,
    );
  }
  // The filter must be doing work, not merely finding nothing to do: the same
  // 20 hanchan under the pre-filter policy filed 第一打字牌切り, 不聴時ドラ切り,
  // 裸単騎, ドラ切り後の手出し, 暗槓条件違反, 即引っかけ and 役満関連牌切り.
  assert(seen.size > 0, "the ledger cannot be empty — 持ち点8000点未満 always fires");
});
