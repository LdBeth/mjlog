// The `play` driver's dojo wiring.
//
// `cmdPlay` cannot be driven from a test — it refuses to run off a tty and its
// decisions come from the keyboard — so what is tested here is the piece it
// shares with self-play: the hooks from `makeDojoHooks`, spread into the same
// option bag `runMatch` gets. Without them the penalty rules never run: the
// 違反台帳 panel stays empty for a whole hanchan and `Observation.tsumogiriLock`
// is stuck false, because that flag is armed by a rule and not by the engine.

import { assert, assertEquals } from "@std/assert";
import { HeuristicPolicy } from "../src/ai/heuristic.ts";
import { RandomPolicy } from "../src/ai/random.ts";
import { cpuKindAt, makeDojoHooks } from "../src/main.ts";
import { runMatchSync } from "../src/match.ts";
import { observe } from "../src/observe.ts";
import { DOJO_DEFAULT, JANKI } from "../src/rules.ts";
import { scorer } from "../src/score.ts";
import type { Table } from "../src/table.ts";
import type { PublicEvent, Seat, Violation } from "../src/types.ts";
import { SEATS } from "../src/types.ts";

interface Run {
  ledger: Violation[];
  /** Violations as the TUI receives them — `App.onEvent` fills the panel here. */
  announced: Violation[];
  tables: Table[];
}

/** A hanchan run exactly the way `cmdPlay` runs one, minus the terminal. */
function play(seed: number, wired: boolean): Run {
  const announced: Violation[] = [];
  const tables: Table[] = [];
  const hooks = wired ? makeDojoHooks(DOJO_DEFAULT) : {};
  const r = runMatchSync(
    SEATS.map((s) => new RandomPolicy(`R${s}`, seed * 4 + s)),
    {
      seed,
      cfg: JANKI,
      dojo: DOJO_DEFAULT,
      scorer,
      sink: (e: PublicEvent) => {
        if (e.e === "violation") announced.push(e.v);
      },
      onTable: (t: Table) => tables.push(t),
      ...hooks,
    },
  );
  return { ledger: r.ledger, announced, tables };
}

Deno.test("play wiring: dojo hooks put violations in the ledger AND on the sink", () => {
  let total = 0;
  const seats = new Set<Seat>();
  for (let seed = 1; seed <= 6; seed++) {
    const r = play(seed, true);
    total += r.ledger.length;
    // The panel is fed by the sink, the final standings by the ledger: a
    // violation that reaches only one of the two would be invisible somewhere.
    assertEquals(
      r.announced.map((v) => `${v.seat}:${v.rule}`),
      r.ledger.map((v) => `${v.seat}:${v.rule}`),
      `seed ${seed}: ledger and 違反台帳 disagree`,
    );
    for (const v of r.ledger) seats.add(v.seat);
  }
  assert(total > 0, "arbitrary legal play should trip some 禁じ手");
  // Seat 0 may be the human in `play` (the seat is drawn from the seed); its
  // violations must reach the panel too.
  assert(seats.has(0), "no violation was ever filed against seat 0");
});

Deno.test("play wiring: without the hooks the ledger is inert", () => {
  for (let seed = 1; seed <= 6; seed++) {
    const r = play(seed, false);
    assertEquals(r.ledger.length, 0);
    assertEquals(r.announced.length, 0);
    for (const t of r.tables) {
      assertEquals(t.tsumogiriLock, [false, false, false, false], `seed ${seed}`);
    }
  }
});

Deno.test("play wiring: tsumogiriLock is armed, and an Observation sees it", () => {
  let armed = 0;
  for (let seed = 1; seed <= 6; seed++) {
    for (const t of play(seed, true).tables) {
      for (const seat of SEATS) {
        if (!t.tsumogiriLock[seat]) continue;
        armed++;
        // The flag only matters if it survives the trip into an Observation —
        // that is the one view a policy gets.
        assert(observe(t, seat, [], null, scorer).tsumogiriLock, "lock lost in observe()");
      }
    }
  }
  assert(armed > 0, "ドラ切り後の手出し never armed the lock in 6 hanchan");
});

// ---------------------------------------------------------------------------
// The result event carries the 手牌公開 snapshot the TUI overlay reveals.

Deno.test("play wiring: every result event snapshots all four hands", () => {
  let rounds = 0;
  for (let seed = 1; seed <= 4; seed++) {
    const results: Array<Extract<PublicEvent, { e: "result" }>> = [];
    runMatchSync(
      SEATS.map((s) => new RandomPolicy(`R${s}`, seed * 4 + s)),
      {
        seed,
        cfg: JANKI,
        dojo: DOJO_DEFAULT,
        scorer,
        sink: (e: PublicEvent) => {
          if (e.e === "result") results.push(e);
        },
        ...makeDojoHooks(DOJO_DEFAULT),
      },
    );
    assert(results.length > 0, `seed ${seed}: a hanchan has rounds`);
    for (const r of results) {
      rounds++;
      assertEquals(r.hands?.length, 4, `seed ${seed}: four hands`);
      assertEquals(r.melds?.length, 4, `seed ${seed}: four meld sets`);
      for (const s of SEATS) {
        const hand = r.hands![s];
        const melds = r.melds![s];
        assert(hand.length > 0, `seed ${seed}: P${s} holds nothing`);
        assert(hand.length <= 14, `seed ${seed}: P${s} holds ${hand.length} tiles`);
        // 13 concealed minus three per meld, plus the winner's drawn 14th.
        assertEquals(
          hand.length + 3 * melds.length <= 14,
          true,
          `seed ${seed}: P${s} hand ${hand.length} + ${melds.length} melds`,
        );
        // A snapshot, not a live alias: sorted, and detached from the table.
        assertEquals(
          hand,
          [...hand].sort((a, b) => a - b),
          `seed ${seed}: P${s}'s revealed hand is unsorted`,
        );
      }
    }
  }
  assert(rounds >= 4, "the snapshot should be checked over real rounds");
});

// ---------------------------------------------------------------------------
// Per-seat round counters: リーチ率 / 副露率 in the self-play stats read these.

Deno.test("match result: riichis / furoRounds count rounds, and 暗槓 is not 副露", () => {
  let riichiSeen = 0;
  let furoSeen = 0;
  let ankanOnlySeen = 0;
  // Both CPUs: the random one calls constantly (and kans freely, which is what
  // exercises the 暗槓 exclusion), the heuristic one is the only one that ever
  // reaches menzen tenpai and so the only one that ever declares riichi.
  const cpus = {
    r: (s: Seat, seed: number) => new RandomPolicy(`R${s}`, seed * 4 + s),
    h: (s: Seat, seed: number) => new HeuristicPolicy(`H${s}`, seed * 4 + s),
  };
  for (const [kind, make] of Object.entries(cpus)) {
    // The heuristic thinks (shanten + ukeire per decision), so fewer hanchan of
    // it: 4 is already ~30 riichi.
    for (let seed = 1; seed <= (kind === "h" ? 4 : 8); seed++) {
      const tag = `${kind} seed ${seed}`;
      // The result events carry each seat's melds at the moment every round
      // ended — the instant the counters are snapshotted, so they must agree.
      const snaps: Array<Extract<PublicEvent, { e: "result" }>> = [];
      const r = runMatchSync(SEATS.map((s) => make(s, seed)), {
        seed,
        cfg: JANKI,
        dojo: DOJO_DEFAULT,
        scorer,
        sink: (e: PublicEvent) => {
          if (e.e === "result") snaps.push(e);
        },
        ...makeDojoHooks(DOJO_DEFAULT),
      });
      const n = r.rounds.length;
      assertEquals(r.riichis?.length, 4, `${tag}: riichis is per seat`);
      assertEquals(r.furoRounds?.length, 4, `${tag}: furoRounds is per seat`);
      assertEquals(snaps.length, n, `${tag}: one result event per round`);

      for (const s of SEATS) {
        for (
          const [name, v] of [["riichis", r.riichis![s]], ["furoRounds", r.furoRounds![s]]] as const
        ) {
          assert(Number.isInteger(v), `${tag}: ${name}[${s}] = ${v}`);
          assert(v >= 0 && v <= n, `${tag}: ${name}[${s}] = ${v} out of 0..${n}`);
        }
        // Cross-check 副露 against the independent melds snapshots: a seat that
        // opened its hand in `k` rounds must have furoRounds === k, and rounds
        // where its only melds were 暗槓 must not be among them.
        const open = snaps.filter((e) =>
          e.melds![s].some((m) => m.kind !== "ankan" && m.kind !== "nuki")
        ).length;
        assertEquals(
          r.furoRounds![s],
          open,
          `${tag}: P${s} furoRounds ${r.furoRounds![s]} vs ${open} open-meld rounds`,
        );
        ankanOnlySeen += snaps.filter((e) =>
          e.melds![s].length > 0 && e.melds![s].every((m) =>
            m.kind === "ankan"
          )
        ).length;
        riichiSeen += r.riichis![s];
        furoSeen += r.furoRounds![s];
      }
    }
  }
  assert(riichiSeen > 0, "12 hanchan should include some riichi");
  assert(furoSeen > 0, "12 hanchan should include some 副露");
  assert(ankanOnlySeen > 0, "no 暗槓-only round appeared; the exclusion is untested");
});

Deno.test("cpuKindAt: play deals --seats kinds past the human seat in order", () => {
  // human at seat 0: CPU seats 1,2,3 get n,h,h (first three chars of --seats)
  assertEquals([1, 2, 3].map((s) => cpuKindAt("nhh", 0, s as Seat)), ["n", "h", "h"]);
  // human at seat 2: CPU seats 0,1,3 get n,h,h — the neural CPU stays exactly one
  assertEquals([0, 1, 3].map((s) => cpuKindAt("nhh", 2, s as Seat)), ["n", "h", "h"]);
  // human at seat 3, mixed kinds keep their order
  assertEquals([0, 1, 2].map((s) => cpuKindAt("rnh", 3, s as Seat)), ["r", "n", "h"]);
  // every human seat: "nhh" yields exactly one "n" across the three CPU seats
  for (let h = 0; h < 4; h++) {
    const kinds = [0, 1, 2, 3].filter((s) => s !== h)
      .map((s) => cpuKindAt("nhh", h as Seat, s as Seat));
    assertEquals(kinds.filter((k) => k === "n").length, 1);
  }
});
