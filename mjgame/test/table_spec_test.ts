// The modular seat layer (`spec.ts`): SeatSpec/TableSpec as the construction
// unit, `resolveTable` as the ONE home of the legacy conventions, `--table` as
// the explicit form.
//
// Three claims. The legacy path is DEFINED as the resolved path (structural
// equivalence, checked by introspecting the built seats — the pinned suites
// already hold it behaviorally). Four copies of one component with four
// different weight files really are four different players (the acceptance
// test for the modular redesign). And `paired`'s explicit control arm REFUSES
// an environment that differs from arm A's — the structural fix for the
// confound that mis-crowned the M11 champion.

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { argError } from "../src/cli/args.ts";
import { closeArm, headless, headlessParallel, openArm } from "../src/harness.ts";
import { pairedRun } from "../src/paired.ts";
import { kindString, loadKtune, loadTable, resolveTable, sameEnvironment } from "../src/spec.ts";
import type { KTune, TableSpec } from "../src/spec.ts";

const SEED = 8191;
const GAMES = 3;

/** A vector distinguishable by introspection: `shanten` is 1000 by default. */
const vec = (shanten: number): KTune => ({ heuristic: { shanten } });

// deno-lint-ignore no-explicit-any
const weightsOf = (arm: ReturnType<typeof openArm>, seat: number): any =>
  // deno-lint-ignore no-explicit-any
  (arm.built[seat].policy as any).w;

// ---------------------------------------------------------------------------
// resolveTable — the legacy conventions, reproduced structurally
// ---------------------------------------------------------------------------

Deno.test("resolveTable: a string arm and its resolved table build the same seats", () => {
  // All-k, because only a "k" seat reads the `heuristic` section (an "h"
  // opponent takes just the hand/riichi blocks) — the introspection needs a
  // kind that shows the vector.
  const opts = { ktune: vec(1111), ktuneOpp: vec(2222), standings: true };
  const a = openArm("kkkk", opts);
  const b = openArm(resolveTable("kkkk", opts), opts);
  try {
    assertEquals(b.seats, a.seats);
    for (let s = 0; s < 4; s++) {
      assertEquals(weightsOf(b, s), weightsOf(a, s));
    }
    // The gates themselves: subject carries its vector, opponents the opp one.
    assertEquals(weightsOf(a, 0).shanten, 1111);
    assertEquals(weightsOf(a, 1).shanten, 2222);
    assertEquals(weightsOf(a, 3).shanten, 2222);
  } finally {
    closeArm(a);
    closeArm(b);
  }
});

Deno.test("resolveTable: the two forms play identical hanchan", () => {
  const opts = { ktune: vec(1111), ktuneOpp: vec(2222) };
  const byString = headless(GAMES, SEED, "khhh", opts);
  const byTable = headless(GAMES, SEED, resolveTable("khhh", opts), opts);
  assertEquals(byTable.results, byString.results);
});

// ---------------------------------------------------------------------------
// the acceptance test: multiple copies, different components and weights
// ---------------------------------------------------------------------------

Deno.test("table: four copies of one seat kind carry four different weight sets", () => {
  const table: TableSpec = [
    { kind: "k", ktune: vec(1001), plan: true },
    { kind: "k", ktune: vec(1002) },
    { kind: "k", ktune: { ...vec(1003), hand: {} } },
    { kind: "k", ktune: { ...vec(1004), riichi: {} } },
  ];
  const arm = openArm(table);
  try {
    assertEquals(arm.seats, "kkkk");
    assertEquals(kindString(table), "kkkk");
    for (let s = 0; s < 4; s++) assertEquals(weightsOf(arm, s).shanten, 1001 + s);
    // Component blocks land on THEIR seat and no other.
    // deno-lint-ignore no-explicit-any
    const p = (s: number) => arm.built[s].policy as any;
    assertEquals([0, 1, 2, 3].map((s) => p(s).hand !== null), [false, false, true, false]);
    assertEquals([0, 1, 2, 3].map((s) => p(s).riichiHead !== null), [false, false, false, true]);
  } finally {
    closeArm(arm);
  }
});

Deno.test("table: different per-seat weights are a different game", () => {
  const uniform: TableSpec = resolveTable("kkkk", { ktune: vec(1001) });
  const varied: TableSpec = [
    { kind: "k", ktune: vec(1001) },
    { kind: "k", ktune: vec(3000) },
    { kind: "k", ktune: vec(1001) },
    { kind: "k", ktune: vec(1001) },
  ];
  const a = headless(GAMES, SEED, uniform);
  const b = headless(GAMES, SEED, varied);
  assertNotEquals(a.results, b.results);
});

Deno.test("table: --jobs shards a TableSpec run byte-identically", async () => {
  const table: TableSpec = [
    { kind: "k", ktune: vec(1001) },
    { kind: "k", ktune: { hand: {} } },
    { kind: "h" },
    { kind: "h" },
  ];
  const one = headless(4, SEED, table);
  const par = await headlessParallel(4, SEED, table, 2);
  assertEquals(par.results, one.results);
});

// ---------------------------------------------------------------------------
// paired --table-b: the environment guard
// ---------------------------------------------------------------------------

Deno.test("paired tableB: an identical environment is accepted, seat 0 varies", () => {
  const tA: TableSpec = [{ kind: "k", ktune: vec(1111) }, { kind: "h" }, { kind: "h" }, {
    kind: "h",
  }];
  const tB: TableSpec = [{ kind: "k", ktune: vec(2222) }, { kind: "h" }, { kind: "h" }, {
    kind: "h",
  }];
  const st = pairedRun(2, SEED, tA, { tableB: tB });
  assertEquals(st.seats, "khhh");
  assertEquals(st.games, 2);
});

Deno.test("paired tableB: a differing environment is refused", () => {
  const tA: TableSpec = [{ kind: "k" }, { kind: "h" }, { kind: "h" }, { kind: "h" }];
  const tB: TableSpec = [{ kind: "k" }, { kind: "h", ktune: { hand: {} } }, { kind: "h" }, {
    kind: "h",
  }];
  assertThrows(
    () => pairedRun(1, SEED, tA, { tableB: tB }),
    Error,
    "環境",
  );
});

Deno.test("sameEnvironment: equality is of values, not key order", () => {
  const a: TableSpec = [
    { kind: "k" },
    {
      kind: "k",
      ktune: { hand: {}, heuristic: { shanten: 5 } },
    },
    { kind: "h" },
    { kind: "h" },
  ];
  const b: TableSpec = [
    { kind: "h" },
    {
      ktune: { heuristic: { shanten: 5 }, hand: {} },
      kind: "k",
    } as TableSpec[number],
    { kind: "h" },
    { kind: "h" },
  ];
  assert(sameEnvironment(a, b));
});

// ---------------------------------------------------------------------------
// the file formats
// ---------------------------------------------------------------------------

Deno.test("loadTable: kinds, inline vectors and table-relative paths", () => {
  const dir = Deno.makeTempDirSync();
  Deno.writeTextFileSync(`${dir}/v.json`, JSON.stringify(vec(4321)));
  Deno.writeTextFileSync(
    `${dir}/table.json`,
    JSON.stringify({
      seats: [
        { kind: "k", ktune: "v.json", plan: true },
        // An "h" seat takes NOTHING (frozen, 2026-08-25 epoch) — component
        // blocks belong on "k" seats.
        { kind: "k", ktune: { hand: {} } },
        { kind: "n", weights: "manifest.json", temp: 1 },
        { kind: "h" },
      ],
    }),
  );
  const table = loadTable(`${dir}/table.json`);
  assertEquals(kindString(table), "kknh");
  assertEquals(table[0].ktune?.heuristic?.shanten, 4321);
  assertEquals(table[0].plan, true);
  assertEquals(table[1].ktune?.hand, {});
  // n-seat paths resolve relative to the table file too.
  assertEquals(table[2].weights, `${dir}/manifest.json`);
  assertEquals(table[2].temp, 1);
});

Deno.test("loadKtune: the riichi section survives the load", () => {
  // Regression: the section whitelist had dropped `riichi` for a day of M12's
  // life — a file-carried head silently never reached the seat.
  const dir = Deno.makeTempDirSync();
  Deno.writeTextFileSync(`${dir}/r.json`, JSON.stringify({ riichi: { bias: -5 } }));
  assertEquals(loadKtune(`${dir}/r.json`).riichi, { bias: -5 });
});

// ---------------------------------------------------------------------------
// the CLI rules
// ---------------------------------------------------------------------------

Deno.test("argError: --table subsumes the per-seat flags", () => {
  const base = { cmd: "selfplay", seats: "khhh", calibrate: "", tablePath: "t.json" };
  assertEquals(argError({ ...base }), null);
  assert(argError({ ...base, cmd: "play" }));
  assert(argError({ ...base, seatsGiven: true }));
  assert(argError({ ...base, ktunePath: "k.json" }));
  assert(argError({ ...base, ktuneOppPath: "o.json" }));
  assert(argError({ ...base, consumerPath: "c.json" }));
  assert(argError({ ...base, standings: true }));
  assert(argError({ ...base, plan: true }));
  assert(argError({ ...base, curriculum: 0.5 }));
  assert(argError({ ...base, weightsGiven: true }));
  assert(argError({ ...base, tempGiven: true }));
});

Deno.test("argError: --table-b needs --table and owns the control arm", () => {
  const base = { cmd: "paired", seats: "khhh", calibrate: "" };
  assert(argError({ ...base, tableBPath: "b.json" }));
  assertEquals(argError({ ...base, tablePath: "t.json", tableBPath: "b.json" }), null);
  assert(argError({ ...base, tablePath: "t.json", tableBPath: "b.json", ktuneBPath: "k.json" }));
  assert(
    argError({ ...base, tablePath: "t.json", tableBPath: "b.json", consumerBPath: "c.json" }),
  );
  // A table-based paired run must state BOTH arms: the hhhh fallback carries
  // no environment guard, and an incumbent flag against a table would measure
  // A against A and print 0.000.
  assert(argError({ ...base, tablePath: "t.json" }));
  assert(argError({ ...base, tablePath: "t.json", ktuneBPath: "k.json" }));
  assert(argError({ ...base, tablePath: "t.json", consumerBPath: "c.json" }));
  // selfplay/bench have no control arm; a lone --table stays legal there.
  assertEquals(argError({ ...base, cmd: "selfplay", tablePath: "t.json" }), null);
});
