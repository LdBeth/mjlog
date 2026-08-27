// `--ktune-opp`: the opponents (seats 1–3) carry their own 感性 vector.
//
// Three claims, and the first is the one that protects every pinned test in the
// suite: WITHOUT the option nothing moves by a bit. The second is that with it
// the opponents really are different players — otherwise the flag would be a
// no-op nobody notices until a transfer measurement comes back flat. The third
// is that the vector survives the worker boundary, so `--jobs=N` still shards a
// run into the SAME four seats.
//
// `kkkk` throughout: since the 2026-08-25 epoch a vector reaches ONLY "k"
// seats — the "h" seat is the frozen baseline and takes nothing — so both the
// subject and the opponents must be "k" for the routing claims to be visible.

import { assert, assertEquals } from "@std/assert";
import { argError, parseArgs } from "../src/cli/args.ts";
import { headless, headlessParallel, loadKtune } from "../src/harness.ts";
import type { KTune } from "../src/harness.ts";
import { pairedRun } from "../src/paired.ts";

const SEED = 8191;
const GAMES = 4;
const SEATS = "kkkk";

/** The shipped vector: calibrated 計算 scalars AND an M11 `hand` block. */
const CHAMPION: KTune = loadKtune(
  new URL("../weights/champion.json", import.meta.url).pathname,
);
/** No sections at all — every seat built from it is the untuned default. */
const EMPTY: KTune = {};

// ---------------------------------------------------------------------------
// (a) identity: opponents given the SAME vector change nothing
// ---------------------------------------------------------------------------

Deno.test("--ktune-opp: the same vector on both sides is the run without the flag", () => {
  const plain = headless(GAMES, SEED, SEATS, { ktune: CHAMPION });
  const echoed = headless(GAMES, SEED, SEATS, { ktune: CHAMPION, ktuneOpp: CHAMPION });
  // Scores, ledgers, outcomes, rounds — `MatchResult` is the whole record of a
  // hanchan, so deep equality here is equality of everything the run produced.
  assertEquals(echoed.results, plain.results);
});

Deno.test("--ktune-opp: absent, the option is not merely equal but unreachable", () => {
  // The default path must not depend on the flag's own machinery: an undefined
  // `ktuneOpp` has to collapse to `opts.ktune` for all four seats, which is what
  // keeps computed_test / calibration_test / champion_test pinned.
  const off = headless(GAMES, SEED, SEATS, { ktune: CHAMPION, ktuneOpp: undefined });
  assertEquals(off.results, headless(GAMES, SEED, SEATS, { ktune: CHAMPION }).results);
});

// ---------------------------------------------------------------------------
// (b) live: a different opponent vector is a different table
// ---------------------------------------------------------------------------

Deno.test("--ktune-opp: a different vector builds a different opponent population", () => {
  const shared = headless(GAMES, SEED, SEATS, { ktune: CHAMPION });
  const split = headless(GAMES, SEED, SEATS, { ktune: CHAMPION, ktuneOpp: EMPTY });
  // Same walls, same seat 0 file — only seats 1–3 lost the champion's `hand`
  // block, and that is enough to be a different game. No exact values pinned:
  // the claim is that the vector REACHES the opponents, not what it makes them
  // do.
  const moved = split.results.some((r, i) =>
    JSON.stringify(r.scores) !== JSON.stringify(shared.results[i].scores)
  );
  assert(moved, "an opponents-only vector left every hanchan untouched");
});

Deno.test("--ktune-opp: the subject keeps its own vector (opp does not overwrite seat 0)", () => {
  // If the gate leaked and seat 0 took the opp file, these two would agree:
  // both would be four seats built from EMPTY.
  const subject = headless(GAMES, SEED, SEATS, { ktune: CHAMPION, ktuneOpp: EMPTY });
  const nobody = headless(GAMES, SEED, SEATS, { ktune: EMPTY, ktuneOpp: EMPTY });
  const differs = subject.results.some((r, i) =>
    JSON.stringify(r.scores) !== JSON.stringify(nobody.results[i].scores)
  );
  assert(differs, "seat 0 appears to have been built from --ktune-opp");
});

// ---------------------------------------------------------------------------
// (c) the worker boundary: `--jobs` shards the same four seats
// ---------------------------------------------------------------------------

Deno.test("--ktune-opp: a sharded run reproduces the sequential one", async () => {
  const opts = { ktune: CHAMPION, ktuneOpp: EMPTY };
  const one = headless(GAMES, SEED, SEATS, opts);
  const many = await headlessParallel(GAMES, SEED, SEATS, 2, opts);
  // `ktuneOpp` is plain JSON and must cross `postMessage` with the rest of the
  // ArmSpec; a worker that dropped it would silently rebuild the opponents from
  // the SUBJECT's vector and the results would fork here.
  assertEquals(many.results.length, GAMES);
  assertEquals(many.results, one.results);
  many.results.forEach((r, i) => assertEquals(r.seed, SEED + i, `game ${i} out of order`));
});

// ---------------------------------------------------------------------------
// paired: the opponents are the ENVIRONMENT, identical on both arms
// ---------------------------------------------------------------------------

Deno.test("--ktune-opp: paired hands the same opponents to arm A and the control arm", () => {
  // Incumbent form (the only paired form the flag composes with since the
  // epoch — the default hhhh control is frozen and unreachable by a vector):
  // both arms carry the SAME subject file, so the two are the same player and
  // every hanchan must end in a dead tie — but only if the control arm
  // received the opponents' vector too. If `ktuneOpp` were dropped from the
  // shared spread, arm B's opponents would rebuild from the subject's file
  // and the runs would diverge immediately.
  const st = pairedRun(3, SEED, SEATS, {
    ktune: CHAMPION,
    ktuneB: CHAMPION,
    ktuneOpp: EMPTY,
  });
  assertEquals(st.tie, 3, "the two arms faced different opponents");
  assertEquals(st.dScore.mean, 0);
  assertEquals(st.scoreA, st.scoreB);
});

Deno.test("--ktune-opp: --ktune-b swaps the subject only, never the opponents", () => {
  // Arm A: subject CHAMPION. Arm B: subject EMPTY. Both: opponents CHAMPION.
  // The measurement is a difference of SUBJECTS, so the run must not be the same
  // as one where B's opponents were rebuilt from B's own file.
  const st = pairedRun(3, SEED, SEATS, {
    ktune: CHAMPION,
    ktuneB: EMPTY,
    ktuneOpp: CHAMPION,
  });
  const swapped = pairedRun(3, SEED, SEATS, { ktune: CHAMPION, ktuneB: EMPTY, ktuneOpp: EMPTY });
  assert(
    st.scoreB !== swapped.scoreB || st.scoreA !== swapped.scoreA,
    "the opponents' vector appears not to reach the arms at all",
  );
});

// ---------------------------------------------------------------------------
// the flag itself
// ---------------------------------------------------------------------------

Deno.test("--ktune-opp: parsed for the headless drivers, refused under play", () => {
  // hand-calibrated.json: the archived M11 fixture — champion.json no longer
  // carries a `hand` section (removed 2026-08-25, see champion_test), and this
  // test only needs SOME tracked file whose section survives the load.
  const path = new URL("../weights/hand-calibrated.json", import.meta.url).pathname;
  // kkkk: since the epoch the flag needs a "k" opponent to reach (a khhh
  // layout dies inside parseArgs — the frozen seats take no vector).
  const a = parseArgs(["selfplay", `--ktune-opp=${path}`, "--seats=kkkk"]);
  assertEquals(a.ktuneOppPath, path);
  assert(a.ktuneOpp?.hand, "the opponents' file should have loaded its hand block");

  for (const cmd of ["selfplay", "bench"]) {
    assertEquals(argError({ cmd, seats: SEATS, calibrate: "", ktuneOppPath: path }), null);
  }
  // Under `paired` the flag needs an incumbent control since the epoch — the
  // default hhhh arm is frozen and no vector reaches it.
  assert(argError({ cmd: "paired", seats: SEATS, calibrate: "", ktuneOppPath: path }));
  assertEquals(
    argError({ cmd: "paired", seats: SEATS, calibrate: "", ktuneOppPath: path, ktuneBPath: "b" }),
    null,
  );
  // …and it needs a "k" opponent to reach at all: frozen seats take nothing.
  const frozen = argError({ cmd: "selfplay", seats: "khhh", calibrate: "", ktuneOppPath: path });
  assert(frozen?.includes("k席"), `expected the frozen-seat refusal, got: ${frozen}`);
  // `cmdPlay` builds its seats without `openArm`, so the flag would reach nobody.
  const err = argError({ cmd: "play", seats: SEATS, calibrate: "", ktuneOppPath: path });
  assert(err?.includes("--ktune-opp"), `play should refuse the flag, got: ${err}`);
});
