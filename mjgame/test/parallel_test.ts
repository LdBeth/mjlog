// `selfplay --jobs=N`: the same run, over N workers, byte for byte.
//
// The claim under test is not "roughly the same statistics" — it is that a
// sharded run and the sequential one produce the IDENTICAL dataset and the
// identical `MatchResult` list, in the identical order. Anything weaker and the
// flag would silently fork the training data.

import { assert, assertEquals } from "@std/assert";
import { argError } from "../src/cli/args.ts";
import type { ArgCheck } from "../src/cli/args.ts";
import { headless, headlessParallel } from "../src/harness.ts";
import type { RunReport } from "../src/harness.ts";
import { toTenhouXml } from "../src/export.ts";
import { JANKI } from "../src/rules.ts";

const SEED = 4242;
const GAMES = 6;

/** One run of `GAMES` hanchan of hhhh, recording every seat, into `path`. */
function sequential(path: string): RunReport {
  return headless(GAMES, SEED, "hhhh", { record: path, recordAll: true });
}

function parallel(path: string, jobs: number): Promise<RunReport> {
  return headlessParallel(GAMES, SEED, "hhhh", jobs, { record: path, recordAll: true });
}

Deno.test("--jobs: 2 workers reproduce the sequential run byte for byte", async () => {
  const dir = Deno.makeTempDirSync({ prefix: "mjgame-jobs-" });
  try {
    const one = sequential(`${dir}/one.jsonl`);
    const many = await parallel(`${dir}/many.jsonl`, 2);

    // The dataset: the same lines, in the same order, with the same bytes.
    assertEquals(
      Deno.readFileSync(`${dir}/many.jsonl`),
      Deno.readFileSync(`${dir}/one.jsonl`),
    );
    // …and the counts the CLI prints about it.
    assertEquals(many.traj, one.traj);
    assert((one.traj?.m ?? 0) === GAMES, "one 'm' line per hanchan");
    assert((one.traj?.d ?? 0) > 0, "recordAll should record heuristic seats too");

    // The results, in game order — this is what the stats tables and --export
    // are computed from, so equality here is equality of both.
    assertEquals(many.results.length, GAMES);
    assertEquals(many.results, one.results);
    many.results.forEach((r, i) => assertEquals(r.seed, SEED + i, `game ${i} out of order`));
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("--jobs: 5 workers over 6 games, and the exported 牌譜 match", async () => {
  const dir = Deno.makeTempDirSync({ prefix: "mjgame-jobs-" });
  try {
    // More workers than one game each can keep busy, and a job count that does
    // not divide the game count: the round-robin tail is where an off-by-one
    // would put game 5 in the wrong slot.
    const one = headless(GAMES, SEED, "hhhh");
    const many = await headlessParallel(GAMES, SEED, "hhhh", 5, {});
    assertEquals(many.results, one.results);
    // The exporter is a pure function of a MatchResult, but it is also the one
    // consumer that reads the deep `game`/`rounds` structure a postMessage had
    // to clone — so check its actual bytes rather than trusting the deep equal.
    for (let i = 0; i < GAMES; i++) {
      assertEquals(toTenhouXml(many.results[i], JANKI), toTenhouXml(one.results[i], JANKI));
    }
    assertEquals(many.traj, null);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("--jobs: more jobs than games is clamped, not an error", async () => {
  const r = await headlessParallel(2, SEED, "hhhh", 16, {});
  assertEquals(r.results.length, 2);
  assertEquals(r.results.map((m) => m.seed), [SEED, SEED + 1]);
});

Deno.test("--jobs: flag matrix", () => {
  const base: ArgCheck = { cmd: "selfplay", seats: "hhhh", calibrate: "" };
  // 1 job is the loop every command already runs, so it is never refused.
  for (const cmd of ["play", "selfplay", "paired", "bench"]) {
    assertEquals(argError({ ...base, cmd, jobs: 1 }), null, cmd);
  }
  // Anything above that has only one command to shard.
  assertEquals(argError({ ...base, jobs: 4 }), null);
  for (const cmd of ["play", "paired", "bench"]) {
    assertEquals(argError({ ...base, cmd, jobs: 4 }), "--jobs は selfplay 専用です", cmd);
  }
  // --record / --export ride along; --calibrate does not.
  assertEquals(argError({ ...base, jobs: 4, record: "t.jsonl", recordAll: true }), null);
  assertEquals(argError({ ...base, jobs: 4, exportPath: "g" }), null);
  assertEquals(
    argError({ ...base, seats: "khhh", calibrate: "c.jsonl", jobs: 4 }),
    "--calibrate と --jobs は併用できません (較正記録は1スレッドで書きます)",
  );
  // …and one worker is still fine with it, because that is the sequential path.
  assertEquals(argError({ ...base, seats: "khhh", calibrate: "c.jsonl", jobs: 1 }), null);
});
