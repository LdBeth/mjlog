/// <reference lib="deno.worker" />
// One shard of a `selfplay --jobs=N` run.
//
// The whole protocol is three messages wide: the main thread posts one
// `ShardInit` (the arm to build and the games this worker owns), the worker
// answers "ready" once its arm exists, then one "game" per finished hanchan, then
// "done". Nothing is ever pulled — a worker plays its whole list and the main
// thread reorders. See the sharding/identity note in `harness.ts`, which owns the
// message types and the reasoning; this file is only the loop.
//
// TWO THINGS THIS FILE MUST NOT DO, both load-bearing for byte-identity:
//
//   * touch the dataset file. `--record` names ONE file for the run, and four
//     workers opening it would truncate it four times and interleave the rest.
//     The arm is built on a BUFFERING writer instead, drained per game, and the
//     text is posted back for the main thread to append in game order.
//   * touch the export files. Those are written from `results[]` on the main
//     thread, in game order, for the same reason.
//
// The kernel warm-up before "ready" is not a nicety: `native/mjkernel.cc` fills
// its lazy tables without a lock, so the FIRST caller must be alone. The main
// thread spawns worker 0, waits for this message, and only then spawns the rest.

import { closeArm, openArm, playGame } from "./harness.ts";
import type { ShardInit, ShardOut } from "./harness.ts";
import { shanten } from "./kernel.ts";
import { TrajectoryWriter } from "./rl/record.ts";

const EMPTY = { d: 0, r: 0, m: 0 };

function post(m: ShardOut): void {
  self.postMessage(m);
}

function run(init: ShardInit): void {
  // Allocates (and starts memoising) the native kernel's shared lookup tables
  // while this worker is still the only one running. A hand of nothing is a
  // perfectly good probe: the entry point allocates before it looks at anything.
  shanten(new Array(34).fill(0));
  const writer = init.record ? TrajectoryWriter.buffering() : null;
  // `init.table` is already resolved — the main thread ran `resolveTable`
  // once, so every shard builds from the same four specs.
  const arm = openArm(init.table, { ...init.opts, writer });
  post({ k: "ready" });
  try {
    for (const g of init.games) {
      const result = playGame(arm, g.seed);
      const t = writer?.drain();
      post({ k: "game", i: g.i, result, traj: t?.text ?? "", counts: t?.counts ?? EMPTY });
    }
  } finally {
    closeArm(arm);
  }
  // No `self.close()`: the main thread terminates the worker once it has the
  // "done", so a message can never be dropped by a shutdown racing the queue.
  post({ k: "done" });
}

self.onmessage = (ev: MessageEvent<ShardInit>) => {
  try {
    run(ev.data);
  } catch (e) {
    post({ k: "error", message: e instanceof Error ? e.message : String(e) });
  }
};
