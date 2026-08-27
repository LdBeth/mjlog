// The league of frozen snapshots — one pin block per weights/league/*.json.
//
// THESE PINS NEVER REGENERATE. A snapshot is the champion configuration as it
// stood on its freeze date, fully resolved by `scripts/freeze.ts`; the league
// exists so that candidates are graded against mixed fields of PAST selves,
// and a snapshot that can move is not a past self. When a later default-field
// addition changes a snapshot's behaviour (a new field its JSON does not
// carry, filled from a live default by the merge), the fix is adding the
// EXPLICIT old value to the snapshot JSON — never re-pinning. That is the
// opposite of `test/champion_test.ts`, whose pins DO regenerate on deliberate
// promotion: champion.json is the present, the league is the past.
//
// (frozen-0825, the first snapshot, is the same seat as the epoch-frozen "h"
// letter — its fingerprints below EQUAL frozen_test's EPOCH_PIN. That equality
// is the mechanism check of the JSON snapshot path, and it is expected to hold
// forever: both are never-regenerate pins of the same 2026-08-25 seat.)

import { assertEquals } from "@std/assert";
import { headless } from "../src/harness.ts";
import type { TableSpec } from "../src/spec.ts";
import { loadKtune } from "../src/spec.ts";

// Same fingerprint as test/frozen_test.ts — deliberately self-contained, so a
// pin file depends on nothing that could drift underneath it.
function fingerprint(seed: number, table: TableSpec): string {
  const r = headless(1, seed, table).results[0];
  const body = JSON.stringify({
    scores: r.scores,
    outcomes: r.outcomes,
    ledger: r.ledger.map((v) => `${v.seat}:${v.label}`),
    riichis: r.riichis,
  });
  let x = 0x811c9dc5;
  for (let i = 0; i < body.length; i++) x = Math.imul(x ^ body.charCodeAt(i), 0x01000193) >>> 0;
  return `${r.scores.join("/")}#${x.toString(16).padStart(8, "0")}`;
}

const LEAGUE_DIR = new URL("../weights/league/", import.meta.url).pathname;

const LEAGUE_PIN: Record<string, Record<number, string>> = {
  "frozen-0825.json": {
    101: "28600/23300/51400/16700#af5c064d",
    505: "43600/23300/27900/25200#545ecdaf",
    909: "47600/45000/18100/9300#a9aab631",
  },
};

Deno.test("league: every snapshot plays its freeze-day hanchan forever", () => {
  for (const [file, pins] of Object.entries(LEAGUE_PIN)) {
    const ktune = loadKtune(LEAGUE_DIR + file, `league ${file}`);
    const seat = { kind: "k" as const, ktune };
    const table: TableSpec = [{ ...seat }, { ...seat }, { ...seat }, { ...seat }];
    for (const [seed, want] of Object.entries(pins)) {
      assertEquals(
        fingerprint(Number(seed), table),
        want,
        `${file} 種${seed}: 凍結スナップショットが動いた — JSONに明示値を足して直します (再ピン禁止)`,
      );
    }
  }
});

Deno.test("league: no snapshot exists unpinned", () => {
  const files = [...Deno.readDirSync(LEAGUE_DIR)]
    .filter((e) => e.isFile && e.name.endsWith(".json"))
    .map((e) => e.name)
    .sort();
  assertEquals(
    files,
    Object.keys(LEAGUE_PIN).sort(),
    "weights/league/ と LEAGUE_PIN がずれています — 凍結したらピンを貼ります",
  );
});
