// Replay interop: a game we played must survive a round trip through Tenhou
// XML and mjrender's parser, event for event. That is what lets the existing
// commentary/snapshot toolchain run on our own games.

import { assert, assertEquals } from "@std/assert";
import { finalStandings } from "mjrender/core.ts";
import { decodeMeld } from "mjrender/meld.ts";
import { parseGame } from "mjrender/parse.ts";
import type { Meld } from "mjrender/model.ts";
import { argError } from "../src/cli/args.ts";
import { RandomPolicy } from "../src/ai/random.ts";
import { encodeMeld, exportPaths, toSidecar, toTenhouXml, writeExport } from "../src/export.ts";
import { headless } from "../src/harness.ts";
import { runMatchSync } from "../src/match.ts";
import { settlement } from "../src/rl/record.ts";
import { DOJO_HEADLESS, JANKI } from "../src/rules.ts";
import { scorer } from "../src/score.ts";
import { SEATS } from "../src/types.ts";
import { repoPath } from "./helpers.ts";

function play(seed: number) {
  return runMatchSync(
    SEATS.map((s) => new RandomPolicy(`P${s}`, seed * 4 + s)),
    { seed, cfg: JANKI, dojo: DOJO_HEADLESS, scorer },
  );
}

/**
 * Hanchan that a random walk never produces: the heuristic seats declare riichi
 * and actually win hands, which is what puts sticks in the pot and score
 * movement in the log. Everything about `sc` and 終局 below is a claim about
 * those, so the tests that make them must not be driven by `play` above.
 */
function hanchans(games: number, seed: number) {
  return headless(games, seed, "hhhh").results;
}

Deno.test("encodeMeld inverts decodeMeld for every meld in the sample log", async () => {
  const xml = await Deno.readTextFile(repoPath("1.xml"));
  const codes = [...xml.matchAll(/<N who="(\d)" m="(\d+)"\s*\/>/g)];
  assert(codes.length > 10, `expected melds in the sample, found ${codes.length}`);

  let checked = 0;
  for (const [, whoS, mS] of codes) {
    const who = Number(whoS);
    const m = Number(mS);
    const meld = decodeMeld(who, m);
    if (meld.kind === "nuki") continue; // sanma only
    assertEquals(encodeMeld(meld), m, `${meld.kind} who=${who} m=${m}`);
    checked++;
  }
  assert(checked > 10, "too few melds actually round-tripped");
});

Deno.test("encodeMeld round-trips every meld our own engine produces", () => {
  const kinds = new Set<string>();
  let checked = 0;
  for (let seed = 1; seed <= 30; seed++) {
    for (const round of play(seed).rounds) {
      for (const e of round.events) {
        if (e.t !== "call") continue;
        const m: Meld = e.meld;
        const back = decodeMeld(m.who, encodeMeld(m));
        assertEquals(back.kind, m.kind, `kind for ${JSON.stringify(m)}`);
        assertEquals(back.tiles, m.tiles, `tiles for ${JSON.stringify(m)}`);
        assertEquals(back.calledTile, m.calledTile, `calledTile for ${JSON.stringify(m)}`);
        assertEquals(back.fromWho, m.fromWho, `fromWho for ${JSON.stringify(m)}`);
        kinds.add(m.kind);
        checked++;
      }
    }
  }
  assert(checked > 100, `only ${checked} melds exercised`);
  // Random play calls freely, so all five 4-player meld kinds should appear.
  for (const k of ["chi", "pon", "ankan", "daiminkan", "shouminkan"]) {
    assert(kinds.has(k), `no ${k} produced; saw [${[...kinds].join(",")}]`);
  }
});

Deno.test("exported XML parses back into the same event stream", () => {
  for (let seed = 1; seed <= 10; seed++) {
    const m = play(seed);
    const xml = toTenhouXml(m, JANKI);
    const back = parseGame(xml);

    assertEquals(back.rounds.length, m.rounds.length, `seed ${seed}: round count`);
    for (let i = 0; i < m.rounds.length; i++) {
      const a = m.rounds[i];
      const b = back.rounds[i];
      assertEquals(b.kyoku, a.kyoku, `seed ${seed} r${i}: kyoku`);
      assertEquals(b.honba, a.honba, `seed ${seed} r${i}: honba`);
      assertEquals(b.dealer, a.dealer, `seed ${seed} r${i}: dealer`);
      assertEquals(b.firstDora, a.firstDora, `seed ${seed} r${i}: dora`);
      assertEquals(b.startHands, a.startHands, `seed ${seed} r${i}: deal`);
      assertEquals(
        b.events.map((e) => JSON.stringify(e)),
        a.events.map((e) => JSON.stringify(e)),
        `seed ${seed} r${i}: events`,
      );
    }
  }
});

Deno.test("the sidecar carries what Tenhou XML cannot", () => {
  const m = play(3);
  const side = toSidecar(m, JANKI);
  // Two red 5-pin is exactly the thing standard XML cannot express.
  assertEquals(side.akaIds, [52, 53]);
  assertEquals(side.ruleset, "janki");
  assertEquals(side.scores, m.scores);
});

/**
 * Tenhou's `sc` is (score AT THE RESULT, delta), and a riichi stick has already
 * left that score. Basing it on the deal instead double-counts every stick — the
 * error is invisible inside a round (each INIT restates the scores) and shows up
 * as a reader's running total drifting 1000 per declaration. So: chain them.
 */
Deno.test("each round's sc lands exactly on the next round's INIT ten", () => {
  let sticks = 0;
  for (const m of hanchans(10, 100)) {
    const seed = m.seed;
    // One segment per round, each holding that round's INIT and its result(s).
    const segments = toTenhouXml(m, JANKI).split("<INIT ").slice(1);
    assertEquals(segments.length, m.rounds.length, `seed ${seed}: INIT count`);

    let prevEnd: number[] | null = null;
    for (let i = 0; i < segments.length; i++) {
      const ten = segments[i].match(/^seed="[^"]*" ten="([^"]*)"/)![1].split(",").map(Number);
      if (prevEnd) assertEquals(ten, prevEnd, `seed ${seed}: round ${i - 1} → ${i}`);
      // Double ron writes two AGARI in one round; both carry the same base, so
      // the round ends where the last of them lands.
      const sc = [...segments[i].matchAll(/<(?:AGARI|RYUUKYOKU)[^>]*\ssc="([^"]*)"/g)]
        .map((r) => r[1].split(",").map(Number));
      assert(sc.length > 0, `seed ${seed}: round ${i} has no result element`);
      const last = sc[sc.length - 1];
      prevEnd = SEATS.map((s) => last[2 * s] + last[2 * s + 1]);
      sticks += [...segments[i].matchAll(/<REACH [^>]*step="2"/g)].length;
    }
  }
  // Without a declaration in the sample the chain above is trivially satisfied
  // by the very bug it exists to catch.
  assert(sticks > 5, `only ${sticks} riichi in the sample — the check proves nothing`);
});

Deno.test("終局 rides on the last result, settled the way the game settles it", () => {
  for (const m of hanchans(10, 100)) {
    const seed = m.seed;
    const back = parseGame(toTenhouXml(m, JANKI));
    const standings = finalStandings(back);
    assert(standings, `seed ${seed}: no 終局 record in the exported log`);
    const net = settlement(m.scores, JANKI);
    for (const row of standings) {
      assertEquals(row.score, m.scores[row.seat], `seed ${seed}: seat ${row.seat} score`);
      assertEquals(row.points, net[row.seat], `seed ${seed}: seat ${row.seat} 精算`);
    }
    // Sorted 1st→4th, so the top row is the top score.
    assertEquals(standings[0].score, Math.max(...m.scores), `seed ${seed}: winner`);
  }
});

Deno.test("--export names the pair from one basename", () => {
  assertEquals(exportPaths("games/seed42"), {
    xml: "games/seed42.xml",
    sidecar: "games/seed42.mjgame.json",
  });
  // An explicit .xml is used verbatim — the sidecar still takes the basename.
  assertEquals(exportPaths("games/seed42.xml"), {
    xml: "games/seed42.xml",
    sidecar: "games/seed42.mjgame.json",
  });
  assertEquals(exportPaths("g", "-0007"), { xml: "g-0007.xml", sidecar: "g-0007.mjgame.json" });
});

Deno.test("--export is refused where nothing would write one match per file", () => {
  const base = { cmd: "selfplay", seats: "hhhh", calibrate: "" };
  assertEquals(argError({ ...base, exportPath: "g" }), null);
  assertEquals(argError({ ...base, cmd: "play", exportPath: "g" }), null);
  for (const cmd of ["paired", "bench"]) {
    assertEquals(
      argError({ ...base, cmd, exportPath: "g" }),
      "--export は play / selfplay 専用です",
      cmd,
    );
  }
});

Deno.test("a selfplay run writes a readable pair per game", async () => {
  const dir = await Deno.makeTempDir({ prefix: "mjgame-export-" });
  try {
    const { results } = headless(2, 4242, "hhhh");
    const written = results.map((r, i) =>
      writeExport(r, JANKI, `${dir}/run`, `-${String(i + 1).padStart(4, "0")}`)
    );
    assertEquals(written.map((w) => w.xml), [`${dir}/run-0001.xml`, `${dir}/run-0002.xml`]);

    for (let i = 0; i < results.length; i++) {
      const xml = Deno.readTextFileSync(written[i].xml);
      assert(xml.startsWith(`<mjloggm ver="2.3">`), "no mjlog root");
      assert(xml.endsWith("</mjloggm>"), "unterminated mjlog");
      // Well-formed as far as the consumer that matters is concerned.
      const back = parseGame(xml);
      assertEquals(back.rounds.length, results[i].rounds.length);
      assert(finalStandings(back), "no 終局 record");

      const side = JSON.parse(Deno.readTextFileSync(written[i].sidecar));
      assertEquals(side.seed, results[i].seed);
      assertEquals(side.scores, results[i].scores);
      assertEquals(side.akaIds, [...JANKI.akaIds]);
      // The ledger is the sidecar's whole reason for existing: no Tenhou element
      // can carry a 禁じ手, so a lossy copy here would lose it silently.
      assertEquals(side.violations, JSON.parse(JSON.stringify(results[i].ledger)));
    }
  } finally {
    await Deno.remove(dir, { recursive: true });
  }
});
