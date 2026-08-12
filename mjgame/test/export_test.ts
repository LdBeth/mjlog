// Replay interop: a game we played must survive a round trip through Tenhou
// XML and mjrender's parser, event for event. That is what lets the existing
// commentary/snapshot toolchain run on our own games.

import { assert, assertEquals } from "@std/assert";
import { decodeMeld } from "mjrender/meld.ts";
import { parseGame } from "mjrender/parse.ts";
import type { Meld } from "mjrender/model.ts";
import { RandomPolicy } from "../src/ai/random.ts";
import { encodeMeld, toSidecar, toTenhouXml } from "../src/export.ts";
import { runMatchSync } from "../src/match.ts";
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
