// MJAI notation arithmetic, pinned against a REAL riichi.dev wire capture
// (runs/arena/validate-0827-*.jsonl): the arena's start_kyoku hand
// ["3m","7m","3p","3p","9p","9p","3s","4s","5s","7s","E","S","F"] arrived in
// its decoded observation as ids [9,25,45,46,68,70,81,86,89,97,108,113,128],
// indicator "F" as 129 — the Tenhou 136-id scheme, reds = first copies.

import { assert, assertEquals } from "@std/assert";
import {
  ARENA_AKA_IDS,
  ARENA_CFG,
  idMatchesPai,
  idToPai,
  matchId,
  paiIsRed,
  paiToType,
  typeToPai,
} from "../src/net/mjai.ts";

Deno.test("mjai: the captured hand round-trips id → pai", () => {
  const ids = [9, 25, 45, 46, 68, 70, 81, 86, 89, 97, 108, 113, 128];
  const pais = ["3m", "7m", "3p", "3p", "9p", "9p", "3s", "4s", "5s", "7s", "E", "S", "F"];
  assertEquals(ids.map(idToPai), pais);
  ids.forEach((id, i) => assert(idMatchesPai(id, pais[i])));
});

Deno.test("mjai: every type round-trips through the wire string", () => {
  for (let ty = 0; ty < 34; ty++) {
    assertEquals(paiToType(typeToPai(ty)), ty);
  }
  // The red variants of the three fives.
  for (const [pai, ty] of [["5mr", 4], ["5pr", 13], ["5sr", 22]] as const) {
    assertEquals(paiToType(pai), ty);
    assert(paiIsRed(pai));
    assertEquals(typeToPai(ty, true), pai);
  }
});

Deno.test("mjai: red identity — first copy of each five, and only it", () => {
  assertEquals([...ARENA_AKA_IDS].sort((a, b) => a - b), [16, 52, 88]);
  assertEquals(idToPai(16), "5mr");
  assertEquals(idToPai(17), "5m");
  assert(idMatchesPai(52, "5pr") && !idMatchesPai(52, "5p"));
  assert(!idMatchesPai(53, "5pr") && idMatchesPai(53, "5p"));
});

Deno.test("mjai: matchId honors redness and never invents", () => {
  const hand = [17, 16, 52, 129];
  assertEquals(matchId(hand, "5m"), 17); // plain never takes the red copy
  assertEquals(matchId(hand, "5mr"), 16);
  assertEquals(matchId(hand, "5pr"), 52);
  assertEquals(matchId(hand, "F"), 129);
  assertEquals(matchId(hand, "5sr"), null);
  assertEquals(matchId(hand, "1m"), null);
});

Deno.test("mjai: the arena config differs from JANKI only in aka", () => {
  assertEquals(ARENA_CFG.akaIds, ARENA_AKA_IDS);
});
