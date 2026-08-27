// The wire↔Action mapping, and the reach two-step.
//
// The mappers are pure — synthetic possible_actions in, mjgame actions out —
// so the aka/consumed/hora corners are pinned without a server. The two-step
// runs through a real ChampionChooser (shadow and all) with the decide seam
// forcing the riichi branch: the reply must be the server's `reach` entry, and
// the FOLLOW-UP request must be answered from memory — the policy re-deciding
// there would see pre-riichi state and could split declaration from discard.

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { MjaiAction, RequestAction } from "../src/net/arena.ts";
import { ChampionChooser, mapClaimActions, mapTurnActions } from "../src/net/champion.ts";
import type { Action } from "../src/types.ts";

Deno.test("map: dahai entries pick concrete ids, aka and tsumogiri honored", () => {
  // Hand holds 5pr(52), 5p(53), 1m(0); drawn is the red five.
  const entries: MjaiAction[] = [
    { type: "dahai", pai: "5pr" },
    { type: "dahai", pai: "5p" },
    { type: "dahai", pai: "1m" },
  ];
  const mapped = mapTurnActions(entries, {
    hand: [0, 53, 52],
    drawn: 52,
    openMelds: 0,
    menzen: true,
  });
  const acts = mapped.map((m) => m.action);
  assertEquals(acts, [
    { t: "discard", tile: 52, riichi: false, tsumogiri: true },
    { t: "discard", tile: 53, riichi: false, tsumogiri: false },
    { t: "discard", tile: 0, riichi: false, tsumogiri: false },
  ]);
  mapped.forEach((m, i) => assertEquals(m.entry, entries[i]));
});

Deno.test("map: reach variants only for tenpai-keeping discards, paired to the reach entry", () => {
  // 123m 456m 789m 123p 5s + drawn 9s: cutting 5s or 9s keeps tenpai, 1m does not.
  const hand = [0, 4, 8, 12, 17, 20, 24, 28, 32, 36, 40, 44, 88 + 1, 104];
  const entries: MjaiAction[] = [
    { type: "dahai", pai: "1m" },
    { type: "dahai", pai: "5s" },
    { type: "dahai", pai: "9s" },
    { type: "reach" },
    { type: "none" },
  ];
  const mapped = mapTurnActions(entries, { hand, drawn: 104, openMelds: 0, menzen: true });
  const riichis = mapped.filter((m) => m.action.t === "discard" && m.action.riichi);
  assertEquals(riichis.map((m) => (m.action as { tile: number }).tile), [89, 104]);
  for (const m of riichis) assertEquals(m.entry, entries[3]);
  // The riichi variant follows its plain twin, as turnActions orders them.
  const seq = mapped.map((m) =>
    m.action.t === "discard" ? `${m.action.tile}${m.action.riichi ? "R" : ""}` : m.action.t
  );
  assertEquals(seq, ["0", "89", "89R", "104", "104R"]);
});

Deno.test("map: turn hora is tsumo, ankan by type, kakan by held id", () => {
  const entries: MjaiAction[] = [
    { type: "dahai", pai: "E" },
    { type: "hora", actor: 0, target: 0 },
    { type: "ankan", consumed: ["1s", "1s", "1s", "1s"] },
    { type: "kakan", pai: "C", consumed: ["C", "C", "C"] },
  ];
  const hand = [72, 73, 74, 75, 108, 132];
  const mapped = mapTurnActions(entries, { hand, drawn: 132, openMelds: 2, menzen: false });
  const byT = new Map(mapped.map((m) => [m.action.t, m]));
  assertEquals(byT.get("tsumo")!.entry, entries[1]);
  assertEquals(byT.get("ankan")!.action, { t: "ankan", type: 18 });
  assertEquals(byT.get("kakan")!.action, { t: "kakan", tile: 132 });
});

Deno.test("map: claims — pass first, aka consumed distinct, daiminkan, ron", () => {
  const entries: MjaiAction[] = [
    { type: "chi", pai: "6p", consumed: ["5pr", "4p"], target: 3 },
    { type: "pon", pai: "5p", consumed: ["5pr", "5p"], target: 3 },
    { type: "kan", pai: "5p", consumed: ["5p", "5p", "5pr"], target: 3 },
    { type: "hora", actor: 0, target: 3, pai: "5p" },
    { type: "none" },
  ];
  const hand = [48, 52, 53, 54];
  const mapped = mapClaimActions(entries, hand, 55);
  assertEquals(mapped[0].action, { t: "pass" });
  const chi = mapped.find((m) => m.action.t === "chi")!.action as Extract<Action, { t: "chi" }>;
  assertEquals(chi.tiles, [52, 48]);
  assertEquals(chi.called, 55);
  const pon = mapped.find((m) => m.action.t === "pon")!.action as Extract<Action, { t: "pon" }>;
  assertEquals(pon.tiles, [52, 53]); // two "5p"-ish strings, two DISTINCT copies
  assertEquals(mapped.find((m) => m.action.t === "daiminkan")!.action, {
    t: "daiminkan",
    called: 55,
  });
  assert(mapped.some((m) => m.action.t === "ron"));
});

Deno.test("map: a consumed tile missing from hand throws (desync, not chombo)", () => {
  assertThrows(() =>
    mapClaimActions(
      [{ type: "pon", pai: "5p", consumed: ["5pr", "5pr"], target: 1 }, { type: "none" }],
      [52, 53],
      54,
    )
  );
});

// ---------------------------------------------------------------------------
// the reach two-step
// ---------------------------------------------------------------------------

function req(id: number, possible: MjaiAction[]): RequestAction {
  return { type: "request_action", request_id: id, possible_actions: possible };
}

Deno.test("champion: reach replies reach, then answers the remembered dahai from memory", () => {
  let phase: "declare" | "silent" = "declare";
  const chooser = new ChampionChooser({
    decide: (obs) => {
      if (phase === "silent") throw new Error("step 2 で policy が呼ばれた");
      const riichi = obs.legal.find((a) => a.t === "discard" && a.riichi && a.tsumogiri);
      assert(riichi, "riichi 打牌が legal に現れない");
      return riichi;
    },
  });
  chooser.onEvent({ type: "start_game", id: 0 });
  chooser.onEvent({
    type: "start_kyoku",
    bakaze: "E",
    kyoku: 1,
    honba: 0,
    kyotaku: 0,
    oya: 0,
    scores: [25000, 25000, 25000, 25000],
    dora_marker: "1m",
    tehais: [
      ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5s"],
      Array(13).fill("?"),
      Array(13).fill("?"),
      Array(13).fill("?"),
    ],
  });
  chooser.onEvent({ type: "tsumo", actor: 0, pai: "9s" });

  const entries: MjaiAction[] = [
    { type: "dahai", pai: "1m" },
    { type: "dahai", pai: "5s" },
    { type: "dahai", pai: "9s" },
    { type: "reach" },
  ];
  const first = chooser.choose(req(1, entries));
  assertEquals(first, entries[3]); // the reach entry, verbatim

  // The server acknowledges the declaration, then asks for the discard.
  chooser.onEvent({ type: "reach", actor: 0 });
  phase = "silent";
  const followup: MjaiAction[] = [{ type: "dahai", pai: "5s" }, { type: "dahai", pai: "9s" }];
  const second = chooser.choose(req(2, followup));
  assertEquals(second, followup[1]); // the remembered 9s, policy untouched
  assertEquals(chooser.fallbacks, 0);
});
