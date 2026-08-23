// The 助言 row: a 計算 seat consulted on the human's own Observation. It is a
// mirror, not a player — it sees exactly what the player sees and its answer
// is rendered, never submitted. The discard trace it leaves behind is a record
// of the scoring that happened anyway, so asking for it must not change the
// choice it explains.

import { assert, assertEquals } from "@std/assert";
import type { Tile } from "mjrender/model.ts";
import { HeuristicPolicy } from "../src/ai/heuristic.ts";
import { makePolicy } from "../src/harness.ts";
import { JANKI } from "../src/rules.ts";
import type { Action } from "../src/types.ts";
import type { Observation } from "../src/observe.ts";
import { adviceRow, overlay } from "../src/tui/widgets.ts";
import type { Ctx } from "../src/tui/widgets.ts";
import { lineWidth } from "../src/tui/screen.ts";
import { tiles } from "./helpers.ts";

function turnObs(): Observation {
  const hand = tiles("123456789m1234p");
  const drawn = tiles("東")[0];
  const all = [...hand, drawn];
  const legal: Action[] = all.map((tile) => ({
    t: "discard",
    tile,
    riichi: false,
    tsumogiri: tile === drawn,
  }));
  return {
    seat: 0,
    kyoku: 0,
    honba: 0,
    kyotaku: 0,
    junme: 1,
    wallRemaining: 69,
    hand: all,
    drawn,
    melds: [[], [], [], []],
    rivers: [[], [], [], []],
    scores: [30000, 30000, 30000, 30000],
    riichi: [false, false, false, false],
    riichiJunme: [-1, -1, -1, -1],
    doraIndicators: [0 as Tile],
    seatWind: 27,
    roundWind: 27,
    akaIds: JANKI.akaIds,
    shanten: 1,
    waits: [],
    ronnable: [],
    katagari: false,
    ukeire: [],
    doraCount: 0,
    furiten: { permanent: false, temporary: false, riichi: false },
    discardInfo: new Map(),
    tsumogiriLock: false,
    danger: new Map(),
    violations: [0, 0, 0, 0],
    legal,
  };
}

function ctxWith(obs: Observation, advice: Ctx["advice"]): Ctx {
  return {
    obs,
    glyph: { mode: "ascii", aka: JANKI.akaIds },
    names: ["あなた", "CPU東", "CPU南", "CPU西"],
    slots: obs.hand,
    drawnIndex: 13,
    cursor: 0,
    selectable: obs.hand.map(() => true),
    riichiArmed: false,
    phase: "turn",
    timer: { turnMs: 0, bankMs: 0, turnLeftMs: 0, bankLeftMs: 0, overMs: 0 },
    ledger: [],
    log: [],
    message: "",
    riverRows: 2,
    claim: null,
    advice,
  };
}

function text(l: ReturnType<typeof adviceRow>): string {
  return l.map((s) => s.text).join("");
}

Deno.test("advice: the trace records the very discard the seat chose", () => {
  const seat = makePolicy({ kind: "k", name: "助言", seed: 7 });
  const p = seat.policy as HeuristicPolicy;
  const obs = turnObs();
  const a = p.decide(obs);
  assertEquals(a.t, "discard");
  const tr = p.lastTrace!;
  assert(tr);
  if (a.t === "discard") assertEquals(tr.chosen, a.tile);
  assertEquals(tr.candidates[0].tile, tr.chosen, "best-first order");
  assertEquals(tr.candidates.length, 14, "every distinct tile was scored");
  // 1p: 123 456 789m 234p + 東 is tenpai on a ダブ東 tanki — the honour stays.
  assertEquals(tr.chosen, tiles("1p")[0]);
  seat.close();
});

Deno.test("advice: the row names the tile, the overlay ranks the candidates", () => {
  const seat = makePolicy({ kind: "k", name: "助言", seed: 7 });
  const p = seat.policy as HeuristicPolicy;
  const obs = turnObs();
  const action = p.decide(obs);
  const ctx = ctxWith(obs, { action, trace: p.lastTrace });
  const row = text(adviceRow(ctx, 80));
  assert(row.startsWith("助言  打 "), row);
  assert(row.includes("次点"), row);
  assertEquals(lineWidth(adviceRow(ctx, 80)), 80, "padded to the panel width");
  const ov = overlay(ctx, { kind: "advice" });
  assertEquals(ov.title, "助言の根拠");
  assert(ov.body.length >= 3 + 14, "header rows + one per candidate");
  assert(ov.body.some((l) => l[0]?.text === "▶ "), "the chosen tile is marked");
  seat.close();
});

Deno.test("advice: no advisor ⇒ a placeholder, never a crash", () => {
  const ctx = ctxWith(turnObs(), null);
  assertEquals(text(adviceRow(ctx, 40)).trimEnd(), "助言  —");
  assertEquals(overlay(ctx, { kind: "advice" }).body.length, 1);
});
