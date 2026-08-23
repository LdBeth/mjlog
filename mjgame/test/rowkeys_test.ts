// The number row discards in one keystroke. On a JIS keyboard 1-9, 0, -, ^, ¥
// are thirteen keys in a line — a hand's width — read left to right like the
// tiles on screen; `\` is what most terminals send for the ¥ key. The drawn
// tile is not on the row (it is `t`), so the row never shifts as the hand does.

import { assertEquals } from "@std/assert";
import type { Tile } from "mjrender/model.ts";
import { App } from "../src/tui/app.ts";
import { JANKI } from "../src/rules.ts";
import type { Action } from "../src/types.ts";
import type { Observation } from "../src/observe.ts";
import { tiles } from "./helpers.ts";

function app(): App {
  return new App({
    glyphs: "ascii",
    aka: JANKI.akaIds,
    names: ["あなた", "CPU東", "CPU南", "CPU西"],
    timerTurnMs: 0,
    timerBankMs: 0,
    cpuDelayMs: 0,
    cfg: JANKI,
    noIntro: true,
    write: () => {},
  });
}

/** 13 sorted tiles + a drawn tile, every discard legal. */
function turnObs(): { obs: Observation; sorted: Tile[]; drawn: Tile } {
  const hand = tiles("123456789m1234p");
  const drawn = tiles("5s")[0];
  const all = [...hand, drawn];
  const legal: Action[] = all.map((tile) => ({
    t: "discard",
    tile,
    riichi: false,
    tsumogiri: tile === drawn,
  }));
  const obs: Observation = {
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
    shanten: 2,
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
  return { obs, sorted: [...hand].sort((x, y) => x - y), drawn };
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

const ROW = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "^", "¥"];

Deno.test("row keys: each of the thirteen keys discards that slot at once", async () => {
  for (let i = 0; i < ROW.length; i++) {
    const a = app();
    const { obs, sorted } = turnObs();
    const p = a.awaitDecision(obs);
    await sleep(20);
    a.feed(ROW[i]);
    const chosen = await p;
    assertEquals(chosen.t, "discard");
    if (chosen.t === "discard") assertEquals(chosen.tile, sorted[i], `key ${ROW[i]}`);
    a.stop();
  }
});

Deno.test("row keys: backslash is the ¥ key, and t is still the drawn tile", async () => {
  const a = app();
  const { obs, sorted } = turnObs();
  const p = a.awaitDecision(obs);
  await sleep(20);
  a.feed("\\");
  const chosen = await p;
  if (chosen.t === "discard") assertEquals(chosen.tile, sorted[12]);
  a.stop();

  const b = app();
  const t = turnObs();
  const q = b.awaitDecision(t.obs);
  await sleep(20);
  b.feed("t");
  const tg = await q;
  if (tg.t === "discard") assertEquals(tg.tile, t.drawn);
  b.stop();
});
