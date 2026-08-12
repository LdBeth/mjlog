// The opening sequence: title card, seating, then the deal animation.
// Verified by capturing the frames the app writes rather than by eye.

import { assert, assertEquals } from "@std/assert";
import { App } from "../src/tui/app.ts";
import { JANKI } from "../src/rules.ts";
import type { Observation } from "../src/observe.ts";
import type { Tile } from "mjrender/model.ts";
import type { Action } from "../src/types.ts";
import { tiles } from "./helpers.ts";

/** Strip SGR/cursor escapes so assertions read the visible text only. */
function plain(s: string): string {
  // deno-lint-ignore no-control-regex -- matching ESC is the entire point
  return s.replace(/\x1b\[[0-9;]*[A-Za-z]/g, "");
}

function app(noIntro: boolean, sink: string[]): App {
  return new App({
    glyphs: "ascii",
    aka: JANKI.akaIds,
    names: ["あなた", "CPU東", "CPU南", "CPU西"],
    thinkLimitMs: 3000,
    timerTurnMs: 3_000, // the shipping defaults: 3s a turn, 10s for the match
    timerBankMs: 10_000,
    cpuDelayMs: 0,
    cfg: JANKI,
    noIntro,
    write: (s) => sink.push(s),
  });
}

function turnObs(): Observation {
  const hand = tiles("123456789m123p9p");
  const drawn = hand[hand.length - 1];
  const legal: Action[] = hand.map((tile) => ({
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
    hand,
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
    ukeire: [],
    doraCount: 0,
    furiten: { permanent: false, temporary: false, riichi: false },
    danger: new Map(),
    violations: [0, 0, 0, 0],
    legal,
  };
}

Deno.test("intro: shows the ruleset and the seating before play", async () => {
  const frames: string[] = [];
  const a = app(false, frames);
  await a.intro();
  a.stop();

  const all = plain(frames.join("\n"));
  assert(all.includes("雀"), "title card should name the ruleset");
  assert(all.includes("東南戦"), "should state the match length");
  assert(all.includes("30000点持ち"), "should state the starting score");
  assert(all.includes("東家"), "should show the seat winds");
  assert(all.includes("あなた"), "should seat the human");
  assert(all.includes("配 牌"), "should hand off into the deal");
  assert(frames.length > 5, `expected a built-up sequence, got ${frames.length} frames`);
});

Deno.test("intro: --no-intro emits nothing at all", async () => {
  const frames: string[] = [];
  const a = app(true, frames);
  await a.intro();
  a.stop();
  assertEquals(frames.length, 0, "skipping the intro must not paint");
});

Deno.test("intro: a keypress cuts the opening short", async () => {
  const frames: string[] = [];
  const a = app(false, frames);
  const p = a.intro();
  a.feed("x"); // any key
  await p;
  a.stop();
  const all = plain(frames.join("\n"));
  // It still painted something, but stopped well before the full sequence.
  assert(frames.length >= 1, "should have painted at least the first frame");
  assert(!all.includes("配 牌"), "a skipped intro should not reach the deal card");
});

Deno.test("deal animation: the hand builds up 4-4-4-1 before the clock starts", async () => {
  const frames: string[] = [];
  const a = app(false, frames);
  // A deal event arms the animation; the next decision plays it.
  a.onEvent({
    e: "deal",
    kyoku: 0,
    honba: 0,
    kyotaku: 0,
    dealer: 0,
    scores: [30000, 30000, 30000, 30000],
    indicator: 0 as Tile,
  });
  frames.length = 0;

  const t0 = Date.now();
  const p = a.awaitDecision(turnObs());
  // The animation must finish before the countdown is armed, so a full base
  // allowance is still available once it ends.
  await new Promise((r) => setTimeout(r, 900));
  const t = a.timerSnapshot();
  assert(
    t.turnLeftMs > 2_500,
    `animation must not burn the clock; ${t.turnLeftMs}ms left after ${Date.now() - t0}ms`,
  );
  assertEquals(t.bankLeftMs, 10_000, "bank untouched by the animation");

  assert(frames.length >= 5, `expected staged deal frames, got ${frames.length}`);
  a.feed("\r");
  await p;
  a.stop();
});
