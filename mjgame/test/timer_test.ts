// The decision countdown: a fresh allowance each turn, plus one match bank.
// It is informational — running out costs 評価点 (the 長考 penalty), never the
// turn itself. A single-player game must not have moves taken away from it.

import { assert, assertEquals } from "@std/assert";
import type { Tile } from "mjrender/model.ts";
import { App } from "../src/tui/app.ts";
import { JANKI } from "../src/rules.ts";
import type { Action } from "../src/types.ts";
import type { Observation } from "../src/observe.ts";
import { tiles } from "./helpers.ts";

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

function app(turnMs: number, bankMs: number): App {
  return new App({
    glyphs: "ascii",
    aka: JANKI.akaIds,
    names: ["あなた", "CPU東", "CPU南", "CPU西"],
    thinkLimitMs: 3000,
    timerTurnMs: turnMs,
    timerBankMs: bankMs,
    cpuDelayMs: 0,
    cfg: JANKI,
    noIntro: true, // the animation must not eat the clock under test
    write: () => {}, // keep the suite output readable
  });
}

/** A minimal turn observation: 13 tiles + a drawn tile, all discards legal. */
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
    ronnable: [],
    katagari: false,
    discardInfo: new Map(),
    ukeire: [],
    doraCount: 0,
    furiten: { permanent: false, temporary: false, riichi: false },
    danger: new Map(),
    violations: [0, 0, 0, 0],
    legal,
  };
}

/** A claim observation: someone discarded and we may pon or pass. */
function claimObs(): Observation {
  const o = turnObs();
  const called = tiles("5s")[0];
  return {
    ...o,
    drawn: null,
    legal: [
      { t: "pass" },
      { t: "pon", tiles: [tiles("5s")[1], tiles("5s")[2]], called },
    ],
  };
}

Deno.test("timer: expiring never plays for you", async () => {
  const a = app(60, 40);
  let settled = false;
  const p = a.awaitDecision(turnObs()).then((x) => {
    settled = true;
    return x;
  });

  // Well past allowance + bank: nothing should have been submitted.
  await sleep(400);
  assertEquals(settled, false, "the clock must not take the turn");
  assert(a.timerSnapshot().overMs > 0, "it should be in overtime by now");

  a.feed("t"); // ツモ切り, on our own terms
  const chosen = await p;
  assertEquals(chosen.t, "discard");
  a.stop();
});

Deno.test("timer: an expired claim is still ours to answer", async () => {
  const a = app(50, 30);
  let settled = false;
  const p = a.awaitDecision(claimObs()).then((x) => {
    settled = true;
    return x;
  });
  await sleep(300);
  assertEquals(settled, false, "a call must never be passed for us");

  a.feed("p"); // pon, deliberately, long after the clock ran out
  const chosen = await p;
  assertEquals(chosen.t, "pon");
  a.stop();
});

Deno.test("timer: overtime keeps counting up, without bound", async () => {
  const a = app(40, 20);
  const p = a.awaitDecision(turnObs());
  await sleep(200);
  const first = a.timerSnapshot();
  await sleep(200);
  const second = a.timerSnapshot();

  assertEquals(first.turnLeftMs, 0);
  assertEquals(first.bankLeftMs, 0);
  assert(
    second.overMs > first.overMs,
    `overtime should keep growing: ${first.overMs} -> ${second.overMs}`,
  );
  a.feed("t");
  await p;
  a.stop();
});

Deno.test("timer: acting inside the turn allowance leaves the bank untouched", async () => {
  const a = app(400, 300);
  const obs = turnObs();
  const p = a.awaitDecision(obs);
  await sleep(60);
  // Discard the first tile in hand via the app's own submit path.
  a.feed("\r");
  const chosen = await p;
  assertEquals(chosen.t, "discard");
  assertEquals(a.bankRemainingMs(), 300, "bank should be intact");
  a.stop();
});

Deno.test("timer: overrun is charged to the bank, and the bank does not refill", async () => {
  const a = app(60, 200);
  const p1 = a.awaitDecision(turnObs());
  await sleep(400); // 60ms turn allowance, then the whole 200ms bank, then overtime
  a.feed("t");
  await p1;
  assertEquals(a.bankRemainingMs(), 0, "a long think drains the bank");

  // The bank is spent for the rest of the match: the next decision goes
  // straight from base into overtime.
  const p2 = a.awaitDecision(turnObs());
  await sleep(150);
  const t = a.timerSnapshot();
  assertEquals(t.bankLeftMs, 0, "the bank must not refill");
  assert(t.overMs > 0, "with no bank left, overtime starts as soon as the turn allowance goes");
  a.feed("t");
  await p2;
  a.stop();
});

Deno.test("timer: the countdown reports the turn allowance first, then the bank", async () => {
  const a = app(200, 200);
  const p = a.awaitDecision(turnObs());
  await sleep(40);
  const early = a.timerSnapshot();
  assert(early.turnLeftMs > 0 && early.turnLeftMs <= 200, `turn ${early.turnLeftMs}`);
  assertEquals(early.bankLeftMs, 200, "bank untouched while the turn allowance remains");

  await sleep(230);
  const late = a.timerSnapshot();
  assertEquals(late.turnLeftMs, 0, "turn allowance exhausted");
  assert(late.bankLeftMs < 200, "bank should now be draining");
  assertEquals(late.overMs, 0, "not yet overtime while the bank holds out");
  a.feed("t");
  await p;
  a.stop();
});
