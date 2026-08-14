// The round-result gate: a 局結果 overlay is dismissed by the player, never by
// a timer. Nothing — not the next decision, not a CPU's paced move, not the
// final standings — may proceed past it until a key arrives.
//
// Also covers the own-seat river line, which shares the same screen real estate
// the result overlay covers.

import { assert, assertEquals } from "@std/assert";
import type { Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import { App } from "../src/tui/app.ts";
import { ownPanel } from "../src/tui/widgets.ts";
import type { Ctx } from "../src/tui/widgets.ts";
import { tileText } from "../src/tui/glyph.ts";
import type { MatchResult } from "../src/match.ts";
import { JANKI } from "../src/rules.ts";
import type { Action, RoundOutcome } from "../src/types.ts";
import type { Observation } from "../src/observe.ts";
import { tiles } from "./helpers.ts";

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

function app(): App {
  return new App({
    glyphs: "ascii",
    aka: JANKI.akaIds,
    names: ["あなた", "CPU東", "CPU南", "CPU西"],
    thinkLimitMs: 3000,
    timerTurnMs: 3_000,
    timerBankMs: 10_000,
    cpuDelayMs: 0,
    cfg: JANKI,
    noIntro: true, // the deal animation must not stand in for the gate
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
    tsumogiriLock: false,
    ukeire: [],
    doraCount: 0,
    furiten: { permanent: false, temporary: false, riichi: false },
    danger: new Map(),
    violations: [0, 0, 0, 0],
    legal,
  };
}

/** 荒牌平局 with the dealer tenpai — the cheapest real `RoundOutcome`. */
function drawOutcome(): RoundOutcome {
  return {
    kind: "ryuukyoku",
    draw: "exhaustive",
    tenpai: [true, false, false, false],
    tenpaiHands: [],
    deltas: [3000, -1000, -1000, -1000],
    dealerRepeat: true,
  };
}

function matchResult(): MatchResult {
  return {
    seed: 1,
    scores: [33000, 29000, 29000, 29000],
    rounds: [],
    outcomes: [drawOutcome()],
    ledger: [],
    ledgerCuts: [0],
    game: {
      version: "2.3",
      rules: { raw: 0, aka: true, kuitan: JANKI.kuitan, sanma: false, hanchan: JANKI.hanchan },
      players: [0, 1, 2, 3].map((seat) => ({ seat, name: `P${seat}` })),
      rounds: [],
    },
  };
}

Deno.test("result gate: the next decision waits for a keypress, not a timer", async () => {
  const a = app();
  a.onEvent({ e: "result", outcome: drawOutcome() });

  const decision = a.awaitDecision(turnObs());
  let paced: number | null = null;
  const pace = a.paceDelay().then((ms) => (paced = ms));

  await sleep(80);
  // The clock only starts once the decision is actually presented, so the timer
  // is the honest witness for "has this decision begun?".
  const presented = a.timerSnapshot().turnLeftMs < 3_000;
  assertEquals(presented, false, "the result overlay must hold the next decision");
  assertEquals(paced, null, "a CPU must not move behind the result overlay");

  a.feed("x"); // any key dismisses 局結果
  await pace;
  assertEquals(paced, 0, "the CPU pace resolves once the overlay is gone");

  await sleep(20);
  a.feed("t"); // ツモ切り — the decision is live again and answerable
  const chosen = await decision;
  assertEquals(chosen.t, "discard");
  a.stop();
});

Deno.test("result gate: the final standings wait behind the last 局結果", async () => {
  const a = app();
  a.onEvent({ e: "result", outcome: drawOutcome() });

  let done = false;
  const final = a.showFinal(matchResult()).then(() => (done = true));

  await sleep(60);
  assertEquals(done, false, "最終結果 must not clobber the deciding hand's result");

  a.feed("x"); // dismiss 局結果 — now the standings go up
  await sleep(20);
  assertEquals(done, false, "the standings themselves still need a key");

  a.feed("x"); // dismiss 最終結果
  await final;
  assertEquals(done, true);
  a.stop();
});

// ---------------------------------------------------------------------------

function river(spec: string): RiverEntry[] {
  return tiles(spec).map((tile, i) => ({
    tile,
    junme: i + 1,
    tsumogiri: false,
    riichiDeclare: false,
  }));
}

function ctxWith(entries: RiverEntry[]): Ctx {
  const o = turnObs();
  return {
    obs: { ...o, rivers: [entries, [], [], []] },
    glyph: { mode: "ascii", aka: JANKI.akaIds },
    names: ["あなた", "CPU東", "CPU南", "CPU西"],
    slots: [...o.hand],
    drawnIndex: o.hand.length - 1,
    cursor: 0,
    selectable: o.hand.map(() => true),
    riichiArmed: false,
    phase: "turn",
    timer: { turnMs: 3000, bankMs: 10000, turnLeftMs: 3000, bankLeftMs: 10000, overMs: 0 },
    ledger: [],
    log: [],
    message: "",
    riverRows: 2,
    claim: null,
  };
}

Deno.test("own panel: the player's own river is rendered above the hand", () => {
  const entries = river("東1p9s");
  const c = ctxWith(entries);
  const lines = ownPanel(c, 78);

  assertEquals(lines.length, 4, "head, river, hand, caret");
  const text = lines[1].map((s) => s.text).join("");
  assert(text.startsWith("河"), `river line should carry the 河 label: ${text}`);
  for (const e of entries) {
    assert(
      text.includes(tileText(e.tile, c.glyph)),
      `${tileText(e.tile, c.glyph)} missing from the own river: ${text}`,
    );
  }
  // The hand must still be its own line, one row below.
  assert(lines[2].map((s) => s.text).join("").startsWith("手牌"));
});

// ---------------------------------------------------------------------------
// 手牌公開: the result overlay reveals every seat's hand, not just the winner's.

Deno.test("result overlay: 手牌公開 shows all four hands when the event carries them", () => {
  const a = app();
  const hands = [
    tiles("123456789m123p9p"),
    tiles("11122233344455s"),
    tiles("東南西北白發中東南西"),
    tiles("456789m456789p"),
  ];
  a.onEvent({
    e: "riichi",
    who: 1,
    step: 1,
  });
  a.onEvent({
    e: "result",
    outcome: drawOutcome(),
    hands,
    melds: [[], [], [], []],
  });

  const lines = a.overlayLines();
  assert(lines, "a 局結果 overlay must be up");
  const head = lines.indexOf("手牌公開");
  assert(head >= 0, `no 手牌公開 header in the overlay: ${JSON.stringify(lines)}`);

  const seatLines = lines.slice(head + 1, head + 5);
  assertEquals(seatLines.length, 4, "one revealed line per seat");
  for (let s = 0; s < 4; s++) {
    const text = seatLines[s];
    assert(text.startsWith(`P${s}`), `seat ${s} line should be labelled: ${text}`);
    for (const t of hands[s]) {
      assert(
        text.includes(tileText(t, { mode: "ascii", aka: JANKI.akaIds })),
        `${tileText(t, { mode: "ascii", aka: JANKI.akaIds })} missing from P${s}: ${text}`,
      );
    }
  }
  // The declaring seat is called out in words; the reveal sits above 点棒移動.
  assert(seatLines[1].includes("リーチ"), `P1 declared riichi: ${seatLines[1]}`);
  assert(!seatLines[0].includes("リーチ"), `P0 did not: ${seatLines[0]}`);
  assert(
    lines.slice(head).some((l) => l.startsWith("点棒移動")),
    "点棒移動 must still follow the reveal",
  );
  a.stop();
});

Deno.test("result overlay: no 手牌公開 block at all when the event omits hands", () => {
  const a = app();
  a.onEvent({ e: "result", outcome: drawOutcome() });

  const lines = a.overlayLines();
  assert(lines, "a 局結果 overlay must be up");
  assertEquals(lines.indexOf("手牌公開"), -1, "an old-shape event reveals nothing");
  assert(lines.some((l) => l.startsWith("点棒移動")), "the rest of the overlay is unchanged");
  a.stop();
});

Deno.test("result overlay: melds are rendered alongside the concealed tiles", () => {
  const a = app();
  const pon = tiles("111p");
  a.onEvent({
    e: "result",
    outcome: drawOutcome(),
    hands: [tiles("123456789m12p"), tiles("2345678999m12p"), tiles("11122233344455s"), []],
    melds: [
      [],
      [],
      [],
      [{ kind: "pon", who: 3, fromWho: 0, calledTile: pon[0], tiles: pon }],
    ],
  });

  const lines = a.overlayLines()!;
  const seatLines = lines.slice(lines.indexOf("手牌公開") + 1, lines.indexOf("手牌公開") + 5);
  const glyph = { mode: "ascii", aka: JANKI.akaIds } as const;
  assert(
    seatLines[3].includes("[" + pon.map((t) => tileText(t, glyph)).join("") + "]"),
    `P3's pon should be bracketed on its line: ${seatLines[3]}`,
  );
  a.stop();
});
