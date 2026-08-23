// 手牌価値の較正記録 (M11 unit C) — the writer, its labels, and the reader.
//
// Three claims, and the tests are grouped by them:
//   1. a sample survives the round trip UNCHANGED — the facts a policy handed
//      the writer are the facts a fit reads back, field for field, and the
//      model's own answer comes back exactly (that is what lets the fit prove
//      it has not forked from the play path);
//   2. the LABELS say what happened: a win, a 放銃, and a 流局 are three
//      different rows, and the buffering means every sample of one round wears
//      that round's outcome and no other's;
//   3. a file from another version is REFUSED, never read with the missing
//      columns filled in.

import { assertEquals, assertThrows } from "@std/assert";
import type { Meld, Tile } from "mjrender/model.ts";
import type { HandFacts } from "../src/ai/handvalue.ts";
import { DEFAULT_HAND, handOutlook } from "../src/ai/handvalue.ts";
import type { HandSample } from "../src/ai/handcalib.ts";
import {
  HAND_CALIB_KIND,
  HAND_CALIB_VERSION,
  HandCalibrationWriter,
  parseHandCalibration,
  scanHandCalibration,
} from "../src/ai/handcalib.ts";
import { sfc32 } from "../src/rng.ts";
import { JANKI } from "../src/rules.ts";
import { Table } from "../src/table.ts";
import type { RoundOutcome, Seat, WinInfo } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { Wall } from "../src/wall.ts";

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

function facts(o: Partial<HandFacts> = {}): HandFacts {
  return {
    shanten: 1,
    ukeire: 12,
    ukeireTypes: 3,
    unseenTotal: 100,
    turnsLeft: 12,
    junme: 6,
    dora: 1,
    open: 0,
    closed: true,
    riichi: false,
    yakuhaiTriplets: 0,
    yakuhaiPairs: 1,
    honitsu: false,
    ronnable: true,
    furiten: false,
    dealer: false,
    oppTenpai: [0, 0.5, 1],
    honba: 0,
    kyotaku: 0,
    ...o,
  };
}

/** A sample the way the policy makes one: the facts, and the model's answer. */
function sampleOf(f: HandFacts): HandSample {
  const { pwin, value } = handOutlook(f, DEFAULT_HAND);
  return { facts: f, pwin, value };
}

function makeTable(kyoku = 0, honba = 0): Table {
  return new Table(
    {
      kyoku,
      honba,
      kyotaku: 0,
      dealer: (kyoku % 4) as Seat,
      scores: [25000, 25000, 25000, 25000],
      wall: Wall.shuffled(sfc32(11)),
      dice: [0, 0],
    },
    JANKI,
    SEATS.map((seat) => ({ seat, name: `P${seat}` })),
  );
}

/** Put `n` discards in seat 0's river — the writer reads its length as 巡目. */
function fillRiver(t: Table, seat: Seat, n: number): void {
  for (let i = 0; i < n; i++) {
    t.board.rivers[seat].push({
      tile: i as Tile,
      junme: i + 1,
      tsumogiri: false,
      riichiDeclare: false,
    });
  }
}

function win(who: Seat, fromWho: Seat): WinInfo {
  return {
    who,
    fromWho,
    winTile: 0 as Tile,
    han: 3,
    fu: 30,
    points: 3900,
    limit: 0,
    yaku: [],
    yakuman: [],
    doraIndicators: [],
    uraIndicators: [],
    hand: [] as Tile[],
    melds: [] as Meld[],
  };
}

function agari(wins: WinInfo[], deltas: number[]): RoundOutcome {
  return { kind: "agari", wins, deltas, dealerRepeat: false };
}

function ryuukyoku(deltas: number[]): RoundOutcome {
  return {
    kind: "ryuukyoku",
    draw: "exhaustive",
    tenpai: [true, false, false, false],
    tenpaiHands: [],
    deltas,
    dealerRepeat: false,
  };
}

function tmpPath(): string {
  return Deno.makeTempFileSync({ prefix: "mjgame-hand-", suffix: ".jsonl" });
}

// ---------------------------------------------------------------------------
// 1. round trip
// ---------------------------------------------------------------------------

Deno.test("handcalib: header and samples survive the round trip", () => {
  const path = tmpPath();
  const w = new HandCalibrationWriter(path, {
    seats: "khhh",
    seed: 4242,
    games: 2,
    w: DEFAULT_HAND,
  });
  const a = sampleOf(facts({ shanten: 2, ukeire: 20 }));
  const b = sampleOf(facts({ shanten: 0, ukeire: 4, junme: 9, dealer: true }));
  w.beginGame(4242);
  w.record(a);
  w.record(b);
  const t = makeTable(1, 2);
  fillRiver(t, 0, 11);
  w.endRound(t, agari([win(0, 2)], [8000, 0, -8000, 0]));
  w.close();

  const { header, records } = parseHandCalibration(Deno.readTextFileSync(path), path);
  assertEquals(header.v, HAND_CALIB_VERSION);
  assertEquals(header.kind, HAND_CALIB_KIND);
  assertEquals(header.seats, "khhh");
  assertEquals(header.seed, 4242);
  assertEquals(header.games, 2);
  assertEquals(header.w, DEFAULT_HAND);
  assertEquals(records.length, 2);

  // The facts come back FIELD FOR FIELD — the fit re-enters `handOutlook` with
  // them, so a dropped or renamed column would silently change the model.
  assertEquals(records[0].facts, a.facts);
  assertEquals(records[1].facts, b.facts);
  // …and so does the answer the live seat got, which is the reproduction check
  // the fit and the report both lean on.
  assertEquals(records[0].pwin, a.pwin);
  assertEquals(records[0].value, a.value);
  assertEquals(
    records[1].pwin,
    handOutlook(records[1].facts, header.w).pwin,
  );
  // Decision index runs over the whole GAME, in play order; the seed is stamped
  // by the writer because the policy has no business knowing it.
  assertEquals(records.map((r) => r.n), [0, 1]);
  assertEquals(records.map((r) => r.s), [4242, 4242]);
  assertEquals(w.stats(), { games: 1, rows: 2, dropped: 0 });

  Deno.removeSync(path);
});

Deno.test("handcalib: scan streams the same records the parser returns", async () => {
  const path = tmpPath();
  const w = new HandCalibrationWriter(path, { seats: "hhhh", seed: 1, games: 1, w: DEFAULT_HAND });
  w.beginGame(1);
  w.record(sampleOf(facts()));
  w.record(sampleOf(facts({ shanten: 3 })));
  const t = makeTable();
  fillRiver(t, 0, 7);
  w.endRound(t, ryuukyoku([1500, -1500, -1500, 1500]));
  w.close();

  const seen: number[] = [];
  const header = await scanHandCalibration(path, (rec) => seen.push(rec.facts.shanten));
  assertEquals(header.seats, "hhhh");
  assertEquals(seen, [1, 3]);
  Deno.removeSync(path);
});

// ---------------------------------------------------------------------------
// 2. labels
// ---------------------------------------------------------------------------

Deno.test("handcalib: labels — 和了 / 放銃 / 流局", () => {
  const path = tmpPath();
  const w = new HandCalibrationWriter(path, { seats: "khhh", seed: 7, games: 1, w: DEFAULT_HAND });
  w.beginGame(7);

  // 東1: we win off seat 2. `winPoints` is the round's real settlement, 本場 and
  // 供託 included — which is exactly what the value model claims to predict.
  w.record(sampleOf(facts()));
  const t1 = makeTable(0, 1);
  fillRiver(t1, 0, 9);
  w.endRound(t1, agari([win(0, 2)], [8300, 0, -8300, 0]));

  // 東2: seat 1 rons OUR discard.
  w.record(sampleOf(facts({ shanten: 2 })));
  w.record(sampleOf(facts({ shanten: 1 })));
  const t2 = makeTable(1, 0);
  fillRiver(t2, 0, 14);
  w.endRound(t2, agari([win(1, 0)], [-5800, 5800, 0, 0]));

  // 東3: 流局 — neither label fires, and `winPoints` stays 0 even though the
  // 聴牌料 moved the seat's score.
  w.record(sampleOf(facts({ shanten: 0 })));
  const t3 = makeTable(2, 0);
  fillRiver(t3, 0, 18);
  w.endRound(t3, ryuukyoku([3000, -1000, -1000, -1000]));
  w.close();

  const { records } = parseHandCalibration(Deno.readTextFileSync(path), path);
  assertEquals(records.length, 4);
  assertEquals(records.map((r) => r.k), [0, 1, 1, 2]);
  assertEquals(records.map((r) => r.b), [1, 0, 0, 0]);
  assertEquals(records.map((r) => r.n), [0, 1, 2, 3]);
  assertEquals(records.map((r) => r.won), [1, 0, 0, 0]);
  assertEquals(records.map((r) => r.winPoints), [8300, 0, 0, 0]);
  assertEquals(records.map((r) => r.dealtIn), [0, 1, 1, 0]);
  assertEquals(records.map((r) => r.endJunme), [9, 14, 14, 18]);
  assertEquals(records.map((r) => r.outcome), ["agari", "agari", "agari", "ryuukyoku"]);

  Deno.removeSync(path);
});

Deno.test("handcalib: a tsumo by another seat is nobody's 放銃, a double ron is still a win", () => {
  const path = tmpPath();
  const w = new HandCalibrationWriter(path, { seats: "khhh", seed: 9, games: 1, w: DEFAULT_HAND });
  w.beginGame(9);

  // `fromWho === who` is a tsumo: seat 0 pays, but it did not deal in.
  w.record(sampleOf(facts()));
  const t1 = makeTable(0, 0);
  fillRiver(t1, 0, 6);
  w.endRound(t1, agari([win(2, 2)], [-2000, -1000, 4000, -1000]));

  // Double ron off seat 3 with seat 0 among the winners: the seat won the round.
  w.record(sampleOf(facts()));
  const t2 = makeTable(1, 0);
  fillRiver(t2, 0, 12);
  w.endRound(t2, agari([win(0, 3), win(1, 3)], [3900, 2000, 0, -5900]));
  w.close();

  const { records } = parseHandCalibration(Deno.readTextFileSync(path), path);
  assertEquals(records.map((r) => r.won), [0, 1]);
  assertEquals(records.map((r) => r.dealtIn), [0, 0]);
  assertEquals(records.map((r) => r.winPoints), [0, 3900]);
  Deno.removeSync(path);
});

Deno.test("handcalib: a game boundary drops unlabelled samples rather than mislabelling them", () => {
  const path = tmpPath();
  const w = new HandCalibrationWriter(path, { seats: "khhh", seed: 1, games: 2, w: DEFAULT_HAND });
  w.beginGame(1);
  w.record(sampleOf(facts()));
  // No `endRound`: the round never finished, so there is no truth to stamp.
  w.beginGame(2);
  w.record(sampleOf(facts({ shanten: 3 })));
  const t = makeTable();
  fillRiver(t, 0, 5);
  w.endRound(t, ryuukyoku([0, 0, 0, 0]));
  w.close();

  const { records } = parseHandCalibration(Deno.readTextFileSync(path), path);
  assertEquals(records.length, 1);
  assertEquals(records[0].s, 2);
  // The index restarts with the game, so `n` is a position within one hanchan.
  assertEquals(records[0].n, 0);
  assertEquals(w.stats(), { games: 2, rows: 1, dropped: 1 });
  Deno.removeSync(path);
});

// ---------------------------------------------------------------------------
// 3. version discipline
// ---------------------------------------------------------------------------

Deno.test("handcalib: a foreign or future file is refused, never coerced", () => {
  const line = (h: unknown) => JSON.stringify(h) + "\n";
  assertThrows(
    () => parseHandCalibration(line({ v: 2, kind: HAND_CALIB_KIND, w: DEFAULT_HAND })),
    Error,
    "版が違います",
  );
  assertThrows(
    () => parseHandCalibration(line({ v: 1, kind: "mjgame-calib", w: DEFAULT_HAND })),
    Error,
    "手牌価値の較正記録ではありません",
  );
  assertThrows(() => parseHandCalibration(""), Error, "空のファイルです");
});
