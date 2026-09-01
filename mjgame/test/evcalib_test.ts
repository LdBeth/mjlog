// EV核の較正記録 (M15b) — the writer, its labels, its reader, and the two
// refusals the whole fit rests on.
//
// Five claims, and the tests are grouped by them:
//   1. a WIRE survives the round trip UNCHANGED — the 528 numbers the policy
//      packed are the 528 numbers a fit reads back, element for element, and
//      the four predictions stored beside them come back exactly;
//   2. the REPRODUCTION holds: replaying a stored wire through a fresh core
//      built from the file's OWN header returns those four numbers bit for bit.
//      That is the property `scripts/ev_fit.ts` refuses to start without, and
//      it is what makes a fitted vector mean the same thing to the fit and at
//      the table;
//   3. the LABELS say what happened — a win, a 放銃, an opponent's win and a
//      流局 are four different rows, `tenpaiEnd` is judged the way the 局 ended
//      rather than by a single rule, and the buffering means every sample of
//      one 局 wears that 局's outcome and no other's;
//   4. the CLI REFUSES what would poison the lane: a non-"k" seat 0, `--jobs`,
//      a driver that does not play whole rounds, and — the rule the M11 lesson
//      bought — a ktune carrying an `ev` block;
//   5. the recorder is an OBSERVER: `paired` strips it from the control arm,
//      and a run with the lane plays the same 半荘 as a run without it.

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { Meld, Tile } from "mjrender/model.ts";
import { argError } from "../src/cli/args.ts";
import type { ArgCheck } from "../src/cli/args.ts";
import { buildEv, closeEv, evNative } from "../src/ai/ev.ts";
import {
  D_COSTIN,
  D_EXPLOSS,
  D_GAIN,
  D_PIN,
  D_RISK,
  D_TENPAI,
  DBLS_LEN,
  EV_ABI,
  I_CLOSED,
  I_HAND,
  I_JUNME,
  I_MODE,
  I_ROUND_WIND,
  I_SEAT_WIND,
  I_T,
  I_UNSEEN,
  INTS_LEN,
} from "../src/ai/evlayout.ts";
import { DEFAULT_EV, mergeEv } from "../src/ai/evparams.ts";
import type { EvSample } from "../src/ai/evcalib.ts";
import {
  EV_CALIB_ACCEPTED,
  EV_CALIB_KIND,
  EV_CALIB_VERSION,
  evaluateWire,
  EvCalibrationWriter,
  evEngineHash,
  evLabels,
  parseEvCalibration,
  scanEvCalibration,
  tenpaiAtEnd,
} from "../src/ai/evcalib.ts";
import { headless, headlessParallel } from "../src/harness.ts";
import type { KTune } from "../src/harness.ts";
import { pairedRun } from "../src/paired.ts";
import { sfc32 } from "../src/rng.ts";
import { JANKI } from "../src/rules.ts";
import { Table } from "../src/table.ts";
import type { RoundOutcome, Seat, WinInfo } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { Wall } from "../src/wall.ts";
import { reproVerdict } from "../scripts/ev_fit.ts";
import { tiles } from "./helpers.ts";

const CHAMPION: KTune = JSON.parse(
  Deno.readTextFileSync(new URL("../weights/champion.json", import.meta.url).pathname),
);

/**
 * Can this machine build a core? Decided ONCE, by trying — the writer OWNS an
 * `EvCore`, so every test that opens one is gated on the dylib exactly as
 * `ev_wiring_test` gates its own.
 */
const SKIP = (() => {
  try {
    const c = buildEv(mergeEv({}));
    closeEv(c);
    return false;
  } catch (e) {
    console.log(
      `[M15b] libmjev が使えないので EV レーンのテストの一部を飛ばします ` +
        `(evNative=${evNative()}): ${e instanceof Error ? e.message.split("\n")[0] : e}`,
    );
    return true;
  }
})();

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

/**
 * A hand-built rest wire. Not a realistic board — the point of these tests is
 * that WHATEVER 528 numbers go in come back out and evaluate the same way — but
 * a legal one: mode 1, a 13-tile hand, a plausible pool and a horizon.
 */
function wireOf(hand: string, junme: number, T: number): EvSample {
  const ints = new Int32Array(INTS_LEN);
  const dbls = new Float64Array(DBLS_LEN);
  const counts = new Array(34).fill(0);
  for (const t of tiles(hand)) counts[t >> 2]++;
  for (let i = 0; i < 34; i++) ints[I_HAND + i] = counts[i];
  ints[I_SEAT_WIND] = 28; // 南
  ints[I_ROUND_WIND] = 27; // 東
  ints[I_JUNME] = junme;
  ints[I_T] = T;
  ints[I_CLOSED] = 1;
  for (let i = 0; i < 34; i++) ints[I_UNSEEN + i] = 4 - counts[i];
  ints[I_MODE] = 1;
  // The policy half: three quiet opponents and a plain danger row.
  dbls[D_TENPAI + 0] = 0.1;
  dbls[D_TENPAI + 1] = 0.2;
  dbls[D_TENPAI + 2] = 0.05;
  dbls[D_EXPLOSS + 0] = 5200;
  dbls[D_EXPLOSS + 1] = 4800;
  dbls[D_EXPLOSS + 2] = 6100;
  for (let i = 0; i < 34; i++) {
    dbls[D_PIN + i] = 0.001 * (i + 1);
    dbls[D_COSTIN + i] = 100 * (i % 7);
  }
  dbls[D_GAIN] = 1;
  dbls[D_RISK] = 1;
  return { ints, dbls, shanten: 2 };
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
    hand: [],
    melds: [] as Meld[],
  };
}

const agari = (wins: WinInfo[], deltas: number[]): RoundOutcome => ({
  kind: "agari",
  wins,
  deltas,
  dealerRepeat: false,
});

const ryuukyoku = (tenpai: boolean[], deltas: number[]): RoundOutcome => ({
  kind: "ryuukyoku",
  draw: "exhaustive",
  tenpai,
  tenpaiHands: [],
  deltas,
  dealerRepeat: false,
});

function tmpPath(): string {
  return Deno.makeTempFileSync({ prefix: "mjgame-ev-", suffix: ".jsonl" });
}

const header = (seed: number, games: number) => ({
  seats: "khhh",
  seed,
  games,
  ev: mergeEv({}),
});

/**
 * The same header with the DP wound right down. The two whole-run tests below
 * are about WIRING — does the sink reach seat 0, does the control arm stay out
 * of the file — and the writer evaluates every wire it is handed, so at
 * `DEFAULT_EV` they spend ten seconds each computing numbers nothing asserts.
 * The tests that do assert on the numbers (round trip, reproduction) keep the
 * real parameters.
 */
const fastHeader = (seed: number, games: number) => ({
  seats: "khhh",
  seed,
  games,
  ev: mergeEv({ exactShanten: 0, sameShantenRungs: 0, maxNodes: 200 }),
});

// ---------------------------------------------------------------------------
// 1. round trip
// ---------------------------------------------------------------------------

Deno.test({
  name: "evcalib: wire と予測が往復して変わらない",
  ignore: SKIP,
  fn: () => {
    const path = tmpPath();
    const w = new EvCalibrationWriter(path, header(4, 1));
    const s = wireOf("123m456p789s東東南南", 5, 12);
    w.beginGame(4);
    w.record(s);
    const t = makeTable();
    fillRiver(t, 0, 11);
    w.endRound(t, ryuukyoku([true, false, false, false], [1500, -500, -500, -500]));
    w.close();

    const { header: h, records } = parseEvCalibration(Deno.readTextFileSync(path), path);
    assertEquals(h.v, EV_CALIB_VERSION);
    assertEquals(h.kind, EV_CALIB_KIND);
    assertEquals(h.evAbi, EV_ABI);
    assertEquals(h.ev, DEFAULT_EV);
    // The ENGINE's identity, beside the wire's: the ABI says whether the file
    // can be read, the hash says whether its stored predictions still belong to
    // the DP on disk.
    assertEquals(h.engineHash, evEngineHash());
    assertEquals(records.length, 1);

    const r = records[0];
    assertEquals(r.ints.length, INTS_LEN);
    assertEquals(r.dbls.length, DBLS_LEN);
    // The INTEGER half is stored verbatim; the REAL half is rounded to six
    // significant digits BEFORE the writer evaluated it, so the file's numbers
    // and the evaluated numbers are one and the same (claim 2 below depends on
    // exactly this).
    for (let i = 0; i < INTS_LEN; i++) assertEquals(r.ints[i], s.ints[i], `ints[${i}]`);
    for (let i = 0; i < DBLS_LEN; i++) {
      const want = s.dbls[i] === 0 ? 0 : Number(s.dbls[i].toPrecision(6));
      assertEquals(r.dbls[i], want, `dbls[${i}]`);
    }
    // The bucket keys the report groups by, carried out of the wire.
    assertEquals(r.junme, 5);
    assertEquals(r.T, 12);
    assertEquals(r.sh, 2);
    assert(Number.isFinite(r.pW) && r.pW >= 0 && r.pW <= 1, `pW=${r.pW}`);
    assert(Number.isFinite(r.pT) && r.pT >= 0 && r.pT <= 1, `pT=${r.pT}`);

    Deno.removeSync(path);
  },
});

Deno.test({
  name: "evcalib: scan は parse と同じ記録を流す",
  ignore: SKIP,
  fn: async () => {
    const path = tmpPath();
    const w = new EvCalibrationWriter(path, header(6, 1));
    w.beginGame(6);
    w.record(wireOf("123m456p789s東東南南", 3, 15));
    w.record(wireOf("111m456p789s東東南南", 4, 14));
    const t = makeTable();
    fillRiver(t, 0, 9);
    w.endRound(t, agari([win(2, 2)], [-1000, -1000, 3000, -1000]));
    w.close();

    const seen: unknown[] = [];
    const h = await scanEvCalibration(path, (rec) => seen.push(rec));
    assertEquals(h.evAbi, EV_ABI);
    assertEquals(seen, parseEvCalibration(Deno.readTextFileSync(path), path).records);
    Deno.removeSync(path);
  },
});

// ---------------------------------------------------------------------------
// 2. reproduction — the property the fit refuses to start without
// ---------------------------------------------------------------------------

Deno.test({
  name: "evcalib: 保存した wire をヘッダの母数で再評価すると予測がビット単位で一致する",
  ignore: SKIP,
  fn: () => {
    const path = tmpPath();
    const w = new EvCalibrationWriter(path, header(8, 1));
    w.beginGame(8);
    for (const spec of ["123m456p789s東東南南", "111m234p567s99m東東", "19m19p19s東南西北白發中"]) {
      w.record(wireOf(spec, 7, 9));
    }
    const t = makeTable();
    fillRiver(t, 0, 13);
    w.endRound(t, agari([win(0, 1)], [7700, -7700, 0, 0]));
    w.close();

    const { header: h, records } = parseEvCalibration(Deno.readTextFileSync(path), path);
    // A DIFFERENT core than the writer's, built from the file's own header —
    // which is exactly what `ev_fit.ts` does before it takes a single step.
    const core = buildEv(mergeEv(h.ev));
    try {
      for (const r of records) {
        const p = evaluateWire(core, r.ints, r.dbls);
        assertEquals(p.pT, r.pT);
        assertEquals(p.pW, r.pW);
        assertEquals(p.eV, r.eV);
        assertEquals(p.eCost, r.eCost);
        assertEquals(p.value, r.value);
      }
    } finally {
      closeEv(core);
    }
    Deno.removeSync(path);
  },
});

Deno.test({
  name: "evcalib: 母数を動かすと予測も動く (当てはめるものが本当にある)",
  ignore: SKIP,
  fn: () => {
    const s = wireOf("123m456p789s東東南南", 6, 10);
    const a = buildEv(mergeEv({}));
    const b = buildEv(mergeEv({ ronFactor: 0.15, oppHazard: 0.28 }));
    try {
      const pa = evaluateWire(a, s.ints, s.dbls);
      const pb = evaluateWire(b, s.ints, s.dbls);
      assert(pa.pW !== pb.pW, `pW が母数に反応していない (${pa.pW})`);
    } finally {
      closeEv(a);
      closeEv(b);
    }
  },
});

// ---------------------------------------------------------------------------
// 3. labels
// ---------------------------------------------------------------------------

Deno.test("evcalib: labels — 和了 / 放銃 / 他家和了 / 流局", () => {
  const t = makeTable(1, 2);
  fillRiver(t, 0, 14);

  const wonL = evLabels(t, agari([win(0, 3)], [8300, 0, 0, -8300]));
  assertEquals([wonL.won, wonL.dealtIn, wonL.oppWon], [1, 0, 0]);
  assertEquals(wonL.winPoints, 8300);
  assertEquals(wonL.dealtInPoints, 0);
  assertEquals(wonL.tenpaiEnd, 1);
  assertEquals([wonL.k, wonL.b, wonL.endJunme], [1, 2, 14]);

  const dealt = evLabels(t, agari([win(1, 0)], [-5800, 5800, 0, 0]));
  assertEquals([dealt.won, dealt.dealtIn, dealt.oppWon], [0, 1, 0]);
  assertEquals(dealt.winPoints, 0);
  // The label is a POSITIVE magnitude — `eCost` is a cost, and the audit table
  // compares the two directly.
  assertEquals(dealt.dealtInPoints, 5800);

  // A tsumo by somebody else is nobody's 放銃, but it IS an opponent ending the
  // hand — which is the event `oppHazard`/`oppGrowth` price.
  const opp = evLabels(t, agari([win(2, 2)], [-2000, -1000, 4000, -1000]));
  assertEquals([opp.won, opp.dealtIn, opp.oppWon], [0, 0, 1]);
  assertEquals(opp.dealtInPoints, 0);

  const draw = evLabels(t, ryuukyoku([true, false, false, false], [3000, -1000, -1000, -1000]));
  assertEquals([draw.won, draw.dealtIn, draw.oppWon], [0, 0, 0]);
  assertEquals(draw.outcome, "ryuukyoku");
  assertEquals(draw.tenpaiEnd, 1);
  assertEquals(
    evLabels(t, ryuukyoku([false, true, true, true], [-3000, 1000, 1000, 1000])).tenpaiEnd,
    0,
  );
});

Deno.test("evcalib: 二人和了で席0が含まれれば勝ち、放銃点は取らない", () => {
  const t = makeTable();
  fillRiver(t, 0, 8);
  const l = evLabels(t, agari([win(0, 3), win(1, 3)], [3900, 2000, 0, -5900]));
  assertEquals([l.won, l.dealtIn, l.oppWon], [1, 0, 0]);
  assertEquals(l.winPoints, 3900);
});

Deno.test("evcalib: tenpaiEnd — 他家が決めた局は席0の手から判定する", () => {
  const t = makeTable();
  fillRiver(t, 0, 10);
  const out = agari([win(2, 1)], [0, -3900, 3900, 0]);

  // 4面子ができていて 4m の単騎待ち — 立派な聴牌。
  t.board.hands[0] = tiles("456m123p789p345s4m");
  assertEquals(t.board.hands[0].length, 13);
  assertEquals(tenpaiAtEnd(t, out), 1);

  // 同じ13枚でも一枚差し替えれば 1向聴 — 456m/123p の二面子に 78p・34s・99s
  // が残る形で、札は 0 になる。
  t.board.hands[0] = tiles("1m456m123p78p34s99s");
  assertEquals(t.board.hands[0].length, 13);
  assertEquals(tenpaiAtEnd(t, out), 0);
});

Deno.test({
  name: "evcalib: 半荘の切れ目は札の無い標本を捨てる",
  ignore: SKIP,
  fn: () => {
    const path = tmpPath();
    const w = new EvCalibrationWriter(path, header(1, 2));
    w.beginGame(1);
    w.record(wireOf("123m456p789s東東南南", 2, 16));
    // No `endRound`: the 局 never finished, so there is no truth to stamp.
    w.beginGame(2);
    w.record(wireOf("123m456p789s東東南南", 3, 15));
    const t = makeTable();
    fillRiver(t, 0, 5);
    w.endRound(t, ryuukyoku([false, false, false, false], [0, 0, 0, 0]));
    w.close();

    const { records } = parseEvCalibration(Deno.readTextFileSync(path), path);
    assertEquals(records.length, 1);
    assertEquals(records[0].s, 2);
    assertEquals(records[0].n, 0);
    const st = w.stats();
    assertEquals([st.games, st.rows, st.dropped], [2, 1, 1]);
    Deno.removeSync(path);
  },
});

// ---------------------------------------------------------------------------
// 4. version / ABI discipline
// ---------------------------------------------------------------------------

Deno.test("evcalib: 別版・別ABI・別種のファイルは読まずに拒否する", () => {
  const line = (h: unknown) => JSON.stringify(h) + "\n";
  assertThrows(
    () => parseEvCalibration(line({ v: 9, kind: EV_CALIB_KIND, ev: DEFAULT_EV, evAbi: EV_ABI })),
    Error,
    "版が違います",
  );
  // v1 (no `engineHash`) is a strict SUBSET of v2 and stays readable — the
  // calibration v2→v3 rule. Its predictions simply have no engine to check.
  assertEquals(EV_CALIB_ACCEPTED.has(1), true);
  assertEquals(EV_CALIB_ACCEPTED.has(EV_CALIB_VERSION), true);
  assertEquals(
    parseEvCalibration(line({ v: 1, kind: EV_CALIB_KIND, ev: DEFAULT_EV, evAbi: EV_ABI }))
      .header.engineHash,
    undefined,
  );
  assertThrows(
    () => parseEvCalibration(line({ v: 1, kind: "mjgame-hand", ev: DEFAULT_EV, evAbi: EV_ABI })),
    Error,
    "EV核の較正記録ではありません",
  );
  assertThrows(
    () =>
      parseEvCalibration(line({ v: 1, kind: EV_CALIB_KIND, ev: DEFAULT_EV, evAbi: EV_ABI + 1 })),
    Error,
    "ABI が違います",
  );
  assertThrows(() => parseEvCalibration(""), Error, "空のファイルです");
});

// ---------------------------------------------------------------------------
// 5. the CLI refusals
// ---------------------------------------------------------------------------

function args(o: Partial<ArgCheck>): ArgCheck {
  return { cmd: "selfplay", seats: "khhh", calibrate: "", evcalib: "runs/ev/x.jsonl", ...o };
}

Deno.test("evcalib: --evcalib は selfplay / paired の k席0・1スレッドに限る", () => {
  assertEquals(argError(args({})), null);
  assertEquals(argError(args({ cmd: "paired" })), null);
  const play = argError(args({ cmd: "play" }));
  assert(play?.includes("selfplay / paired 専用"), play ?? "(no error)");
  const h0 = argError(args({ seats: "hhhh" }));
  assert(h0?.includes("k席"), h0 ?? "(no error)");
  const jobs = argError(args({ jobs: 4 }));
  assert(jobs?.includes("--jobs"), jobs ?? "(no error)");
});

Deno.test("evcalib: --evcalib と ev ブロックは併用できない (レーンは素の席で録る)", () => {
  // THE RULE THE FIT RESTS ON. A lane recorded under the DP would be censored
  // by the DP's own folds — `handvalue.ts` measured the refit on such a lane at
  // +0.11 WORSE — so the flag refuses ANY `ev` section, sub-switches included.
  for (const ev of [{}, { riichi: false, calls: false }, { discard: false }]) {
    const e = argError(args({ ktune: { ...CHAMPION, ev } }));
    assert(e?.includes("ev ブロック"), `${JSON.stringify(ev)}: ${e ?? "(no error)"}`);
  }
  // …and the plain champion, which carries none, is accepted.
  assertEquals(argError(args({ ktune: CHAMPION })), null);
});

// ---------------------------------------------------------------------------
// 6. the recorder is an observer
// ---------------------------------------------------------------------------

Deno.test({
  name: "evcalib: レーンを録っても打牌は変わらない (khhh + champion)",
  ignore: SKIP,
  fn: () => {
    const path = tmpPath();
    const plain = headless(3, 20250830, "khhh", { ktune: CHAMPION });
    const writer = new EvCalibrationWriter(path, fastHeader(20250830, 3));
    let recorded: number;
    try {
      const taped = headless(3, 20250830, "khhh", { ktune: CHAMPION, evCalib: writer });
      assertEquals(taped.results, plain.results);
    } finally {
      writer.close();
      recorded = writer.stats().rows;
    }
    assert(recorded > 100, `記録が少なすぎる: ${recorded}行`);
    Deno.removeSync(path);
  },
});

Deno.test({
  name: "evcalib: paired の対照腕はレーンを持たない (A腕だけが録る)",
  ignore: SKIP,
  fn: () => {
    // BOTH arms are `khhh` on the champion (`ktuneB` makes the control an
    // INCUMBENT rather than the plain `hhhh` baseline), so nothing but the
    // structural strip in `pairedRun` can keep the control arm from recording.
    // If it did, the file would carry two arms' decisions under one seed.
    const solo = tmpPath();
    const both = tmpPath();
    const wSolo = new EvCalibrationWriter(solo, fastHeader(31337, 2));
    let soloRows = 0;
    try {
      headless(2, 31337, "khhh", { ktune: CHAMPION, evCalib: wSolo });
    } finally {
      wSolo.close();
      soloRows = wSolo.stats().rows;
    }
    const wBoth = new EvCalibrationWriter(both, fastHeader(31337, 2));
    let bothRows = 0;
    try {
      pairedRun(2, 31337, "khhh", { ktune: CHAMPION, ktuneB: CHAMPION, evCalib: wBoth });
    } finally {
      wBoth.close();
      bothRows = wBoth.stats().rows;
    }
    assert(soloRows > 0, "A腕が何も録っていない");
    assertEquals(bothRows, soloRows);
    Deno.removeSync(solo);
    Deno.removeSync(both);
  },
});

Deno.test({
  name: "evcalib: --jobs では分割できない (書き手が核を1つ持つため)",
  ignore: SKIP,
  fn: async () => {
    const path = tmpPath();
    const w = new EvCalibrationWriter(path, fastHeader(5, 2));
    try {
      // Refused BEFORE a worker is spawned — the message names the flag, not a
      // structured-clone failure four frames deep.
      await headlessParallel(2, 5, "khhh", 2, { ktune: CHAMPION, evCalib: w })
        .then(() => {
          throw new Error("分割が拒否されなかった");
        }, (e: Error) => {
          assert(e.message.includes("--evcalib"), e.message);
        });
    } finally {
      w.close();
      Deno.removeSync(path);
    }
  },
});

// ---------------------------------------------------------------------------
// 7. engine drift — a notice, never a refusal
// ---------------------------------------------------------------------------

Deno.test("evcalib: エンジンの指紋が変われば再現検査は落ちずに飛ぶ", () => {
  const now = evEngineHash();
  assert(now !== undefined && now.length === 64, `指紋が取れていない: ${now}`);

  // Same source ⇒ the check runs and the fit is entitled to demand bit equality.
  assertEquals(reproVerdict(now, now).check, true);

  // A CHANGED engine is the expected state during M15's repairs: the wire and
  // the labels are facts about the game and survive it, so the fit says so and
  // carries on. Refusing here would throw away the half that is still true.
  const other = reproVerdict("0".repeat(64), now);
  assertEquals(other.check, false);
  assert(other.why.includes("エンジンが変わっています"), other.why);

  // …and a v1 lane, recorded before the fingerprint existed, is the same case.
  const old = reproVerdict(undefined, now);
  assertEquals(old.check, false);
  assert(old.why.includes("指紋がありません"), old.why);
});
