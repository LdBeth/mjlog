// The M13 fold head: the push/fold gate behind an absent-by-default `--ktune`
// block, its ε-flip lane, and the recorder that writes the lane down.
//
// The load-bearing claim is INIT-EQUIVALENCE, and it is stronger here than
// M12's: `INIT_FOLD` is not "a head that happens to answer the same way", it is
// the incumbent comparison rearranged — one linear layer with `w[margin] = −1`,
// so `forward(x)[0] > 0` IS `margin < 0`. Three tests attack it from three
// sides: whole-hanchan equality (`fold:{}` ≡ absent), per-DECISION equality
// (every gated verdict of two hanchan, through the recorder, head vs gate), and
// the finiteness the identity rests on (one NaN and `0·NaN` poisons the sum).
//
// Then the lane, which has a discipline of its own: at ε = 0 it must be
// INVISIBLE — same games, same bytes, no random number drawn — because a
// recorder that moves the seat is measuring itself.

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { argError } from "../src/cli/args.ts";
import type { FoldFacts, FoldSample } from "../src/ai/fold.ts";
import {
  decideFold,
  FOLD_FEATURES,
  FOLD_FV,
  FOLD_INPUTS,
  foldVector,
  INIT_FOLD,
  mergeFold,
} from "../src/ai/fold.ts";
import { FoldCalibrationWriter, parseFoldCalibration } from "../src/ai/foldcalib.ts";
import { buildMlp, closeMlp } from "../src/ai/mlp.ts";
import { AugmentedHeuristic } from "../src/ai/augmented.ts";
import { computedReads } from "../src/ai/computed.ts";
import { closeArm, headless, loadKtune, openArm } from "../src/harness.ts";
import type { KTune, TableSpec } from "../src/harness.ts";
import { pairedRun } from "../src/paired.ts";
import type { Rng } from "../src/rng.ts";
import { sfc32 } from "../src/rng.ts";
import { JANKI } from "../src/rules.ts";
import { Table } from "../src/table.ts";
import type { Meld, Tile } from "mjrender/model.ts";
import type { RoundOutcome, Seat, Violation, WinInfo } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { Wall } from "../src/wall.ts";
import { playHanchan } from "./helpers.ts";

const SEED = 8191;
const GAMES = 4;

/** The shipped vector — the head has to compose with everything already on it. */
const CHAMPION: KTune = loadKtune(
  new URL("../weights/champion.json", import.meta.url).pathname,
);

/** An arbitrary but realistic gated decision. */
function facts(o: Partial<FoldFacts> = {}): FoldFacts {
  return {
    margin: -0.2,
    push: 0.45,
    pressure: 1,
    gain: 1,
    risk: 1,
    buffer: 1,
    shanten: 1,
    ukeire: 12,
    ukeireTypes: 3,
    dora: 1,
    junme: 8,
    turnsLeft: 10,
    dealer: 0,
    open: 0,
    oppRiichi: 1,
    furoThreats: 0,
    threat0: 1,
    threat1: 0,
    threat2: 0,
    expLoss0: 5.2,
    expLoss1: 0,
    expLoss2: 0,
    pwin: 0.18,
    value: 5.4,
    ev: 0.97,
    score: 25,
    leadTop: -3.2,
    leadBottom: 4.1,
    kyoku: 2,
    honba: 0,
    kyotaku: 0,
    safeTypes: 2,
    lowTypes: 3,
    unassessedTypes: 4,
    genbutsuAll: 2,
    genbutsuMin: 2,
    sensePressure: 0,
    ...o,
  };
}

// ---------------------------------------------------------------------------
// 1. the vector and the identity head
// ---------------------------------------------------------------------------

Deno.test("fold: the feature list is 37 names, unique, and `foldVector` writes them in order", () => {
  assertEquals(FOLD_FEATURES.length, 37);
  assertEquals(FOLD_INPUTS, 37);
  assertEquals(new Set(FOLD_FEATURES).size, 37);
  // The order claim, checked by NAME rather than trusted: give each field its
  // own distinctive value and read the row back through the name list.
  const f = facts() as unknown as Record<string, number>;
  const marked: Record<string, number> = {};
  FOLD_FEATURES.forEach((name, i) => (marked[name] = 1000 + i));
  const x = foldVector({ ...f, ...marked } as unknown as FoldFacts);
  FOLD_FEATURES.forEach((name, i) => assertEquals(x[i], 1000 + i, `列 ${i} は ${name} のはず`));
});

Deno.test("fold: INIT is the old gate — `forward > 0` exactly when `margin < 0`", () => {
  const m = buildMlp(INIT_FOLD);
  const scratch = new Float32Array(FOLD_INPUTS);
  try {
    for (const margin of [-5, -1, -1e-7, 0, 1e-7, 1, 5]) {
      assertEquals(
        decideFold(m, facts({ margin }), scratch),
        margin < 0,
        `margin=${margin}`,
      );
    }
    // Nothing else in the vector may move it: every other weight is 0.
    assertEquals(decideFold(m, facts({ margin: 1, pressure: 99, ev: -99 }), scratch), false);
    assertEquals(decideFold(m, facts({ margin: -1, pressure: 0, ev: 99 }), scratch), true);
  } finally {
    closeMlp(m);
  }
});

Deno.test("fold: a non-finite feature becomes 0 rather than poisoning the sum", () => {
  const m = buildMlp(INIT_FOLD);
  const scratch = new Float32Array(FOLD_INPUTS);
  try {
    // `0 · NaN` is NaN, so an unguarded NaN anywhere in the row would make the
    // identity head answer `NaN > 0` = false on a hand it should fold. The
    // guard lives in `foldVector`, which is why every one of these still reads
    // the sign of `margin`.
    for (const bad of [NaN, Infinity, -Infinity]) {
      assertEquals(decideFold(m, facts({ margin: -1, ev: bad }), scratch), true);
      assertEquals(decideFold(m, facts({ margin: 1, pwin: bad }), scratch), false);
      // …and a poisoned `margin` itself reads as 0, i.e. "push".
      assertEquals(decideFold(m, facts({ margin: bad }), scratch), false);
    }
    const x = foldVector(facts({ ev: NaN, pwin: Infinity }));
    assert(x.every((v) => Number.isFinite(v)), "行に非有限の値が残っています");
  } finally {
    closeMlp(m);
  }
});

Deno.test("fold: mergeFold fills partials against the identity and refuses the rest", () => {
  assertEquals(mergeFold(), INIT_FOLD);
  assertEquals(mergeFold({}), INIT_FOLD);
  assertEquals(mergeFold({ fv: FOLD_FV }), INIT_FOLD);
  // A real block passes through untouched.
  const two = {
    fv: FOLD_FV,
    layers: [
      { in: 37, out: 2, act: "relu" as const, w: new Array(74).fill(0.5), b: [0, 0] },
      { in: 2, out: 1, act: "none" as const, w: [1, 1], b: [0] },
    ],
  };
  assertEquals(mergeFold(two), two);

  const bad = (spec: unknown, msg: string) =>
    assertThrows(() => mergeFold(spec as Partial<typeof INIT_FOLD>), Error, msg);
  bad({ fv: 99 }, "特徴量版");
  bad({ layers: [] }, "layers が空です");
  bad({ layers: [{ in: 36, out: 1, act: "none", w: new Array(36).fill(0), b: [0] }] }, "入力次元");
  bad(
    { layers: [{ in: 37, out: 2, act: "none", w: new Array(74).fill(0), b: [0, 0] }] },
    "出力次元",
  );
  bad({ layers: [{ in: 37, out: 1, act: "tanh", w: new Array(37).fill(0), b: [0] }] }, "act");
  bad({ layers: [{ in: 37, out: 1, act: "none", w: new Array(36).fill(0), b: [0] }] }, "長さ");
  bad(
    { layers: [{ in: 37, out: 1, act: "none", w: new Array(37).fill(NaN), b: [0] }] },
    "有限の数値ではありません",
  );
});

// ---------------------------------------------------------------------------
// 2. `fold: {}` ≡ absent, over real hanchan
// ---------------------------------------------------------------------------

Deno.test("fold head: `{}` (⇒ INIT) plays the identical hanchan — kkkk", () => {
  const plain = headless(GAMES, SEED, "kkkk", {});
  const inited = headless(GAMES, SEED, "kkkk", { ktune: { fold: {} } });
  assertEquals(inited.results, plain.results);
});

Deno.test("fold head: `{}` (⇒ INIT) plays the identical hanchan — khhh + champion", () => {
  // champion.json carries no `fold` block today (`champion_test` asserts it),
  // so the base is the file itself; stripping is spelled out anyway so this
  // test keeps meaning the same thing after a promotion.
  const base = { ...CHAMPION, fold: undefined };
  const plain = headless(GAMES, SEED, "khhh", { ktune: base });
  const inited = headless(GAMES, SEED, "khhh", { ktune: { ...base, fold: {} } });
  assertEquals(inited.results, plain.results);
});

Deno.test("fold head: the block reaches k seats only — an h seat builds no head", () => {
  const smuggled: TableSpec = [
    { kind: "k", ktune: { fold: {} } },
    { kind: "h", ktune: { fold: {} } },
    { kind: "h" },
    { kind: "h" },
  ];
  const arm = openArm(smuggled);
  try {
    // deno-lint-ignore no-explicit-any
    const p = (s: number) => arm.built[s].policy as any;
    assertNotEquals(p(0).foldHead, null, "k席にはヘッドが載るはず");
    assertEquals(p(1).foldHead, null, "h席は凍結 — ヘッドは載らない (D11)");
    assertEquals(p(2).foldHead, null);
  } finally {
    closeArm(arm);
  }
});

// ---------------------------------------------------------------------------
// 3. per-decision equality, instrumented
// ---------------------------------------------------------------------------

/**
 * Every gated fold verdict of one hanchan, in decision order.
 *
 * The sink is the instrument: with it attached (and ε 0) `computeFold` takes
 * the head path either way, so the two runs differ ONLY in whether a head
 * answers or `margin < 0` does — which is exactly the claim. The seat is built
 * by hand rather than through `makePolicy` so the head can be handed in
 * directly.
 */
function verdicts(seed: number, head: ReturnType<typeof buildMlp> | undefined): boolean[] {
  const out: boolean[] = [];
  playHanchan(
    seed,
    (s) =>
      s === 0
        ? new AugmentedHeuristic(`K${s}`, seed * 4 + s, computedReads(), {
          fold: head,
          foldSink: (r: FoldSample) => out.push(r.verdict),
        })
        : new AugmentedHeuristic(`K${s}`, seed * 4 + s, computedReads()),
  );
  return out;
}

Deno.test("fold head: INIT reproduces the old gate on EVERY decision of two hanchan", () => {
  const head = buildMlp(INIT_FOLD);
  try {
    let total = 0;
    for (const seed of [101, 505]) {
      const gate = verdicts(seed, undefined);
      const init = verdicts(seed, head);
      assert(gate.length > 0, `種${seed}: ゲートに到達した判断が1件もない`);
      assertEquals(init, gate, `種${seed}: 判断ごとの判定が一致しない`);
      total += gate.length;
    }
    // The claim is worthless if the region is empty — say how big it was.
    assert(total > 100, `検査した判断が少なすぎます: ${total}`);
  } finally {
    closeMlp(head);
  }
});

// ---------------------------------------------------------------------------
// 4. the block is live
// ---------------------------------------------------------------------------

/** A head whose bias alone decides: +1000 folds always, −1000 never. */
function biased(bias: number) {
  return {
    fv: FOLD_FV,
    layers: [{
      in: FOLD_INPUTS,
      out: 1,
      act: "none" as const,
      w: new Array(FOLD_INPUTS).fill(0),
      b: [bias],
    }],
  };
}

Deno.test("fold head: a hostile bias changes the fold rate and the games", () => {
  const plain = headless(GAMES, SEED, "kkkk", {});
  const alwaysFold = headless(GAMES, SEED, "kkkk", { ktune: { fold: biased(1000) } });
  const neverFold = headless(GAMES, SEED, "kkkk", { ktune: { fold: biased(-1000) } });
  assertNotEquals(alwaysFold.results, plain.results, "常に降りるヘッドが盤面を動かしていない");
  assertNotEquals(neverFold.results, plain.results, "常に押すヘッドが盤面を動かしていない");
  assertNotEquals(alwaysFold.results, neverFold.results);

  // …and the rate itself, read through the recorder rather than inferred.
  const rate = (fold?: ReturnType<typeof biased>) => {
    const head = fold ? buildMlp(fold) : undefined;
    try {
      const v = verdicts(101, head);
      return v.filter((x) => x).length / v.length;
    } finally {
      if (head) closeMlp(head);
    }
  };
  assertEquals(rate(biased(1000)), 1);
  assertEquals(rate(biased(-1000)), 0);
});

// ---------------------------------------------------------------------------
// 5. the lane is invisible at ε = 0, and real above it
// ---------------------------------------------------------------------------

function tmpPath(): string {
  return Deno.makeTempFileSync({ prefix: "mjgame-fold-", suffix: ".jsonl" });
}

function lane(path: string, eps: number, seed = 4242, games = 3) {
  const w = new FoldCalibrationWriter(path, {
    seats: "khhh",
    seed,
    games,
    eps,
    head: "gate",
  });
  try {
    const run = headless(games, seed, "khhh", { foldCalib: w, foldEps: eps });
    return { run, stats: w.stats() };
  } finally {
    w.close();
  }
}

Deno.test("fold lane: ε=0 records without moving a single tile, and draws no random number", () => {
  const path = tmpPath();
  try {
    const plain = headless(3, 4242, "khhh", {});
    const { run, stats } = lane(path, 0);
    assertEquals(run.results, plain.results, "記録係が盤面を動かしています");
    assert(stats.rows > 0, "記録が空です");
    assertEquals(stats.flips, 0);
    assertEquals(stats.dropped, 0);
    const { header, records } = parseFoldCalibration(Deno.readTextFileSync(path), path);
    assertEquals(header.eps, 0);
    assertEquals(header.head, "gate");
    assertEquals(header.fv, FOLD_FV);
    assertEquals(header.features, FOLD_FEATURES);
    // Nothing was flipped, so `taken` is `verdict` everywhere and the propensity
    // is 1 — the lane is a pure observation of the incumbent.
    for (const r of records) {
      assertEquals(r.taken, r.verdict);
      assertEquals(r.p, 1);
      assertEquals(r.flipped, 0);
      // THE REPRODUCTION CHECK, the one `fold_report.ts` prints: on a gate lane
      // the verdict IS the sign of column 0.
      assertEquals(r.x[0] < 0, r.verdict);
      assertEquals(r.x.length, FOLD_INPUTS);
    }
  } finally {
    Deno.removeSync(path);
  }
});

Deno.test("fold lane: ε=0.3 flips about 30% of decisions and changes the games", () => {
  const a = tmpPath();
  const b = tmpPath();
  try {
    const plain = headless(3, 4242, "khhh", {});
    const { run, stats } = lane(a, 0.3);
    assertNotEquals(run.results, plain.results, "反転が盤面に届いていません");
    const rate = stats.flips / stats.rows;
    assert(rate > 0.15 && rate < 0.45, `反転率が ${rate.toFixed(3)} — 0.3 のはず`);
    // Same seed, same lane: the flip stream is a function of the seed like
    // everything else, so the two files are byte-identical.
    lane(b, 0.3);
    assertEquals(Deno.readTextFileSync(b), Deno.readTextFileSync(a));

    const { records } = parseFoldCalibration(Deno.readTextFileSync(a), a);
    for (const r of records) {
      assertEquals(r.taken, r.flipped ? !r.verdict : r.verdict);
      assertEquals(r.p, r.flipped ? 0.3 : 0.7);
      assertEquals(r.x[0] < 0, r.verdict);
    }
  } finally {
    Deno.removeSync(a);
    Deno.removeSync(b);
  }
});

Deno.test("fold lane: exactly one draw per recorded decision", () => {
  // `shouldFold` is asked two or three times per decision — the C7 planner
  // first, then `context` — and each extra ask must NOT consume another random
  // number, or adding a caller would silently re-shuffle the flip schedule of
  // every seed. `computeFold` is memoised on the Observation, so the claim is
  // exact and checkable by counting: one draw per EMITTED sample, no more.
  let draws = 0;
  const real = sfc32(1234);
  const counting: Rng = {
    u32: () => real.u32(),
    int: (n) => real.int(n),
    float: () => {
      draws++;
      return real.float();
    },
    fork: (t) => real.fork(t),
  };
  const samples: FoldSample[] = [];
  const seed = 101;
  playHanchan(seed, (s) => {
    const p = new AugmentedHeuristic(`K${s}`, seed * 4 + s, computedReads(), {
      ...(s === 0
        ? { foldSink: (r: FoldSample) => samples.push(r), foldExplore: { eps: 0.5 } }
        : {}),
    });
    // deno-lint-ignore no-explicit-any
    if (s === 0) (p as any).foldRng = counting;
    return p;
  });
  assert(samples.length > 0, "ゲートに到達した判断が1件もない");
  assertEquals(draws, samples.length, "1判断あたり1回を超えて乱数を引いています");
  // …and ε 0 must draw nothing at all: the rng is never even made.
  const quiet: FoldSample[] = [];
  playHanchan(seed, (s) =>
    new AugmentedHeuristic(`K${s}`, seed * 4 + s, computedReads(), {
      ...(s === 0 ? { foldSink: (r: FoldSample) => quiet.push(r) } : {}),
    }));
  // (Not the same COUNT — a flipped verdict re-writes the rest of the hanchan,
  // so the two runs face different boards. The claim is only that ε = 0 flips
  // nothing while still recording.)
  assert(quiet.length > 0);
  assertEquals(quiet.filter((r) => r.flipped).length, 0);
});

Deno.test("fold lane: the paired control arm never receives the writer", () => {
  const path = tmpPath();
  try {
    const w = new FoldCalibrationWriter(path, {
      seats: "khhh",
      seed: 909,
      games: 1,
      eps: 0,
      head: "gate",
    });
    try {
      pairedRun(1, 909, "khhh", { foldCalib: w });
    } finally {
      w.close();
    }
    const solo = tmpPath();
    const w2 = new FoldCalibrationWriter(solo, {
      seats: "khhh",
      seed: 909,
      games: 1,
      eps: 0,
      head: "gate",
    });
    try {
      headless(1, 909, "khhh", { foldCalib: w2 });
    } finally {
      w2.close();
    }
    // Arm B is `hhhh` (no "k" seat 0 at all), so a paired run must write
    // EXACTLY arm A's rows — not two interleaved streams.
    assertEquals(Deno.readTextFileSync(path), Deno.readTextFileSync(solo));
    Deno.removeSync(solo);
  } finally {
    Deno.removeSync(path);
  }
});

// ---------------------------------------------------------------------------
// 6. the writer, in isolation
// ---------------------------------------------------------------------------

function sample(o: Partial<FoldSample> = {}): FoldSample {
  return {
    x: FOLD_FEATURES.map((_, i) => i / 10),
    verdict: true,
    taken: true,
    p: 1,
    flipped: false,
    turn: true,
    ...o,
  };
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

function violation(seat: Seat): Violation {
  return {
    rule: "test",
    label: "試験",
    seat,
    kyoku: 0,
    junme: 3,
    points: 10,
    tier: "A",
    confidence: 1,
    detail: "",
  };
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

Deno.test("foldcalib: samples survive the round trip and wear their round's reward", () => {
  const path = tmpPath();
  const w = new FoldCalibrationWriter(path, {
    seats: "khhh",
    seed: 7,
    games: 1,
    eps: 0.05,
    head: "gate",
  });
  w.beginGame(7);

  // 東1本場1: we win off seat 2. THE REWARD IS `deltas[0]`, not the winning
  // hand's points — and not `handLabels`' `winPoints`, which is the same number
  // here only because we won.
  w.record(sample({ x: FOLD_FEATURES.map((n) => (n === "junme" ? 4 : 1)) }));
  const t1 = makeTable(0, 1);
  fillRiver(t1, 0, 9);
  w.endRound(t1, agari([win(0, 2)], [8300, 0, -8300, 0]));

  // 東2: seat 1 rons OUR discard — the reward is NEGATIVE, which is the whole
  // reason `winPoints` could not be reused.
  w.record(sample({ verdict: false, taken: true, p: 0.95 }));
  w.record(sample({ verdict: true, taken: false, p: 0.05, flipped: true, turn: false }));
  const t2 = makeTable(1, 0);
  fillRiver(t2, 0, 14);
  t2.addViolation(violation(0));
  t2.addViolation(violation(2));
  t2.addViolation(violation(0));
  w.endRound(t2, agari([win(1, 0)], [-5800, 5800, 0, 0]));

  // 東3: 流局. The 聴牌料 IS the reward for this round.
  w.record(sample({ verdict: false, taken: false }));
  const t3 = makeTable(2, 0);
  fillRiver(t3, 0, 18);
  w.endRound(t3, ryuukyoku([3000, -1000, -1000, -1000]));
  w.close();

  const { header, records } = parseFoldCalibration(Deno.readTextFileSync(path), path);
  assertEquals(header.eps, 0.05);
  assertEquals(header.features, FOLD_FEATURES);
  assertEquals(records.length, 4);
  assertEquals(records.map((r) => r.k), [0, 1, 1, 2]);
  assertEquals(records.map((r) => r.b), [1, 0, 0, 0]);
  assertEquals(records.map((r) => r.n), [0, 1, 2, 3]);
  assertEquals(records.map((r) => r.s), [7, 7, 7, 7]);
  // D7's reward, per round, shared by every decision inside it.
  assertEquals(records.map((r) => r.delta), [8300, -5800, -5800, 3000]);
  assertEquals(records.map((r) => r.won), [1, 0, 0, 0]);
  assertEquals(records.map((r) => r.dealtIn), [0, 1, 1, 0]);
  assertEquals(records.map((r) => r.outcome), ["agari", "agari", "agari", "ryuukyoku"]);
  assertEquals(records.map((r) => r.endJunme), [9, 14, 14, 18]);
  // Seat 0's ledger entries of THAT round — seat 2's is not ours, and the Table
  // is per-round so nothing from 東1 leaks in.
  assertEquals(records.map((r) => r.vio0), [0, 2, 2, 0]);
  // The positional columns, and the row itself, verbatim.
  assertEquals(records.map((r) => r.turn), [1, 1, 0, 1]);
  // `junme` is read back out of the row itself (column 10), so only the first
  // sample — the one built with a marked row — carries 4.
  assertEquals(records.map((r) => r.junme), [4, 1, 1, 1]);
  assertEquals(records.map((r) => r.flipped), [0, 0, 1, 0]);
  assertEquals(records.map((r) => r.p), [1, 0.95, 0.05, 1]);
  assertEquals(records[1].x, sample().x);
  Deno.removeSync(path);
});

Deno.test("foldcalib: a game boundary drops unlabelled samples rather than mislabelling them", () => {
  const path = tmpPath();
  const w = new FoldCalibrationWriter(path, {
    seats: "khhh",
    seed: 1,
    games: 2,
    eps: 0,
    head: "gate",
  });
  w.beginGame(1);
  w.record(sample());
  // No `endRound`: the round never finished, so there is no settlement to stamp.
  w.beginGame(2);
  w.record(sample({ verdict: false, taken: false }));
  const t = makeTable();
  fillRiver(t, 0, 5);
  w.endRound(t, ryuukyoku([0, 0, 0, 0]));
  w.close();

  const { records } = parseFoldCalibration(Deno.readTextFileSync(path), path);
  assertEquals(records.length, 1);
  assertEquals(records[0].s, 2);
  assertEquals(records[0].n, 0);
  const st = w.stats();
  assertEquals(st.games, 2);
  assertEquals(st.rows, 1);
  assertEquals(st.dropped, 1);
  Deno.removeSync(path);
});

Deno.test("foldcalib: a foreign, old or wrong-fv file is refused, never coerced", () => {
  const line = (h: unknown) => JSON.stringify(h) + "\n";
  const base = { kind: "mjgame-fold", eps: 0, fv: FOLD_FV, features: FOLD_FEATURES, head: "gate" };
  assertThrows(
    () => parseFoldCalibration(line({ ...base, v: 2 })),
    Error,
    "版が違います",
  );
  assertThrows(
    () => parseFoldCalibration(line({ ...base, v: 1, kind: "mjgame-hand" })),
    Error,
    "押し引きの較正記録ではありません",
  );
  assertThrows(
    () => parseFoldCalibration(line({ ...base, v: 1, fv: 99 })),
    Error,
    "特徴量版が違います",
  );
  assertThrows(() => parseFoldCalibration(""), Error, "空のファイルです");
});

// ---------------------------------------------------------------------------
// 7. the CLI refusals
// ---------------------------------------------------------------------------

Deno.test("fold flags: argError refuses every combination that would silently do nothing", () => {
  const base = { cmd: "selfplay", seats: "khhh", calibrate: "" };
  assertEquals(argError({ ...base, foldcalib: "x.jsonl" }), null);
  assertEquals(argError({ ...base, foldcalib: "x.jsonl", foldEps: 0.05 }), null);
  assertEquals(argError({ ...base, cmd: "paired", foldcalib: "x.jsonl" }), null);
  // A driver that does not play whole rounds has no settlement to label with.
  for (const cmd of ["play", "bench"]) {
    assert(argError({ ...base, cmd, foldcalib: "x.jsonl" })?.includes("selfplay / paired"));
  }
  // The head routes to "k" seats only, so an "h" seat 0 has nothing to record.
  assert(argError({ ...base, seats: "hhhh", foldcalib: "x.jsonl" })?.includes("k席"));
  assert(argError({ ...base, seats: "nhhh", foldcalib: "x.jsonl" })?.includes("k席"));
  // Four buffers flushing into one file in wall-clock order.
  assert(argError({ ...base, foldcalib: "x.jsonl", jobs: 4 })?.includes("--jobs"));
  // Playing worse and recording nothing.
  assert(argError({ ...base, foldEps: 0.05 })?.includes("--foldcalib"));
});

// ---------------------------------------------------------------------------
// 8. the freeze round trip
// ---------------------------------------------------------------------------

Deno.test("fold: a resolved snapshot round-trips through mergeFold", () => {
  // What `scripts/freeze.ts` writes for a seat carrying a head: the block,
  // resolved and validated. Reading it back must give the same object, and the
  // seat it builds must play the same game.
  const src = biased(0.25);
  const resolved = mergeFold(src);
  assertEquals(mergeFold(JSON.parse(JSON.stringify(resolved))), resolved);
  const a = headless(2, 1234, "kkkk", { ktune: { fold: src } });
  const b = headless(2, 1234, "kkkk", { ktune: { fold: resolved } });
  assertEquals(b.results, a.results);
  // …and the identity survives the same trip.
  assertEquals(mergeFold(JSON.parse(JSON.stringify(INIT_FOLD))), INIT_FOLD);
});
