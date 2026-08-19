// Trajectory recording: the JSONL a Python/MLX trainer eats.
//
// The file layout is a frozen contract, so this test reads the bytes back the
// way the trainer will — decode base64, count bytes, re-derive the settlement —
// rather than trusting the writer's own helpers to agree with themselves.

import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { RandomPolicy } from "../src/ai/random.ts";
import { runMatchSync } from "../src/match.ts";
import type { MatchResult } from "../src/match.ts";
import { makeDojoHooks } from "../src/main.ts";
import { encodeOracle, encodeSeq, FEATURES, ORACLE_LEN, PLANE_LEN } from "../src/rl/features.ts";
import type { Observation } from "../src/observe.ts";
import type { SyncPolicy } from "../src/policy.ts";
import { ACTIONS } from "../src/rl/actionspace.ts";
import {
  f32leBytes,
  fromBase64,
  RecordingPolicy,
  settlement,
  toBase64,
  TrajectoryWriter,
  violationPoints,
  writeMatchEnd,
} from "../src/rl/record.ts";
import { DOJO_HEADLESS, JANKI } from "../src/rules.ts";
import { sfc32 } from "../src/rng.ts";
import { scorer } from "../src/score.ts";
import type { Table } from "../src/table.ts";
import type { Action, RoundOutcome, Seat, Violation } from "../src/types.ts";
import { SEATS } from "../src/types.ts";

const SCALAR_BYTES = 42 * 4; // 168: the frozen scalar block (unchanged from v3)

interface Recorded {
  result: MatchResult;
  lines: Record<string, unknown>[];
  stats: { d: number; r: number; m: number };
}

/** One recorded hanchan, written to `path`. Four seats share ONE writer. */
function record(path: string, seed: number): Recorded {
  const writer = new TrajectoryWriter(path);
  const policies = SEATS.map((s) =>
    new RecordingPolicy(new RandomPolicy(`R${s}`, seed * 4 + s), writer)
  );
  const result = runMatchSync(policies, {
    seed,
    cfg: JANKI,
    dojo: DOJO_HEADLESS,
    scorer,
    ...makeDojoHooks(DOJO_HEADLESS),
  });
  writeMatchEnd(writer, result, JANKI);
  const stats = writer.stats();
  writer.close();

  const text = Deno.readTextFileSync(path);
  assert(text.endsWith("\n"), "every line must be newline-terminated");
  const lines = text.split("\n").filter((l) => l !== "").map((l, i) => {
    try {
      return JSON.parse(l) as Record<string, unknown>;
    } catch (e) {
      throw new Error(`line ${i + 1} is not JSON: ${e instanceof Error ? e.message : e}`);
    }
  });
  return { result, lines, stats };
}

/**
 * 精算, re-derived here by a different route than `settlement` (rank counted
 * per seat rather than by sorting), so a change to the shipped formula shows up
 * as a failure instead of two copies of the same bug agreeing.
 */
function expectedNet(scores: number[]): number[] {
  assert(JANKI.truncateSub1000, "this derivation assumes 1000点未満切り捨て");
  return scores.map((s, seat) => {
    // Ties go to the lower seat, exactly as the sort's `a.seat - b.seat` does.
    const rank = scores.filter((o, i) => o > s || (o === s && i < seat)).length;
    return Math.trunc((s - JANKI.returnScore) / 1000) + JANKI.uma[rank];
  });
}

/**
 * 供託 the last round left behind — its sticks, unless a 和了 swept them. This
 * is what `finalize` then pays out to the top finisher, so a seed that leaves
 * one exercises that payout.
 */
function leftoverKyotaku(r: MatchResult): number {
  let k = 0;
  for (let i = 0; i < r.rounds.length; i++) {
    k = r.outcomes[i].kind === "agari" ? 0 : r.rounds[i].kyotaku;
  }
  return k;
}

Deno.test("record: 1半荘の JSONL が契約どおりに書かれる", async (t) => {
  const path = Deno.makeTempFileSync({ prefix: "mjgame_rl_", suffix: ".jsonl" });
  try {
    // The seed is chosen so the hanchan is NOT degenerate: it ends with a 供託
    // stick still on the table and at least one 和了, and one final score is
    // below 返し点 — which is what makes the conservation check and the
    // truncate-toward-zero half of 精算 actually bite. If engine changes ever
    // trip the guards below, re-pick a seed with the same three properties
    // rather than dropping them.
    const { result, lines, stats } = record(path, 111);
    assert(leftoverKyotaku(result) > 0, "fixture seed no longer leaves a 供託 stick");
    assert(result.outcomes.some((o) => o.kind === "agari"), "fixture seed has no 和了");
    assert(result.scores.some((s) => s < JANKI.returnScore), "fixture seed has no minus 精算");

    await t.step("行の種別は d/r/m のみ、m は最後に1本", () => {
      assert(lines.length > 0, "nothing was recorded");
      for (const [i, l] of lines.entries()) {
        assert(
          l.k === "d" || l.k === "r" || l.k === "m",
          `line ${i + 1}: unexpected kind ${JSON.stringify(l.k)}`,
        );
      }
      const kinds = lines.map((l) => l.k);
      assertEquals(kinds.filter((k) => k === "r").length, result.outcomes.length);
      assertEquals(kinds.filter((k) => k === "m").length, 1);
      assertEquals(kinds[kinds.length - 1], "m");
      // The writer's own tally must agree with what landed on disk.
      assertEquals(stats, {
        d: kinds.filter((k) => k === "d").length,
        r: result.outcomes.length,
        m: 1,
      });
      assert(stats.d > 0, "no decisions were recorded");
    });

    await t.step("d 行: 特徴量のバイト数・マスク・選択行動", () => {
      const ds = lines.filter((l) => l.k === "d");
      for (const [i, d] of ds.entries()) {
        const where = `d line ${i + 1}`;
        assertEquals(fromBase64(d.planes as string).length, PLANE_LEN, `${where}: planes`);
        assertEquals(fromBase64(d.scalars as string).length, SCALAR_BYTES, `${where}: scalars`);
        // The two numbers the header comment promises, spelled out: the v4
        // planes/scalars are v3's — 48 × 34 Int8 and 42 little-endian float32.
        assertEquals(PLANE_LEN, 1632);
        assertEquals(fromBase64(d.planes as string).length, 1632, `${where}: 1632 plane bytes`);
        assertEquals(fromBase64(d.scalars as string).length, 168, `${where}: 168 scalar bytes`);
        // The trainer refuses a line whose feature version it does not know, so
        // every line must carry it — a missing "v" means "v1" over there.
        assertEquals(d.v, 4, `${where}: feature version`);
        assertEquals(d.v, FEATURES.version, `${where}: version tracks the encoder`);

        // "seq": 4 packed int8 per discard, at most 96 tokens. The empty string
        // is legal and means L = 0 — the first decision of a hand has one.
        assert(typeof d.seq === "string", `${where}: no seq field`);
        const seq = fromBase64(d.seq as string);
        assertEquals(seq.length % 4, 0, `${where}: seq is not whole tokens`);
        assert(seq.length <= 4 * 96, `${where}: ${seq.length / 4} tokens`);
        for (let j = 0; j < seq.length; j += 4) {
          // Read back as SIGNED bytes, the way the trainer's int8 view will.
          const [type, rel, idx, flags] = [...seq.slice(j, j + 4)].map((b) => (b << 24) >> 24);
          assert(type >= 0 && type < 34, `${where}: token type ${type}`);
          assert(rel >= 0 && rel < 4, `${where}: token seatRel ${rel}`);
          assert(idx >= 0 && idx < 24, `${where}: token idx ${idx}`);
          assert(flags >= 0 && flags <= 7, `${where}: token flags ${flags}`);
        }

        const mask = d.mask as number[];
        assert(Array.isArray(mask) && mask.length > 0, `${where}: empty mask`);
        for (let j = 1; j < mask.length; j++) {
          assert(mask[j] > mask[j - 1], `${where}: mask not ascending (${mask.join(",")})`);
        }
        for (const m of mask) {
          assert(Number.isInteger(m) && m >= 0 && m < ACTIONS, `${where}: mask slot ${m}`);
        }
        assert(mask.includes(d.a as number), `${where}: chosen ${d.a} not in mask`);

        const seat = d.seat as number;
        assert(Number.isInteger(seat) && seat >= 0 && seat <= 3, `${where}: seat ${seat}`);
        for (const f of ["kyoku", "honba", "junme"]) {
          assert(typeof d[f] === "number" && Number.isFinite(d[f]), `${where}: ${f}=${d[f]}`);
        }
        assert((d.kyoku as number) >= 0 && (d.kyoku as number) <= 7, `${where}: kyoku`);
        assert((d.honba as number) >= 0, `${where}: honba`);
        assert((d.junme as number) >= 0, `${where}: junme`);
      }
    });

    await t.step("r 行: 局結果は outcomes と一致する", () => {
      const rs = lines.filter((l) => l.k === "r");
      rs.forEach((r, i) => {
        const o = result.outcomes[i];
        assertEquals(r.deltas, o.deltas);
        assertEquals(r.outcome, o.kind === "agari" ? "agari" : "draw");
      });
    });

    await t.step("r 行: kyoku/honba でその局を名乗る", () => {
      const rs = lines.filter((l) => l.k === "r");
      assertEquals(rs.length, result.rounds.length);
      rs.forEach((r, i) => {
        assertEquals(r.kyoku, result.rounds[i].kyoku, `r line ${i + 1}: kyoku`);
        assertEquals(r.honba, result.rounds[i].honba, `r line ${i + 1}: honba`);
      });
      // The pair is what a loader joins on, so it must identify the round: no
      // two "r" lines of one match may carry the same one.
      const seen = new Set(rs.map((r) => `${r.kyoku}/${r.honba}`));
      assertEquals(seen.size, rs.length, "(kyoku,honba) repeats inside one match");
      // …and it must be the SAME pair the d lines of that round carry, which is
      // the whole point: the join is d ↔ r, not d ↔ position.
      const dPairs = new Set(lines.filter((l) => l.k === "d").map((d) => `${d.kyoku}/${d.honba}`));
      for (const p of dPairs) assert(seen.has(p), `d lines sit in round ${p}, which has no r line`);
    });

    await t.step("r 行: viol はその局で科された罰符だけを持つ", () => {
      const rs = lines.filter((l) => l.k === "r");
      // Re-slice the ledger here rather than calling the writer's helper: the
      // point is that the file agrees with the ledger, not with itself.
      assertEquals(result.ledgerCuts.length, result.outcomes.length);
      rs.forEach((r, i) => {
        const viol = r.viol as number[];
        assert(Array.isArray(viol), `r line ${i + 1}: no viol array`);
        assertEquals(viol.length, 4, `r line ${i + 1}: viol length`);
        const lo = result.ledgerCuts[i - 1] ?? 0;
        const hi = result.ledgerCuts[i];
        assert(hi >= lo, `r line ${i + 1}: cuts went backwards (${lo} > ${hi})`);
        const want = [0, 0, 0, 0];
        for (const v of result.ledger.slice(lo, hi)) want[v.seat] += v.points;
        assertEquals(viol, want, `r line ${i + 1}: viol != ledger[${lo},${hi})`);
        for (const p of viol) assert(p >= 0, `r line ${i + 1}: negative 評価点マイナス`);
      });
      // Every entry of the ledger lands in exactly one round.
      assertEquals(result.ledgerCuts[result.ledgerCuts.length - 1], result.ledger.length);
      // The attribution is only interesting if it actually distinguishes
      // rounds. Random play fouls in every round, so the guard is that the
      // rounds do not all carry the SAME vector (and none of them carries the
      // match total, which is what the old match-level charge amounted to).
      const dirty = rs.filter((r) => (r.viol as number[]).some((p) => p > 0));
      assert(dirty.length > 0, "fixture seed attributes no 罰符 to any round");
      assert(rs.length > 1, "fixture seed is a single round: attribution is trivial");
      const shapes = new Set(rs.map((r) => JSON.stringify(r.viol)));
      assert(shapes.size > 1, "every round got an identical viol vector");
      const total = JSON.stringify(violationPoints(result.ledger));
      assert(!shapes.has(total), `a single round was charged the whole match (${total})`);
    });

    await t.step("m 行: 点棒保存・精算・違反点", () => {
      const m = lines[lines.length - 1];
      const scores = m.scores as number[];
      assertEquals(scores.length, 4);
      for (const s of scores) assert(typeof s === "number" && Number.isFinite(s));
      assertEquals(scores, result.scores);

      // Exactly 4×30000: the 供託 the last round left behind is not lost, it is
      // paid to the top finisher by `finalize` (the Tenhou convention), so the
      // recorded scores account for every point that went in.
      const sum = scores.reduce((a, b) => a + b, 0);
      assertEquals(sum, JANKI.startScore * 4, `${sum} != ${JANKI.startScore * 4}`);

      assertEquals(m.net, settlement(result.scores, JANKI));
      assertEquals(m.net, expectedNet(scores));
      // ウマ is a pure bonus in this ruleset, so the nets sum to the uma total
      // plus however far the table's points sit from 4×返し点.
      const netSum = (m.net as number[]).reduce((a, b) => a + b, 0);
      const umaSum = JANKI.uma.reduce((a, b) => a + b, 0);
      assert(netSum <= umaSum, `net sum ${netSum} exceeds uma total ${umaSum}`);

      const vio = [0, 0, 0, 0];
      for (const v of result.ledger) vio[v.seat] += v.points;
      assertEquals(m.violations, vio);
      // Random play cuts honors and kans freely, so the ledger is never empty
      // here — without this the equality above could pass on two empty arrays.
      assert(vio.some((p) => p > 0), "no 評価点マイナス was recorded at all");

      // The match total is exactly the rounds' attribution summed: a loader may
      // use either, and reward built from "viol" must not drift from "m".
      const perRound = [0, 0, 0, 0];
      for (const r of lines.filter((l) => l.k === "r")) {
        (r.viol as number[]).forEach((p, seat) => perRound[seat] += p);
      }
      assertEquals(perRound, m.violations);
    });
  } finally {
    Deno.removeSync(path);
  }
});

// ---------------------------------------------------------------------------
// round-level 罰符 attribution
// ---------------------------------------------------------------------------

function fakeViolation(seat: Seat, kyoku: number, points: number): Violation {
  return {
    rule: "test",
    label: "テスト",
    seat,
    kyoku,
    junme: 1,
    points,
    tier: "A",
    confidence: 1,
    detail: "",
  };
}

function fakeOutcome(): RoundOutcome {
  return {
    kind: "ryuukyoku",
    draw: "exhaustive",
    tenpai: [false, false, false, false],
    tenpaiHands: [],
    deltas: [0, 0, 0, 0],
    dealerRepeat: false,
  };
}

Deno.test("record: 一席だけ記録しても r 行は全局を名乗る", () => {
  // The mixed-population shape (`--seats=nhhh`): only the neural seat is
  // wrapped, so the "d" lines cover a SUBSET of the rounds while the "r" lines
  // must still cover all of them. That asymmetry is exactly what positional
  // reconstruction could not survive, and the labels are what fix it.
  const path = Deno.makeTempFileSync({ prefix: "mjgame_mixed_", suffix: ".jsonl" });
  try {
    const writer = new TrajectoryWriter(path);
    const policies = SEATS.map((s) => {
      const inner = new RandomPolicy(`R${s}`, 966 * 4 + s);
      return s === 0 ? new RecordingPolicy(inner, writer) : inner;
    });
    const result = runMatchSync(policies, {
      seed: 966,
      cfg: JANKI,
      dojo: DOJO_HEADLESS,
      scorer,
      ...makeDojoHooks(DOJO_HEADLESS),
    });
    writeMatchEnd(writer, result, JANKI);
    writer.close();

    const lines = Deno.readTextFileSync(path).split("\n").filter((l) => l !== "")
      .map((l) => JSON.parse(l) as Record<string, unknown>);
    const rs = lines.filter((l) => l.k === "r");
    const ds = lines.filter((l) => l.k === "d");
    assertEquals(rs.length, result.rounds.length, "one r line per round played");
    assert(ds.length > 0, "the wrapped seat recorded nothing");
    // Every d line belongs to a seat the driver wrapped…
    for (const d of ds) assertEquals(d.seat, 0, "an unwrapped seat was recorded");
    // …and every d line's round is named by an r line.
    const rPairs = new Set(rs.map((r) => `${r.kyoku}/${r.honba}`));
    assertEquals(rPairs.size, rs.length, "(kyoku,honba) repeats inside one match");
    for (const d of ds) {
      assert(
        rPairs.has(`${d.kyoku}/${d.honba}`),
        `d line in unlabelled round ${d.kyoku}/${d.honba}`,
      );
    }
    // 連荘/流局 push honba past 0 on this seed, so the join really does need
    // both halves of the pair — kyoku alone would not be unique.
    assert(rs.some((r) => (r.honba as number) > 0), "fixture seed never leaves 0本場");
  } finally {
    Deno.removeSync(path);
  }
});

Deno.test("record: 罰符は発生した局の r 行にだけ載る", () => {
  const path = Deno.makeTempFileSync({ prefix: "mjgame_viol_", suffix: ".jsonl" });
  try {
    // Four rounds; 罰符 only in round 0 (seat 1) and round 2 (seats 1 and 3).
    // Round 1 and 3 are clean, so a match-level total would wrongly charge them.
    const ledger = [
      fakeViolation(1, 0, 5),
      fakeViolation(1, 2, 3),
      fakeViolation(3, 2, 7),
    ];
    const writer = new TrajectoryWriter(path);
    writeMatchEnd(writer, {
      scores: [30000, 30000, 30000, 30000],
      // 東1 → 東1-1本場 (連荘) → 東2 → 東2-1本場: honba is not always 0, and the
      // (kyoku,honba) pair is what has to come back off the line.
      rounds: [
        { kyoku: 0, honba: 0 },
        { kyoku: 0, honba: 1 },
        { kyoku: 1, honba: 0 },
        { kyoku: 1, honba: 1 },
      ],
      outcomes: [fakeOutcome(), fakeOutcome(), fakeOutcome(), fakeOutcome()],
      ledger,
      ledgerCuts: [1, 1, 3, 3],
    }, JANKI);
    writer.close();

    const lines = Deno.readTextFileSync(path).split("\n").filter((l) => l !== "")
      .map((l) => JSON.parse(l) as Record<string, unknown>);
    const rs = lines.filter((l) => l.k === "r");
    assertEquals(rs.length, 4);
    assertEquals(rs.map((r) => [r.kyoku, r.honba]), [[0, 0], [0, 1], [1, 0], [1, 1]]);
    assertEquals(rs[0].viol, [0, 5, 0, 0]);
    assertEquals(rs[1].viol, [0, 0, 0, 0]);
    assertEquals(rs[2].viol, [0, 3, 0, 7]);
    assertEquals(rs[3].viol, [0, 0, 0, 0]);
    assertEquals(lines[lines.length - 1].violations, [0, 8, 0, 7]);
  } finally {
    Deno.removeSync(path);
  }
});

Deno.test("record: ledgerCuts が outcomes と合わなければ書かずに落ちる", () => {
  const path = Deno.makeTempFileSync({ prefix: "mjgame_viol_", suffix: ".jsonl" });
  const writer = new TrajectoryWriter(path);
  try {
    assertThrows(
      () =>
        writeMatchEnd(writer, {
          scores: [30000, 30000, 30000, 30000],
          rounds: [{ kyoku: 0, honba: 0 }, { kyoku: 1, honba: 0 }],
          outcomes: [fakeOutcome(), fakeOutcome()],
          ledger: [],
          ledgerCuts: [0], // one cut, two rounds
        }, JANKI),
      Error,
      "ledgerCuts/outcomes mismatch",
    );
    assertEquals(writer.stats(), { d: 0, r: 0, m: 0 });
  } finally {
    writer.close();
    Deno.removeSync(path);
  }
});

Deno.test("record: rounds が outcomes と合わなければ書かずに落ちる", () => {
  const path = Deno.makeTempFileSync({ prefix: "mjgame_rid_", suffix: ".jsonl" });
  const writer = new TrajectoryWriter(path);
  try {
    // An "r" line labelled with the wrong round is worse than an unlabelled
    // one, because the loader trusts the label — so refuse the whole match.
    assertThrows(
      () =>
        writeMatchEnd(writer, {
          scores: [30000, 30000, 30000, 30000],
          rounds: [{ kyoku: 0, honba: 0 }], // one round, two outcomes
          outcomes: [fakeOutcome(), fakeOutcome()],
          ledger: [],
          ledgerCuts: [0, 0],
        }, JANKI),
      Error,
      "rounds/outcomes mismatch",
    );
    assertEquals(writer.stats(), { d: 0, r: 0, m: 0 });
  } finally {
    writer.close();
    Deno.removeSync(path);
  }
});

Deno.test("record: 実戦でも局ごとの合計は台帳と一致する", () => {
  for (const seed of [966, 12345, 40000, 777]) {
    const policies = SEATS.map((s) => new RandomPolicy(`R${s}`, seed * 4 + s));
    const r = runMatchSync(policies, {
      seed,
      cfg: JANKI,
      dojo: DOJO_HEADLESS,
      scorer,
      ...makeDojoHooks(DOJO_HEADLESS),
    });
    const cuts = r.ledgerCuts;
    assertEquals(cuts.length, r.outcomes.length, `seed ${seed}: one cut per round`);
    for (let i = 1; i < cuts.length; i++) {
      assert(cuts[i] >= cuts[i - 1], `seed ${seed}: cuts not monotone`);
    }
    assertEquals(cuts[cuts.length - 1], r.ledger.length, `seed ${seed}: cuts miss the tail`);

    // Per-round attribution partitions the ledger: the sums must reconstruct
    // the match total exactly, with nothing double-counted or dropped.
    const total = [0, 0, 0, 0];
    for (let k = 0; k < cuts.length; k++) {
      const slice = r.ledger.slice(cuts[k - 1] ?? 0, cuts[k]);
      // Every violation in a round's slice really happened in that kyoku.
      for (const v of slice) {
        assertEquals(v.kyoku, r.rounds[k].kyoku, `seed ${seed}: round ${k} kyoku mismatch`);
      }
      for (const v of slice) total[v.seat] += v.points;
    }
    const whole = [0, 0, 0, 0];
    for (const v of r.ledger) whole[v.seat] += v.points;
    assertEquals(total, whole, `seed ${seed}: attribution lost points`);
  }
});

// ---------------------------------------------------------------------------
// oracle block (optional, trainer-side only)
// ---------------------------------------------------------------------------

/**
 * The same hanchan recorded twice: once with the driver's oracle tap wired up,
 * once without it. `--record` always wires it, but the field is optional on the
 * wire — a dataset written without it must still be a legal dataset.
 */
function recordOracle(path: string, seed: number, wired: boolean): Record<string, unknown>[] {
  const writer = new TrajectoryWriter(path);
  const ref: { t: Table | null } = { t: null };
  const policies = SEATS.map((s) =>
    new RecordingPolicy(
      new RandomPolicy(`R${s}`, seed * 4 + s),
      writer,
      wired ? (sq) => encodeOracle(ref.t!, sq as Seat) : undefined,
    )
  );
  const result = runMatchSync(policies, {
    seed,
    cfg: JANKI,
    dojo: DOJO_HEADLESS,
    scorer,
    tableRef: ref,
    ...makeDojoHooks(DOJO_HEADLESS),
  });
  writeMatchEnd(writer, result, JANKI);
  writer.close();
  return Deno.readTextFileSync(path).split("\n").filter((l) => l !== "").map(
    (l) => JSON.parse(l) as Record<string, unknown>,
  );
}

Deno.test("record: オラクル情報は全 d 行に o/sh として乗る", () => {
  const path = Deno.makeTempFileSync({ prefix: "mjgame_oracle_", suffix: ".jsonl" });
  try {
    const ds = recordOracle(path, 966, true).filter((l) => l.k === "d");
    assert(ds.length > 0, "nothing was recorded");
    let sawKnownOpponent = false;
    for (const [i, d] of ds.entries()) {
      const where = `d line ${i + 1}`;
      const o = d.o as string;
      assertEquals(typeof o, "string", `${where}: no oracle block`);
      assertEquals(o.length, 228, `${where}: base64 length (170 bytes)`);
      const bytes = fromBase64(o);
      assertEquals(bytes.length, ORACLE_LEN, `${where}: oracle bytes`);
      // v3 widened the POLICY input only ("planes"/"scalars"); the hidden-state
      // block is byte for byte the 170 it has always been.
      assertEquals(ORACLE_LEN, 170);
      for (const b of bytes) assert(b >= 0 && b <= 4, `${where}: oracle cell ${b}`);
      // Each opponent holds 13 tiles minus whatever it has melded, so the first
      // three planes are never all zero on a real board.
      const held = bytes.subarray(0, 3 * 34).reduce((a, b) => a + b, 0);
      assert(held > 0, `${where}: no opponent tiles at all`);
      if (held === 39) sawKnownOpponent = true;

      const sh = d.sh as number[];
      assert(Array.isArray(sh) && sh.length === 3, `${where}: sh ${JSON.stringify(sh)}`);
      for (const v of sh) {
        assert(Number.isInteger(v) && v >= -1 && v <= 8, `${where}: shanten ${v}`);
      }
      // The oracle block does not move the version on its own — "v" is the
      // POLICY encoder's, and it says v4 because of the river token stream.
      assertEquals(d.v, FEATURES.version, `${where}: feature version`);
      assertEquals(d.v, 4, `${where}: feature version`);
      assertEquals(fromBase64(d.planes as string).length, PLANE_LEN, `${where}: planes`);
    }
    assert(sawKnownOpponent, "no decision saw three fully concealed opponents");
  } finally {
    Deno.removeSync(path);
  }
});

Deno.test("record: タップ無しでは o も sh も現れず、他の列は1バイトも変わらない", () => {
  const bare = Deno.makeTempFileSync({ prefix: "mjgame_bare_", suffix: ".jsonl" });
  const wired = Deno.makeTempFileSync({ prefix: "mjgame_wired_", suffix: ".jsonl" });
  try {
    const without = recordOracle(bare, 966, false);
    const with_ = recordOracle(wired, 966, true);
    for (const l of without) {
      assert(!("o" in l), "an oracle block appeared without a tap");
      assert(!("sh" in l), "shanten labels appeared without a tap");
    }
    // Same seed, same policies: the oracle must be a pure addition, so stripping
    // it reproduces the old file line for line, key order included.
    assertEquals(with_.length, without.length);
    with_.forEach((l, i) => {
      const stripped = { ...l };
      delete stripped.o;
      delete stripped.sh;
      assertEquals(JSON.stringify(stripped), JSON.stringify(without[i]), `line ${i + 1}`);
    });
  } finally {
    Deno.removeSync(bare);
    Deno.removeSync(wired);
  }
});

// ---------------------------------------------------------------------------
// encoding helpers
// ---------------------------------------------------------------------------

Deno.test("record: toBase64/fromBase64 が往復する", () => {
  const rng = sfc32(0xc0ffee);
  // 0 and 1 are the edges; 0x8000 is exactly the chunk boundary `toBase64`
  // splits on, and 0x8001/0x14000 straddle it.
  for (const n of [0, 1, 2, 3, 255, 1000, 0x7fff, 0x8000, 0x8001, 0x14000]) {
    const bytes = new Uint8Array(n);
    for (let i = 0; i < n; i++) bytes[i] = rng.int(256);
    const round = fromBase64(toBase64(bytes));
    assertEquals(round.length, n, `length for n=${n}`);
    assertEquals(round, bytes, `round trip for n=${n}`);
  }
  // Every byte value survives, including the ones that are not valid UTF-8.
  const all = new Uint8Array(256);
  for (let i = 0; i < 256; i++) all[i] = i;
  assertEquals(fromBase64(toBase64(all)), all);
  assertEquals(toBase64(new Uint8Array([0x4d, 0x61, 0x6e])), "TWFu"); // the canonical vector
});

/**
 * Every "seq" a real hanchan wrote, decoded and matched against `encodeSeq` run
 * on the very Observation that produced the line.
 *
 * The wrapper is a SPY, not a re-encode after the fact: an Observation mutates
 * as the round goes on (the rivers it exposes are the board's own arrays), so
 * the only honest comparison is the one taken at decision time — which is also
 * the only way to catch a writer that encoded a stale or a shared buffer.
 */
Deno.test("record: d 行の seq は決断時の encodeSeq のバイト列そのもの", () => {
  const path = Deno.makeTempFileSync({ prefix: "mjgame_seq_", suffix: ".jsonl" });
  try {
    const writer = new TrajectoryWriter(path);
    const seen: Int8Array[] = [];
    const policies = SEATS.map((s) => {
      const inner = new RandomPolicy(`R${s}`, 966 * 4 + s);
      const spy: SyncPolicy = {
        name: inner.name,
        reset(seed: number) {
          inner.reset(seed);
        },
        decide(obs: Observation): Action {
          const a = inner.decide(obs);
          // AFTER the inner decision, exactly where RecordingPolicy encodes.
          seen.push(encodeSeq(obs));
          return a;
        },
      };
      return new RecordingPolicy(spy, writer);
    });
    const result = runMatchSync(policies, {
      seed: 966,
      cfg: JANKI,
      dojo: DOJO_HEADLESS,
      scorer,
      ...makeDojoHooks(DOJO_HEADLESS),
    });
    writeMatchEnd(writer, result, JANKI);
    writer.close();

    const ds = Deno.readTextFileSync(path).split("\n").filter((l) => l !== "")
      .map((l) => JSON.parse(l) as Record<string, unknown>).filter((l) => l.k === "d");
    assertEquals(ds.length, seen.length, "one d line per decision");
    assert(ds.length > 0, "nothing was recorded");

    let empty = 0;
    ds.forEach((d, i) => {
      const want = seen[i];
      const got = fromBase64(d.seq as string);
      assertEquals(got.length, want.length, `d line ${i + 1}: seq byte count`);
      // Compare as SIGNED bytes — base64 round-trips unsigned, and the trainer
      // reads the block back as int8.
      assertEquals(
        Array.from(new Int8Array(got.buffer, got.byteOffset, got.byteLength)),
        Array.from(want),
        `d line ${i + 1}: seq bytes`,
      );
      if (want.length === 0) empty++;
    });
    // The L = 0 case is not hypothetical: it is every 親's first decision.
    assert(empty > 0, "no decision was made before anyone discarded");
    assert(empty < ds.length, "every seq was empty — the rivers never reached the encoder");
    // …and the field survives an empty stream as an empty STRING, not a hole.
    const first = ds.find((d) => (d.seq as string) === "");
    assert(first !== undefined, 'an empty seq was not written as ""');
    assertEquals(fromBase64("").length, 0);
  } finally {
    Deno.removeSync(path);
  }
});

Deno.test("record: f32leBytes はリトルエンディアン", () => {
  // 1.0 = 0x3F800000, 2.5 = 0x40200000, −1.0 = 0xBF800000 — low byte first.
  assertEquals(
    Array.from(f32leBytes(new Float32Array([1, 2.5, -1]))),
    [0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0x40, 0x00, 0x00, 0x80, 0xbf],
  );
  assertEquals(f32leBytes(new Float32Array(0)).length, 0);

  // ...and the trainer's own read-back (DataView, littleEndian=true) agrees.
  const rng = sfc32(7);
  const a = new Float32Array(FEATURES.scalars);
  for (let i = 0; i < a.length; i++) a[i] = (rng.float() * 2 - 1) * 100;
  const bytes = f32leBytes(a);
  assertEquals(bytes.length, SCALAR_BYTES);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < a.length; i++) assertAlmostEquals(dv.getFloat32(i * 4, true), a[i], 0);
});
