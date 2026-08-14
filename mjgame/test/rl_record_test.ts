// Trajectory recording: the JSONL a Python/MLX trainer eats.
//
// The file layout is a frozen contract, so this test reads the bytes back the
// way the trainer will — decode base64, count bytes, re-derive the settlement —
// rather than trusting the writer's own helpers to agree with themselves.

import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import { RandomPolicy } from "../src/ai/random.ts";
import { runMatchSync } from "../src/match.ts";
import type { MatchResult } from "../src/match.ts";
import { makeDojoHooks } from "../src/main.ts";
import { FEATURES, PLANE_LEN } from "../src/rl/features.ts";
import { ACTIONS } from "../src/rl/actionspace.ts";
import {
  f32leBytes,
  fromBase64,
  RecordingPolicy,
  settlement,
  toBase64,
  TrajectoryWriter,
  writeMatchEnd,
} from "../src/rl/record.ts";
import { DOJO_HEADLESS, JANKI } from "../src/rules.ts";
import { sfc32 } from "../src/rng.ts";
import { scorer } from "../src/score.ts";
import { SEATS } from "../src/types.ts";

const SCALAR_BYTES = 39 * 4; // 156: the frozen scalar block (feature v2)

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

/** 供託 still on the table at the end — the last round's sticks, unless won. */
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
    const { result, lines, stats } = record(path, 966);
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
        assertEquals(PLANE_LEN, 1224); // pins the plane count itself (36 × 34)
        // The trainer refuses a line whose feature version it does not know, so
        // every line must carry it — a missing "v" means "v1" over there.
        assertEquals(d.v, 2, `${where}: feature version`);
        assertEquals(d.v, FEATURES.version, `${where}: version tracks the encoder`);

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

    await t.step("m 行: 点棒保存・精算・違反点", () => {
      const m = lines[lines.length - 1];
      const scores = m.scores as number[];
      assertEquals(scores.length, 4);
      for (const s of scores) assert(typeof s === "number" && Number.isFinite(s));
      assertEquals(scores, result.scores);

      // 4×30000 minus whatever is still lying on the table as 供託.
      const sum = scores.reduce((a, b) => a + b, 0);
      assertEquals(
        sum + leftoverKyotaku(result) * 1000,
        JANKI.startScore * 4,
        `${sum} + ${leftoverKyotaku(result)} sticks != ${JANKI.startScore * 4}`,
      );

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
    });
  } finally {
    Deno.removeSync(path);
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

Deno.test("record: f32leBytes はリトルエンディアン", () => {
  // 1.0 = 0x3F800000, 2.5 = 0x40200000, −1.0 = 0xBF800000 — low byte first.
  assertEquals(
    Array.from(f32leBytes(new Float32Array([1, 2.5, -1]))),
    [0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0x40, 0x00, 0x00, 0x80, 0xbf],
  );
  assertEquals(f32leBytes(new Float32Array(0)).length, 0);

  // ...and the trainer's own read-back (DataView, littleEndian=true) agrees.
  const rng = sfc32(7);
  const a = new Float32Array(39);
  for (let i = 0; i < a.length; i++) a[i] = (rng.float() * 2 - 1) * 100;
  const bytes = f32leBytes(a);
  assertEquals(bytes.length, SCALAR_BYTES);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < a.length; i++) assertAlmostEquals(dv.getFloat32(i * 4, true), a[i], 0);
});
