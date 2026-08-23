// M11 wiring: the own-hand value model as the policy consumes it.
//
// The MODEL is tested in `handvalue_test.ts` and the FILE in `handcalib_test.ts`;
// what is left — and what this file is for — is the seam between them, which has
// exactly three claims to make:
//
//   1. ABSENT MEANS ABSENT. No `hand` block, no change: not "almost", not
//      "statistically indistinguishable", the same hanchan tile for tile. The
//      pinned decision streams in `computed_test.ts` / `calibration_test.ts`
//      carry most of that claim already; the half they cannot carry is the
//      RECORDER, which runs on a seat with no `hand` block at all and must
//      therefore be invisible to it. Its mirror is also here: with a `hand`
//      block the seat plays a DIFFERENT game, or the model is wired to nothing.
//   2. `threatOf` is `pressureOf` broken out per seat, and the augmented seat
//      answers it from `tenpaiP` when it has one.
//   3. One sample per turn decision, and nothing the fit cannot use.

import { assert, assertEquals, assertNotEquals } from "@std/assert";
import type { DangerAssessment } from "mjrender/danger.ts";
import type { HandSample } from "../src/ai/handcalib.ts";
import type { HandWeights } from "../src/ai/handvalue.ts";
import { DEFAULT_HAND, handOutlook, mergeHand } from "../src/ai/handvalue.ts";
import type { Reads } from "../src/ai/augmented.ts";
import { AugmentedHeuristic } from "../src/ai/augmented.ts";
import { HeuristicPolicy } from "../src/ai/heuristic.ts";
import type { Observation } from "../src/observe.ts";
import type { Action } from "../src/types.ts";
import { playHanchan } from "./helpers.ts";

const SEEDS = [101, 404, 505];

// ---------------------------------------------------------------------------
// 1. wired, and only when asked for
// ---------------------------------------------------------------------------

/** Seat 0 carries the options under test; the other three are plain. */
function stream(
  seed: number,
  opts: { hand?: HandWeights; sink?: (r: HandSample) => void },
): string {
  const r = playHanchan(seed, (s) =>
    s === 0
      ? new HeuristicPolicy(`H${s}`, seed * 4 + s, {
        hand: opts.hand,
        handSink: opts.sink,
      })
      : new HeuristicPolicy(`H${s}`, seed * 4 + s));
  return JSON.stringify({
    scores: r.scores,
    outcomes: r.outcomes,
    ledger: r.ledger.map((v) => `${v.seat}:${v.label}`),
    riichis: r.riichis,
    furo: r.furoRounds,
  });
}

Deno.test("M11: without a hand block the seat is the old policy, bit for bit", () => {
  for (const seed of SEEDS) {
    const plain = stream(seed, {});
    // The recorder is an observer, not a participant: a lane has to be played by
    // the seat that ships, so the sink must not cost a single decision. (It is
    // also the only thing in M11 that runs with no weights attached — the
    // samples are evaluated under DEFAULT_HAND, which the file header records.)
    let seen = 0;
    assertEquals(
      stream(seed, { sink: () => seen++ }),
      plain,
      `種${seed}: handSink が打牌を動かしている`,
    );
    assert(seen > 0, `種${seed}: 記録が一件も出ていないなら、この比較は空`);
  }
});

Deno.test("M11: a hand block reaches the decision — the streams diverge", () => {
  let diverged = 0;
  for (const seed of SEEDS) {
    if (stream(seed, { hand: DEFAULT_HAND }) !== stream(seed, {})) diverged++;
  }
  // Not per-seed: the model replaces the push table and the value term, and a
  // hanchan where neither ever decided anything is a legitimate outcome. Zero
  // out of three is not — that would mean the option is inert.
  assert(diverged > 0, "hand ブロックを与えても何一つ変わらないなら、配線されていない");
});

Deno.test("M11: mergeHand feeds the policy a complete vector from a partial", () => {
  // `--ktune`'s `hand` section is a PARTIAL, and `makePolicy` merges it before
  // the policy ever sees it. A vector missing `pushScale` would divide the
  // fold gate's EV by undefined and fold on NaN — silently, and always.
  const w = mergeHand({ valuePerDora: 2500 });
  assertEquals(w.valuePerDora, 2500);
  assertEquals(w.pushScale, DEFAULT_HAND.pushScale);
  assertEquals(w.meanUkeire, DEFAULT_HAND.meanUkeire);
  assertNotEquals(stream(SEEDS[0], { hand: w }), "", "そのまま一局打てる完全なベクトル");
});

// ---------------------------------------------------------------------------
// 2. threatOf — pressureOf, per seat
// ---------------------------------------------------------------------------

/** Reaches the protected hook the way `handFacts` does (augmented_test.ts:266). */
class Probe extends AugmentedHeuristic {
  threatWith(obs: Observation, reads: Reads | null): number[] {
    this.reads = reads;
    try {
      return this.threatOf(obs);
    } finally {
      this.reads = null;
    }
  }
}

/** A furo threat detail against ABSOLUTE seat `seat`, as mjrender writes them. */
function furoDanger(seat: number): DangerAssessment {
  return {
    level: "危険度中",
    seats: [seat],
    details: [{ seat, level: "危険度中", kind: "furo", openMeldCount: 2, notes: [] }],
  };
}

function threatObs(seat: number, riichi: boolean[], danger: Map<number, DangerAssessment>) {
  return { seat, riichi, danger } as unknown as Observation;
}

Deno.test("M11: threatOf reads 立直 full and a 副露 threat half, in relative order", () => {
  const p = new Probe("P", 1, () => null);
  // We are absolute seat 2. 対面 (absolute 0) is relative index 1; 上家
  // (absolute 1) is relative index 2 and has declared.
  const obs = threatObs(2, [false, false, false, true], new Map([[0, furoDanger(0)]]));
  assertEquals(p.threatWith(obs, null), [0, 0.5, 1]);
});

Deno.test("M11: a declared 立直 outranks the 副露 reading on the same seat", () => {
  const p = new Probe("P", 1, () => null);
  // 下家 (relative 1) is both open and declared: the public fact wins.
  const obs = threatObs(0, [false, true, false, false], new Map([[5, furoDanger(1)]]));
  assertEquals(p.threatWith(obs, null), [1, 0, 0]);
});

Deno.test("M11: the augmented seat answers threatOf from tenpaiP", () => {
  const p = new Probe("P", 1, () => null);
  const obs = threatObs(0, [false, false, false, true], new Map());

  // No channel ⇒ the base reading stands: only the declared 上家 (relative 2).
  assertEquals(p.threatWith(obs, {}), [0, 0, 1]);

  // With one, the estimate is used — except where a declaration outranks it, and
  // except outside [0,1], which no probability may leave.
  assertEquals(p.threatWith(obs, { tenpaiP: [0.4, 0, 0.1] }), [0.4, 0, 1]);
  assertEquals(p.threatWith(obs, { tenpaiP: [-0.2, 1.7, 0] }), [0, 1, 1]);
});

// ---------------------------------------------------------------------------
// 3. the recorder
// ---------------------------------------------------------------------------

/** Counts the turn decisions that actually reach the discard chooser. */
class Counting extends HeuristicPolicy {
  turns = 0;
  override decide(obs: Observation): Action {
    const a = super.decide(obs);
    // A forced move short-circuits before `chooseDiscard`, and a tsumo/kan/call
    // never gets there either — so the sample is owed exactly when a turn
    // decision with a real choice ends in a discard.
    if (obs.drawn !== null && obs.legal.length > 1 && a.t === "discard") this.turns++;
    return a;
  }
}

Deno.test("M11: one sample per turn decision, and every one of them is usable", () => {
  const seed = 707;
  const samples: HandSample[] = [];
  let counter!: Counting;
  playHanchan(seed, (s) => {
    if (s !== 0) return new HeuristicPolicy(`H${s}`, seed * 4 + s);
    counter = new Counting(`H${s}`, seed * 4 + s, {
      hand: DEFAULT_HAND,
      handSink: (r) => samples.push(r),
    });
    return counter;
  });

  assert(samples.length > 0, "半荘を打って自摸番が一度もないことはない");
  assertEquals(samples.length, counter.turns, "自摸番の数と記録数は一致していなければならない");

  for (const s of samples) {
    assert(s.pwin >= 0 && s.pwin <= 1, `pwin が確率でない: ${s.pwin}`);
    assert(s.value > 0, `和了価値が正でない: ${s.value}`);
    // The record carries the FACTS, and the fit re-enters `handOutlook` through
    // them — so a fact vector that does not describe a 13-tile resting shape
    // would poison the fit rather than merely mispredict.
    const f = s.facts;
    assert(f.shanten >= 0, `向聴が負: ${f.shanten}`);
    assert(f.ukeire >= 0 && f.ukeire <= f.unseenTotal, "受け入れは未見枚数を超えない");
    assert(f.unseenTotal > 0, "未見が0枚なら分母が壊れている");
    assert(f.turnsLeft >= 0, "残り自摸が負");
    assertEquals(f.oppTenpai.length, 3, "相手は三人");
    // 13 concealed + up to four kan tiles, and a 赤5筒 that is also indicator
    // dora counts twice.
    assert(f.dora >= 0 && f.dora <= 34, `ドラ枚数が範囲外: ${f.dora}`);
  }
});

Deno.test("M11: the recorded facts re-enter the model and reproduce it exactly", () => {
  // THE POINT OF STORING FACTS AND NOT FEATURES. `scripts/hand_fit.ts` re-scores
  // every record under a candidate weight vector by calling `handOutlook` on the
  // stored `facts`. That is only sound if the facts ALONE determine what the seat
  // predicted — no hidden state left on the policy, nothing rounded away on the
  // way out. So: replay each sample through the same function under the same
  // weights and demand the identical double, not a close one.
  const seed = 808;
  const samples: HandSample[] = [];
  playHanchan(
    seed,
    (s) =>
      s === 0
        // No `hand` block on purpose: this is the pre-fit lane, whose samples are
        // the DEFAULT_HAND reading of a seat that does not consume them.
        ? new HeuristicPolicy(`H${s}`, seed * 4 + s, { handSink: (r) => samples.push(r) })
        : new HeuristicPolicy(`H${s}`, seed * 4 + s),
  );

  assert(samples.length > 0);
  for (const s of samples) {
    const again = handOutlook(s.facts, DEFAULT_HAND);
    assertEquals(again.pwin, s.pwin, "再評価した和了確率が記録と食い違う");
    assertEquals(again.value, s.value, "再評価した和了価値が記録と食い違う");
    assertEquals(again.ev, again.pwin * again.value, "ev は pwin × value そのもの");
  }
});
