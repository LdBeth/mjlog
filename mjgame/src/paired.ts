// `paired` — the measuring instrument: the same walls played twice, once by the
// candidate seat layout and once by the control, so the difference between the
// two arms is the only thing left in the numbers.

import { closeArm, openArm, playGame } from "./harness.ts";
import type { Arm, HeadlessOptions } from "./harness.ts";
import type { MatchResult } from "./match.ts";
import { JANKI } from "./rules.ts";
import { finalStandings } from "./score.ts";
import type { Seat } from "./types.ts";

/** Seat 0's placement by raw score. Ties go to the lower seat, as in selfplay. */
function rankOfSeat0(scores: number[]): number {
  let rank = 1;
  for (let s = 1; s < 4; s++) if (scores[s] > scores[0]) rank++;
  return rank;
}

/**
 * Seat 0's placement by the DOJO's own ranking: every seat carrying a ledger
 * entry finishes below every clean seat, whatever the scores say
 * (`score.ts::finalStandings`). This is the number the 雀鬼会 would read off the
 * table, and the one a compliance filter is actually optimising — a seat can
 * gain 8000 points and still lose the placement by filing one violation, which
 * the raw 順位差 line cannot show.
 *
 * 起家 is read off the first round rather than assumed: `match.ts::roundInit`
 * deals East-1 to seat 0 today, but the tie-break should not depend on that.
 */
function dojoRankOfSeat0(r: MatchResult): number {
  const east = (r.rounds[0]?.dealer ?? 0) as Seat;
  return finalStandings(r.scores, east, r.ledger, JANKI)
    .find((s) => s.seat === 0)!.place;
}

export interface Diff {
  mean: number;
  sd: number;
  ci: number;
}

/** Sample mean, sample SD (n-1) and the 95% half-width 1.96·sd/√n. */
export function summarize(xs: number[]): Diff {
  const n = xs.length;
  if (n === 0) return { mean: 0, sd: 0, ci: 0 };
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  if (n < 2) return { mean, sd: 0, ci: 0 };
  const varSum = xs.reduce((a, x) => a + (x - mean) * (x - mean), 0);
  const sd = Math.sqrt(varSum / (n - 1));
  return { mean, sd, ci: 1.96 * sd / Math.sqrt(n) };
}

export interface PairedStats {
  games: number;
  seed: number;
  seats: string;
  /** Seat 0's mean rank / mean final score in the test arm and the control. */
  rankA: number;
  rankB: number;
  scoreA: number;
  scoreB: number;
  /** The same means under the dojo's ranking (violators below clean players). */
  rankDojoA: number;
  rankDojoB: number;
  /** Per-seed differences A−B: negative rank diff = the test arm placed better. */
  dRank: Diff;
  /** Same difference, ranked the way the dojo ranks. */
  dRankDojo: Diff;
  dScore: Diff;
  /** Sign counts on the rank diff. */
  better: number;
  tie: number;
  worse: number;
  /** Total ledger entries across all seats, per arm. */
  vioA: number;
  vioB: number;
  /**
   * Seat 0's ledger entries by rule label, per arm. Seat 0 is the only seat
   * whose policy differs between the arms, so this is the breakdown that says
   * WHICH rules an augmented seat trades away — the all-seat totals above
   * cannot (seats 1–3 drift too, because the games diverge).
   */
  vioA0: Map<string, number>;
  vioB0: Map<string, number>;
  ms: number;
}

/**
 * The paired ablation harness: every seed is played TWICE, once with `seats`
 * and once all-heuristic, with identical match seeds and identical per-seat
 * policy seeds. Identical seeds mean identical walls AND identical policy dice,
 * so the only thing that differs between the two arms is what seat 0 knows —
 * which is what makes a per-seed difference meaningful and the variance small
 * enough to see a rank effect in hundreds rather than tens of thousands of
 * games.
 *
 * Both arms are BUILT ONCE and re-seeded per seed (`playGame`), so a thousand
 * seeds cost two sets of seats rather than eight thousand.
 *
 * Exported so a test (and any later sweep script) can measure without a CLI.
 */
export function pairedRun(
  games: number,
  seed: number,
  seats: string,
  opts: HeadlessOptions = {},
): PairedStats {
  const dRank: number[] = [];
  const dRankDojo: number[] = [];
  const dScore: number[] = [];
  let rankA = 0, rankB = 0, scoreA = 0, scoreB = 0, vioA = 0, vioB = 0;
  let rankDojoA = 0, rankDojoB = 0;
  let better = 0, tie = 0, worse = 0;
  const vioA0 = new Map<string, number>();
  const vioB0 = new Map<string, number>();
  const bump = (m: Map<string, number>, k: string) => m.set(k, (m.get(k) ?? 0) + 1);
  const t0 = performance.now();

  const armA = openArm(seats, opts);
  // The control arm is handed ONE option, and it is not a tuning knob: the
  // manifest path, so an "n" seat in `seats` has a comparable baseline to load.
  // Nothing that could move seat 0's play — `oracle`, `noise`, `plan`, `ktune`,
  // `standings` — may cross this line, or the difference below would be measured
  // against a baseline the candidate had already edited. (hhhh has no "k" or "o"
  // seat to read them anyway; the point is that it stays that way.)
  //
  // …UNLESS `consumerB` (M9c-b) or `ktuneB` (M10d) names an INCUMBENT. Then the
  // control arm is the same seat, the same everything, carrying a DIFFERENT
  // curve set / a DIFFERENT 感性 vector — and the difference measured is
  // candidate minus incumbent rather than candidate minus baseline. Whichever of
  // the two is absent is inherited from arm A, so the two arms still differ in
  // exactly one file. See `consumerB` / `ktuneB`.
  const incumbent = opts.consumerB !== undefined || opts.ktuneB !== undefined;
  // Neither writer crosses either: a second `record` would open a second handle
  // on arm A's trajectory file, and the control arm is not what is being
  // calibrated — its decisions would interleave into arm A's records under the
  // same seed. Stripped structurally rather than overridden, so a field added to
  // `HeadlessOptions` later cannot reintroduce them by being spread.
  const { record: _record, calibrate: _calibrate, ...shared } = opts;
  const armB: Arm = incumbent
    ? openArm(seats, {
      ...shared,
      consumer: opts.consumerB ?? opts.consumer,
      ktune: opts.ktuneB ?? opts.ktune,
    })
    : openArm("hhhh", { weights: opts.weights });

  try {
    for (let g = 0; g < games; g++) {
      const s = seed + g;
      const a = playGame(armA, s);
      const b = playGame(armB, s);
      const ra = rankOfSeat0(a.scores);
      const rb = rankOfSeat0(b.scores);
      const da = dojoRankOfSeat0(a);
      const db = dojoRankOfSeat0(b);
      rankA += ra;
      rankB += rb;
      rankDojoA += da;
      rankDojoB += db;
      dRankDojo.push(da - db);
      scoreA += a.scores[0];
      scoreB += b.scores[0];
      vioA += a.ledger.length;
      vioB += b.ledger.length;
      for (const v of a.ledger) if (v.seat === 0) bump(vioA0, v.label);
      for (const v of b.ledger) if (v.seat === 0) bump(vioB0, v.label);
      dRank.push(ra - rb);
      dScore.push(a.scores[0] - b.scores[0]);
      if (ra < rb) better++;
      else if (ra === rb) tie++;
      else worse++;
    }
  } finally {
    closeArm(armA);
    closeArm(armB);
  }

  return {
    games,
    seed,
    seats,
    rankA: rankA / games,
    rankB: rankB / games,
    scoreA: scoreA / games,
    scoreB: scoreB / games,
    rankDojoA: rankDojoA / games,
    rankDojoB: rankDojoB / games,
    dRank: summarize(dRank),
    dRankDojo: summarize(dRankDojo),
    dScore: summarize(dScore),
    better,
    tie,
    worse,
    vioA,
    vioB,
    vioA0,
    vioB0,
    ms: performance.now() - t0,
  };
}

/**
 * `paired --json`: the same measurement as the tables in `main.ts`, in one line
 * a search loop can parse. Only the fields a tuner grades on — the Maps become
 * plain objects, and everything the human tables derive (勝敗 counts, the
 * per-arm score means) is left out on purpose: what is not printed cannot be
 * optimised against by accident.
 */
export function pairedJson(st: PairedStats): string {
  const obj = {
    games: st.games,
    seed: st.seed,
    seats: st.seats,
    rankA: st.rankA,
    rankB: st.rankB,
    dRank: st.dRank,
    dRankDojo: st.dRankDojo,
    dScore: st.dScore,
    vioA0: Object.fromEntries(st.vioA0),
    vioB0: Object.fromEntries(st.vioB0),
    ms: st.ms,
  };
  return JSON.stringify(obj);
}
