// 押し引きの較正記録 (M13) — what the fold gate decided, what was actually
// played, and how the 局 settled for the seat that decided it.
//
// WHY A LANE AND NOT A LABEL. `handcalib.ts` records a PREDICTION and grades it
// against a truth the round reveals ("did this hand cash, for how much"). There
// is no such truth for a push/fold decision: the round tells us what happened
// after we pushed, and says nothing whatever about what folding would have paid.
// So this lane is a CONTEXTUAL BANDIT, not a supervised set. The seat plays its
// own verdict with probability 1−ε and the OPPOSITE one with probability ε, the
// propensity is written down beside the action, and `train/fold_fit.py` fits a
// doubly-robust estimate of the difference between the two arms. ε > 0 means
// the seat deliberately plays worse than it knows how to — which is why the
// flag is explicit, per-run and never the default.
//
// THE REWARD (plan D7). `delta` — the round's `deltas[0]`, in points — is the
// ONE objective. It is the seat's real settlement for the 局: 本場, 供託, noten
// payments and deal-ins all included, and negative when the round cost us. The
// ledger and the game-end columns below are recorded as DATA and are not in the
// objective, per the project's reward philosophy: violations are minimised as a
// byproduct of long-term reward, never by per-decision shaping.
//
// THE INDEPENDENCE ASSUMPTION, stated once here and again in the fit. Every
// fold decision of one round shares that round's single settlement, so a round
// in which two decisions were flipped credits both flips with the same number.
// The estimator treats decisions as independent draws, which is false at the
// round level; the fit reports the fraction of rounds carrying more than one
// flip so the size of the lie is visible (at ε = 0.05 and ~4 gated decisions a
// round it is small, and it shrinks with ε).
//
// ===========================================================================
// THE FILE FORMAT — JSONL, one header line then one line per FOLD DECISION
// ===========================================================================
//
// Header (line 1):
//   {"v":1,"kind":"mjgame-fold","seats":"khhh","seed":N,"games":N,"eps":0.05,
//    "fv":1,"features":[...37 names...],"head":"gate"}
//   `features` is `FOLD_FEATURES` verbatim, so a file names its own columns;
//   `head` is "gate" when the recorded seat carried NO fold block (the verdict
//   is the incumbent `margin < 0`, which `scripts/fold_report.ts` re-derives
//   from `x[0]` as a reproduction check) and "mlp" when it carried one.
//
// Record (one per fold decision that reached the head region — seat 0):
//   x        the 37 features, in `FOLD_FEATURES` order
//   verdict  what the seat's own rule said (true = fold)
//   taken    what was PLAYED (`verdict` unless the flip fired)
//   p        P(taken | state) under the behaviour policy: 1−ε, or ε when flipped
//   flipped  1 = the ε-flip fired here
//   s        game seed      n  decision index within the game (all rounds)
//   k        kyoku (0 = 東1) b  本場      turn  1 = a draw decision, 0 = a claim
//   junme    the 巡 the decision was made on
//   delta    THE REWARD: the round's `deltas[0]`, in points
//   won / dealtIn / outcome / endJunme  the round's shape, as `handLabels` reads it
//   vio0     ledger entries charged to seat 0 IN THIS ROUND (data, not objective)
//
// LIKE THE HAND LANE, BUFFERED BY ROUND: the reward does not exist until the 局
// ends. A round that never ends (the last round of an aborted run) drops its
// samples rather than inventing a settlement; `close` says how many.
//
// SIZE. One line per gated fold decision — far fewer than the hand lane's one
// per turn, since most turns face a quiet table and never reach the gate — at
// ~700 bytes each (37 floats). A 3000-半荘 lane is tens of megabytes.

import type { FoldSample } from "./fold.ts";
import { FOLD_FEATURES, FOLD_FV } from "./fold.ts";
import { handLabels } from "./handcalib.ts";
import type { Table } from "../table.ts";
import type { RoundOutcome } from "../types.ts";

/** Bumped whenever a field changes meaning. A reader refuses anything else. */
export const FOLD_CALIB_VERSION = 1;
export const FOLD_CALIB_KIND = "mjgame-fold";

/** One line of the file: a sample, where it happened, and the round's reward. */
export interface FoldRecord extends Omit<FoldSample, "flipped" | "turn"> {
  flipped: 0 | 1;
  s: number;
  n: number;
  k: number;
  b: number;
  /** 1 = a draw (turn) decision, 0 = a claim decision. */
  turn: 0 | 1;
  junme: number;
  /** THE REWARD: the round's `deltas[0]` in points (D7). */
  delta: number;
  won: 0 | 1;
  dealtIn: 0 | 1;
  outcome: "agari" | "ryuukyoku";
  endJunme: number;
  /** Ledger entries charged to seat 0 in this round. Data, never the objective. */
  vio0: number;
}

export interface FoldCalibHeader {
  v: number;
  kind: string;
  seats?: string;
  seed?: number;
  games?: number;
  /** The flip probability the lane was played at; 0 = a pure observation lane. */
  eps: number;
  /** `FOLD_FV` — the feature version the columns below belong to. */
  fv: number;
  /** `FOLD_FEATURES`, verbatim. */
  features: readonly string[];
  /** Which rule produced `verdict`: the incumbent gate, or a fitted head. */
  head: "gate" | "mlp";
}

/**
 * What the policy hands the writer, plus what only the DECISION's position
 * knows. The policy emits a `FoldSample` and nothing else; the writer owns the
 * seed, the index and the round — exactly `handcalib.ts`'s division.
 */
interface Pending {
  sample: FoldSample;
  n: number;
  turn: 0 | 1;
  junme: number;
}

// ---------------------------------------------------------------------------
// the writer
// ---------------------------------------------------------------------------

/**
 * Streams fold-calibration JSONL. Opens (and TRUNCATES) the path: a new run is
 * a new dataset, exactly as every other recorder in this tree does.
 *
 * `record` is bound, so `harness.ts` can hand it to a seat as a plain sink.
 */
export class FoldCalibrationWriter {
  readonly path: string;
  private file: Deno.FsFile;
  private enc = new TextEncoder();
  private seed = 0;
  private n = 0;
  private rows = 0;
  private games = 0;
  private dropped = 0;
  private flips = 0;
  /** Rounds in which more than one decision was flipped — the D7 caveat, counted. */
  private multiFlipRounds = 0;
  private rounds = 0;
  /** Samples of the round in progress, in the order they were decided. */
  private pending: Pending[] = [];
  /**
   * `junme` is already IN the feature row (the head reads it), so the record's
   * own 巡 column is read back from there rather than duplicated through a
   * second channel: one number, one source. The index is resolved once.
   */
  private junmeIdx = FOLD_FEATURES.indexOf("junme");

  constructor(path: string, header: Omit<FoldCalibHeader, "v" | "kind" | "fv" | "features">) {
    this.path = path;
    const dir = path.slice(0, path.lastIndexOf("/"));
    if (dir) Deno.mkdirSync(dir, { recursive: true });
    this.file = Deno.openSync(path, { create: true, write: true, truncate: true });
    this.writeLine({
      v: FOLD_CALIB_VERSION,
      kind: FOLD_CALIB_KIND,
      ...header,
      fv: FOLD_FV,
      features: FOLD_FEATURES,
    });
  }

  /** A new game: stamp its seed on everything that follows, restart the index. */
  beginGame(seed: number): void {
    this.seed = seed;
    this.n = 0;
    this.games++;
    this.dropped += this.pending.length;
    this.pending.length = 0;
  }

  /** The sink handed to the policy. Bound, so it can be passed around. */
  readonly record = (sample: FoldSample): void => {
    this.pending.push({
      sample,
      n: this.n++,
      // Stamped by the policy: a claim decision (`obs.drawn === null`) rests on
      // the hand as it already lies, and only the seat can tell the two apart.
      turn: sample.turn ? 1 : 0,
      junme: sample.x[this.junmeIdx] ?? 0,
    });
    if (sample.flipped) this.flips++;
  };

  /** The round is over: label everything it produced and write it out. */
  endRound(t: Table, outcome: RoundOutcome): void {
    // Reused verbatim: the round's shape is the same fact for both lanes, and
    // two implementations of "did seat 0 win this" would be one too many.
    const lab = handLabels(t, outcome);
    // …but the REWARD is not `handLabels`' `winPoints`, which is 0 unless we
    // won. D7's objective is the settlement itself, deal-ins and noten payments
    // included, or the fit would see every loss as a zero.
    const delta = outcome.deltas[0];
    let vio0 = 0;
    for (const v of t.ledger) if (v.seat === 0) vio0++;
    let flipsHere = 0;
    for (const p of this.pending) {
      const { flipped, turn: _turn, ...rest } = p.sample;
      if (flipped) flipsHere++;
      this.writeLine(
        {
          ...rest,
          flipped: flipped ? 1 : 0,
          s: this.seed,
          n: p.n,
          k: lab.k,
          b: lab.b,
          turn: p.turn,
          junme: p.junme,
          delta,
          won: lab.won,
          dealtIn: lab.dealtIn,
          outcome: lab.outcome,
          endJunme: lab.endJunme,
          vio0,
        } satisfies FoldRecord,
      );
      this.rows++;
    }
    if (this.pending.length > 0) this.rounds++;
    if (flipsHere > 1) this.multiFlipRounds++;
    this.pending.length = 0;
  }

  private writeLine(obj: unknown): void {
    const bytes = this.enc.encode(JSON.stringify(obj) + "\n");
    let n = 0;
    while (n < bytes.length) n += this.file.writeSync(bytes.subarray(n));
  }

  stats(): {
    games: number;
    rows: number;
    dropped: number;
    flips: number;
    rounds: number;
    multiFlipRounds: number;
  } {
    return {
      games: this.games,
      rows: this.rows,
      dropped: this.dropped + this.pending.length,
      flips: this.flips,
      rounds: this.rounds,
      multiFlipRounds: this.multiFlipRounds,
    };
  }

  close(): void {
    this.dropped += this.pending.length;
    this.pending.length = 0;
    this.file.close();
  }
}

// ---------------------------------------------------------------------------
// reading one back
// ---------------------------------------------------------------------------

function checkHeader(h: FoldCalibHeader, path: string): void {
  if (h.kind !== FOLD_CALIB_KIND) throw new Error(`${path}: 押し引きの較正記録ではありません`);
  if (h.v !== FOLD_CALIB_VERSION) {
    throw new Error(
      `${path}: 版が違います (v${h.v}, 期待 v${FOLD_CALIB_VERSION})。` +
        "selfplay --foldcalib で取り直してください",
    );
  }
  if (h.fv !== FOLD_FV) {
    throw new Error(
      `${path}: 特徴量版が違います (fv${h.fv}, 期待 fv${FOLD_FV})。` +
        "selfplay --foldcalib で取り直してください",
    );
  }
}

export interface FoldCalibFile {
  header: FoldCalibHeader;
  records: FoldRecord[];
}

/** Parse a JSONL fold-calibration file. A wrong version is fatal, never coerced. */
export function parseFoldCalibration(text: string, path = "<memory>"): FoldCalibFile {
  const lines = text.split("\n").filter((l) => l.trim() !== "");
  if (lines.length === 0) throw new Error(`${path}: 空のファイルです`);
  const header = JSON.parse(lines[0]) as FoldCalibHeader;
  checkHeader(header, path);
  return { header, records: lines.slice(1).map((l) => JSON.parse(l) as FoldRecord) };
}

export function readFoldCalibration(path: string): FoldCalibFile {
  return parseFoldCalibration(Deno.readTextFileSync(path), path);
}

/**
 * The same file, one record at a time and nothing retained — `scanHandCalibration`'s
 * twin. The header is parsed before any record is delivered, so a version
 * mismatch fails before the caller has accumulated anything.
 */
export async function scanFoldCalibration(
  path: string,
  onRecord: (rec: FoldRecord, header: FoldCalibHeader) => void,
): Promise<FoldCalibHeader> {
  const file = await Deno.open(path, { read: true });
  const dec = new TextDecoder();
  let header: FoldCalibHeader | null = null;
  let rest = "";
  const take = (line: string) => {
    if (line.trim() === "") return;
    if (header === null) {
      header = JSON.parse(line) as FoldCalibHeader;
      checkHeader(header, path);
      return;
    }
    onRecord(JSON.parse(line) as FoldRecord, header);
  };
  try {
    const buf = new Uint8Array(1 << 20);
    for (;;) {
      const n = await file.read(buf);
      if (n === null) break;
      rest += dec.decode(buf.subarray(0, n), { stream: true });
      let i = rest.indexOf("\n");
      while (i >= 0) {
        take(rest.slice(0, i));
        rest = rest.slice(i + 1);
        i = rest.indexOf("\n");
      }
    }
    take(rest);
  } finally {
    file.close();
  }
  if (header === null) throw new Error(`${path}: 空のファイルです`);
  return header;
}
