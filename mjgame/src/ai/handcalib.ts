// 手牌価値の較正記録 (M11) — what the own-hand model predicted, and how the
// round actually ended for the seat that predicted it.
//
// WHY. `src/ai/calibration.ts` grades the OTHER half of the 計算 seat: what the
// three opponents are holding. This grades OUR half — `handOutlook`'s two
// numbers, P(this hand wins the round) and what it pays when it does. Those two
// numbers are consumed in exactly two places (the fold gate's push term and the
// discard score's EV term), so a systematic error in either is a systematic
// error in every push/fold decision the seat makes, and no rank measurement can
// say which of the two is wrong. A per-decision record can.
//
// THE PRINCIPLE, unchanged from M10a: FEATURE EXTRACTION AT RECORD TIME,
// CLOSED-FORM MODEL AT FIT TIME. A record carries the `HandFacts` the policy
// built — parameter-free counts off the Observation (shanten, ukeire, unseen,
// dora, melds, 巡目, the three opponents' tenpai reads) — plus the answer the
// live seat got from them. So an offline fit re-evaluates any candidate
// `HandWeights` by calling `handOutlook` on the cached facts: the SAME function
// the seat called, never a second implementation of it. `pwin`/`value` are
// stored alongside precisely so the fit can prove that (re-deriving them under
// the header's own weights must reproduce them exactly), which is the job
// `digestRow` does on the deal-in side.
//
// ===========================================================================
// THE FILE FORMAT — JSONL, one header line then one line per TURN DECISION
// ===========================================================================
//
// Header (line 1):
//   {"v":1,"kind":"mjgame-hand","seats":"khhh","seed":N,"games":N,
//    "w":{…HandWeights…}}
//   `w` is the weight vector the predictions were produced with — the merged
//   `hand` block of the `--ktune` file, or `DEFAULT_HAND` when there is none.
//   Every number in the file is a function of it, so a file is self-describing.
//
// Record (one per TURN decision of the recorded seat — seat 0 — in play order):
//   facts  the `HandFacts` of the CHOSEN resting 13-tile shape, verbatim: the
//          field names are the interface's own (see `handvalue.ts`), because a
//          record this small has nothing to gain from short keys and everything
//          to lose from a second naming scheme to keep in step.
//   pwin   `handOutlook(facts, w).pwin` as the live seat computed it
//   value  …and `.value`. Both are the REPRODUCTION CHECK, not data the fit
//          consumes: under the header's own weights the fit must recompute them
//          bit for bit or the record and the policy have forked.
//   s      game seed      n  decision index within the game (all rounds)
//   k      kyoku (0 = 東1) b  本場
//   won        1 = seat 0 was among the winners of the round this decision is in
//   winPoints  that round's `deltas[0]` when `won`, else 0 — the round's REAL
//              settlement for the seat (本場・供託 included), which is what
//              `handOutlook`'s `value` claims to predict
//   dealtIn    1 = the round ended on a ron off seat 0's discard
//   endJunme   seat 0's river length when the round ended (the hand's length)
//   outcome    "agari" | "ryuukyoku"
//
// THE LABEL IS PER ROUND, NOT PER DECISION, and that is the point: `pwin` is a
// prediction about how the ROUND ends, so every turn decision of one round
// carries the same outcome and they differ only in the facts that produced the
// prediction. That is also why the writer buffers: a sample cannot be written
// until the round it belongs to is over.
//
// SIZE. One line per own turn, so ~15 per round and ~120 per hanchan — a couple
// of hundred bytes each. A 2000-game lane is tens of megabytes, three orders of
// magnitude below a `--calibrate` lane, so this one may live in the repo's
// scratch space without ceremony.

import type { HandFacts, HandWeights } from "./handvalue.ts";
import type { Table } from "../table.ts";
import type { RoundOutcome } from "../types.ts";

/** Bumped whenever a field changes meaning. A reader refuses anything else. */
export const HAND_CALIB_VERSION = 1;
export const HAND_CALIB_KIND = "mjgame-hand";

/**
 * What the policy hands the writer: the facts of one resting shape and the
 * answer the model gave for them. The policy knows nothing about seeds, rounds
 * or outcomes — the writer owns all four, exactly as `CalibrationWriter` owns
 * the seed and the decision index.
 */
export interface HandSample {
  facts: HandFacts;
  pwin: number;
  value: number;
}

/** One line of the file: a sample plus the round's ground truth. */
export interface HandRecord extends HandSample {
  s: number;
  k: number;
  b: number;
  n: number;
  won: 0 | 1;
  winPoints: number;
  dealtIn: 0 | 1;
  endJunme: number;
  outcome: "agari" | "ryuukyoku";
}

export interface HandCalibHeader {
  v: number;
  kind: string;
  seats?: string;
  seed?: number;
  games?: number;
  w: HandWeights;
}

/** The labels one finished round stamps on every sample recorded inside it. */
interface RoundLabels {
  k: number;
  b: number;
  won: 0 | 1;
  winPoints: number;
  dealtIn: 0 | 1;
  endJunme: number;
  outcome: "agari" | "ryuukyoku";
}

/**
 * The ground truth for the RECORDED SEAT — absolute seat 0, which is the only
 * seat the harness ever hands a sink to.
 *
 * `winPoints` is `deltas[0]`, the round's real settlement, rather than the
 * winning hand's raw points: 本場 and 供託 are part of what a hand is worth and
 * `handOutlook`'s value model adds both explicitly. A double ron in which seat 0
 * is one of the winners still reads `won` — the seat won the round — and
 * `deltas[0]` is what it collected.
 */
export function handLabels(t: Table, outcome: RoundOutcome): RoundLabels {
  let won: 0 | 1 = 0;
  let dealtIn: 0 | 1 = 0;
  if (outcome.kind === "agari") {
    for (const w of outcome.wins) {
      if (w.who === 0) won = 1;
      // 放銃: someone else won off OUR discard. `fromWho === who` is a tsumo,
      // which is nobody's deal-in.
      else if (w.fromWho === 0) dealtIn = 1;
    }
  }
  return {
    k: t.round.kyoku,
    b: t.round.honba,
    won,
    winPoints: won ? outcome.deltas[0] : 0,
    dealtIn,
    // The hand's LENGTH as the seat lived it: how many times it got to discard.
    // `t.junme` advances on the dealer's draw and drifts when calls skip seats,
    // so the river is the honest count (see the note on `Table.turnIndex`).
    endJunme: t.board.rivers[0].length,
    outcome: outcome.kind,
  };
}

// ---------------------------------------------------------------------------
// the writer
// ---------------------------------------------------------------------------

/**
 * Streams hand-calibration JSONL. Opens (and TRUNCATES) the path: a new run is
 * a new dataset, exactly as `CalibrationWriter` and `TrajectoryWriter` do.
 *
 * BUFFERED BY ROUND, which is the one structural difference from the deal-in
 * recorder. A deal-in record is complete the instant the decision is made — the
 * oracle reads the truth off the live Table right there. A hand record is not:
 * its label is how the round ENDS, so samples wait in memory until `endRound`
 * and are written in decision order once the truth exists. A round that never
 * ends (the last round of an aborted run) drops its samples rather than
 * inventing an outcome for them; `close` says how many.
 */
export class HandCalibrationWriter {
  readonly path: string;
  private file: Deno.FsFile;
  private enc = new TextEncoder();
  private seed = 0;
  private n = 0;
  private rows = 0;
  private games = 0;
  private dropped = 0;
  /** Samples of the round in progress, in the order they were decided. */
  private pending: Array<{ sample: HandSample; n: number }> = [];

  constructor(path: string, header: Omit<HandCalibHeader, "v" | "kind">) {
    this.path = path;
    const dir = path.slice(0, path.lastIndexOf("/"));
    if (dir) Deno.mkdirSync(dir, { recursive: true });
    this.file = Deno.openSync(path, { create: true, write: true, truncate: true });
    this.writeLine({ v: HAND_CALIB_VERSION, kind: HAND_CALIB_KIND, ...header });
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
  readonly record = (sample: HandSample): void => {
    this.pending.push({ sample, n: this.n++ });
  };

  /** The round is over: label everything it produced and write it out. */
  endRound(t: Table, outcome: RoundOutcome): void {
    const lab = handLabels(t, outcome);
    for (const { sample, n } of this.pending) {
      this.writeLine({ ...sample, s: this.seed, n, ...lab } satisfies HandRecord);
      this.rows++;
    }
    this.pending.length = 0;
  }

  private writeLine(obj: unknown): void {
    const bytes = this.enc.encode(JSON.stringify(obj) + "\n");
    let n = 0;
    while (n < bytes.length) n += this.file.writeSync(bytes.subarray(n));
  }

  stats(): { games: number; rows: number; dropped: number } {
    return { games: this.games, rows: this.rows, dropped: this.dropped + this.pending.length };
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

function versionError(path: string, v: number): Error {
  return new Error(
    `${path}: 版が違います (v${v}, 期待 v${HAND_CALIB_VERSION})。` +
      "selfplay --handcalib で取り直してください",
  );
}

function checkHeader(h: HandCalibHeader, path: string): void {
  if (h.kind !== HAND_CALIB_KIND) throw new Error(`${path}: 手牌価値の較正記録ではありません`);
  if (h.v !== HAND_CALIB_VERSION) throw versionError(path, h.v);
}

export interface HandCalibFile {
  header: HandCalibHeader;
  records: HandRecord[];
}

/** Parse a JSONL hand-calibration file. A wrong version is fatal, never coerced. */
export function parseHandCalibration(text: string, path = "<memory>"): HandCalibFile {
  const lines = text.split("\n").filter((l) => l.trim() !== "");
  if (lines.length === 0) throw new Error(`${path}: 空のファイルです`);
  const header = JSON.parse(lines[0]) as HandCalibHeader;
  checkHeader(header, path);
  return { header, records: lines.slice(1).map((l) => JSON.parse(l) as HandRecord) };
}

export function readHandCalibration(path: string): HandCalibFile {
  return parseHandCalibration(Deno.readTextFileSync(path), path);
}

/**
 * The same file, one record at a time and nothing retained.
 *
 * A hand lane is small enough to read whole (`readHandCalibration` exists for
 * fixtures), but the fit and the report both make one pass over it and neither
 * needs the parsed objects afterwards, so the streaming door is the one they
 * use. The header is parsed before any record is delivered, so a version
 * mismatch fails before the caller has accumulated anything.
 */
export async function scanHandCalibration(
  path: string,
  onRecord: (rec: HandRecord, header: HandCalibHeader) => void,
): Promise<HandCalibHeader> {
  const file = await Deno.open(path, { read: true });
  const dec = new TextDecoder();
  let header: HandCalibHeader | null = null;
  let rest = "";
  const take = (line: string) => {
    if (line.trim() === "") return;
    if (header === null) {
      header = JSON.parse(line) as HandCalibHeader;
      checkHeader(header, path);
      return;
    }
    onRecord(JSON.parse(line) as HandRecord, header);
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
