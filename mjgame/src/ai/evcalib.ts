// EV核の較正記録 (M15b) — what the expected-value DP predicted about OUR OWN
// hand, and how the 局 actually ended for the seat that held it.
//
// WHY. M15's first grade said the engine is internally consistent and the MODEL
// is wrong: `ronFactor` 0.5 gives P(win) 0.89 from a ryanmen tenpai in 12 turns
// where the champion's realised 和了率 is ~21%, and a 18-cell screen over
// `ronFactor × oppGrowth × dealinRate` moved nothing because a screen grades
// PLACEMENT — four scalars away from the quantity they actually distort. This
// lane grades the quantity: `mjev_eval_rest` reports `R_PWIN` and `R_PTENPAI`
// for a 13-tile rest, and a 局 reports whether that hand won and whether it was
// tenpai when the music stopped. Two probabilities, two labels, one BCE.
//
// THE PRINCIPLE, from `handcalib.ts` and unchanged: FEATURE EXTRACTION AT
// RECORD TIME, MODEL RE-EVALUATION AT FIT TIME. What a record carries is the
// FULL packed wire of one `mjev_eval_rest` call — `ints` (INTS_LEN) and `dbls`
// (DBLS_LEN) exactly as `packEvInputs` built them — which is by construction
// everything the DP is allowed to see (`evlayout.ts` IS that specification).
// So the fit re-evaluates any candidate `EvParams` by building a core and
// replaying the wire: the SAME engine the seat would run, never a second
// implementation of it. No closed form is possible here (the DP has no
// TypeScript twin) and none is wanted.
//
// THE M11 LESSON, applied twice over. The lane is recorded on the PLAIN
// champion — a seat with NO `ev` block — so the labels are the champion's own
// continuation of the hand. A lane played by the block being fitted would be
// censored by that block's own folds, which measured +0.11 WORSE when M11 tried
// it. Hence `--evcalib` refuses a ktune carrying an `ev` section, and the seat
// that records is not the seat that is evaluated: the WRITER owns the `EvCore`
// (built once, from the header's own params) and evaluates each wire as it
// arrives, so the policy never touches the FFI and plays bit-for-bit its
// ordinary game.
//
// ===========================================================================
// THE FILE FORMAT — JSONL, one header line then one line per TURN DECISION
// ===========================================================================
//
// Header (line 1):
//   {"v":2,"kind":"mjgame-ev","seats":"khhh","seed":N,"games":N,
//    "ev":{…EvParams…},"evAbi":1,"engineHash":"<sha256 of native/mjev.cc>"}
//   `ev` is the merged parameter vector the stored predictions were produced
//   with — `DEFAULT_EV`, always, because the recording seat carries no block —
//   `evAbi` the wire version, since every `ints`/`dbls` offset in the file is
//   meaningless under a different one, and `engineHash` the identity of the DP
//   that produced the stored predictions.
//
// WHY A HASH OF THE ENGINE SOURCE, AND WHY IT IS NOT A REFUSAL. The wire and
// the labels are properties of the GAME: a hand, a board, and how the 局 ended.
// They survive any amount of work on `native/mjev.cc`. The four stored
// predictions do not — they are that engine's answers, and a corrected DP is
// SUPPOSED to answer differently. So the two are separated: the ABI (which
// governs whether the wire can be read at all) is a REFUSAL, and the engine
// hash (which governs only whether the stored predictions still reproduce) is a
// NOTICE. `scripts/ev_fit.ts` skips its reproduction check, says so out loud
// and fits on regardless, because a lane whose predictions no longer reproduce
// is still a lane of valid inputs and honest labels. A lane recorded before the
// hash existed (v1) reads the same way.
//
// Record (one per TURN decision of seat 0, in play order):
//   ints/dbls  the packed wire of the CHOSEN 13-tile rest (mode 1), verbatim.
//              `dbls` is rounded to 6 significant digits BEFORE the writer's
//              own evaluation, so the numbers stored and the numbers evaluated
//              are the same numbers and a fit reproduces `pT`/`pW`/`eV`/`eCost`
//              bit for bit. (The hidden-information block is zero: 計算 reads
//              no hidden channel.)
//   pT/pW      `R_PTENPAI` / `R_PWIN` under the header's params AND the engine
//              named by `engineHash` — the reproduction check's right-hand side,
//              never data the fit consumes
//   eV/eCost   `R_EVALUE` / `R_ECOST`, and `value` the DP's own answer
//   sh/junme/T the rest's shanten, 巡目 and remaining own draws — bucket keys,
//              all three derivable from the wire, carried out for the report
//   s/k/b/n    game seed, 局, 本場, decision index within the game
//   won/winPoints/dealtIn/dealtInPoints/oppWon/outcome/endJunme/tenpaiEnd
//              the 局's ground truth for seat 0 (see `evLabels`)
//
// THE LABEL IS PER ROUND, NOT PER DECISION — `pW` is a claim about how the 局
// ends, so every turn decision inside one 局 wears the same outcome and they
// differ only in the wire that produced the claim. Which is why the writer
// BUFFERS: a sample is half a record until its 局 is over.
//
// SIZE. ~1.7 KB per line against `handcalib`'s ~200 B — the wire is 528 numbers
// — so ~220 KB per 半荘 (measured: 2,000 半荘 = 264,035 rows = 448 MB). Scratch
// space, not the repo.

import { createHash } from "node:crypto";
import { tileType } from "mjrender/tiles.ts";
import { shanten } from "../kernel.ts";
import type { Table } from "../table.ts";
import type { RoundOutcome } from "../types.ts";
import type { EvCore } from "./ev.ts";
import { buildEv, closeEv, evEvalRest } from "./ev.ts";
import {
  DBLS_LEN,
  EV_ABI,
  I_JUNME,
  I_T,
  INTS_LEN,
  R_ECOST,
  R_EVALUE,
  R_PTENPAI,
  R_PWIN,
  R_TRUNC,
  R_VALUE,
} from "./evlayout.ts";
import type { EvParams } from "./evparams.ts";
import { mergeEv } from "./evparams.ts";

/**
 * Bumped whenever a field changes meaning; a reader refuses anything not in
 * `EV_CALIB_ACCEPTED`. v2 adds `engineHash` — a strict SUPERSET, so the v1
 * lanes recorded before it stay readable and simply have no engine identity to
 * check against (calibration v2→v3's rule, M14).
 */
export const EV_CALIB_VERSION = 2;
export const EV_CALIB_ACCEPTED = new Set([1, 2]);
export const EV_CALIB_KIND = "mjgame-ev";

/**
 * What the policy hands the writer: one packed rest-root wire, plus the one
 * bucket key the wire does not carry (the rest's shanten — `ints` holds the
 * hand, not its distance to tenpai, and re-deriving it in the writer would be a
 * second witness to a number the policy already paid for).
 *
 * The arrays are READ, never retained: the policy reuses one scratch pair for
 * the life of the seat, exactly as `EvCore` does.
 */
export interface EvSample {
  ints: ArrayLike<number>;
  dbls: ArrayLike<number>;
  shanten: number;
}

/** The DP's four numbers for one wire, under the header's own parameters. */
export interface EvPrediction {
  value: number;
  pT: number;
  pW: number;
  eV: number;
  eCost: number;
}

/** One line of the file: the wire, the prediction, and the 局's ground truth. */
export interface EvRecord extends EvPrediction {
  ints: number[];
  dbls: number[];
  s: number;
  k: number;
  b: number;
  n: number;
  junme: number;
  sh: number;
  T: number;
  won: 0 | 1;
  winPoints: number;
  dealtIn: 0 | 1;
  dealtInPoints: number;
  /** Somebody else won the 局, and not off our discard. */
  oppWon: 0 | 1;
  outcome: "agari" | "ryuukyoku";
  endJunme: number;
  /** Was seat 0 tenpai when the 局 ended? See `evLabels`. */
  tenpaiEnd: 0 | 1;
}

/** Everything one record carries that is NOT the 局's label or its index. */
export type EvPending = Omit<EvRecord, keyof EvLabels | "s" | "n">;

export interface EvCalibHeader {
  v: number;
  kind: string;
  seats?: string;
  seed?: number;
  games?: number;
  ev: EvParams;
  evAbi: number;
  /**
   * sha256 of `native/mjev.cc` as it stood when the lane was recorded. Absent
   * in a v1 lane. See the module header: an ABI mismatch is a refusal, an
   * engine mismatch is a notice.
   */
  engineHash?: string;
}

/** The labels one finished 局 stamps on every sample recorded inside it. */
export interface EvLabels {
  k: number;
  b: number;
  won: 0 | 1;
  winPoints: number;
  dealtIn: 0 | 1;
  dealtInPoints: number;
  oppWon: 0 | 1;
  outcome: "agari" | "ryuukyoku";
  endJunme: number;
  tenpaiEnd: 0 | 1;
}

/**
 * Was seat 0 tenpai at the moment the 局 ended?
 *
 * THREE CASES, and only one of them needs computing. At a 流局 the TABLE has
 * already judged it (that judgement is what the 罰符 was paid on, so no second
 * opinion is admissible). When seat 0 won, it was tenpai by definition. When
 * somebody else ended it, nobody judged our hand — so we ask `shanten` of the
 * concealed tiles still in front of us, with our own melds, which is the same
 * question `observe` would have asked one instant earlier.
 *
 * `R_PTENPAI` is "P(the hand is ever tenpai before it ends)", so this label is
 * a LOWER bound on the event it predicts: a hand that reached tenpai and then
 * broke it to fold reads 0 here. That is stated rather than corrected — the
 * alternative is to reconstruct every 巡 of the hand from the river, which is a
 * different lane — and it means a fitted `pT` is calibrated against "tenpai at
 * the end", the quantity the 聴牌率 the seat is graded on actually measures.
 */
export function tenpaiAtEnd(t: Table, outcome: RoundOutcome): 0 | 1 {
  if (outcome.kind === "ryuukyoku") return outcome.tenpai[0] ? 1 : 0;
  if (outcome.wins.some((w) => w.who === 0)) return 1;
  const counts = new Array<number>(34).fill(0);
  for (const tile of t.board.hands[0]) counts[tileType(tile)]++;
  const melds = t.board.melds[0];
  return shanten(counts, melds.length, melds.every((m) => m.kind === "ankan")) <= 0 ? 1 : 0;
}

/**
 * The ground truth for the RECORDED SEAT — absolute seat 0, the only seat the
 * harness ever hands a sink to. `handLabels`' three columns plus three more the
 * EV model makes claims about: what a deal-in COST (`R_ECOST` is a points
 * figure, so its audit needs points), whether the 局 was ended by somebody else
 * at all (which is what `oppHazard`/`oppGrowth` price), and tenpai at the end.
 */
export function evLabels(t: Table, outcome: RoundOutcome): EvLabels {
  let won: 0 | 1 = 0;
  let dealtIn: 0 | 1 = 0;
  let oppWon: 0 | 1 = 0;
  if (outcome.kind === "agari") {
    for (const w of outcome.wins) {
      if (w.who === 0) won = 1;
      // 放銃: somebody else won off OUR discard. `fromWho === who` is a tsumo,
      // which is nobody's deal-in but is still an opponent ending the hand.
      else if (w.fromWho === 0) dealtIn = 1;
      else oppWon = 1;
    }
    if (won) oppWon = 0;
  }
  const d = outcome.deltas[0];
  return {
    k: t.round.kyoku,
    b: t.round.honba,
    won,
    winPoints: won ? d : 0,
    dealtIn,
    dealtInPoints: dealtIn && !won ? -d : 0,
    oppWon,
    outcome: outcome.kind,
    // The hand's LENGTH as the seat lived it (see `handLabels`): `t.junme`
    // drifts when calls skip seats, the river does not.
    endJunme: t.board.rivers[0].length,
    tenpaiEnd: tenpaiAtEnd(t, outcome),
  };
}

// ---------------------------------------------------------------------------
// the engine's identity
// ---------------------------------------------------------------------------

/** The one place `native/mjev.cc` is looked for, module-relative (`ev.ts`'s rule). */
export const EV_SOURCE_URL = new URL("../../native/mjev.cc", import.meta.url);

/**
 * sha256 of the DP's source, hex. THE SOURCE and not the dylib: a rebuilt
 * dylib with unchanged source answers the same numbers, and a changed source
 * that has not been rebuilt yet is exactly the state a lane must not be
 * recorded in.
 *
 * Returns `undefined` rather than throwing when the file cannot be read — a
 * checkout without the C++ half can still parse and label a lane, and a missing
 * identity is the same "cannot check" that a v1 lane carries.
 */
export function evEngineHash(): string | undefined {
  let bytes: Uint8Array;
  try {
    bytes = Deno.readFileSync(EV_SOURCE_URL);
  } catch {
    return undefined;
  }
  // `node:crypto` and not `crypto.subtle`, for one reason: the digest is taken
  // from a CONSTRUCTOR (the writer stamps its header before the first game) and
  // `subtle.digest` is async. This is the only synchronous sha256 Deno ships.
  return createHash("sha256").update(bytes).digest("hex");
}

// ---------------------------------------------------------------------------
// the writer
// ---------------------------------------------------------------------------

/**
 * Six significant digits. Applied to `dbls` BEFORE the writer's own evaluation,
 * which is the whole point: a file whose stored numbers were evaluated at full
 * precision and re-evaluated at printed precision would fail its own
 * reproduction check by a few ULP and nobody could tell that from a fork.
 */
function sig6(v: number): number {
  if (v === 0 || !Number.isFinite(v)) return v;
  return Number(v.toPrecision(6));
}

/**
 * Streams EV-calibration JSONL. Opens (and TRUNCATES) the path, buffers by 局,
 * and — unlike every other recorder here — OWNS AN EvCore.
 *
 * That ownership is the design. The seat being recorded carries no `ev` block
 * (the CLI refuses one: the lane must be the champion's own continuation), so
 * somebody has to evaluate the wire, and the writer is the only party that can
 * do it without changing what the seat plays. It builds exactly one core, from
 * the header's own parameters, and frees it in `close()`.
 */
export class EvCalibrationWriter {
  readonly path: string;
  readonly params: EvParams;
  private file: Deno.FsFile;
  private core: EvCore;
  private enc = new TextEncoder();
  private seed = 0;
  private n = 0;
  private rows = 0;
  private games = 0;
  private dropped = 0;
  private evals = 0;
  private truncated = 0;
  /** Samples of the 局 in progress, in the order they were decided. */
  private pending: Array<{ rec: EvPending; n: number }> = [];

  constructor(path: string, header: Omit<EvCalibHeader, "v" | "kind" | "evAbi">) {
    this.path = path;
    this.params = mergeEv(header.ev);
    // BEFORE the file is opened: `buildEv` throws when the dylib is missing or
    // `--allow-ffi` was not granted, and a truncated empty lane left behind by
    // a run that never started is worse than no lane.
    this.core = buildEv(this.params);
    const dir = path.slice(0, path.lastIndexOf("/"));
    if (dir) Deno.mkdirSync(dir, { recursive: true });
    this.file = Deno.openSync(path, { create: true, write: true, truncate: true });
    this.writeLine(
      {
        v: EV_CALIB_VERSION,
        kind: EV_CALIB_KIND,
        ...header,
        ev: this.params,
        evAbi: EV_ABI,
        engineHash: evEngineHash(),
      } satisfies EvCalibHeader,
    );
  }

  /** A new game: stamp its seed on everything that follows, restart the index. */
  beginGame(seed: number): void {
    this.seed = seed;
    this.n = 0;
    this.games++;
    this.dropped += this.pending.length;
    this.pending.length = 0;
  }

  /**
   * The sink handed to the policy. Bound, so it can be passed around.
   *
   * The native call happens HERE rather than at `endRound`, so the buffered
   * memory of a long 局 is 528 numbers per decision and not a growing pile of
   * live wire buffers the policy would have to stop reusing.
   */
  readonly record = (sample: EvSample): void => {
    if (sample.ints.length !== INTS_LEN) {
      throw new Error(`evcalib: ints は${INTS_LEN}要素です: ${sample.ints.length}`);
    }
    if (sample.dbls.length !== DBLS_LEN) {
      throw new Error(`evcalib: dbls は${DBLS_LEN}要素です: ${sample.dbls.length}`);
    }
    const ints = new Array<number>(INTS_LEN);
    for (let i = 0; i < INTS_LEN; i++) ints[i] = sample.ints[i];
    const dbls = new Array<number>(DBLS_LEN);
    for (let i = 0; i < DBLS_LEN; i++) dbls[i] = sig6(sample.dbls[i]);
    const p = evaluateWire(this.core, ints, dbls);
    this.evals++;
    if (this.core.meta[R_TRUNC] > 0) this.truncated++;
    this.pending.push({
      rec: { ints, dbls, ...p, junme: ints[I_JUNME], sh: sample.shanten, T: ints[I_T] },
      n: this.n++,
    });
  };

  /** The 局 is over: label everything it produced and write it out. */
  endRound(t: Table, outcome: RoundOutcome): void {
    const lab = evLabels(t, outcome);
    for (const { rec, n } of this.pending) {
      this.writeLine({ ...rec, s: this.seed, n, ...lab } satisfies EvRecord);
      this.rows++;
    }
    this.pending.length = 0;
  }

  private writeLine(obj: unknown): void {
    const bytes = this.enc.encode(JSON.stringify(obj) + "\n");
    let n = 0;
    while (n < bytes.length) n += this.file.writeSync(bytes.subarray(n));
  }

  stats(): { games: number; rows: number; dropped: number; truncated: number } {
    return {
      games: this.games,
      rows: this.rows,
      dropped: this.dropped + this.pending.length,
      truncated: this.truncated,
    };
  }

  close(): void {
    this.dropped += this.pending.length;
    this.pending.length = 0;
    this.file.close();
    closeEv(this.core);
  }
}

/**
 * Replay one stored wire through a core and read the meta out. THE one place
 * this happens — the writer calls it to produce a record and the fitter calls it
 * to reproduce (and then to re-price under a candidate vector), so the two can
 * never disagree about what "evaluating a record" means.
 */
export function evaluateWire(
  core: EvCore,
  ints: ArrayLike<number>,
  dbls: ArrayLike<number>,
): EvPrediction {
  for (let i = 0; i < INTS_LEN; i++) core.ints[i] = ints[i];
  for (let i = 0; i < DBLS_LEN; i++) core.dbls[i] = dbls[i];
  // The return value IS `meta[R_VALUE]`; read from the meta so one buffer
  // answers all five and a future field cannot drift from the return.
  evEvalRest(core);
  const m = core.meta;
  return {
    value: m[R_VALUE],
    pT: m[R_PTENPAI],
    pW: m[R_PWIN],
    eV: m[R_EVALUE],
    eCost: m[R_ECOST],
  };
}

// ---------------------------------------------------------------------------
// reading one back
// ---------------------------------------------------------------------------

function checkHeader(h: EvCalibHeader, path: string): void {
  if (h.kind !== EV_CALIB_KIND) throw new Error(`${path}: EV核の較正記録ではありません`);
  if (!EV_CALIB_ACCEPTED.has(h.v)) {
    throw new Error(
      `${path}: 版が違います (v${h.v}, 読めるのは v${[...EV_CALIB_ACCEPTED].join("/")})。` +
        "selfplay --evcalib で取り直してください",
    );
  }
  if (h.evAbi !== EV_ABI) {
    throw new Error(
      `${path}: ABI が違います (evAbi=${h.evAbi}, 期待 ${EV_ABI}) — ` +
        "wire のオフセットが変わっているので、この記録は読めません",
    );
  }
}

export interface EvCalibFile {
  header: EvCalibHeader;
  records: EvRecord[];
}

/** Parse a JSONL EV-calibration file. A wrong version/ABI is fatal, never coerced. */
export function parseEvCalibration(text: string, path = "<memory>"): EvCalibFile {
  const lines = text.split("\n").filter((l) => l.trim() !== "");
  if (lines.length === 0) throw new Error(`${path}: 空のファイルです`);
  const header = JSON.parse(lines[0]) as EvCalibHeader;
  checkHeader(header, path);
  return { header, records: lines.slice(1).map((l) => JSON.parse(l) as EvRecord) };
}

export function readEvCalibration(path: string): EvCalibFile {
  return parseEvCalibration(Deno.readTextFileSync(path), path);
}

/**
 * The same file, one record at a time and nothing retained — `scanHandCalibration`
 * exactly, and for its reason: the fit makes one pass and a lane is hundreds of
 * megabytes. The header is parsed before any record is delivered, so a version
 * or ABI mismatch fails before the caller has accumulated anything.
 */
export async function scanEvCalibration(
  path: string,
  onRecord: (rec: EvRecord, header: EvCalibHeader) => void,
): Promise<EvCalibHeader> {
  const file = await Deno.open(path, { read: true });
  const dec = new TextDecoder();
  let header: EvCalibHeader | null = null;
  let rest = "";
  const take = (line: string) => {
    if (line.trim() === "") return;
    if (header === null) {
      header = JSON.parse(line) as EvCalibHeader;
      checkHeader(header, path);
      return;
    }
    onRecord(JSON.parse(line) as EvRecord, header);
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
