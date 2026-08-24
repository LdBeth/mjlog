// Trajectory recording: JSONL that the Python/MLX trainer eats directly.
//
// One file holds many matches. Line kinds (FROZEN contract):
//
//   {"k":"d","v":4,"seat":0,"kyoku":0,"honba":0,"junme":3,
//    "planes":"<base64 of the 1632 Int8 bytes>",
//    "scalars":"<base64 of the 42 little-endian float32 = 168 bytes>",
//    "seq":"<base64 of the 4×L packed river tokens, L ≤ 96>",
//    "mask":[slot indices the action was sampled from — legal minus the
//            compliance filter's vetoes],"a":<chosen index>,
//    "o":"<base64 of the 170 oracle Int8 bytes>","sh":[3 opponent shanten]}
//   {"k":"r","kyoku":0,"honba":0,
//    "deltas":[4 point deltas, ABSOLUTE seats],"outcome":"agari"|"draw",
//    "viol":[4 評価点マイナス, ABSOLUTE seats]}
//   {"k":"m","scores":[4],"net":[4],"violations":[4]}
//
// "kyoku"/"honba" on an "r" line NAME the round it reports, carrying exactly the
// values that round's "d" lines carry. They exist because an "r" line was
// otherwise anonymous: the writer buffers a match's rounds and flushes them all
// after that match's decisions, so the only thing joining a decision to its
// round result was POSITION — count the (kyoku,honba) blocks of the "d" lines
// and pair the k-th block with the k-th "r" line. Position is not enough.
// Recording wraps only the seats the driver asked for, so under a mixed
// population an entire round can pass with no "d" line at all (an opponent wins
// before any recorded seat acts), and a decision-less round is invisible to
// that reconstruction — two consecutive ones are not even distinguishable in
// principle. With the pair on the line the join is direct, and (kyoku,honba) is
// unique within a match: it is constant through a round and changes at every
// boundary (honba increments on 連荘 and on a draw, kyoku advances otherwise).
// Files written before this field existed simply lack it — a loader must read a
// missing "kyoku"/"honba" as "old data" and fall back to the positional
// reconstruction, the same compatibility rule "viol" documents for itself just
// below. That fallback is only sound when every round holds at least one
// recorded decision, so a loader taking it must say so loudly when the block
// count and the "r" line count disagree rather than guess an alignment.
//
// "viol" on an "r" line is the 評価点マイナス INCURRED IN THAT ROUND, per
// absolute seat (positive magnitudes, 0 when clean). It exists so credit
// assignment stays causal: the match total charges a 罰符 to every decision in
// the hanchan, including the ones made before it happened. Files written before
// this field existed simply lack it — a loader should read a missing "viol" as
// [0,0,0,0] and fall back to the "m" line. The "m" line's "violations" is
// UNCHANGED: still the whole-match total, kept for that fallback and for
// reporting, and it equals the per-seat sum of every "r" line's "viol".
//
// "v" is the FEATURE version the planes/scalars were encoded with, so a loader
// can reject a dataset that predates the current encoder instead of reshaping
// stale bytes into the wrong network.
//
// "seq" is feature v4's river token stream (`encodeSeq` in features.ts): the
// four rivers as `4 × L` packed Int8 — `[type, seatRel, idx, flags]` per
// discard, self first then relative seats 1/2/3, chronological, each river cut
// to its first 24 entries, so L ≤ 96. Unlike "o"/"sh" it is NOT optional and it
// DID move the version: the planes and scalars are byte-identical to v3, so
// "v":4 means precisely "this line carries seq", and that is the only thing a
// loader can check the field's presence against. An EMPTY string is a perfectly
// valid value — it is what the first decision of a hand encodes, before anyone
// has discarded — and must be read as L = 0, never as a missing field. The
// trainer requires v4 and rejects v3 datasets outright, as v3 did to v2.
//
// "o" and "sh" are OPTIONAL and appear only when the driver supplied an oracle
// tap (`RecordingPolicy`'s third constructor argument); a consumer must read
// their absence as "this dataset carries no oracle data" and fall back to a
// symmetric critic rather than erroring. They are HIDDEN state — the three
// opponents' concealed hands, the unseen remainder, the ura indicators, and the
// opponents' shanten — meant for an asymmetric critic and for auxiliary
// (opponent-speed) losses at TRAINING time only. TS inference never reads them,
// and they are no part of the policy input, so the oracle block never moves
// "v" on its own: it was added without a bump, and feature v3 (the per-opponent
// planes) left it untouched. Layout: `encodeOracle` in features.ts
// (5 planes × 34 types, then `sh` in the same relative seat order, raw shanten
// values — -1 means a complete hand, they are labels, not clamped).
//
// "d" lines are written by `RecordingPolicy` as play happens; "r" and "m" come
// from the driver once a match finishes (the policy cannot see round results).
// `seat` on a "d" line is the ABSOLUTE seat, while everything inside `planes`
// and `scalars` is relative to it — the absolute seat is there for debugging
// and for credit assignment against the "r"/"m" lines, not as a feature.

import type { Observation } from "../observe.ts";
import type { SyncPolicy } from "../policy.ts";
import type { RuleConfig } from "../rules.ts";
import type { Action, PublicEvent, RoundOutcome, Violation } from "../types.ts";
import { actionIndex, maskIndices } from "./actionspace.ts";
import type { EncodingCache } from "./features.ts";
import { encode, encodeSeq, FEATURES } from "./features.ts";

// ---------------------------------------------------------------------------
// encoding helpers (no dependencies: btoa plus chunked fromCharCode)
// ---------------------------------------------------------------------------

export function toBase64(bytes: Uint8Array): string {
  let s = "";
  const CHUNK = 0x8000; // fromCharCode blows the argument limit above this
  for (let i = 0; i < bytes.length; i += CHUNK) {
    s += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(s);
}

export function fromBase64(s: string): Uint8Array {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** Little-endian bytes of a float32 array, written explicitly (not a view). */
export function f32leBytes(a: Float32Array): Uint8Array {
  const out = new Uint8Array(a.length * 4);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < a.length; i++) dv.setFloat32(i * 4, a[i], true);
  return out;
}

// ---------------------------------------------------------------------------
// writer
// ---------------------------------------------------------------------------

/** Line counts by kind, as `TrajectoryWriter.stats` reports them. */
export interface LineCounts {
  d: number;
  r: number;
  m: number;
}

/**
 * One open file, shared by all four seats of a run. Each line is handed to
 * `writeSync` on its own, so a killed run still leaves every completed line on
 * disk (only the partial tail, if any, is lost — and there is no partial tail,
 * because a line is a single write).
 *
 * A writer can also hold its lines IN MEMORY instead (`TrajectoryWriter.
 * buffering()`), which is what `--jobs` is built on: a worker has no business
 * writing into the run's one dataset — the file belongs to the main thread,
 * which appends each game's buffered text in GAME ORDER (`writeRaw`) so the
 * result is byte-identical to the sequential run. `drain` hands one game's
 * lines over and clears the buffer, so a worker's memory does not grow with the
 * shard.
 */
export class TrajectoryWriter {
  readonly path: string;
  private file: Deno.FsFile | null;
  /** Non-null exactly in memory mode; `file` is then null. */
  private buf: string[] | null;
  private enc = new TextEncoder();
  private counts = { d: 0, r: 0, m: 0 };
  /** `drain`'s cursor: counts as of the last drain. */
  private drained = { d: 0, r: 0, m: 0 };

  /** Opens (and TRUNCATES) `path`: a new run is a new dataset. */
  constructor(path: string) {
    this.path = path;
    if (path === "") {
      // Memory mode — see `buffering()`, the only caller that passes "".
      this.file = null;
      this.buf = [];
      return;
    }
    const dir = path.slice(0, path.lastIndexOf("/"));
    if (dir) Deno.mkdirSync(dir, { recursive: true });
    this.file = Deno.openSync(path, { create: true, write: true, truncate: true });
    this.buf = null;
  }

  /** A writer that accumulates lines for `drain()` and never touches disk. */
  static buffering(): TrajectoryWriter {
    return new TrajectoryWriter("");
  }

  writeLine(obj: Record<string, unknown>): void {
    const text = JSON.stringify(obj) + "\n";
    const k = obj.k as "d" | "r" | "m";
    if (k in this.counts) this.counts[k]++;
    if (this.buf) {
      this.buf.push(text);
      return;
    }
    const bytes = this.enc.encode(text);
    let n = 0;
    while (n < bytes.length) n += this.file!.writeSync(bytes.subarray(n));
  }

  /**
   * Append already-serialised lines (a `drain()` result from a worker) verbatim,
   * folding their counts in. The bytes are written exactly as the worker built
   * them, which is what makes a `--jobs=N` dataset identical to a `--jobs=1` one
   * rather than merely equivalent.
   */
  writeRaw(text: string, counts: LineCounts): void {
    this.counts.d += counts.d;
    this.counts.r += counts.r;
    this.counts.m += counts.m;
    if (this.buf) {
      this.buf.push(text);
      return;
    }
    if (text === "") return;
    const bytes = this.enc.encode(text);
    let n = 0;
    while (n < bytes.length) n += this.file!.writeSync(bytes.subarray(n));
  }

  /**
   * Memory mode only: everything written since the last drain, as one string,
   * plus the counts of just those lines. Clears the buffer.
   */
  drain(): { text: string; counts: LineCounts } {
    if (!this.buf) throw new Error("drain() is for a buffering writer only");
    const text = this.buf.join("");
    this.buf.length = 0;
    const counts: LineCounts = {
      d: this.counts.d - this.drained.d,
      r: this.counts.r - this.drained.r,
      m: this.counts.m - this.drained.m,
    };
    this.drained = { ...this.counts };
    return { text, counts };
  }

  /** Line counts by kind — what the CLI prints after a recording run. */
  stats(): LineCounts {
    return { ...this.counts };
  }

  close(): void {
    this.file?.close();
    this.file = null;
  }
}

// ---------------------------------------------------------------------------
// per-decision recording
// ---------------------------------------------------------------------------

/** The hidden state for one decision, as `encodeOracle` returns it. */
export type OracleTap = (seat: number) => { oplanes: Int8Array; oppShanten: number[] };

/** Wraps any sync policy and logs one "d" line per decision it makes. */
export class RecordingPolicy implements SyncPolicy {
  readonly name: string;
  readonly sync = true;

  /**
   * `oracle` is optional: with it every "d" line also carries "o"/"sh" (hidden
   * state for an asymmetric critic), without it the line is byte-identical to
   * what this class has always written. The tap is a callback rather than a
   * Table because the policy has no business holding the table — the driver
   * owns it and hands over only the reading.
   */
  constructor(
    readonly inner: SyncPolicy,
    readonly writer: TrajectoryWriter,
    readonly oracle?: OracleTap,
  ) {
    this.name = inner.name;
  }

  notify(e: PublicEvent): void {
    this.inner.notify?.(e);
  }

  reset(seed: number): void {
    this.inner.reset?.(seed);
  }

  decide(obs: Observation): Action {
    const a = this.inner.decide(obs);
    // The inner policy has usually just encoded this very Observation — a
    // `NeuralPolicy` cannot decide without doing so — and then offers what it
    // built. Take the offer only when it names THIS Observation by reference,
    // and encode here otherwise: a heuristic seat offers nothing, and a v3 net
    // offers no seq. The bytes are identical either way, `encode`/`encodeSeq`
    // being pure functions of the Observation.
    //
    // INVARIANT the offer relies on: every field is serialised BY VALUE right
    // here (`toBase64` and `f32leBytes` both copy) and the line is written
    // before this method returns, so nothing survives into the next decision.
    // Whatever scratch buffers the encoders reuse are therefore invisible.
    //
    // The offer also carries the SUPPORT the decision was made over, which for
    // a neural seat is `obs.legal` after its dojo compliance filter — and that
    // is what `mask` must record, since PPO reweights by a ratio taken against
    // the distribution the action was actually sampled from. A heuristic inner
    // offers nothing, so the full legal mask is recorded for it: its own veto
    // lives in the choice it makes, not in the distribution behind it.
    const offered = (this.inner as Partial<EncodingCache>).lastEncoding ?? null;
    const cached = offered !== null && offered.obs === obs ? offered : null;
    const { planes, scalars } = cached ?? encode(obs);
    const seq = cached?.seq ?? encodeSeq(obs);
    const line: Record<string, unknown> = {
      k: "d",
      v: FEATURES.version,
      seat: obs.seat,
      kyoku: obs.kyoku,
      honba: obs.honba,
      junme: obs.junme,
      planes: toBase64(new Uint8Array(planes.buffer, planes.byteOffset, planes.byteLength)),
      scalars: toBase64(f32leBytes(scalars)),
      seq: toBase64(new Uint8Array(seq.buffer, seq.byteOffset, seq.byteLength)),
      mask: maskIndices(cached?.legal ?? obs.legal),
      a: actionIndex(a),
    };
    if (this.oracle) {
      const { oplanes, oppShanten } = this.oracle(obs.seat);
      line.o = toBase64(new Uint8Array(oplanes.buffer, oplanes.byteOffset, oplanes.byteLength));
      line.sh = oppShanten;
    }
    this.writer.writeLine(line);
    return a;
  }
}

// ---------------------------------------------------------------------------
// round / match lines
// ---------------------------------------------------------------------------

/** Which round an "r" line reports — the same pair its "d" lines carry. */
export interface RoundId {
  kyoku: number;
  honba: number;
}

/**
 * One "r" line. `id`: the round's (kyoku, honba), which is what joins it to the
 * "d" lines of the same round — see the header. `viol`: 評価点マイナス incurred
 * in THIS round, per absolute seat (length 4).
 *
 * Module-private: an "r" line only ever makes sense as part of the batch
 * `writeMatchEnd` flushes, which is where the per-round ledger slicing lives.
 */
function writeRoundEnd(
  w: TrajectoryWriter,
  id: RoundId,
  outcome: RoundOutcome,
  viol: number[],
): void {
  w.writeLine({
    k: "r",
    kyoku: id.kyoku,
    honba: id.honba,
    deltas: [...outcome.deltas],
    outcome: outcome.kind === "agari" ? "agari" : "draw",
    viol: [...viol],
  });
}

/**
 * 精算: `(点数 − 返し点) / 1000` (truncated toward zero when the ruleset says
 * so) plus ウマ by rank, ties broken by seat. Identical to the TUI's
 * `finalStandings` — the two must not drift, since this is the reward the
 * trainer optimises and that is the number the player is shown.
 */
export function settlement(scores: number[], cfg: RuleConfig): number[] {
  const order = scores
    .map((s, seat) => ({ seat, s }))
    .sort((a, b) => b.s - a.s || a.seat - b.seat);
  const net = [0, 0, 0, 0];
  order.forEach((o, i) => {
    const raw = o.s - cfg.returnScore;
    const pts = cfg.truncateSub1000 ? Math.trunc(raw / 1000) : raw / 1000;
    net[o.seat] = pts + cfg.uma[i];
  });
  return net;
}

/** Summed 評価点マイナス per absolute seat. */
export function violationPoints(ledger: Violation[]): number[] {
  const out = [0, 0, 0, 0];
  for (const v of ledger) out[v.seat] += v.points;
  return out;
}

/**
 * `rounds` supplies each "r" line's (kyoku, honba). It is `MatchResult.rounds`
 * — the engine pushes one `Round` and one outcome per finished round, in the
 * same breath, so the two arrays are parallel by construction and the identity
 * needs no separate bookkeeping.
 */
export function writeMatchEnd(
  w: TrajectoryWriter,
  result: {
    scores: number[];
    rounds: RoundId[];
    outcomes: RoundOutcome[];
    ledger: Violation[];
    ledgerCuts: number[];
  },
  cfg: RuleConfig,
): void {
  const cuts = result.ledgerCuts;
  if (cuts.length !== result.outcomes.length) {
    // Silently recording a mis-attributed ledger would poison the dataset in a
    // way no downstream check could see, so refuse to write the match at all.
    throw new Error(
      `ledgerCuts/outcomes mismatch: ${cuts.length} cuts for ${result.outcomes.length} rounds`,
    );
  }
  if (result.rounds.length !== result.outcomes.length) {
    // Same refusal, same reason: an "r" line labelled with the wrong round is
    // worse than no label, because the loader would trust it.
    throw new Error(
      `rounds/outcomes mismatch: ${result.rounds.length} rounds for ` +
        `${result.outcomes.length} outcomes`,
    );
  }
  result.outcomes.forEach((o, k) => {
    writeRoundEnd(
      w,
      { kyoku: result.rounds[k].kyoku, honba: result.rounds[k].honba },
      o,
      violationPoints(result.ledger.slice(cuts[k - 1] ?? 0, cuts[k])),
    );
  });
  w.writeLine({
    k: "m",
    scores: [...result.scores],
    net: settlement(result.scores, cfg),
    violations: violationPoints(result.ledger),
  });
}
