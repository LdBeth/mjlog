// Trajectory recording: JSONL that the Python/MLX trainer eats directly.
//
// One file holds many matches. Line kinds (FROZEN contract):
//
//   {"k":"d","v":2,"seat":0,"kyoku":0,"honba":0,"junme":3,
//    "planes":"<base64 of the 1224 Int8 bytes>",
//    "scalars":"<base64 of the 39 little-endian float32 = 156 bytes>",
//    "mask":[legal action indices],"a":<chosen index>}
//   {"k":"r","deltas":[4 point deltas, ABSOLUTE seats],"outcome":"agari"|"draw"}
//   {"k":"m","scores":[4],"net":[4],"violations":[4]}
//
// "v" is the FEATURE version the planes/scalars were encoded with, so a loader
// can reject a dataset that predates the current encoder instead of reshaping
// stale bytes into the wrong network.
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
import { encode, FEATURES } from "./features.ts";

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

/**
 * One open file, shared by all four seats of a run. Each line is handed to
 * `writeSync` on its own, so a killed run still leaves every completed line on
 * disk (only the partial tail, if any, is lost — and there is no partial tail,
 * because a line is a single write).
 */
export class TrajectoryWriter {
  readonly path: string;
  private file: Deno.FsFile;
  private enc = new TextEncoder();
  private counts = { d: 0, r: 0, m: 0 };

  /** Opens (and TRUNCATES) `path`: a new run is a new dataset. */
  constructor(path: string) {
    this.path = path;
    const dir = path.slice(0, path.lastIndexOf("/"));
    if (dir) Deno.mkdirSync(dir, { recursive: true });
    this.file = Deno.openSync(path, { create: true, write: true, truncate: true });
  }

  writeLine(obj: Record<string, unknown>): void {
    const bytes = this.enc.encode(JSON.stringify(obj) + "\n");
    let n = 0;
    while (n < bytes.length) n += this.file.writeSync(bytes.subarray(n));
    const k = obj.k as "d" | "r" | "m";
    if (k in this.counts) this.counts[k]++;
  }

  /** Line counts by kind — what the CLI prints after a recording run. */
  stats(): { d: number; r: number; m: number } {
    return { ...this.counts };
  }

  close(): void {
    this.file.close();
  }
}

// ---------------------------------------------------------------------------
// per-decision recording
// ---------------------------------------------------------------------------

/** Wraps any sync policy and logs one "d" line per decision it makes. */
export class RecordingPolicy implements SyncPolicy {
  readonly name: string;
  readonly sync = true;

  constructor(readonly inner: SyncPolicy, readonly writer: TrajectoryWriter) {
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
    const { planes, scalars } = encode(obs);
    this.writer.writeLine({
      k: "d",
      v: FEATURES.version,
      seat: obs.seat,
      kyoku: obs.kyoku,
      honba: obs.honba,
      junme: obs.junme,
      planes: toBase64(new Uint8Array(planes.buffer, planes.byteOffset, planes.byteLength)),
      scalars: toBase64(f32leBytes(scalars)),
      mask: maskIndices(obs.legal),
      a: actionIndex(a, obs.akaIds),
    });
    return a;
  }
}

// ---------------------------------------------------------------------------
// round / match lines
// ---------------------------------------------------------------------------

export function writeRoundEnd(w: TrajectoryWriter, outcome: RoundOutcome): void {
  w.writeLine({
    k: "r",
    deltas: [...outcome.deltas],
    outcome: outcome.kind === "agari" ? "agari" : "draw",
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

export function writeMatchEnd(
  w: TrajectoryWriter,
  result: { scores: number[]; outcomes: RoundOutcome[]; ledger: Violation[] },
  cfg: RuleConfig,
): void {
  for (const o of result.outcomes) writeRoundEnd(w, o);
  w.writeLine({
    k: "m",
    scores: [...result.scores],
    net: settlement(result.scores, cfg),
    violations: violationPoints(result.ledger),
  });
}
