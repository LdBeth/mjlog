// Seat specifications — the modular description of WHO sits at the table.
//
// A `SeatSpec` is one seat, completely: its kind and every component block and
// weight file that seat carries. A `TableSpec` is four of them. Everything here
// is plain JSON — no writers, no taps, no closures — so a table crosses the
// `--jobs` worker boundary verbatim and two tables can be compared field by
// field, which is what makes a paired run's "same environment" claim checkable
// by machine instead of by discipline.
//
// The legacy surface (`--seats=khhh` plus the flat option bag) does not go
// away: `resolveTable` is the ONE place its conventions live — seat-0 gating
// of `standings`/`consumer`/`curriculum`, the `--ktune`/`--ktune-opp` split —
// and it preserves their ROUTING. (It no longer claims bit-compatibility with
// the pre-2026-08-25 world: the "h" seat's own meaning changed that day — see
// `ai/frozen.ts` — so `runs/` numbers involving h seats are pre-epoch.) New
// capability is additive: `--table` hands the driver an explicit TableSpec.

import type { AugmentedWeights } from "./ai/augmented.ts";
import type { ComputedWeights } from "./ai/computed.ts";
import type { ConsumerParams } from "./ai/consumer.ts";
import { parseConsumerParams } from "./ai/consumer.ts";
import type { HandWeights } from "./ai/handvalue.ts";
import type { HeuristicWeights } from "./ai/heuristic.ts";
import type { RiichiWeights } from "./ai/riichi.ts";
import { die } from "./cli/die.ts";

/**
 * A tuned 感性 vector: the weight sections a heuristic-family seat is built
 * from, each a PARTIAL merged over its own defaults by the constructor that
 * receives it.
 *
 * Deliberately un-validated. `scripts/tune.ts` writes these files and
 * `ComputedWeights` grows fields as the reader learns to count more; a key
 * whitelist here would have to be edited in lockstep with that, and would
 * reject a forward-compatible file by silently dropping the very term under
 * test. Unknown keys are simply spread onto the defaults and ignored by
 * whatever does not read them.
 */
export interface KTune {
  heuristic?: Partial<HeuristicWeights>;
  augment?: Partial<AugmentedWeights>;
  computed?: Partial<ComputedWeights>;
  /**
   * M11's 手牌価値 scalars. ABSENT means the model is off and the heuristic
   * family plays the game it has always played — so this section, unlike the
   * three above, is what SWITCHES a behaviour on rather than retuning one.
   * `scripts/hand_fit.ts` writes a file whose only key is this one.
   */
  hand?: Partial<HandWeights>;
  /**
   * M12's riichi head — the learned declare-vs-damaten decision
   * (`ai/riichi.ts`). Like `hand`, a switch rather than a retune: ABSENT means
   * `wantRiichi` declares unconditionally inside its gates, exactly as before;
   * `{}` merges to `INIT_RIICHI`, which reproduces that same behaviour bit for
   * bit by construction.
   */
  riichi?: Partial<RiichiWeights>;
}

/**
 * Read a `--ktune` file. Unreadable or malformed is fatal, never silent.
 * `flag` names the option in the diagnostics, so `--ktune-b` (and a table
 * file's `seats[n].ktune`) reports itself.
 */
export function loadKtune(path: string, flag = "--ktune"): KTune {
  let text: string;
  try {
    text = Deno.readTextFileSync(path);
  } catch (e) {
    return die(`${flag} のファイルが読めません: ${path}\n${e instanceof Error ? e.message : e}`);
  }
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (e) {
    return die(`${flag} のJSONが壊れています: ${path}\n${e instanceof Error ? e.message : e}`);
  }
  if (typeof json !== "object" || json === null || Array.isArray(json)) {
    die(
      `${flag} はオブジェクト {heuristic, augment, computed, hand, riichi} である必要があります: ${path}`,
    );
  }
  // Sections only — the contents pass through verbatim. (`riichi` was silently
  // dropped here for one day of M12's life; the whitelist and the interface
  // above must move together.)
  const k = json as KTune;
  return {
    heuristic: k.heuristic,
    augment: k.augment,
    computed: k.computed,
    hand: k.hand,
    riichi: k.riichi,
  };
}

/**
 * Read a `--consumer` file. Unreadable, malformed or incomplete is fatal: a
 * silently-defaulted consumer would measure the hand-written score and call it
 * the fitted one.
 */
export function loadConsumer(path: string, flag = "--consumer"): ConsumerParams {
  let text: string;
  try {
    text = Deno.readTextFileSync(path);
  } catch (e) {
    return die(`${flag} のファイルが読めません: ${path}\n${e instanceof Error ? e.message : e}`);
  }
  try {
    return parseConsumerParams(JSON.parse(text));
  } catch (e) {
    return die(`${flag} のJSONが不正です: ${path}\n${e instanceof Error ? e.message : e}`);
  }
}

/** The seat letters `--seats` has always used. */
export type SeatKind = "h" | "k" | "o" | "n" | "r";

/**
 * One seat, completely described. Every field is meaningful only to the kinds
 * that read it (`ktune`/`plan`/`curriculum` to the heuristic family,
 * `weights`/`temp` to "n") — the others ignore it, exactly as the flat option
 * bag always worked. Plain JSON by construction.
 */
export interface SeatSpec {
  kind: SeatKind;
  /** This seat's complete vector — 感性 sections AND the hand/riichi models. */
  ktune?: KTune;
  /** Engage the C7 planner (a `ktune` naming `planner` outranks it). */
  plan?: boolean;
  /** 順位効用 on this seat. */
  standings?: boolean;
  /** M9's learned consumer for this seat. */
  consumer?: ConsumerParams;
  /** M9c curriculum rate for this seat's reader ("k" only). */
  curriculum?: number;
  /** Manifest path for an "n" seat. */
  weights?: string;
  /** Softmax temperature for an "n" seat; 0 or omitted = greedy. */
  temp?: number;
}

/** Four seats. Absolute seat order — index 0 is the subject by convention. */
export type TableSpec = [SeatSpec, SeatSpec, SeatSpec, SeatSpec];

/** The kind-letter string a table answers to in reports ("khhh"). */
export function kindString(table: TableSpec): string {
  return table.map((s) => s.kind).join("");
}

/** What `resolveTable` reads out of the flat option bag — a structural subset
 * of `HeadlessOptions`, named here so `spec.ts` does not import the harness. */
export interface LegacySeatOptions {
  ktune?: KTune;
  ktuneOpp?: KTune;
  plan?: boolean;
  standings?: boolean;
  consumer?: ConsumerParams;
  curriculum?: number;
  weights?: string;
  temp?: number;
}

const KINDS = new Set<string>(["h", "k", "o", "n", "r"]);

/** `--seats` validation, shared by the resolver and the CLI. */
export function parseKinds(seats: string, flag = "--seats"): SeatKind[] {
  if (seats.length !== 4 || [...seats].some((c) => !KINDS.has(c))) {
    die(`${flag} は h/k/o/n/r の4文字です: ${seats}`);
  }
  return [...seats] as SeatKind[];
}

/**
 * The legacy conventions, in one place and nowhere else:
 *
 *   - seat 0 carries `--ktune`; seats 1–3 carry `--ktune-opp`, or the
 *     subject's vector when no opp file was given;
 *   - `standings` / `consumer` / `curriculum` reach seat 0 alone (a layer
 *     handed to every seat would move both sides of a paired comparison);
 *   - every configurable reaches ONLY the kind that reads it: the vectors and
 *     the seat-0 layers go to "k" seats, `weights`/`temp` to "n" seats.
 *
 * `openArm` on a seats-string is DEFINED as `openArm(resolveTable(...))`, so
 * the equivalence is structural, and the pinned suites hold it behaviorally.
 *
 * 2026-08-25 EPOCH NOTE. Flag ROUTING is what this preserves — the flags keep
 * their meanings — but the "h" seat itself was re-bound that day to a frozen
 * copy of the default 計算 seat (`ai/frozen.ts`), configurable by nothing. So
 * a vector no longer reaches an "h" seat at all (before the epoch, the
 * `hand`/`riichi` model blocks deliberately did), and numbers in `runs/`
 * recorded before the epoch were measured against the OLD h population — they
 * are not comparable forward.
 */
export function resolveTable(seats: string, opts: LegacySeatOptions = {}): TableSpec {
  return parseKinds(seats).map((kind, seat) => ({
    kind,
    ktune: kind !== "k" ? undefined : seat === 0 ? opts.ktune : opts.ktuneOpp ?? opts.ktune,
    plan: kind === "k" ? opts.plan : undefined,
    standings: (opts.standings ?? false) && seat === 0 && kind === "k" ? true : undefined,
    consumer: seat === 0 && kind === "k" ? opts.consumer : undefined,
    curriculum: seat === 0 && kind === "k" ? opts.curriculum : undefined,
    weights: kind === "n" ? opts.weights : undefined,
    temp: kind === "n" ? opts.temp : undefined,
  })) as TableSpec;
}

/**
 * The `--table` file: `{"seats": [<spec>, <spec>, <spec>, <spec>]}`.
 *
 * Keys are whitelisted PER KIND: a table file is explicit per-seat authorship
 * (unlike the legacy flags, which broadcast by convention and let the kinds
 * ignore what they cannot read), so a key the kind cannot honour is refused —
 * an "h" seat is frozen (2026-08-25 epoch) and takes nothing, "k" takes the
 * vector and its layers, "n" the net fields, "o"/"r" nothing.
 */
const KIND_KEYS: Record<SeatKind, ReadonlySet<string>> = {
  h: new Set(["kind"]),
  r: new Set(["kind"]),
  o: new Set(["kind"]),
  k: new Set(["kind", "ktune", "plan", "standings", "consumer", "curriculum"]),
  n: new Set(["kind", "weights", "temp"]),
};

/**
 * Read a `--table` file: four explicit SeatSpecs, each seat's `ktune` either
 * inline or a path resolved RELATIVE TO THE TABLE FILE (so a table next to its
 * weight files stays portable), `consumer` a path likewise.
 *
 * Unlike `KTune`, the spec keys ARE whitelisted: a `KTune` passes unknown
 * sections through to constructors that may learn to read them, but a spec key
 * typo ("standing") would silently drop a whole component from the seat — the
 * exact class of silent mismeasurement this file format exists to end.
 */
export function loadTable(path: string, flag = "--table"): TableSpec {
  let text: string;
  try {
    text = Deno.readTextFileSync(path);
  } catch (e) {
    return die(`${flag} のファイルが読めません: ${path}\n${e instanceof Error ? e.message : e}`);
  }
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (e) {
    return die(`${flag} のJSONが壊れています: ${path}\n${e instanceof Error ? e.message : e}`);
  }
  const seats = (json as { seats?: unknown }).seats;
  if (!Array.isArray(seats) || seats.length !== 4) {
    die(`${flag} は {"seats": [席0, 席1, 席2, 席3]} の4席です: ${path}`);
  }
  const dir = path.replace(/[^/]*$/, "");
  // A relative path inside the file is relative TO the file; an absolute one
  // is taken as written.
  const at_ = (p: string) => p.startsWith("/") ? p : dir + p;
  const table = (seats as unknown[]).map((raw, i) => {
    const at = `${flag} の seats[${i}]`;
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
      die(`${at} はオブジェクトです: ${path}`);
    }
    const o = raw as Record<string, unknown>;
    if (typeof o.kind !== "string" || !KINDS.has(o.kind)) {
      die(`${at}.kind は h/k/o/n/r です: ${String(o.kind)}`);
    }
    const kind = o.kind as SeatKind;
    const allowed = KIND_KEYS[kind];
    for (const k of Object.keys(o)) {
      if (allowed.has(k)) continue;
      if (kind === "h") {
        die(
          `${at}: h席は凍結済みで設定を受け付けません (2026-08-25 epoch — ` +
            `設定できる席が要るなら kind を "k" にします): "${k}"`,
        );
      }
      die(`${at} (${kind}席) が取らないキー "${k}" (${[...allowed].join("/")})`);
    }
    const spec: SeatSpec = { kind };
    if (o.ktune !== undefined) {
      spec.ktune = typeof o.ktune === "string"
        ? loadKtune(at_(o.ktune), `${at}.ktune`)
        : o.ktune as KTune;
    }
    if (o.consumer !== undefined) {
      if (typeof o.consumer !== "string") die(`${at}.consumer はファイルパスです`);
      spec.consumer = loadConsumer(at_(o.consumer), `${at}.consumer`);
    }
    if (o.plan !== undefined) spec.plan = Boolean(o.plan);
    if (o.standings !== undefined) spec.standings = Boolean(o.standings);
    if (o.curriculum !== undefined) {
      if (typeof o.curriculum !== "number") die(`${at}.curriculum は数値です`);
      spec.curriculum = o.curriculum;
    }
    if (o.weights !== undefined) {
      if (typeof o.weights !== "string") die(`${at}.weights はファイルパスです`);
      spec.weights = at_(o.weights);
    }
    if (o.temp !== undefined) {
      if (typeof o.temp !== "number") die(`${at}.temp は数値です`);
      spec.temp = o.temp;
    }
    return spec;
  });
  return table as unknown as TableSpec;
}

/** JSON with recursively sorted keys, so equality is of VALUES, not key order. */
function canonical(v: unknown): string {
  if (Array.isArray(v)) return `[${v.map(canonical).join(",")}]`;
  if (typeof v === "object" && v !== null) {
    const keys = Object.keys(v as Record<string, unknown>).sort();
    return `{${
      keys
        .filter((k) => (v as Record<string, unknown>)[k] !== undefined)
        .map((k) => `${JSON.stringify(k)}:${canonical((v as Record<string, unknown>)[k])}`)
        .join(",")
    }}`;
  }
  return JSON.stringify(v) ?? "null";
}

/**
 * The paired environment guard: two arms may differ in SEAT 0 ONLY. Seats 1–3
 * are the environment, and an environment that differs between arms turns the
 * paired difference into a difference of fields — the exact confound that
 * mis-crowned the M11 champion. Compared on the RESOLVED specs (after every
 * ktune path has been loaded) and on canonical JSON, so two paths to one file,
 * or one vector written with its sections in a different order, still count as
 * the same environment.
 */
export function sameEnvironment(a: TableSpec, b: TableSpec): boolean {
  for (let s = 1; s < 4; s++) {
    if (canonical(a[s]) !== canonical(b[s])) return false;
  }
  return true;
}
