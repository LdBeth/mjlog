// `native/libmjev` — the FFI front door to the 計算 seat's expected-value core.
//
// WHY THIS ONE HAS NO FALLBACK. `src/kernel.ts`, `src/ai/mlp.ts` and
// `src/rl/net.ts` all wrap an ACCELERATOR: the TypeScript beside them is the
// reference implementation, so a missing dylib is a slowdown and nothing more.
// `libmjev` is not that. It has no TypeScript twin (owner decision,
// 2026-08-30) — the DP that prices a discard exists only in C++ — so a seat
// carrying an `ev` block cannot degrade gracefully: it would silently play a
// DIFFERENT game. Hence the flag discipline of CLAUDE.md applied at the
// loader: absent block ⇒ this module is never imported into a decision and no
// FFI is touched; present block ⇒ the dylib is REQUIRED, `--allow-ffi` is
// REQUIRED, and `MJGAME_NATIVE=0` is REFUSED rather than honoured.
//
// So there are exactly two outcomes here: a working core, or a throw that names
// the build line. There is no third.
//
// Everything else — where the dylib is looked for, `querySync` instead of
// `request`, the ABI check, `closeEvLib()` for tests that flip the gate — is
// deliberately identical to `src/ai/mlp.ts`.

import { DBLS_LEN, EV_ABI, INTS_LEN, OUT_LEN, REST_META_LEN } from "./evlayout.ts";
import type { EvParams } from "./evparams.ts";
import { packEvParams } from "./evparams.ts";

const EV_SYMBOLS = {
  mjev_abi: { parameters: [], result: "i32" },
  mjev_create: { parameters: ["buffer", "i32"], result: "i64" },
  mjev_destroy: { parameters: ["i64"], result: "void" },
  mjev_score: { parameters: ["buffer", "buffer"], result: "i32" },
  mjev_shanten: { parameters: ["i64", "buffer", "i32", "i32"], result: "i32" },
  mjev_ukeire_mask: { parameters: ["i64", "buffer", "i32", "i32", "i32"], result: "u64" },
  mjev_eval_discard: { parameters: ["i64", "buffer", "buffer", "buffer"], result: "i32" },
  mjev_eval_rest: { parameters: ["i64", "buffer", "buffer", "buffer"], result: "f64" },
} as const;

export type EvLib = Deno.DynamicLibrary<typeof EV_SYMBOLS>;

const LIB_EXT = Deno.build.os === "windows"
  ? ".dll"
  : Deno.build.os === "darwin"
  ? ".dylib"
  : ".so";

/** The one place the core is looked for, module-relative (src/ai/ → ../../native/). */
export const EV_LIB_URL = new URL(`../../native/libmjev${LIB_EXT}`, import.meta.url);

const BUILD_HINT = `mjgame/ で \`deno task build-ev\` ` +
  `(sh native/build_ev.sh) を実行し、--allow-ffi をつけて起動してください`;

/**
 * The one refusal message. Every cause names every remedy on purpose: whoever
 * reads this has an `ev` block in a ktune and needs to know that the block is
 * native-only, that `MJGAME_NATIVE=0` will not talk them out of it, and which
 * line builds the thing.
 */
function refuse(why: string): Error {
  return new Error(
    `ev ブロックは native の libmjev を必須とします (${why}) — ${BUILD_HINT}。` +
      `ev では MJGAME_NATIVE=0 は拒否されます (TypeScript の代替実装が無く、` +
      `黙って別の打牌になるため)`,
  );
}

/** Asked, never taken — `querySync` prompts for nothing. */
function granted(desc: Deno.PermissionDescriptor): boolean {
  try {
    return Deno.permissions.querySync(desc).state === "granted";
  } catch {
    return false;
  }
}

/** `MJGAME_NATIVE=0`, read only when `--allow-env=MJGAME_NATIVE` was granted. */
function gateOff(): boolean {
  if (!granted({ name: "env", variable: "MJGAME_NATIVE" })) return false;
  return Deno.env.get("MJGAME_NATIVE") === "0";
}

// Resolved on the first `buildEv`/`evScore` rather than at import: the gate is
// an environment variable, and deciding at import would freeze it before a
// caller (a test, mostly) ever had the chance to set it.
let lib: EvLib | null = null;

/** The loaded core, or a throw. Never returns null. */
function openOrThrow(): EvLib {
  if (lib) return lib;
  if (gateOff()) throw refuse("MJGAME_NATIVE=0");
  if (!granted({ name: "ffi", path: EV_LIB_URL })) throw refuse("--allow-ffi がありません");
  let opened: EvLib;
  try {
    opened = Deno.dlopen(EV_LIB_URL, EV_SYMBOLS);
  } catch (e) {
    throw refuse(e instanceof Error ? e.message : String(e));
  }
  const abi = opened.symbols.mjev_abi();
  if (abi !== EV_ABI) {
    opened.close();
    throw refuse(`ABI が合いません (dylib=${abi}, 期待=${EV_ABI})`);
  }
  lib = opened;
  return lib;
}

/**
 * Releases the dylib and re-arms the gate. For tests that flip MJGAME_NATIVE or
 * move the artifact aside; production never calls it (the core lives for the
 * process). Contexts already created keep their handles — free those with
 * `closeEv` FIRST.
 */
export function closeEvLib(): void {
  if (lib) lib.close();
  lib = null;
}

/** Whether the core is currently open. Never opens it — `buildEv` does that. */
export function evNative(): boolean {
  return lib !== null;
}

// ---------------------------------------------------------------------------
// the context
// ---------------------------------------------------------------------------

/**
 * One seat's evaluation context: the native handle, the parameters it was built
 * from, and the four buffers every call reuses. Reused rather than allocated
 * per decision because a self-play run makes millions of them and the FFI
 * boundary copies nothing — the dylib reads and writes these arrays in place.
 */
export interface EvCore {
  handle: bigint | number;
  params: EvParams;
  ints: Int32Array;
  dbls: Float64Array;
  out: Float64Array;
  meta: Float64Array;
}

/**
 * Build a context. THROWS — with the build line named — when the dylib is
 * missing or stale, when `--allow-ffi` was not granted, or when
 * `MJGAME_NATIVE=0` asked for a TypeScript path that does not exist.
 */
export function buildEv(params: EvParams): EvCore {
  const l = openOrThrow();
  const packed = packEvParams(params);
  const raw = l.symbols.mjev_create(packed, packed.length);
  const handle = typeof raw === "bigint" ? raw : BigInt(raw);
  if (handle === 0n) {
    throw refuse(`mjev_create が失敗しました (params=${packed.length})`);
  }
  return {
    handle,
    params,
    ints: new Int32Array(INTS_LEN),
    dbls: new Float64Array(DBLS_LEN),
    out: new Float64Array(OUT_LEN),
    meta: new Float64Array(REST_META_LEN),
  };
}

/**
 * The handle as the FFI wants it. `EvCore.handle` is typed `bigint | number`
 * because that is what a `i64` result may come back as; the dylib only ever
 * takes a bigint.
 */
function h(core: EvCore): bigint {
  return typeof core.handle === "bigint" ? core.handle : BigInt(core.handle);
}

/**
 * Frees the native context. Idempotent; the buffers stay alive for the GC.
 * Close contexts BEFORE `closeEvLib()` — with the library already released
 * there is nothing left to call `mjev_destroy` through, and the allocation
 * simply rides to process exit.
 */
export function closeEv(core: EvCore): void {
  const handle = h(core);
  if (handle === 0n) return;
  core.handle = 0n;
  if (lib) lib.symbols.mjev_destroy(handle);
}

// ---------------------------------------------------------------------------
// entry points
// ---------------------------------------------------------------------------

/**
 * The stateless scorer. `inp` is `SCORE_IN_LEN` long in the `S_*` layout, `out`
 * `SCORE_OUT_LEN` long in the `SO_*` one. No context is needed: nothing about a
 * finished hand is memoised.
 */
export function evScore(inp: Int32Array, out: Int32Array): void {
  const rc = openOrThrow().symbols.mjev_score(inp, out);
  if (rc !== 0) throw new Error(`mjev_score が ${rc} を返しました`);
}

// One buffer for the whole process, as in `src/kernel.ts`: shanten is a leaf
// call, nothing re-enters it between the fill and the crossing, and a fresh
// Uint8Array per call would be pure garbage at 10^7 calls per run.
const countsBuf = new Uint8Array(34);

function fill(counts: number[]): void {
  if (counts.length !== 34) throw new Error(`counts の長さが ${counts.length} (期待 34)`);
  for (let t = 0; t < 34; t++) countsBuf[t] = counts[t];
}

/**
 * Minimum shanten across standard / chiitoitsu / kokushi — `shanten()` of
 * `src/kernel.ts` exactly, but off this context's own memo (so two seats in one
 * process share nothing and need no warm-up).
 */
export function evShanten(
  core: EvCore,
  counts: number[],
  openMelds = 0,
  closed = true,
): number {
  fill(counts);
  return openOrThrow().symbols.mjev_shanten(
    h(core),
    countsBuf,
    openMelds,
    closed ? 1 : 0,
  );
}

/**
 * Tile types whose addition lowers shanten below `base` — `ukeireTypes()` of
 * `src/kernel.ts` exactly, types already held four times skipped.
 */
export function evUkeireTypes(
  core: EvCore,
  counts: number[],
  openMelds = 0,
  closed = true,
  base = evShanten(core, counts, openMelds, closed),
): number[] {
  fill(counts);
  const raw = openOrThrow().symbols.mjev_ukeire_mask(
    h(core),
    countsBuf,
    openMelds,
    closed ? 1 : 0,
    base,
  );
  const mask = typeof raw === "bigint" ? raw : BigInt(raw);
  const lo = Number(mask & 0xffffffffn);
  const hi = Number((mask >> 32n) & 0x3n);

  const out: number[] = [];
  for (let t = 0; t < 32; t++) if ((lo >>> t) & 1) out.push(t);
  for (let t = 32; t < 34; t++) if ((hi >>> (t - 32)) & 1) out.push(t);
  return out;
}

/**
 * The discard DP. Reads `core.ints`/`core.dbls`, writes `core.out` — the caller
 * fills the first two and reads the third; nothing is passed or returned.
 * A non-zero return code (a malformed hidden-information distribution, an
 * unfilled unit) throws rather than leaving a half-written `out` to be argmaxed.
 */
export function evEvalDiscard(core: EvCore): void {
  const rc = openOrThrow().symbols.mjev_eval_discard(
    h(core),
    core.ints,
    core.dbls,
    core.out,
  );
  if (rc !== 0) throw new Error(`mjev_eval_discard が ${rc} を返しました`);
}

/** The 13-tile DP: the value of holding this hand. Meta lands in `core.meta`. */
export function evEvalRest(core: EvCore): number {
  return openOrThrow().symbols.mjev_eval_rest(
    h(core),
    core.ints,
    core.dbls,
    core.meta,
  );
}
