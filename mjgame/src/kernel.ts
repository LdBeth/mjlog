// The shanten / ukeire compute kernel — mjgame's front door to `mjrender/shanten.ts`.
//
// Every mjgame call site imports shanten/ukeireTypes from HERE, never from
// mjrender directly. The two implementations behind this module are the pure
// TypeScript one (mjrender's, unchanged and authoritative) and an optional C++
// shim, `native/libmjkernel.dylib`, that answers the same questions ~40x faster.
// Which one runs is invisible: `test/kernel_native_test.ts` fuzzes them against
// each other by the million and the results are identical, not approximate —
// unlike the float32 inference shim next door, this is integer arithmetic and
// there is no tolerance to argue about.
//
// The direction of the dependency matters: mjrender knows nothing about mjgame
// and nothing about the dylib. It stays a standalone oracle; the acceleration
// lives on this side of the fence.
//
// Gate, dylib location, permission etiquette and failure behaviour are all
// deliberately identical to `src/rl/net.ts` — see the comment there. In short:
//
//   MJGAME_NATIVE=0   force TypeScript
//   MJGAME_NATIVE=1   require native; anything missing throws, naming the build
//   unset             try native, fall silently back to TypeScript
//
// Reading the variable needs `--allow-env=MJGAME_NATIVE`; the dlopen needs
// `--allow-ffi`. Both are asked for with `querySync`, never taken, so a run that
// never wanted FFI is never interrupted by a prompt.

import {
  countsFromTiles,
  shanten as shantenTS,
  ukeireTypes as ukeireTS,
} from "mjrender/shanten.ts";

export { countsFromTiles };

const KERNEL_SYMBOLS = {
  mj_kernel_version: { parameters: [], result: "i32" },
  mj_shanten: { parameters: ["buffer", "i32", "i32"], result: "i32" },
  mj_ukeire_mask: { parameters: ["buffer", "i32", "i32", "i32"], result: "u64" },
} as const;

export type KernelLib = Deno.DynamicLibrary<typeof KERNEL_SYMBOLS>;

const LIB_EXT = Deno.build.os === "windows"
  ? ".dll"
  : Deno.build.os === "darwin"
  ? ".dylib"
  : ".so";

/** The one place the kernel is looked for, module-relative (src/ → ../native/). */
export const KERNEL_LIB_URL = new URL(`../native/libmjkernel${LIB_EXT}`, import.meta.url);

/** The ABI this module was written against; a mismatch means a stale dylib. */
export const KERNEL_ABI = 1;

const BUILD_HINT = `mjgame/ で \`deno task build-kernel\` ` +
  `(sh native/build_kernel.sh) を実行し、--allow-ffi をつけて起動してください`;

type Gate = "off" | "require" | "auto";

/** Asked, never taken — `querySync` prompts for nothing. */
function granted(desc: Deno.PermissionDescriptor): boolean {
  try {
    return Deno.permissions.querySync(desc).state === "granted";
  } catch {
    return false;
  }
}

function gate(): Gate {
  if (!granted({ name: "env", variable: "MJGAME_NATIVE" })) return "auto";
  const v = Deno.env.get("MJGAME_NATIVE");
  if (v === "0") return "off";
  if (v === "1") return "require";
  return "auto";
}

// Resolved once, on the first shanten/ukeire call rather than at import: the
// gate is an environment variable, and deciding at import would freeze it before
// a caller (a test, mostly) ever had the chance to set it.
let lib: KernelLib | null | undefined;

function open(): KernelLib | null {
  if (lib !== undefined) return lib;
  const g = gate();
  if (g === "off") {
    lib = null;
    return null;
  }
  const required = g === "require";
  let error = "";
  if (!granted({ name: "ffi", path: KERNEL_LIB_URL })) {
    error = "--allow-ffi がありません";
  } else {
    try {
      const opened = Deno.dlopen(KERNEL_LIB_URL, KERNEL_SYMBOLS);
      const abi = opened.symbols.mj_kernel_version();
      if (abi !== KERNEL_ABI) {
        opened.close();
        error = `ABI が合いません (dylib=${abi}, 期待=${KERNEL_ABI})`;
      } else {
        lib = opened;
        return lib;
      }
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  }
  lib = null;
  if (required) {
    throw new Error(
      `MJGAME_NATIVE=1 ですが native カーネルを読み込めません: ${error} — ${BUILD_HINT}`,
    );
  }
  return null;
}

/** Whether shanten/ukeireTypes currently go through the dylib. Resolves the gate. */
export function kernelNative(): boolean {
  return open() !== null;
}

/**
 * Releases the dylib and re-arms the gate. For tests that flip MJGAME_NATIVE;
 * production never calls it (the kernel lives for the process).
 */
export function closeKernel(): void {
  if (lib) lib.close();
  lib = undefined;
}

// One buffer for the whole process. Shanten is a leaf call — nothing re-enters
// it between the fill and the FFI call — so a fresh Uint8Array per call would be
// pure garbage, and at ~10^7 calls per self-play run that garbage is the point.
const buf = new Uint8Array(34);

/** False when `counts` is not the 34-length vector the dylib's ABI assumes. */
function fill(counts: number[]): boolean {
  if (counts.length !== 34) return false;
  for (let t = 0; t < 34; t++) buf[t] = counts[t];
  return true;
}

/**
 * Minimum shanten across standard / chiitoitsu / kokushi.
 * `closed` disables the two closed-only forms (i.e. when melds were called).
 */
export function shanten(counts: number[], openMelds = 0, closed = true): number {
  const k = open();
  if (k === null || !fill(counts)) return shantenTS(counts, openMelds, closed);
  return k.symbols.mj_shanten(buf, openMelds, closed ? 1 : 0);
}

/**
 * Tile types whose addition lowers shanten of a resting (3n+1) hand.
 * At shanten 0 these are exactly the winning (wait) tiles.
 * Callers that already computed the hand's shanten can pass it as `base`.
 *
 * This is the call the kernel exists for: the TypeScript costs 34 shanten
 * evaluations, the dylib costs one crossing.
 */
export function ukeireTypes(
  counts: number[],
  openMelds = 0,
  closed = true,
  base = shanten(counts, openMelds, closed),
): number[] {
  const k = open();
  if (k === null || !fill(counts)) return ukeireTS(counts, openMelds, closed, base);

  const raw = k.symbols.mj_ukeire_mask(buf, openMelds, closed ? 1 : 0, base);
  const mask = typeof raw === "bigint" ? raw : BigInt(raw);
  const lo = Number(mask & 0xffffffffn);
  const hi = Number((mask >> 32n) & 0x3n);

  const out: number[] = [];
  for (let t = 0; t < 32; t++) if ((lo >>> t) & 1) out.push(t);
  for (let t = 32; t < 34; t++) if ((hi >>> (t - 32)) & 1) out.push(t);
  return out;
}
