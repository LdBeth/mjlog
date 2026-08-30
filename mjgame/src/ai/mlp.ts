// The shared small-MLP runtime — one forward pass, three implementations that
// must agree to the last bit.
//
// WHAT IT IS FOR. The 計算 seat's two most consequential hand-written decisions
// are moving to learned heads: M13, the push/fold gate (`ai/fold.ts`, 1 output),
// and M14, the deal-in read (`ai/dealin.ts`, 34 rows per opponent). Both are
// tiny fully-connected nets — hundreds to a few thousand weights — that ship
// INLINE in a `--ktune` JSON, so this module owns their shape, their
// validation, and their arithmetic. It knows no mahjong: it is dims,
// activations and floats.
//
// THE BIT-EXACT CONTRACT. A seat's decision fingerprints are pinned
// (`computed_test`/`calibration_test` pin whole-hanchan streams), so the head
// may not answer one ulp differently depending on whether a dylib happened to
// be built. The reference is the `src/rl/net.ts` forward loop, transliterated
// here verbatim:
//
//   * the accumulator is a JS number, i.e. an IEEE double, seeded with the f32
//     bias and grown by `w[i] * x[i]` in ASCENDING i — sequential rounding, not
//     a reordered/vectorised sum;
//   * relu is applied to that DOUBLE (`acc > 0 ? acc : 0`) and only then is the
//     result STORED into a Float32Array, i.e. rounded once to f32
//     (round-to-nearest-even, which is what C's `(float)` cast does too);
//   * every layer therefore reads f32 inputs and writes f32 outputs — doubles
//     never cross a layer boundary.
//
// (The middle rule is defence in depth: a double accumulator over a few dozen
// float32 products absorbs a reordering into the f32 store, and the fuzz
// confirms it — but the reorderings a LIBRARY performs are float32 partial sums
// and vector reductions, which do show, and that is the reason for the rule.)
//
// `native/mlp.c` mirrors those three rules literally and is built with
// `-ffp-contract=off` so the compiler may not fuse `acc + w*x` into an FMA that
// JavaScript would never have performed. `train/common.py`'s `mlp_forward_np`
// is the third mirror (an explicit Python loop — `np.dot` reorders the sum) and
// `test/fixtures/mlp-parity.json` is what pins it to this file.
//
// WHY NOT ACCELERATE (plan D1). `native/rlnet.c` already has a generic MLP over
// `cblas_sgemv`, and reusing it would have been free — but BLAS reorders
// summation, which is why `test/rl_native_test.ts` grades that path with a
// 1e-4 tolerance. A tolerance is fine for a policy net whose output is argmaxed
// over 78 slots; it is not fine for a gate whose sign decides a fold, because a
// single flipped decision re-writes a whole hanchan. So `libmjmlp` is a
// SEPARATE dylib of plain C loops, graded at zero tolerance, and it buys only
// the removal of the interpreter overhead.
//
// ERRORS. `validateMlp` THROWS rather than calling `die`: it is called from
// `merge*` functions that tests exercise directly, and a `Deno.exit(2)` is not
// assertable. The CLI layer that loads a ktune file is where a bad block turns
// into a die message; the `where` argument is what makes that message name the
// block.
//
// Gate, dylib location and failure behaviour are deliberately identical to
// `src/kernel.ts` and `src/rl/net.ts`:
//
//   MJGAME_NATIVE=0   force TypeScript
//   MJGAME_NATIVE=1   require native; anything missing throws, naming the build
//   unset             try native, fall silently back to TypeScript

/** Activation of one layer. Only these two exist; the JSON is validated. */
export type MlpAct = "relu" | "none";

/** One dense layer. `w` is row-major `[out][in]`, `b` is `[out]`. */
export interface MlpLayer {
  in: number;
  out: number;
  act: MlpAct;
  /** `[out][in]` row-major — `w[o * in + i]`. */
  w: number[];
  /** `[out]`. */
  b: number[];
}

/**
 * A whole head as it rides inside a ktune JSON. `fv` is the FEATURE version of
 * the vector this net expects: bumping the feature list bumps `fv`, and a stale
 * weight file is then refused instead of being fed the wrong columns.
 */
export interface MlpSpec {
  fv: number;
  layers: MlpLayer[];
}

/** A built head: float32 weights, scratch buffers, and maybe a native handle. */
export interface Mlp {
  readonly fv: number;
  readonly inputs: number;
  readonly outputs: number;
  readonly layers: readonly {
    in: number;
    out: number;
    act: MlpAct;
    w: Float32Array;
    b: Float32Array;
  }[];
  /** One output buffer per layer, reused for the life of the head. */
  readonly scratch: readonly Float32Array[];
  native?: { handle: bigint; lib: MlpLib };
}

// ---------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------

function fail(where: string, msg: string): never {
  throw new Error(`${where}: ${msg}`);
}

function isPosInt(v: unknown): v is number {
  return typeof v === "number" && Number.isInteger(v) && v > 0;
}

function numbers(v: unknown, want: number, where: string, what: string): number[] {
  if (!Array.isArray(v)) fail(where, `${what} は数値の配列です`);
  const a = v as unknown[];
  if (a.length !== want) fail(where, `${what} の長さが ${a.length} (期待 ${want})`);
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    if (typeof x !== "number" || !Number.isFinite(x)) {
      fail(where, `${what}[${i}] が有限の数値ではありません (${String(x)})`);
    }
  }
  return a as number[];
}

/**
 * Check an untrusted `MlpSpec` against the shape the caller's feature list
 * demands. Everything is refused loudly: an empty net, a broken in/out chain, a
 * width that does not match the feature vector, a stale `fv`, an unknown
 * activation, a weight array of the wrong length, and any NaN/Infinity.
 *
 * Returns the same object, typed — nothing is copied here; `buildMlp` is what
 * takes float32 copies.
 */
export function validateMlp(
  spec: unknown,
  want: { inputs: number; outputs: number; fv: number },
  where: string,
): MlpSpec {
  if (typeof spec !== "object" || spec === null || Array.isArray(spec)) {
    fail(where, `{fv, layers} のオブジェクトです`);
  }
  const s = spec as Record<string, unknown>;
  if (typeof s.fv !== "number" || !Number.isFinite(s.fv)) {
    fail(where, `fv が数値ではありません (${String(s.fv)})`);
  }
  if (s.fv !== want.fv) {
    fail(where, `fv=${s.fv} は特徴量版 ${want.fv} と違います (古い重みファイルです)`);
  }
  if (!Array.isArray(s.layers) || s.layers.length === 0) {
    fail(where, "layers が空です");
  }
  const raw = s.layers as unknown[];
  const layers: MlpLayer[] = [];
  for (let li = 0; li < raw.length; li++) {
    const at = `layers[${li}]`;
    const l = raw[li];
    if (typeof l !== "object" || l === null || Array.isArray(l)) {
      fail(where, `${at} は {in, out, act, w, b} です`);
    }
    const o = l as Record<string, unknown>;
    if (!isPosInt(o.in)) fail(where, `${at}.in が正整数ではありません (${String(o.in)})`);
    if (!isPosInt(o.out)) fail(where, `${at}.out が正整数ではありません (${String(o.out)})`);
    if (o.act !== "relu" && o.act !== "none") {
      fail(where, `${at}.act は "relu" か "none" です (${String(o.act)})`);
    }
    const nIn = o.in as number;
    const nOut = o.out as number;
    const prev = li === 0 ? want.inputs : layers[li - 1].out;
    if (nIn !== prev) {
      fail(
        where,
        li === 0
          ? `${at}.in=${nIn} は入力次元 ${want.inputs} と違います`
          : `${at}.in=${nIn} は前層の out=${prev} と違います`,
      );
    }
    layers.push({
      in: nIn,
      out: nOut,
      act: o.act,
      w: numbers(o.w, nIn * nOut, where, `${at}.w`),
      b: numbers(o.b, nOut, where, `${at}.b`),
    });
  }
  const last = layers[layers.length - 1].out;
  if (last !== want.outputs) {
    fail(where, `最終層の out=${last} は出力次元 ${want.outputs} と違います`);
  }
  return { fv: s.fv, layers };
}

// ---------------------------------------------------------------------------
// the native shim — same gate as src/kernel.ts, its own dylib
// ---------------------------------------------------------------------------

const MLP_SYMBOLS = {
  mjmlp_abi: { parameters: [], result: "i32" },
  mjmlp_create: { parameters: ["i32", "buffer", "buffer", "buffer"], result: "i64" },
  mjmlp_forward: { parameters: ["i64", "buffer", "buffer"], result: "void" },
  mjmlp_forward_batch: { parameters: ["i64", "i32", "buffer", "buffer"], result: "void" },
  mjmlp_destroy: { parameters: ["i64"], result: "void" },
} as const;

export type MlpLib = Deno.DynamicLibrary<typeof MLP_SYMBOLS>;

const LIB_EXT = Deno.build.os === "windows"
  ? ".dll"
  : Deno.build.os === "darwin"
  ? ".dylib"
  : ".so";

/** The one place the shim is looked for, module-relative (src/ai/ → ../../native/). */
export const MLP_LIB_URL = new URL(`../../native/libmjmlp${LIB_EXT}`, import.meta.url);

/** The ABI this module was written against; a mismatch means a stale dylib. */
export const MLP_ABI = 1;

const BUILD_HINT = `mjgame/ で \`deno task build-mlp\` ` +
  `(sh native/build_mlp.sh) を実行し、--allow-ffi をつけて起動してください`;

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

// Resolved on the first `buildMlp` rather than at import: the gate is an
// environment variable, and deciding at import would freeze it before a caller
// (a test, mostly) ever had the chance to set it.
let lib: MlpLib | null | undefined;

function open(): MlpLib | null {
  if (lib !== undefined) return lib;
  const g = gate();
  if (g === "off") {
    lib = null;
    return null;
  }
  const required = g === "require";
  let error = "";
  if (!granted({ name: "ffi", path: MLP_LIB_URL })) {
    error = "--allow-ffi がありません";
  } else {
    try {
      const opened = Deno.dlopen(MLP_LIB_URL, MLP_SYMBOLS);
      const abi = opened.symbols.mjmlp_abi();
      if (abi !== MLP_ABI) {
        opened.close();
        error = `ABI が合いません (dylib=${abi}, 期待=${MLP_ABI})`;
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
      `MJGAME_NATIVE=1 ですが native MLP を読み込めません: ${error} — ${BUILD_HINT}`,
    );
  }
  return null;
}

/**
 * Releases the dylib and re-arms the gate. For tests that flip MJGAME_NATIVE;
 * production never calls it (the shim lives for the process). Heads already
 * built keep their handles — close those with `closeMlp` FIRST.
 */
export function closeMlpLib(): void {
  if (lib) lib.close();
  lib = undefined;
}

// ---------------------------------------------------------------------------
// build / free
// ---------------------------------------------------------------------------

/**
 * Weights and biases in one flat float32 buffer: per layer, in order, the
 * row-major `[out][in]` matrix followed by the `[out]` bias — the same layout
 * `src/rl/net.ts` writes for `policy.f32`, and what `mjmlp_create` reads.
 */
export function packMlp(m: Mlp): Float32Array {
  let n = 0;
  for (const l of m.layers) n += l.w.length + l.b.length;
  const blob = new Float32Array(n);
  let off = 0;
  for (const l of m.layers) {
    blob.set(l.w, off);
    off += l.w.length;
    blob.set(l.b, off);
    off += l.b.length;
  }
  return blob;
}

/**
 * Build a runnable head from a validated spec. Weights are COPIED into
 * `Float32Array`s (so the JSON may be mutated or dropped afterwards) and, if
 * the gate allows it, handed to the dylib as well.
 *
 * Under `MJGAME_NATIVE=1` a missing/stale dylib throws HERE, at construction,
 * rather than silently on the first decision.
 */
export function buildMlp(spec: MlpSpec): Mlp {
  const layers = spec.layers.map((l) => ({
    in: l.in,
    out: l.out,
    act: l.act,
    w: Float32Array.from(l.w),
    b: Float32Array.from(l.b),
  }));
  const m: Mlp = {
    fv: spec.fv,
    inputs: layers[0].in,
    outputs: layers[layers.length - 1].out,
    layers,
    scratch: layers.map((l) => new Float32Array(l.out)),
  };

  const g = gate();
  if (g === "off") return m;
  const l = open();
  if (!l) return m;
  const dims = new Int32Array(layers.length + 1);
  const acts = new Uint8Array(layers.length);
  dims[0] = layers[0].in;
  layers.forEach((la, i) => {
    dims[i + 1] = la.out;
    acts[i] = la.act === "relu" ? 1 : 0;
  });
  const handle = l.symbols.mjmlp_create(layers.length, dims, acts, packMlp(m));
  const h = typeof handle === "bigint" ? handle : BigInt(handle);
  if (h === 0n) {
    if (g === "require") {
      throw new Error(`MJGAME_NATIVE=1 ですが mjmlp_create が失敗しました (in=${m.inputs})`);
    }
    return m;
  }
  m.native = { handle: h, lib: l };
  return m;
}

/** Whether this head's forward pass currently runs natively. */
export function mlpNative(m: Mlp): boolean {
  return m.native !== undefined;
}

/**
 * Releases the native context, if any; the head keeps working on the TS path.
 * Optional — process exit frees everything — but a run that builds one head per
 * seat should close them in `close()`.
 */
export function closeMlp(m: Mlp): void {
  if (!m.native) return;
  m.native.lib.symbols.mjmlp_destroy(m.native.handle);
  m.native = undefined;
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

/**
 * The reference loop, `src/rl/net.ts:785-795` transliterated with offsets so a
 * batch row needs no subarray allocation. Double accumulate in ascending `i`,
 * relu on the double, one f32 store per output — see the contract at the top.
 */
function forwardTS(m: Mlp, x: Float32Array, xOff: number, dst: Float32Array, dOff: number): void {
  let src = x;
  let sOff = xOff;
  const n = m.layers.length;
  for (let li = 0; li < n; li++) {
    const l = m.layers[li];
    const last = li === n - 1;
    const y = last ? dst : m.scratch[li];
    const yOff = last ? dOff : 0;
    for (let o = 0; o < l.out; o++) {
      const base = o * l.in;
      let acc = l.b[o];
      for (let i = 0; i < l.in; i++) acc += l.w[base + i] * src[sOff + i];
      y[yOff + o] = l.act === "relu" ? (acc > 0 ? acc : 0) : acc;
    }
    src = y;
    sOff = yOff;
  }
}

/**
 * One forward pass. `x` must be `m.inputs` long.
 *
 * With `out` given (length `m.outputs`) the result is written there and
 * returned; otherwise it goes into a per-head scratch buffer that the NEXT call
 * on this head overwrites — copy it if you need to keep it. `x` and `out` must
 * not overlap.
 */
export function mlpForward(m: Mlp, x: Float32Array, out?: Float32Array): Float32Array {
  if (x.length !== m.inputs) {
    throw new Error(`入力長 ${x.length} は ${m.inputs} であるべきです`);
  }
  const dst = out ?? m.scratch[m.scratch.length - 1];
  if (dst.length !== m.outputs) {
    throw new Error(`出力長 ${dst.length} は ${m.outputs} であるべきです`);
  }
  if (m.native) {
    m.native.lib.symbols.mjmlp_forward(m.native.handle, x, dst);
    return dst;
  }
  forwardTS(m, x, 0, dst, 0);
  return dst;
}

/**
 * `n` forward passes in one go. `xs` holds the inputs contiguously
 * (`n * m.inputs`), `outs` receives the outputs the same way
 * (`n * m.outputs`). Natively this is ONE crossing — which is the whole point
 * for M14, where a decision asks 34 rows per opponent.
 *
 * Bit-identical to `n` separate `mlpForward` calls, by construction: the C
 * batch entry point is the single-row loop run `n` times.
 */
export function mlpForwardBatch(m: Mlp, n: number, xs: Float32Array, outs: Float32Array): void {
  if (n < 0 || !Number.isInteger(n)) throw new Error(`行数 ${n} が不正です`);
  if (xs.length < n * m.inputs) {
    throw new Error(`入力長 ${xs.length} は ${n * m.inputs} 以上であるべきです`);
  }
  if (outs.length < n * m.outputs) {
    throw new Error(`出力長 ${outs.length} は ${n * m.outputs} 以上であるべきです`);
  }
  if (n === 0) return;
  if (m.native) {
    m.native.lib.symbols.mjmlp_forward_batch(m.native.handle, n, xs, outs);
    return;
  }
  for (let r = 0; r < n; r++) forwardTS(m, xs, r * m.inputs, outs, r * m.outputs);
}
