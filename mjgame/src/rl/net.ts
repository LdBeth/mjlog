// Inference-only MLP: manifest + flat float32 blob → a forward pass.
//
// Training happens in Python/MLX; this side never learns, it only reads. The
// on-disk format is a FROZEN contract:
//
//   manifest.json
//     {"version":1,"arch":"mlp","features":{"planes":48,"scalars":42},
//      "actions":78,
//      "layers":[{"in":1738,"out":512,"act":"relu"},
//                {"in":512,"out":256,"act":"relu"},
//                {"in":256,"out":79,"act":"none"}],
//      "blob":"policy.f32","attn":"attn.f32"}
//
//   policy.f32 — for each layer in order: the weight matrix row-major
//   [out][in], then the bias [out]; all little-endian float32, concatenated
//   with no header and no padding. `blob` is relative to the manifest file.
//
//   attn.f32 — feature v4 ONLY, same [out][in]+bias convention, in the order
//   `train/V4_SPEC.md` freezes. `attn` is relative to the manifest file too.
//
// `version` is the FILE format's version, which is 1 and has not changed; the
// feature layout the weights were trained on is what `features` names, and
// `checkManifest` rejects anything that is not the encoder's current one.
//
// FEATURE v3 vs v4 — the one thing `features` cannot say. v4 left the planes
// and scalars alone (48/42 either way), so the marker for "these weights want
// the river encoder" is the PRESENCE of the `attn` entry, and it is what sets
// the width the first layer must declare:
//
//   no "attn"  ⇒ v3: fc1 takes 1674 = planes ++ scalars
//   "attn"     ⇒ v4: fc1 takes 1738 = planes ++ scalars ++ z(64)
//
// Both are loadable, always: play and bench against a v3 snapshot must keep
// working after the encoder lands, so a v3 manifest is not a stale file to be
// rejected — it is a net that simply has no seq path.
//
// Output = 79: elements 0..77 are the action logits and element 78 is the VALUE
// head, which inference ignores — it is carried in the same tensor purely so
// the trainer can share the trunk.

import {
  expandToken,
  FEATURES,
  INPUT_LEN,
  SEQ_DENSE,
  SEQ_MAX,
  SEQ_TOKEN_BYTES,
} from "./features.ts";

export type Activation = "relu" | "none";

export interface LayerSpec {
  in: number;
  out: number;
  act: Activation;
}

export interface Manifest {
  version: number;
  arch: string;
  features: { planes: number; scalars: number };
  actions: number;
  layers: LayerSpec[];
  blob: string;
  /**
   * Feature v4 only: the river encoder's weight file, beside `blob`. Its
   * presence IS the v4 marker — see the file header.
   */
  attn?: string;
}

export interface Layer extends LayerSpec {
  w: Float32Array; // [out][in], row-major
  b: Float32Array; // [out]
}

export interface Net {
  manifest: Manifest;
  path: string;
  layers: Layer[];
  /** Width of the network's output, = actions + 1 (the value head). */
  outputs: number;
  /** Feature v4 only: the river encoder. Absent ⇒ this is a v3 net. */
  attn?: Attn;
  /** Set by `loadNative` when the Accelerate shim took this net over. */
  native?: NativeCtx;
}

/** A live `rlnet` context. Opaque to everything but `net.ts`. */
export interface NativeCtx {
  handle: bigint;
  lib: RlnetLib;
}

/** Logits width the caller should read; index `ACTION_OUTPUTS` is the value. */
export const VALUE_INDEX = FEATURES.actions;

// ---------------------------------------------------------------------------
// feature v4: the attention river encoder (train/V4_SPEC.md)
// ---------------------------------------------------------------------------

/** Encoder width. */
export const SEQ_D_MODEL = 64;
/** Attention heads; head dim is `SEQ_D_MODEL / SEQ_HEADS` = 16. */
export const SEQ_HEADS = 4;
/** Head dimension. */
export const SEQ_HEAD_DIM = SEQ_D_MODEL / SEQ_HEADS;
/**
 * The softmax scale, FROZEN as 1/4 by the spec. It happens to equal
 * 1/√(head dim), but it is written as a constant on purpose: the spec names the
 * number, not the formula, so a future head-dim change must not silently move
 * the scale in one implementation and not the others.
 */
export const SEQ_SCALE = 1 / 4;
/** Width of `z`, the vector the encoder appends to the policy input. */
export const Z_LEN = SEQ_D_MODEL;
/** What a v4 first layer takes: planes ++ scalars ++ z. */
export const SEQ_INPUT_LEN = INPUT_LEN + Z_LEN;

/**
 * Float count of `attn.f32` — 23,616 (94,464 bytes), derived from the tensor
 * list rather than transcribed from the spec's prose. Any other file size is
 * REJECTED outright: the file has no header, so a wrong length means the
 * tensors would silently misalign.
 */
export const ATTN_FLOATS = SEQ_D_MODEL * SEQ_DENSE + SEQ_D_MODEL + // W_in, b_in
  4 * (SEQ_D_MODEL * SEQ_D_MODEL + SEQ_D_MODEL) + // Wq/Wk/Wv/Wo + biases
  SEQ_D_MODEL + // u (a vector, no bias)
  SEQ_D_MODEL * SEQ_D_MODEL + SEQ_D_MODEL; // Wz, bz

/** The river encoder's parameters, in the order `attn.f32` stores them. */
export interface Attn {
  wIn: Float32Array; // [64][42]
  bIn: Float32Array;
  wq: Float32Array; // [64][64]
  bq: Float32Array;
  wk: Float32Array;
  bk: Float32Array;
  wv: Float32Array;
  bv: Float32Array;
  wo: Float32Array;
  bo: Float32Array;
  u: Float32Array; // [64] — the learned pooling query
  wz: Float32Array;
  bz: Float32Array;
  /** Where the blob came from, for error messages. */
  path: string;
  /** Set by `loadNativeAttn` when the shim exports the v4 entry points. */
  native?: NativeAttnCtx;
}

/** A live `rlnet_attn` context. */
export interface NativeAttnCtx {
  handle: Deno.PointerValue;
  lib: RlnetAttnLib;
}

function dirOf(path: string): string {
  const i = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return i < 0 ? "." : path.slice(0, i);
}

function joinPath(dir: string, rel: string): string {
  if (rel.startsWith("/") || /^[A-Za-z]:[\\/]/.test(rel)) return rel;
  return `${dir}/${rel}`;
}

const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/** `count` little-endian float32 starting at `offset`. */
function readF32LE(bytes: Uint8Array, offset: number, count: number): Float32Array {
  if (LITTLE_ENDIAN) {
    // A copy, not a view: `Deno.readFileSync` gives no alignment guarantee, and
    // the slice keeps the net independent of the file buffer's lifetime.
    return new Float32Array(bytes.buffer.slice(
      bytes.byteOffset + offset,
      bytes.byteOffset + offset + count * 4,
    ));
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = dv.getFloat32(offset + i * 4, true);
  return out;
}

function fail(path: string, why: string): never {
  throw new Error(`重み ${path}: ${why}`);
}

/**
 * Names the feature version a (planes, scalars) pair belongs to, if known.
 *
 * v3 and v4 share (48, 42) — v4 appended a token stream, not a plane — so the
 * pair alone cannot tell them apart and `hasAttn` is what does.
 */
function versionLabel(
  planes: number | undefined,
  scalars: number | undefined,
  hasAttn = false,
): string {
  if (planes === 22 && scalars === 33) return "v1";
  if (planes === 36 && scalars === 39) return "v2";
  if (planes === FEATURES.planes && scalars === FEATURES.scalars) {
    return hasAttn ? `v${FEATURES.version}` : "v3";
  }
  return "不明な版";
}

function checkManifest(path: string, m: Manifest): void {
  if (m.version !== 1) fail(path, `version ${m.version} は未対応 (1 のみ)`);
  if (m.arch !== "mlp") fail(path, `arch "${m.arch}" は未対応 (mlp のみ)`);
  if (m.features?.planes !== FEATURES.planes || m.features?.scalars !== FEATURES.scalars) {
    const p = m.features?.planes, sc = m.features?.scalars;
    const dim = typeof p === "number" && typeof sc === "number" ? p * 34 + sc : NaN;
    fail(
      path,
      `特徴量が一致しません: この重みは特徴量 ${versionLabel(p, sc, m.attn !== undefined)} ` +
        `(${p}×34+${sc}=${dim}) 用、` +
        `本体は v${FEATURES.version} (${FEATURES.planes}×34+${FEATURES.scalars}=${INPUT_LEN}) — ` +
        `再学習するか train/randinit.py で重みを作り直してください`,
    );
  }
  if (m.actions !== FEATURES.actions) {
    fail(path, `行動数が一致しません: manifest ${m.actions} / 本体 ${FEATURES.actions}`);
  }
  if (!Array.isArray(m.layers) || m.layers.length === 0) fail(path, "layers が空です");
  if (typeof m.blob !== "string" || m.blob === "") fail(path, "blob のパスがありません");
  if (m.attn !== undefined && (typeof m.attn !== "string" || m.attn === "")) {
    fail(path, "attn のパスが不正です");
  }

  // The `attn` entry decides the first layer's width — see the file header.
  let expectIn = m.attn === undefined ? INPUT_LEN : SEQ_INPUT_LEN;
  m.layers.forEach((l, i) => {
    if (!Number.isInteger(l.in) || !Number.isInteger(l.out) || l.in <= 0 || l.out <= 0) {
      fail(path, `layer ${i} の in/out が不正です`);
    }
    if (l.in !== expectIn) fail(path, `layer ${i} の in=${l.in} は ${expectIn} であるべきです`);
    if (l.act !== "relu" && l.act !== "none") fail(path, `layer ${i} の act "${l.act}" は未対応`);
    expectIn = l.out;
  });
  const last = m.layers[m.layers.length - 1].out;
  if (last !== FEATURES.actions + 1) {
    fail(path, `最終層の out=${last} は ${FEATURES.actions + 1} (行動 + 価値) であるべきです`);
  }
}

export function loadNet(manifestPath: string): Net {
  let text: string;
  try {
    text = Deno.readTextFileSync(manifestPath);
  } catch (e) {
    throw new Error(
      `重み ${manifestPath} を読めません (${e instanceof Error ? e.message : String(e)})`,
    );
  }
  let manifest: Manifest;
  try {
    manifest = JSON.parse(text) as Manifest;
  } catch (e) {
    fail(manifestPath, `JSON として読めません (${e instanceof Error ? e.message : String(e)})`);
  }
  checkManifest(manifestPath, manifest);

  const blobPath = joinPath(dirOf(manifestPath), manifest.blob);
  let bytes: Uint8Array;
  try {
    bytes = Deno.readFileSync(blobPath);
  } catch (e) {
    throw new Error(
      `重み本体 ${blobPath} を読めません (${e instanceof Error ? e.message : String(e)})`,
    );
  }
  const need = manifest.layers.reduce((n, l) => n + l.out * l.in + l.out, 0) * 4;
  if (bytes.byteLength !== need) {
    fail(manifestPath, `${blobPath} の長さが ${bytes.byteLength} バイト (期待 ${need})`);
  }

  let off = 0;
  const layers: Layer[] = manifest.layers.map((l) => {
    const w = readF32LE(bytes, off, l.out * l.in);
    off += l.out * l.in * 4;
    const b = readF32LE(bytes, off, l.out);
    off += l.out * 4;
    return { ...l, w, b };
  });

  const net: Net = {
    manifest,
    path: manifestPath,
    layers,
    outputs: manifest.layers[manifest.layers.length - 1].out,
  };
  if (manifest.attn !== undefined) {
    net.attn = loadAttn(manifestPath, joinPath(dirOf(manifestPath), manifest.attn));
    loadNativeAttn(net.attn);
  }
  loadNative(net);
  return net;
}

/**
 * Reads `attn.f32` into the parameter set the encoder walks. The file is a bare
 * concatenation in the spec's order, so the only validation possible is its
 * length — which is exactly why the length check is strict and its message
 * spells the expected count out.
 */
export function loadAttn(manifestPath: string, blobPath: string): Attn {
  let bytes: Uint8Array;
  try {
    bytes = Deno.readFileSync(blobPath);
  } catch (e) {
    throw new Error(
      `注意機構の重み ${blobPath} を読めません (${e instanceof Error ? e.message : String(e)})`,
    );
  }
  if (bytes.byteLength !== ATTN_FLOATS * 4) {
    fail(
      manifestPath,
      `${blobPath} の長さが ${bytes.byteLength} バイト ` +
        `(期待 ${ATTN_FLOATS * 4} = float32 ${ATTN_FLOATS} 個)`,
    );
  }

  let off = 0;
  const take = (n: number): Float32Array => {
    const a = readF32LE(bytes, off, n);
    off += n * 4;
    return a;
  };
  const D = SEQ_D_MODEL;
  const mat = (): [Float32Array, Float32Array] => [take(D * D), take(D)];

  const wIn = take(D * SEQ_DENSE);
  const bIn = take(D);
  const [wq, bq] = mat();
  const [wk, bk] = mat();
  const [wv, bv] = mat();
  const [wo, bo] = mat();
  const u = take(D);
  const [wz, bz] = mat();
  return { wIn, bIn, wq, bq, wk, bk, wv, bv, wo, bo, u, wz, bz, path: blobPath };
}

// ---------------------------------------------------------------------------
// optional native inference (Accelerate via Deno FFI)
// ---------------------------------------------------------------------------
//
// `native/rlnet.c` is a generic MLP over cblas_sgemv. It is an OPTIONAL
// accelerator: with the dylib absent, or FFI not permitted, or the gate turned
// off, every net keeps running the pure-TS loop below and behaves identically.
//
// The dylib is looked for at ONE place: `native/librlnet.dylib` resolved
// relative to THIS MODULE (…/mjgame/src/rl/net.ts → …/mjgame/native/), so the
// working directory does not matter. Build it with `deno task build-native`.
//
// Gate — the `MJGAME_NATIVE` environment variable:
//   "0"    force the TypeScript path, even with the dylib built
//   "1"    require the native path; anything missing throws with the build line
//   unset  try native, fall back silently
// Reading it needs `--allow-env`; when that is denied the gate reads as unset.
//
// LIFETIME: a context is created per loaded `Net` and lives for the process
// (self-play and the TUI both run one process per session). `closeNet` exists
// for callers — tests, mostly — that churn through many nets. A context owns
// its scratch buffers and is NOT re-entrant: one caller at a time per net,
// which is exactly how a `NeuralPolicy` uses its own.

const NATIVE_SYMBOLS = {
  rlnet_create: { parameters: ["i32", "buffer", "buffer", "buffer"], result: "i64" },
  rlnet_forward: { parameters: ["i64", "buffer", "buffer"], result: "void" },
  rlnet_destroy: { parameters: ["i64"], result: "void" },
} as const;

export type RlnetLib = Deno.DynamicLibrary<typeof NATIVE_SYMBOLS>;

/**
 * The feature-v4 entry points, opened SEPARATELY from the MLP ones on purpose.
 *
 * `Deno.dlopen` resolves every symbol in one go and throws if any is missing,
 * so putting these three in `NATIVE_SYMBOLS` would make a dylib built before v4
 * landed fail to open AT ALL — the MLP would lose its accelerator over an
 * encoder it does not use. Two dlopen calls against the same file cost nothing
 * (dyld caches the image) and let the encoder degrade on its own.
 */
const ATTN_SYMBOLS = {
  rlnet_attn_create: { parameters: ["buffer"], result: "pointer" },
  rlnet_attn_encode: { parameters: ["pointer", "buffer", "i32", "buffer"], result: "void" },
  rlnet_attn_destroy: { parameters: ["pointer"], result: "void" },
} as const;

export type RlnetAttnLib = Deno.DynamicLibrary<typeof ATTN_SYMBOLS>;

const LIB_EXT = Deno.build.os === "windows"
  ? ".dll"
  : Deno.build.os === "darwin"
  ? ".dylib"
  : ".so";

/** The one place the shim is looked for, module-relative. */
export const NATIVE_LIB_URL = new URL(`../../native/librlnet${LIB_EXT}`, import.meta.url);

const BUILD_HINT = `mjgame/ で \`deno task build-native\` ` +
  `(clang -O3 -dynamiclib -framework Accelerate -o native/librlnet${LIB_EXT} native/rlnet.c) ` +
  `を実行し、--allow-ffi をつけて起動してください`;

type Gate = "off" | "require" | "auto";

/**
 * Whether a permission is already granted. This is ASKED, never taken: reading
 * the env var or calling `dlopen` without the flag would put an interactive
 * prompt in front of a self-play run that never wanted native inference in the
 * first place. `querySync` prompts for nothing.
 */
function granted(desc: Deno.PermissionDescriptor): boolean {
  try {
    return Deno.permissions.querySync(desc).state === "granted";
  } catch {
    return false;
  }
}

function gate(): Gate {
  // --allow-env withheld: the gate reads as unset, and native is merely tried.
  if (!granted({ name: "env", variable: "MJGAME_NATIVE" })) return "auto";
  const v = Deno.env.get("MJGAME_NATIVE");
  if (v === "0") return "off";
  if (v === "1") return "require";
  return "auto";
}

// dlopen once per process; `null` remembers a failure so every later net skips
// the retry (and, under gate "1", so the same reason is reported again).
let libCache: RlnetLib | null | undefined;
let libError = "";

function openLib(required: boolean): RlnetLib | null {
  if (libCache === undefined) {
    if (!granted({ name: "ffi", path: NATIVE_LIB_URL })) {
      libCache = null;
      libError = "--allow-ffi がありません";
    } else {
      try {
        libCache = Deno.dlopen(NATIVE_LIB_URL, NATIVE_SYMBOLS);
      } catch (e) {
        libCache = null;
        libError = e instanceof Error ? e.message : String(e);
      }
    }
  }
  if (libCache === null && required) {
    throw new Error(
      `MJGAME_NATIVE=1 ですが native 推論を読み込めません: ${libError} — ${BUILD_HINT}`,
    );
  }
  return libCache;
}

/** Layer weights+biases back into one `policy.f32`-shaped buffer. */
function packBlob(net: Net): Float32Array {
  let n = 0;
  for (const l of net.layers) n += l.w.length + l.b.length;
  const blob = new Float32Array(n);
  let off = 0;
  for (const l of net.layers) {
    blob.set(l.w, off);
    off += l.w.length;
    blob.set(l.b, off);
    off += l.b.length;
  }
  return blob;
}

/**
 * Hands `net` to the native shim, if the gate and the filesystem allow it.
 * Returns whether `forward` will now go through Accelerate. Idempotent; called
 * for you by `loadNet`, and safe to call on a hand-built `Net` too.
 */
export function loadNative(net: Net): boolean {
  if (net.native) return true;
  const g = gate();
  if (g === "off") return false;
  const lib = openLib(g === "require");
  if (!lib) return false;

  const dims = new Int32Array(net.layers.length + 1);
  const acts = new Uint8Array(net.layers.length);
  dims[0] = net.layers[0].in;
  net.layers.forEach((l, i) => {
    dims[i + 1] = l.out;
    acts[i] = l.act === "relu" ? 1 : 0;
  });

  const handle = lib.symbols.rlnet_create(net.layers.length, dims, acts, packBlob(net));
  if (handle === 0n) {
    if (g === "require") {
      throw new Error(`MJGAME_NATIVE=1 ですが rlnet_create が失敗しました (${net.path})`);
    }
    return false;
  }
  net.native = { handle, lib };
  return true;
}

/** Whether this net's `forward` currently runs natively. */
export function isNative(net: Net): boolean {
  return net.native !== undefined;
}

// ---------------------------------------------------------------------------
// optional native river encoder
// ---------------------------------------------------------------------------
//
// Same gate, ONE deliberate difference: a missing `rlnet_attn_*` symbol never
// throws, not even under MJGAME_NATIVE=1. The gate's contract is about whether
// the shim is used at all, and a dylib that predates feature v4 exports the MLP
// entry points perfectly well — refusing to run because a NEWER symbol is
// absent would turn "use the accelerator" into "rebuild or die". The encoder
// falls back to `attnForwardTS`, which is the reference implementation anyway.
// MJGAME_NATIVE=0 still turns it off, as it turns everything off.

let attnLibCache: RlnetAttnLib | null | undefined;

function openAttnLib(): RlnetAttnLib | null {
  if (attnLibCache === undefined) {
    if (!granted({ name: "ffi", path: NATIVE_LIB_URL })) {
      attnLibCache = null;
    } else {
      try {
        attnLibCache = Deno.dlopen(NATIVE_LIB_URL, ATTN_SYMBOLS);
      } catch {
        // No dylib, or one without the v4 entry points. Either way: TS encoder.
        attnLibCache = null;
      }
    }
  }
  return attnLibCache;
}

/**
 * Hands one `Attn` to the native encoder, if it is there. Returns whether
 * `attnEncode` will now go through Accelerate. Idempotent.
 *
 * The native side re-reads `attn.f32` from disk rather than being handed the
 * parsed arrays — the ABI takes a path — so the TS copy stays loaded as the
 * fallback and as the thing tests compare against.
 */
export function loadNativeAttn(attn: Attn): boolean {
  if (attn.native) return true;
  if (gate() === "off") return false;
  const lib = openAttnLib();
  if (!lib) return false;
  const path = new TextEncoder().encode(`${attn.path}\0`);
  let handle: Deno.PointerValue;
  try {
    handle = lib.symbols.rlnet_attn_create(path);
  } catch {
    return false;
  }
  if (handle === null) return false;
  attn.native = { handle, lib };
  return true;
}

/** Releases the native encoder context; the TS path keeps working. */
export function closeAttn(attn: Attn): void {
  if (!attn.native) return;
  attn.native.lib.symbols.rlnet_attn_destroy(attn.native.handle);
  attn.native = undefined;
}

// ---------------------------------------------------------------------------
// the river encoder's forward
// ---------------------------------------------------------------------------

/** `y[i] = W h[i] + b` for each of `L` rows; `w` is [D][D], row-major. */
function rowsAffine(w: Float32Array, b: Float32Array, h: Float32Array, L: number): Float32Array {
  const D = SEQ_D_MODEL;
  const y = new Float32Array(L * D);
  for (let i = 0; i < L; i++) {
    const hi = i * D;
    for (let o = 0; o < D; o++) {
      const base = o * D;
      let acc = b[o];
      for (let d = 0; d < D; d++) acc += w[base + d] * h[hi + d];
      y[i * D + o] = acc;
    }
  }
  return y;
}

/**
 * The reference implementation of the encoder, exactly as `train/V4_SPEC.md`
 * numbers it: relu input projection, one bidirectional multi-head attention
 * block (no causal mask — a discard from three turns ago may read one from
 * ten), a learned-query pooling into a single vector, and a final projection.
 *
 * The L = 0 case is the one every implementation has to agree on and the one
 * that actually happens: the very first decision of a hand has no discards at
 * all. p = 0, so z is `bz` and nothing else — NOT a zero vector, which would
 * make "no river yet" indistinguishable from a river the encoder happened to
 * read as neutral.
 */
function attnForwardTS(a: Attn, tokens: Int8Array): Float32Array {
  const D = SEQ_D_MODEL, H = SEQ_HEADS, HD = SEQ_HEAD_DIM;
  const L = tokens.length / SEQ_TOKEN_BYTES;
  if (L === 0) return a.bz.slice();

  // 1. h = relu(W_in x + b_in)
  const h = new Float32Array(L * D);
  const x = new Float32Array(SEQ_DENSE);
  for (let i = 0; i < L; i++) {
    expandToken(tokens, i, x);
    for (let o = 0; o < D; o++) {
      const base = o * SEQ_DENSE;
      let acc = a.bIn[o];
      for (let d = 0; d < SEQ_DENSE; d++) acc += a.wIn[base + d] * x[d];
      h[i * D + o] = acc > 0 ? acc : 0;
    }
  }

  // 2. q / k / v
  const q = rowsAffine(a.wq, a.bq, h, L);
  const k = rowsAffine(a.wk, a.bk, h, L);
  const v = rowsAffine(a.wv, a.bv, h, L);

  // 3. per-head softmax attention, heads concatenated back into one [L, D]
  const o = new Float32Array(L * D);
  const w = new Float32Array(L);
  for (let head = 0; head < H; head++) {
    const off = head * HD;
    for (let i = 0; i < L; i++) {
      let max = -Infinity;
      for (let j = 0; j < L; j++) {
        let s = 0;
        for (let d = 0; d < HD; d++) s += q[i * D + off + d] * k[j * D + off + d];
        s *= SEQ_SCALE;
        w[j] = s;
        if (s > max) max = s;
      }
      let total = 0;
      for (let j = 0; j < L; j++) {
        const e = Math.exp(w[j] - max);
        w[j] = e;
        total += e;
      }
      for (let j = 0; j < L; j++) w[j] /= total;
      for (let d = 0; d < HD; d++) {
        let acc = 0;
        for (let j = 0; j < L; j++) acc += w[j] * v[j * D + off + d];
        o[i * D + off + d] = acc;
      }
    }
  }

  // 4. m = Wo O + bo
  const m = rowsAffine(a.wo, a.bo, o, L);

  // 5. α = softmax(u · m); p = Σ α m
  const alpha = new Float32Array(L);
  let max = -Infinity;
  for (let i = 0; i < L; i++) {
    let s = 0;
    for (let d = 0; d < D; d++) s += a.u[d] * m[i * D + d];
    alpha[i] = s;
    if (s > max) max = s;
  }
  let total = 0;
  for (let i = 0; i < L; i++) {
    const e = Math.exp(alpha[i] - max);
    alpha[i] = e;
    total += e;
  }
  const p = new Float32Array(D);
  for (let i = 0; i < L; i++) {
    const c = alpha[i] / total;
    for (let d = 0; d < D; d++) p[d] += c * m[i * D + d];
  }

  // 6. z = Wz p + bz
  const z = new Float32Array(D);
  for (let oi = 0; oi < D; oi++) {
    const base = oi * D;
    let acc = a.bz[oi];
    for (let d = 0; d < D; d++) acc += a.wz[base + d] * p[d];
    z[oi] = acc;
  }
  return z;
}

/**
 * `z` for one packed token stream — native when the shim exports the v4 entry
 * points, `attnForwardTS` otherwise. Fresh `Z_LEN`-long buffer per call.
 */
export function attnEncode(attn: Attn, tokens: Int8Array): Float32Array {
  // Both out-of-range rules the spec fixes, applied HERE so the two paths agree
  // by construction: a partial trailing token is dropped, and anything past
  // token 96 is ignored (`encodeSeq` truncates to 96, so a longer stream is a
  // caller bug — but it must not be a bug the two implementations disagree on).
  const ntok = Math.min(Math.floor(tokens.length / SEQ_TOKEN_BYTES), SEQ_MAX);
  const packed = ntok * SEQ_TOKEN_BYTES === tokens.length
    ? tokens
    : tokens.subarray(0, ntok * SEQ_TOKEN_BYTES);
  if (attn.native) {
    const z = new Float32Array(Z_LEN);
    attn.native.lib.symbols.rlnet_attn_encode(attn.native.handle, packed, ntok, z);
    return z;
  }
  return attnForwardTS(attn, packed);
}

/** The width this net's first layer takes: 1674 (v3) or 1738 (v4). */
export function inputLen(net: Net): number {
  return net.layers[0].in;
}

/** Whether this net reads the v4 token stream. */
export function isSeqNet(net: Net): boolean {
  return net.attn !== undefined;
}

/**
 * The vector `forward` wants, built from the two halves the encoder produces.
 *
 * On a v3 net `flat` IS the input and `tokens` is ignored, which is what keeps
 * one call site working against both generations of snapshot. On a v4 net the
 * 64 encoder dims are appended at 1674..1737 — appended, never interleaved, so
 * the v3 columns of a widened first layer still line up with the v3 features.
 */
export function seqInput(net: Net, flat: Float32Array, tokens: Int8Array): Float32Array {
  if (!net.attn) return flat;
  const out = new Float32Array(INPUT_LEN + Z_LEN);
  out.set(flat.subarray(0, INPUT_LEN), 0);
  out.set(attnEncode(net.attn, tokens), INPUT_LEN);
  return out;
}

/**
 * Releases the native context, if any; the net keeps working on the TS path.
 * Optional — process exit frees everything — but tests that build many nets
 * should call it.
 */
export function closeNet(net: Net): void {
  if (net.attn) closeAttn(net.attn);
  if (!net.native) return;
  net.native.lib.symbols.rlnet_destroy(net.native.handle);
  net.native = undefined;
}

/**
 * One forward pass. `input` must be `inputLen(net)` long — 1674 for a v3 net,
 * 1738 for a v4 one, `seqInput` being what builds the latter. The result is
 * fresh.
 */
export function forward(net: Net, input: Float32Array): Float32Array {
  const want = inputLen(net);
  if (input.length !== want) {
    throw new Error(`入力長 ${input.length} は ${want} であるべきです`);
  }
  if (net.native) {
    // A fresh output buffer per call, as the contract promises — it doubles as
    // the FFI destination, so nothing is copied afterwards.
    const out = new Float32Array(net.outputs);
    net.native.lib.symbols.rlnet_forward(net.native.handle, input, out);
    return out;
  }
  let x = input;
  for (const l of net.layers) {
    const y = new Float32Array(l.out);
    for (let o = 0; o < l.out; o++) {
      const base = o * l.in;
      let acc = l.b[o];
      for (let i = 0; i < l.in; i++) acc += l.w[base + i] * x[i];
      y[o] = l.act === "relu" ? (acc > 0 ? acc : 0) : acc;
    }
    x = y;
  }
  return x as Float32Array;
}
