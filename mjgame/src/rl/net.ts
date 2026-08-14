// Inference-only MLP: manifest + flat float32 blob → a forward pass.
//
// Training happens in Python/MLX; this side never learns, it only reads. The
// on-disk format is a FROZEN contract:
//
//   manifest.json
//     {"version":1,"arch":"mlp","features":{"planes":36,"scalars":39},
//      "actions":78,
//      "layers":[{"in":1263,"out":512,"act":"relu"},
//                {"in":512,"out":256,"act":"relu"},
//                {"in":256,"out":79,"act":"none"}],
//      "blob":"policy.f32"}
//
//   policy.f32 — for each layer in order: the weight matrix row-major
//   [out][in], then the bias [out]; all little-endian float32, concatenated
//   with no header and no padding. `blob` is relative to the manifest file.
//
// `version` is the FILE format's version, which is 1 and has not changed; the
// feature layout the weights were trained on is what `features` names, and
// `checkManifest` rejects anything that is not the encoder's current one.
//
// Input = planes (as floats) ++ scalars = 1263. Output = 79: elements 0..77 are
// the action logits and element 78 is the VALUE head, which inference ignores —
// it is carried in the same tensor purely so the trainer can share the trunk.

import { FEATURES, INPUT_LEN } from "./features.ts";

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

/** Names the feature version a (planes, scalars) pair belongs to, if known. */
function versionLabel(planes: number | undefined, scalars: number | undefined): string {
  if (planes === 22 && scalars === 33) return "v1";
  if (planes === 36 && scalars === 39) return "v2";
  if (planes === FEATURES.planes && scalars === FEATURES.scalars) return `v${FEATURES.version}`;
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
      `特徴量が一致しません: この重みは特徴量 ${versionLabel(p, sc)} (${p}×34+${sc}=${dim}) 用、` +
        `本体は v${FEATURES.version} (${FEATURES.planes}×34+${FEATURES.scalars}=${INPUT_LEN}) — ` +
        `再学習するか train/randinit.py で重みを作り直してください`,
    );
  }
  if (m.actions !== FEATURES.actions) {
    fail(path, `行動数が一致しません: manifest ${m.actions} / 本体 ${FEATURES.actions}`);
  }
  if (!Array.isArray(m.layers) || m.layers.length === 0) fail(path, "layers が空です");
  if (typeof m.blob !== "string" || m.blob === "") fail(path, "blob のパスがありません");

  let expectIn = INPUT_LEN;
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
  loadNative(net);
  return net;
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

/**
 * Releases the native context, if any; the net keeps working on the TS path.
 * Optional — process exit frees everything — but tests that build many nets
 * should call it.
 */
export function closeNet(net: Net): void {
  if (!net.native) return;
  net.native.lib.symbols.rlnet_destroy(net.native.handle);
  net.native = undefined;
}

/** One forward pass. `input` must be `INPUT_LEN` long; the result is fresh. */
export function forward(net: Net, input: Float32Array): Float32Array {
  if (input.length !== INPUT_LEN) {
    throw new Error(`入力長 ${input.length} は ${INPUT_LEN} であるべきです`);
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
