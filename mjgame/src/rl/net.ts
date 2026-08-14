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

  return {
    manifest,
    path: manifestPath,
    layers,
    outputs: manifest.layers[manifest.layers.length - 1].out,
  };
}

/** One forward pass. `input` must be `INPUT_LEN` long; the result is fresh. */
export function forward(net: Net, input: Float32Array): Float32Array {
  if (input.length !== INPUT_LEN) {
    throw new Error(`入力長 ${input.length} は ${INPUT_LEN} であるべきです`);
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
