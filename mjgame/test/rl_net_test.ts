// The inference-only MLP loader: the on-disk contract, the arithmetic, and one
// full hanchan driven by four `NeuralPolicy` seats.
//
// Everything on disk here is written by hand, so the manifest/blob format is
// pinned from the OUTSIDE: if `net.ts` ever changes how it walks the blob, these
// files stop decoding to the numbers the tests expect.

import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { FEATURES, INPUT_LEN, SEQ_DENSE, SEQ_MAX, SEQ_RIVER_MAX } from "../src/rl/features.ts";
import type { Attn, LayerSpec } from "../src/rl/net.ts";
import {
  ATTN_FLOATS,
  attnEncode,
  forward,
  inputLen,
  isSeqNet,
  loadAttn,
  loadNet,
  SEQ_D_MODEL,
  SEQ_HEAD_DIM,
  SEQ_HEADS,
  SEQ_INPUT_LEN,
  SEQ_SCALE,
  seqInput,
  VALUE_INDEX,
  Z_LEN,
} from "../src/rl/net.ts";
import { NeuralPolicy } from "../src/rl/policy.ts";
import { runMatchSync } from "../src/match.ts";
import { makeDojoHooks } from "../src/main.ts";
import { DOJO_HEADLESS, JANKI } from "../src/rules.ts";
import { scorer } from "../src/score.ts";
import type { Rng } from "../src/rng.ts";
import { sfc32 } from "../src/rng.ts";
import type { SyncPolicy } from "../src/policy.ts";
import { SEATS } from "../src/types.ts";

// ---------------------------------------------------------------------------
// on-disk helpers
// ---------------------------------------------------------------------------

interface Params {
  w: Float32Array; // [out][in], row-major
  b: Float32Array; // [out]
}

/** Weights then bias per layer, concatenated as little-endian float32. */
function packParams(params: Params[]): Uint8Array {
  const n = params.reduce((acc, p) => acc + p.w.length + p.b.length, 0);
  const out = new Uint8Array(n * 4);
  const dv = new DataView(out.buffer);
  let i = 0;
  for (const p of params) {
    for (const v of p.w) dv.setFloat32(i++ * 4, v, true);
    for (const v of p.b) dv.setFloat32(i++ * 4, v, true);
  }
  return out;
}

interface WriteOpts {
  /** Fields spliced over the otherwise-valid manifest, to break it on purpose. */
  patch?: Record<string, unknown>;
  /** Replaces the packed blob wholesale (for length-mismatch cases). */
  blob?: Uint8Array;
  /** Skip writing the blob file at all. */
  noBlob?: boolean;
  /** Feature v4: writes `attn.f32` and names it in the manifest. */
  attn?: Uint8Array;
  /** Name `attn.f32` in the manifest WITHOUT writing it. */
  attnMissing?: boolean;
}

/** Writes `manifest.json` + `policy.f32` into `dir`; returns the manifest path. */
function writeNet(dir: string, layers: LayerSpec[], params: Params[], o: WriteOpts = {}): string {
  const manifestPath = `${dir}/manifest.json`;
  const wantAttn = o.attn !== undefined || o.attnMissing === true;
  Deno.writeTextFileSync(
    manifestPath,
    JSON.stringify({
      version: 1,
      arch: "mlp",
      features: { planes: FEATURES.planes, scalars: FEATURES.scalars },
      actions: FEATURES.actions,
      layers,
      blob: "policy.f32",
      ...(wantAttn ? { attn: "attn.f32" } : {}),
      ...o.patch,
    }),
  );
  // The dir is reused across cases, so a case without an attn must remove one
  // an earlier case left behind.
  try {
    Deno.removeSync(`${dir}/attn.f32`);
  } catch {
    // never written
  }
  if (o.attn !== undefined) Deno.writeFileSync(`${dir}/attn.f32`, o.attn);
  if (o.noBlob) {
    // The dir is reused across cases, so "no blob" means removing an earlier one.
    try {
      Deno.removeSync(`${dir}/policy.f32`);
    } catch {
      // never written in the first place
    }
  } else {
    Deno.writeFileSync(`${dir}/policy.f32`, o.blob ?? packParams(params));
  }
  return manifestPath;
}

function tempDir(): string {
  return Deno.makeTempDirSync({ prefix: "mjgame_rl_net_" });
}

// ---------------------------------------------------------------------------
// a valid RANDOM net with the production architecture (reused below)
// ---------------------------------------------------------------------------

function randomF32(rng: Rng, n: number, scale: number): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (rng.float() * 2 - 1) * scale;
  return a;
}

/**
 * The architecture named in `net.ts`'s frozen contract — 1263→512→256→79 — with
 * deterministic pseudo-random weights (sfc32, never `Math.random`, so a failure
 * here reproduces exactly). Exported shape, not exported symbol: other tests in
 * this file call it, and nothing outside does.
 */
export function writeRandomNet(dir: string, seed: number): string {
  const layers: LayerSpec[] = [
    { in: INPUT_LEN, out: 512, act: "relu" },
    { in: 512, out: 256, act: "relu" },
    { in: 256, out: FEATURES.actions + 1, act: "none" },
  ];
  const rng = sfc32(seed);
  // 1/√fan-in keeps three layers of activations at O(1) instead of exploding.
  const params = layers.map((l) => ({
    w: randomF32(rng, l.out * l.in, 1 / Math.sqrt(l.in)),
    b: randomF32(rng, l.out, 0.01),
  }));
  return writeNet(dir, layers, params);
}

// ---------------------------------------------------------------------------
// 1. hand-computable forward passes
// ---------------------------------------------------------------------------

/** Quarters, so every input is exact in float32. Cycles with period 7. */
function testInput(): Float32Array {
  const x = new Float32Array(INPUT_LEN);
  for (let i = 0; i < INPUT_LEN; i++) x[i] = ((i % 7) - 3) / 4;
  return x;
}

Deno.test(`net: 単層 ${INPUT_LEN}→79 の forward が手計算と一致する`, () => {
  const dir = tempDir();
  try {
    const IN = INPUT_LEN, OUT = FEATURES.actions + 1;
    // w[o][i] = 1 when i === o, plus a fixed 2 on the very last input, so every
    // output reads one distinct input AND one shared one: out[o] = x[o] + 2·x[IN−1] + b[o].
    const w = new Float32Array(OUT * IN);
    const b = new Float32Array(OUT);
    for (let o = 0; o < OUT; o++) {
      w[o * IN + o] = 1;
      w[o * IN + (IN - 1)] = 2;
      b[o] = o / 8;
    }
    const layers: LayerSpec[] = [{ in: IN, out: OUT, act: "none" }];
    const net = loadNet(writeNet(dir, layers, [{ w, b }]));

    assertEquals(net.outputs, 79);
    assertEquals(net.layers.length, 1);
    assertEquals(VALUE_INDEX, 78);

    const x = testInput();
    const y = forward(net, x);
    assertEquals(y.length, 79);
    for (let o = 0; o < OUT; o++) {
      assertAlmostEquals(y[o], x[o] + 2 * x[IN - 1] + o / 8, 1e-6, `output ${o}`);
    }
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test(`net: 二層 ${INPUT_LEN}→4(relu)→79 で relu と連鎖が効く`, () => {
  const dir = tempDir();
  try {
    const IN = INPUT_LEN, H = 4, OUT = FEATURES.actions + 1;
    // Every whole period of 7 sums to 0, so only the tail survives — and a
    // partial period runs −3,−2,… , making the total ≤ 0 at any INPUT_LEN.
    // That sign is what makes h3's relu clamp below actually fire.
    const x = testInput();
    let sum = 0;
    for (const v of x) sum += v;
    assert(sum <= 0, `input sum ${sum} should be ≤ 0`);

    const w0 = new Float32Array(H * IN);
    const b0 = new Float32Array(H);
    for (let i = 0; i < IN; i++) {
      w0[0 * IN + i] = 1; // h0 = relu(sum + 4)     = sum + 4
      w0[1 * IN + i] = -1; // h1 = relu(−sum)        = −sum
      w0[3 * IN + i] = 1; // h3 = relu(sum)         = 0   ← the relu clamp
    }
    w0[2 * IN + 5] = 1; // h2 = relu(x[5] + 0.25) = 0.75
    b0[0] = 4;
    b0[2] = 0.25;

    const w1 = new Float32Array(OUT * H);
    const b1 = new Float32Array(OUT);
    for (let o = 0; o < OUT; o++) {
      for (let j = 0; j < H; j++) w1[o * H + j] = ((o + j) % 4 + 1) / 8;
      b1[o] = o / 64;
    }

    const layers: LayerSpec[] = [
      { in: IN, out: H, act: "relu" },
      { in: H, out: OUT, act: "none" },
    ];
    const net = loadNet(writeNet(dir, layers, [{ w: w0, b: b0 }, { w: w1, b: b1 }]));
    const y = forward(net, x);
    assertEquals(y.length, 79);

    const h = [sum + 4, -sum, 0.75, 0];
    for (let o = 0; o < OUT; o++) {
      let want = o / 64;
      for (let j = 0; j < H; j++) want += h[j] * (((o + j) % 4 + 1) / 8);
      assertAlmostEquals(y[o], want, 1e-6, `output ${o}`);
    }
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("net: 入力長が違えば forward は拒否する", () => {
  const dir = tempDir();
  try {
    const IN = INPUT_LEN, OUT = FEATURES.actions + 1;
    const net = loadNet(writeNet(dir, [{ in: IN, out: OUT, act: "none" }], [{
      w: new Float32Array(OUT * IN),
      b: new Float32Array(OUT),
    }]));
    assertThrows(() => forward(net, new Float32Array(INPUT_LEN - 1)), Error, "であるべきです");
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

// ---------------------------------------------------------------------------
// 2. rejection cases
// ---------------------------------------------------------------------------

Deno.test("net: 壊れた manifest / blob は理由つきで拒否される", () => {
  const dir = tempDir();
  try {
    const IN = INPUT_LEN, OUT = FEATURES.actions + 1;
    const good: LayerSpec[] = [{ in: IN, out: OUT, act: "none" }];
    const goodParams: Params[] = [{ w: new Float32Array(OUT * IN), b: new Float32Array(OUT) }];

    // manifest missing entirely
    assertThrows(
      () => loadNet(`${dir}/does-not-exist.json`),
      Error,
      "を読めません",
    );

    // not JSON
    Deno.writeTextFileSync(`${dir}/manifest.json`, "{ not json");
    assertThrows(() => loadNet(`${dir}/manifest.json`), Error, "JSON として読めません");

    // wrong version
    assertThrows(
      () => loadNet(writeNet(dir, good, goodParams, { patch: { version: 2 } })),
      Error,
      "version 2 は未対応",
    );

    // wrong arch
    assertThrows(
      () => loadNet(writeNet(dir, good, goodParams, { patch: { arch: "transformer" } })),
      Error,
      'arch "transformer" は未対応',
    );

    // features that do not match FEATURES
    assertThrows(
      () =>
        loadNet(writeNet(dir, good, goodParams, {
          patch: { features: { planes: FEATURES.planes - 1, scalars: FEATURES.scalars } },
        })),
      Error,
      "特徴量が一致しません",
    );
    // ...and a whole v1 weight set is named as such, not just as "mismatch".
    assertThrows(
      () =>
        loadNet(writeNet(dir, good, goodParams, {
          patch: { features: { planes: 22, scalars: 33 } },
        })),
      Error,
      "この重みは特徴量 v1 (22×34+33=781) 用",
    );
    assertThrows(
      () =>
        loadNet(writeNet(dir, good, goodParams, {
          patch: { features: { planes: 22, scalars: 33 } },
        })),
      Error,
      `本体は v${FEATURES.version} (${FEATURES.planes}×34+${FEATURES.scalars}=${INPUT_LEN})`,
    );
    assertThrows(
      () =>
        loadNet(writeNet(dir, good, goodParams, {
          patch: { features: { planes: FEATURES.planes, scalars: FEATURES.scalars + 1 } },
        })),
      Error,
      "特徴量が一致しません",
    );

    // action count mismatch
    assertThrows(
      () => loadNet(writeNet(dir, good, goodParams, { patch: { actions: FEATURES.actions - 1 } })),
      Error,
      "行動数が一致しません",
    );

    // no layers
    assertThrows(
      () => loadNet(writeNet(dir, [], [], { patch: { layers: [] } })),
      Error,
      "layers が空です",
    );

    // first layer does not take the 1263-wide input
    assertThrows(
      () => loadNet(writeNet(dir, [{ in: 780, out: OUT, act: "none" }], [])),
      Error,
      `layer 0 の in=780 は ${INPUT_LEN} であるべきです`,
    );

    // the chain breaks between layer 0 and layer 1
    assertThrows(
      () =>
        loadNet(writeNet(dir, [
          { in: IN, out: 64, act: "relu" },
          { in: 32, out: OUT, act: "none" },
        ], [])),
      Error,
      "layer 1 の in=32 は 64 であるべきです",
    );

    // the head is not `actions + 1` wide
    assertThrows(
      () => loadNet(writeNet(dir, [{ in: IN, out: FEATURES.actions, act: "none" }], [])),
      Error,
      `最終層の out=${FEATURES.actions} は ${FEATURES.actions + 1}`,
    );

    // unknown activation
    assertThrows(
      () =>
        loadNet(
          writeNet(dir, [{ in: IN, out: OUT, act: "gelu" as unknown as "relu" }], []),
        ),
      Error,
      '"gelu" は未対応',
    );

    // blob file missing
    assertThrows(
      () => loadNet(writeNet(dir, good, goodParams, { noBlob: true })),
      Error,
      "重み本体",
    );

    // blob one float short
    const need = (OUT * IN + OUT) * 4;
    assertThrows(
      () => loadNet(writeNet(dir, good, goodParams, { blob: new Uint8Array(need - 4) })),
      Error,
      `の長さが ${need - 4} バイト (期待 ${need})`,
    );
    // ...and one float long
    assertThrows(
      () => loadNet(writeNet(dir, good, goodParams, { blob: new Uint8Array(need + 4) })),
      Error,
      `(期待 ${need})`,
    );
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

// ---------------------------------------------------------------------------
// 3. four NeuralPolicy seats play a real hanchan
// ---------------------------------------------------------------------------

// ONE hanchan only: the 1263×512 matmul runs on every decision of every seat, so
// a second match buys no coverage and costs seconds. The contract's widths are
// deliberately NOT shrunk — the point is that production dimensions survive.
Deno.test("net: NeuralPolicy 4席で半荘を1回打ち切れる", () => {
  const dir = tempDir();
  try {
    const manifest = writeRandomNet(dir, 0x5eed_1234);
    const net = loadNet(manifest);
    assertEquals(net.layers.map((l) => [l.in, l.out, l.act]), [
      [INPUT_LEN, 512, "relu"],
      [512, 256, "relu"],
      [256, 79, "none"],
    ]);

    const policies: SyncPolicy[] = SEATS.map((s) =>
      new NeuralPolicy(`N${s}`, 4242 * 4 + s, manifest)
    );
    const result = runMatchSync(policies, {
      seed: 4242,
      cfg: JANKI,
      dojo: DOJO_HEADLESS,
      scorer,
      ...makeDojoHooks(DOJO_HEADLESS),
    });

    assertEquals(result.scores.length, 4);
    assert(result.rounds.length > 0, "no rounds were played");
    assert(result.rounds.length <= 64, `${result.rounds.length} rounds`);
    assertEquals(result.outcomes.length, result.rounds.length);
    for (const s of result.scores) assert(Number.isFinite(s), `score ${s} is not finite`);

    // Same conservation law as the self-play smoke test: every point that went
    // in is on the table at the end, the last round's leftover 供託 included —
    // `finalize` pays those sticks to the top finisher rather than dropping them.
    const sum = result.scores.reduce((a, b) => a + b, 0);
    assertEquals(sum, JANKI.startScore * 4);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("net: 同じ重みの NeuralPolicy は同じ半荘を再現する", () => {
  const dir = tempDir();
  try {
    const manifest = writeRandomNet(dir, 99);
    const play = () =>
      runMatchSync(
        SEATS.map((s) => new NeuralPolicy(`N${s}`, s, manifest)),
        { seed: 11, cfg: JANKI, dojo: DOJO_HEADLESS, scorer, ...makeDojoHooks(DOJO_HEADLESS) },
      );
    const a = play();
    const b = play();
    assertEquals(a.scores, b.scores);
    assertEquals(
      JSON.stringify(a.rounds.map((r) => r.events)),
      JSON.stringify(b.rounds.map((r) => r.events)),
    );
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

// ---------------------------------------------------------------------------
// 4. feature v4: the attention river encoder
// ---------------------------------------------------------------------------
//
// Same philosophy as the MLP half above: `attn.f32` is written here by hand, so
// the loader's walk through the blob is pinned from the OUTSIDE, and every
// forward is checked against arithmetic done in this file rather than against
// another copy of `net.ts`'s loop.

/** The tensors of `attn.f32`, in the order the spec stores them. */
const ATTN_ORDER: Array<[string, number]> = [
  ["wIn", 64 * 42],
  ["bIn", 64],
  ["wq", 64 * 64],
  ["bq", 64],
  ["wk", 64 * 64],
  ["bk", 64],
  ["wv", 64 * 64],
  ["bv", 64],
  ["wo", 64 * 64],
  ["bo", 64],
  ["u", 64],
  ["wz", 64 * 64],
  ["bz", 64],
];

/** Named tensors → the flat little-endian blob; anything omitted is zeros. */
function packAttn(parts: Record<string, Float32Array> = {}): Uint8Array {
  const out = new Uint8Array(ATTN_FLOATS * 4);
  const dv = new DataView(out.buffer);
  let i = 0;
  for (const [name, n] of ATTN_ORDER) {
    const a = parts[name] ?? new Float32Array(n);
    assertEquals(a.length, n, `tensor ${name} should be ${n} floats`);
    for (const v of a) dv.setFloat32(i++ * 4, v, true);
  }
  assertEquals(i, ATTN_FLOATS);
  return out;
}

/** The n×n identity, row-major. */
function eye(n: number): Float32Array {
  const a = new Float32Array(n * n);
  for (let i = 0; i < n; i++) a[i * n + i] = 1;
  return a;
}

/** W_in = [64][42] with a 1 on the diagonal ⇒ h = relu(x), zero-padded to 64. */
function embed(): Float32Array {
  const a = new Float32Array(64 * 42);
  for (let o = 0; o < 42; o++) a[o * 42 + o] = 1;
  return a;
}

/** A vector with a single 1. */
function unit(n: number, at: number): Float32Array {
  const a = new Float32Array(n);
  a[at] = 1;
  return a;
}

/** Packs `[type, seatRel, idx, flags]` tuples the way `encodeSeq` would. */
function pack(tuples: number[][]): Int8Array {
  const a = new Int8Array(tuples.length * 4);
  tuples.forEach((t, i) => a.set(t, i * 4));
  return a;
}

/**
 * The dense 42-vector of one token, written out INDEPENDENTLY of features.ts —
 * the point is to compare two derivations of the spec, not one function with
 * itself.
 */
function denseRef([type, seatRel, idx, flags]: number[]): number[] {
  const x = new Array(42).fill(0);
  x[type] = 1;
  x[34 + seatRel] = 1;
  x[38] = idx / 24;
  x[39] = flags & 1 ? 1 : 0;
  x[40] = flags & 2 ? 1 : 0;
  x[41] = flags & 4 ? 1 : 0;
  return x;
}

/** Loads an `attn.f32` written into a fresh temp dir; caller removes the dir. */
function attnFrom(dir: string, parts: Record<string, Float32Array>): Attn {
  const p = `${dir}/attn.f32`;
  Deno.writeFileSync(p, packAttn(parts));
  return loadAttn(`${dir}/manifest.json`, p);
}

Deno.test("attn: the frozen shapes and the file's float count", () => {
  assertEquals(SEQ_D_MODEL, 64);
  assertEquals(SEQ_HEADS, 4);
  assertEquals(SEQ_HEAD_DIM, 16);
  assertEquals(SEQ_SCALE, 1 / 4);
  assertEquals(Z_LEN, 64);
  assertEquals(SEQ_DENSE, 42);
  assertEquals(SEQ_INPUT_LEN, 1738);
  assertEquals(INPUT_LEN + Z_LEN, SEQ_INPUT_LEN);
  // 64*42+64 + 4*(64*64+64) + 64 + 64*64+64. The spec's prose briefly said
  // 23,872; the tensor list has always summed to this, and 94,464 bytes is the
  // ONLY size the loader accepts.
  assertEquals(ATTN_FLOATS, 23616);
  assertEquals(ATTN_FLOATS * 4, 94464);
  assertEquals(ATTN_ORDER.reduce((n, [, k]) => n + k, 0), ATTN_FLOATS);
});

Deno.test("attn: loadAttn slices the blob into the spec's tensors, in order", () => {
  const dir = tempDir();
  try {
    // Each tensor gets its own constant, so a mis-slice shows up as a value
    // from the neighbouring tensor rather than as a length error.
    const parts: Record<string, Float32Array> = {};
    ATTN_ORDER.forEach(([name, n], i) => {
      parts[name] = new Float32Array(n).fill(i + 1);
    });
    const a = attnFrom(dir, parts);
    const got: Record<string, Float32Array> = {
      wIn: a.wIn,
      bIn: a.bIn,
      wq: a.wq,
      bq: a.bq,
      wk: a.wk,
      bk: a.bk,
      wv: a.wv,
      bv: a.bv,
      wo: a.wo,
      bo: a.bo,
      u: a.u,
      wz: a.wz,
      bz: a.bz,
    };
    ATTN_ORDER.forEach(([name, n], i) => {
      assertEquals(got[name].length, n, `${name} length`);
      assertEquals(got[name][0], i + 1, `${name} head`);
      assertEquals(got[name][n - 1], i + 1, `${name} tail`);
    });
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("attn: L = 0 ⇒ z = bz exactly, whatever the rest of the weights are", () => {
  const dir = tempDir();
  try {
    const rng = sfc32(0xbeef);
    const bz = randomF32(rng, 64, 1);
    const a = attnFrom(dir, {
      wIn: randomF32(rng, 64 * 42, 1),
      bIn: randomF32(rng, 64, 1),
      wq: randomF32(rng, 64 * 64, 1),
      wk: randomF32(rng, 64 * 64, 1),
      wv: randomF32(rng, 64 * 64, 1),
      wo: randomF32(rng, 64 * 64, 1),
      u: randomF32(rng, 64, 1),
      wz: randomF32(rng, 64 * 64, 1),
      bz,
    });
    const z = attnEncode(a, new Int8Array(0));
    assertEquals(z.length, Z_LEN);
    // Not "close to bz" — bit for bit. p = 0 means no arithmetic happens at all.
    assertEquals(Array.from(z), Array.from(bz));
    // …and it is a FRESH buffer, not the stored bias handed out.
    z[0] = 12345;
    assert(a.bz[0] !== 12345, "encodeAttn returned the bias array itself");
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("attn: with q = k = 0 the attention is uniform and z is the token MEAN", () => {
  const dir = tempDir();
  try {
    // W_in embeds x into the first 42 of 64 dims (relu is a no-op: every dense
    // entry is ≥ 0), q and k stay 0 so every score is 0 and softmax is uniform,
    // and v/o/z are all the identity. So:
    //     O_i = mean_j x_j  (the same for every i)
    //     ⇒ m_i identical ⇒ α uniform ⇒ p = mean_j x_j
    //     ⇒ z = mean_j x_j + bz
    const bz = new Float32Array(64);
    for (let d = 0; d < 64; d++) bz[d] = (d + 1) / 100;
    const a = attnFrom(dir, {
      wIn: embed(),
      wv: eye(64),
      wo: eye(64),
      wz: eye(64),
      bz,
    });

    const tuples = [
      [0, 0, 0, 0],
      [33, 1, 5, 1],
      [17, 3, 23, 6],
      [0, 2, 1, 7],
      [9, 0, 12, 2],
    ];
    const z = attnEncode(a, pack(tuples));

    const want = new Array(64).fill(0);
    for (const t of tuples) {
      const x = denseRef(t);
      for (let d = 0; d < 42; d++) want[d] += x[d] / tuples.length;
    }
    for (let d = 0; d < 64; d++) {
      assertAlmostEquals(z[d], want[d] + bz[d], 1e-6, `z[${d}]`);
    }
    // Spot-checks with the arithmetic spelled out: type 0 appears twice of five,
    // relative seat 0 twice of five, and the idx dim is the mean of idx/24.
    assertAlmostEquals(z[0] - bz[0], 2 / 5, 1e-6, "onehot34(type 0)");
    assertAlmostEquals(z[34] - bz[34], 2 / 5, 1e-6, "onehot4(seatRel 0)");
    assertAlmostEquals(z[38] - bz[38], (0 + 5 + 23 + 1 + 12) / 24 / 5, 1e-6, "mean idx/24");
    assertAlmostEquals(z[41] - bz[41], 2 / 5, 1e-6, "called-away bit");
    // Dims 42..63 are outside the embedding, so they carry the bias alone.
    for (let d = 42; d < 64; d++) assertAlmostEquals(z[d], bz[d], 1e-7, `padding ${d}`);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("attn: heads are CONTIGUOUS 16-column slices — head 1 alone goes non-uniform", () => {
  const dir = tempDir();
  try {
    // bq = e16 puts the only non-zero query component at dim 16, which belongs
    // to head 1 (dims 16..31) under the contiguous-slice convention. k = x, so
    //     score(head 1)_ij = 0.25 · x_j[16] = 0.25 · [type_j == 16]
    // while heads 0, 2 and 3 see an all-zero query slice and stay uniform. If
    // the split were strided instead, dim 16 would land in head 0 and this test
    // would read the uniform mean where it expects the weighted one.
    const a = attnFrom(dir, {
      wIn: embed(),
      bq: unit(64, 16),
      wk: eye(64),
      wv: eye(64),
      wo: eye(64),
      wz: eye(64),
    });

    const tuples = [
      [16, 0, 0, 0], // type 16 ⇒ x[16] = 1
      [20, 0, 1, 0],
      [5, 1, 0, 1],
      [16, 2, 3, 7], // type 16 again
    ];
    const z = attnEncode(a, pack(tuples));

    // Head 1's weights, worked out by hand.
    const hot = Math.exp(SEQ_SCALE); // the two type-16 tokens
    const cold = Math.exp(0);
    const total = 2 * hot + 2 * cold;
    const wHot = hot / total, wCold = cold / total;
    assertAlmostEquals(2 * wHot + 2 * wCold, 1, 1e-9);

    // dims 16..31: the WEIGHTED mean.
    assertAlmostEquals(z[16], 2 * wHot, 1e-6, "types 16 carry the hot weight");
    assertAlmostEquals(z[20], wCold, 1e-6, "type 20 carries the cold weight");
    assert(Math.abs(z[16] - 2 / 4) > 1e-3, "head 1 came out uniform after all");
    for (let d = 17; d < 32; d++) {
      if (d === 20) continue;
      assertAlmostEquals(z[d], 0, 1e-6, `no token has type ${d}`);
    }

    // dims 0..15 (head 0) and 32..63 (heads 2/3): the plain mean, 1/4 each.
    assertAlmostEquals(z[5], 1 / 4, 1e-6, "head 0 stays uniform");
    assertAlmostEquals(z[34], 2 / 4, 1e-6, "head 2: two tokens at relative seat 0");
    assertAlmostEquals(z[38], (0 + 1 + 0 + 3) / 24 / 4, 1e-6, "head 2: mean idx/24");
    assertAlmostEquals(z[39], 2 / 4, 1e-6, "head 2: two ツモ切り");
    assertAlmostEquals(z[41], 1 / 4, 1e-6, "head 2: one called away");
    for (let d = 42; d < 64; d++) assertAlmostEquals(z[d], 0, 1e-7, `padding ${d}`);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("attn: the learned query u reweights the pooled rows", () => {
  const dir = tempDir();
  try {
    // Pooling can only be SEEN when the rows differ, and rows differ only when
    // the attention is per-row — so Wq and Wk are both the identity here:
    //     q_i = k_i = x_i  ⇒  score_ij = 0.25 · (x_i · x_j) per head slice.
    // With two tokens of DIFFERENT types (both < 16, so head 0 sees them):
    //     score_00 = score_11 = 0.25,  score_01 = score_10 = 0
    // giving row 0 the weights (A, B) and row 1 the mirror (B, A):
    const hot = Math.exp(SEQ_SCALE), cold = 1;
    const A = hot / (hot + cold), B = cold / (hot + cold);
    assertAlmostEquals(A + B, 1, 1e-12);
    assert(A > B, "the self-match must be the heavier one");
    //     m_0[0] = A, m_0[1] = B ;  m_1[0] = B, m_1[1] = A
    // so the pooled p[0] = α_0·A + α_1·B — a straight readout of α.
    const tuples = [[0, 0, 0, 0], [1, 0, 0, 0]];
    const weights = { wIn: embed(), wq: eye(64), wk: eye(64), wv: eye(64), wo: eye(64) };

    // u = 0 ⇒ α uniform ⇒ p[0] = p[1] = (A + B)/2 = 1/2, exactly.
    const flat = attnEncode(attnFrom(dir, { ...weights, wz: eye(64) }), pack(tuples));
    assertAlmostEquals(flat[0], 0.5, 1e-6, "uniform pooling");
    assertAlmostEquals(flat[1], 0.5, 1e-6, "uniform pooling");

    // u = e0 ⇒ the scores are m_0[0] = A and m_1[0] = B, so row 0 wins.
    const picked = attnEncode(
      attnFrom(dir, { ...weights, u: unit(64, 0), wz: eye(64) }),
      pack(tuples),
    );
    const a0 = Math.exp(A) / (Math.exp(A) + Math.exp(B));
    const a1 = 1 - a0;
    assert(a0 > a1, "u = e0 should favour row 0");
    assertAlmostEquals(picked[0], a0 * A + a1 * B, 1e-6, "p[0] = α·m[0]");
    assertAlmostEquals(picked[1], a0 * B + a1 * A, 1e-6, "p[1] = α·m[1]");
    assert(picked[0] > 0.5 && picked[1] < 0.5, "α did not move the pooled vector");

    // …and −u mirrors it, which no uniform pooling could do.
    const flipped = attnEncode(
      attnFrom(dir, { ...weights, u: new Float32Array(unit(64, 0)).map((v) => -v), wz: eye(64) }),
      pack(tuples),
    );
    assertAlmostEquals(flipped[0], picked[1], 1e-6);
    assertAlmostEquals(flipped[1], picked[0], 1e-6);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("attn: a single token attends to itself and z is just that token", () => {
  const dir = tempDir();
  try {
    const a = attnFrom(dir, { wIn: embed(), wv: eye(64), wo: eye(64), wz: eye(64) });
    const t = [7, 2, 23, 5];
    const z = attnEncode(a, pack([t]));
    const x = denseRef(t);
    for (let d = 0; d < 42; d++) assertAlmostEquals(z[d], x[d], 1e-6, `z[${d}]`);
    for (let d = 42; d < 64; d++) assertAlmostEquals(z[d], 0, 1e-7, `z[${d}]`);
    assertAlmostEquals(z[38], 23 / SEQ_RIVER_MAX, 1e-6, "the last live river slot");
    assertEquals([z[39], z[40], z[41]].map((v) => Math.round(v)), [1, 0, 1], "flags 5 = bit0|bit2");
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("attn: relu actually clamps in the input projection", () => {
  const dir = tempDir();
  try {
    // Row 0 of W_in reads dense dim 0 with a −1: for a type-0 token the
    // pre-activation is −1, so h[0] must come out 0, not −1.
    const wIn = embed();
    wIn[0 * 42 + 0] = -1;
    const a = attnFrom(dir, { wIn, wv: eye(64), wo: eye(64), wz: eye(64) });
    const z = attnEncode(a, pack([[0, 0, 0, 0]]));
    assertEquals(z[0], 0, "relu did not clamp");
    assertAlmostEquals(z[34], 1, 1e-6, "the rest of the token is untouched");
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("attn: a stream longer than 96 tokens is clamped, not read past", () => {
  const dir = tempDir();
  try {
    const a = attnFrom(dir, { wIn: embed(), wv: eye(64), wo: eye(64), wz: eye(64) });
    // 96 tokens of type 0, then 4 of type 1 — the tail must not reach z.
    const tuples: number[][] = [];
    for (let i = 0; i < SEQ_MAX; i++) tuples.push([0, i % 4, i % 24, 0]);
    for (let i = 0; i < 4; i++) tuples.push([1, 0, 0, 0]);
    const z = attnEncode(a, pack(tuples));
    assertAlmostEquals(z[0], 1, 1e-6, "every counted token is type 0");
    assertAlmostEquals(z[1], 0, 1e-6, "the tokens past 96 were read");
    // A trailing partial token is dropped rather than read as garbage.
    const ragged = new Int8Array(4 * 3 + 2);
    ragged.set(pack([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]));
    ragged[12] = 1;
    ragged[13] = 1;
    const zr = attnEncode(a, ragged);
    assertAlmostEquals(zr[0], 1, 1e-6, "the ragged tail changed the answer");
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

// ---------------------------------------------------------------------------
// 5. the v4 network: z concatenated at 1674..1737
// ---------------------------------------------------------------------------

/** A production-shaped v4 net: 1738→512→256→79 plus a random `attn.f32`. */
function writeRandomNetV4(dir: string, seed: number): string {
  const layers: LayerSpec[] = [
    { in: SEQ_INPUT_LEN, out: 512, act: "relu" },
    { in: 512, out: 256, act: "relu" },
    { in: 256, out: FEATURES.actions + 1, act: "none" },
  ];
  const rng = sfc32(seed);
  const params = layers.map((l) => ({
    w: randomF32(rng, l.out * l.in, 1 / Math.sqrt(l.in)),
    b: randomF32(rng, l.out, 0.01),
  }));
  const parts: Record<string, Float32Array> = {};
  // 0.02 is the trainer's own init scale for this file (train/widen4.py).
  for (const [name, n] of ATTN_ORDER) parts[name] = randomF32(rng, n, 0.02);
  return writeNet(dir, layers, params, { attn: packAttn(parts) });
}

Deno.test("net: an attn entry makes it a v4 net — 1738 wide, z at 1674..1737", () => {
  const dir = tempDir();
  try {
    const manifest = writeRandomNetV4(dir, 0x4444);
    const net = loadNet(manifest);
    assert(isSeqNet(net), "the attn entry did not make a seq net");
    assertEquals(inputLen(net), SEQ_INPUT_LEN);
    assertEquals(inputLen(net), 1738);
    assertEquals(net.layers[0].in, 1738);
    assert(net.attn !== undefined);

    const flat = testInput();
    const tokens = pack([[3, 0, 0, 0], [9, 2, 4, 3]]);
    const x = seqInput(net, flat, tokens);
    assertEquals(x.length, SEQ_INPUT_LEN);
    // The v3 half is copied through UNTOUCHED — this is the offset contract the
    // trainer's widened first layer depends on.
    for (let i = 0; i < INPUT_LEN; i++) assertEquals(x[i], flat[i], `plane/scalar cell ${i}`);
    const z = attnEncode(net.attn!, tokens);
    for (let d = 0; d < Z_LEN; d++) assertEquals(x[INPUT_LEN + d], z[d], `z dim ${d}`);

    // …and the 1738-wide vector is what forward accepts, the 1674-wide one not.
    const y = forward(net, x);
    assertEquals(y.length, 79);
    for (const v of y) assert(Number.isFinite(v), `logit ${v}`);
    assertThrows(() => forward(net, flat), Error, "1738 であるべきです");
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("net: on a v4 net the seq REACHES the logits", () => {
  const dir = tempDir();
  try {
    // One layer, 1738→79: row o reads plane/scalar cell o AND z dim o (o < 64),
    // so a changed river must move the first 64 logits and nothing else.
    const IN = SEQ_INPUT_LEN, OUT = FEATURES.actions + 1;
    const w = new Float32Array(OUT * IN);
    const b = new Float32Array(OUT);
    for (let o = 0; o < OUT; o++) {
      w[o * IN + o] = 1;
      if (o < Z_LEN) w[o * IN + INPUT_LEN + o] = 1;
    }
    const parts = { wIn: embed(), wv: eye(64), wo: eye(64), wz: eye(64) };
    const manifest = writeNet(dir, [{ in: IN, out: OUT, act: "none" }], [{ w, b }], {
      attn: packAttn(parts),
    });
    const net = loadNet(manifest);

    const flat = testInput();
    const empty = forward(net, seqInput(net, flat, new Int8Array(0)));
    // bz = 0 here, so an empty river contributes exactly nothing.
    for (let o = 0; o < OUT; o++) assertAlmostEquals(empty[o], flat[o], 1e-6, `logit ${o}`);

    // One token of type 3 at relative seat 1, idx 12, tsumogiri: z is that
    // token's dense vector, so logit 3 gains 1 and logit 35 gains 1.
    const one = forward(net, seqInput(net, flat, pack([[3, 1, 12, 1]])));
    assertAlmostEquals(one[3], flat[3] + 1, 1e-6, "onehot34(type 3)");
    assertAlmostEquals(one[35], flat[35] + 1, 1e-6, "onehot4(seatRel 1)");
    assertAlmostEquals(one[38], flat[38] + 12 / 24, 1e-6, "idx/24");
    assertAlmostEquals(one[39], flat[39] + 1, 1e-6, "tsumogiri");
    assertAlmostEquals(one[40], flat[40], 1e-6, "no riichi declaration");
    for (let o = Z_LEN; o < OUT; o++) {
      assertAlmostEquals(one[o], flat[o], 1e-6, `logit ${o} has no z column`);
    }
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("net: v4 manifests are rejected for the same reasons v3 ones are", () => {
  const dir = tempDir();
  try {
    const OUT = FEATURES.actions + 1;
    const good: Params[] = [{
      w: new Float32Array(OUT * SEQ_INPUT_LEN),
      b: new Float32Array(OUT),
    }];
    const layers: LayerSpec[] = [{ in: SEQ_INPUT_LEN, out: OUT, act: "none" }];

    // an attn entry with a v3-wide first layer
    assertThrows(
      () =>
        loadNet(writeNet(dir, [{ in: INPUT_LEN, out: OUT, act: "none" }], [{
          w: new Float32Array(OUT * INPUT_LEN),
          b: new Float32Array(OUT),
        }], { attn: packAttn() })),
      Error,
      `layer 0 の in=${INPUT_LEN} は ${SEQ_INPUT_LEN} であるべきです`,
    );
    // …and the mirror image: a 1738-wide first layer with no attn entry.
    assertThrows(
      () => loadNet(writeNet(dir, layers, good)),
      Error,
      `layer 0 の in=${SEQ_INPUT_LEN} は ${INPUT_LEN} であるべきです`,
    );

    // attn named but not on disk
    assertThrows(
      () => loadNet(writeNet(dir, layers, good, { attnMissing: true })),
      Error,
      "注意機構の重み",
    );

    // attn one float short, and one long — the file has no header, so a wrong
    // length would silently misalign every tensor after the first.
    const short = packAttn().subarray(0, ATTN_FLOATS * 4 - 4);
    assertThrows(
      () => loadNet(writeNet(dir, layers, good, { attn: short })),
      Error,
      `の長さが ${ATTN_FLOATS * 4 - 4} バイト (期待 ${ATTN_FLOATS * 4}`,
    );
    const long = new Uint8Array(ATTN_FLOATS * 4 + 4);
    long.set(packAttn());
    assertThrows(
      () => loadNet(writeNet(dir, layers, good, { attn: long })),
      Error,
      `(期待 ${ATTN_FLOATS * 4} = float32 ${ATTN_FLOATS} 個)`,
    );

    // an empty attn path is not "no attn", it is a broken manifest
    assertThrows(
      () => loadNet(writeNet(dir, layers, good, { attn: packAttn(), patch: { attn: "" } })),
      Error,
      "attn のパスが不正です",
    );
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

// ---------------------------------------------------------------------------
// 6. v3 snapshots keep working
// ---------------------------------------------------------------------------

Deno.test("net: a v3 manifest still loads, still plays, and grows no seq path", () => {
  const dir = tempDir();
  try {
    // The exact file `writeRandomNet` has always produced — no attn entry, a
    // 1674-wide first layer. Play and bench against old snapshots must survive
    // the encoder landing, so this is a REGRESSION guard, not a compatibility
    // nicety.
    const manifest = writeRandomNet(dir, 0x3333);
    const net = loadNet(manifest);
    assertEquals(net.manifest.attn, undefined);
    assertEquals(net.attn, undefined);
    assertEquals(isSeqNet(net), false);
    assertEquals(inputLen(net), INPUT_LEN);
    assertEquals(inputLen(net), 1674);

    const flat = testInput();
    const y = forward(net, flat);
    assertEquals(y.length, 79);
    // `seqInput` is a no-op on a v3 net — same array, not merely equal values,
    // so the shared call site costs an old net nothing.
    assertEquals(seqInput(net, flat, pack([[1, 1, 1, 1]])), flat);
    assert(seqInput(net, flat, new Int8Array(0)) === flat);
    assertThrows(() => forward(net, new Float32Array(SEQ_INPUT_LEN)), Error, "1674 であるべきです");

    // …and four v3 seats still finish a hanchan through NeuralPolicy.
    const result = runMatchSync(
      SEATS.map((s) => new NeuralPolicy(`N${s}`, 7 * 4 + s, net)),
      { seed: 7, cfg: JANKI, dojo: DOJO_HEADLESS, scorer, ...makeDojoHooks(DOJO_HEADLESS) },
    );
    assertEquals(result.scores.reduce((a, b) => a + b, 0), JANKI.startScore * 4);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});

Deno.test("net: NeuralPolicy 4席が v4 の重みで半荘を1回打ち切れる", () => {
  const dir = tempDir();
  try {
    const manifest = writeRandomNetV4(dir, 0x5eed_4444);
    const net = loadNet(manifest);
    assertEquals(net.layers.map((l) => [l.in, l.out, l.act]), [
      [SEQ_INPUT_LEN, 512, "relu"],
      [512, 256, "relu"],
      [256, 79, "none"],
    ]);

    const policies: SyncPolicy[] = SEATS.map((s) => new NeuralPolicy(`N${s}`, 909 * 4 + s, net));
    const result = runMatchSync(policies, {
      seed: 909,
      cfg: JANKI,
      dojo: DOJO_HEADLESS,
      scorer,
      ...makeDojoHooks(DOJO_HEADLESS),
    });
    assert(result.rounds.length > 0, "no rounds were played");
    assertEquals(result.scores.reduce((a, b) => a + b, 0), JANKI.startScore * 4);
  } finally {
    Deno.removeSync(dir, { recursive: true });
  }
});
