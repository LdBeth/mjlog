// The inference-only MLP loader: the on-disk contract, the arithmetic, and one
// full hanchan driven by four `NeuralPolicy` seats.
//
// Everything on disk here is written by hand, so the manifest/blob format is
// pinned from the OUTSIDE: if `net.ts` ever changes how it walks the blob, these
// files stop decoding to the numbers the tests expect.

import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { FEATURES, INPUT_LEN } from "../src/rl/features.ts";
import type { LayerSpec } from "../src/rl/net.ts";
import { forward, loadNet, VALUE_INDEX } from "../src/rl/net.ts";
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
}

/** Writes `manifest.json` + `policy.f32` into `dir`; returns the manifest path. */
function writeNet(dir: string, layers: LayerSpec[], params: Params[], o: WriteOpts = {}): string {
  const manifestPath = `${dir}/manifest.json`;
  Deno.writeTextFileSync(
    manifestPath,
    JSON.stringify({
      version: 1,
      arch: "mlp",
      features: { planes: FEATURES.planes, scalars: FEATURES.scalars },
      actions: FEATURES.actions,
      layers,
      blob: "policy.f32",
      ...o.patch,
    }),
  );
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
