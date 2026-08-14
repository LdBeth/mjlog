// The optional Accelerate shim: same numbers as the TypeScript loop, and a gate
// that can always send you back to it.
//
// Everything here is skipped — with a printed reason, never silently — when the
// dylib cannot be produced (no clang, no `--allow-run`, no `--allow-ffi`),
// because native inference is an accelerator and never a requirement.
//
// Shapes are read from `FEATURES`/`INPUT_LEN`, never hardcoded: the feature
// version moves and this file must follow it.

import { assert, assertEquals } from "@std/assert";
import { FEATURES, INPUT_LEN } from "../src/rl/features.ts";
import type { LayerSpec, Net } from "../src/rl/net.ts";
import { closeNet, forward, isNative, loadNative, loadNet, NATIVE_LIB_URL } from "../src/rl/net.ts";
import type { Rng } from "../src/rng.ts";
import { sfc32 } from "../src/rng.ts";

// ---------------------------------------------------------------------------
// build the dylib if it is not there yet
// ---------------------------------------------------------------------------

/** Empty when native is testable here, otherwise why it is not. */
function ensureDylib(): string {
  try {
    Deno.statSync(NATIVE_LIB_URL);
    return "";
  } catch {
    // not built yet — fall through and build it
  }
  const src = new URL("rlnet.c", NATIVE_LIB_URL);
  const args = [
    "-O3",
    "-Wall",
    "-Wextra",
    "-dynamiclib",
    "-framework",
    "Accelerate",
    "-o",
    NATIVE_LIB_URL.pathname,
    src.pathname,
  ];
  let out: Deno.CommandOutput;
  try {
    out = new Deno.Command("clang", { args, stderr: "piped", stdout: "piped" }).outputSync();
  } catch (e) {
    return `clang を実行できません (${e instanceof Error ? e.message : String(e)})`;
  }
  if (!out.success) {
    return `clang が失敗しました: ${new TextDecoder().decode(out.stderr).trim()}`;
  }
  return "";
}

const SKIP_REASON = ensureDylib();
if (SKIP_REASON) console.log(`native テストを飛ばします: ${SKIP_REASON}`);
const SKIP = SKIP_REASON !== "";

// ---------------------------------------------------------------------------
// in-memory nets
// ---------------------------------------------------------------------------

function randomF32(rng: Rng, n: number, scale: number): Float32Array {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = (rng.float() * 2 - 1) * scale;
  return a;
}

/**
 * A `Net` built in memory — no manifest.json, no blob file. `loadNative` only
 * reads `layers`, so this is enough to compare the two forward paths, and it
 * keeps the shapes free of the on-disk contract's constraints (the last layer
 * need not be the policy head here).
 */
function makeNet(specs: LayerSpec[], seed: number): Net {
  const rng = sfc32(seed);
  const layers = specs.map((l) => ({
    ...l,
    // 1/√fan-in: three layers of this stay O(1) instead of exploding, so an
    // absolute tolerance below is a meaningful one.
    w: randomF32(rng, l.out * l.in, 1 / Math.sqrt(l.in)),
    b: randomF32(rng, l.out, 0.1),
  }));
  return {
    manifest: {
      version: 1,
      arch: "mlp",
      features: { planes: FEATURES.planes, scalars: FEATURES.scalars },
      actions: FEATURES.actions,
      layers: specs,
      blob: "policy.f32",
    },
    path: "<memory>",
    layers,
    outputs: specs[specs.length - 1].out,
  };
}

/** The production architecture, whatever `FEATURES` currently says it is. */
function realShape(): LayerSpec[] {
  return [
    { in: INPUT_LEN, out: 512, act: "relu" },
    { in: 512, out: 256, act: "relu" },
    { in: 256, out: FEATURES.actions + 1, act: "none" },
  ];
}

function randomInput(rng: Rng): Float32Array {
  return randomF32(rng, INPUT_LEN, 1);
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  assertEquals(a.length, b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

/** Runs `body` with `MJGAME_NATIVE` set (or deleted), then puts it back. */
function withGate(value: string | undefined, body: () => void): void {
  const had = Deno.env.get("MJGAME_NATIVE");
  if (value === undefined) Deno.env.delete("MJGAME_NATIVE");
  else Deno.env.set("MJGAME_NATIVE", value);
  try {
    body();
  } finally {
    if (had === undefined) Deno.env.delete("MJGAME_NATIVE");
    else Deno.env.set("MJGAME_NATIVE", had);
  }
}

// ---------------------------------------------------------------------------
// 1. equivalence
// ---------------------------------------------------------------------------

// sgemv reorders the summation, so this is an agreement test, not a bit-for-bit
// one; 1e-4 is far below any margin an argmax over logits cares about.
const TOL = 1e-4;

Deno.test({
  name: "native: 各形状で native と TS の forward が一致する",
  ignore: SKIP,
  fn: () => {
    const cases: { label: string; specs: LayerSpec[] }[] = [
      // one layer: the shortest path through the shim (input → caller's buffer)
      { label: "1層", specs: [{ in: INPUT_LEN, out: 13, act: "none" }] },
      {
        label: "2層",
        specs: [
          { in: INPUT_LEN, out: 7, act: "relu" },
          { in: 7, out: 5, act: "none" },
        ],
      },
      {
        // a relu on the LAST layer too — the shim must clamp the caller's buffer
        label: "2層(末尾relu)",
        specs: [
          { in: INPUT_LEN, out: 64, act: "relu" },
          { in: 64, out: 33, act: "relu" },
        ],
      },
      {
        label: "3層(細)",
        specs: [
          { in: INPUT_LEN, out: 32, act: "relu" },
          { in: 32, out: 16, act: "relu" },
          { in: 16, out: FEATURES.actions + 1, act: "none" },
        ],
      },
      { label: `3層(実寸 ${INPUT_LEN}→512→256→${FEATURES.actions + 1})`, specs: realShape() },
    ];

    withGate("1", () => {
      for (const [ci, c] of cases.entries()) {
        const ts = makeNet(c.specs, 0x51de_0000 + ci);
        const nat = makeNet(c.specs, 0x51de_0000 + ci); // identical weights
        assert(loadNative(nat), `${c.label}: native を読み込めませんでした`);
        assert(!isNative(ts), `${c.label}: TS 側に ctx が付いています`);

        try {
          const rng = sfc32(7000 + ci);
          let worst = 0;
          for (let k = 0; k < 100; k++) {
            const x = randomInput(rng);
            const a = forward(ts, x);
            const b = forward(nat, x);
            assertEquals(b.length, c.specs[c.specs.length - 1].out);
            worst = Math.max(worst, maxAbsDiff(a, b));
          }
          assert(worst < TOL, `${c.label}: 最大差 ${worst} が許容 ${TOL} を超えました`);
        } finally {
          closeNet(nat);
        }
      }
    });
  },
});

Deno.test({
  name: "native: closeNet の後は同じ net が TS 経路で同じ値を返す",
  ignore: SKIP,
  fn: () => {
    withGate("1", () => {
      const net = makeNet(realShape(), 0xc105_e1);
      assert(loadNative(net));
      const x = randomInput(sfc32(4));
      const withNative = forward(net, x);
      closeNet(net);
      assert(!isNative(net), "closeNet の後も ctx が残っています");
      const withTs = forward(net, x);
      assert(maxAbsDiff(withNative, withTs) < TOL);
      // Idempotent: a second close is a no-op, not a double free.
      closeNet(net);
    });
  },
});

// ---------------------------------------------------------------------------
// 2. the gate
// ---------------------------------------------------------------------------

Deno.test({
  name: "native: MJGAME_NATIVE=0 は dylib があっても TS 経路のまま",
  ignore: SKIP,
  fn: () => {
    withGate("0", () => {
      const net = makeNet(realShape(), 11);
      assertEquals(loadNative(net), false);
      assertEquals(isNative(net), false);
      // ...and it still computes: the fallback is the whole point.
      const y = forward(net, randomInput(sfc32(5)));
      assertEquals(y.length, FEATURES.actions + 1);
    });
  },
});

Deno.test({
  name: "native: 既定 (未設定) と =1 はどちらも native を掴む",
  ignore: SKIP,
  fn: () => {
    for (const gate of [undefined, "1"]) {
      withGate(gate, () => {
        const net = makeNet([{ in: INPUT_LEN, out: 9, act: "none" }], 12);
        assertEquals(loadNative(net), true, `gate=${gate}`);
        // Idempotent: a second call reuses the context rather than leaking one.
        assertEquals(loadNative(net), true);
        closeNet(net);
      });
    }
  },
});

// ---------------------------------------------------------------------------
// 3. through loadNet, from disk
// ---------------------------------------------------------------------------

/** Writes manifest.json + policy.f32 for `specs` into `dir`. */
function writeNet(dir: string, specs: LayerSpec[], seed: number): string {
  const net = makeNet(specs, seed);
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
  Deno.writeFileSync(`${dir}/policy.f32`, new Uint8Array(blob.buffer));
  const path = `${dir}/manifest.json`;
  Deno.writeTextFileSync(path, JSON.stringify(net.manifest));
  return path;
}

Deno.test({
  name: "native: loadNet が読んだ重みは gate 次第で native / TS になる",
  ignore: SKIP,
  fn: () => {
    const dir = Deno.makeTempDirSync({ prefix: "mjgame_rl_native_" });
    try {
      const path = writeNet(dir, realShape(), 0xbeef);
      const x = randomInput(sfc32(6));

      let nativeOut: Float32Array | undefined;
      withGate("1", () => {
        const net = loadNet(path);
        assert(isNative(net), "MJGAME_NATIVE=1 で native が付いていません");
        nativeOut = forward(net, x);
        closeNet(net);
      });

      withGate("0", () => {
        const net = loadNet(path);
        assertEquals(isNative(net), false);
        assert(maxAbsDiff(nativeOut!, forward(net, x)) < TOL, "経路で結果が食い違いました");
      });
    } finally {
      Deno.removeSync(dir, { recursive: true });
    }
  },
});

// ---------------------------------------------------------------------------
// 4. micro-bench (logged, never asserted)
// ---------------------------------------------------------------------------

Deno.test({
  name: "native: 実寸ネットの µs/forward を記録する",
  ignore: SKIP,
  fn: () => {
    withGate("1", () => {
      const specs = realShape();
      const ts = makeNet(specs, 1);
      const nat = makeNet(specs, 1);
      assert(loadNative(nat));
      try {
        const rng = sfc32(8);
        const inputs = Array.from({ length: 32 }, () => randomInput(rng));
        const N = 200;
        const time = (net: Net) => {
          let sink = 0;
          for (let i = 0; i < 20; i++) sink += forward(net, inputs[i % inputs.length])[0]; // warm-up
          const t0 = performance.now();
          for (let i = 0; i < N; i++) sink += forward(net, inputs[i % inputs.length])[0];
          const us = (performance.now() - t0) * 1000 / N;
          return { us, sink };
        };
        const a = time(ts), b = time(nat);
        console.log(
          `  forward ${INPUT_LEN}→512→256→${specs[2].out}: ` +
            `TS ${a.us.toFixed(1)}µs / native ${b.us.toFixed(1)}µs ` +
            `(×${(a.us / b.us).toFixed(1)})`,
        );
        assert(Number.isFinite(a.sink + b.sink));
      } finally {
        closeNet(nat);
      }
    });
  },
});
