// The small-MLP runtime: validation, the reference arithmetic, and the Python
// parity fixture.
//
// This file judges the TYPESCRIPT path only — it forces `MJGAME_NATIVE=0`
// around every `buildMlp`, so a shell that demanded native cannot turn a
// reference test into a test of the dylib, and a missing dylib cannot fail it.
// The dylib is judged next door, in `mlp_native_test.ts`.
//
// The fixture (`test/fixtures/mlp-parity.json`, written by
// `train/mlp_selftest.py`) is the third mirror: MLX/numpy trains the heads, so
// its own double-accumulate/float32-store reference has to produce the same
// bits this engine will. Equality here is EXACT — `assertEquals` on the array,
// no tolerance — because a head's sign decides a fold.

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { Mlp, MlpSpec } from "../src/ai/mlp.ts";
import {
  buildMlp,
  mlpForward,
  mlpForwardBatch,
  mlpNative,
  packMlp,
  validateMlp,
} from "../src/ai/mlp.ts";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/**
 * `buildMlp` with the gate forced off, restored immediately — the same
 * save/latch/restore dance `kernel_native_test.ts` performs, for the opposite
 * reason: there, so an ambient `0` cannot silently compare TypeScript with
 * itself; here, so an ambient `1` cannot make a reference test depend on a
 * build step.
 */
function tsBuild(spec: MlpSpec): Mlp {
  const ambient = Deno.env.get("MJGAME_NATIVE");
  Deno.env.set("MJGAME_NATIVE", "0");
  try {
    // Gate "off" is decided before the dylib is even looked at, so this needs
    // no `closeMlpLib()` — and must not call one: another test file's heads may
    // be holding the very library handle it would close.
    return buildMlp(spec);
  } finally {
    if (ambient === undefined) Deno.env.delete("MJGAME_NATIVE");
    else Deno.env.set("MJGAME_NATIVE", ambient);
  }
}

function layer(nIn: number, nOut: number, act: "relu" | "none", w: number[], b: number[]) {
  return { in: nIn, out: nOut, act, w, b };
}

/** in 2 → 2 (relu) → 1 (none), with numbers that are exact in float32. */
const HAND: MlpSpec = {
  fv: 1,
  layers: [
    layer(2, 2, "relu", [1, 2, -1, 0.5], [0.5, 1]),
    layer(2, 1, "none", [3, -4], [0.25]),
  ],
};

const WANT = { inputs: 2, outputs: 1, fv: 1 };

/** A rejection case: the mutation, and the substring the message must carry. */
function rejects(name: string, mutate: (s: MlpSpec) => unknown, needle: string) {
  Deno.test(`mlp: validateMlp が拒む — ${name}`, () => {
    const spec = JSON.parse(JSON.stringify(HAND)) as MlpSpec;
    const bad = mutate(spec);
    const e = assertThrows(() => validateMlp(bad, WANT, "fold"), Error);
    assert(
      (e as Error).message.includes(needle),
      `メッセージが "${needle}" を含みません: ${(e as Error).message}`,
    );
    assert((e as Error).message.startsWith("fold:"), `where が先頭にありません: ${e}`);
  });
}

// ---------------------------------------------------------------------------
// validation
// ---------------------------------------------------------------------------

Deno.test("mlp: validateMlp は正しいブロックを通す", () => {
  const ok = validateMlp(JSON.parse(JSON.stringify(HAND)), WANT, "fold");
  assertEquals(ok.fv, 1);
  assertEquals(ok.layers.length, 2);
  assertEquals(ok.layers[0].w.length, 4);
});

rejects("layers が空", (s) => ({ ...s, layers: [] }), "layers が空");
rejects("layers が無い", (s) => ({ fv: s.fv }), "layers が空");
rejects("オブジェクトでない", () => [1, 2, 3], "{fv, layers}");
rejects("fv 違い", (s) => ({ ...s, fv: 2 }), "特徴量版 1 と違います");
rejects("fv が数値でない", (s) => ({ ...s, fv: "1" }), "fv が数値ではありません");
rejects("入力次元違い", (s) => {
  s.layers[0].in = 3;
  s.layers[0].w = new Array(6).fill(0);
  return s;
}, "入力次元 2 と違います");
rejects("層の連鎖が切れている", (s) => {
  s.layers[1].in = 3;
  s.layers[1].w = [1, 2, 3];
  return s;
}, "前層の out=2 と違います");
rejects("出力次元違い", (s) => {
  s.layers[1].out = 2;
  s.layers[1].w = [1, 2, 3, 4];
  s.layers[1].b = [0, 0];
  return s;
}, "出力次元 1 と違います");
rejects("act が不正", (s) => {
  (s.layers[0] as { act: string }).act = "tanh";
  return s;
}, '"relu" か "none"');
rejects("w の長さ違い", (s) => {
  s.layers[0].w = [1, 2, 3];
  return s;
}, "w の長さが 3 (期待 4)");
rejects("b の長さ違い", (s) => {
  s.layers[0].b = [0.5];
  return s;
}, "b の長さが 1 (期待 2)");
rejects("w に NaN", (s) => {
  s.layers[0].w[2] = NaN;
  return s;
}, "w[2] が有限の数値ではありません");
rejects("b に Infinity", (s) => {
  s.layers[1].b[0] = Infinity;
  return s;
}, "b[0] が有限の数値ではありません");
rejects("w が配列でない", (s) => {
  (s.layers[0] as { w: unknown }).w = 3;
  return s;
}, "w は数値の配列です");
rejects("in が整数でない", (s) => {
  s.layers[0].in = 2.5;
  return s;
}, "in が正整数ではありません");
rejects("out が 0", (s) => {
  s.layers[1].out = 0;
  return s;
}, "out が正整数ではありません");
rejects("層がオブジェクトでない", (s) => ({ ...s, layers: [1] }), "{in, out, act, w, b}");

// ---------------------------------------------------------------------------
// forward — hand-checked
// ---------------------------------------------------------------------------

Deno.test("mlp: 手計算の2層順伝播", () => {
  const m = tsBuild(validateMlp(HAND, WANT, "fold"));
  assert(!mlpNative(m), "この検査は TS 経路のものです");
  assertEquals(m.inputs, 2);
  assertEquals(m.outputs, 1);

  // x = [2, 1]
  //   h0 = 0.5 + 1*2 + 2*1   =  4.5  → relu → 4.5
  //   h1 = 1   + -1*2 + 0.5*1 = -0.5 → relu → 0
  //   y  = 0.25 + 3*4.5 + -4*0 = 13.75
  assertEquals(Array.from(mlpForward(m, new Float32Array([2, 1]))), [13.75]);

  // x = [-1, -1]
  //   h0 = 0.5 - 1 - 2 = -2.5 → 0
  //   h1 = 1 + 1 - 0.5 =  1.5 → 1.5
  //   y  = 0.25 + 0 + -4*1.5 = -5.75
  assertEquals(Array.from(mlpForward(m, new Float32Array([-1, -1]))), [-5.75]);
});

Deno.test("mlp: out を渡すとそこに書き、返り値は同じ配列", () => {
  const m = tsBuild(validateMlp(HAND, WANT, "fold"));
  const out = new Float32Array(1);
  const got = mlpForward(m, new Float32Array([2, 1]), out);
  assert(got === out);
  assertEquals(out[0], 13.75);
});

Deno.test("mlp: 長さが違えば投げる", () => {
  const m = tsBuild(validateMlp(HAND, WANT, "fold"));
  assertThrows(() => mlpForward(m, new Float32Array(3)), Error, "入力長 3");
  assertThrows(
    () => mlpForward(m, new Float32Array(2), new Float32Array(2)),
    Error,
    "出力長 2",
  );
  assertThrows(() => mlpForwardBatch(m, -1, new Float32Array(2), new Float32Array(1)), Error);
  assertThrows(
    () => mlpForwardBatch(m, 2, new Float32Array(2), new Float32Array(2)),
    Error,
    "入力長 2",
  );
});

Deno.test("mlp: バッチは単発の連結と同じ", () => {
  const m = tsBuild(validateMlp(HAND, WANT, "fold"));
  const rows = [[2, 1], [-1, -1], [0, 0], [3.5, -2.25]];
  const xs = new Float32Array(rows.flat());
  const outs = new Float32Array(rows.length);
  mlpForwardBatch(m, rows.length, xs, outs);
  const singles = rows.map((r) => mlpForward(m, new Float32Array(r))[0]);
  assertEquals(Array.from(outs), singles);

  // n = 0 writes nothing.
  const before = Array.from(outs);
  mlpForwardBatch(m, 0, xs, outs);
  assertEquals(Array.from(outs), before);
});

Deno.test("mlp: 単層 (act none) は素の線形", () => {
  const spec: MlpSpec = { fv: 1, layers: [layer(3, 2, "none", [1, 0, -1, 0.5, 0.5, 0.5], [0, 1])] };
  const m = tsBuild(validateMlp(spec, { inputs: 3, outputs: 2, fv: 1 }, "fold"));
  assertEquals(Array.from(mlpForward(m, new Float32Array([1, 2, 4]))), [-3, 4.5]);
});

Deno.test("mlp: packMlp は層ごとに w のあと b", () => {
  const m = tsBuild(validateMlp(HAND, WANT, "fold"));
  assertEquals(
    Array.from(packMlp(m)),
    [1, 2, -1, 0.5, 0.5, 1, 3, -4, 0.25],
  );
});

Deno.test("mlp: 重みは複製される (JSON を書き換えても影響しない)", () => {
  const spec = JSON.parse(JSON.stringify(HAND)) as MlpSpec;
  const m = tsBuild(validateMlp(spec, WANT, "fold"));
  spec.layers[0].w[0] = 1000;
  spec.layers[1].b[0] = -1000;
  assertEquals(Array.from(mlpForward(m, new Float32Array([2, 1]))), [13.75]);
});

// ---------------------------------------------------------------------------
// the Python parity fixture
// ---------------------------------------------------------------------------

interface ParityDoc {
  nets: { spec: MlpSpec; inputs: number[][]; outputs: number[][] }[];
}

Deno.test("mlp: Python の参照実装とビット一致 (fixture)", () => {
  const path = new URL("fixtures/mlp-parity.json", import.meta.url);
  const doc = JSON.parse(Deno.readTextFileSync(path)) as ParityDoc;
  assert(doc.nets.length >= 3, "fixture が薄すぎます");

  let vectors = 0;
  for (const [n, net] of doc.nets.entries()) {
    const inputs = net.spec.layers[0].in;
    const outputs = net.spec.layers[net.spec.layers.length - 1].out;
    const m = tsBuild(validateMlp(net.spec, { inputs, outputs, fv: 1 }, `fixture[${n}]`));
    assertEquals(net.inputs.length, net.outputs.length);
    for (const [r, x] of net.inputs.entries()) {
      assertEquals(
        Array.from(mlpForward(m, new Float32Array(x))),
        net.outputs[r],
        `net ${n} row ${r}`,
      );
      vectors++;
    }
    // …and the batch entry point sees the same numbers.
    const xs = new Float32Array(net.inputs.flat());
    const outs = new Float32Array(net.inputs.length * outputs);
    mlpForwardBatch(m, net.inputs.length, xs, outs);
    assertEquals(Array.from(outs), net.outputs.flat(), `net ${n} batch`);
  }
  assert(vectors >= 15, `検査したベクトルが ${vectors} 本しかありません`);
});
