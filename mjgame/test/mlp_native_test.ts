// `native/libmjmlp`, judged against the TypeScript it replaces.
//
// There is no tolerance here and no "close enough": the two implementations
// must produce the SAME BITS, because a head's sign decides a fold and the
// seat's decision streams are pinned. So the comparison is on the float32 bit
// patterns, not on `===` — that also settles -0 vs 0 and any NaN by itself.
//
// Skipped — loudly, never silently — when the dylib cannot be produced (no
// clang, no `--allow-run`, no `--allow-ffi`), the same way the kernel and rlnet
// tests skip: the shim is an accelerator, never a requirement.
//
// The fuzz is over SHAPES as much as numbers: 1–3 layers, widths 1–64, relu and
// identity mixed, and a deliberate spread of magnitudes so the relu gate is
// exercised in both directions. Every net is checked on 10 inputs, one row at a
// time and again as a batch.

import { assert, assertEquals, assertThrows } from "@std/assert";
import type { Mlp, MlpSpec } from "../src/ai/mlp.ts";
import {
  buildMlp,
  closeMlp,
  closeMlpLib,
  MLP_LIB_URL,
  mlpForward,
  mlpForwardBatch,
  mlpNative,
} from "../src/ai/mlp.ts";
import type { Rng } from "../src/rng.ts";
import { sfc32 } from "../src/rng.ts";

// ---------------------------------------------------------------------------
// build the dylib if it is not there yet
// ---------------------------------------------------------------------------

/** Empty when the shim is testable here, otherwise why it is not. */
function ensureDylib(): string {
  const src = new URL("../native/mlp.c", import.meta.url);
  // Rebuild when the dylib is MISSING or STALE — a stale artifact would either
  // fail the ABI check and turn this whole file into a silent skip, or worse,
  // compare the TypeScript against yesterday's C.
  try {
    const lib = Deno.statSync(MLP_LIB_URL).mtime?.getTime() ?? 0;
    const cc = Deno.statSync(src).mtime?.getTime() ?? 0;
    if (lib >= cc) return "";
  } catch {
    // not built yet — fall through and build it
  }
  const args = [
    "-std=c11",
    "-O3",
    // Keep in step with native/build_mlp.sh — the forward pass is bit-exact
    // only if the compiler is forbidden to contract a multiply-add into an FMA.
    "-ffp-contract=off",
    "-Wall",
    "-Wextra",
    "-fvisibility=hidden",
    "-dynamiclib",
    "-o",
    MLP_LIB_URL.pathname,
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
if (SKIP_REASON) console.log(`mlp native テストを飛ばします: ${SKIP_REASON}`);

const AMBIENT_GATE = Deno.env.get("MJGAME_NATIVE");

/** Restore whatever the shell asked for; every helper below ends with this. */
function restoreGate(): void {
  if (AMBIENT_GATE === undefined) Deno.env.delete("MJGAME_NATIVE");
  else Deno.env.set("MJGAME_NATIVE", AMBIENT_GATE);
}

/** Build with the gate forced ON — the head under test. */
function nativeBuild(spec: MlpSpec): Mlp {
  Deno.env.set("MJGAME_NATIVE", "1");
  try {
    return buildMlp(spec);
  } finally {
    restoreGate();
  }
}

/** Build with the gate forced OFF — the reference. Never touches the dylib. */
function tsBuild(spec: MlpSpec): Mlp {
  Deno.env.set("MJGAME_NATIVE", "0");
  try {
    return buildMlp(spec);
  } finally {
    restoreGate();
  }
}

// A stray MJGAME_NATIVE=0 in the shell would otherwise turn this whole file
// into a comparison of TypeScript with itself, so the probe forces the gate.
const NATIVE = SKIP_REASON === "" && (() => {
  try {
    const probe = nativeBuild({ fv: 1, layers: [{ in: 1, out: 1, act: "none", w: [1], b: [0] }] });
    const ok = mlpNative(probe);
    closeMlp(probe);
    if (!ok) console.log("mlp native テストを飛ばします: dylib を開けません");
    return ok;
  } catch (e) {
    console.log(`mlp native テストを飛ばします: ${e instanceof Error ? e.message : e}`);
    return false;
  }
})();

const SKIP = !NATIVE;

// ---------------------------------------------------------------------------
// bit-exact comparison
// ---------------------------------------------------------------------------

/** The float32 bit patterns, which is the only equality this file accepts. */
function bits(a: Float32Array): number[] {
  const u = new Uint32Array(a.buffer, a.byteOffset, a.length);
  return Array.from(u);
}

function shapeOf(spec: MlpSpec): string {
  return spec.layers.map((l) => `${l.in}x${l.out}${l.act === "relu" ? "R" : ""}`).join("→");
}

// ---------------------------------------------------------------------------
// the fuzz
// ---------------------------------------------------------------------------

const NETS = 300;
const ROWS = 10;

function randomSpec(rng: Rng): MlpSpec {
  const n = 1 + rng.int(3);
  const dims = [1 + rng.int(64)];
  for (let i = 0; i < n; i++) dims.push(1 + rng.int(64));
  // A spread of magnitudes: tiny weights keep sums near cancellation (where a
  // reordered summation would show up first), large ones drive relu hard.
  const scale = [0.01, 0.25, 1, 4][rng.int(4)];
  const layers = [];
  for (let i = 0; i < n; i++) {
    const nIn = dims[i];
    const nOut = dims[i + 1];
    const w: number[] = [];
    for (let k = 0; k < nIn * nOut; k++) w.push((rng.float() * 2 - 1) * scale);
    const b: number[] = [];
    for (let k = 0; k < nOut; k++) b.push((rng.float() * 2 - 1) * scale);
    // The last layer is identity as often as not; hidden layers are usually relu.
    const act = (i === n - 1 ? rng.int(2) === 0 : rng.int(4) !== 0) ? "relu" : "none";
    layers.push({ in: nIn, out: nOut, act: act as "relu" | "none", w, b });
  }
  return { fv: 1, layers };
}

Deno.test({
  name: "mlp native: TS とビット単位で一致する (fuzz)",
  ignore: SKIP,
  fn: () => {
    const rng = sfc32("mlp-native-fuzz");
    const diffs: string[] = [];
    let checks = 0;
    for (let net = 0; net < NETS && diffs.length < 10; net++) {
      const spec = randomSpec(rng);
      const nv = nativeBuild(spec);
      const ts = tsBuild(spec);
      assert(mlpNative(nv), "native 経路になっていません");
      assert(!mlpNative(ts), "参照が native 経路になっています");
      try {
        for (let r = 0; r < ROWS; r++) {
          const x = new Float32Array(nv.inputs);
          for (let i = 0; i < x.length; i++) x[i] = (rng.float() * 2 - 1) * 3;
          const a = bits(mlpForward(ts, x, new Float32Array(ts.outputs)));
          const b = bits(mlpForward(nv, x, new Float32Array(nv.outputs)));
          checks++;
          for (let o = 0; o < a.length; o++) {
            if (a[o] !== b[o] && diffs.length < 10) {
              diffs.push(
                `net ${net} [${shapeOf(spec)}] row ${r} out ${o}: ` +
                  `TS=0x${a[o].toString(16)} native=0x${b[o].toString(16)}`,
              );
            }
          }
        }
      } finally {
        closeMlp(nv);
      }
    }
    assertEquals(diffs, [], `${checks} 本の入力で不一致`);
    assert(checks === NETS * ROWS, `検査本数 ${checks}`);
  },
});

Deno.test({
  name: "mlp native: バッチは単発 n 回と同じ",
  ignore: SKIP,
  fn: () => {
    const rng = sfc32("mlp-native-batch");
    const diffs: string[] = [];
    for (let net = 0; net < 40 && diffs.length < 10; net++) {
      const spec = randomSpec(rng);
      const nv = nativeBuild(spec);
      const ts = tsBuild(spec);
      try {
        const n = 1 + rng.int(34); // 34 is M14's row count
        const xs = new Float32Array(n * nv.inputs);
        for (let i = 0; i < xs.length; i++) xs[i] = (rng.float() * 2 - 1) * 3;

        const batch = new Float32Array(n * nv.outputs);
        mlpForwardBatch(nv, n, xs, batch);

        // …against n single native calls, and against the TS batch.
        const singles = new Float32Array(n * nv.outputs);
        for (let r = 0; r < n; r++) {
          singles.set(
            mlpForward(nv, xs.slice(r * nv.inputs, (r + 1) * nv.inputs)),
            r * nv.outputs,
          );
        }
        const tsBatch = new Float32Array(n * ts.outputs);
        mlpForwardBatch(ts, n, xs, tsBatch);

        const a = bits(batch);
        const b = bits(singles);
        const c = bits(tsBatch);
        for (let i = 0; i < a.length && diffs.length < 10; i++) {
          if (a[i] !== b[i]) diffs.push(`net ${net} slot ${i}: batch != singles`);
          if (a[i] !== c[i]) diffs.push(`net ${net} slot ${i}: native batch != TS batch`);
        }
      } finally {
        closeMlp(nv);
      }
    }
    assertEquals(diffs, []);
  },
});

Deno.test({
  name: "mlp native: M14 の実寸 (34行 × 54入力 → 32 → 1) がビット単位で一致する",
  ignore: SKIP,
  fn: () => {
    // The fuzz above reaches this geometry only by accident (random widths, a
    // random row count). M14 serves EXACTLY this shape, once per opponent per
    // decision — 34 tile types through `DEALIN_F` = 54 columns — so it gets a
    // deterministic case of its own, at the batch entry point the head uses.
    const rng = sfc32("mlp-native-m14");
    const w0: number[] = [];
    for (let k = 0; k < 54 * 32; k++) w0.push((rng.float() * 2 - 1) * 0.25);
    const w1: number[] = [];
    for (let k = 0; k < 32; k++) w1.push((rng.float() * 2 - 1) * 0.5);
    const spec: MlpSpec = {
      fv: 1,
      layers: [
        { in: 54, out: 32, act: "relu", w: w0, b: Array.from({ length: 32 }, () => 0.05) },
        { in: 32, out: 1, act: "none", w: w1, b: [-2] },
      ],
    };
    const nv = nativeBuild(spec);
    const ts = tsBuild(spec);
    try {
      const n = 34;
      const xs = new Float32Array(n * 54);
      for (let i = 0; i < xs.length; i++) xs[i] = (rng.float() * 2 - 1) * 3;
      const batch = new Float32Array(n);
      mlpForwardBatch(nv, n, xs, batch);
      const singles = new Float32Array(n);
      for (let r = 0; r < n; r++) singles[r] = mlpForward(nv, xs.slice(r * 54, (r + 1) * 54))[0];
      const tsBatch = new Float32Array(n);
      mlpForwardBatch(ts, n, xs, tsBatch);
      assertEquals(bits(batch), bits(singles), `${shapeOf(spec)}: batch != singles`);
      assertEquals(bits(batch), bits(tsBatch), `${shapeOf(spec)}: native != TS`);
    } finally {
      closeMlp(nv);
    }
  },
});

Deno.test({
  name: "mlp native: closeMlp のあとは TS 経路",
  ignore: SKIP,
  fn: () => {
    const spec: MlpSpec = {
      fv: 1,
      layers: [{ in: 2, out: 1, act: "none", w: [1.5, -2], b: [0.25] }],
    };
    const m = nativeBuild(spec);
    assert(mlpNative(m));
    const before = mlpForward(m, new Float32Array([2, 1]), new Float32Array(1))[0];
    closeMlp(m);
    assert(!mlpNative(m));
    assertEquals(mlpForward(m, new Float32Array([2, 1]), new Float32Array(1))[0], before);
    assertEquals(before, 1.25);
    closeMlp(m); // idempotent
  },
});

// ---------------------------------------------------------------------------
// the gate — these run LAST because the last one moves the dylib aside
// ---------------------------------------------------------------------------

Deno.test({
  name: "mlp native: MJGAME_NATIVE=0 なら native を使わない",
  ignore: SKIP,
  fn: () => {
    const m = tsBuild({ fv: 1, layers: [{ in: 1, out: 1, act: "none", w: [2], b: [1] }] });
    assert(!mlpNative(m), "ゲート 0 で native が開いています");
    assertEquals(mlpForward(m, new Float32Array([3]), new Float32Array(1))[0], 7);
  },
});

Deno.test({
  name: "mlp native: MJGAME_NATIVE=1 で dylib が無ければ投げる (ビルド手順を名指しで)",
  ignore: SKIP,
  fn: () => {
    const path = MLP_LIB_URL.pathname;
    const aside = `${path}.aside`;
    Deno.renameSync(path, aside);
    closeMlpLib(); // re-arm the gate so the missing file is actually noticed
    try {
      Deno.env.set("MJGAME_NATIVE", "1");
      const e = assertThrows(
        () => buildMlp({ fv: 1, layers: [{ in: 1, out: 1, act: "none", w: [1], b: [0] }] }),
        Error,
      );
      const msg = (e as Error).message;
      assert(msg.includes("MJGAME_NATIVE=1"), msg);
      assert(msg.includes("build-mlp"), `ビルド手順を名指ししていません: ${msg}`);
      assert(msg.includes("native/build_mlp.sh"), msg);
    } finally {
      restoreGate();
      Deno.renameSync(aside, path);
      closeMlpLib();
    }
  },
});
