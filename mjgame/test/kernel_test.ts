// `src/kernel.ts` as a contract, independent of whether the dylib exists.
//
// The native-vs-TypeScript differential lives in `kernel_native_test.ts` and
// skips without a dylib. THIS file must pass on a machine with no clang++ at
// all, so it asserts the only thing that is true on both paths — and it is the
// thing that matters: the wrapper answers exactly what `mjrender/shanten.ts`
// answers, for every hand, at every openMelds, at every base.

import { assertEquals } from "@std/assert";
import {
  countsFromTiles as countsRef,
  shanten as shantenTS,
  ukeireTypes as ukeireTS,
} from "mjrender/shanten.ts";
import { closeKernel, countsFromTiles, shanten, ukeireTypes } from "../src/kernel.ts";
import { sfc32 } from "../src/rng.ts";

/** `countsFromTiles` is re-exported, not reimplemented — same function object. */
Deno.test("kernel: countsFromTiles は mjrender のものそのもの", () => {
  assertEquals(countsFromTiles, countsRef);
  assertEquals(countsFromTiles([0, 1, 2, 3, 4, 4, 5]).length, 34);
});

function randomHands(seed: number, n: number): number[][] {
  const rng = sfc32(seed);
  const out: number[][] = [];
  const sizes = [0, 1, 4, 7, 10, 13, 14];
  for (let i = 0; i < n; i++) {
    const counts = new Array<number>(34).fill(0);
    const avail = new Array<number>(34).fill(4);
    let left = 136;
    const size = sizes[rng.int(sizes.length)];
    for (let d = 0; d < size; d++) {
      let k = rng.int(left);
      for (let t = 0; t < 34; t++) {
        if (k < avail[t]) {
          counts[t]++;
          avail[t]--;
          left--;
          break;
        }
        k -= avail[t];
      }
    }
    out.push(counts);
  }
  return out;
}

Deno.test("kernel: shanten / ukeireTypes が mjrender と完全一致する", () => {
  for (const counts of randomHands(340200, 400)) {
    for (let om = 0; om <= 5; om++) {
      for (const cl of [true, false]) {
        const base = shantenTS(counts, om, cl);
        assertEquals(shanten(counts, om, cl), base, `shanten [${counts}] open=${om} closed=${cl}`);
        for (const b of [base, base + 1, base - 1]) {
          assertEquals(
            ukeireTypes(counts, om, cl, b),
            ukeireTS(counts, om, cl, b),
            `ukeire [${counts}] open=${om} closed=${cl} base=${b}`,
          );
        }
      }
    }
  }
});

Deno.test("kernel: 既定引数 (openMelds=0, closed=true, base=shanten) が mjrender と揃う", () => {
  for (const counts of randomHands(340201, 200)) {
    assertEquals(shanten(counts), shantenTS(counts));
    assertEquals(ukeireTypes(counts), ukeireTS(counts));
  }
});

Deno.test("kernel: shanten は counts を書き換えない", () => {
  const counts = countsFromTiles([0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 108, 16, 20]);
  const before = counts.slice();
  shanten(counts, 0, true);
  ukeireTypes(counts, 0, true);
  assertEquals(counts, before);
});

Deno.test("kernel: 34長でない counts は TS 実装にそのまま委ねる", () => {
  // The dylib's ABI is a fixed 34-byte vector, so anything else must never
  // reach it — a short array would have it reading past the buffer. The wrapper
  // hands those to mjrender instead and matches whatever mjrender makes of them.
  //
  // Only the OVER-long case is exercised: mjrender's DFS reads counts[i] for
  // i < 34 unconditionally, so a SHORT array makes the reference itself recurse
  // until the stack ends. That is a property of the reference, not of the
  // wrapper, and there is nothing here to be bit-identical to.
  const long = new Array<number>(40).fill(0);
  for (const t of [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5, 34, 39]) long[t]++;
  assertEquals(shanten(long, 0, true), shantenTS(long.slice(), 0, true));
  assertEquals(ukeireTypes(long, 0, true), ukeireTS(long.slice(), 0, true));
  assertEquals(shanten(long, 2, false), shantenTS(long.slice(), 2, false));
});

Deno.test("kernel: MJGAME_NATIVE=0 は dylib があっても TS 経路のまま", () => {
  // Needs --allow-env to force the gate; without it there is nothing to assert
  // beyond what the tests above already cover on either path.
  let may = false;
  try {
    may =
      Deno.permissions.querySync({ name: "env", variable: "MJGAME_NATIVE" }).state === "granted";
  } catch {
    may = false;
  }
  if (!may) return;

  const had = Deno.env.get("MJGAME_NATIVE");
  try {
    Deno.env.set("MJGAME_NATIVE", "0");
    closeKernel();
    for (const counts of randomHands(340202, 120)) {
      assertEquals(shanten(counts, 0, true), shantenTS(counts, 0, true));
      assertEquals(ukeireTypes(counts, 1, false), ukeireTS(counts, 1, false));
    }
  } finally {
    if (had === undefined) Deno.env.delete("MJGAME_NATIVE");
    else Deno.env.set("MJGAME_NATIVE", had);
    closeKernel();
  }
});
