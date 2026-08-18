// The native compute kernel, judged against the TypeScript it replaces.
//
// There is no tolerance here and no "close enough": shanten is an integer and
// ukeire is a set, so the only acceptable result is IDENTICAL. Every case that
// disagrees is printed with its count vector, because a shanten bug that is not
// localised is a shanten bug that is not fixed.
//
// Skipped — loudly, never silently — when the dylib cannot be produced (no
// clang++, no `--allow-run`, no `--allow-ffi`), the same way the rlnet tests
// skip: the kernel is an accelerator, never a requirement.
//
// VOLUME. The reference TypeScript costs ~14 µs per shanten and ~680 µs per
// ukeire probe (34 shanten each), so the *judge* is what makes a fuzz slow, not
// the thing being judged — and past a few thousand hands the random loop stops
// finding anything the targeted batteries below do not already cover
// exhaustively. So the random count is deliberately modest and a knob:
// `MJGAME_FUZZ` sets the number of ukeire-checked hands (default 1_000, with
// twenty times that many shanten-checked ones) for anyone who wants to grind on
// it after touching mjkernel.cc. The batteries — every kokushi subset, every
// chiitoi pair/single split, the saturated and over-long degenerates — run in
// full at every setting, and they are where the edge cases actually live.

import { assert, assertEquals } from "@std/assert";
import { shanten as shantenTS, ukeireTypes as ukeireTS } from "mjrender/shanten.ts";
import { closeKernel, KERNEL_LIB_URL, kernelNative, shanten, ukeireTypes } from "../src/kernel.ts";
import { sfc32 } from "../src/rng.ts";

// ---------------------------------------------------------------------------
// build the dylib if it is not there yet
// ---------------------------------------------------------------------------

/** Empty when the kernel is testable here, otherwise why it is not. */
function ensureDylib(): string {
  try {
    Deno.statSync(KERNEL_LIB_URL);
    return "";
  } catch {
    // not built yet — fall through and build it
  }
  const src = new URL("mjkernel.cc", KERNEL_LIB_URL);
  const args = [
    "-std=c++17",
    "-O3",
    "-flto",
    "-Wall",
    "-Wextra",
    "-fvisibility=hidden",
    "-dynamiclib",
    "-o",
    KERNEL_LIB_URL.pathname,
    src.pathname,
  ];
  let out: Deno.CommandOutput;
  try {
    out = new Deno.Command("clang++", { args, stderr: "piped", stdout: "piped" }).outputSync();
  } catch (e) {
    return `clang++ を実行できません (${e instanceof Error ? e.message : String(e)})`;
  }
  if (!out.success) {
    return `clang++ が失敗しました: ${new TextDecoder().decode(out.stderr).trim()}`;
  }
  return "";
}

const SKIP_REASON = ensureDylib();
if (SKIP_REASON) console.log(`kernel native テストを飛ばします: ${SKIP_REASON}`);

// The gate is an env var and the kernel latches its decision on first use, so
// force it here: a stray MJGAME_NATIVE=0 in the shell would otherwise turn this
// whole file into a silent comparison of TypeScript with itself.
const AMBIENT_GATE = Deno.env.get("MJGAME_NATIVE");
Deno.env.set("MJGAME_NATIVE", "1");
closeKernel();
const NATIVE = SKIP_REASON === "" && (() => {
  try {
    return kernelNative();
  } catch (e) {
    console.log(`kernel native テストを飛ばします: ${e instanceof Error ? e.message : e}`);
    return false;
  }
})();
if (AMBIENT_GATE === undefined) Deno.env.delete("MJGAME_NATIVE");
else Deno.env.set("MJGAME_NATIVE", AMBIENT_GATE);

const SKIP = !NATIVE;

const DEFAULT_FUZZ = 1000;

/** `MJGAME_FUZZ`, read only if --allow-env would not have to prompt for it. */
function fuzzCount(): number {
  try {
    if (Deno.permissions.querySync({ name: "env", variable: "MJGAME_FUZZ" }).state !== "granted") {
      return DEFAULT_FUZZ;
    }
  } catch {
    return DEFAULT_FUZZ;
  }
  const n = Number(Deno.env.get("MJGAME_FUZZ"));
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_FUZZ;
}

// ---------------------------------------------------------------------------
// the comparison
// ---------------------------------------------------------------------------

class Judge {
  shantenChecks = 0;
  ukeireChecks = 0;
  n = 0;
  readonly diffs: string[] = [];

  private note(what: string, counts: number[], om: number, cl: boolean, ts: string, nv: string) {
    this.n++;
    if (this.diffs.length < 20) {
      this.diffs.push(
        `${what} counts=[${counts.join(",")}] open=${om} closed=${cl}: TS=${ts} native=${nv}`,
      );
    }
  }

  shanten(counts: number[], om: number, cl: boolean): number {
    this.shantenChecks++;
    const ts = shantenTS(counts, om, cl);
    const nv = shanten(counts, om, cl);
    if (ts !== nv) this.note("shanten", counts, om, cl, String(ts), String(nv));
    return ts;
  }

  /**
   * `base` is a caller-supplied parameter, not always `shanten()`, so the mask
   * is checked at the neighbouring bases too — a kernel that only agrees at
   * base = shanten() would still be wrong for `heuristic.ts`'s call-after-call.
   */
  ukeire(counts: number[], om: number, cl: boolean, bases: number[]): void {
    for (const base of bases) {
      this.ukeireChecks++;
      const a = ukeireTS(counts, om, cl, base).join(",");
      const b = ukeireTypes(counts, om, cl, base).join(",");
      if (a !== b) this.note(`ukeire base=${base}`, counts, om, cl, a, b);
    }
  }

  /** Shanten over every (openMelds, closed) that behaves differently, incl. cap < 0. */
  shantenAll(counts: number[]): void {
    for (let om = 0; om <= 5; om++) {
      this.shanten(counts, om, true);
      this.shanten(counts, om, false);
    }
  }

  report(label: string): void {
    for (const d of this.diffs) console.error(`差分 [${label}] ${d}`);
    assertEquals(this.n, 0, `${label}: ${this.n} 件の不一致`);
  }
}

// ---------------------------------------------------------------------------
// hand generators
// ---------------------------------------------------------------------------

const HAND_SIZES = [0, 1, 2, 4, 7, 10, 13, 14];

interface IntRng {
  int(n: number): number;
}

/** A count vector dealt from a real (≤4 of a kind) wall restricted to `pool`. */
function deal(rng: IntRng, size: number, pool: number[]): number[] {
  const counts = new Array<number>(34).fill(0);
  const avail = new Array<number>(34).fill(0);
  for (const t of pool) avail[t] = 4;
  let left = pool.length * 4;
  for (let i = 0; i < size && left > 0; i++) {
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
  return counts;
}

const ALL34 = Array.from({ length: 34 }, (_, i) => i);
const HONORS = Array.from({ length: 7 }, (_, i) => 27 + i);
const YAOCHU = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
const ONE_SUIT = Array.from({ length: 9 }, (_, i) => i);
/** Four kinds only ⇒ every draw piles onto a tiny support: kan-heavy shapes. */
const NARROW = [4, 13, 22, 31];
const POOLS = [ALL34, ALL34, ALL34, HONORS, YAOCHU, ONE_SUIT, NARROW];

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

Deno.test({
  name: "kernel native: 乱数手牌の差分ファズ (shanten / ukeire)",
  ignore: SKIP,
  fn: () => {
    const want = fuzzCount();
    const rng = sfc32(340000);
    const j = new Judge();
    let hands = 0;
    for (let n = 0; n < want * 20; n++) {
      const counts = deal(
        rng,
        HAND_SIZES[rng.int(HAND_SIZES.length)],
        POOLS[rng.int(POOLS.length)],
      );
      const om = rng.int(6); // 5 exercises the cap < 0 branch
      const cl = rng.int(2) === 0;
      const ts = j.shanten(counts, om, cl);
      hands++;
      if (n % 20 === 0) j.ukeire(counts, om, cl, [ts, ts + 1, ts - 1]);
    }
    j.report(`random x${hands}`);
    console.log(
      `  ファズ: ${hands} 手牌 / shanten ${j.shantenChecks} 件 / ukeire ${j.ukeireChecks} 件 一致`,
    );
  },
});

Deno.test({
  name: "kernel native: 国士無双 — 么九13種の全部分集合",
  ignore: SKIP,
  fn: () => {
    const j = new Judge();
    for (let mask = 0; mask < 1 << 13; mask++) {
      const counts = new Array<number>(34).fill(0);
      for (let k = 0; k < 13; k++) if (mask & (1 << k)) counts[YAOCHU[k]] = 1;
      j.shantenAll(counts);
      if ((mask & 15) === 0) j.ukeire(counts, 0, true, [j.shanten(counts, 0, true)]);

      // the same subset with one kind doubled: the "pair secured" variants
      for (let k = 0; k < 13; k++) {
        if (!(mask & (1 << k))) continue;
        counts[YAOCHU[k]] = 2;
        const ts = j.shanten(counts, 0, true);
        j.shanten(counts, 1, false);
        if ((mask & 15) === 0) j.ukeire(counts, 0, true, [ts]);
        counts[YAOCHU[k]] = 1;
      }
    }
    j.report("kokushi subsets");
  },
});

Deno.test({
  name: "kernel native: 七対子 — ペア数 × 孤立牌数の全組み合わせ",
  ignore: SKIP,
  fn: () => {
    const j = new Judge();
    const rng = sfc32(340001);
    for (let pairs = 0; pairs <= 7; pairs++) {
      for (let singles = 0; singles + pairs * 2 <= 14; singles++) {
        for (let rep = 0; rep < 12; rep++) {
          const counts = new Array<number>(34).fill(0);
          const perm = ALL34.slice();
          for (let i = perm.length - 1; i > 0; i--) {
            const k = rng.int(i + 1);
            const tmp = perm[i];
            perm[i] = perm[k];
            perm[k] = tmp;
          }
          let p = 0;
          for (let i = 0; i < pairs; i++) counts[perm[p++]] = 2;
          for (let i = 0; i < singles; i++) counts[perm[p++]] = 1;
          j.shantenAll(counts);
          if (rep === 0) j.ukeire(counts, 0, true, [j.shanten(counts, 0, true)]);
        }
      }
    }
    j.report("chiitoi shapes");
  },
});

Deno.test({
  name: "kernel native: 縮退ケース (空手牌 / 槓だらけ / 字牌のみ / 15枚超)",
  ignore: SKIP,
  fn: () => {
    const j = new Judge();
    const zero = new Array<number>(34).fill(0);
    j.shantenAll(zero);
    j.ukeire(zero, 0, true, [j.shanten(zero, 0, true), 3]);

    // every tile alone, doubled, tripled, quadrupled
    for (let t = 0; t < 34; t++) {
      for (const n of [1, 2, 3, 4]) {
        const c = zero.slice();
        c[t] = n;
        j.shantenAll(c);
        j.ukeire(c, 0, true, [j.shanten(c, 0, true)]);
        j.ukeire(c, 2, false, [j.shanten(c, 2, false)]);
      }
    }
    // three kans at once: the density the table's clamps are about
    for (let a = 0; a < 34; a++) {
      for (const b of [a + 1, a + 5, a + 9, a + 17]) {
        if (b > 33) continue;
        const c = zero.slice();
        c[a] = 4;
        c[b] = 4;
        c[(a + 3) % 34] += 4;
        j.shantenAll(c);
        j.ukeire(c, 0, true, [j.shanten(c, 0, true)]);
      }
    }
    // honours only, saturated
    for (let n = 1; n <= 4; n++) {
      const c = zero.slice();
      for (const t of HONORS) c[t] = n;
      j.shantenAll(c);
      j.ukeire(c, 0, true, [j.shanten(c, 0, true)]);
    }
    // deliberately over-long hands: outside the table's domain, so these land on
    // the reference DFS inside the dylib — and must still be exact
    const rng = sfc32(340002);
    for (let n = 0; n < 200; n++) {
      const c = deal(rng, 15 + rng.int(6), ALL34);
      j.shantenAll(c);
      if (n % 10 === 0) j.ukeire(c, 0, true, [j.shanten(c, 0, true)]);
    }
    j.report("degenerate");
  },
});

Deno.test({
  name: "kernel native: 聴牌形の受け入れが待ち牌そのものになる",
  ignore: SKIP,
  fn: () => {
    // A shanten-0 hand accepts exactly its winning tiles. Every caller in src/
    // leans on that, so it gets a hand-checked assertion of its own rather than
    // only being covered by the fuzz.
    const c = new Array<number>(34).fill(0);
    for (const t of [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5]) c[t]++;
    assertEquals(shanten(c, 0, true), 0);
    assertEquals(ukeireTypes(c, 0, true), [3, 6]);
    assertEquals(ukeireTypes(c, 0, true), ukeireTS(c, 0, true));
    // the 34th type (中) really is reachable through the mask's high word
    const kokushi = new Array<number>(34).fill(0);
    for (const t of YAOCHU) kokushi[t] = 1;
    kokushi[0] = 2; // 14 tiles: a complete kokushi, i.e. shanten -1
    assertEquals(shanten(kokushi, 0, true), -1);
    assertEquals(ukeireTypes(kokushi, 0, true), ukeireTS(kokushi, 0, true));
  },
});

Deno.test({
  name: "kernel native: ゲート — 0 で TS 経路、1 で native、閉じて開き直せる",
  ignore: SKIP,
  fn: () => {
    const had = Deno.env.get("MJGAME_NATIVE");
    const c = new Array<number>(34).fill(0);
    for (const t of [0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5]) c[t]++;
    try {
      Deno.env.set("MJGAME_NATIVE", "0");
      closeKernel();
      assert(!kernelNative(), "MJGAME_NATIVE=0 なのに native が有効です");
      assertEquals(shanten(c, 0, true), shantenTS(c, 0, true));

      Deno.env.set("MJGAME_NATIVE", "1");
      closeKernel();
      assert(kernelNative(), "MJGAME_NATIVE=1 で native が付いていません");
      assertEquals(shanten(c, 0, true), shantenTS(c, 0, true));
    } finally {
      if (had === undefined) Deno.env.delete("MJGAME_NATIVE");
      else Deno.env.set("MJGAME_NATIVE", had);
      closeKernel(); // back to the ambient decision for whatever runs next
    }
  },
});
