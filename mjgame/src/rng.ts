// Seedable PRNG for gameplay.
//
// sfc32 — 128 bits of state, four uint32 ops per draw, no BigInt. Chosen over
// xoshiro256** (which needs BigInt in JS and runs ~20x slower) because RL
// self-play does millions of rollouts. Tenhou's MT19937 lives in
// `tenhou_wall.ts` and is used only to validate our wall layout against real
// logs — never for gameplay.

export interface Rng {
  /** Uniform uint32. */
  u32(): number;
  /** Uniform integer in [0, n). */
  int(n: number): number;
  /** Uniform float in [0, 1). */
  float(): number;
  /** An independent substream, so a match seed reproduces every wall
   *  regardless of how many numbers the policies consumed. */
  fork(tag: number): Rng;
}

/** splitmix32 — used to expand a single seed into sfc32's four words. */
function splitmix32(a: number): () => number {
  return () => {
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15);
    t = Math.imul(t, 0x735a2d97);
    return (t ^ (t >>> 15)) >>> 0;
  };
}

function hashString(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function sfc32(seed: number | string): Rng {
  const base = typeof seed === "string" ? hashString(seed) : seed >>> 0;
  const sm = splitmix32(base);
  let a = sm(), b = sm(), c = sm(), d = sm();

  const rng: Rng = {
    u32(): number {
      a |= 0;
      b |= 0;
      c |= 0;
      d |= 0;
      const t = (((a + b) | 0) + d) | 0;
      d = (d + 1) | 0;
      a = b ^ (b >>> 9);
      b = (c + (c << 3)) | 0;
      c = (c << 21) | (c >>> 11);
      c = (c + t) | 0;
      return t >>> 0;
    },
    int(n: number): number {
      // Rejection-free is not needed here: n <= 136, and the modulo bias at
      // 2^32 is < 2^-25. Tenhou itself accepts the same bias (tenhou.cc:86).
      return rng.u32() % n;
    },
    float(): number {
      return rng.u32() / 4294967296;
    },
    fork(tag: number): Rng {
      return sfc32((base ^ Math.imul(tag + 1, 0x9e3779b9)) >>> 0);
    },
  };
  return rng;
}
