// Tenhou's own wall generator — TEST AND REPLAY ONLY.
//
// A port of mt19937ar.cc + the hash/shuffle stage of tenhou.cc:55-93. Gameplay
// uses `Wall.shuffled(sfc32(...))` instead; this module exists so we can prove
// our dead-wall / rinshan / draw-order layout against real logs (test/wall_test.ts),
// and so a recorded Tenhou game can later be re-simulated in the engine.
//
// Pipeline per round: 288 uint32 from MT19937 -> 9 SHA-512 blocks of 128 bytes
// each -> 144 uint32 (`rnd`) -> Fisher-Yates over 136 tiles using rnd[0..134],
// dice from rnd[135] and rnd[136]. rnd[137..143] are unused.

import { createHash } from "node:crypto";
import type { Tile } from "mjrender/model.ts";

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

export class MT19937 {
  private mt = new Uint32Array(N);
  private mti = N + 1;

  initGenrand(s: number): void {
    this.mt[0] = s >>> 0;
    for (this.mti = 1; this.mti < N; this.mti++) {
      const prev = this.mt[this.mti - 1];
      this.mt[this.mti] = (Math.imul(1812433253, prev ^ (prev >>> 30)) + this.mti) >>> 0;
    }
  }

  initByArray(key: Uint32Array): void {
    this.initGenrand(19650218);
    let i = 1, j = 0;
    let k = Math.max(N, key.length);
    for (; k; k--) {
      const prev = this.mt[i - 1];
      this.mt[i] = (((this.mt[i] ^ Math.imul(prev ^ (prev >>> 30), 1664525)) >>> 0) +
        key[j] + j) >>> 0;
      i++;
      j++;
      if (i >= N) {
        this.mt[0] = this.mt[N - 1];
        i = 1;
      }
      if (j >= key.length) j = 0;
    }
    for (k = N - 1; k; k--) {
      const prev = this.mt[i - 1];
      this.mt[i] = (((this.mt[i] ^ Math.imul(prev ^ (prev >>> 30), 1566083941)) >>> 0) -
        i) >>> 0;
      i++;
      if (i >= N) {
        this.mt[0] = this.mt[N - 1];
        i = 1;
      }
    }
    this.mt[0] = UPPER_MASK;
  }

  genrandInt32(): number {
    let y: number;
    if (this.mti >= N) {
      if (this.mti === N + 1) this.initGenrand(5489);
      let kk = 0;
      for (; kk < N - M; kk++) {
        y = ((this.mt[kk] & UPPER_MASK) | (this.mt[kk + 1] & LOWER_MASK)) >>> 0;
        this.mt[kk] = (this.mt[kk + M] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0)) >>>
          0;
      }
      for (; kk < N - 1; kk++) {
        y = ((this.mt[kk] & UPPER_MASK) | (this.mt[kk + 1] & LOWER_MASK)) >>> 0;
        this.mt[kk] = (this.mt[kk + (M - N)] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0)) >>> 0;
      }
      y = ((this.mt[N - 1] & UPPER_MASK) | (this.mt[0] & LOWER_MASK)) >>> 0;
      this.mt[N - 1] = (this.mt[M - 1] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0)) >>>
        0;
      this.mti = 0;
    }
    y = this.mt[this.mti++];
    y ^= y >>> 11;
    y = (y ^ ((y << 7) & 0x9d2c5680)) >>> 0;
    y = (y ^ ((y << 15) & 0xefc60000)) >>> 0;
    y = (y ^ (y >>> 18)) >>> 0;
    return y >>> 0;
  }
}

const SEED_PREFIX = "mt19937ar-sha512-n288-base64,";

/** Decode a log's `SHUFFLE seed` attribute into MT19937's 624-word key. */
export function decodeSeed(seedAttr: string): Uint32Array {
  const b64 = seedAttr.startsWith(SEED_PREFIX) ? seedAttr.slice(SEED_PREFIX.length) : seedAttr;
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  if (bytes.length < N * 4) {
    throw new Error(`seed too short: ${bytes.length} bytes, need ${N * 4}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const key = new Uint32Array(N);
  for (let i = 0; i < N; i++) key[i] = view.getUint32(i * 4, true);
  return key;
}

export interface TenhouRoundWall {
  tiles: Tile[]; // 136, in Wall's index convention
  dice: [number, number];
}

/** Regenerate the first `rounds` walls of a game from its seed. */
export function tenhouWalls(seedAttr: string, rounds: number): TenhouRoundWall[] {
  const mt = new MT19937();
  mt.initByArray(decodeSeed(seedAttr));

  const out: TenhouRoundWall[] = [];
  for (let r = 0; r < rounds; r++) {
    // 288 uint32 -> little-endian bytes, hashed 128 bytes at a time.
    const src = new Uint8Array(288 * 4);
    const srcView = new DataView(src.buffer);
    for (let i = 0; i < 288; i++) srcView.setUint32(i * 4, mt.genrandInt32(), true);

    const digest = new Uint8Array(9 * 64);
    for (let i = 0; i < 9; i++) {
      const block = src.subarray(i * 128, (i + 1) * 128);
      digest.set(createHash("sha512").update(block).digest(), i * 64);
    }
    const dv = new DataView(digest.buffer);
    const rnd = new Uint32Array(144);
    for (let i = 0; i < 144; i++) rnd[i] = dv.getUint32(i * 4, true);

    const tiles: Tile[] = new Array(136);
    for (let i = 0; i < 136; i++) tiles[i] = i;
    for (let i = 0; i < 135; i++) {
      const j = i + (rnd[i] % (136 - i));
      const tmp = tiles[i];
      tiles[i] = tiles[j];
      tiles[j] = tmp;
    }
    out.push({ tiles, dice: [rnd[135] % 6, rnd[136] % 6] });
  }
  return out;
}
