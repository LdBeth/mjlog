// The wall, using Tenhou's exact index layout.
//
// Copied from the verifier at the repo root (tenhou.cc:78-95) and CLAUDE.md's
// "Dead Wall Layout" section, so that games we generate can be exported as
// Tenhou XML and checked with the same tooling:
//
//   tiles[0..13]        dead wall
//     tiles[1,0,3,2]      rinshan draws, in that order (ord[] = {1,0,3,2})
//     tiles[5,7,9,11,13]  dora indicators   (first from INIT seed[5])
//     tiles[4,6,8,10,12]  ura-dora indicators
//   tiles[14..135]      live wall, 122 tiles, drawn from the TOP: the i-th
//                       live draw is tiles[135 - i]
//
// Each kan moves one tile from the bottom of the live wall (index 14, 15, ...)
// into the dead wall, so total drawable = 122 - kanCount, i.e. after the 52-tile
// deal there are always exactly 70 post-deal draws (live + rinshan) in a round.

import type { Tile } from "mjrender/model.ts";
import type { Rng } from "./rng.ts";
import type { Seat } from "./types.ts";

/** Dead-wall slots used by successive rinshan draws. */
const RINSHAN_ORD: readonly number[] = [1, 0, 3, 2];

/** Post-deal draws available in a round (the 122-tile live wall minus 52 dealt). */
export const DRAWS_PER_ROUND = 70;

export class Wall {
  readonly tiles: Tile[];

  /** Number of live-wall tiles taken so far, including the 52 dealt. */
  private liveTaken = 0;
  /** Rinshan draws taken so far (== number of kans). */
  private kans = 0;
  /** Dora indicators revealed so far (>= 1 after the deal). */
  private indicators = 0;
  /** Post-deal draws, live + rinshan. */
  private drawn = 0;

  constructor(tiles: Tile[]) {
    if (tiles.length !== 136) {
      throw new Error(`wall must be 136 tiles, got ${tiles.length}`);
    }
    this.tiles = tiles;
  }

  /**
   * Fisher-Yates in the same direction as tenhou.cc:84-86 —
   * `for i in 0..134: swap(yama[i], yama[i + rnd[i] % (136 - i)])`.
   */
  static shuffled(rng: Rng): Wall {
    const t: Tile[] = new Array(136);
    for (let i = 0; i < 136; i++) t[i] = i;
    for (let i = 0; i < 135; i++) {
      const j = i + rng.int(136 - i);
      const tmp = t[i];
      t[i] = t[j];
      t[j] = tmp;
    }
    return new Wall(t);
  }

  /** Two dice, as Tenhou rolls them (rnd[135] % 6, rnd[136] % 6). */
  static roll(rng: Rng): [number, number] {
    return [rng.int(6), rng.int(6)];
  }

  // -------------------------------------------------------------------------

  /** Tiles left to draw before an exhaustive draw. */
  get remaining(): number {
    return DRAWS_PER_ROUND - this.drawn;
  }

  get exhausted(): boolean {
    return this.remaining <= 0;
  }

  get kanCount(): number {
    return this.kans;
  }

  get indicatorCount(): number {
    return this.indicators;
  }

  /**
   * Deal 13 tiles to each seat, in Tenhou's order: three blocks of four to
   * dealer, dealer+1, dealer+2, dealer+3, then one more each in the same order.
   * (Mirrors MjLogCtrl -startHand: in mjlog.cc:112-119.)
   */
  deal(dealer: Seat): Tile[][] {
    const hands: Tile[][] = [[], [], [], []];
    for (let block = 0; block < 3; block++) {
      for (let k = 0; k < 4; k++) {
        const seat = (dealer + k) % 4;
        for (let n = 0; n < 4; n++) hands[seat].push(this.takeLive());
      }
    }
    for (let k = 0; k < 4; k++) {
      hands[(dealer + k) % 4].push(this.takeLive());
    }
    return hands;
  }

  /** A normal draw from the live wall. */
  draw(): Tile {
    if (this.exhausted) throw new Error("live wall exhausted");
    this.drawn++;
    return this.takeLive();
  }

  /** A replacement draw after a kan. */
  drawRinshan(): Tile {
    if (this.kans >= 4) throw new Error("fifth kan");
    if (this.exhausted) throw new Error("live wall exhausted");
    this.drawn++;
    return this.tiles[RINSHAN_ORD[this.kans++]];
  }

  /** Reveal the next dora indicator (the first one is the round's INIT dora). */
  revealIndicator(): Tile {
    if (this.indicators >= 5) throw new Error("sixth dora indicator");
    return this.tiles[5 + 2 * this.indicators++];
  }

  /**
   * The tile the (k+1)-th FUTURE live draw would return, without taking it:
   * `peekLive(0)` is what the very next `draw()` yields. Null once `k` runs past
   * what the round still has to give.
   *
   * LIVE ONLY — rinshan draws come out of the dead wall (`RINSHAN_ORD`), so a
   * kan does not move this sequence along; it only shortens it (`remaining`
   * counts live and rinshan draws together). Callers that want "the tile seat X
   * draws next" must do their own turn arithmetic on top, and own the
   * assumption that no call intervenes.
   *
   * A pure read: no state changes, so an oracle may call it freely.
   *
   * `remaining` already bounds the index away from the dead wall: with L live
   * draws and R rinshan draws taken, `remaining = 70 - L - R` and the index is
   * `83 - L - k`, so `k < remaining` implies `index >= 14 + R`.
   */
  peekLive(k: number): Tile | null {
    if (k < 0 || k >= this.remaining) return null;
    return this.tiles[135 - this.liveTaken - k];
  }

  /** What the next `revealIndicator()` would turn up, or null after the 5th. */
  peekNextIndicator(): Tile | null {
    if (this.indicators >= 5) return null;
    return this.tiles[5 + 2 * this.indicators];
  }

  /** Indicators revealed so far, in order. */
  doraIndicators(): Tile[] {
    const out: Tile[] = [];
    for (let k = 0; k < this.indicators; k++) out.push(this.tiles[5 + 2 * k]);
    return out;
  }

  /** The ura counterparts of the revealed indicators. */
  uraIndicators(): Tile[] {
    const out: Tile[] = [];
    for (let k = 0; k < this.indicators; k++) out.push(this.tiles[4 + 2 * k]);
    return out;
  }

  private takeLive(): Tile {
    const idx = 135 - this.liveTaken++;
    if (idx < 14 + this.kans) throw new Error("live wall underflow");
    return this.tiles[idx];
  }
}
