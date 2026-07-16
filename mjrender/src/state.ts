// BoardState: a queryable replay engine for one round.
//
// Owns ALL mutable table state (hands, melds, rivers, visible counts, dora,
// riichi, live scores, wall) and replays GameEvents against it. Two consumers:
//   - render.ts steps it event-by-event while formatting the transcript;
//   - the snapshot/query side replays to an arbitrary position (replayTo) and
//     reads the state off — the transcript and the query tools can never
//     disagree because they share this single replay semantics.

import type { Game, GameEvent, Meld, Round, Tile } from "./model.ts";
import type { RiichiThreat } from "./danger.ts";
import { concealedTilesUsed } from "./meld.ts";
import { countsFromTiles, shanten, ukeireTypes } from "./shanten.ts";
import { doraFromIndicatorType, isAka, tileType } from "./tiles.ts";

/** One discarded tile in a player's river (kawa), in discard order. */
export interface RiverEntry {
  tile: Tile;
  junme: number; // the go-around this discard happened in
  tsumogiri: boolean; // true = drawn tile discarded unchanged; false = tedashi
  riichiDeclare: boolean; // this is the sideways riichi tile
  calledBy?: number; // seat that pon/chi/kan'd this tile away, if any
}

/** Resting (3n+1) hand analysis: shanten + ukeire kinds / live-tile count. */
export interface RestInfo {
  shanten: number;
  kinds: number;
  count: number;
  types: number[];
}

/** Number of live-wall draws in a 4-player game: 136 − 13*4 deal − 14 dead wall. */
const LIVE_WALL = 70;

export class BoardState {
  readonly game: Game;
  readonly round: Round;

  // per seat
  readonly hands: Tile[][];
  readonly melds: Meld[][] = [[], [], [], []];
  readonly rivers: RiverEntry[][] = [[], [], [], []];
  readonly discardTypes: Array<Set<number>> = [new Set(), new Set(), new Set(), new Set()];
  readonly riichiActive = [false, false, false, false];
  readonly riichiJunme = [-1, -1, -1, -1];
  /** Tile types 100% safe vs each seat's riichi (their genbutsu set). */
  readonly safe: Array<Set<number>> = [new Set(), new Set(), new Set(), new Set()];
  readonly lastDraw = [-1, -1, -1, -1];
  /** Post-action shanten per seat, kept current by discard()/applyMeld() callers. */
  readonly restShanten = [0, 0, 0, 0];

  // table
  /** Copies of each tile type visible to everyone (discards, melds, indicators).
   *  NOTE: a called discard is currently counted twice (once by the discard,
   *  once by the meld) — a pre-existing quirk preserved for output stability. */
  readonly publicVisible = new Array<number>(34).fill(0);
  readonly indicators: Tile[];
  readonly doraCount = new Int8Array(34);
  readonly doraTypeSet = new Set<number>();
  /** Live scores in units of 100 (riichi sticks debited as declared). */
  readonly scores: number[];
  /** Riichi sticks actually placed this round (REACH step 2 — a declaration
   *  whose tile is ronned never places its stick). */
  sticksPlaced = 0;
  junme = 0;
  /** Live-wall tiles not yet drawn (each kan shifts one tile into the dead wall,
   *  so rinshan draws consume from this budget too). */
  wallRemaining = LIVE_WALL;

  /** Round wind as a tile type (東=27..北=30). */
  readonly roundWindType: number;
  /** Per seat: honor types that are yakuhai for it (round wind + seat wind + dragons). */
  readonly valueHonorsBySeat: ReadonlyArray<ReadonlySet<number>>;

  constructor(game: Game, round: Round) {
    this.game = game;
    this.round = round;
    this.hands = round.startHands.map((h) => [...h]);
    this.scores = [...round.startScores];
    this.roundWindType = 27 + (Math.floor(round.kyoku / 4) % 4);
    this.valueHonorsBySeat = [0, 1, 2, 3].map((s) =>
      new Set<number>([this.roundWindType, this.seatWindType(s), 31, 32, 33])
    );
    this.indicators = [round.firstDora];
    this.bumpVisible(round.firstDora);
    this.addDora(round.firstDora);
  }

  /** Seat wind as a tile type (東=27..北=30), from the seat's offset to the dealer. */
  seatWindType(seat: number): number {
    return 27 + ((seat - this.round.dealer + 4) % 4);
  }

  private bumpVisible(id: Tile): void {
    this.publicVisible[tileType(id)]++;
  }

  private addDora(id: Tile): void {
    const dt = doraFromIndicatorType(tileType(id));
    this.doraCount[dt]++;
    this.doraTypeSet.add(dt);
  }

  // ---- mutators (one per event kind; applyEvent dispatches for the query path) ----

  draw(who: number, tile: Tile, rinshan: boolean): void {
    if (!rinshan && who === this.round.dealer) this.junme++;
    this.wallRemaining--;
    this.hands[who].push(tile);
    this.lastDraw[who] = tile;
  }

  /** Apply a discard. Returns false when the tile wasn't in hand (corrupt log). */
  discard(who: number, tile: Tile, tsumogiri: boolean, riichi: boolean): boolean {
    const di = this.hands[who].indexOf(tile);
    if (di >= 0) this.hands[who].splice(di, 1);
    this.rivers[who].push({ tile, junme: this.junme, tsumogiri, riichiDeclare: riichi });
    this.discardTypes[who].add(tileType(tile));
    this.bumpVisible(tile);
    if (riichi) {
      this.riichiActive[who] = true;
      this.riichiJunme[who] = this.junme;
      this.safe[who] = new Set(this.discardTypes[who]);
    }
    for (let s = 0; s < 4; s++) {
      if (this.riichiActive[s]) this.safe[s].add(tileType(tile));
    }
    return di >= 0;
  }

  applyMeld(m: Meld): void {
    if (m.kind === "shouminkan") {
      // upgrade the existing pon of this type to a kan
      const tt = tileType(m.calledTile);
      const idx = this.melds[m.who].findIndex(
        (x) => x.kind === "pon" && tileType(x.tiles[0]) === tt,
      );
      if (idx >= 0) this.melds[m.who].splice(idx, 1);
      const i = this.hands[m.who].indexOf(m.calledTile);
      if (i >= 0) this.hands[m.who].splice(i, 1);
      this.melds[m.who].push(m);
    } else {
      for (const t of concealedTilesUsed(m)) {
        const i = this.hands[m.who].indexOf(t);
        if (i >= 0) this.hands[m.who].splice(i, 1);
      }
      this.melds[m.who].push(m);
      // mark the called tile in the discarder's river (chi/pon/daiminkan take it)
      if (m.fromWho !== m.who) {
        const river = this.rivers[m.fromWho];
        const last = river[river.length - 1];
        if (last && last.tile === m.calledTile) last.calledBy = m.who;
      }
    }
    for (const t of m.tiles) this.publicVisible[tileType(t)]++;
    this.restShanten[m.who] = shanten(
      countsFromTiles(this.hands[m.who]),
      this.melds[m.who].length,
      false,
    );
  }

  reach(who: number, step: 1 | 2, scores?: number[]): void {
    // step 1 is the declaration (the flagged discard carries the state change);
    // step 2 places the 1000-point stick — the log's `ten` is authoritative.
    if (step === 2) {
      this.sticksPlaced++;
      if (scores && scores.length === 4) {
        for (let s = 0; s < 4; s++) this.scores[s] = scores[s];
      } else {
        this.scores[who] -= 10;
      }
    }
  }

  revealDora(indicator: Tile): void {
    this.indicators.push(indicator);
    this.bumpVisible(indicator);
    this.addDora(indicator);
  }

  applyEvent(e: GameEvent): void {
    switch (e.t) {
      case "draw":
        this.draw(e.who, e.tile, e.rinshan);
        break;
      case "discard":
        this.discard(e.who, e.tile, e.tsumogiri, e.riichi);
        this.restShanten[e.who] = shanten(
          countsFromTiles(this.hands[e.who]),
          this.melds[e.who].length,
          this.melds[e.who].length === 0,
        );
        break;
      case "call":
        this.applyMeld(e.meld);
        break;
      case "reach":
        this.reach(e.who, e.step, e.scores);
        break;
      case "dora":
        this.revealDora(e.indicator);
        break;
    }
  }

  // ---- queries ----

  /** Resting (3n+1) hand analysis for a seat, against live public information. */
  restInfo(seat: number): RestInfo {
    const counts = countsFromTiles(this.hands[seat]);
    const open = this.melds[seat].length;
    const closed = open === 0;
    const s = shanten(counts, open, closed);
    const types = ukeireTypes(counts, open, closed);
    let total = 0;
    for (const t of types) total += Math.max(0, 4 - this.publicVisible[t] - counts[t]);
    return { shanten: s, kinds: types.length, count: total, types };
  }

  /** Dora tiles in a seat's hand + melds (indicator multiplicity + aka). */
  countDora(seat: number): number {
    const aka = this.game.rules.aka;
    let n = 0;
    const tally = (id: Tile) => {
      n += this.doraCount[tileType(id)];
      if (aka && isAka(id)) n++;
    };
    for (const id of this.hands[seat]) tally(id);
    for (const m of this.melds[seat]) for (const id of m.tiles) tally(id);
    return n;
  }

  /** Active riichi threats against `who`'s next discard. */
  threats(who: number): RiichiThreat[] {
    const out: RiichiThreat[] = [];
    for (let s = 0; s < 4; s++) {
      if (s !== who && this.riichiActive[s]) {
        out.push({
          seat: s,
          safeTypes: this.safe[s],
          valueHonors: this.valueHonorsBySeat[s] as Set<number>,
        });
      }
    }
    return out;
  }
}

/** Position within a round: an event index (inclusive) or a go-around number. */
export type RoundPosition = { eventIndex: number } | { junme: number };

/**
 * Replay a round up to `pos` and return the resulting state.
 * - `{eventIndex: k}` applies events[0..k] inclusive (-1 = the deal only).
 * - `{junme: j}` applies everything through the END of go-around j (stops right
 *   before the dealer draw that would begin j+1).
 * Out-of-range positions clamp to the full round.
 */
export function replayTo(game: Game, round: Round, pos: RoundPosition): BoardState {
  const st = new BoardState(game, round);
  const ev = round.events;
  let stop = ev.length;
  for (let i = 0; i < ev.length; i++) {
    const e = ev[i];
    if ("eventIndex" in pos && i > pos.eventIndex) {
      stop = i;
      break;
    }
    if (
      "junme" in pos && e.t === "draw" && !e.rinshan && e.who === round.dealer &&
      st.junme >= pos.junme
    ) {
      stop = i;
      break;
    }
    st.applyEvent(e);
  }
  // Absorb trailing bookkeeping: a riichi discard's stick placement (REACH
  // step 2) follows the discard event, and a position pointing at the discard
  // should already see the 1000 debited (the stick is shown as 供託).
  for (let i = stop; i < ev.length; i++) {
    const e = ev[i];
    if (e.t === "reach" && e.step === 2) st.applyEvent(e);
    else break;
  }
  return st;
}
