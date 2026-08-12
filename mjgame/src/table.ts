// Authoritative table state for one round, with mjrender's BoardState as a
// derived mirror.
//
// Division of labour, deliberately non-overlapping:
//   BoardState  — hands, melds, rivers, discard sets, genbutsu, dora, junme,
//                 threats/furoThreats. It replays the same GameEvents we log,
//                 so it can never disagree with the transcript we export.
//   Table       — everything BoardState has no concept of: the actual wall
//                 (tile identity, not just a counter), full-point scores,
//                 ippatsu/double-riichi, three-valued furiten, the 見せ牌/腰
//                 ron blocks, 和了放棄 sanctions, and the violation ledger.
//
// Hands are NOT duplicated: `board.hands` is the single copy. Both objects are
// driven from one `emit()`, so there is no second source of truth to drift.

import type { Game, GameEvent, Meld, PlayerInfo, Round, Tile } from "mjrender/model.ts";
import { BoardState } from "mjrender/state.ts";
import type { FuroThreat, RiichiThreat } from "mjrender/danger.ts";
import { tileType } from "mjrender/tiles.ts";
import { zeros34 } from "./tiles.ts";
import type { RuleConfig } from "./rules.ts";
import type { PublicEvent, Seat, Violation } from "./types.ts";
import { SEATS } from "./types.ts";
import { Wall } from "./wall.ts";

/** Furiten is three-valued; mjrender's `isFuriten` covers only `permanent`. */
export interface Furiten {
  /** A wait appears in this seat's own discards. */
  permanent: boolean;
  /** Passed on a ron this go-around; clears on the seat's next draw. */
  temporary: boolean;
  /** Passed on a ron after declaring riichi — locked for the rest of the round. */
  riichi: boolean;
}

export function anyFuriten(f: Furiten): boolean {
  return f.permanent || f.temporary || f.riichi;
}

export interface RoundInit {
  kyoku: number;
  honba: number;
  kyotaku: number;
  dealer: Seat;
  scores: number[]; // full points
  wall: Wall;
  dice: [number, number];
}

export type EventSink = (e: PublicEvent) => void;

export class Table {
  readonly cfg: RuleConfig;
  readonly wall: Wall;
  readonly round: Round;
  readonly board: BoardState;
  readonly game: Game;

  /** Full points. `board.scores` mirrors this in units of 100. */
  readonly scores: number[];

  readonly riichi = [false, false, false, false];
  readonly doubleRiichi = [false, false, false, false];
  readonly ippatsu = [false, false, false, false];
  readonly furiten: Furiten[] = SEATS.map(() => ({
    permanent: false,
    temporary: false,
    riichi: false,
  }));

  /** Tile types this seat may not 出和了 on (見せ牌 exposes a suit, 腰 a tile+suji). */
  readonly ronBlocked: Array<Set<number>> = SEATS.map(() => new Set<number>());
  /** Tile ids this seat has exposed (空ポン/空チー), driving the 見せ牌 block. */
  readonly exposed: Array<Set<Tile>> = SEATS.map(() => new Set<Tile>());
  /** 和了放棄: may not win or call for the rest of the round. */
  readonly sanctioned = [false, false, false, false];
  /** Set when a pre-tenpai dora discard was called; the seat must tsumogiri after. */
  readonly tsumogiriLock = [false, false, false, false];

  readonly ledger: Violation[] = [];

  /**
   * Exact count of completed turns, unlike `board.junme` which advances on the
   * dealer's draw and so drifts when calls skip seats. Rule predicates that
   * must be exact (第一打, 同巡) use this; the UI shows board.junme for
   * consistency with mjrender transcripts.
   */
  turnIndex = 0;
  /** True while no call and no discard other than the opening go-around has
   *  happened — the window for 天和/地和/ダブリー. */
  firstTurnIntact = true;
  /** Kans declared this round, across all seats (四開槓 watch). */
  kanTotal = 0;
  /** Tile types the seat that just called may not immediately discard
   *  (同巡内食い替え禁止). Cleared by that seat's discard. */
  kuikaeBan: { seat: Seat; types: Set<number> } | null = null;

  private sink: EventSink;

  constructor(init: RoundInit, cfg: RuleConfig, players: PlayerInfo[], sink: EventSink = () => {}) {
    this.cfg = cfg;
    this.wall = init.wall;
    this.scores = [...init.scores];
    this.sink = sink;

    const hands = init.wall.deal(init.dealer);
    const firstDora = init.wall.revealIndicator();

    this.round = {
      kyoku: init.kyoku,
      honba: init.honba,
      kyotaku: init.kyotaku,
      dealer: init.dealer,
      dice: init.dice,
      startScores: init.scores.map((s) => s / 100),
      startHands: hands,
      firstDora,
      events: [],
      results: [],
    };
    this.game = {
      version: "2.3",
      rules: {
        raw: 0,
        aka: cfg.akaIds.size > 0,
        kuitan: cfg.kuitan,
        sanma: false,
        hanchan: cfg.hanchan,
      },
      players,
      rounds: [this.round],
    };
    this.board = new BoardState(this.game, this.round);

    this.emitPublic({
      e: "deal",
      kyoku: init.kyoku,
      honba: init.honba,
      kyotaku: init.kyotaku,
      dealer: init.dealer,
      scores: [...init.scores],
      indicator: firstDora,
    });
  }

  // --- shorthand accessors onto the mirror -----------------------------------

  get hands(): Tile[][] {
    return this.board.hands;
  }
  get melds(): Meld[][] {
    return this.board.melds;
  }
  get junme(): number {
    return this.board.junme;
  }
  get dealer(): Seat {
    return this.round.dealer as Seat;
  }
  get kyoku(): number {
    return this.round.kyoku;
  }

  isMenzen(seat: Seat): boolean {
    return this.melds[seat].every((m) => m.kind === "ankan");
  }

  seatWindType(seat: Seat): number {
    return this.board.seatWindType(seat);
  }
  get roundWindType(): number {
    return this.board.roundWindType;
  }
  valueHonors(seat: Seat): ReadonlySet<number> {
    return this.board.valueHonorsBySeat[seat];
  }

  threats(seat: Seat): RiichiThreat[] {
    return this.board.threats(seat);
  }
  furoThreats(seat: Seat): FuroThreat[] {
    return this.board.furoThreats(seat);
  }

  /** Dora indicators revealed so far (index 0 is the round's opening dora). */
  get indicators(): Tile[] {
    return this.board.indicators;
  }

  // --- event plumbing --------------------------------------------------------

  /** Record a game event: append to the replay log, advance the mirror, notify. */
  emit(g: GameEvent, p: PublicEvent): void {
    this.round.events.push(g);
    this.board.applyEvent(g);
    this.emitPublic(p);
  }

  emitPublic(p: PublicEvent): void {
    this.sink(p);
  }

  addViolation(v: Violation): void {
    this.ledger.push(v);
    this.emitPublic({ e: "violation", v });
  }

  /** Keep `board.scores` (units of 100) in step after a settlement. */
  syncScores(): void {
    for (const s of SEATS) this.board.scores[s] = this.scores[s] / 100;
  }

  // --- visibility ------------------------------------------------------------

  /**
   * Copies of each tile type `seat` can see: every uncalled river tile, every
   * meld tile, every revealed indicator, and its own concealed hand.
   *
   * This deliberately replaces `board.publicVisible`, which double-counts a
   * called discard (mjrender/src/state.ts:54-56 — a quirk preserved there for
   * golden-output stability). Danger evidence and ukeire live-counts both read
   * from here instead, so the quirk never reaches mjgame.
   */
  visibleCounts(seat: Seat): number[] {
    const c = zeros34();
    for (const s of SEATS) {
      for (const r of this.board.rivers[s]) {
        if (r.calledBy === undefined) c[tileType(r.tile)]++;
      }
      for (const m of this.melds[s]) {
        for (const t of m.tiles) c[tileType(t)]++;
      }
    }
    for (const ind of this.board.indicators) c[tileType(ind)]++;
    for (const t of this.hands[seat]) c[tileType(t)]++;
    return c;
  }

  /** Dora value of a seat's hand+melds, counting indicators and aka. */
  countDora(seat: Seat): number {
    let n = 0;
    const all = [...this.hands[seat], ...this.melds[seat].flatMap((m) => m.tiles)];
    for (const t of all) {
      n += this.board.doraCount[tileType(t)];
      if (this.cfg.akaIds.has(t)) n++;
    }
    return n;
  }

  // --- furiten ---------------------------------------------------------------

  clearTemporaryFuriten(seat: Seat): void {
    this.furiten[seat].temporary = false;
  }

  /** Called when a seat declines (or is unable to take) a ron it was offered. */
  markPassedRon(seat: Seat): void {
    this.furiten[seat].temporary = true;
    if (this.riichi[seat]) this.furiten[seat].riichi = true;
  }

  /** Recompute permanent furiten from the seat's own discards vs its waits. */
  refreshPermanentFuriten(seat: Seat, waits: readonly number[]): void {
    const mine = this.board.discardTypes[seat];
    this.furiten[seat].permanent = waits.some((w) => mine.has(w));
  }

  /** Ippatsu survives only until the declarer's next draw, and dies on any call. */
  breakIppatsu(seat?: Seat): void {
    if (seat === undefined) {
      for (const s of SEATS) this.ippatsu[s] = false;
    } else {
      this.ippatsu[seat] = false;
    }
  }
}
