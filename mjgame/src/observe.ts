// The imperfect-information projection a policy sees.
//
// Every per-seat array is indexed RELATIVE to the observing seat
// (0 = self, 1 = shimocha, 2 = toimen, 3 = kamicha). That is what lets one
// trained policy play all four seats without learning a seat embedding.

import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import type { DangerAssessment } from "mjrender/danger.ts";
import { assessDanger } from "mjrender/danger.ts";
import { countsFromTiles, shanten, ukeireTypes } from "mjrender/shanten.ts";
import { tileType } from "mjrender/tiles.ts";
import type { Furiten } from "./table.ts";
import { Table } from "./table.ts";
import type { Action, Seat } from "./types.ts";
import { relSeat, SEATS } from "./types.ts";

export interface Ukeire {
  type: number;
  live: number; // copies not visible to this seat
}

export interface Observation {
  seat: Seat;
  kyoku: number;
  honba: number;
  kyotaku: number;
  junme: number;
  wallRemaining: number;

  hand: Tile[];
  drawn: Tile | null;
  /** Relative index 0..3. */
  melds: Meld[][];
  rivers: RiverEntry[][];
  scores: number[];
  riichi: boolean[];
  riichiJunme: number[];

  doraIndicators: Tile[];
  seatWind: number;
  roundWind: number;
  akaIds: ReadonlySet<Tile>;

  shanten: number;
  waits: number[];
  ukeire: Ukeire[];
  doraCount: number;
  furiten: Furiten;

  /** Per candidate discard *type*, computed lazily-ish on construction. */
  danger: Map<number, DangerAssessment>;

  violations: number[]; // relative counts
  legal: Action[];
}

export function observe(t: Table, seat: Seat, legal: Action[], drawn: Tile | null): Observation {
  const rel = <T>(pick: (s: Seat) => T): T[] => SEATS.map((s) => pick(((seat + s) % 4) as Seat));

  const hand = [...t.hands[seat]];
  const open = t.melds[seat].length;
  const closed = t.isMenzen(seat);
  const visible = t.visibleCounts(seat);

  // Resting-hand analysis is done on the 3n+1 shape: drop the drawn tile if we
  // are mid-turn, so shanten/ukeire mean "after discarding nothing yet".
  const resting = drawn === null ? hand : hand.filter((_, i) => i !== hand.lastIndexOf(drawn));
  const counts = countsFromTiles(resting);
  const sh = shanten(counts, open, closed);
  const waitTypes = sh <= 0 ? ukeireTypes(counts, open, closed, sh) : [];
  const ukeire: Ukeire[] = ukeireTypes(counts, open, closed, sh).map((type) => ({
    type,
    live: Math.max(0, 4 - visible[type]),
  }));

  const threats = t.threats(seat);
  const furo = t.furoThreats(seat);
  const ownCounts = countsFromTiles(hand);
  const danger = new Map<number, DangerAssessment>();
  if (threats.length > 0 || furo.length > 0) {
    for (const ty of new Set(hand.map(tileType))) {
      const d = assessDanger(ty, threats, furo, visible, ownCounts);
      if (d) danger.set(ty, d);
    }
  }

  const vio = SEATS.map((s) => t.ledger.filter((v) => v.seat === ((seat + s) % 4)).length);

  return {
    seat,
    kyoku: t.kyoku,
    honba: t.round.honba,
    kyotaku: t.round.kyotaku,
    junme: t.junme,
    wallRemaining: t.wall.remaining,
    hand,
    drawn,
    melds: rel((s) => t.melds[s]),
    rivers: rel((s) => t.board.rivers[s]),
    scores: rel((s) => t.scores[s]),
    riichi: rel((s) => t.riichi[s]),
    riichiJunme: rel((s) => t.board.riichiJunme[s]),
    doraIndicators: [...t.indicators],
    seatWind: t.seatWindType(seat),
    roundWind: t.roundWindType,
    akaIds: t.cfg.akaIds,
    shanten: sh,
    waits: waitTypes,
    ukeire,
    doraCount: t.countDora(seat),
    furiten: { ...t.furiten[seat] },
    danger,
    violations: vio,
    legal,
  };
}

/** Absolute seat for a relative index in an Observation. */
export function absSeat(obs: Observation, relIndex: number): Seat {
  return ((obs.seat + relIndex) % 4) as Seat;
}

export { relSeat };
