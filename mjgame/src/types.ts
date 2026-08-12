// Engine-level types: seats, actions, requests, and the public event stream.
//
// Deliberately distinct from mjrender's `GameEvent` (src/model.ts), which is the
// *replay log* format and records every drawn tile. `PublicEvent` is what a
// policy is allowed to see, so it hides other seats' draws.

import type { Meld, Tile } from "mjrender/model.ts";

export type Seat = 0 | 1 | 2 | 3;

export const SEATS: readonly Seat[] = [0, 1, 2, 3];

/** Seat `from` viewed relative to `self`: 0=self, 1=shimocha, 2=toimen, 3=kamicha. */
export function relSeat(self: Seat, from: Seat): 0 | 1 | 2 | 3 {
  return ((from - self + 4) % 4) as 0 | 1 | 2 | 3;
}

export function nextSeat(s: Seat): Seat {
  return ((s + 1) % 4) as Seat;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/**
 * A player decision. Note that dojo 禁じ手 (daiminkan, kakan, first-turn honor
 * discards, non-tenpai dora cuts) are all *representable and legal* here — they
 * are recorded by the penalty ledger after the fact, never blocked. See
 * `legal.ts` for the narrow set of things that genuinely are illegal.
 */
export type Action =
  | { t: "discard"; tile: Tile; riichi: boolean; tsumogiri: boolean }
  | { t: "chi"; tiles: [Tile, Tile]; called: Tile }
  | { t: "pon"; tiles: [Tile, Tile]; called: Tile }
  | { t: "daiminkan"; called: Tile }
  | { t: "kakan"; tile: Tile }
  | { t: "ankan"; type: number }
  | { t: "ron" }
  | { t: "tsumo" }
  | { t: "pass" };

export type ActionKind = Action["t"];

/** A request from the game master for one seat to choose among `legal`. */
export type Request =
  | { k: "turn"; seat: Seat; drawn: Tile | null; legal: Action[] }
  | { k: "claim"; seat: Seat; tile: Tile; from: Seat; legal: Action[] };

// ---------------------------------------------------------------------------
// Violations (dojo penalty ledger)
// ---------------------------------------------------------------------------

export type RuleTier = "A" | "B";

export interface Violation {
  rule: string;
  label: string; // Japanese, for the ledger UI
  seat: Seat;
  kyoku: number;
  junme: number;
  points: number; // 評価点マイナス (positive magnitude)
  tier: RuleTier;
  confidence: number; // 1.0 for Tier A; < 1 for approximations
  detail: string; // human-readable evidence
  sanction?: "和了放棄" | "振込禁止";
}

// ---------------------------------------------------------------------------
// Public event stream
// ---------------------------------------------------------------------------

export type DrawKind =
  | "exhaustive" // 荒牌平局
  | "sanchahou" // 三家和
  | "suukaikan" // 四開槓
  | "suucha-riichi" // 四人立直
  | "nagashi"; // 流し満貫

export type PublicEvent =
  | {
    e: "deal";
    kyoku: number;
    honba: number;
    kyotaku: number;
    dealer: Seat;
    scores: number[];
    indicator: Tile;
  }
  | { e: "draw"; who: Seat; rinshan: boolean; tile: Tile | null } // tile non-null only for `who`
  | { e: "discard"; who: Seat; tile: Tile; tsumogiri: boolean; riichi: boolean }
  | { e: "call"; meld: Meld }
  | { e: "riichi"; who: Seat; step: 1 | 2 }
  | { e: "dora"; indicator: Tile }
  | { e: "violation"; v: Violation }
  | { e: "result"; outcome: RoundOutcome };

// ---------------------------------------------------------------------------
// Outcomes
// ---------------------------------------------------------------------------

export interface WinInfo {
  who: Seat;
  fromWho: Seat; // === who ⇒ tsumo
  winTile: Tile;
  han: number;
  fu: number;
  points: number; // base points before honba/kyotaku
  limit: number; // 0 normal, 1 mangan, 2 haneman, 3 baiman, 4 yakuman
  yaku: Array<{ id: number; han: number }>;
  yakuman: number[];
  doraIndicators: Tile[];
  uraIndicators: Tile[];
  hand: Tile[];
  melds: Meld[];
}

export type RoundOutcome =
  | { kind: "agari"; wins: WinInfo[]; deltas: number[]; dealerRepeat: boolean }
  | {
    kind: "ryuukyoku";
    draw: DrawKind;
    tenpai: boolean[];
    /** Hands revealed at an exhaustive draw — Tenhou XML records these. */
    tenpaiHands: Array<{ who: Seat; hand: Tile[] }>;
    deltas: number[];
    dealerRepeat: boolean;
  };
