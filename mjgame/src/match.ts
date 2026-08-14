// Match (hanchan) progression: kyoku sequence, dealer rotation, 連荘, honba,
// kyotaku carry-over, and the two drivers that turn `runRound` into a game.
//
// `runMatchSync` is the RL/self-play path: no promises anywhere, so a rollout is
// a tight loop. `runMatch` is the interactive path and awaits each policy. They
// share the identical generator, so a bug can never appear in only one of them.

import type { Game, PlayerInfo, Round } from "mjrender/model.ts";
import { observe } from "./observe.ts";
import type { Policy, SyncPolicy } from "./policy.ts";
import type { Rng } from "./rng.ts";
import { sfc32 } from "./rng.ts";
import { type RoundDeps, runRound } from "./round.ts";
import type { Table } from "./table.ts";
import type { Action, PublicEvent, RoundOutcome, Seat, Violation } from "./types.ts";
import { SEATS } from "./types.ts";
import { Wall } from "./wall.ts";

export interface MatchOptions extends RoundDeps {
  seed: number;
  players?: PlayerInfo[];
  /** Hard cap on kyoku played, so a bug cannot hang a self-play run. */
  maxRounds?: number;
}

export interface MatchResult {
  seed: number;
  scores: number[];
  rounds: Round[];
  outcomes: RoundOutcome[];
  ledger: Violation[];
  /**
   * Round boundaries into `ledger`: `ledgerCuts[k]` is the ledger length once
   * round `k` finished, so round `k` owns `ledger[ledgerCuts[k-1] ?? 0 ..
   * ledgerCuts[k])`. Non-decreasing, one entry per outcome, last === ledger
   * length. `kyoku` alone cannot do this — 連荘 repeats the same kyoku number.
   */
  ledgerCuts: number[];
  game: Game;
}

interface MatchState {
  kyoku: number;
  honba: number;
  kyotaku: number;
  scores: number[];
  rounds: Round[];
  outcomes: RoundOutcome[];
  ledger: Violation[];
  ledgerCuts: number[];
  rng: Rng;
  players: PlayerInfo[];
  game: Game | null;
}

function initState(opts: MatchOptions): MatchState {
  const players = opts.players ??
    SEATS.map((seat) => ({ seat, name: `P${seat}` }));
  return {
    kyoku: 0,
    honba: 0,
    kyotaku: 0,
    scores: SEATS.map(() => opts.cfg.startScore),
    rounds: [],
    outcomes: [],
    ledger: [],
    ledgerCuts: [],
    rng: sfc32(opts.seed),
    players,
    game: null,
  };
}

function roundInit(m: MatchState) {
  // A per-kyoku substream keeps wall generation independent of how many random
  // numbers the policies consumed, so a seed reproduces the same walls exactly.
  const r = m.rng.fork(m.kyoku * 64 + m.honba);
  return {
    kyoku: m.kyoku,
    honba: m.honba,
    kyotaku: m.kyotaku,
    dealer: (m.kyoku % 4) as Seat,
    scores: [...m.scores],
    wall: Wall.shuffled(r),
    dice: Wall.roll(r) as [number, number],
  };
}

/** Fold a finished round back into the match, and report whether to continue. */
function advance(m: MatchState, t: Table, outcome: RoundOutcome, opts: MatchOptions): boolean {
  m.scores = [...t.scores];
  m.rounds.push(t.round);
  m.outcomes.push(outcome);
  m.ledger.push(...t.ledger);
  // Snapshot the boundary in the same breath as the outcome: this is the only
  // place the two can be kept in lockstep, and it is what lets a recorder charge
  // a 罰符 to the round that earned it rather than to the whole match.
  m.ledgerCuts.push(m.ledger.length);
  if (!m.game) m.game = t.game;
  else m.game.rounds.push(t.round);

  const dealer = t.dealer;
  if (outcome.kind === "agari") {
    m.kyotaku = 0; // the winner swept the sticks
    m.honba = outcome.dealerRepeat ? m.honba + 1 : 0;
  } else {
    m.kyotaku = t.round.kyotaku;
    m.honba += 1;
  }
  if (!outcome.dealerRepeat) m.kyoku += 1;

  const cfg = opts.cfg;
  const lastKyoku = cfg.hanchan ? 7 : 3;

  if (m.kyoku > lastKyoku) return false; // no 西入
  if (m.rounds.length >= (opts.maxRounds ?? 64)) return false;

  // アガリやめ: the dealer wins オーラス while leading, and the match ends.
  if (
    cfg.agariYame && outcome.kind === "agari" && outcome.dealerRepeat &&
    m.kyoku - (outcome.dealerRepeat ? 0 : 1) > lastKyoku - 1 &&
    m.scores[dealer] === Math.max(...m.scores) &&
    m.scores[dealer] >= cfg.returnScore
  ) {
    return false;
  }
  return true;
}

function finalize(m: MatchState, opts: MatchOptions): MatchResult {
  const game = m.game ?? {
    version: "2.3",
    rules: {
      raw: 0,
      aka: opts.cfg.akaIds.size > 0,
      kuitan: opts.cfg.kuitan,
      sanma: false,
      hanchan: opts.cfg.hanchan,
    },
    players: m.players,
    rounds: m.rounds,
  };
  return {
    seed: opts.seed,
    scores: m.scores,
    rounds: m.rounds,
    outcomes: m.outcomes,
    ledger: m.ledger,
    ledgerCuts: m.ledgerCuts,
    game,
  };
}

// ---------------------------------------------------------------------------

/** Fast path: every policy must be synchronous. */
export function runMatchSync(policies: SyncPolicy[], opts: MatchOptions): MatchResult {
  const m = initState(opts);
  for (;;) {
    let table: Table | null = null;
    const deps = {
      ...opts,
      onTable: (t: Table) => {
        table = t;
        opts.onTable?.(t);
      },
    };
    const gen = runRound(roundInit(m), deps, m.players);
    let reply: Action | undefined;
    for (;;) {
      const step = reply === undefined ? gen.next() : gen.next(reply);
      if (step.done) {
        if (!advance(m, step.value.table, step.value.outcome, opts)) return finalize(m, opts);
        break;
      }
      const req = step.value;
      const obs = observe(
        table!,
        req.seat,
        req.legal,
        req.k === "turn" ? req.drawn : null,
        opts.scorer,
        req.k === "claim" ? req.tile : null,
      );
      reply = policies[req.seat].decide(obs);
    }
  }
}

/** Interactive path: policies may await (the human does). */
export async function runMatch(policies: Policy[], opts: MatchOptions): Promise<MatchResult> {
  const m = initState(opts);
  for (;;) {
    let table: Table | null = null;
    const deps = {
      ...opts,
      onTable: (t: Table) => {
        table = t;
        opts.onTable?.(t);
      },
    };
    const gen = runRound(roundInit(m), deps, m.players);
    let reply: Action | undefined;
    for (;;) {
      const step = reply === undefined ? gen.next() : gen.next(reply);
      if (step.done) {
        if (!advance(m, step.value.table, step.value.outcome, opts)) return finalize(m, opts);
        break;
      }
      const req = step.value;
      const obs = observe(
        table!,
        req.seat,
        req.legal,
        req.k === "turn" ? req.drawn : null,
        opts.scorer,
        req.k === "claim" ? req.tile : null,
      );
      reply = await policies[req.seat].decide(obs);
    }
  }
}

export type { PublicEvent };
