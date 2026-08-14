// Match (hanchan) progression: kyoku sequence, dealer rotation, 連荘, honba,
// kyotaku carry-over, and the two drivers that turn `runRound` into a game.
//
// `runMatchSync` is the RL/self-play path: no promises anywhere, so a rollout is
// a tight loop. `runMatch` is the interactive path and awaits each policy. They
// share the identical generator, so a bug can never appear in only one of them.

import type { Game, Meld, PlayerInfo, Round } from "mjrender/model.ts";
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
  /**
   * A mutable cell the driver keeps pointed at the round currently in play, so
   * a caller that needs the FULL state (an RL recorder taking oracle features,
   * say) can reach it without the Table being threaded through every policy.
   * Set when a round starts, nulled when the match finishes. Nothing in the
   * engine reads it back — it is a one-way tap.
   */
  tableRef?: { t: Table | null };
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
  /**
   * Per seat (absolute), the number of rounds in which that seat's riichi was
   * live at the end of the round — read from `Table.riichi`, the engine's own
   * accepted-declaration flag, not by counting `riichi` events (a step-1
   * declaration that never completes must not count).
   *
   * Optional only so older `MatchResult` literals keep typechecking; every
   * match `runMatch`/`runMatchSync` finishes populates it (length 4).
   */
  riichis?: number[];
  /**
   * Per seat (absolute), the number of rounds in which the seat held at least
   * one OPEN meld (chi/pon/daiminkan/shouminkan). Ankan leaves a hand
   * concealed, so it is excluded — same test as `Table.isMenzen`. Optional on
   * the same terms as `riichis`.
   */
  furoRounds?: number[];
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
  riichis: number[];
  furoRounds: number[];
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
    riichis: SEATS.map(() => 0),
    furoRounds: SEATS.map(() => 0),
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

/**
 * Does this meld break concealment? Ankan does not (it is still 門前), and
 * `nuki` cannot occur in a 4-player game. Mirrors `Table.isMenzen`.
 */
function isOpenMeld(m: Meld): boolean {
  return m.kind !== "ankan" && m.kind !== "nuki";
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
  // Same breath, same reason: `t` is the only witness to riichi/meld state, and
  // it is discarded the moment this function returns.
  for (const s of SEATS) {
    if (t.riichi[s]) m.riichis[s]++;
    if (t.melds[s].some(isOpenMeld)) m.furoRounds[s]++;
  }
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

/**
 * Who takes the 供託 left on the table when the match ends: the top finisher,
 * ties broken by the lower seat index. That is the same tiebreak the two
 * settlement paths use — `settlement` (src/rl/record.ts) and the TUI's
 * `finalStandings` (src/tui/app.ts) both sort by `b.s - a.s || a.seat - b.seat`
 * — so the sticks can never land on a seat those two rank second.
 */
export function topFinisher(scores: readonly number[]): Seat {
  let top: Seat = 0;
  for (const s of SEATS) if (scores[s] > scores[top]) top = s;
  return top;
}

function finalize(m: MatchState, opts: MatchOptions): MatchResult {
  // The last round's Table is history now; leaving it reachable would let a
  // recorder read a finished table as if it were live.
  if (opts.tableRef) opts.tableRef.t = null;
  // 供託の残り: a match that ends on a drawn round leaves riichi sticks on the
  // table, and `advance` faithfully carries them (a win sweeps them to 0, a draw
  // rolls them over) — with no next round to collect them. The Tenhou convention
  // is that the top finisher takes them; without this the four scores no longer
  // sum to the 100,000 that went in.
  if (m.kyotaku > 0) {
    m.scores[topFinisher(m.scores)] += m.kyotaku * 1000;
    m.kyotaku = 0;
  }
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
    riichis: m.riichis,
    furoRounds: m.furoRounds,
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
        if (opts.tableRef) opts.tableRef.t = t;
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
        if (opts.tableRef) opts.tableRef.t = t;
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
