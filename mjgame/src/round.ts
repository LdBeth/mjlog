// The game master for one round (kyoku), as a synchronous generator.
//
// Written as a generator rather than an async loop so that self-play can drive
// it in a promise-free `for(;;) { gen.next(action) }` — millions of RL rollouts
// should not pay a microtask per decision. The TUI wraps this exact generator in
// an async driver that awaits the human; nothing else differs between the two.
//
//   DEAL -> TURN(dealer)
//   TURN:  draw (unless we just called) -> yield "turn" -> act
//            tsumo            -> settle
//            ankan/kakan      -> dora, 槍槓 window, rinshan draw, TURN(same seat)
//            discard          -> CLAIM
//   CLAIM: yield "claim" to each seat with a real choice, then resolve
//            ron  -> settle (頭ハネ; three rons abort instead)
//            call -> TURN(caller, no draw)
//            none -> abortive checks, exhaustion check, TURN(next seat)

import type { GameEvent, PlayerInfo, Tile } from "mjrender/model.ts";
import { tileType } from "mjrender/tiles.ts";
import { isYaochu } from "./tiles.ts";
import type { DojoConfig, RuleConfig } from "./rules.ts";
import {
  buildMeld,
  claimActions,
  kuikaeTypes,
  resolveClaims,
  turnActions,
  type WinOracle,
} from "./legal.ts";
import { type RoundInit, Table } from "./table.ts";
import type {
  Action,
  DrawKind,
  PublicEvent,
  Request,
  RoundOutcome,
  Seat,
  WinInfo,
} from "./types.ts";
import { SEATS } from "./types.ts";

export interface WinFlags {
  tsumo: boolean;
  riichi: boolean;
  doubleRiichi: boolean;
  ippatsu: boolean;
  rinshan: boolean;
  chankan: boolean;
  haitei: boolean;
  houtei: boolean;
  tenhou: boolean;
  chiihou: boolean;
}

/**
 * Everything the round needs from the scorer. Injected so `round.ts` has no
 * dependency on yaku detection — M1 ran against `stubScorer`, M2 swaps in the
 * real one without touching this file.
 */
export interface Scorer extends WinOracle {
  scoreWin(t: Table, who: Seat, fromWho: Seat, winTile: Tile, flags: WinFlags): WinInfo | null;
  tenpaiAtDraw(t: Table, seat: Seat): boolean;
  winDeltas(t: Table, wins: WinInfo[]): number[];
  drawDeltas(t: Table, tenpai: boolean[], kind: DrawKind, nagashi: Seat[]): number[];
}

export interface RoundDeps {
  cfg: RuleConfig;
  dojo: DojoConfig;
  scorer: Scorer;
  /**
   * Called after an action has been APPLIED to the table — the penalty rules
   * ask questions like "is the hand tenpai now?" and "is this the seat's first
   * river tile?", which are only answerable post-application.
   */
  onAction?: (t: Table, seat: Seat, action: Action, drawn: Tile | null) => void;
  /** Called once the round is settled, for end-of-round rules. */
  onRoundEnd?: (t: Table, outcome: RoundOutcome) => void;
  /** Called once, before the first yield — how a driver gets hold of the Table
   *  it needs to build observations from. */
  onTable?: (t: Table) => void;
  sink?: (e: PublicEvent) => void;
}

export interface RoundResult {
  outcome: RoundOutcome;
  table: Table;
}

export function* runRound(
  init: RoundInit,
  deps: RoundDeps,
  players: PlayerInfo[] = defaultPlayers(),
): Generator<Request, RoundResult, Action> {
  const { cfg, scorer } = deps;
  const t = new Table(init, cfg, players, deps.sink);
  deps.onTable?.(t);

  let seat: Seat = init.dealer;
  let drawn: Tile | null = null;
  let rinshanPending = false;
  /** Dora reveal deferred until after the discard (minkan/kakan convention). */
  let deferredDora = false;
  let riichiPending: Seat | null = null;

  const emitDraw = (who: Seat, tile: Tile, rinshan: boolean) => {
    t.emit({ t: "draw", who, tile, rinshan } as GameEvent, {
      e: "draw",
      who,
      rinshan,
      tile,
    });
  };

  const finish = (outcome: RoundOutcome): RoundResult => {
    t.syncScores();
    deps.onRoundEnd?.(t, outcome);
    // Snapshot every seat's hand BEFORE anything downstream can disturb the
    // table: the 局結果 overlay reveals all four hands, not just the winner's.
    // Copies, sorted by tile id — `board.hands` is append-ordered, which would
    // make the reveal jump around between rounds.
    t.emitPublic({
      e: "result",
      outcome,
      hands: SEATS.map((s) => [...t.hands[s]].sort((a, b) => a - b)),
      melds: SEATS.map((s) => [...t.melds[s]]),
    });
    return { outcome, table: t };
  };

  const settleWin = (wins: WinInfo[]): RoundResult => {
    // Win rules (e.g. the 南2局以降 浮き restriction) judge the table as it
    // stood when the hand was declared, so they run before the transfer.
    //
    // On a tsumo the winning tile is passed as `drawn`, because it IS the drawn
    // tile and it is still sitting in the hand: the 後付け/片和了り rules judge
    // the wait the hand held one tile ago, and without it they cannot subtract
    // the win from the hand. On a ron there is nothing to subtract — the tile is
    // in the discarder's river.
    for (const w of wins) {
      const tsumo = w.fromWho === w.who;
      deps.onAction?.(t, w.who, { t: tsumo ? "tsumo" : "ron" }, tsumo ? w.winTile : null);
    }
    const deltas = scorer.winDeltas(t, wins);
    for (const s of SEATS) t.scores[s] += deltas[s];
    const dealerRepeat = wins.some((w) => w.who === init.dealer);
    return finish({ kind: "agari", wins, deltas, dealerRepeat });
  };

  const settleDraw = (kind: DrawKind): RoundResult => {
    const tenpai = SEATS.map((s) => kind === "exhaustive" && scorer.tenpaiAtDraw(t, s));
    const nagashi = kind === "exhaustive" && cfg.nagashiMangan
      ? SEATS.filter((s) => isNagashi(t, s))
      : [];
    const deltas = scorer.drawDeltas(t, tenpai, nagashi.length ? "nagashi" : kind, nagashi);
    for (const s of SEATS) t.scores[s] += deltas[s];
    const dealerRepeat = kind !== "exhaustive"
      ? true
      : cfg.dealerRepeatOnTenpai
      ? tenpai[init.dealer]
      : false;
    const tenpaiHands = SEATS.filter((s) => tenpai[s]).map((s) => ({
      who: s,
      hand: [...t.hands[s]],
    }));
    return finish({ kind: "ryuukyoku", draw: kind, tenpai, tenpaiHands, deltas, dealerRepeat });
  };

  // -------------------------------------------------------------------------

  // The dealer opens by drawing.
  drawn = t.wall.draw();
  emitDraw(seat, drawn, false);

  for (;;) {
    // ---- TURN ----------------------------------------------------------
    const legal = turnActions(t, seat, drawn, scorer, rinshanPending);
    const drawnNow = drawn;
    const action = yield { k: "turn", seat, drawn, legal };

    if (action.t === "tsumo") {
      const flags = winFlags(t, seat, true, {
        rinshan: rinshanPending,
        chankan: false,
        haitei: t.wall.remaining === 0,
        houtei: false,
      });
      const win = scorer.scoreWin(t, seat, seat, drawn!, flags);
      if (win) return settleWin([win]);
    }

    if (action.t === "ankan" || action.t === "kakan") {
      const meld = buildMeld(seat, seat, action, t);
      t.emit({ t: "call", meld }, { e: "call", meld });
      t.kanTotal++;
      t.firstTurnIntact = false;
      t.breakIppatsu();
      deps.onAction?.(t, seat, action, drawnNow);

      // 槍槓: only an added kan can be robbed (an ankan cannot, outside kokushi
      // house rules we do not adopt).
      if (action.t === "kakan") {
        const robbed = yield* claimPhase(t, seat, action.tile, scorer, true);
        if (robbed && robbed.kind === "ron") {
          const wins = robbed.seats
            .map((w) =>
              scorer.scoreWin(
                t,
                w,
                seat,
                action.tile,
                winFlags(t, w, false, {
                  rinshan: false,
                  chankan: true,
                  haitei: false,
                  houtei: false,
                }),
              )
            )
            .filter((w): w is WinInfo => w !== null);
          if (wins.length) return settleWin(wins);
        }
      }

      if (cfg.kanDora) {
        if (action.t === "ankan") {
          const ind = t.wall.revealIndicator();
          t.emit({ t: "dora", indicator: ind }, { e: "dora", indicator: ind });
        } else {
          deferredDora = true;
        }
      }

      if (t.kanTotal >= 4 && cfg.suukaikanDraw && !singleKanner(t)) {
        return settleDraw("suukaikan");
      }

      drawn = t.wall.drawRinshan();
      rinshanPending = true;
      emitDraw(seat, drawn, true);
      continue;
    }

    if (action.t !== "discard") {
      throw new Error(`illegal turn action: ${action.t}`);
    }

    // ---- DISCARD -------------------------------------------------------
    const tile = action.tile;
    if (action.riichi) {
      t.emit({ t: "reach", who: seat, step: 1 }, { e: "riichi", who: seat, step: 1 });
      riichiPending = seat;
      if (t.firstTurnIntact) t.doubleRiichi[seat] = true;
    }
    t.emit(
      { t: "discard", who: seat, tile, tsumogiri: action.tsumogiri, riichi: action.riichi },
      { e: "discard", who: seat, tile, tsumogiri: action.tsumogiri, riichi: action.riichi },
    );
    t.riichi[seat] = t.board.riichiActive[seat];
    t.kuikaeBan = null;
    t.clearTemporaryFuriten(seat);
    t.turnIndex++;
    deps.onAction?.(t, seat, action, drawnNow);
    rinshanPending = false;
    const wasLastTile = t.wall.remaining === 0;

    // ---- CLAIM ---------------------------------------------------------
    const claim = yield* claimPhase(t, seat, tile, scorer, false);

    if (claim?.kind === "sanchahou") {
      return settleDraw("sanchahou");
    }
    if (claim?.kind === "ron") {
      const wins = claim.seats
        .map((w) =>
          scorer.scoreWin(
            t,
            w,
            seat,
            tile,
            winFlags(t, w, false, {
              rinshan: false,
              chankan: false,
              haitei: false,
              houtei: wasLastTile,
            }),
          )
        )
        .filter((w): w is WinInfo => w !== null);
      if (wins.length) return settleWin(wins);
    }

    // The riichi stick is only committed once the declaring tile survives.
    if (riichiPending !== null) {
      const who = riichiPending;
      riichiPending = null;
      t.scores[who] -= 1000;
      t.round.kyotaku += 1;
      t.syncScores();
      t.emit(
        { t: "reach", who, step: 2, scores: t.board.scores.slice() },
        { e: "riichi", who, step: 2 },
      );
      t.ippatsu[who] = true;
      if (cfg.suuchaRiichiDraw && SEATS.every((s) => t.riichi[s])) {
        return settleDraw("suucha-riichi");
      }
    }

    if (deferredDora) {
      deferredDora = false;
      const ind = t.wall.revealIndicator();
      t.emit({ t: "dora", indicator: ind }, { e: "dora", indicator: ind });
    }

    if (claim?.kind === "call") {
      const meld = buildMeld(claim.seat, seat, claim.action, t);
      t.emit({ t: "call", meld }, { e: "call", meld });
      t.firstTurnIntact = false;
      t.breakIppatsu();
      deps.onAction?.(t, claim.seat, claim.action, null);
      if (claim.action.t === "daiminkan") {
        t.kanTotal++;
        if (t.kanTotal >= 4 && cfg.suukaikanDraw && !singleKanner(t)) {
          return settleDraw("suukaikan");
        }
        if (cfg.kanDora) deferredDora = true;
        seat = claim.seat;
        drawn = t.wall.drawRinshan();
        rinshanPending = true;
        emitDraw(seat, drawn, true);
        continue;
      }
      t.kuikaeBan = { seat: claim.seat, types: kuikaeTypes(claim.action) };
      seat = claim.seat;
      drawn = null;
      continue;
    }

    // ---- NEXT ----------------------------------------------------------
    if (t.wall.exhausted) return settleDraw("exhaustive");

    seat = ((seat + 1) % 4) as Seat;
    t.breakIppatsu(seat);
    t.clearTemporaryFuriten(seat);
    drawn = t.wall.draw();
    emitDraw(seat, drawn, false);
  }
}

// ---------------------------------------------------------------------------

type ClaimOutcome = ReturnType<typeof resolveClaims>;

/** Poll the three other seats on a discard (or an added kan). */
function* claimPhase(
  t: Table,
  from: Seat,
  tile: Tile,
  scorer: Scorer,
  chankan: boolean,
): Generator<Request, ClaimOutcome, Action> {
  const replies = new Map<Seat, Action>();
  for (const s of SEATS) {
    if (s === from) continue;
    const legal = claimActions(t, s, tile, from, scorer, chankan);
    if (legal.length <= 1) continue; // only "pass" — do not bother the policy
    const a = yield { k: "claim", seat: s, tile, from, legal };
    replies.set(s, a);
    // Declining a ron you could have taken is what makes you temporarily
    // furiten — and permanently so if you are in riichi.
    if (a.t !== "ron" && legal.some((x) => x.t === "ron")) t.markPassedRon(s);
  }
  return resolveClaims(t, from, replies);
}

function winFlags(
  t: Table,
  seat: Seat,
  tsumo: boolean,
  extra: Pick<WinFlags, "rinshan" | "chankan" | "haitei" | "houtei">,
): WinFlags {
  return {
    tsumo,
    riichi: t.riichi[seat],
    doubleRiichi: t.doubleRiichi[seat],
    ippatsu: t.ippatsu[seat],
    tenhou: t.cfg.tenhouChiihou && tsumo && t.firstTurnIntact && seat === t.dealer &&
      t.turnIndex === 0,
    chiihou: t.cfg.tenhouChiihou && tsumo && t.firstTurnIntact && seat !== t.dealer &&
      t.turnIndex < 4,
    ...extra,
  };
}

/** 四開槓 does not abort when all four kans belong to one player. */
function singleKanner(t: Table): boolean {
  const kanSeats = new Set<number>();
  for (const s of SEATS) {
    for (const m of t.melds[s]) {
      if (m.kind === "ankan" || m.kind === "daiminkan" || m.kind === "shouminkan") {
        kanSeats.add(s);
      }
    }
  }
  return kanSeats.size <= 1;
}

/** 流し満貫: every discard is a yaochu tile and none of them was called. */
function isNagashi(t: Table, seat: Seat): boolean {
  const river = t.board.rivers[seat];
  if (river.length === 0) return false;
  return river.every((r) => r.calledBy === undefined && isYaochu(tileType(r.tile)));
}

function defaultPlayers() {
  return SEATS.map((seat) => ({ seat, name: `P${seat}` }));
}
