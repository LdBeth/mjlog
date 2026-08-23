// The referee, asked hypothetically: "if I did this, would you write it down?"
//
// `rules.ts` judges actions that have already committed. A policy has to decide
// BEFORE that, so this module puts the action on the table, runs the very same
// `DojoRule.check` predicates through the very same `runHook`, and takes the
// action back off again. Nothing here re-implements a rule — that is the whole
// point: a hand-rolled copy of a predicate drifts from the ledger, and a policy
// steering by a drifted copy files violations it believes it avoided.
//
// HOW THE HYPOTHETICAL IS BUILT. Each preview mirrors, by hand, exactly the
// mutation `round.ts` performs before it calls `onAction` — a discard leaves the
// hand and lands in the river, a call takes its tiles out of the hand and puts a
// meld on the table — then restores from a snapshot in a `finally`. The mutation
// is synchronous and nothing observes the table in between, so the same trick
// `observe.ts::discardInfoFor` and `rules.ts::onPreWinHand` already use is safe
// here. It also means the preview reproduces the ledger's reading of the state
// AS THE HOOK WILL SEE IT, timing included (see 暗槓条件違反 below), which is
// what a filter needs: the question is not "is this move sound?" but "will the
// referee charge for it?".
//
// A preview is only valid for the tick it was made in. `observe()` hands one to
// the policy for the decision it is about to make; holding onto it past that
// decision would ask questions of a table that has moved on.
//
// ---------------------------------------------------------------------------
// RULE → PREVIEWABLE (every rule in `RULES`, no exceptions hidden)
// ---------------------------------------------------------------------------
//
//   first-honor            post-discard   YES  previewDiscard
//   noten-dora             post-discard   YES  previewDiscard
//   dora-pon-lock          post-discard   YES  previewDiscard (the tedashi half).
//                          on-call        n/a  the call half only ARMS the lock
//                                              and never charges; the preview
//                                              runs it and rolls the flag back.
//   minkan                 on-kan/on-call YES  previewKan / previewCall
//   ankan-form             on-kan         YES  previewKan
//   riichi-kan-skip        post-discard   YES  previewDiscard (taking the action)
//                                              AND previewSkipKan (declining to
//                                              kan — the one rule that fires on
//                                              an OMISSION).
//   hadaka-tanki           post-discard   YES  previewDiscard, plus the call
//                                              lookahead in previewCall: with
//                                              four melds every discard is a
//                                              violation, so the call that makes
//                                              the fourth meld is the last point
//                                              at which it can be avoided.
//   jigoku-tanki           on-riichi      YES  previewDiscard({riichi:true})
//   suuankou-riichi        on-riichi      YES  ditto
//   yakuman-threat-riichi  on-riichi      YES  ditto
//   hikkake                on-riichi      YES  ditto
//   yakuman-related        post-discard   YES  previewDiscard
//   under-8000             on-round-end   NO   fires on the PAYMENTS at the end
//                                              of a round, not on an action; no
//                                              single move can veto it.
//   katagari               on-win         NO   fires on taking a win. A policy
//                                              could only avoid it by declining
//                                              one, which makes it 見逃し
//                                              (furiten) — a worse position by
//                                              the engine's own rules. Prevention
//                                              lives one step earlier, in
//                                              `heuristic.ts::dojoCost`, which
//                                              prices the DISCARD that leaves a
//                                              split wait — and, since a price
//                                              can be outbid, vetoes it too
//                                              (`compliantDiscards`).
//   atozuke                on-win         NO   same, handled the same way
//                                              (`yakulessTenpai`).
//   uki-crush              on-win     (B) NO   same — would require declining.
//   misehai                on-call    (B) NO   needs `Table.exposed`, filled by
//                                              a 空ポン the engine cannot even
//                                              represent as an Action.
//
// ---------------------------------------------------------------------------
// 暗槓条件違反 AND THE POINT IN TIME IT IS ASKED AT
// ---------------------------------------------------------------------------
// `ankan-form` runs on the `on-kan` hook, which `round.ts` fires AFTER the kan
// meld is on the table: the four tiles are out of the hand and the meld is among
// `t.melds[seat]` by the time the rule asks its questions. `wouldChangeWait`
// therefore takes a `committed` flag and rebuilds the pre-kan hand from it.
// (It used to assume the opposite, counting the kan once from the meld list and
// once from `melds.length + 1`, which made every ankan in the game "changes the
// wait" — and, in riichi, made the fourth copy a charged dilemma: 暗槓条件違反
// for kanning, 立直後カン見送り for declining.)
//
// `previewKan` mirrors that timing exactly — it lays the meld, asks, and rolls
// back — so the answer it gives is the answer the ledger will give. A clean kan
// now previews clean, `mandatoryKan` takes it, and the dilemma only survives for
// kans that really do move the wait; `pickLesserEvil` below still resolves those
// by points, with the tie (both 中) falling to declining.

import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import { tileType } from "mjrender/tiles.ts";
import type { WinOracle } from "../legal.ts";
import { ANY_WIN, kuikaeTypes } from "../legal.ts";
import type { DojoConfig } from "../rules.ts";
import type { Table } from "../table.ts";
import type { Action, Seat, Violation } from "../types.ts";
import { SEATS } from "../types.ts";
import type { RuleCtx } from "./mod.ts";
import { runHook } from "./rules.ts";

/** What a policy asks about the action it is considering. */
export interface ActionPreview {
  /**
   * Violations a discard (and, when flagged, the riichi declaration it carries)
   * would draw. `drawn` is this turn's draw; a tsumogiri implies it.
   */
  discard(action: Extract<Action, { t: "discard" }>, drawn?: Tile | null): Violation[];
  /** Violations a pon/chi/大明槓 would draw, including the forced-discard trap. */
  call(action: Action): Violation[];
  /** Violations a kan would draw. `drawn` is this turn's draw, as `round.ts` passes it. */
  kan(action: Action, drawn: Tile | null): Violation[];
  /** Violations DECLINING to kan would draw (立直後カン見送り). */
  skipKan(drawn: Tile | null): Violation[];
}

interface Ctxish {
  t: Table;
  seat: Seat;
  dojo: DojoConfig;
  oracle: WinOracle;
}

/**
 * Bind a preview to one seat at one table. `observe()` calls this; a policy
 * receives the result on the Observation and never needs the Table itself.
 */
export function makePreview(
  t: Table,
  seat: Seat,
  dojo: DojoConfig,
  oracle: WinOracle = ANY_WIN,
): ActionPreview {
  return {
    discard: (a, drawn = null) => previewDiscard(t, seat, a, dojo, oracle, drawn),
    call: (a) => previewCall(t, seat, a, dojo, oracle),
    kan: (a, drawn) => previewKan(t, seat, a, drawn, dojo, oracle),
    skipKan: (drawn) => previewSkipKan(t, seat, drawn, dojo, oracle),
  };
}

// --- the guard ---------------------------------------------------------------

/** The table a `guarded` call currently holds a snapshot of, if any. */
let guardHolder: Table | null = null;

/**
 * Two rules WRITE to the table when they fire (ドラ切り後の手出し arms
 * `tsumogiriLock`; 見せ牌 closes ron types). A hypothetical must not leave
 * those behind, so every preview runs inside this.
 *
 * `ronBlocked` is only snapshotted when Tier B is live: 見せ牌 is its only
 * writer, and copying four Sets per candidate discard is not free.
 *
 * NESTING. `previewCall`'s forced-discard lookahead runs a `previewDiscard` per
 * escape route inside its own guard, and a snapshot taken there would only copy
 * state the outer guard already holds and already restores. So a guard entered
 * while one is open on the SAME table is a pass-through. That is exact, not an
 * approximation: both writers are on the `on-call` side (the arming half of
 * ドラ切り後の手出し, and 見せ牌, both registered on `on-call` alone), so no
 * hook a nested `previewDiscard` can reach writes to either channel and nothing
 * leaks between sibling iterations. Previews are synchronous start to finish,
 * so a module-level marker is enough to see the open guard.
 */
function guarded<T>(t: Table, dojo: DojoConfig, fn: () => T): T {
  if (guardHolder === t) return fn();
  const lock = [...t.tsumogiriLock];
  const blocked = dojo.tierB ? t.ronBlocked.map((s) => new Set(s)) : null;
  guardHolder = t;
  try {
    return fn();
  } finally {
    guardHolder = null;
    for (const s of SEATS) {
      t.tsumogiriLock[s] = lock[s];
      if (blocked) {
        t.ronBlocked[s].clear();
        for (const ty of blocked[s]) t.ronBlocked[s].add(ty);
      }
    }
  }
}

function ctxOf(c: Ctxish, action: Action, drawn: Tile | null): RuleCtx {
  return {
    t: c.t,
    seat: c.seat,
    action,
    drawn,
    cfg: c.t.cfg,
    dojo: c.dojo,
    oracle: c.oracle,
  };
}

/** Total 評価点 a set of violations would cost — the tie-break for a dilemma. */
export function violationPoints(vs: readonly Violation[]): number {
  return vs.reduce((n, v) => n + v.points, 0);
}

// --- discard -----------------------------------------------------------------

/**
 * The discard, previewed. Mirrors `round.ts`'s pre-hook mutation exactly:
 * the tile leaves the hand and lands in the seat's own river, a declaration
 * turns `riichi` on (and `doubleRiichi` when the first go-around is intact,
 * which is what makes 第一打字牌切り and 即引っかけ exempt a ダブリー).
 *
 * `drawn` is the tile this turn's draw put in the hand, which is what `round.ts`
 * hands the hook and what 立直後カン見送り reads. It defaults to the discarded
 * tile on a tsumogiri (which is what a tsumogiri means) and to null otherwise —
 * a tedashi leaves the drawn tile IN the hand, and the rule declines to judge
 * that case anyway.
 *
 * `only` narrows the reading to one rule id, for a caller that wants exactly
 * one verdict; the result is the full reading filtered by `v.rule === only`.
 */
export function previewDiscard(
  t: Table,
  seat: Seat,
  action: Extract<Action, { t: "discard" }>,
  dojo: DojoConfig,
  oracle: WinOracle = ANY_WIN,
  drawn: Tile | null = null,
  only?: string,
): Violation[] {
  const hand = t.hands[seat];
  const i = hand.lastIndexOf(action.tile);
  if (i < 0) return []; // not a tile this seat holds: nothing to preview
  const c: Ctxish = { t, seat, dojo, oracle };

  return guarded(t, dojo, () => {
    const river = t.board.rivers[seat];
    const riichiWas = t.riichi[seat];
    const doubleWas = t.doubleRiichi[seat];
    const entry: RiverEntry = {
      tile: action.tile,
      junme: t.junme,
      tsumogiri: action.tsumogiri,
      riichiDeclare: action.riichi,
    };
    hand.splice(i, 1);
    river.push(entry);
    if (action.riichi) {
      t.riichi[seat] = true;
      if (t.firstTurnIntact) t.doubleRiichi[seat] = true;
    }
    try {
      const ctx = ctxOf(c, action, drawn ?? (action.tsumogiri ? action.tile : null));
      const out = runHook("post-discard", ctx, only);
      if (action.riichi) out.push(...runHook("on-riichi", ctx, only));
      return out;
    } finally {
      river.pop();
      hand.splice(i, 0, action.tile);
      t.riichi[seat] = riichiWas;
      t.doubleRiichi[seat] = doubleWas;
    }
  });
}

/**
 * 立直後カン見送り, the omission side: what declining the kan costs. It is a
 * discard rule (the tsumogiri that passed the kan up is the charged action), so
 * this is the discard preview asked of that one rule — no second reading of the
 * predicate, just a narrower question (and the other seven predicates, whose
 * answers this would have thrown away, are never run).
 */
export function previewSkipKan(
  t: Table,
  seat: Seat,
  drawn: Tile | null,
  dojo: DojoConfig,
  oracle: WinOracle = ANY_WIN,
): Violation[] {
  if (drawn === null) return [];
  const action: Extract<Action, { t: "discard" }> = {
    t: "discard",
    tile: drawn,
    riichi: false,
    tsumogiri: true,
  };
  return previewDiscard(t, seat, action, dojo, oracle, drawn, "riichi-kan-skip");
}

// --- calls -------------------------------------------------------------------

/** The seat whose last river tile is `called` — the discarder a call takes from. */
function discarderOf(t: Table, called: Tile): Seat | null {
  for (const s of SEATS) {
    const river = t.board.rivers[s];
    const last = river[river.length - 1];
    if (last && last.tile === called && last.calledBy === undefined) return s;
  }
  return null;
}

function meldFor(seat: Seat, from: Seat, action: Action, t: Table): Meld | null {
  switch (action.t) {
    case "chi":
    case "pon":
      return {
        kind: action.t,
        who: seat,
        fromWho: from,
        tiles: [...action.tiles, action.called].sort((a, b) => a - b),
        calledTile: action.called,
      };
    case "daiminkan": {
      const ty = tileType(action.called);
      const held = t.hands[seat].filter((id) => tileType(id) === ty).slice(0, 3);
      return {
        kind: "daiminkan",
        who: seat,
        fromWho: from,
        tiles: [...held, action.called].sort((a, b) => a - b),
        calledTile: action.called,
      };
    }
    case "ankan": {
      const held = t.hands[seat].filter((id) => tileType(id) === action.type);
      return {
        kind: "ankan",
        who: seat,
        fromWho: seat,
        tiles: [...held].sort((a, b) => a - b),
        calledTile: held[0],
      };
    }
    default:
      return null;
  }
}

/** Put a meld on the table the way `BoardState.applyMeld` does; return the undo. */
function applyMeld(t: Table, meld: Meld): () => void {
  const seat = meld.who as Seat;
  const hand = t.hands[seat];
  const handWas = [...hand];
  const melds = t.melds[seat];
  for (const id of meld.tiles) {
    if (id === meld.calledTile && meld.fromWho !== meld.who) continue;
    const i = hand.indexOf(id);
    if (i >= 0) hand.splice(i, 1);
  }
  melds.push(meld);
  let river: RiverEntry | null = null;
  if (meld.fromWho !== meld.who) {
    const r = t.board.rivers[meld.fromWho];
    const last = r[r.length - 1];
    if (last && last.tile === meld.calledTile) {
      river = last;
      last.calledBy = meld.who;
    }
  }
  return () => {
    melds.pop();
    hand.length = 0;
    hand.push(...handWas);
    if (river) delete river.calledBy;
  };
}

/**
 * A pon / chi / 大明槓, previewed.
 *
 * Two questions, not one. The first is what the `on-call` rules say about the
 * call itself. The second is the trap a call sets for the turn that follows it:
 * a caller MUST discard, and it discards with no drawn tile, so every option is
 * a tedashi and the seat cannot escape into a tsumogiri. Two rules bite exactly
 * there — 裸単騎 (with four melds every discard is the naked wait) and ドラ切り
 * 後の手出し (while locked, every tedashi is charged) — and in both the call is
 * the last moment at which the violation could still be avoided. So a call whose
 * every follow-up discard is charged is itself reported as charged, priced at
 * the cheapest of those discards.
 *
 * 食い替え is excluded from the escape routes, since the engine will not offer
 * those discards either (`legal.ts::kuikaeTypes`).
 */
export function previewCall(
  t: Table,
  seat: Seat,
  action: Action,
  dojo: DojoConfig,
  oracle: WinOracle = ANY_WIN,
): Violation[] {
  if (action.t !== "pon" && action.t !== "chi" && action.t !== "daiminkan") return [];
  const from = discarderOf(t, action.called);
  if (from === null) return [];
  const meld = meldFor(seat, from, action, t);
  if (!meld) return [];
  const c: Ctxish = { t, seat, dojo, oracle };

  return guarded(t, dojo, () => {
    const undo = applyMeld(t, meld);
    try {
      const out = runHook("on-call", ctxOf(c, action, null));
      if (out.length > 0) return out;
      if (action.t === "daiminkan") return out; // a kan draws, so a tsumogiri exists

      // The forced-discard lookahead: charge the call only if EVERY discard it
      // leaves us with is charged.
      const banned = kuikaeTypes(action);
      let cheapest: Violation[] | null = null;
      for (const tile of new Set(t.hands[seat])) {
        if (banned.has(tileType(tile))) continue;
        const vs = previewDiscard(
          t,
          seat,
          { t: "discard", tile, riichi: false, tsumogiri: false },
          dojo,
          oracle,
        );
        if (vs.length === 0) return out; // a clean way out exists
        if (!cheapest || violationPoints(vs) < violationPoints(cheapest)) cheapest = vs;
      }
      return cheapest ?? out;
    } finally {
      undo();
    }
  });
}

// --- kan ---------------------------------------------------------------------

/**
 * An 暗槓 / 加槓 / 大明槓 declared on our own turn, previewed.
 *
 * 明槓 needs no state at all (the rule reads the action), but an ankan does, and
 * it must be the state the hook will actually see: `round.ts` emits the meld
 * BEFORE calling `onAction`, so the four tiles are out of the hand and the kan
 * is already among the melds by the time 暗槓条件違反 asks its questions. See the
 * header note on how `wouldChangeWait` reads that state.
 */
export function previewKan(
  t: Table,
  seat: Seat,
  action: Action,
  drawn: Tile | null,
  dojo: DojoConfig,
  oracle: WinOracle = ANY_WIN,
): Violation[] {
  if (action.t !== "ankan" && action.t !== "kakan" && action.t !== "daiminkan") return [];
  const c: Ctxish = { t, seat, dojo, oracle };
  if (action.t !== "ankan") {
    // 加槓 and 大明槓 are charged on sight; no hypothetical state is needed, and
    // faking a shouminkan upgrade would only risk disagreeing with the engine.
    return guarded(t, dojo, () => runHook("on-kan", ctxOf(c, action, drawn)));
  }
  const meld = meldFor(seat, seat, action, t);
  if (!meld || meld.tiles.length !== 4) return [];

  return guarded(t, dojo, () => {
    const undo = applyMeld(t, meld);
    try {
      return runHook("on-kan", ctxOf(c, action, drawn));
    } finally {
      undo();
    }
  });
}

/**
 * Which of two ledgerable options to take when both are: the cheaper one, ties
 * going to `a`. Used for the 立直後カン見送り / 暗槓条件違反 dilemma, where the
 * caller passes "decline" first so a tie declines.
 */
export function pickLesserEvil(a: readonly Violation[], b: readonly Violation[]): "a" | "b" {
  return violationPoints(b) < violationPoints(a) ? "b" : "a";
}
