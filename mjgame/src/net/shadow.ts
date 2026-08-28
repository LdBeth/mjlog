// Shadow table: a real `Table` reconstructed from the MJAI wire, so that the
// champion's Observation is built by the REAL `observe()` — no duplicated
// derivation. The alternative (hand-assembling an Observation from wire state)
// was rejected because the policy DEGRADES SILENTLY when `danger`/`discardInfo`
// /`preview` are missing or subtly wrong: drift would never throw, it would
// just quietly play the un-tuned default.
//
// The trick is that `observe()` consumes only public information plus the
// observing seat's own hand (verified read-by-read; see the parity test), so a
// Table whose OPPONENT hands hold arbitrary placeholder ids and whose wall is a
// fabrication produces the identical Observation — provided every public tile,
// junme, wall count and furiten flag is right. This file's job is exactly that
// bookkeeping, mirroring `round.ts`'s emit order and Table mutations event for
// event (each mirror cites the round.ts line it copies).
//
// Id discipline: the 136-permutation invariant is kept at all times. Masked
// draws ("?") allocate a free id; when a concealed opponent tile becomes public
// (tedashi, meld) it is materialized by swapping a placeholder for a
// type-and-red matching copy — placeholders live only in opponent hands, which
// nothing observable reads, so retyping one is invisible. `syncFromDecoded`
// then makes public ids EXACT against the server's own decoded observation
// (riichi.dev's base64 blob shares this tile scheme — see mjai.ts), which is
// what lets the parity test demand id-level equality, not just type-level.
//
// Fake-wall caveat: `t.wall.tiles` never reaches the board (draw ids are burned
// for the counter and replaced by wire ids), and the one reader of dead-wall
// identity — `uraIndicatorsOf` via `hasYaku`, when WE are in riichi — feeds han
// counting only: ura cannot open or close a yaku, so `ronnable`/`katagari`/
// `yakuless` are unaffected by the fabricated ura tiles.

import type { GameEvent, Meld, Tile } from "mjrender/model.ts";
import { tileType } from "mjrender/tiles.ts";
import { countsFromTiles, shanten, ukeireTypes } from "../kernel.ts";
import { buildMeld, completesHand, kuikaeTypes } from "../legal.ts";
import type { Observation } from "../observe.ts";
import { observe } from "../observe.ts";
import type { RuleConfig } from "../rules.ts";
import { DOJO_HEADLESS } from "../rules.ts";
import { scorer } from "../score.ts";
import { type RoundInit, Table } from "../table.ts";
import type { Action, Seat } from "../types.ts";
import { SEATS } from "../types.ts";
import { Wall } from "../wall.ts";
import { ARENA_CFG, idMatchesPai, matchId, paiToType } from "./mjai.ts";

/** The shadow lost the server: state can no longer be trusted this kyoku. */
export class ShadowDesyncError extends Error {}

interface MjaiEventish {
  type: string;
  [k: string]: unknown;
}

const BAKAZE_BASE: Record<string, number> = { E: 0, S: 4, W: 8, N: 12 };

export class ShadowGame {
  /** Absolute seat of the bot; -1 before start_game. */
  me: Seat | -1 = -1;

  #cfg: RuleConfig;
  #t: Table | null = null;
  /** Our current drawn tile; null after our discard / off-turn. */
  #drawn: Tile | null = null;
  /** Seat of the most recent dahai this kyoku — the claim source the wire
   *  never names (riichi.dev claim entries omit `target`, and the decoded
   *  blob's `last_discard` changed meaning between RiichiEnv versions). */
  #lastDiscarder: Seat | null = null;
  /** reach declared (wire `reach`); that actor's next dahai is the riichi tile. */
  #reachPending: Seat | null = null;
  /** riichi discard emitted; the stick commits when the tile survives claims —
   *  i.e. at `reach_accepted`, or defensively at the next non-hora event. */
  #stickPending: Seat | null = null;
  /** A kan was declared; that actor's next tsumo is the rinshan draw. */
  #rinshanPending: Seat | null = null;
  /** An opponent discard completed OUR hand and no ron has (yet) ended the
   *  round: 見逃し furiten lands at the next event, exactly when `claimPhase`
   *  would have applied it (round.ts:392) — never before our claim decision. */
  #minogashiPending = false;

  constructor(cfg: RuleConfig = ARENA_CFG) {
    this.#cfg = cfg;
  }

  get table(): Table | null {
    return this.#t;
  }

  drawnTile(): Tile | null {
    return this.#drawn;
  }

  /** Who discarded last (null before the kyoku's first dahai). */
  lastDiscarder(): Seat | null {
    return this.#lastDiscarder;
  }

  /** Feed every server event in arrival order. Unknown types are ignored;
   *  anything that contradicts the board throws `ShadowDesyncError`. */
  apply(e: MjaiEventish): void {
    try {
      this.#apply(e);
    } catch (err) {
      if (err instanceof ShadowDesyncError) throw err;
      throw new ShadowDesyncError(`event ${e.type} の適用に失敗: ${err}`);
    }
  }

  /** Build the champion's Observation off the live shadow table. The referee
   *  (preview) stays ON: the champion's grading regime included it, and the
   *  placeholder hands are safe under it (the one cross-seat read is
   *  rollback-restored and returns no violations). */
  observe(legal: Action[], drawn: Tile | null, claimTile: Tile | null): Observation {
    if (!this.#t || this.me < 0) throw new ShadowDesyncError("局の外で observe された");
    return observe(this.#t, this.me as Seat, legal, drawn, scorer, claimTile, DOJO_HEADLESS);
  }

  // -------------------------------------------------------------------------
  // event application
  // -------------------------------------------------------------------------

  #apply(e: MjaiEventish): void {
    switch (e.type) {
      case "start_game":
        this.me = Number(e.id ?? -1) as Seat;
        return;
      case "start_kyoku":
        this.#startKyoku(e);
        return;
      case "end_kyoku":
      case "end_game":
        this.#t = null;
        this.#drawn = null;
        this.#lastDiscarder = null;
        this.#clearPendings();
        return;
    }
    if (!this.#t) return; // between kyoku, nothing to shadow

    switch (e.type) {
      case "tsumo":
        this.#resolvePendings(e);
        this.#tsumo(this.#seatOf(e.actor), String(e.pai));
        break;
      case "dahai":
        this.#resolvePendings(e);
        this.#dahai(this.#seatOf(e.actor), String(e.pai), e.tsumogiri === true);
        break;
      case "reach":
        this.#resolvePendings(e);
        this.#reachPending = this.#seatOf(e.actor);
        break;
      case "reach_accepted":
        this.#commitStick();
        break;
      case "chi":
      case "pon":
        this.#resolvePendings(e);
        this.#call(e, e.type);
        break;
      case "daiminkan":
      case "kan":
        this.#resolvePendings(e);
        this.#call(e, "daiminkan");
        break;
      case "ankan":
        this.#resolvePendings(e);
        this.#ankan(this.#seatOf(e.actor), this.#strings(e.consumed, 4));
        break;
      case "kakan":
        this.#resolvePendings(e);
        this.#kakan(this.#seatOf(e.actor), String(e.pai));
        break;
      case "dora":
        this.#resolvePendings(e);
        this.#dora(String(e.dora_marker));
        break;
      case "hora":
        // A ron on the riichi tile means the stick was never placed
        // (round.ts:302 runs only after the claim phase yields no win), and a
        // win by US retracts the pending 見逃し. Round is over either way.
        if (this.#minogashiPending && this.#seatOf(e.actor) !== this.me) this.#markMinogashi();
        this.#clearPendings();
        this.#applyDeltas(e);
        break;
      case "ryukyoku":
        // Next kyoku's scores/kyotaku arrive authoritatively in start_kyoku,
        // so pending resolution after the round is moot — just drop it.
        this.#clearPendings();
        this.#applyDeltas(e);
        break;
      default:
        break; // unknown event types are the server's business, not a desync
    }
  }

  #startKyoku(e: MjaiEventish): void {
    if (this.me < 0) throw new ShadowDesyncError("start_game 前の start_kyoku");
    this.#lastDiscarder = null;
    const me = this.me as Seat;
    const dealer = Number(e.oya) as Seat;
    const base = BAKAZE_BASE[String(e.bakaze)] ?? 0;
    const kyoku = base + (Number(e.kyoku) - 1);
    const tehais = e.tehais as unknown;
    if (!Array.isArray(tehais) || !Array.isArray(tehais[me]) || tehais[me].length !== 13) {
      throw new ShadowDesyncError("start_kyoku.tehais が読めない");
    }
    const myPais = (tehais[me] as unknown[]).map(String);
    if (myPais.includes("?")) throw new ShadowDesyncError("自席の配牌が伏せられている");

    // Allocate ids: our 13 from the wire strings, the indicator, then 39
    // placeholders — all distinct, the rest fills the fabricated wall.
    const free = new Set<Tile>();
    for (let id = 0; id < 136; id++) free.add(id as Tile);
    const takeByPai = (pai: string): Tile => {
      for (const id of free) {
        if (idMatchesPai(id, pai)) {
          free.delete(id);
          return id;
        }
      }
      throw new ShadowDesyncError(`配牌/ドラの5枚目: ${pai}`);
    };
    const takeAny = (): Tile => {
      const id = free.values().next().value as Tile;
      free.delete(id);
      return id;
    };
    const myIds = myPais.map(takeByPai);
    const doraId = takeByPai(String(e.dora_marker));

    // Wall fabrication: `Wall.deal` hands out tiles[135], tiles[134], … in
    // Tenhou block order (wall.ts:92-104), and `revealIndicator` reads
    // tiles[5]. Place our ids at OUR dealt positions, placeholders at the
    // others, the indicator at 5, and the leftovers anywhere else.
    const seatSeq: Seat[] = [];
    for (let block = 0; block < 3; block++) {
      for (let k = 0; k < 4; k++) {
        const s = ((dealer + k) % 4) as Seat;
        seatSeq.push(s, s, s, s);
      }
    }
    for (let k = 0; k < 4; k++) seatSeq.push(((dealer + k) % 4) as Seat);
    const tiles = new Array<Tile>(136).fill(-1 as Tile);
    tiles[5] = doraId;
    let mine = 0;
    seatSeq.forEach((s, j) => {
      tiles[135 - j] = s === me ? myIds[mine++] : takeAny();
    });
    for (let i = 0; i < 84; i++) {
      if (i !== 5) tiles[i] = takeAny();
    }

    const init: RoundInit = {
      kyoku,
      honba: Number(e.honba ?? 0),
      kyotaku: Number(e.kyotaku ?? 0),
      dealer,
      scores: Array.isArray(e.scores)
        ? (e.scores as unknown[]).map(Number)
        : [25000, 25000, 25000, 25000],
      wall: new Wall(tiles),
      dice: [0, 0],
    };
    this.#t = new Table(init, this.#cfg, SEATS.map((seat) => ({ seat, name: `P${seat}` })));
    this.#drawn = null;
    this.#clearPendings();
  }

  #tsumo(actor: Seat, pai: string): void {
    const t = this.#t!;
    // round.ts:353-354 — the seat about to draw sheds its ippatsu window and
    // its temporary furiten (harmlessly repeated for the dealer's opening draw).
    t.breakIppatsu(actor);
    t.clearTemporaryFuriten(actor);

    const rinshan = this.#rinshanPending === actor;
    if (rinshan) {
      this.#rinshanPending = null;
      t.wall.drawRinshan(); // burn the counter; the id below is the wire's
    } else {
      t.wall.draw();
    }
    let id: Tile;
    if (actor === this.me) {
      if (pai === "?") throw new ShadowDesyncError("自分のツモ牌が伏せられている");
      id = this.#allocByPai(pai);
      this.#drawn = id;
    } else {
      id = this.#allocAny();
    }
    t.emit({ t: "draw", who: actor, tile: id, rinshan } as GameEvent, {
      e: "draw",
      who: actor,
      rinshan,
      tile: id,
    });
  }

  #dahai(actor: Seat, pai: string, tsumogiri: boolean): void {
    const t = this.#t!;
    this.#lastDiscarder = actor;
    const riichi = this.#reachPending === actor;
    if (riichi) {
      // round.ts:252-255 — step 1 precedes the flagged discard; ダブリー arms
      // off the untouched first go-around.
      this.#reachPending = null;
      t.emit({ t: "reach", who: actor, step: 1 }, { e: "riichi", who: actor, step: 1 });
      if (t.firstTurnIntact) t.doubleRiichi[actor] = true;
      this.#stickPending = actor;
    }
    let id: Tile;
    if (actor === this.me) {
      const found = matchId(t.hands[this.me], pai);
      if (found === null) throw new ShadowDesyncError(`自分の打牌 ${pai} が手中にない`);
      id = found;
    } else {
      id = this.#materialize(actor, pai);
    }
    t.emit(
      { t: "discard", who: actor, tile: id, tsumogiri, riichi },
      { e: "discard", who: actor, tile: id, tsumogiri, riichi },
    );
    // round.ts:261-271, in order.
    t.riichi[actor] = t.board.riichiActive[actor];
    t.kuikaeBan = null;
    if (actor === this.me) {
      t.clearTemporaryFuriten(actor);
      t.refreshPermanentFuriten(actor, this.#currentWaits());
      this.#drawn = null;
    }
    t.turnIndex++;
    // 見逃し: a discard that completes our hand furitens us whether or not the
    // server offers the ron (yakuless shapes never generate a request) — but
    // only once the round moves PAST our claim window (round.ts:392).
    if (actor !== this.me && completesHand(t, this.me as Seat, id)) {
      this.#minogashiPending = true;
    }
  }

  #call(e: MjaiEventish, kind: "chi" | "pon" | "daiminkan"): void {
    const t = this.#t!;
    const actor = this.#seatOf(e.actor);
    const from = this.#seatOf(e.target);
    const river = t.board.rivers[from];
    const called = river[river.length - 1]?.tile;
    if (called === undefined || !idMatchesPai(called, String(e.pai))) {
      throw new ShadowDesyncError(`${kind} の対象牌 ${e.pai} が河末尾と一致しない`);
    }
    const consumed = this.#claimConsumed(
      actor,
      this.#strings(e.consumed, kind === "chi" ? 2 : kind === "pon" ? 2 : 3),
    );

    let meld: Meld;
    if (kind === "daiminkan") {
      // Not via `buildMeld`: its hand filter would sweep up same-type
      // placeholder ids sitting in an opponent's concealed hand. Field-for-field
      // the literal below matches legal.ts:382-392.
      meld = {
        kind: "daiminkan",
        who: actor,
        fromWho: from,
        tiles: [...consumed, called].sort((a, b) => a - b),
        calledTile: called,
      };
    } else {
      const action: Action = { t: kind, tiles: [consumed[0], consumed[1]], called };
      meld = buildMeld(actor, from, action, t);
      // round.ts:343 — the caller's next discard may not swap the call back.
      t.kuikaeBan = { seat: actor, types: kuikaeTypes(action) };
    }
    t.emit({ t: "call", meld }, { e: "call", meld });
    // round.ts:328-336.
    t.firstTurnIntact = false;
    t.breakIppatsu();
    if (kind === "daiminkan") {
      t.kanTotal++;
      t.kuikaeBan = null;
      this.#rinshanPending = actor;
    }
  }

  #ankan(actor: Seat, consumed: string[]): void {
    const t = this.#t!;
    const ids = this.#claimConsumed(actor, consumed);
    // Field-for-field the literal of legal.ts:393-402, except `calledTile`,
    // where the engine takes the hand-order-first copy — unknowable from the
    // wire and observationally inert (nothing reads an ankan's calledTile), so
    // the sorted-first copy stands in. The parity test normalizes it.
    const tiles = [...ids].sort((a, b) => a - b);
    const meld: Meld = { kind: "ankan", who: actor, fromWho: actor, tiles, calledTile: tiles[0] };
    t.emit({ t: "call", meld }, { e: "call", meld });
    // round.ts:196-199, 240-242.
    t.kanTotal++;
    t.firstTurnIntact = false;
    t.breakIppatsu();
    this.#rinshanPending = actor;
  }

  #kakan(actor: Seat, pai: string): void {
    const t = this.#t!;
    const added = actor === this.me
      ? matchId(t.hands[this.me], pai)
      : this.#materialize(actor, pai);
    if (added === null) throw new ShadowDesyncError(`加槓牌 ${pai} が手中にない`);
    const action: Action = { t: "kakan", tile: added };
    const meld = buildMeld(actor, actor, action, t); // finds the pon; throws without one
    t.emit({ t: "call", meld }, { e: "call", meld });
    t.kanTotal++;
    t.firstTurnIntact = false;
    t.breakIppatsu();
    this.#rinshanPending = actor;
    // 槍槓 見逃し runs through the same claim window (round.ts:388-392).
    if (actor !== this.me && completesHand(t, this.me as Seat, added)) {
      this.#minogashiPending = true;
    }
  }

  #dora(pai: string): void {
    const t = this.#t!;
    const id = this.#allocByPai(pai);
    t.wall.revealIndicator(); // burn the dead-wall counter; the id is the wire's
    t.emit({ t: "dora", indicator: id }, { e: "dora", indicator: id });
  }

  #applyDeltas(e: MjaiEventish): void {
    const t = this.#t!;
    const deltas = e.deltas;
    if (Array.isArray(deltas) && deltas.length === 4) {
      for (const s of SEATS) t.scores[s] += Number(deltas[s]) || 0;
      t.syncScores();
    }
  }

  // -------------------------------------------------------------------------
  // pending resolution (the claim-phase timing round.ts gets from its yields)
  // -------------------------------------------------------------------------

  #clearPendings(): void {
    this.#reachPending = null;
    this.#stickPending = null;
    this.#rinshanPending = null;
    this.#minogashiPending = false;
  }

  /** Any gameplay event that is not the ron itself means the last discard
   *  survived the claim window: commit the pending stick (round.ts:302-313)
   *  and land the pending 見逃し (round.ts:392). */
  #resolvePendings(_next: MjaiEventish): void {
    this.#commitStick();
    if (this.#minogashiPending) this.#markMinogashi();
  }

  #commitStick(): void {
    const who = this.#stickPending;
    if (who === null || !this.#t) return;
    this.#stickPending = null;
    const t = this.#t;
    t.scores[who] -= 1000;
    t.round.kyotaku += 1;
    t.syncScores();
    t.emit(
      { t: "reach", who, step: 2, scores: t.board.scores.slice() },
      { e: "riichi", who, step: 2 },
    );
    t.ippatsu[who] = true;
  }

  #markMinogashi(): void {
    this.#minogashiPending = false;
    if (this.#t && this.me >= 0) this.#t.markPassedRon(this.me as Seat);
  }

  /** round.ts:403-409 verbatim (private there): the waits of our resting hand. */
  #currentWaits(): number[] {
    const t = this.#t!;
    const me = this.me as Seat;
    const counts = countsFromTiles(t.hands[me]);
    const open = t.melds[me].length;
    const closed = t.isMenzen(me);
    const s = shanten(counts, open, closed);
    return s <= 0 ? ukeireTypes(counts, open, closed, s) : [];
  }

  // -------------------------------------------------------------------------
  // the id ledger
  // -------------------------------------------------------------------------

  /** Every id currently on the board (hands incl. placeholders, rivers, melds,
   *  indicators). The complement is the free pool. */
  #usedIds(): Set<Tile> {
    const t = this.#t!;
    const used = new Set<Tile>();
    for (const s of SEATS) {
      for (const id of t.hands[s]) used.add(id);
      for (const r of t.board.rivers[s]) used.add(r.tile);
      for (const m of t.melds[s]) for (const id of m.tiles) used.add(id);
    }
    for (const id of t.board.indicators) used.add(id);
    return used;
  }

  #allocAny(): Tile {
    const used = this.#usedIds();
    for (let id = 0; id < 136; id++) {
      if (!used.has(id as Tile)) return id as Tile;
    }
    throw new ShadowDesyncError("136枚を使い切った");
  }

  /** A free id answering to `pai`; if all matching copies sit in opponent
   *  hands as placeholders, retype one (invisible) and take its id. `keep`
   *  ids are already claimed by the caller and may not be (re)taken. */
  #allocByPai(pai: string, keep: readonly Tile[] = []): Tile {
    const t = this.#t!;
    const used = this.#usedIds();
    const ty = paiToType(pai);
    const copies = [ty * 4, ty * 4 + 1, ty * 4 + 2, ty * 4 + 3] as Tile[];
    const matching = copies.filter((id) => idMatchesPai(id, pai) && !keep.includes(id));
    for (const id of matching) {
      if (!used.has(id)) return id;
    }
    for (const id of matching) {
      const holder = this.#placeholderHolder(id);
      if (holder !== null) {
        const hand = t.hands[holder];
        // Retype the placeholder to any wholly-free id, freeing this one.
        let free: Tile | null = null;
        for (let cand = 0; cand < 136; cand++) {
          if (!used.has(cand as Tile)) {
            free = cand as Tile;
            break;
          }
        }
        if (free === null) throw new ShadowDesyncError("代替牌が尽きた");
        hand[hand.indexOf(id)] = free;
        return id;
      }
    }
    throw new ShadowDesyncError(`${pai} の5枚目が現れた`);
  }

  /** The opponent hand holding `id` as a placeholder, if any. */
  #placeholderHolder(id: Tile): Seat | null {
    const t = this.#t!;
    for (const s of SEATS) {
      if (s === this.me) continue;
      if (t.hands[s].includes(id)) return s;
    }
    return null;
  }

  /** A concealed opponent tile turns public: make sure a matching id is IN
   *  their hand (swapping out a placeholder), and return it. `keep` protects
   *  ids already claimed by the same call from being overwritten or reused. */
  #materialize(seat: Seat, pai: string, keep: readonly Tile[] = []): Tile {
    const t = this.#t!;
    const hand = t.hands[seat];
    const already = hand.find((x) => idMatchesPai(x, pai) && !keep.includes(x));
    if (already !== undefined) return already;
    let slot = -1;
    for (let i = hand.length - 1; i >= 0; i--) {
      if (!keep.includes(hand[i])) {
        slot = i;
        break;
      }
    }
    if (slot < 0) throw new ShadowDesyncError(`席${seat} に空き枠がない (${pai})`);
    const id = this.#allocByPai(pai, keep);
    hand[slot] = id;
    return id;
  }

  /** Consumed tiles of a call: exact ids from our real hand, or materialized
   *  ones for an opponent (distinct even for equal strings). */
  #claimConsumed(actor: Seat, pais: string[]): Tile[] {
    const t = this.#t!;
    const out: Tile[] = [];
    for (const pai of pais) {
      if (actor === this.me) {
        const id = t.hands[actor].find((x) => idMatchesPai(x, pai) && !out.includes(x));
        if (id === undefined) throw new ShadowDesyncError(`晒す牌 ${pai} が手中にない`);
        out.push(id);
      } else {
        out.push(this.#materialize(actor, pai, out));
      }
    }
    return out;
  }

  // -------------------------------------------------------------------------
  // decoded-observation resync
  // -------------------------------------------------------------------------

  /**
   * Make public ids EXACT against the server's decoded observation blob (own
   * hand, drawn tile, rivers, indicators, melds where the shape is readable).
   * Type-or-red level disagreement is a real desync and throws; unreadable or
   * differently-conventioned sections are skipped — they are already correct
   * at the type level, which is all any consumer distinguishes.
   */
  syncFromDecoded(decoded: unknown): void {
    if (!this.#t || this.me < 0) return;
    if (typeof decoded !== "object" || decoded === null) return;
    const d = decoded as Record<string, unknown>;
    const me = this.me as Seat;
    const t = this.#t;

    // Public structures first: `#relocate` can swap a wanted id out of ANY
    // hand (ours included), but the own-hand pass below refuses ids that are
    // still sitting in a public slot — which they legitimately are until the
    // rivers/melds/indicators have been re-paired onto the server's ids.
    const ind = d.dora_indicators;
    if (Array.isArray(ind) && ind.every((x) => typeof x === "number")) {
      const ours = t.board.indicators;
      if (ind.length !== ours.length) {
        throw new ShadowDesyncError(`ドラ表示牌の枚数不一致: ${ind.length} vs ${ours.length}`);
      }
      ind.forEach((id, i) => this.#relocate(ours[i], id as Tile, `ドラ表示牌#${i}`));
    }

    const disc = d.discards;
    if (Array.isArray(disc) && disc.length === 4) {
      for (const s of SEATS) {
        const D = disc[s];
        if (!Array.isArray(D) || !D.every((x) => typeof x === "number")) continue;
        const river = t.board.rivers[s];
        // Two river conventions are conceivable: called tiles kept or dropped.
        let entries = river;
        if (D.length !== river.length) {
          entries = river.filter((r) => r.calledBy === undefined);
          if (D.length !== entries.length) continue; // unknown convention: skip
        }
        entries.forEach((r, i) => this.#relocate(r.tile, D[i] as Tile, `席${s} 河#${i}`));
      }
    }

    const melds = d.melds;
    if (Array.isArray(melds) && melds.length === 4) {
      const meldIds = (raw: unknown): Tile[] | null => {
        const ids = Array.isArray(raw)
          ? raw
          : (raw && typeof raw === "object" && Array.isArray((raw as { tiles?: unknown }).tiles))
          ? (raw as { tiles: unknown[] }).tiles
          : null;
        return ids && ids.every((x) => typeof x === "number") ? ids as Tile[] : null;
      };
      // A meld's type signature is invariant under `#relocate`, so it stays a
      // valid matching key while the swaps below rewrite the live arrays.
      const sig = (ids: readonly Tile[]) => ids.map(tileType).sort((a, b) => a - b).join(",");
      for (const s of SEATS) {
        const M = melds[s];
        if (!Array.isArray(M) || M.length !== t.melds[s].length) continue;
        // Match by content, NOT by index: a 加槓 upgrade moves the meld to the
        // END of our list (mjrender BoardState.applyMeld splices then pushes)
        // while the server keeps it in the pon's original slot, so index
        // pairing hands `#relocate` two different melds and kills the kyoku.
        const taken = M.map(() => false);
        t.melds[s].forEach((meld, i) => {
          const mine = sig(meld.tiles);
          let j = -1;
          let unreadable = false;
          for (let x = 0; x < M.length && j < 0; x++) {
            if (taken[x]) continue;
            const ids = meldIds(M[x]);
            if (!ids) unreadable = true;
            else if (ids.length === meld.tiles.length && sig(ids) === mine) j = x;
          }
          if (j < 0) {
            // Unreadable shape: skip, the type level already holds. But a
            // READABLE slot that matches nothing is a real divergence — the
            // old index pairing threw here, and staying silent would leave the
            // champion reading opponents off melds that are not on the table.
            if (unreadable) return;
            throw new ShadowDesyncError(`席${s} 副露#${i}: 一致する副露がない`);
          }
          taken[j] = true;
          const theirsIds = meldIds(M[j])!;
          // Pair only the set differences: pairing full sorted snapshots
          // would go stale as each swap rewrites the live array.
          const extra = meld.tiles.filter((x) => !theirsIds.includes(x)).sort((a, b) => a - b);
          const missing = theirsIds.filter((x) => !meld.tiles.includes(x)).sort((a, b) => a - b);
          extra.forEach((id, k) => this.#relocate(id, missing[k], `席${s} 副露#${i}`));
          meld.tiles.sort((a, b) => a - b); // buildMeld's invariant
        });
      }
    }

    const hands = d.hands;
    if (Array.isArray(hands)) {
      const mine = hands[me];
      if (Array.isArray(mine) && mine.length > 0 && mine.every((x) => typeof x === "number")) {
        this.#syncOwnHand(mine as Tile[]);
      } else {
        // Orientation is unverified beyond seat 0: a populated hand at any
        // OTHER index would mean the arrays are seat-relative. Refuse to guess.
        const populated = hands.findIndex((h) => Array.isArray(h) && h.length > 0);
        if (populated >= 0 && populated !== me) {
          throw new ShadowDesyncError(
            `decoded.hands[${populated}] に手牌がある (自席は ${me}) — 配列の向きが絶対席でない疑い`,
          );
        }
      }
    }
    if (typeof d.drawn_tile === "number") this.#drawn = d.drawn_tile as Tile;
  }

  /** Wholesale own-hand replacement in the server's exact ids AND order. */
  #syncOwnHand(mine: Tile[]): void {
    const t = this.#t!;
    const me = this.me as Seat;
    const hand = t.hands[me];
    if (mine.length !== hand.length) {
      throw new ShadowDesyncError(`手牌枚数の不一致: ${mine.length} vs ${hand.length}`);
    }
    const key = (id: Tile) => `${tileType(id)}${this.#cfg.akaIds.has(id) ? "r" : ""}`;
    const mult = (ids: readonly Tile[]) => {
      const m = new Map<string, number>();
      for (const id of ids) m.set(key(id), (m.get(key(id)) ?? 0) + 1);
      return m;
    };
    const a = mult(mine), b = mult(hand);
    for (const [k, n] of a) {
      if (b.get(k) !== n) throw new ShadowDesyncError(`手牌の種類不一致 (${k})`);
    }
    const surplus = hand.filter((id) => !mine.includes(id));
    for (const id of mine) {
      if (hand.includes(id)) continue;
      const holder = this.#placeholderHolder(id);
      if (holder !== null) {
        const h = t.hands[holder];
        h[h.indexOf(id)] = surplus.pop()!;
      } else if (this.#usedIds().has(id)) {
        throw new ShadowDesyncError(`手牌の牌 ${id} が既に公開されている`);
      }
    }
    hand.length = 0;
    hand.push(...mine);
  }

  /** Put `id` where `oldId` currently is (river/meld/indicator), preserving
   *  the permutation: same type+red required, and any holder of `id` takes
   *  `oldId` in exchange (only a concealed placeholder may be holding it). */
  #relocate(oldId: Tile, id: Tile, what: string): void {
    if (oldId === id) return;
    if (
      tileType(oldId) !== tileType(id) || this.#cfg.akaIds.has(oldId) !== this.#cfg.akaIds.has(id)
    ) {
      throw new ShadowDesyncError(`${what}: 牌 ${oldId} と ${id} の種類が食い違う`);
    }
    const t = this.#t!;
    // A plain swap keeps every invariant — whether `id` was free, a concealed
    // placeholder, or another public slot being positionally re-paired in the
    // same sync pass: walk the board, exchange the two.
    for (const s of SEATS) {
      const h = t.hands[s];
      for (let i = 0; i < h.length; i++) h[i] = h[i] === oldId ? id : h[i] === id ? oldId : h[i];
      for (const r of t.board.rivers[s]) {
        r.tile = r.tile === oldId ? id : r.tile === id ? oldId : r.tile;
      }
      for (const m of t.melds[s]) {
        for (let i = 0; i < m.tiles.length; i++) {
          m.tiles[i] = m.tiles[i] === oldId ? id : m.tiles[i] === id ? oldId : m.tiles[i];
        }
        m.calledTile = m.calledTile === oldId ? id : m.calledTile === id ? oldId : m.calledTile;
      }
    }
    const ind = t.board.indicators;
    for (let i = 0; i < ind.length; i++) {
      ind[i] = ind[i] === oldId ? id : ind[i] === id ? oldId : ind[i];
    }
  }

  // -------------------------------------------------------------------------

  #seatOf(v: unknown): Seat {
    const n = Number(v);
    if (!Number.isInteger(n) || n < 0 || n > 3) {
      throw new ShadowDesyncError(`席番号が読めない: ${v}`);
    }
    return n as Seat;
  }

  #strings(v: unknown, n: number): string[] {
    if (!Array.isArray(v) || v.length !== n) {
      throw new ShadowDesyncError(`晒し牌リストが読めない: ${JSON.stringify(v)}`);
    }
    return v.map(String);
  }
}
