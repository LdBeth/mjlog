// The baseline CPU: efficiency-first, with a push/fold gate driven by
// mjrender's danger assessor and a dojo-aware filter on which actions it will
// even consider.
//
// The shape of the decision — score every legal action, take the argmax — is
// the same shape a learned policy will use, so replacing this with a network
// changes `score`, not the plumbing. Every magic number lives in one weights
// object so the thing can be tuned or ablated without surgery.
//
// It is deterministic by default: no Date.now, no Math.random. `epsilon` (off
// unless asked for) exists so self-play can generate varied trajectories.

import type { DangerLevel } from "mjrender/danger.ts";
import type { Meld, Tile } from "mjrender/model.ts";
import { countsFromTiles, shanten, ukeireTypes } from "mjrender/shanten.ts";
import { doraFromIndicatorType, rankOfType, suitOfType, tileType } from "mjrender/tiles.ts";
import { isHonor } from "../tiles.ts";
import { confirmedYaku } from "../hand.ts";
import type { Observation } from "../observe.ts";
import type { SyncPolicy } from "../policy.ts";
import type { Rng } from "../rng.ts";
import { sfc32 } from "../rng.ts";
import type { Action } from "../types.ts";

export interface HeuristicWeights {
  /** Per shanten step. Deliberately dominates every other efficiency term. */
  shanten: number;
  /** Per live ukeire tile (copies of a useful type not yet visible). */
  ukeire: number;
  /** Per distinct ukeire type — breadth of the wait, not just its count. */
  ukeireType: number;
  /** Per dora (including aka) retained in the concealed hand. */
  dora: number;
  /** Retaining a concealed value-honor pair: a yaku seed, and a safe holding. */
  yakuhaiPair: number;
  /** Per lone honor still held, scaled by how late it is. */
  isolatedHonor: number;
  /** Cost of a discard by assessed danger level. */
  danger: Record<DangerLevel, number>;
  /**
   * 第一打字牌切り and 不聴時ドラ切り, as costs rather than hard filters: a
   * ledger entry is a price, not a wall, and the engine will happily let the
   * CPU pay it if literally every alternative is worse. Both are set above a
   * shanten step so in practice they decide the discard.
   */
  firstHonor: number;
  notenDora: number;
  /**
   * 片和了り. Set above a shanten step on purpose: when a clean tenpai is
   * available this only steers between equals, but when every tenpai is split
   * and riichi cannot cure it, outweighing `shanten` is what makes the policy
   * break the shape and give the hand up — which for an open hand is the only
   * move left.
   */
  katagari: number;
  /** Any tedashi once 不聴時ドラ切り has been called on us. */
  tsumogiriLock: number;
  /** Efficiency is scaled by this while folding. */
  foldEfficiency: number;
  /** Danger is scaled by this while folding. */
  foldDanger: number;
}

export const DEFAULT_WEIGHTS: HeuristicWeights = {
  shanten: 1000,
  ukeire: 12,
  ukeireType: 4,
  dora: 60,
  yakuhaiPair: 40,
  isolatedHonor: 6,
  danger: { "安全": 0, "危険度低": 30, "危険度中": 90, "危険度高": 200 },
  firstHonor: 4000,
  notenDora: 2500,
  katagari: 1500,
  tsumogiriLock: 2500,
  foldEfficiency: 0.05,
  foldDanger: 10,
};

export interface HeuristicOptions {
  weights?: Partial<HeuristicWeights>;
  /** 喰いタン. Only affects whether an open tanyao counts as a confirmed yaku. */
  kuitan?: boolean;
  /**
   * Obey the dojo 禁じ手 the CPU can see for itself (no 明槓, no 後付け calls,
   * no first-turn honor discard, no 不聴時ドラ切り, no 地獄単騎/引っかけ riichi).
   * This is an approximation of `penalty/rules.ts`, not a use of it: the ledger
   * judges committed actions, and a policy has to decide beforehand.
   */
  dojo?: boolean;
  /** Probability of taking a uniformly random legal action instead. */
  epsilon?: number;
}

/** Everything derived once per decision and shared by the per-action scorers. */
interface Ctx {
  obs: Observation;
  open: number;
  closed: boolean;
  doraTypes: Set<number>;
  valueHonors: Set<number>;
  folding: boolean;
  /** Riichi is on the table this turn — which puts a yaku on every wait. */
  canRiichi: boolean;
  eff: number;
  def: number;
}

export class HeuristicPolicy implements SyncPolicy {
  readonly name: string;
  readonly sync = true;
  private w: HeuristicWeights;
  private kuitan: boolean;
  private dojo: boolean;
  private epsilon: number;
  private rng: Rng;

  constructor(name: string, seed: number, opts: HeuristicOptions = {}) {
    this.name = name;
    // `danger` is a nested record: spreading `opts.weights` wholesale would let
    // a partial override drop levels, and a missing level scores every discard
    // NaN — which silently degrades to "discard the first tile in hand".
    this.w = {
      ...DEFAULT_WEIGHTS,
      ...opts.weights,
      danger: { ...DEFAULT_WEIGHTS.danger, ...opts.weights?.danger },
    };
    this.kuitan = opts.kuitan ?? true;
    this.dojo = opts.dojo ?? true;
    this.epsilon = opts.epsilon ?? 0;
    this.rng = sfc32(seed);
  }

  reset(seed: number): void {
    this.rng = sfc32(seed);
  }

  decide(obs: Observation): Action {
    const { legal } = obs;
    if (legal.length === 1) return legal[0];
    if (this.epsilon > 0 && this.rng.float() < this.epsilon) {
      return legal[this.rng.int(legal.length)];
    }

    // A win is always taken. 片和了り and the 8000点未満 rule could argue
    // otherwise; neither is worth modelling before the hand values are.
    const win = legal.find((a) => a.t === "tsumo") ?? legal.find((a) => a.t === "ron");
    if (win) return win;

    const ctx = this.context(obs);

    const kan = this.chooseKan(ctx, legal);
    if (kan) return kan;

    const call = this.chooseCall(ctx, legal);
    if (call) return call;

    const discard = this.chooseDiscard(ctx, legal);
    if (discard) return discard;

    return legal.find((a) => a.t === "pass") ?? legal[0];
  }

  // ---------------------------------------------------------------- context

  private context(obs: Observation): Ctx {
    const doraTypes = new Set(obs.doraIndicators.map((t) => doraFromIndicatorType(tileType(t))));
    const valueHonors = new Set([31, 32, 33, obs.seatWind, obs.roundWind]);
    const folding = this.shouldFold(obs);
    return {
      obs,
      open: obs.melds[0].length,
      closed: obs.melds[0].every((m) => m.kind === "ankan"),
      doraTypes,
      valueHonors,
      folding,
      canRiichi: obs.legal.some((a) => a.t === "discard" && a.riichi),
      eff: folding ? this.w.foldEfficiency : 1,
      def: folding ? this.w.foldDanger : 1,
    };
  }

  /**
   * Push/fold. `push` is how much this hand is worth carrying forward, `pressure`
   * is how loud the table is. Folding is not all-or-nothing — it re-weights the
   * discard score rather than switching to a different algorithm.
   */
  private shouldFold(obs: Observation): boolean {
    // Committed: after riichi the only legal discard is the drawn tile anyway.
    if (obs.riichi[0]) return false;

    const pressure = this.pressure(obs);
    if (pressure === 0) return false;

    let push = obs.shanten <= 0 ? 1.0 : obs.shanten === 1 ? 0.45 : obs.shanten === 2 ? 0.15 : 0;
    push += 0.12 * obs.doraCount;
    // Late and far from tenpai is not a hand worth defending with.
    if (obs.shanten >= 2 && obs.junme >= 10) push = 0;
    // A dealer has more to lose by folding (連荘) — nudge, don't override.
    if (obs.seatWind === 27) push += 0.08;

    // 持ち点8000未満になる打ち方禁止. The ledger charges for *being* short, and
    // by then it is too late to play differently — so the buffer, not the
    // breach, is what the policy watches. A deal-in runs ~6000, so a stack
    // inside that of the line is one bad discard away from the violation.
    const buffer = obs.scores[0] - 8000;
    if (buffer < 6000) push *= 0.35;
    else if (buffer < 12000) push *= 0.7;

    return push < 0.5 * pressure;
  }

  /** Threat volume: a declared riichi counts full, a loud open hand counts half. */
  private pressure(obs: Observation): number {
    let p = 0;
    for (let s = 1; s < 4; s++) if (obs.riichi[s]) p += 1;
    // Danger entries carry the furo threats the assessor decided were real.
    const furoSeats = new Set<number>();
    for (const d of obs.danger.values()) {
      for (const detail of d.details) {
        if (detail.kind === "furo") furoSeats.add(detail.seat);
      }
    }
    p += 0.5 * furoSeats.size;
    return p;
  }

  // ---------------------------------------------------------------- discard

  private chooseDiscard(ctx: Ctx, legal: Action[]): Action | null {
    const discards = legal.filter((a): a is Extract<Action, { t: "discard" }> => a.t === "discard");
    if (discards.length === 0) return null;

    // One entry per distinct tile — the riichi flag is decided separately.
    const byTile = new Map<Tile, Extract<Action, { t: "discard" }>[]>();
    for (const d of discards) {
      const g = byTile.get(d.tile);
      if (g) g.push(d);
      else byTile.set(d.tile, [d]);
    }

    const candidates = [...byTile.keys()];
    const shantenAfter = new Map<Tile, number>();
    let best = Infinity;
    for (const tile of candidates) {
      const s = this.shantenWithout(ctx, tile);
      shantenAfter.set(tile, s);
      if (s < best) best = s;
    }

    let bestTile = candidates[0];
    let bestScore = -Infinity;
    for (const tile of candidates) {
      const sh = shantenAfter.get(tile)!;
      // Ukeire is the expensive part (34 shanten probes); only the tiles that
      // actually hold the best shanten can win on it.
      const score = this.scoreDiscard(ctx, tile, sh, sh === best);
      if (score > bestScore || (score === bestScore && tile < bestTile)) {
        bestScore = score;
        bestTile = tile;
      }
    }

    const group = byTile.get(bestTile)!;
    const plain = group.find((d) => !d.riichi);
    const riichi = group.find((d) => d.riichi);
    if (riichi && !this.riichiBanned(ctx, bestTile)) {
      // A split wait must not be left damaten. Riichi is the cure and the
      // dojo's own prescription (役なしなら即リーチ), so it overrides the
      // ordinary "is this worth declaring" judgement — but never a 禁じ手.
      const mustCure = this.dojo && (ctx.obs.discardInfo.get(bestTile)?.katagari ?? false);
      if (mustCure || this.wantRiichi(ctx, bestTile)) return riichi;
    }
    return plain ?? group[0];
  }

  private handWithout(ctx: Ctx, tile: Tile): Tile[] {
    const rest = [...ctx.obs.hand];
    rest.splice(rest.lastIndexOf(tile), 1);
    return rest;
  }

  private shantenWithout(ctx: Ctx, tile: Tile): number {
    // The referee already worked this out for every legal discard.
    const known = ctx.obs.discardInfo.get(tile);
    if (known) return known.shanten;
    const counts = countsFromTiles(this.handWithout(ctx, tile));
    return shanten(counts, ctx.open, ctx.closed);
  }

  private scoreDiscard(ctx: Ctx, tile: Tile, sh: number, wideOpen: boolean): number {
    const { obs } = ctx;
    const rest = this.handWithout(ctx, tile);
    const counts = countsFromTiles(rest);

    let eff = -sh * this.w.shanten;

    if (wideOpen) {
      const types = ukeireTypes(counts, ctx.open, ctx.closed, sh);
      let live = 0;
      for (const ty of types) live += this.liveCopies(obs, ty, counts);
      eff += live * this.w.ukeire + types.length * this.w.ukeireType;
    }

    // Value kept. Melded dora is constant across candidates, so hand-only is
    // enough to rank them.
    let dora = 0;
    for (const t of rest) {
      if (ctx.doraTypes.has(tileType(t))) dora++;
      if (obs.akaIds.has(t)) dora++;
    }
    eff += dora * this.w.dora;

    for (let ty = 0; ty < 34; ty++) {
      if (counts[ty] >= 2 && ctx.valueHonors.has(ty)) eff += this.w.yakuhaiPair;
      if (counts[ty] === 1 && isHonor(ty)) eff -= this.w.isolatedHonor * Math.min(obs.junme, 12);
    }

    const level = obs.danger.get(tileType(tile))?.level ?? "安全";
    const risk = this.w.danger[level];

    // The dojo cost is deliberately outside `ctx.eff`: folding must not make a
    // 禁じ手 cheap. The ledger charges the same either way.
    return ctx.eff * eff - ctx.def * risk - this.dojoCost(ctx, tile, sh);
  }

  /** What the ledger would charge for this discard, in score units. */
  private dojoCost(ctx: Ctx, tile: Tile, sh: number): number {
    if (!this.dojo) return 0;
    const { obs } = ctx;
    const ty = tileType(tile);
    let cost = 0;

    // 第一打字牌切り — our river is still empty when we pick this discard.
    if (obs.rivers[0].length === 0 && isHonor(ty)) cost += this.w.firstHonor;

    // 不聴時ドラ切り. Indicator dora only: the aka 5p may be cut before tenpai.
    // 例外: an honor dora already twice in the rivers is spent.
    if (sh > 0 && ctx.doraTypes.has(ty)) {
      if (!(isHonor(ty) && this.visibleCount(obs, ty) >= 2)) cost += this.w.notenDora;
    }

    // 片和了り, but only when riichi is not on offer: riichi is itself a yaku,
    // so declaring it makes every wait scoring and the shape stops being split.
    if (!ctx.canRiichi && obs.discardInfo.get(tile)?.katagari) cost += this.w.katagari;

    // ドラ切りをポンされた後の手出し. `legal.ts` will not stop us — the dojo
    // takes the payment instead — so the price has to be paid here.
    if (obs.tsumogiriLock && tile !== obs.drawn) cost += this.w.tsumogiriLock;

    return cost;
  }

  /** Copies of a type this seat can see: rivers, melds, indicators, own hand. */
  private visibleCount(obs: Observation, type: number): number {
    let n = 0;
    for (const river of obs.rivers) {
      // Skip entries that were called away — the meld loop below counts those.
      for (const e of river) if (e.calledBy === undefined && tileType(e.tile) === type) n++;
    }
    for (const melds of obs.melds) {
      for (const m of melds) {
        for (const t of m.tiles) if (tileType(t) === type) n++;
      }
    }
    for (const t of obs.doraIndicators) if (tileType(t) === type) n++;
    for (const t of obs.hand) if (tileType(t) === type) n++;
    return n;
  }

  /** Copies of `type` this seat cannot see, minus the ones it already holds. */
  private liveCopies(obs: Observation, type: number, counts: number[]): number {
    const u = obs.ukeire.find((x) => x.type === type);
    if (u) return u.live;
    return Math.max(0, 4 - counts[type]);
  }

  // ----------------------------------------------------------------- riichi

  /**
   * The waits this hand is on. Approximate when the chosen discard is not the
   * drawn tile: `obs.waits` describes the resting hand, and only a tsumogiri
   * leaves that shape untouched.
   */
  private waitsOf(obs: Observation): number[] {
    return obs.waits.length ? obs.waits : obs.ukeire.map((u) => u.type);
  }

  /** Riichi declarations the dojo forbids outright. Never overridden. */
  private riichiBanned(ctx: Ctx, discard: Tile): boolean {
    if (!this.dojo) return false;
    const { obs } = ctx;
    const waits = this.waitsOf(obs);
    if (waits.length === 0) return true;
    const live = this.liveWaits(obs, waits);

    // 地獄単騎: a lone honor wait with most copies already gone.
    if (waits.length === 1 && isHonor(waits[0]) && live <= 1) return true;

    // 即引っかけ: a wait one suji away from the tile being cut. The dojo asks
    // for a 巡 of daylight; a policy with no memory of when the shape arrived
    // simply declines.
    const dt = tileType(discard);
    if (!isHonor(dt)) {
      return waits.some((w) =>
        !isHonor(w) && suitOfType(w) === suitOfType(dt) &&
        Math.abs(rankOfType(w) - rankOfType(dt)) === 3
      );
    }
    return false;
  }

  private wantRiichi(ctx: Ctx, _discard: Tile): boolean {
    const { obs } = ctx;
    const waits = this.waitsOf(obs);
    if (waits.length === 0) return false;
    if (this.liveWaits(obs, waits) === 0) return false; // 純カラ — nothing to win on

    // Furiten riichi is legal in the dojo but rarely what you want.
    if (obs.furiten.permanent || obs.furiten.temporary) return false;

    return obs.wallRemaining >= 4;
  }

  private liveWaits(obs: Observation, waits: number[]): number {
    return waits.reduce((n, ty) => n + (obs.ukeire.find((u) => u.type === ty)?.live ?? 0), 0);
  }

  // ------------------------------------------------------------------ calls

  private chooseCall(ctx: Ctx, legal: Action[]): Action | null {
    const calls = legal.filter((a) => a.t === "pon" || a.t === "chi");
    if (calls.length === 0) return null;
    if (ctx.folding) return null;

    let best: Action | null = null;
    let bestSh = ctx.obs.shanten;
    for (const a of calls) {
      if (a.t !== "pon" && a.t !== "chi") continue;
      const after = this.shantenAfterCall(ctx, a);
      if (after === null) continue;
      if (after < bestSh) {
        bestSh = after;
        best = a;
      }
    }
    return best;
  }

  /** Shanten after taking the call, or null if the call is one we refuse. */
  private shantenAfterCall(
    ctx: Ctx,
    a: Extract<Action, { t: "pon" | "chi" }>,
  ): number | null {
    const { obs } = ctx;
    const rest = [...obs.hand];
    for (const t of a.tiles) {
      const i = rest.lastIndexOf(t);
      if (i < 0) return null;
      rest.splice(i, 1);
    }
    const meld: Meld = {
      kind: a.t,
      who: obs.seat,
      fromWho: obs.seat,
      tiles: [...a.tiles, a.called].sort((x, y) => x - y),
      calledTile: a.called,
    };
    const melds = [...obs.melds[0], meld];

    if (this.dojo) {
      // 後付け禁止: a call must leave a yaku that holds in every completion.
      const yaku = confirmedYaku(rest, melds, ctx.valueHonors, this.kuitan);
      if (yaku.length === 0) return null;
    }

    return shanten(countsFromTiles(rest), melds.length, false);
  }

  // -------------------------------------------------------------------- kan

  private chooseKan(ctx: Ctx, legal: Action[]): Action | null {
    const { obs } = ctx;

    const ankan = legal.find((a) => a.t === "ankan");

    if (!this.dojo) {
      // Off the dojo leash, a kan is just value: take whichever is offered.
      return ankan ?? legal.find((a) => a.t === "kakan") ??
        legal.find((a) => a.t === "daiminkan") ?? null;
    }

    // 明槓 (大明槓・加槓) is a 禁じ手, and a bad idea besides: it opens a dora
    // for a table you cannot see. 暗槓 is allowed, but only 門前 + 聴牌 and
    // only when it leaves the wait alone.
    if (!ankan || ankan.t !== "ankan") return null;
    if (!ctx.closed || obs.shanten > 0) return null;
    if (this.kanChangesWait(ctx, ankan.type)) return null;
    return ankan;
  }

  /** Does declaring this ankan change what the hand is waiting on? */
  private kanChangesWait(ctx: Ctx, type: number): boolean {
    const { obs } = ctx;
    const before = new Set(obs.waits);

    const rest: Tile[] = [];
    let removed = 0;
    for (const t of obs.hand) {
      if (tileType(t) === type && removed < 4) removed++;
      else rest.push(t);
    }
    if (removed < 4) return true; // not actually a concealed four — refuse

    const counts = countsFromTiles(rest);
    const sh = shanten(counts, ctx.open + 1, ctx.closed);
    if (sh !== 0) return true;
    const after = ukeireTypes(counts, ctx.open + 1, ctx.closed, sh);
    if (after.length !== before.size) return true;
    return after.some((w) => !before.has(w));
  }
}
