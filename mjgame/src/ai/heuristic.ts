// The baseline CPU: efficiency-first, with a push/fold gate driven by
// mjrender's danger assessor and a dojo-aware filter on which actions it will
// even consider.
//
// THE COMPLIANCE FILTER. `finalStandings` ranks one ledger entry below every
// clean seat regardless of score, so a 禁じ手 is not an expensive move — it is a
// losing one, and pricing it (which is all `dojoCost` can do) only decides how
// expensive the loss was. So when the referee's own predicates are reachable
// (`Observation.preview`, wired by every real driver) this policy asks them
// first and DROPS every action that would be ledgered, choosing by score only
// among what is left. `dojoCost` stays for the fallthrough: when literally every
// candidate is charged, something has to be picked, and the prices are how.
//
// The filter lives where a subclass cannot reach around it: the discard
// candidate set is narrowed inside `chooseDiscard` (private) before scoring, and
// the call/kan hooks (protected, and overridden by the C7 planner) only PROPOSE
// — `decide` vetoes what they return.
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
import { rankOfType, suitOfType, tileType } from "mjrender/tiles.ts";
import { countsFromTiles, shanten, ukeireTypes } from "../kernel.ts";
import { isHonor, isYaochu } from "../tiles.ts";
import type { Observation } from "../observe.ts";
import type { ActionPreview } from "../penalty/preview.ts";
import { pickLesserEvil, violationPoints } from "../penalty/preview.ts";
import type { SyncPolicy } from "../policy.ts";
import type { ConsumerParams } from "./consumer.ts";
import { scoreDiscard as consumeEvidence } from "./consumer.ts";
import type { ContextEvidence, EvidenceHooks } from "./evidence.ts";
import { assembleCandidate, assembleContext } from "./evidence.ts";
import { doraTypesOf, publicUnseen, valueHonorsOf } from "./planner.ts";
import type { StandingsWeights } from "./standings.ts";
import { standingsScales } from "./standings.ts";
import type { Rng } from "../rng.ts";
import { sfc32 } from "../rng.ts";
import type { Action, Violation } from "../types.ts";

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
  /**
   * 後付け: the discard would leave an OPEN hand tenpai on nothing that scores.
   * Weighted like `firstHonor` because it buys exactly that — a Tier A medium
   * ledger entry — and unlike 片和了り there is no riichi available to cure it.
   */
  yakulessTenpai: number;
  /** Any tedashi once 不聴時ドラ切り has been called on us. */
  tsumogiriLock: number;
  /** Efficiency is scaled by this while folding. */
  foldEfficiency: number;
  /** Danger is scaled by this while folding. */
  foldDanger: number;
  /**
   * 順位効用. Absent by DEFAULT, and absent means off: every scale the layer
   * produces is 1 and the policy is bit-for-bit the point-EV agent it has always
   * been. Present, it prices this seat's points by what they do to the FINAL
   * PLACEMENT — see `standings.ts` — and reaches the decision through exactly
   * two multipliers, on the push/fold gate and on the price of danger.
   */
  standings?: StandingsWeights;
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
  yakulessTenpai: 4000,
  tsumogiriLock: 2500,
  foldEfficiency: 0.05,
  foldDanger: 10,
};

export interface HeuristicOptions {
  weights?: Partial<HeuristicWeights>;
  /** 喰いタン. Only affects whether an open tanyao counts as a confirmed yaku. */
  kuitan?: boolean;
  /**
   * Obey the dojo 禁じ手. With a referee preview on the Observation this is
   * literal — every action the ledger would charge for is dropped from the
   * choice set by `penalty/preview.ts`, which runs the ledger's OWN predicates
   * rather than an imitation of them. Without one (a hand-built Observation, a
   * driver that passed no `DojoConfig`) it degrades to the older behaviour: the
   * rules the policy can see for itself, priced by `dojoCost`.
   */
  dojo?: boolean;
  /** Probability of taking a uniformly random legal action instead. */
  epsilon?: number;
  /**
   * M9. The learned consumer of the evidence vector (`consumer.ts`). ABSENT BY
   * DEFAULT, and absent means the hand-written arithmetic below runs unchanged,
   * bit for bit. Present, it replaces the discard score CORE and nothing else:
   * the compliance filter, the `dojoCost` fallthrough pricing and the riichi
   * decision are outside it on both paths, and `initFromWeights` makes the two
   * paths agree exactly.
   */
  consumer?: ConsumerParams;
}

/**
 * Everything derived once per decision and shared by the per-action scorers.
 *
 * Exported because the subclass hooks below (`riskOf`, `drawBonus`, `keepBonus`)
 * take it: a policy that reads hidden information overrides those, and cannot
 * name their parameter type otherwise.
 */
export interface Ctx {
  obs: Observation;
  open: number;
  closed: boolean;
  doraTypes: Set<number>;
  valueHonors: Set<number>;
  /**
   * Copies of each of the 34 types this seat cannot see (`publicUnseen`).
   *
   * THE ONLY LIVENESS ACCOUNT. Every "how many are left" question in the
   * decision reads this vector: the ukeire count, the 不聴時ドラ切り exception's
   * visible-copy test (`4 − unseen`), and the planner's availability model.
   * There used to be two — `Observation.ukeire[].live` for a type the resting
   * hand happened to accept and `4 − own copies` for everything else — which
   * priced the same tile differently depending on which candidate was asking,
   * and always upward, because the second one counts neither the rivers nor the
   * melds nor the indicators.
   */
  unseen: number[];
  folding: boolean;
  /** Riichi is on the table this turn — which puts a yaku on every wait. */
  canRiichi: boolean;
  eff: number;
  def: number;
}

/**
 * Everything the consumer path needs that is fixed for one decision: the bound
 * hooks and the per-decision context evidence, assembled once and shared by
 * every candidate. Null whenever no consumer is set, which is what makes the
 * old path free.
 */
interface EvidenceRun {
  hooks: EvidenceHooks;
  context: ContextEvidence;
}

/**
 * The per-decision memo of the four PURE per-observation quantities.
 *
 * All four are closed forms over one `Observation` and were computed two or
 * three times per decision — `shouldFold` from `context` and again from the C7
 * planner's `updatePlan`, `pressureOf`/`bufferScale`/`standingsOf` from inside
 * `shouldFold` and again from the evidence assembler. Keyed on the Observation's
 * IDENTITY rather than cleared by hand: `decide` receives one object and every
 * caller inside the decision is handed that same object, so a new decision
 * invalidates the memo by construction and nothing has to remember to.
 *
 * It memoizes the CALL, never the method: `pressureOf` and `bufferScale` are
 * hooks an augmented policy overrides, so the first call still dispatches
 * virtually and the memo holds whatever that policy answered.
 */
interface DecisionMemo {
  obs: Observation;
  pressure?: number;
  buffer?: number;
  standings?: { gain: number; risk: number };
  fold?: boolean;
}

export class HeuristicPolicy implements SyncPolicy {
  readonly name: string;
  readonly sync = true;
  protected w: HeuristicWeights;
  private kuitan: boolean;
  private dojo: boolean;
  private epsilon: number;
  private rng: Rng;
  /** M9's learned consumer, or null for the hand-written score. */
  private consumer: ConsumerParams | null;
  private memo: DecisionMemo | null = null;

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
    this.consumer = opts.consumer ?? null;
    this.rng = sfc32(seed);
  }

  /**
   * The evidence assembler's window onto this instance's own methods. Built as
   * closures rather than passed as a `this` reference so that the hook set is a
   * NAMED, minimal contract — and, more to the point, so that the calls dispatch
   * virtually: an `AugmentedHeuristic`'s `riskOf`/`pressureOf` overrides fill
   * the evidence automatically, with no cooperation from `evidence.ts`.
   */
  private evidenceHooks(): EvidenceHooks {
    return {
      handWithout: (ctx, tile) => this.handWithout(ctx, tile),
      riskOf: (ctx, tile) => this.riskOf(ctx, tile),
      drawBonus: (ctx, tile) => this.drawBonus(ctx, tile),
      keepBonus: (ctx, tile) => this.keepBonus(ctx, tile),
      pressureOf: (obs) => this.pressure(obs),
      bufferScale: (obs) => this.buffer(obs),
      standings: (obs) => this.standingsOf(obs),
    };
  }

  reset(seed: number): void {
    this.rng = sfc32(seed);
    this.memo = null;
  }

  /** The memo slot for this decision, freshly emptied when the board moved. */
  private cache(obs: Observation): DecisionMemo {
    const m = this.memo;
    if (m && m.obs === obs) return m;
    return (this.memo = { obs });
  }

  /** `pressureOf`, once per decision. */
  private pressure(obs: Observation): number {
    const m = this.cache(obs);
    return m.pressure ??= this.pressureOf(obs);
  }

  /** `bufferScale`, once per decision. */
  private buffer(obs: Observation): number {
    const m = this.cache(obs);
    return m.buffer ??= this.bufferScale(obs);
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

    // The veto sits HERE, not inside the hooks: `chooseKan` and `chooseCall` are
    // protected and a subclass (the C7 planner) overrides them, so a filter
    // applied inside them would be one the plan could talk its way out of.
    const kan = this.chooseKan(ctx, legal);
    if (kan && this.compliant(ctx, kan)) return kan;

    // 立直後カン見送り is the one rule that fires on an OMISSION: while in
    // riichi, passing up a kan that leaves the wait alone is itself the foul.
    const forced = this.mandatoryKan(ctx, legal);
    if (forced) return forced;

    const call = this.chooseCall(ctx, legal);
    if (call && this.compliant(ctx, call)) return call;

    const discard = this.chooseDiscard(ctx, legal);
    if (discard) return discard;

    return legal.find((a) => a.t === "pass") ?? legal[0];
  }

  // ------------------------------------------------------- compliance filter

  /** The referee's hypothetical judgement, when the driver wired one up. */
  private referee(ctx: Ctx): ActionPreview | undefined {
    return this.dojo ? ctx.obs.preview : undefined;
  }

  /** Would the ledger stay silent on this call/kan? True when it cannot be asked. */
  private compliant(ctx: Ctx, a: Action): boolean {
    const pv = this.referee(ctx);
    if (!pv) return true;
    if (a.t === "pon" || a.t === "chi") return pv.call(a).length === 0;
    if (a.t === "daiminkan") {
      return pv.call(a).length === 0 && pv.kan(a, ctx.obs.drawn).length === 0;
    }
    if (a.t === "ankan" || a.t === "kakan") return pv.kan(a, ctx.obs.drawn).length === 0;
    return true;
  }

  /**
   * The kan the dojo requires. Declining is charged (立直後カン見送り), so the
   * only question is whether accepting is charged too — and if both are, the
   * cheaper option wins with the tie going to declining, which is `pickLesserEvil`
   * called with the decline first.
   */
  private mandatoryKan(ctx: Ctx, legal: Action[]): Action | null {
    const pv = this.referee(ctx);
    if (!pv || !ctx.obs.riichi[0] || ctx.obs.drawn === null) return null;
    const skip = pv.skipKan(ctx.obs.drawn);
    if (skip.length === 0) return null;
    let best: { a: Action; vs: Violation[] } | null = null;
    for (const a of legal) {
      if (a.t !== "ankan") continue;
      const vs = pv.kan(a, ctx.obs.drawn);
      if (!best || violationPoints(vs) < violationPoints(best.vs)) best = { a, vs };
    }
    if (!best) return null;
    return pickLesserEvil(skip, best.vs) === "b" ? best.a : null;
  }

  // ---------------------------------------------------------------- context

  private context(obs: Observation): Ctx {
    const folding = this.shouldFold(obs);
    return {
      obs,
      open: obs.melds[0].length,
      closed: obs.melds[0].every((m) => m.kind === "ankan"),
      doraTypes: doraTypesOf(obs),
      valueHonors: valueHonorsOf(obs.roundWind, obs.seatWind),
      unseen: publicUnseen(obs),
      folding,
      canRiichi: obs.legal.some((a) => a.t === "discard" && a.riichi),
      eff: folding ? this.w.foldEfficiency : 1,
      // 順位効用 rides on `def` and not on `eff`: how rank-sensitive this seat's
      // points are is a statement about the PRICE OF DANGER, so a protected lead
      // pays more for every risky tile and folds earlier, while a hopeless
      // deficit pays less and pushes tiles a point-EV agent would never let go.
      // Off (the default) the factor is exactly 1 and this is the old value.
      def: (folding ? this.w.foldDanger : 1) * this.standingsOf(obs).risk,
    };
  }

  /**
   * 順位効用's two multipliers, or the neutral pair when the layer is off — which
   * it is unless a weights object asked for it, so this is `1 × 1` and changes
   * nothing by default.
   *
   * Memoized per decision (`DecisionMemo`): `Ctx.def`, the push/fold gate and
   * the evidence assembler all ask for it, and the closed form behind it is ten
   * `rankStats` evaluations — not free enough to run three times for one
   * discard.
   */
  private standingsOf(obs: Observation): { gain: number; risk: number } {
    const w = this.w.standings;
    if (!w) return { gain: 1, risk: 1 };
    const m = this.cache(obs);
    return m.standings ??= standingsScales(obs, w);
  }

  /**
   * Push/fold. `push` is how much this hand is worth carrying forward, `pressure`
   * is how loud the table is. Folding is not all-or-nothing — it re-weights the
   * discard score rather than switching to a different algorithm.
   *
   * Protected, not because the base policy shares it, but because a subclass
   * that runs work BEFORE `decide` (the C7 planner, which must not re-plan while
   * the hand is being abandoned) has no other way to ask. Pure — and memoized on
   * the Observation for exactly that reason: the planner asks, and then `context`
   * asks again for the same board.
   */
  protected shouldFold(obs: Observation): boolean {
    const m = this.cache(obs);
    return m.fold ??= this.computeFold(obs);
  }

  private computeFold(obs: Observation): boolean {
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

    push *= this.buffer(obs);

    // 順位効用: the same hand against the same table is worth pushing for a
    // different amount depending on what the points would DO. Off (the default)
    // both scales are 1 and this is the old `push < 0.5 * pressure`, exactly.
    const st = this.standingsOf(obs);
    return push * st.gain < 0.5 * pressure * st.risk;
  }

  /**
   * 持ち点8000未満になる打ち方禁止. The ledger charges for *being* short, and by
   * then it is too late to play differently — so the buffer, not the breach, is
   * what the policy watches: a stack within one deal-in of the line is one bad
   * discard away from the violation.
   *
   * HOOK. `expectedLoss` is what a deal-in is assumed to cost; the base policy
   * has no way to know, so it guesses. A subclass that can price the table
   * overrides this, computes the figure and calls `super` with it.
   */
  protected bufferScale(obs: Observation, expectedLoss = 6000): number {
    const buffer = obs.scores[0] - 8000;
    if (buffer < expectedLoss) return 0.35;
    if (buffer < 2 * expectedLoss) return 0.7;
    return 1;
  }

  /**
   * Threat volume: a declared riichi counts full, a loud open hand counts half.
   *
   * HOOK. Both figures are guesses standing in for "how likely is that seat to
   * be tenpai, and for how much" — exactly what an estimator replaces.
   */
  protected pressureOf(obs: Observation): number {
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

    // THE FILTER. Ask the referee about every candidate before scoring any of
    // them, and if some subset is clean, that subset IS the choice set — no
    // score, from this class or a subclass hook, can nominate a tile outside it.
    // An empty result means every discard is charged, and the priced fallthrough
    // below (`dojoCost`) decides which charge to take.
    const clean = this.compliantDiscards(ctx, byTile);
    const candidates = [...byTile.keys()].filter((t) => clean === null || clean.has(t));
    const shantenAfter = new Map<Tile, number>();
    let best = Infinity;
    for (const tile of candidates) {
      const s = this.shantenWithout(ctx, tile);
      shantenAfter.set(tile, s);
      if (s < best) best = s;
    }

    // M9. One assembly per decision when a consumer is set, and not a single
    // line of work when one is not.
    let run: EvidenceRun | null = null;
    if (this.consumer) {
      const hooks = this.evidenceHooks();
      run = { hooks, context: assembleContext(hooks, ctx) };
    }

    let bestTile = candidates[0];
    let bestScore = -Infinity;
    for (const tile of candidates) {
      const sh = shantenAfter.get(tile)!;
      // Ukeire is the expensive part (34 shanten probes); only the tiles that
      // actually hold the best shanten can win on it.
      const score = this.scoreDiscard(ctx, tile, sh, sh === best, run);
      if (score > bestScore || (score === bestScore && tile < bestTile)) {
        bestScore = score;
        bestTile = tile;
      }
    }

    const group = byTile.get(bestTile)!;
    const plain = group.find((d) => !d.riichi);
    const riichi = group.find((d) => d.riichi);
    if (riichi && !this.riichiBanned(ctx, bestTile) && this.riichiClean(ctx, riichi)) {
      // A split wait must not be left damaten. Riichi is the cure and the
      // dojo's own prescription (役なしなら即リーチ), so it overrides the
      // ordinary "is this worth declaring" judgement — but never a 禁じ手.
      const mustCure = this.dojo && (ctx.obs.discardInfo.get(bestTile)?.katagari ?? false);
      if (mustCure || this.wantRiichi(ctx, bestTile)) return riichi;
    }
    return plain ?? group[0];
  }

  /**
   * The discards the ledger would let pass, or null when the question cannot be
   * asked (no preview wired, or the dojo leash is off) — and null again when the
   * answer is "none of them", which is the fallthrough the prices exist for.
   *
   * Each tile is judged by the plain discard: the riichi variant carries extra
   * rules (地獄単騎, 即引っかけ …) and is settled separately, once the tile is
   * chosen, by `riichiClean`.
   *
   * 片和了り and 後付け are vetoed here too, even though the preview cannot see
   * them. They are the only two ledger rules whose CHARGE lands at win time —
   * unpreviewable by construction, since declining the win would be 見逃し — but
   * whose only PREVENTION is a discard, right here. Leaving them to `dojoCost`
   * makes them a price, and a price loses: the C7 planner's `planKeep` malus
   * (5000) simply outbids both. So they are vetoes like everything else, and the
   * prices stay for the fallthrough, where ranking damage is all that is left.
   */
  private compliantDiscards(
    ctx: Ctx,
    byTile: Map<Tile, Extract<Action, { t: "discard" }>[]>,
  ): Set<Tile> | null {
    const pv = this.referee(ctx);
    if (!pv) return null;
    const ok = new Set<Tile>();
    for (const [tile, group] of byTile) {
      const a = group.find((d) => !d.riichi) ?? group[0];
      if (pv.discard(a, ctx.obs.drawn).length !== 0) continue;
      const info = ctx.obs.discardInfo.get(tile);
      // 片和了り, but only when riichi is not on offer: declaring is itself a
      // yaku, so it makes every wait scoring and the shape stops being split.
      if (!ctx.canRiichi && info?.katagari) continue;
      // 後付け: only an open hand is stuck with a yakuless tenpai; a closed one
      // can still cure the same shape by declaring.
      if (!ctx.closed && info?.yakuless) continue;
      ok.add(tile);
    }
    return ok.size > 0 ? ok : null;
  }

  /** Would declaring on this discard stay off the ledger? */
  private riichiClean(ctx: Ctx, a: Extract<Action, { t: "discard" }>): boolean {
    const pv = this.referee(ctx);
    return !pv || pv.discard(a, ctx.obs.drawn).length === 0;
  }

  /** The 13-tile shape left behind by this discard. Protected: the hooks want it. */
  protected handWithout(ctx: Ctx, tile: Tile): Tile[] {
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

  /**
   * One discard candidate's score.
   *
   * TWO PATHS, ONE SURROUND. With a consumer set (M9) the CORE — the efficiency
   * aggregate, the price of danger, and the two hook bonuses — is computed from
   * the named evidence vector by `consumer.ts` instead of by the arithmetic
   * below; `dojoCost` is subtracted identically either way, because a 禁じ手 is
   * priced by the ledger and not by anything a fit is allowed to move. At init
   * the two paths agree bit for bit (`initFromWeights`), which is what makes the
   * swap measurable rather than merely plausible.
   */
  private scoreDiscard(
    ctx: Ctx,
    tile: Tile,
    sh: number,
    wideOpen: boolean,
    run: EvidenceRun | null,
  ): number {
    if (this.consumer && run) {
      const ev = {
        context: run.context,
        candidate: assembleCandidate(run.hooks, ctx, tile, sh, wideOpen),
      };
      return consumeEvidence(ev, this.consumer) - this.dojoCost(ctx, tile, sh);
    }

    const { obs } = ctx;
    const rest = this.handWithout(ctx, tile);
    const counts = countsFromTiles(rest);

    let eff = -sh * this.w.shanten;

    if (wideOpen) {
      const types = ukeireTypes(counts, ctx.open, ctx.closed, sh);
      let live = 0;
      for (const ty of types) live += ctx.unseen[ty];
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

    // The dojo cost is deliberately outside `ctx.eff`: folding must not make a
    // 禁じ手 cheap. The ledger charges the same either way. The two bonus hooks
    // sit outside it too, and outside `ctx.eff`/`ctx.def`: they are already in
    // score units and already know whether the policy is folding.
    return ctx.eff * eff - ctx.def * this.riskOf(ctx, tile) +
      this.drawBonus(ctx, tile) - this.keepBonus(ctx, tile) -
      this.dojoCost(ctx, tile, sh);
  }

  /**
   * What letting this tile go costs defensively, in score units, BEFORE the
   * fold multiplier.
   *
   * HOOK. The base policy reads mjrender's four danger levels off the
   * observation; a policy holding a per-tile deal-in probability and a payment
   * to go with it computes the product instead. Whatever the source, "安全"
   * must stay free — that level means provably safe (genbutsu), and no estimate
   * outranks a proof.
   */
  protected riskOf(ctx: Ctx, tile: Tile): number {
    return this.ruleRisk(this.dangerLevelOf(ctx, tile));
  }

  /**
   * The assessor's own reading of this tile, or `undefined` when it was not
   * looking at all (no declared threat on the table).
   *
   * Split from `riskOf` because an override needs BOTH halves and the two are
   * one map lookup: `AugmentedHeuristic` prices its estimate against the rule
   * ladder while also honouring an EXPLICIT 安全 as a proof, and an absent entry
   * is not that proof.
   */
  protected dangerLevelOf(ctx: Ctx, tile: Tile): DangerLevel | undefined {
    return ctx.obs.danger.get(tileType(tile))?.level;
  }

  /** The rule ladder's price for a level; an absent reading costs nothing. */
  protected ruleRisk(level: DangerLevel | undefined): number {
    return this.w.danger[level ?? "安全"];
  }

  /**
   * A bonus ADDED to this discard's score. Zero here: the base policy has no
   * one-turn lookahead. Overridden by a policy that knows what is coming off
   * the wall — the discard whose kept shape accepts the incoming tile, the one
   * that keeps a copy of a dora about to be flipped.
   */
  protected drawBonus(_ctx: Ctx, _tile: Tile): number {
    return 0;
  }

  /**
   * A malus SUBTRACTED from this discard's score: a reason to hold the tile
   * back this turn rather than spend it. Zero here. Overridden by a policy that
   * can see a tile is about to become genbutsu anyway, and would rather spend a
   * less useful safe tile first.
   */
  protected keepBonus(_ctx: Ctx, _tile: Tile): number {
    return 0;
  }

  /**
   * What the ledger would charge for this discard, in score units.
   *
   * With the compliance filter live this is mostly moot for the rules it shares
   * with the preview — a charged discard never reaches the score loop unless
   * EVERY discard is charged, and then these prices are exactly what ranks the
   * damage. The two win-time rules are different: 片和了り and 後付け fire when a
   * hand is CASHED, which a policy cannot decline without going furiten, so the
   * only prevention is here, at the discard that builds the wait.
   */
  private dojoCost(ctx: Ctx, tile: Tile, sh: number): number {
    if (!this.dojo) return 0;
    const { obs } = ctx;
    const ty = tileType(tile);
    let cost = 0;

    // 第一打字牌切り — our river is still empty when we pick this discard.
    if (obs.rivers[0].length === 0 && isHonor(ty)) cost += this.w.firstHonor;

    // 不聴時ドラ切り. Indicator dora only: the aka 5p may be cut before tenpai.
    // Charged only from 3向聴 out — 2向聴以内 is allowed.
    // 例外: an honor dora already twice in the rivers is spent.
    // `4 − unseen` IS the visible count: rivers, melds, indicators and own hand
    // are exactly what `publicUnseen` subtracts, and there are only four copies.
    if (sh > 2 && ctx.doraTypes.has(ty)) {
      if (!(isHonor(ty) && ctx.unseen[ty] <= 2)) cost += this.w.notenDora;
    }

    // 片和了り, but only when riichi is not on offer: riichi is itself a yaku,
    // so declaring it makes every wait scoring and the shape stops being split.
    if (!ctx.canRiichi && obs.discardInfo.get(tile)?.katagari) cost += this.w.katagari;

    // 後付け: an open hand tenpai on nothing that scores. A closed hand can cure
    // the same shape by declaring, so only the open case is a violation — and
    // the only prevention is not making that discard.
    if (!ctx.closed && obs.discardInfo.get(tile)?.yakuless) cost += this.w.yakulessTenpai;

    // ドラ切りをポンされた後の手出し. `legal.ts` will not stop us — the dojo
    // takes the payment instead — so the price has to be paid here.
    if (obs.tsumogiriLock && tile !== obs.drawn) cost += this.w.tsumogiriLock;

    return cost;
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

  /**
   * Which pon/chi to take, if any: the one that buys a shanten step, subject to
   * the open hand still having a route to a yaku.
   *
   * HOOK (protected only for that reason — the base behaviour is unchanged): a
   * policy with a locked-on target overrides this to accept only the calls its
   * plan asked for, and to decline the rest even when they are faster.
   */
  protected chooseCall(ctx: Ctx, legal: Action[]): Action | null {
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

    if (this.dojo && !this.hasYakuProspect(rest, melds, ctx.valueHonors, this.kuitan)) {
      return null;
    }

    return shanten(countsFromTiles(rest), melds.length, false);
  }

  /**
   * Is there a yaku this open shape can plausibly still land on?
   *
   * GUIDANCE ONLY, and deliberately looser than the ledger's 後付け rule. That
   * rule judges the finished waiting hand with the real scorer (`yakuless` in
   * `DiscardInfo`, priced in `dojoCost`); this one only has to stop the CPU
   * from opening a hand with no route to a yaku at all. A concealed 役牌 pair —
   * the classic バック — passes here on purpose: under the new reading the crime
   * is the yakuless WAIT, not the hopeful call, so refusing every バック would
   * cost the policy hands the dojo has no objection to.
   */
  private hasYakuProspect(
    rest: Tile[],
    melds: Meld[],
    valueHonors: Set<number>,
    kuitan: boolean,
  ): boolean {
    const restTypes = rest.map(tileType);
    const meldTypes = melds.map((m) => m.tiles.map(tileType));

    // 役牌: already melded, or held concealed as a pair waiting on the third.
    for (const m of melds) {
      if (m.kind !== "chi" && valueHonors.has(tileType(m.tiles[0]))) return true;
    }
    const counts = new Map<number, number>();
    for (const ty of restTypes) counts.set(ty, (counts.get(ty) ?? 0) + 1);
    for (const [ty, n] of counts) if (n >= 2 && valueHonors.has(ty)) return true;

    // 断幺九: nothing melded touches a yaochu, and the concealed part holds at
    // most one — which the discard that follows this call can throw away.
    if (kuitan) {
      const meldClean = meldTypes.every((ts) => ts.every((ty) => !isYaochu(ty)));
      if (meldClean && restTypes.filter(isYaochu).length <= 1) return true;
    }

    // 混一色/清一色: the melds sit in one suit and at most two concealed tiles
    // are stranded in another (honors are always welcome in a 混一色).
    const meldSuits = new Set(
      meldTypes.flat().filter((ty) => !isHonor(ty)).map((ty) => suitOfType(ty)),
    );
    if (meldSuits.size === 1) {
      const [suit] = [...meldSuits];
      const strays = restTypes.filter((ty) => !isHonor(ty) && suitOfType(ty) !== suit);
      if (strays.length <= 2) return true;
    }

    // 対々和: no chi anywhere means the hand is still a pure triplet build.
    if (melds.every((m) => m.kind !== "chi")) return true;

    return false;
  }

  // -------------------------------------------------------------------- kan

  /** HOOK, same reasoning as `chooseCall`; no subclass overrides it yet. */
  protected chooseKan(ctx: Ctx, legal: Action[]): Action | null {
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
