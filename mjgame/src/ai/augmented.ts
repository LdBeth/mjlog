// The heuristic CPU with a channel of hidden information plugged in.
//
// M8-ZERO is an ABLATION, not a net: the estimates the reader is meant to
// produce are replaced with the TRUTH the engine already holds (every hand,
// the live wall, the dead wall), one channel at a time, and each channel is
// measured against the plain heuristic with `paired`. A channel that cannot
// convert perfect truth into rank is dropped before a single label is recorded.
//
// The split that makes that work: `Reads` is the DATA the decision terms
// consume, and a `ReadsProvider` is where it comes from. `oracleReads` cheats;
// M8b's `NetReads` will run a supervised reader and fill the same shape. The
// consumption terms below — the `AugmentedHeuristic` overrides — are written
// and tuned once, here, against the oracle, and never change again.
//
// Nothing here touches `dojoCost`, and the plan carries no ledger term at all.
// It needs none: compliance is enforced at ACTION SELECTION, not by pricing.
// The base class narrows the discard candidate set to what the referee would let
// pass before any score — `planKeep` included — is computed, and vetoes whatever
// `chooseCall` returns, this override included. A plan can therefore be as
// single-minded as it likes about which tiles it wants kept and which calls it
// wants taken; it can never buy its way into a 禁じ手, however profitable the
// future it is holding out for.

import type { Tile } from "mjrender/model.ts";
import { countsFromTiles, shanten, ukeireTypes } from "mjrender/shanten.ts";
import { doraFromIndicatorType, tileType } from "mjrender/tiles.ts";
import type { Ctx, HeuristicOptions } from "./heuristic.ts";
import { HeuristicPolicy } from "./heuristic.ts";
import type { PlannerOptions, TargetPlan } from "./planner.ts";
import { availabilityFrom, enumerateTargets, relock, RELOCK_MARGIN } from "./planner.ts";
import type { Observation } from "../observe.ts";
import { sfc32 } from "../rng.ts";
import type { Scorer } from "../round.ts";
import { ronValue } from "../score.ts";
import type { Table } from "../table.ts";
import { anyFuriten } from "../table.ts";
import type { Action, Seat } from "../types.ts";

/**
 * What the augmented decision terms consume. Later a learned reader produces
 * the same shape; in M8-zero `oracleReads` fills it with truth.
 *
 * Every field is OPTIONAL and every opponent array is length 3 in RELATIVE
 * order (index 0 = shimocha = absolute (self+1)%4, and so on — the convention
 * `observe.ts` and `rl/features.ts` already use). Absence is meaningful: a term
 * whose data is missing falls back to the base heuristic's own reasoning, which
 * is exactly what makes `--oracle=C4` measure C4 and nothing else.
 *
 * DEVIATION from the M8b sketch in the plan, deliberate: `tenpaiP`, `dealinP`
 * and `expLoss` are optional here. The ablation needs "this channel is off" to
 * be distinguishable from "this channel says zero" — a `dealinP` of all zeros
 * is a claim that nothing can deal in, not an absence of information.
 */
export interface Reads {
  /** P(opponent is tenpai) — {0,1} from the oracle. */
  tenpaiP?: number[];
  /** Per opponent, per tile type: P(that opponent rons this type NOW). */
  dealinP?: Float32Array[];
  /** Per opponent: expected ron payment in points, over its ronnable types. */
  expLoss?: number[];
  /** Per opponent, per tile type: the exact payment. Oracle only. */
  dealinValue?: Float32Array[];
  /** Tile TYPE of `wall.peekLive(0)` — the next tile off the live wall. */
  nextDraw?: number | null;
  /**
   * Tile TYPE this seat would draw next, no calls intervening. NOT in the plan's
   * field list; see the note on `oracleReads` for why `nextDraw` alone is the
   * wrong tile to hand a discard scorer.
   */
  ownNextDraw?: number | null;
  /** Dora TYPE (not indicator) the next kan-dora reveal would flip. */
  nextDora?: number | null;
  /** Per riichi opponent: the tile TYPE they draw next. Null for the rest. */
  riichiNextDraw?: (number | null)[];

  /**
   * C7: the planner is engaged for this decision.
   *
   * NOT in the plan's field list, and unavoidable: C7P deliberately fills
   * neither field below, so "the planner is on" cannot be inferred from the data
   * — which is exactly what makes C7P the architecture-only control.
   */
  planner?: boolean;
  /** C7O: copies of each tile TYPE left in the LIVE wall. Oracle only. */
  wallComposition?: Float32Array;
  /** C7O: each opponent's concealed hand as type counts, relative order. */
  oppConcealed?: Float32Array[];
}

export type ReadsProvider = (obs: Observation) => Reads | null;

/**
 * The information channels of the ablation matrix (plan M8-zero):
 * C1 deal-in truth, C2 tenpai truth, C3 value truth, C4 own next draw,
 * C5 next kan-dora, C6 declared-riichi opponent's next draw.
 *
 * C7O/C7P are the Z6 planner pair and are NOT information channels in the same
 * sense: both switch the same machinery on, and they differ only in what feeds
 * its availability model — true wall composition + true hands (C7O) versus the
 * public unseen counts anyone can count (C7P). Running both separates what the
 * planner is worth as an architecture from what the hidden information is worth.
 */
export type OracleChannel = "C1" | "C2" | "C3" | "C4" | "C5" | "C6" | "C7O" | "C7P";

export const ORACLE_CHANNELS: readonly OracleChannel[] = [
  "C1",
  "C2",
  "C3",
  "C4",
  "C5",
  "C6",
  "C7O",
  "C7P",
];

export function parseChannels(spec: string): Set<OracleChannel> | null {
  const out = new Set<OracleChannel>();
  if (spec.trim() === "" || spec === "none") return out;
  for (const raw of spec.split(",")) {
    const tok = raw.trim().toUpperCase();
    if (!(ORACLE_CHANNELS as readonly string[]).includes(tok)) return null;
    out.add(tok as OracleChannel);
  }
  return out;
}

/** Every magic number the augmented terms add, in one object like the base's. */
export interface AugmentedWeights {
  /**
   * Points → score units for the deal-in term. Units-matched against the base
   * table: P .13 × 6400 × .25 ≈ 200, which is what 危険度高 costs.
   */
  lambda: number;
  /**
   * Floor under the rule-based risk, as a fraction of it. An uncalibrated
   * estimate must never be able to declare a no-suji tile against a riichi
   * free; only the "安全" level (a proof, not an estimate) can do that.
   */
  floor: number;
  /** C4: the kept shape accepts the tile that is about to be drawn. */
  drawRealize: number;
  /** C6: per copy held of the dora type the next indicator would flip. */
  futureDora: number;
  /** C6: holding back a tile that is about to become genbutsu anyway. */
  futureGenbutsu: number;
  /**
   * C7: cutting a tile the locked plan keeps. Set far above a shanten step so
   * the plan, not the local efficiency count, decides which tile goes — but a
   * SCORE, never a filter: while folding it is switched off entirely, and even
   * pushing, the danger term and the dojo ledger can still outbid it.
   */
  planKeep: number;
  /** C7: cutting a tile the plan has no use for. Small — a tie-break. */
  planAdvance: number;
  /** C7: how much better a challenger must be to take the lock. */
  planRelock: number;
}

export const DEFAULT_AUGMENTED_WEIGHTS: AugmentedWeights = {
  lambda: 0.25,
  floor: 0.5,
  drawRealize: 150,
  futureDora: 30,
  futureGenbutsu: 120,
  planKeep: 5000,
  planAdvance: 200,
  planRelock: RELOCK_MARGIN,
};

/** Fallback deal-in cost, in points, wherever the value channel is off. */
const ASSUMED_LOSS = 6000;

// ---------------------------------------------------------------------------
// the oracle provider
// ---------------------------------------------------------------------------

/**
 * Truth, straight off the live Table, filtered to the enabled channels.
 *
 * `tableGet` is the existing `MatchOptions.tableRef` tap (match.ts:24-31): the
 * driver keeps it pointed at the round in play, so this is only usable
 * headless. An empty channel set produces no Reads at all, which makes the
 * augmented policy byte-identical to the plain heuristic — the control arm.
 *
 * TURN ARITHMETIC. At a turn decision our own draw has already left the wall,
 * so `peekLive(0)` is the tile the NEXT seat draws, `peekLive(r-1)` the tile the
 * opponent at relative seat r draws, and `peekLive(3)` our own next draw. All of
 * that assumes no call intervenes (a pon/chi skips seats and a kan inserts a
 * rinshan draw from the dead wall); that is the documented approximation the
 * plan's `rd`/`nx` labels carry too. At a claim decision the arithmetic has no
 * fixed answer, so the per-seat fields are left null and only the wall-order
 * facts (`nextDraw`, `nextDora`) are filled.
 */
export function oracleReads(
  tableGet: () => Table | null,
  scorer: Scorer,
  channels: Set<OracleChannel>,
): ReadsProvider {
  const wantDealin = channels.has("C1");
  const wantTenpai = channels.has("C2");
  const wantValue = channels.has("C3");

  return (obs: Observation): Reads | null => {
    if (channels.size === 0) return null;
    const t = tableGet();
    if (!t) return null;
    const self = obs.seat;
    const reads: Reads = {};

    if (wantDealin || wantTenpai || wantValue) {
      const tenpaiP = wantTenpai ? [0, 0, 0] : undefined;
      const dealinP = wantDealin ? [f34(), f34(), f34()] : undefined;
      const dealinValue = wantDealin ? [f34(), f34(), f34()] : undefined;
      const expLoss = wantValue ? [0, 0, 0] : undefined;

      for (let i = 0; i < 3; i++) {
        const o = ((self + i + 1) % 4) as Seat;
        const counts = countsFromTiles(t.hands[o]);
        const open = t.melds[o].length;
        const closed = t.isMenzen(o);
        const sh = shanten(counts, open, closed);
        const tenpai = sh <= 0;
        if (tenpaiP) tenpaiP[i] = tenpai ? 1 : 0;
        if (!dealinP && !expLoss) continue;

        // The exact "a ron fires right now" predicate — the same chain
        // `claimActions` walks (legal.ts:216-237), minus the shape test, which
        // is what tenpai + ukeire already answers.
        let sum = 0;
        let n = 0;
        const mute = !tenpai || t.sanctioned[o] || anyFuriten(t.furiten[o]);
        if (!mute) {
          for (const ty of ukeireTypes(counts, open, closed, sh)) {
            if (t.ronBlocked[o].has(ty)) continue;
            // Aka-blind: the lowest copy stands in for the type, the same
            // approximation `hand.ts:59` makes. Worth at most ±1 han.
            const tile = ty * 4;
            if (!scorer.hasYaku(t, o, tile, false)) continue;
            const v = ronValue(t, o, self, tile) ?? ASSUMED_LOSS;
            if (dealinP) dealinP[i][ty] = 1;
            if (dealinValue) dealinValue[i][ty] = v;
            sum += v;
            n++;
          }
        }
        if (expLoss) expLoss[i] = n === 0 ? 0 : sum / n;
      }

      if (tenpaiP) reads.tenpaiP = tenpaiP;
      if (dealinP) reads.dealinP = dealinP;
      if (dealinValue) reads.dealinValue = dealinValue;
      if (expLoss) reads.expLoss = expLoss;
    }

    const ourTurn = obs.drawn !== null;

    if (channels.has("C4")) {
      reads.nextDraw = typeOf(t.wall.peekLive(0));
      reads.ownNextDraw = ourTurn ? typeOf(t.wall.peekLive(3)) : null;
    }

    if (channels.has("C5")) {
      const ind = t.wall.peekNextIndicator();
      reads.nextDora = ind === null ? null : doraFromIndicatorType(tileType(ind));
    }

    if (channels.has("C6")) {
      reads.riichiNextDraw = [0, 1, 2].map((i) => {
        const o = ((self + i + 1) % 4) as Seat;
        if (!ourTurn || !t.riichi[o]) return null;
        return typeOf(t.wall.peekLive(i));
      });
    }

    // C7 — the planner. Both variants switch it on; only the oracle one feeds it
    // truth. C7P fills nothing at all, so `availabilityFrom` falls through to the
    // public unseen counts and the arm measures the architecture alone.
    if (channels.has("C7O") || channels.has("C7P")) {
      reads.planner = true;
      if (channels.has("C7O")) {
        const wall = f34();
        for (let k = 0;; k++) {
          const tile = t.wall.peekLive(k);
          if (tile === null) break;
          wall[tileType(tile)]++;
        }
        reads.wallComposition = wall;
        reads.oppConcealed = [0, 1, 2].map((i) => {
          const o = ((self + i + 1) % 4) as Seat;
          const c = f34();
          for (const tile of t.hands[o]) c[tileType(tile)]++;
          return c;
        });
      }
    }

    return reads;
  };
}

const f34 = () => new Float32Array(34);
const typeOf = (tile: Tile | null): number | null => tile === null ? null : tileType(tile);

// ---------------------------------------------------------------------------
// oracle fading
// ---------------------------------------------------------------------------

/**
 * The information GROUPS a decision can lose, in the order the PRNG visits them.
 *
 * A group is what one estimator would produce, so it fails as a unit: a reader
 * that cannot say who is tenpai cannot say what their wait pays either. Keeping
 * `dealinValue` alongside `dealinP` (rather than with `expLoss`) follows
 * `riskOf`, which reads the two together and falls back to `expLoss` when the
 * pair is gone.
 *
 * `planner` is deliberately absent: see `noisyReads`.
 */
const NOISE_GROUPS: readonly (readonly (keyof Reads)[])[] = [
  ["dealinP", "dealinValue"], // dealin
  ["tenpaiP"], // tenpai
  ["expLoss"], // value
  ["nextDraw", "ownNextDraw"], // draw
  ["nextDora"], // dora
  ["riichiNextDraw"], // riichi
  ["wallComposition", "oppConcealed"], // avail
];

/**
 * A Reads provider that FORGETS, per decision, with probability ε per group.
 *
 * THE MODEL. At noise level ε each decision independently drops each
 * information group it was given, so the policy sees perfect truth on one turn
 * and its own rule-of-thumb reasoning on the next. This is Suphx-style oracle
 * fading applied to the M8-zero ablation: sweeping ε measures at what fidelity a
 * real reader would have to operate before the guided heuristic loses the edge
 * the oracle arm demonstrated. Every override in `AugmentedHeuristic` already
 * degrades to `super` when its field is absent, so a dropped group needs no
 * cooperation from the consumer — absence is the existing, tested path.
 *
 * WHY DROPOUT AND NOT BLUR. Perturbing the magnitudes (ε-scaled noise on
 * `dealinP`, on `expLoss`) leaves the ORDER of the tiles almost intact, and
 * order is all a discard scorer consumes: a blurred oracle keeps picking the
 * same tile and the curve under-reports the degradation badly. Dropping the
 * whole group destroys the discrimination outright, which is the honest lower
 * bound — a reader that is wrong is worse than one that is silent, so the true
 * curve for a real net lies below this one.
 *
 * ENDPOINTS. ε = 0 returns `inner` UNCHANGED — the same function object, so the
 * arm is bit-identical to the un-noised run and not merely statistically alike.
 * ε = 1 drops every group on every decision; what survives is `planner`, which
 * is never dropped, so C7O degrades exactly to C7P's public-availability
 * behavior (the estimator goes uninformative, the architecture stays on) and an
 * otherwise all-channel oracle seat degrades to the plain heuristic.
 *
 * STREAM. One uniform is drawn per PRESENT group per call, in `NOISE_GROUPS`
 * order, and none at all when `inner` returns null. So the length of the
 * consumed stream depends only on which groups the channel set fills — a fixed
 * channel set gives a fixed, reproducible schedule of losses for a given seed,
 * which is what `paired` needs to compare two arms on one wall.
 */
export function noisyReads(inner: ReadsProvider, epsilon: number, seed = 0xC0FFEE): ReadsProvider {
  if (!Number.isFinite(epsilon) || epsilon < 0 || epsilon > 1) {
    throw new RangeError(`noisyReads: ε は 0..1 の実数: ${epsilon}`);
  }
  if (epsilon === 0) return inner;
  // ε is part of the seed so that two levels of the sweep do not share a
  // dropout schedule — otherwise ε=.3 and ε=.6 would lose the SAME decisions
  // (the second a superset of the first) and the two points would be correlated
  // in a way the paired variance estimate does not know about.
  const rng = sfc32((seed ^ Math.round(epsilon * 1e6)) >>> 0);

  return (obs: Observation): Reads | null => {
    const r = inner(obs);
    if (r === null) return null;
    const out: Reads = { ...r };
    for (const group of NOISE_GROUPS) {
      if (!group.some((f) => out[f] !== undefined)) continue;
      if (rng.float() >= epsilon) continue;
      for (const f of group) delete out[f];
    }
    return out;
  };
}

// ---------------------------------------------------------------------------
// the consumer
// ---------------------------------------------------------------------------

const clamp = (x: number, lo: number, hi: number) => x < lo ? lo : x > hi ? hi : x;

export interface AugmentedOptions extends HeuristicOptions {
  augment?: Partial<AugmentedWeights>;
  /** Search bounds and ruleset for the C7 planner. */
  plan?: PlannerOptions;
}

/**
 * The heuristic, with each rule-of-thumb term given the option of a better
 * answer. Every override degrades to `super` when its Reads field is absent, so
 * the same class is the baseline, each single-channel ablation arm and the
 * eventual net-driven policy.
 */
export class AugmentedHeuristic extends HeuristicPolicy {
  protected reads: Reads | null = null;
  protected readonly aw: AugmentedWeights;
  private readonly provider: ReadsProvider;
  private readonly planOpts: PlannerOptions;
  /**
   * The C7 lock. Unlike `reads` this SURVIVES the decision — that is the whole
   * mechanism: a target chosen on 6巡目 is what steers 7巡目's discard. It is
   * scoped to one 局 (kyoku, honba); a policy instance plays a whole match, so
   * carrying a plan across the deal would steer the next hand with the last
   * hand's tiles.
   */
  private target: TargetPlan | null = null;
  private targetRound = "";

  constructor(name: string, seed: number, provider: ReadsProvider, opts: AugmentedOptions = {}) {
    super(name, seed, opts);
    this.provider = provider;
    this.aw = { ...DEFAULT_AUGMENTED_WEIGHTS, ...opts.augment };
    this.planOpts = opts.plan ?? {};
  }

  /** The plan currently locked on, if any. Read-only; for the TUI and tests. */
  get plan(): TargetPlan | null {
    return this.target;
  }

  /**
   * One read per decision, held for the life of `super.decide` and dropped
   * after: the hooks below run deep inside the base policy and have no other
   * way to reach it, and a stale read outliving the decision would be a lie.
   */
  override decide(obs: Observation): Action {
    this.reads = this.provider(obs);
    try {
      this.updatePlan(obs);
      return super.decide(obs);
    } finally {
      this.reads = null;
    }
  }

  // ------------------------------------------------------------ C7 planner

  /**
   * Re-price the futures and move the lock if a challenger has earned it.
   *
   * Deliberately does nothing while folding: the plan is what to build, and a
   * hand being abandoned is not being built. The lock is KEPT rather than
   * cleared, so a fold that turns out to be one loud turn does not cost the hand
   * its shape.
   */
  private updatePlan(obs: Observation): void {
    if (!this.reads?.planner) {
      this.target = null;
      return;
    }
    const round = `${obs.kyoku}:${obs.honba}`;
    if (round !== this.targetRound) {
      this.targetRound = round;
      this.target = null;
    }
    // A forced move has nothing to steer, and a formless hand has no future
    // worth naming — the base efficiency terms own that regime.
    if (obs.legal.length <= 1) return;
    if (obs.shanten < 0 || obs.shanten > 3) {
      this.target = null;
      return;
    }
    if (this.shouldFold(obs)) return;

    const avail = availabilityFrom(obs, this.reads);
    const cands = enumerateTargets(obs, avail, this.planOpts);
    if (cands.length === 0) {
      this.target = null;
      return;
    }
    this.target = relock(this.target, cands, this.aw.planRelock);
  }

  /**
   * Plan discipline: take the calls the plan asked for, decline the rest — even
   * the ones that buy a shanten step. A 副露 that is not part of a future is how
   * an open hand ends up fast, cheap and yakuless, which is the shape the dojo
   * charges for.
   *
   * What comes back is a PROPOSAL: `HeuristicPolicy.decide` puts it past the
   * referee preview before playing it, so a plan that asks for a ledgerable call
   * simply does not get it.
   */
  protected override chooseCall(ctx: Ctx, legal: Action[]): Action | null {
    const target = this.target;
    if (!target || ctx.folding) return super.chooseCall(ctx, legal);
    if (target.callable.length === 0) return null; // a closed plan calls nothing
    const want = new Set(target.callable);
    for (const a of legal) {
      if (a.t !== "pon" && a.t !== "chi") continue;
      if (!want.has(tileType(a.called))) continue;
      // The tiles it would spend must be the plan's own, or the call takes the
      // hand somewhere the plan never priced.
      if (!a.tiles.every((t) => target.keep.has(t))) continue;
      return a;
    }
    return null;
  }

  /** P(deal in) × what it pays, floored by the rule-based reading. */
  protected override riskOf(ctx: Ctx, tile: Tile): number {
    const base = super.riskOf(ctx, tile);
    const dealinP = this.reads?.dealinP;
    if (!dealinP) return base;

    // RULE FLOOR, top half: "安全" is genbutsu — a proof, not an assessment.
    // No estimate, however confident, may price a provably safe tile.
    const level = ctx.obs.danger.get(tileType(tile))?.level ?? "安全";
    if (level === "安全") return 0;

    const ty = tileType(tile);
    let risk = 0;
    for (let i = 0; i < 3; i++) {
      const p = dealinP[i][ty];
      if (p <= 0) continue;
      const v = this.reads?.dealinValue?.[i]?.[ty] ?? this.reads?.expLoss?.[i] ?? ASSUMED_LOSS;
      risk += p * v;
    }
    // RULE FLOOR, bottom half.
    return Math.max(this.aw.lambda * risk, this.aw.floor * base);
  }

  /** Threat volume priced by who is actually tenpai, and for how much. */
  protected override pressureOf(obs: Observation): number {
    const tenpaiP = this.reads?.tenpaiP;
    if (!tenpaiP) return super.pressureOf(obs);
    const expLoss = this.reads?.expLoss;
    let p = 0;
    for (let i = 0; i < 3; i++) {
      // A declared riichi is a public fact; it outranks any estimate.
      const t = clamp(Math.max(tenpaiP[i], obs.riichi[i + 1] ? 1 : 0), 0, 1);
      if (t <= 0) continue;
      p += t * Math.min(2, (expLoss?.[i] ?? ASSUMED_LOSS) / ASSUMED_LOSS);
    }
    return p;
  }

  /** The 8000点 buffer measured against what a deal-in would ACTUALLY cost. */
  protected override bufferScale(obs: Observation, expectedLoss = ASSUMED_LOSS): number {
    const tenpaiP = this.reads?.tenpaiP;
    const expLoss = this.reads?.expLoss;
    if (tenpaiP && expLoss) {
      let worst = 0;
      for (let i = 0; i < 3; i++) worst = Math.max(worst, tenpaiP[i] * expLoss[i]);
      expectedLoss = clamp(worst, 2000, 12000);
    }
    return super.bufferScale(obs, expectedLoss);
  }

  /**
   * One-turn lookahead: C4 rewards the discard whose kept shape accepts the
   * tile actually coming, C5 rewards keeping copies of the dora about to flip.
   */
  protected override drawBonus(ctx: Ctx, tile: Tile): number {
    const r = this.reads;
    if (!r) return 0;
    let bonus = 0;

    // `ownNextDraw` is the tile WE get; `nextDraw` is merely the next tile off
    // the wall (the next seat's). Prefer the former, fall back to the latter so
    // a net that only produces the plan's `nx` head still drives this term.
    const incoming = r.ownNextDraw ?? r.nextDraw;
    const dora = r.nextDora;

    if (incoming !== null && incoming !== undefined) {
      const rest = this.handWithout(ctx, tile);
      const counts = countsFromTiles(rest);
      if (counts[incoming] < 4) {
        // "Is `incoming` in this shape's ukeire" without enumerating all 34:
        // it is exactly "does drawing it drop the shanten".
        const before = ctx.obs.discardInfo.get(tile)?.shanten ??
          shanten(counts, ctx.open, ctx.closed);
        counts[incoming]++;
        const after = shanten(counts, ctx.open, ctx.closed);
        counts[incoming]--;
        if (after < before) bonus += this.aw.drawRealize;
      }
    }

    if (dora !== null && dora !== undefined) {
      let held = 0;
      for (const t of ctx.obs.hand) if (t !== tile && tileType(t) === dora) held++;
      bonus += held * this.aw.futureDora;
    }

    // C7, the positive half: this discard is exactly what the plan wanted spent
    // — neither a tile the shape uses nor a copy of a type it is still waiting
    // on. Small: the malus below is what decides, this only breaks ties among
    // the tiles the plan is indifferent to.
    const target = this.target;
    if (
      target && !ctx.folding && !target.keep.has(tile) &&
      !target.required.includes(tileType(tile))
    ) {
      bonus += this.aw.planAdvance;
    }
    return bonus;
  }

  /**
   * C6: a tile a riichi seat is about to draw and tsumogiri becomes genbutsu
   * next turn — so while folding, spend the other safe tiles first and keep
   * this one for the turn after. Only while folding: on a pushing turn the
   * hand's own shape decides, not the safety inventory.
   */
  protected override keepBonus(ctx: Ctx, tile: Tile): number {
    // C7, the half that does the work: cutting a tile the locked plan is built
    // from. Gated on `!folding` on purpose — while the hand is being abandoned
    // the safest tile must stay the cheapest one, and a 5000 malus on the plan's
    // tiles would make the policy defend its shape instead of its points.
    const target = this.target;
    if (target && !ctx.folding && target.keep.has(tile)) return this.aw.planKeep;

    const rd = this.reads?.riichiNextDraw;
    if (!rd || !ctx.folding) return 0;
    const ty = tileType(tile);
    for (let i = 0; i < 3; i++) {
      if (rd[i] !== ty) continue;
      // Unless it is their winning tile, in which case they keep it and this
      // tile is the last one to let go of.
      if ((this.reads?.dealinP?.[i]?.[ty] ?? 0) > 0) continue;
      return this.aw.futureGenbutsu;
    }
    return 0;
  }
}
