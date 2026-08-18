// 順位効用 — the rank-utility layer. What points are WORTH, given the table.
//
// THE THESIS. A hanchan is not scored in points; it is scored in placements, and
// the map from one to the other is a step function over the final standings.
// Every policy in this project maximises point-EV, and the measurements say so:
// a planner arm bought +4400 points a game and converted exactly none of it into
// rank. That is the signature of an agent indifferent to variance — it buys the
// same lottery whether it is 20000 ahead or 20000 behind, because the only thing
// it prices is the mean. Under a step function variance is an asset class: the
// leader wants less of it (the step he stands on is the one he wants to keep),
// the last-place seat wants more of it (no step is worse than the one he stands
// on). Skill, in a placement game, is condition-dependent variance control, and
// this file is where the condition is computed.
//
// WHAT IT IS ALLOWED TO SEE. Scores, kyoku, kyotaku, honba — the four public
// numbers everyone at the table can read off the sticks. From them it builds a
// closed-form model of the final standings: each seat's remaining score change
// is a Gaussian with a FIXED population SD per kyoku, the same number for every
// seat. That is the same constitutional standing as `computed.ts` — a published
// base rate conditioned on public events, applied identically to all three
// opponents. There is no per-opponent inference here, no 河読み, no "he plays
// fast when he is behind". Swapping two opponents' scores swaps two numbers in
// the model and nothing else. The dojo compliance filter (規律) is untouched and
// sits entirely outside this layer.
//
// THE 雀鬼流 CORRECTION — THE RABBIT. Pure rank utility saturates: once first
// place is secure the model says nothing further can be gained, and every scale
// derived from it tells the agent to coast. That is win-by-not-losing, and the
// dojo forbids it — the instruction is to keep WINNING, even from first. So the
// utility carries a fifth, imaginary seat: "a virtual player at placement −1", a
// rabbit permanently `phantomLead` points above where I stand right now. The
// leader is by construction always second, and the marginal value of a win never
// falls to zero. The rabbit is anchored at the PRE-BRANCH score, so both the win
// and the loss branch are measured against the same fixed target — otherwise it
// would run away from every gain and the whole term would cancel. Set
// `phantomWeight: 0` and it vanishes exactly; that is the ablation switch that
// restores pure rank-EV, and the thing the tests use to show the difference.
//
// WHAT IT PRODUCES. Exactly two numbers, and they are the only exported
// consumption surface: `gain`, the marginal rank utility of winning this hand
// (the rabbit included) relative to a flat table at the same stage, and `risk`,
// the marginal cost of dealing in, likewise. Both are ratios centred on 1, so a
// policy consumes them as MULTIPLIERS on machinery it already has — the
// push/fold gate and the price of danger — and a table with four equal stacks
// leaves every decision bit-for-bit where it was. Nothing else here is meant to
// be read by a policy.

import type { Observation } from "../observe.ts";

export interface StandingsWeights {
  /** Population SD of ONE seat's score change over ONE kyoku. */
  sigmaPerKyoku: number;
  /** Nominal hanchan length in kyoku (東1..南4). */
  expectedKyoku: number;
  /** Reference value of winning the hand being decided. */
  refWin: number;
  /** Reference cost of dealing in. Matches `ASSUMED_LOSS` in `augmented.ts`. */
  refLoss: number;
  /**
   * The flat table the two scales are quoted against: everyone on the ruleset's
   * 配給原点 (`JANKI.startScore`, 30000持ち). Named rather than inlined because
   * "a flat table" is the reference the whole file is about — though note the
   * VALUE is arithmetically inert: every term below reads score DIFFERENCES, so
   * a flat 25000 table and a flat 30000 table produce the same reference. It is
   * here to be right, and to stay right if an absolute term is ever added.
   */
  startScore: number;
  /** Floor/ceiling on both scales — a model this coarse must not run away. */
  scaleMin: number;
  scaleMax: number;
  /** How far above me the rabbit always sits. See the header. */
  phantomLead: number;
  /** Its weight in the utility. 0 disables the rabbit exactly. */
  phantomWeight: number;
}

export const DEFAULT_STANDINGS_WEIGHTS: StandingsWeights = {
  sigmaPerKyoku: 5000,
  expectedKyoku: 8,
  refWin: 6000,
  refLoss: 6000,
  startScore: 30000,
  scaleMin: 0.25,
  scaleMax: 4,
  phantomLead: 8000,
  phantomWeight: 1,
};

/**
 * Standard normal CDF, via the Abramowitz–Stegun 7.1.26 rational approximation
 * to erf. Absolute error below 1.5e-7 — far inside anything a 5000-point SD
 * prior deserves — and, more importantly, a closed form with no table, no
 * iteration and no dependency, so the whole layer stays deterministic and cheap
 * enough to call several times per decision.
 */
export function phi(x: number): number {
  const s = x < 0 ? -1 : 1;
  const z = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * z);
  const poly =
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const erf = 1 - poly * Math.exp(-z * z);
  return 0.5 * (1 + s * erf);
}

/**
 * How many kyoku are still to be decided, the current one included — its
 * outcome is precisely what is being decided, so it counts. A 連荘 at All Last
 * repeats the same kyoku number and therefore stays pinned at 1, which is the
 * honest answer: one more hand, and it might be the last.
 */
export function kyokuRemaining(kyoku: number, w: StandingsWeights): number {
  return Math.max(1, w.expectedKyoku - kyoku);
}

/** SD of the difference between two seats' remaining score changes. */
function sigmaDiffOf(kyoku: number, w: StandingsWeights): number {
  return w.sigmaPerKyoku * Math.sqrt(2 * kyokuRemaining(kyoku, w));
}

export interface RankStats {
  /** P(I finish above opponent j) for relative seats 1..3, in that order. */
  pBeat: number[];
  /** P(final placement is r+1), r = 0..3. Sums to 1. */
  rankDist: number[];
  /** Mean of `rankDist`; 1 is best, 4 is worst. */
  expRank: number;
}

/**
 * The final-standings model over the three REAL opponents. Each seat's remaining
 * score change is an independent Gaussian with SD `sigmaPerKyoku` per kyoku, so
 * the DIFFERENCE between me and one opponent has SD `sigmaPerKyoku·√(2·remaining)`
 * and P(I stay ahead) is a single `phi` call. The three pairwise comparisons are
 * then treated as independent Bernoullis and all eight outcomes enumerated,
 * which is what turns pairwise odds into a placement distribution.
 *
 * The independence is a lie, and a deliberate one: the four score changes sum to
 * zero (plus sticks), so beating A and beating B are positively coupled through
 * the common pot. Modelling that needs a covariance the layer has no business
 * estimating, and the error it costs is second-order in a quantity we only ever
 * consume as a RATIO against the flat-table reference — where the same lie is
 * told identically and largely cancels.
 *
 * The rabbit is NOT here. This function reports the honest expected placement,
 * which is a fact about the table and is what a report or a test should read;
 * the virtual −1 seat exists only inside the utility `standingsScales` builds.
 *
 * Ties: at a score difference of zero `phi` returns exactly 0.5, the honest dead
 * heat. The real ruleset breaks that by seat order (起家 first); ignoring it
 * here is deliberate, because a tiebreak worth a fraction of a placement is
 * noise beside a 5000-point-per-kyoku prior, and folding it in would make the
 * flat-table reference asymmetric between seats for no gain.
 */
export function rankStats(
  scoresRel: number[],
  kyoku: number,
  w: StandingsWeights,
): RankStats {
  const sigmaDiff = sigmaDiffOf(kyoku, w);
  const pBeat: number[] = [];
  for (let j = 1; j < 4; j++) pBeat.push(phi((scoresRel[0] - scoresRel[j]) / sigmaDiff));

  const rankDist = [0, 0, 0, 0];
  for (let mask = 0; mask < 8; mask++) {
    let p = 1;
    let above = 0; // opponents who finish above me
    for (let j = 0; j < 3; j++) {
      const beaten = (mask & (1 << j)) !== 0;
      p *= beaten ? pBeat[j] : 1 - pBeat[j];
      if (!beaten) above++;
    }
    rankDist[above] += p;
  }
  let expRank = 0;
  for (let r = 0; r < 4; r++) expRank += (r + 1) * rankDist[r];
  return { pBeat, rankDist, expRank };
}

export interface StandingsScales {
  /** Multiplier on how much this hand is worth pushing for. */
  gain: number;
  /** Multiplier on the price of danger. */
  risk: number;
  /** The model's honest expected placement right now (rabbit excluded). */
  expRank: number;
  /**
   * The common factor divided out of both scales: how much ANY point movement
   * moves the placement here, compared with a flat table at the same stage.
   * Not consumed by any policy — see the normalisation note below — but reported
   * because it is exactly the quantity this layer refuses to have an opinion
   * about, and a reader should be able to see what was removed.
   */
  decisiveness: number;
}

const clamp = (x: number, lo: number, hi: number) => x < lo ? lo : x > hi ? hi : x;

/**
 * The two multipliers, from public facts alone.
 *
 * Utility is −(expected placement + the rabbit's chance of finishing above me),
 * so a change is worth what it does to that. Two hypothetical futures are priced
 * against the table as it stands: winning the current hand, and dealing into it.
 * Each is then divided by the SAME quantity computed at a flat 配給原点-all table
 * at the SAME stage of the game, so that a flat table yields 1 at any kyoku.
 *
 * THE NORMALISATION, and why the flat-table ratio alone is not enough. Those two
 * ratios share a factor nobody wants a policy to consume: how DECIDED the game
 * already is. At All Last a knife-edge flat table is maximally point-sensitive,
 * so a seat 20000 clear divides by a large denominator on BOTH branches and both
 * scales collapse together — which would clamp to the floor and tell the policy
 * that danger is cheap in exactly the game it most wants to protect. So the
 * common factor is measured (the geometric mean of the two raw ratios, reported
 * as `decisiveness`) and divided out, leaving `gain · risk = 1` before clamping.
 * That is a deliberate constraint on the layer's authority: it may ROTATE the
 * balance between offence and defence, and may never turn the whole policy up or
 * down. Everything it has to say is the asymmetry `gain / risk`; a flat table
 * has none, and consumes as exactly 1 × 1.
 */
export function standingsScales(obs: Observation, w: StandingsWeights): StandingsScales {
  const now = obs.scores.slice(0, 4);
  const kyoku = obs.kyoku;
  const sigmaDiff = sigmaDiffOf(kyoku, w);
  // The pot rides on the win branch and on nothing else: 供託 and 本場 are paid
  // by the table to whoever cashes, so they raise what a win is worth without
  // touching what a deal-in costs.
  const winDelta = w.refWin + 1000 * obs.kyotaku + 300 * obs.honba;

  /** −(expected placement + P(the rabbit finishes above me)). */
  const eu = (scores: number[], rabbit: number) =>
    -(rankStats(scores, kyoku, w).expRank +
      w.phantomWeight * phi((rabbit - scores[0]) / sigmaDiff));

  const winFrom = (s: number[]) => {
    // Zero-sum on `refWin`: the reference value comes out of the other three
    // stacks evenly (a tsumo, near enough), while the stick/honba part is pot.
    const out = [s[0] + winDelta, 0, 0, 0];
    for (let j = 1; j < 4; j++) out[j] = s[j] - w.refWin / 3;
    return out;
  };

  /**
   * The WORST deal-in available: the opponent whose collecting hurts the
   * placement most. Deliberately conservative for v1 — the base policy has no
   * per-seat estimate of who is actually waiting, and a layer that guessed one
   * would be reading behaviour. A subclass that already holds per-opponent
   * tenpai probabilities (`computed.ts` computes exactly that) should
   * probability-weight these three branches instead of taking the minimum.
   */
  const lossFrom = (s: number[], rabbit: number) => {
    let worst = Infinity;
    for (let j = 1; j < 4; j++) {
      const out = [s[0] - w.refLoss, s[1], s[2], s[3]];
      out[j] = s[j] + w.refLoss;
      worst = Math.min(worst, eu(out, rabbit));
    }
    return worst;
  };

  const flat = [w.startScore, w.startScore, w.startScore, w.startScore];
  // Anchored on the pre-branch score, once per table: a rabbit that ran with
  // each branch would be no rabbit at all.
  const rabbit = now[0] + w.phantomLead;
  const rabbitFlat = flat[0] + w.phantomLead;

  const euNow = eu(now, rabbit);
  const euFlat = eu(flat, rabbitFlat);

  const rawGain = (eu(winFrom(now), rabbit) - euNow) /
    Math.max(1e-9, eu(winFrom(flat), rabbitFlat) - euFlat);
  const rawRisk = Math.abs(lossFrom(now, rabbit) - euNow) /
    Math.max(1e-9, Math.abs(lossFrom(flat, rabbitFlat) - euFlat));

  const decisiveness = Math.sqrt(Math.max(1e-9, rawGain * rawRisk));
  const gain = clamp(rawGain / decisiveness, w.scaleMin, w.scaleMax);
  let risk = clamp(rawRisk / decisiveness, w.scaleMin, w.scaleMax);

  // 持ち点8000未満になる打ち方禁止. A deal-in that crosses the 8000 line is not
  // a score change the rank model gets to have an opinion about — it is a ledger
  // entry, and `finalStandings` drops a violator below every clean seat whatever
  // the points say. So the layer may raise the price of danger there and may
  // never discount it, no matter how little the placement model cares.
  if (obs.scores[0] - w.refLoss < 8000) risk = Math.max(risk, 1);

  return { gain, risk, expRank: rankStats(now, kyoku, w).expRank, decisiveness };
}
