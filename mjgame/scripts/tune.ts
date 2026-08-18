#!/usr/bin/env -S deno run --allow-read --allow-write --allow-run=deno
// 感性の最適化 — outcome tuning for the "k" seat, in either of two spaces.
//
// WHAT IS BEING OPTIMISED. Not a loss, not a per-decision label: the ONE number
// this project treats as the verdict, `dRankDojo.mean` from `paired` — seat 0's
// mean placement under the dojo's own ranking, minus the same seat's placement
// on the identical wall with the untouched baseline policy. Negative is better.
// Violations are not a term in the objective; they enter only through that
// ranking, which drops any seat carrying a ledger entry below every clean seat.
// A vector that buys points by fouling therefore scores WORSE, with no penalty
// shaping anywhere in this file.
//
// THE ALGORITHM is a cross-entropy method / (μ, λ) evolution strategy, which is
// what the noise level here allows: one candidate's advantage is worth ~0.05
// rank with a per-seed SD near 1, so nothing gradient-shaped survives, and even
// a rank ordering of a population is only trustworthy because both arms of every
// paired game share a wall (common random numbers) and every candidate in a
// generation shares the same seed lane.
//
// TWO THINGS ARE DELIBERATELY NOT DONE:
//  * The measurement lane (--seed=50000) is never played here. Tuning on the
//    lane you later validate on measures memorisation of 250 walls.
//  * In the SCALAR space `heuristic.shanten` (1000) is NOT tuned. It is the
//    NUMERAIRE: every other weight in the table is a price expressed in
//    shanten-steps, so scaling it would just rescale the whole vector and leave
//    the search wandering along a ridge of behaviourally identical points.
//
//   deno run --allow-read --allow-write --allow-run=deno scripts/tune.ts \
//     [--space=scalar|consumer] [--gens=16] [--pop=8] [--seeds=250] \
//     [--elite=3] [--sigma=…] [--sigma-final=…] [--margin-weight=…] \
//     [--base-ktune=runs/tune/best.json] [--params=a,b,c] [--curriculum] \
//     [--jobs=8] [--out=runs/tune] [--seed=1] [--resume]
//
// ===========================================================================
// THE INSTRUMENT — shared by both spaces (M9c-b in consumer, M10d in scalar)
// ===========================================================================
//
// Everything from "THE INSTRUMENT" down to "ABSOLUTE ANCHORING" below is now
// space-independent: a candidate is measured against the INCUMBENT, not against
// plain hhhh, whichever space it lives in. The only difference is which file the
// two arms differ by — `--consumer-b=<incumbent>` for a curve set,
// `--ktune-b=<incumbent>` for a scalar vector. Both are one code path here
// (`EvalSpec.incumbent`, `pairedArgs`), so the acceptance gate, the σ shrink on
// rejection, the every-fourth-generation absolute anchor and the
// incumbent-vs-itself null check apply identically to both.
//
// SCALAR-SPACE BASE (M10d). `--base-ktune` used to be consumer-only. In the
// scalar space it now names a PARTIAL ktune that every candidate file is written
// ON TOP OF: the file the child reads is deepMerge(base, toKtune(vec)), so the
// base supplies whatever the vector does not name (typically the M10c-fitted
// `computed` block — `weights/computed-calibrated.json` — including its twenty
// `tenpaiPrior` cells, which reach the seat through `mergeComputed`'s cell-by-
// cell merge) and the vector supplies the tuned scalars. The INCUMBENT file is
// written the same way, so the two arms still differ only in the tuned
// coordinates. There is NO default: the scalar space's base is empty unless
// asked for, which is exactly its pre-M10d behaviour.
//
// `--params=heuristic.dora,augment.floor` restricts the search to a SUBSET of
// the twelve. The unlisted parameters are not perturbed, not recombined and not
// written — they keep tracking their defaults (or the base's values) and are
// therefore bit-identical in the candidate and in the incumbent. The subset is
// re-ordered into the table's own order, so `--params=b,a` and `--params=a,b`
// are the same run; it is recorded in state.json's `paths`, and a `--resume`
// that names a different subset is refused rather than reinterpreted.
//
// ===========================================================================
// --space=consumer (M9c): the 68 θ of `src/ai/consumer.ts`
// ===========================================================================
//
// The scalar space tunes twelve PRICES that the hand-written score then consumes
// in a fixed arithmetic shape. The consumer space tunes the SHAPE: seventeen
// monotone piecewise-linear curves over named evidence features, four θ each.
// The two compose — a consumer candidate is always evaluated ON TOP of the
// scalar champion (`--base-ktune`, default runs/tune/best.json), because the
// 計算 side of the seat (riskOf's ladder, the fold scales, the counting reader's
// constants) is that vector's business and stays where it was measured.
//
// PERTURBATION, and why a single global σ would be wrong. The θ of this space
// are not commensurate: the shanten curve's increments are ~2000-3000 (a
// shanten step IS a thousand score points) while `atkEff`'s are 0.25 (a
// dimensionless multiplier). A lognormal step, the scalar space's operator, is
// unavailable outright — six curves start at θ = 0 (dangerLevel, safe,
// atkPressure, atkJunme, defPressure, defJunme) and log 0 is not a number, and
// two of the live ones legitimately want to go negative. So:
//
//  (a) EVERY curve carries a hand-set STEP UNIT in the units of its own θ (see
//      `CURVE_STEP`), and one candidate's step is `θ + unit · σ · N(0,1)` per
//      component. σ is therefore dimensionless and anneals as usual; the unit
//      carries the dimension. Live curves get roughly a tenth of their init
//      magnitude; that is a step a 1000-seed paired run can actually resolve.
//  (b) DORMANT curves — the six at θ = 0 or θ = const 1 — get a unit derived
//      from their ROLE, never from their (zero) init, or they could never leave
//      it. `dangerLevel` and `safe` are score-point ladders read alongside
//      `risk`, so they move in tens of points per level; the four context
//      multipliers sit at the constant 1 and move in fractions of it (0.1).
//      Reachability is the whole point of the design: the search must be able to
//      discover that the danger ORDINAL is worth pricing separately from the
//      continuous risk, and that discovery is what the curves are for.
//  (c) θ₀ is a signed LEVEL and is stepped as such. θ₁..θ₃ are INCREMENTS whose
//      sign is meaningless — `curveValues` reads |θᵢ| — so their step reflects
//      at zero (`|θᵢ + unit·σ·N|`), which is the same distribution a free sign
//      flip would give and keeps the stored θ canonical (nonnegative). That
//      canonical form is what makes (d) meaningful.
//  (d) ELITE RECOMBINATION is the plain arithmetic mean in θ space. The scalar
//      space's geometric mean is not available and would be wrong anyway: θ can
//      be exactly 0 (every dormant curve) and θ₀ can be negative (the shanten
//      level is −1000). With increments canonicalised nonnegative by (c), the
//      arithmetic mean of two elites' magnitudes is their average magnitude —
//      which is the reading a mean is supposed to have.
//
// No bounds table. Monotonicity is structural (|θ|), so unlike the scalar space
// there is no constraint a step could violate; the only clamping is a repair of
// non-finite entries.
//
// THE INSTRUMENT (M9c-b), and why the first consumer run failed. A candidate
// used to be measured the way every arm in this project is measured: against
// plain `hhhh`. That is the right control for "is this seat better than the
// baseline", and the WRONG one for "is this candidate better than the one I
// already have". The two arms are different players, every game diverges on the
// first turn, and the per-seed SD of the rank difference is ~1.4 — so even 1000
// seeds leave a ±0.08 CI around each candidate while genuine neighbour
// differences are 0.02-0.04. The first 16-generation run duly selected on noise:
// gen-8 elites printed −0.105, gen-15's mean measured −0.036 on its own lane,
// and the champion validated at −0.007 — worse than the scalar champion it was
// built on. Nothing was wrong with the space; the ruler had no marks on it.
//
// So each candidate is now paired against the INCUMBENT (`--consumer-b`): same
// seeds, same policy seeds, same `--ktune`, same oracle wiring — only the curve
// file differs. A small θ perturbation leaves most games bit-identical to the
// end, and the difference is measured only where the two actually part. Measured
// on this harness, a single-knot nudge over 40 seeds: SD 1.339 against hhhh,
// 0.276 against the incumbent — a 23× variance reduction, and the same seeds now
// resolve what they could not before. An incumbent-vs-itself run returns exactly
// 0.0000 with SD 0, which is the instrument's own null check and is printed
// every generation (candidate 0 IS the incumbent).
//
// ACCEPTANCE. Elites win on the walls they were drawn against; their arithmetic
// mean is a new vector that has won nothing. It is therefore CONFIRMED against
// the incumbent on a fresh lane (600000 + gen·1000, disjoint from every
// generation lane — `lane+500` would not be fresh at all at 1000 seeds) and
// adopted only if that difference is ≤ 0 (`--accept-ci` demands the whole
// interval be below zero instead). A rejected proposal leaves the incumbent
// standing and multiplies σ by 0.85: the step outran the signal, so the next
// question is asked smaller. This is the mechanism that stops the drift.
//
// ABSOLUTE ANCHORING. Every fourth generation the incumbent is also measured the
// old way — against plain `hhhh`, on the generation's lane — purely so the log
// answers "and where are we, in the units the acceptance bar is written in".
// Selection never reads that row.
//
// FITNESS, and the phantom margin. Primary is `dRankDojo.mean`, exactly as
// before. Added to it is a TIE-BREAK on raw points:
//
//     fitness = dRankDojo.mean − marginWeight · (dScore.mean / 1000)
//
// with marginWeight 0.005 in the consumer space (0 in the scalar space, which
// therefore grades on the rank alone, exactly as it always has). Both terms are
// RELATIVE TO THE INCUMBENT in either space — negative still means better, it
// just means better than the vector in hand rather than better than hhhh. At that weight a full 1000-point margin is worth
// 0.005 rank — an order of magnitude below the smallest rank difference this
// harness can resolve, so it can never overturn a real ranking; it only decides
// between candidates the rank statistic calls a tie. This is the EPISODE-LEVEL
// (telescoped) form of the 順位効用 phantom-margin credit: per-kyoku the
// potential-based Φ-deltas of that scheme telescope to Φ(end) − Φ(start), so the
// only part of it that survives into an episodic ES objective is the end-of-game
// margin. Shaping it per decision would add variance and change nothing.
//
// ===========================================================================
// --curriculum (M9c, consumer space only)
// ===========================================================================
//
// A candidate's arm A additionally gets `--oracle=C1,C2,C3 --curriculum=<E>`:
// seat 0's reader answers each information group from the ORACLE with
// probability 1−E and from the 計算 reader with probability E, per decision.
// E anneals across the run from 0.35 to 1.0 with a deliberate DWELL in the
// measured hard region [0.5, 0.75] — piecewise linear in t = gen/(gens−1):
// t∈[0,¼] → 0.35..0.50, t∈[¼,¾] → 0.50..0.75, t∈[¾,1] → 0.75..1.00, i.e. at
// gens=16 four generations of ramp-in, eight of dwell, four of ramp-out.
// The champion is ALWAYS re-scored at E=1 — no oracle flags at all, the seat as
// it ships — before anything is written to best.json, and is compared there
// against the init curves on the same fresh lane. A consumer that only works
// with oracle help does not survive that comparison and is not written.

import { sfc32 } from "../src/rng.ts";
import type { Rng } from "../src/rng.ts";
import { DEFAULT_WEIGHTS } from "../src/ai/heuristic.ts";
import type { HeuristicWeights } from "../src/ai/heuristic.ts";
import { ALL_SPECS, curveValues, initFromWeights, serializeConsumer } from "../src/ai/consumer.ts";
import type { ConsumerParams, CurveKey, CurveParams } from "../src/ai/consumer.ts";

/** Which parameterisation the search is running over. */
export type SearchSpace = "scalar" | "consumer";

// ---------------------------------------------------------------------------
// the parameter space — scalar
// ---------------------------------------------------------------------------

export interface ParamSpec {
  /** Dotted path into the ktune file: section, then key (then danger level). */
  path: string;
  /** The shipped default — generation 0's mean, and the report's reference. */
  def: number;
  min: number;
  max: number;
}

/**
 * The 12 free parameters. Every one is strictly positive, which is why the whole
 * search runs in log space: these are PRICES and RATES, so a candidate should be
 * "half as afraid of a 危険度高 tile", never "30 points less afraid" — and a
 * Gaussian step in linear space would happily propose a negative danger cost,
 * i.e. a seat that seeks out deal-ins.
 *
 * Bounds are default/4 .. default×4 (a factor of 16 of range, which is far more
 * than a hand-set constant is usually wrong by), tightened to [0, 1] for the two
 * genuine fractions — `augment.floor` is a share of the rule-based risk and
 * `computed.tenpaiFloor` is a probability threshold; above 1 both stop meaning
 * anything.
 */
export const PARAMS: readonly ParamSpec[] = [
  // efficiency
  { path: "heuristic.ukeire", def: 12, min: 3, max: 48 },
  { path: "heuristic.ukeireType", def: 4, min: 1, max: 16 },
  { path: "heuristic.dora", def: 60, min: 15, max: 240 },
  // danger, the three graded levels ("安全" is a proof and stays 0)
  { path: "heuristic.danger.危険度低", def: 30, min: 7.5, max: 120 },
  { path: "heuristic.danger.危険度中", def: 90, min: 22.5, max: 360 },
  { path: "heuristic.danger.危険度高", def: 200, min: 50, max: 800 },
  // push/fold
  { path: "heuristic.foldDanger", def: 10, min: 2.5, max: 40 },
  { path: "heuristic.foldEfficiency", def: 0.05, min: 0.0125, max: 0.2 },
  // the augmented consumption terms
  { path: "augment.lambda", def: 0.25, min: 0.0625, max: 1 },
  { path: "augment.floor", def: 0.5, min: 0.125, max: 1 },
  // the counting reader's two calibration constants
  { path: "computed.dealinScale", def: 0.065, min: 0.01625, max: 0.26 },
  { path: "computed.tenpaiFloor", def: 0.25, min: 0.0625, max: 1 },
];

/** Generation 0's mean: the shipped vector, so gen 0 measures the status quo. */
export function defaults(params: readonly ParamSpec[] = PARAMS): number[] {
  return params.map((p) => p.def);
}

/** Elementwise clamp into the table's bounds. */
export function clampVec(v: readonly number[], params: readonly ParamSpec[] = PARAMS): number[] {
  return params.map((p, i) => {
    const x = v[i];
    if (!Number.isFinite(x)) return p.def;
    return x < p.min ? p.min : x > p.max ? p.max : x;
  });
}

// ---------------------------------------------------------------------------
// the ktune file shape
// ---------------------------------------------------------------------------

export type Nested = Record<string, unknown>;

/** Write `value` at a dotted path, creating intermediate objects. Exported for tests. */
export function setPath(root: Nested, path: string, value: number): void {
  const parts = path.split(".");
  let node = root;
  for (let i = 0; i < parts.length - 1; i++) {
    const k = parts[i];
    const cur = node[k];
    if (typeof cur !== "object" || cur === null) node[k] = {};
    node = node[k] as Nested;
  }
  node[parts[parts.length - 1]] = value;
}

/**
 * A vector as a `--ktune` file: `{heuristic, augment, computed}`, each section
 * a partial merged over its own defaults by the constructor that receives it.
 * Only the tuned keys appear — an untuned field must keep tracking its default,
 * including when that default later changes in source.
 */
export function toKtune(v: readonly number[], params: readonly ParamSpec[] = PARAMS): Nested {
  const out: Nested = {};
  params.forEach((p, i) => setPath(out, p.path, v[i]));
  return out;
}

const isPlainObject = (x: unknown): x is Nested =>
  typeof x === "object" && x !== null && !Array.isArray(x);

/**
 * `ov` over `base`, leaf by leaf. Plain objects recurse; ARRAYS AND SCALARS
 * REPLACE, which is the right rule here because the only array a ktune file
 * carries is `computed.tenpaiPrior` and the search vector never names it — so
 * the base's fitted table passes through untouched and is merged cell by cell
 * later, by `mergeComputed` in the child. Neither argument is mutated.
 */
export function deepMerge(base: Nested, ov: Nested): Nested {
  const out: Nested = { ...base };
  for (const [k, v] of Object.entries(ov)) {
    const b = out[k];
    out[k] = isPlainObject(v) && isPlainObject(b) ? deepMerge(b, v) : v;
  }
  return out;
}

/**
 * The ktune file one scalar candidate is written as: the vector's coordinates
 * over the optional base partial. Without a base this is `toKtune` exactly —
 * the pre-M10d file, so a run that asks for no base gets byte-identical output.
 */
export function candidateKtune(
  v: readonly number[],
  params: readonly ParamSpec[] = PARAMS,
  base?: Nested,
): Nested {
  const k = toKtune(v, params);
  return base ? deepMerge(base, k) : k;
}

/**
 * `--params=a,b,c` → the subset of the table to search over, in the TABLE's
 * order (so the coordinate layout is a function of the set, not of the typing
 * order). Throws on an unknown or repeated path: a typo that silently searched
 * eleven parameters would be found only by reading the report.
 */
export function selectParams(spec: string): ParamSpec[] {
  const names = spec.split(",").map((s) => s.trim()).filter((s) => s !== "");
  if (names.length === 0) throw new Error("--params が空です");
  const want = new Set<string>();
  for (const n of names) {
    if (!PARAMS.some((p) => p.path === n)) {
      throw new Error(
        `--params に未知のパラメータ: ${n}\n選べるのは:\n  ${
          PARAMS.map((p) => p.path).join("\n  ")
        }`,
      );
    }
    if (want.has(n)) throw new Error(`--params にパラメータの重複: ${n}`);
    want.add(n);
  }
  return PARAMS.filter((p) => want.has(p.path));
}

// ---------------------------------------------------------------------------
// the parameter space — consumer (M9c)
// ---------------------------------------------------------------------------

/**
 * The step unit of each curve, in the units of that curve's own θ — see the
 * header, (a) and (b). One number per curve, applied to all four of its θ.
 *
 * The live curves are set at roughly a tenth of their init magnitude under the
 * scalar champion; the dormant six are set from their ROLE, since a fraction of
 * their init would be a fraction of zero:
 *
 *   曲線             init θ (champion)          歩幅  根拠
 *   shanten          [-1000,2000,2000,3000]      200  向聴一歩=1000点の1/5
 *   ukeire           [0,60,60,121]                10  受け入れ1枚ぶんの値段
 *   ukeireType       [0,16,16,33]                  4  受け入れ1種類ぶん
 *   dora             [0,132,132,264]              25  ドラ半枚ぶん弱
 *   yakuhaiPair      [0,40,40,80]                 12  役牌対子の1/3
 *   isolatedHonor    [0,72,72,144]                15  孤立字牌×巡目 2.5単位
 *   risk             [0,100,100,200]              15  放銃100点帯あたり15点
 *   dangerLevel      0 (休眠)                      20  段位1つ=数十点 ←役割から
 *   safe             0 (休眠)                      20  現物の証明=数十点 ←役割から
 *   drawBonus        [0,200,200,600]              30  先読み加点の1/6
 *   keepBonus        [0,1000,1500,2500]          150  温存減点の1/7
 *   atkEff           [0,0.25,0.25,0.5]          0.05  倍率、定数1の5%
 *   atkPressure      [1,0,0,0] (休眠)            0.1  定数1まわりの1割 ←役割から
 *   atkJunme         [1,0,0,0] (休眠)            0.1  同上
 *   defScale         [0,1,1,2]                  0.15  倍率
 *   defPressure      [1,0,0,0] (休眠)            0.1  定数1まわりの1割 ←役割から
 *   defJunme         [1,0,0,0] (休眠)            0.1  同上
 *
 * The shanten curve is NOT frozen the way the scalar space freezes
 * `heuristic.shanten`. There it is a pure numeraire and rescaling it is a null
 * move; here it is not, because `drawBonus`, `keepBonus` and `dojoCost` enter
 * the score OUTSIDE `M_atk` in absolute points — so the ratio between the
 * attack group and those absolutes is a real degree of freedom.
 */
export const CURVE_STEP: Readonly<Record<CurveKey, number>> = {
  shanten: 200,
  ukeire: 10,
  ukeireType: 4,
  dora: 25,
  yakuhaiPair: 12,
  isolatedHonor: 15,
  risk: 15,
  dangerLevel: 20,
  safe: 20,
  drawBonus: 30,
  keepBonus: 150,
  atkEff: 0.05,
  atkPressure: 0.1,
  atkJunme: 0.1,
  defScale: 0.15,
  defPressure: 0.1,
  defJunme: 0.1,
};

/** 17 curves × 4 θ, in `ALL_SPECS` order. The search vector's layout. */
export const CONSUMER_DIM = ALL_SPECS.length * 4;

/** `ConsumerParams` → the flat search vector. */
export function consumerToVec(p: ConsumerParams): number[] {
  const out: number[] = [];
  for (const spec of ALL_SPECS) out.push(...p.curves[spec.key]);
  return out;
}

/**
 * The flat search vector → `ConsumerParams`. Increments are canonicalised
 * nonnegative on the way in (see header (c)); a non-finite entry is repaired to
 * 0, which for an increment means "flat here" and for θ₀ means "no level".
 */
export function vecToConsumer(v: readonly number[]): ConsumerParams {
  const curves = {} as Record<CurveKey, CurveParams>;
  ALL_SPECS.forEach((spec, c) => {
    const t = [0, 1, 2, 3].map((j) => {
      const x = v[c * 4 + j];
      if (!Number.isFinite(x)) return 0;
      return j === 0 ? x : Math.abs(x);
    });
    curves[spec.key] = [t[0], t[1], t[2], t[3]];
  });
  return { version: 1, curves };
}

/** Per-component step units, flattened to match `consumerToVec`. */
export function consumerSteps(): number[] {
  const out: number[] = [];
  for (const spec of ALL_SPECS) {
    const u = CURVE_STEP[spec.key];
    out.push(u, u, u, u);
  }
  return out;
}

/**
 * One consumer candidate: an independent Gaussian step per component, scaled by
 * that curve's unit and the generation's σ. Increments reflect at zero, θ₀ does
 * not — the level is signed and means what its sign says.
 */
export function perturbConsumer(
  mean: readonly number[],
  sigma: number,
  rng: Rng,
): number[] {
  const step = consumerSteps();
  return mean.map((m, i) => {
    const x = m + step[i] * sigma * gauss(rng);
    if (!Number.isFinite(x)) return m;
    // Component 0 of each curve is the signed level; 1..3 are |increments|.
    return i % 4 === 0 ? x : Math.abs(x);
  });
}

/** The plain arithmetic mean — see header (d). */
export function meanVec(vs: readonly (readonly number[])[]): number[] {
  if (vs.length === 0) return [];
  const n = vs[0].length;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (const v of vs) s += v[i];
    out.push(s / vs.length);
  }
  return out;
}

/**
 * The merged weights a "k" seat actually holds under a `--ktune` file: defaults,
 * then the file's `heuristic` section, with `danger` merged level by level so a
 * partial override cannot drop a level. The SAME merge `scripts/consumer_init.ts`
 * performs — a consumer fitted on top of the scalar champion has to start from
 * the champion's score, not the shipped one.
 */
export function mergedWeights(ktunePath: string): HeuristicWeights {
  if (!ktunePath) return { ...DEFAULT_WEIGHTS, danger: { ...DEFAULT_WEIGHTS.danger } };
  const json = JSON.parse(Deno.readTextFileSync(ktunePath)) as {
    heuristic?: Partial<HeuristicWeights>;
  };
  return {
    ...DEFAULT_WEIGHTS,
    ...json.heuristic,
    danger: { ...DEFAULT_WEIGHTS.danger, ...json.heuristic?.danger },
  };
}

/** Generation 0's mean in the consumer space: the init curves, as a vector. */
export function consumerDefaults(ktunePath: string): number[] {
  return consumerToVec(initFromWeights(mergedWeights(ktunePath)));
}

/**
 * The curriculum's ε for one generation: 0.35 → 1.0, dwelling in [0.5, 0.75].
 * Piecewise linear in t = gen/(gens−1), with breakpoints at t = ¼ and t = ¾, so
 * a 16-generation run spends 4 generations ramping in, 8 in the hard band and 4
 * ramping out — and the LAST generation is played at ε = 1 exactly.
 */
export function epsAt(gen: number, gens: number): number {
  if (gens <= 1) return 0.35;
  const t = Math.min(1, Math.max(0, gen / (gens - 1)));
  if (t <= 0.25) return 0.35 + (0.5 - 0.35) * (t / 0.25);
  if (t <= 0.75) return 0.5 + (0.75 - 0.5) * ((t - 0.25) / 0.5);
  return 0.75 + (1 - 0.75) * ((t - 0.75) / 0.25);
}

// ---------------------------------------------------------------------------
// the search operators
// ---------------------------------------------------------------------------

/** Standard normal, Box–Muller. Seeded — never `Math.random`: a tuning run must
 *  replay exactly from `--seed` alone, including after `--resume`. */
export function gauss(rng: Rng): number {
  // `float()` can return 0, and log(0) is -Infinity.
  const u = 1 - rng.float();
  const v = rng.float();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * One candidate: an independent lognormal step per dimension, clamped.
 * `sigma` is in LOG units, so 0.25 is "about ±28% per parameter, per step".
 */
export function perturb(
  mean: readonly number[],
  sigma: number,
  rng: Rng,
  params: readonly ParamSpec[] = PARAMS,
): number[] {
  return clampVec(mean.map((m) => Math.exp(Math.log(m) + sigma * gauss(rng))), params);
}

/**
 * The geometric (log-space) mean of the elite. The counterpart of the lognormal
 * step: the arithmetic mean of {x/4, 4x} is 2.1x, which would drag the mean
 * upward on every generation that disagreed with itself; the geometric mean of
 * that pair is x.
 */
export function logMean(vs: readonly (readonly number[])[]): number[] {
  if (vs.length === 0) return [];
  const n = vs[0].length;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (const v of vs) s += Math.log(v[i]);
    out.push(Math.exp(s / vs.length));
  }
  return out;
}

/** Geometric annealing of the step size across the whole run. */
export function sigmaAt(gen: number, gens: number, s0: number, s1: number): number {
  if (gens <= 1) return s0;
  return s0 * Math.pow(s1 / s0, gen / (gens - 1));
}

// ---------------------------------------------------------------------------
// evaluation
// ---------------------------------------------------------------------------

interface Diff {
  mean: number;
  sd: number;
  ci: number;
}

/** `src/main.ts paired --json`, exactly. */
export interface PairedJson {
  games: number;
  seed: number;
  seats: string;
  rankA: number;
  rankB: number;
  dRank: Diff;
  dRankDojo: Diff;
  dScore: Diff;
  vioA0: Record<string, number>;
  vioB0: Record<string, number>;
  ms: number;
}

export interface Candidate {
  /** 0 = the carried-over mean, 1.. = perturbations. */
  idx: number;
  vec: number[];
  /** null when the child process failed; such a candidate can never be elite. */
  stats: PairedJson | null;
  err?: string;
}

/** Total ledger entries in one arm's per-label breakdown. */
export function vioTotal(m: Record<string, number> | undefined): number {
  return m ? Object.values(m).reduce((a, b) => a + b, 0) : 0;
}

/**
 * The graded quantity: dojo rank difference, with raw margin as a tie-break.
 * See the header — `marginWeight` 0 (the scalar space's setting, and the
 * default here) reduces this to the original objective exactly.
 */
export function fitnessOf(st: PairedJson, marginWeight = 0): number {
  return st.dRankDojo.mean - marginWeight * (st.dScore.mean / 1000);
}

/**
 * Elite selection: best (most negative) fitness first. Failed runs sort last
 * and are then dropped, so a crashed candidate cannot poison the new mean by
 * being scored 0 (which on this objective is a perfectly average result).
 * Ties break by index, so selection is a deterministic function of the inputs.
 */
export function select(
  cands: readonly Candidate[],
  elite: number,
  marginWeight = 0,
): Candidate[] {
  const ok = cands.filter((c) => c.stats !== null);
  const sorted = [...ok].sort((a, b) =>
    fitnessOf(a.stats!, marginWeight) - fitnessOf(b.stats!, marginWeight) || a.idx - b.idx
  );
  return sorted.slice(0, Math.max(1, elite));
}

// ---------------------------------------------------------------------------
// the runner
// ---------------------------------------------------------------------------

/** The repo root, so the child `deno run src/main.ts` resolves regardless of cwd. */
const ROOT = decodeURIComponent(new URL("../", import.meta.url).pathname);

/**
 * The measurement lane, reserved. Tuning generations rotate through
 * base + gen·1000 so no generation reuses another's walls (which would let the
 * mean drift toward whatever those particular seeds rewarded) and none of them
 * ever touches the seeds the champion is finally validated on. The two spaces
 * get disjoint bases as well, so a consumer run is never graded on the walls the
 * scalar champion was fitted to.
 */
const VALIDATE_SEED = 50_000;
const TUNE_LANE_BASE = 200_000;
const CONSUMER_LANE_BASE = 400_000;
/**
 * The ACCEPTANCE lanes, disjoint from every generation lane. The obvious
 * `lane + 500` is not fresh at all once `--seeds=1000`: it is the second half of
 * this generation's walls and the first half of the next one's, so a proposal
 * would be confirmed on the seeds that selected it. A separate base is the only
 * offset that is fresh for every seed count this harness is run at.
 */
const ACCEPT_LANE_BASE = 600_000;

/** Lane for a generation. Distinct per generation, never the validation lane. */
export function laneOf(gen: number, space: SearchSpace = "scalar"): number {
  return (space === "consumer" ? CONSUMER_LANE_BASE : TUNE_LANE_BASE) + gen * 1000;
}

/**
 * What a rejected proposal costs the next generation's step size. The search
 * proposed something the instrument could not confirm; the honest reading is
 * that the step outran the signal, so the next question is asked smaller.
 */
const REJECT_SHRINK = 0.85;

/** Lane for a generation's acceptance test. Never a generation lane. */
export function acceptLaneOf(gen: number): number {
  return ACCEPT_LANE_BASE + gen * 1000;
}

/** The oracle channels the curriculum arm reads before dropout. */
const CURRICULUM_CHANNELS = "C1,C2,C3";

/**
 * The `paired` argv for one candidate. Exported so a test — and a reader — can
 * see exactly what reaches the child, which is the only place a candidate can
 * actually differ from the baseline.
 */
export function pairedArgs(
  opt: Options,
  candFile: string,
  lane: number,
  eps?: number,
  incumbent?: string,
): string[] {
  const args = [
    "run",
    "--allow-read",
    "--allow-write",
    "--allow-ffi",
    "--allow-env=MJGAME_NATIVE",
    "src/main.ts",
    "paired",
    `--games=${opt.seeds}`,
    `--seed=${lane}`,
    "--seats=khhh",
  ];
  if (opt.space === "consumer") {
    // The scalar champion underneath, then the candidate's curves on top of it.
    if (opt.baseKtune) args.push(`--ktune=${opt.baseKtune}`);
    args.push(`--consumer=${candFile}`);
    // The INCUMBENT as the control arm — the variance-reduction instrument.
    // Absent, the control is plain hhhh and the measurement is absolute.
    if (incumbent) args.push(`--consumer-b=${incumbent}`);
    // ε undefined = the seat as it ships: no oracle machinery in the path at all.
    if (eps !== undefined) {
      args.push(`--oracle=${CURRICULUM_CHANNELS}`, `--curriculum=${eps}`);
    }
  } else {
    // The candidate file ALREADY carries `--base-ktune` merged underneath it
    // (see `writeCandidate`), so one flag describes the whole seat — and the
    // incumbent file, written the same way, differs from it only in the tuned
    // coordinates. Absent an incumbent the control is plain hhhh, i.e. the
    // absolute measurement the anchor rows want.
    args.push(`--ktune=${candFile}`);
    if (incumbent) args.push(`--ktune-b=${incumbent}`);
  }
  args.push("--json");
  return args;
}

/** Serialize a candidate to the file the child will read. */
async function writeCandidate(vec: number[], file: string, opt: Options): Promise<void> {
  const text = opt.space === "consumer"
    ? serializeConsumer(vecToConsumer(vec))
    : JSON.stringify(candidateKtune(vec, opt.params, opt.baseObj), null, 2) + "\n";
  await Deno.writeTextFile(file, text);
}

/**
 * How one vector is recorded in `history.jsonl`: the curves in the consumer
 * space, the SEARCHED parameters (`--params`, all twelve by default) in the
 * scalar one. Coordinates the run did not search are not recorded, because a
 * row that listed them would suggest they had been measured.
 */
function vecRecord(vec: readonly number[], opt: Options): Nested {
  return opt.space === "consumer"
    ? { curves: vecToConsumer(vec).curves }
    : { params: Object.fromEntries(opt.params.map((p, i) => [p.path, vec[i]])) };
}

/** One measurement: which vector, on which walls, against which control. */
interface EvalSpec {
  vec: number[];
  /** Population index; negative values mark the out-of-population measurements. */
  idx: number;
  lane: number;
  /** Where this candidate's own file is written. */
  file: string;
  eps?: number;
  /**
   * The control arm's file — a curve set in the consumer space, a 感性 vector in
   * the scalar one. Absent ⇒ the control is plain hhhh.
   */
  incumbent?: string;
}

async function evaluate(spec: EvalSpec, opt: Options): Promise<Candidate> {
  const { vec, idx, lane, file, eps, incumbent } = spec;
  await writeCandidate(vec, file, opt);
  const args = pairedArgs(opt, file, lane, eps, incumbent);
  const cmd = new Deno.Command("deno", {
    args,
    cwd: ROOT,
    stdout: "piped",
    stderr: "piped",
  });
  const out = await cmd.output();
  const text = new TextDecoder().decode(out.stdout).trim();
  const errText = new TextDecoder().decode(out.stderr).trim();
  const c: Candidate = { idx, vec, stats: null };
  if (!out.success) {
    c.err = `exit ${out.code}: ${errText.slice(-400)}`;
    return c;
  }
  try {
    const line = text.split("\n").filter((l) => l.trim() !== "").pop() ?? "";
    c.stats = JSON.parse(line) as PairedJson;
  } catch (e) {
    c.err = `JSONを解釈できません: ${e instanceof Error ? e.message : e} :: ${text.slice(-200)}`;
  }
  return c;
}

/**
 * `Promise.all` with a ceiling on how many run at once. A generation is still
 * one batch on one seed lane — this only stops eight native-kernel children from
 * becoming sixteen when a population is widened.
 */
async function mapLimit<T, R>(
  items: readonly T[],
  limit: number,
  fn: (t: T, i: number) => Promise<R>,
): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const width = Math.max(1, Math.min(limit, items.length));
  await Promise.all(
    Array.from({ length: width }, async () => {
      for (;;) {
        const i = next++;
        if (i >= items.length) return;
        out[i] = await fn(items[i], i);
      }
    }),
  );
  return out;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface Options {
  space: SearchSpace;
  gens: number;
  pop: number;
  seeds: number;
  elite: number;
  sigma: number;
  sigmaFinal: number;
  /** Rank-units credited per 1000 points of raw margin. See the header. */
  marginWeight: number;
  /**
   * The `--ktune` partial every candidate sits on top of. In the consumer space
   * it is handed to the child as `--ktune=` (the scalar champion under the
   * curves, default runs/tune/best.json); in the scalar space it is MERGED into
   * each candidate file (`--base-ktune`, no default). Either way the candidate
   * and the incumbent receive it identically, so it cannot move the difference.
   */
  baseKtune: string;
  /**
   * Scalar space: `baseKtune` parsed, or undefined. Loaded once — the candidate
   * files are written from it several times per generation.
   */
  baseObj?: Nested;
  /** Scalar space: the coordinates being searched (`--params`; default: all 12). */
  params: readonly ParamSpec[];
  /** Consumer space: anneal an oracle→計算 curriculum across the generations. */
  curriculum: boolean;
  /**
   * Acceptance rule. Default: the point estimate must be ≤ 0. With `--accept-ci`
   * the whole 95% interval must be below 0 — far stricter, and far slower, but
   * it makes every accepted step a measured improvement rather than a coin that
   * landed the right way.
   */
  acceptCi: boolean;
  /** Ceiling on concurrent `paired` children. */
  jobs: number;
  out: string;
  seed: number;
  resume: boolean;
}

const USAGE = `scripts/tune.ts — 感性ベクトルの成績最適化 (CEM/ES)

  deno run --allow-read --allow-write --allow-run=deno scripts/tune.ts [options]

  --space=scalar     探索空間 (既定 scalar)
                     scalar   = --ktune の12スカラ (対数空間・幾何平均)
                     consumer = M9消費曲線の68θ (score単位の歩幅・算術平均)
  --gens=16          世代数
  --pop=8            1世代の候補数 (1つは現平均の再評価)
  --seeds=250        1候補あたりの対応ペア数 (paired --games)
                     consumer 空間の既定は 1000
  --elite=3          新平均に採る上位数
  --sigma=0.25       初期ステップ幅 (scalar: 対数空間 / consumer: 歩幅の倍率、既定 1.0)
  --sigma-final=0.10 最終ステップ幅 (幾何的に減衰、consumer 既定 0.4)
  --margin-weight=W  素点1000点あたりの順位換算 (consumer 既定 0.005、scalar 既定 0)
  --base-ktune=PATH  下敷きにする --ktune 部分ファイル。
                     consumer 空間: 曲線の下に敷く12スカラ champion を子に
                       --ktune= として渡す (既定 runs/tune/best.json)
                     scalar 空間: 各候補ファイルを deepMerge(base, ベクトル) で
                       書き出す。ベクトルが触らない項 (例 M10c の computed 較正
                       weights/computed-calibrated.json の tenpaiPrior 20セル) は
                       base のまま席に届く。既定なし (=従来どおり素の12スカラ)
  --params=A,B,C     scalar 空間で探索するパラメータを表の部分集合に絞る
                     (ドット区切りのパス。未知のパスはエラー)。挙げなかった項は
                     既定 (または base) のまま、候補にも現行にも同じ値で入る
  --curriculum       consumer 空間で「オラクル→計算」カリキュラムを焚く。
                     εを 0.35→1.0 で焼き鈍し、[0.5,0.75] に長く留まる。
                     最終 champion は必ず ε=1 (純・計算) で採点し直す
  --accept-ci        受理判定を厳しくする (点推定 ≤0 ではなく 95%CI 全体が <0)
  --jobs=8           同時に走らせる paired 子プロセスの上限
  --out=runs/tune    履歴・状態・最良ベクトルの出力先
                     (consumer 空間の既定は runs/tune-consumer)
  --seed=1           探索の乱数シード (牌山シードとは別物)
  --resume           <out>/state.json の続きから再開
  --help, -h         このヘルプ
`;

/**
 * The CLI, as a function. Exported so a test can build an `Options` the same
 * way the run does — the defaults are space-sensitive, and a test that hand-rolled
 * the record would not be testing them.
 */
export function parseOpts(argv: string[]): Options {
  const o: Options = {
    space: "scalar",
    gens: 16,
    pop: 8,
    seeds: 250,
    elite: 3,
    sigma: 0.25,
    sigmaFinal: 0.1,
    marginWeight: 0,
    baseKtune: "",
    params: PARAMS,
    curriculum: false,
    acceptCi: false,
    jobs: 8,
    out: "runs/tune",
    seed: 1,
    resume: false,
  };
  // Which of the space-sensitive defaults the user actually typed. Everything
  // else takes its default from `--space` below, so `--space=consumer` alone is
  // the documented 16×8×1000 run and not a 250-seed one.
  const set = new Set<string>();
  const num = (s: string, name: string) => {
    const v = Number(s);
    if (!Number.isFinite(v)) {
      console.error(`${name} は数値: ${s}`);
      Deno.exit(2);
    }
    return v;
  };
  for (const arg of argv) {
    if (arg === "--help" || arg === "-h") {
      console.log(USAGE);
      Deno.exit(0);
    } else if (arg.startsWith("--space=")) {
      const v = arg.slice(8);
      if (v !== "scalar" && v !== "consumer") {
        console.error(`--space は scalar か consumer: ${v}`);
        Deno.exit(2);
      }
      o.space = v;
    } else if (arg.startsWith("--gens=")) o.gens = num(arg.slice(7), "--gens");
    else if (arg.startsWith("--pop=")) o.pop = num(arg.slice(6), "--pop");
    else if (arg.startsWith("--seeds=")) {
      o.seeds = num(arg.slice(8), "--seeds");
      set.add("seeds");
    } else if (arg.startsWith("--elite=")) o.elite = num(arg.slice(8), "--elite");
    else if (arg.startsWith("--sigma-final=")) {
      o.sigmaFinal = num(arg.slice(14), "--sigma-final");
      set.add("sigmaFinal");
    } else if (arg.startsWith("--sigma=")) {
      o.sigma = num(arg.slice(8), "--sigma");
      set.add("sigma");
    } else if (arg.startsWith("--margin-weight=")) {
      o.marginWeight = num(arg.slice(16), "--margin-weight");
      set.add("marginWeight");
    } else if (arg.startsWith("--base-ktune=")) {
      o.baseKtune = arg.slice(13);
      set.add("baseKtune");
    } else if (arg.startsWith("--params=")) {
      try {
        o.params = selectParams(arg.slice(9));
      } catch (e) {
        console.error(e instanceof Error ? e.message : String(e));
        Deno.exit(2);
      }
      set.add("params");
    } else if (arg === "--curriculum") o.curriculum = true;
    else if (arg === "--accept-ci") o.acceptCi = true;
    else if (arg.startsWith("--jobs=")) o.jobs = num(arg.slice(7), "--jobs");
    else if (arg.startsWith("--out=")) {
      o.out = arg.slice(6);
      set.add("out");
    } else if (arg.startsWith("--seed=")) o.seed = num(arg.slice(7), "--seed");
    else if (arg === "--resume") o.resume = true;
    else {
      console.error(`不明なオプション: ${arg}\n\n${USAGE}`);
      Deno.exit(2);
    }
  }
  if (o.space === "consumer") {
    // Native `paired` does ~25-40 半荘/s, so 1000 seeds is ~70s per candidate —
    // and a 250-seed CI is far too loose to rank eight of them against each other.
    if (!set.has("seeds")) o.seeds = 1000;
    // σ here multiplies a per-curve unit, so its scale has nothing to do with
    // the scalar space's log-space σ.
    if (!set.has("sigma")) o.sigma = 1.0;
    if (!set.has("sigmaFinal")) o.sigmaFinal = 0.4;
    if (!set.has("marginWeight")) o.marginWeight = 0.005;
    if (!set.has("baseKtune")) o.baseKtune = `${ROOT}runs/tune/best.json`;
    if (!set.has("out")) o.out = "runs/tune-consumer";
    // The consumer vector is 68 θ; the scalar table has nothing to do with it.
    if (set.has("params")) {
      console.error("--params は --space=scalar 専用です");
      Deno.exit(2);
    }
  } else if (o.curriculum) {
    console.error("--curriculum は --space=consumer 専用です");
    Deno.exit(2);
  }
  if (o.pop < 1) o.pop = 1;
  if (o.elite < 1) o.elite = 1;
  if (o.elite > o.pop) o.elite = o.pop;
  if (o.jobs < 1) o.jobs = 1;
  // Child processes run with cwd = repo root; a relative --out would then mean
  // two different directories to parent and child.
  if (!o.out.startsWith("/")) o.out = `${Deno.cwd()}/${o.out}`;
  if (o.baseKtune && !o.baseKtune.startsWith("/")) o.baseKtune = `${Deno.cwd()}/${o.baseKtune}`;
  return o;
}

export interface State {
  /** Absent in files written before M9c — those are scalar runs. */
  space?: SearchSpace;
  gen: number;
  mean: number[];
  sigma: number;
  /** The search seed. With `gen`, this IS the RNG state: each generation draws
   *  from `sfc32(seed).fork(gen)`, so resuming needs no serialized stream. */
  seed: number;
  /**
   * The space's coordinates. In the scalar space this IS the `--params` subset,
   * in the table's order, so a resume that names a different subset is refused.
   */
  paths: string[];
  /** What the candidates sit on top of, and whether ε was annealed (consumer). */
  baseKtune?: string;
  curriculum?: boolean;
  /** The control arm the last generation was measured against. */
  incumbentPath?: string;
  /** The accumulated σ shrink from rejected proposals. */
  sigmaScale?: number;
}

/**
 * The identity of the space's coordinates, recorded so a resume can check it —
 * which in the scalar space also means a resume under a different `--params`
 * fails loudly instead of reassigning old numbers to new parameters.
 */
export function spacePaths(opt: Options): string[] {
  return opt.space === "consumer" ? ALL_SPECS.map((s) => s.key) : opt.params.map((p) => p.path);
}

function fmt(x: number): string {
  return Math.abs(x) >= 100 ? x.toFixed(1) : Math.abs(x) >= 1 ? x.toFixed(3) : x.toFixed(5);
}

/** `padEnd` counts code units; the danger levels are CJK and print two columns
 *  wide, so the report would step out of line on exactly the three rows a reader
 *  most wants to compare. */
function padName(s: string, w: number): string {
  let width = 0;
  for (const ch of s) width += /[ᄀ-ᅟ⺀-꓏가-힣豈-﫿︰-﹯＀-｠￠-￦]/.test(ch) ? 2 : 1;
  return s + " ".repeat(Math.max(1, w - width));
}

const signed = (x: number, d = 4) => (x >= 0 ? "+" : "") + x.toFixed(d);

async function main(): Promise<void> {
  const opt = parseOpts(Deno.args);
  await Deno.mkdir(opt.out, { recursive: true });
  const historyPath = `${opt.out}/history.jsonl`;
  const statePath = `${opt.out}/state.json`;
  const bestPath = `${opt.out}/best.json`;
  const consumer = opt.space === "consumer";

  let init: number[];
  if (consumer) {
    try {
      init = consumerDefaults(opt.baseKtune);
    } catch (e) {
      console.error(
        `--base-ktune が読めません: ${opt.baseKtune}\n${e instanceof Error ? e.message : e}`,
      );
      Deno.exit(2);
    }
  } else {
    init = defaults(opt.params);
    // The base partial is merged into every candidate file (see the header), so
    // it is read ONCE, here, and an unreadable one is fatal before any hanchan
    // is played rather than eight subprocess failures later.
    if (opt.baseKtune) {
      try {
        const json = JSON.parse(Deno.readTextFileSync(opt.baseKtune)) as unknown;
        if (!isPlainObject(json)) throw new Error("オブジェクトではありません");
        opt.baseObj = json;
      } catch (e) {
        console.error(
          `--base-ktune が読めません: ${opt.baseKtune}\n${e instanceof Error ? e.message : e}`,
        );
        Deno.exit(2);
      }
    }
  }
  const repair = (v: readonly number[]) =>
    consumer ? consumerToVec(vecToConsumer(v)) : clampVec(v, opt.params);

  let mean = repair(init);
  let startGen = 0;
  // Multiplies the annealed σ. Only the acceptance test moves it, and only
  // downward; it is persisted so a resumed run does not re-widen its step.
  let sigmaScale = 1;
  if (opt.resume) {
    try {
      const st = JSON.parse(await Deno.readTextFile(statePath)) as State;
      // A state file with no `space` predates M9c and is a scalar run by
      // definition; resuming one into the consumer space would reinterpret
      // twelve prices as twelve θ.
      const stSpace: SearchSpace = st.space ?? "scalar";
      if (stSpace !== opt.space) {
        console.error(
          `state.json は ${stSpace} 空間のものです (--space=${opt.space}): ${statePath}`,
        );
        Deno.exit(2);
      }
      // Coordinates are recorded so a resume across a table edit fails loudly
      // rather than silently reassigning old numbers to new parameters.
      const want = spacePaths(opt);
      const same = st.paths?.length === want.length && st.paths.every((p, i) => p === want[i]);
      if (!same) {
        console.error(`state.json のパラメータ表が現在の表と一致しません: ${statePath}`);
        Deno.exit(2);
      }
      mean = repair(st.mean);
      startGen = st.gen;
      if (typeof st.sigmaScale === "number" && st.sigmaScale > 0) sigmaScale = st.sigmaScale;
      console.log(`再開: 世代 ${startGen} から (${statePath})`);
    } catch (e) {
      console.error(`--resume できません: ${e instanceof Error ? e.message : e}`);
      Deno.exit(2);
    }
  }

  console.log(
    `感性最適化 [${opt.space}]: 世代 ${opt.gens}  候補 ${opt.pop}/世代  ペア ${opt.seeds}  ` +
      `エリート ${opt.elite}  σ ${opt.sigma}→${opt.sigmaFinal}  探索シード ${opt.seed}`,
  );
  console.log(
    `出力 ${opt.out}  (評価 ${opt.pop * opt.seeds * 2} 半荘/世代、同時 ${opt.jobs} プロセス)` +
      (opt.baseKtune ? `\n下敷き ${opt.baseKtune}` : "") +
      (consumer ? `  素点係数 ${opt.marginWeight}/1000点` : "") +
      (!consumer && opt.params.length !== PARAMS.length
        ? `\n探索するパラメータ ${opt.params.length}/${PARAMS.length}: ${
          opt.params.map((p) => p.path).join(", ")
        }`
        : "") +
      (opt.curriculum ? "  カリキュラム ε 0.35→1.0 ([0.5,0.75] に滞在)" : ""),
  );

  for (let gen = startGen; gen < opt.gens; gen++) {
    // σ is the schedule's value times whatever the acceptance test has shrunk
    // it to: a rejected proposal means the step was too big for the signal, and
    // the next generation asks a smaller question.
    const sigma = sigmaAt(gen, opt.gens, opt.sigma, opt.sigmaFinal) * sigmaScale;
    const lane = laneOf(gen, opt.space);
    const eps = opt.curriculum ? epsAt(gen, opt.gens) : undefined;
    // One stream per generation, derived from the run seed: resume replays it.
    const rng = sfc32(opt.seed).fork(gen);
    const vecs: number[][] = [repair(mean)];
    for (let i = 1; i < opt.pop; i++) {
      vecs.push(
        consumer ? perturbConsumer(mean, sigma, rng) : perturb(mean, sigma, rng, opt.params),
      );
    }

    // The incumbent, on disk, as the control arm for every candidate this
    // generation — in BOTH spaces (M10d). Written per generation rather than
    // overwritten, so the history says exactly what each row was measured
    // against.
    const incFile = `${opt.out}/incumbent-g${gen}.json`;
    await writeCandidate(mean, incFile, opt);

    console.log("");
    console.log(
      `== 世代 ${gen}  σ ${sigma.toFixed(4)}  牌山レーン ${lane}..${lane + opt.seeds - 1}` +
        (eps !== undefined ? `  ε ${eps.toFixed(3)}` : "") + " ==",
    );
    // The child's argv, verbatim. This is the only place a candidate differs
    // from the control, so a saved log that does not show it cannot be audited.
    console.log(
      `   子: deno ${pairedArgs(opt, "<候補>", lane, eps, incFile).slice(5).join(" ")}`,
    );
    const t0 = performance.now();
    // Every candidate in a generation plays the SAME seeds (common random
    // numbers), so the comparison between them is paired too — running them
    // concurrently is what makes that affordable.
    const jobs: EvalSpec[] = vecs.map((v, i) => ({
      vec: v,
      idx: i,
      lane,
      eps,
      file: `${opt.out}/cand-g${gen}-${i}.json`,
      incumbent: incFile,
    }));
    // Absolute anchoring: every fourth generation the incumbent is also measured
    // the OLD way, against plain hhhh, on this generation's lane. Selection never
    // reads this row — it exists so the log answers "and where are we actually,
    // in the units the acceptance bar is written in".
    const anchoring = gen % 4 === 0;
    if (anchoring) {
      jobs.push({
        vec: mean,
        idx: -1,
        lane,
        eps,
        file: `${opt.out}/anchor-g${gen}.json`,
      });
    }
    const runs = await mapLimit(jobs, opt.jobs, (j) => evaluate(j, opt));
    const cands = runs.slice(0, vecs.length);
    const anchor = anchoring ? runs[vecs.length] : null;
    const secs = (performance.now() - t0) / 1000;

    // The control arm is identical for every candidate in a generation (same
    // seeds, same file), so any one of them supplies the reference violation
    // count — the INCUMBENT's, now, not the baseline's. Both arms are k seats,
    // so this compares like with like.
    const genVioB = vioTotal(cands.find((c) => c.stats)?.stats?.vioB0);

    for (const c of cands) {
      const tag = c.idx === 0 ? "現行" : `#${c.idx}`;
      if (!c.stats) {
        console.log(`  ${tag}  失敗: ${c.err}`);
        continue;
      }
      const vA = vioTotal(c.stats.vioA0);
      // A DIAGNOSTIC, not a rejection. The compliance filter drops ledgered
      // actions before scoring, so no weight vector is supposed to be able to
      // buy violations at all — a flag here means some term outbid the filter's
      // fallthrough, and that is worth reading, not worth hiding.
      const flagged = vA > genVioB * 1.2 && vA > 0;
      const d = c.stats.dRankDojo;
      const fit = fitnessOf(c.stats, opt.marginWeight);
      console.log(
        `  ${tag}  道場順位差 ${signed(d.mean)} ` +
          `±${d.ci.toFixed(4)}  SD ${d.sd.toFixed(3)}  素点差 ${
            c.stats.dScore.mean.toFixed(0).padStart(7)
          }   ` +
          `適合度 ${signed(fit)}   違反 A${vA}/B${genVioB}${flagged ? "  ⚠ 違反増" : ""}`,
      );
      const rec = {
        space: opt.space,
        gen,
        idx: c.idx,
        lane,
        sigma,
        eps,
        /** "" for the absolute instrument, else the incumbent this row is relative to. */
        against: incFile,
        ...vecRecord(c.vec, opt),
        fitness: fit,
        dRankDojo: c.stats.dRankDojo,
        dRank: c.stats.dRank,
        dScore: c.stats.dScore,
        vioA0Total: vA,
        vioB0Total: vioTotal(c.stats.vioB0),
        vioA0: c.stats.vioA0,
        flagged,
      };
      await Deno.writeTextFile(historyPath, JSON.stringify(rec) + "\n", { append: true });
    }

    // ---- instrument telemetry ---------------------------------------------
    // Candidate 0 IS the incumbent, so its measured difference is provably zero
    // and its SD carries no information; the instrument's resolution is the SD
    // over the perturbations. Printed every generation because the redesign this
    // number justifies is only as good as the number.
    const sds = cands.filter((c) => c.stats && c.idx > 0).map((c) => c.stats!.dRankDojo.sd);
    const sdRel = sds.length > 0 ? sds.reduce((a, b) => a + b, 0) / sds.length : NaN;
    if (cands[0]?.stats && cands[0].stats.dRankDojo.sd !== 0) {
      console.log(
        `  ⚠ 現行×現行の差が0ではありません (SD ${cands[0].stats.dRankDojo.sd}) — 器が非対称です`,
      );
    }
    if (anchor?.stats) {
      const a = anchor.stats.dRankDojo;
      console.log(
        `  絶対 (現行 対 hhhh、選抜には未使用)  道場順位差 ${signed(a.mean)} ±${
          a.ci.toFixed(4)
        }  SD ${a.sd.toFixed(3)}  席0平均順位 ${anchor.stats.rankA.toFixed(3)}`,
      );
      await Deno.writeTextFile(
        historyPath,
        JSON.stringify({
          space: opt.space,
          gen,
          idx: "absolute",
          lane,
          eps,
          against: "",
          fitness: fitnessOf(anchor.stats, opt.marginWeight),
          dRankDojo: anchor.stats.dRankDojo,
          dRank: anchor.stats.dRank,
          dScore: anchor.stats.dScore,
          rankA: anchor.stats.rankA,
          vioA0Total: vioTotal(anchor.stats.vioA0),
          vioB0Total: vioTotal(anchor.stats.vioB0),
          vioA0: anchor.stats.vioA0,
          flagged: false,
        }) + "\n",
        { append: true },
      );
      console.log(
        `  器: 候補×現行 SD ${sdRel.toFixed(3)} 対 絶対 SD ${a.sd.toFixed(3)} → 分散削減 ×${
          (Math.pow(a.sd / sdRel, 2)).toFixed(1)
        } (標準偏差で ×${(a.sd / sdRel).toFixed(2)})`,
      );
    } else {
      console.log(
        `  器: 候補×現行 SD ${sdRel.toFixed(3)}  (95%CI ≒ ±${
          (1.96 * sdRel / Math.sqrt(opt.seeds)).toFixed(4)
        })`,
      );
    }

    const elite = select(cands, opt.elite, opt.marginWeight);
    if (elite.length === 0) {
      console.error("この世代は全候補が失敗しました。中断します。");
      Deno.exit(1);
    }
    const proposed = repair(
      consumer ? meanVec(elite.map((c) => c.vec)) : logMean(elite.map((c) => c.vec)),
    );
    const best = elite[0].stats!.dRankDojo;
    console.log(
      `  → エリート ${elite.map((c) => (c.idx === 0 ? "現行" : `#${c.idx}`)).join(",")}  ` +
        `最良 ${signed(best.mean)}  ${secs.toFixed(1)}s`,
    );

    // ---- acceptance --------------------------------------------------------
    // The elites won on THIS generation's walls; the recombination of them is a
    // different vector that won nothing yet. Confirming it against the incumbent
    // on walls neither of them has seen is what stops the mean from drifting on
    // eight lane-lucky draws — the failure that produced the −0.007 champion.
    {
      const accLane = acceptLaneOf(gen);
      const acc = await evaluate({
        vec: proposed,
        idx: 0,
        lane: accLane,
        eps,
        file: `${opt.out}/proposed-g${gen}.json`,
        incumbent: incFile,
      }, opt);
      const d = acc.stats?.dRankDojo;
      // The point estimate by default; `--accept-ci` demands the whole interval
      // be on the right side of zero, which is a much slower but stricter walk.
      const ok = d !== undefined && (opt.acceptCi ? d.mean + d.ci < 0 : d.mean <= 0);
      if (!acc.stats) {
        console.log(`  受理判定 (レーン ${accLane}): 失敗 — 棄却  ${acc.err}`);
      } else {
        console.log(
          `  受理判定 (レーン ${accLane}): 提案 対 現行 ${signed(d!.mean)} ±${
            d!.ci.toFixed(4)
          }  SD ${d!.sd.toFixed(3)} → ${ok ? "採用" : "棄却"}`,
        );
      }
      await Deno.writeTextFile(
        historyPath,
        JSON.stringify({
          space: opt.space,
          gen,
          idx: "accept",
          lane: accLane,
          eps,
          against: incFile,
          accepted: ok,
          ...vecRecord(proposed, opt),
          ...(acc.stats
            ? {
              fitness: fitnessOf(acc.stats, opt.marginWeight),
              dRankDojo: acc.stats.dRankDojo,
              dRank: acc.stats.dRank,
              dScore: acc.stats.dScore,
              vioA0Total: vioTotal(acc.stats.vioA0),
              vioB0Total: vioTotal(acc.stats.vioB0),
              vioA0: acc.stats.vioA0,
            }
            : { err: acc.err }),
          flagged: false,
        }) + "\n",
        { append: true },
      );
      if (ok) {
        mean = proposed;
      } else {
        sigmaScale *= REJECT_SHRINK;
        console.log(
          `     現行を据え置き、次世代の σ を ×${REJECT_SHRINK} (累計 ×${sigmaScale.toFixed(3)})`,
        );
      }
    }

    // State first, then the vector: a run killed between the two resumes into a
    // generation whose best.json is one behind, never the other way round.
    const state: State = {
      space: opt.space,
      gen: gen + 1,
      mean,
      sigma,
      seed: opt.seed,
      paths: spacePaths(opt),
      baseKtune: opt.baseKtune,
      curriculum: opt.curriculum,
      incumbentPath: incFile,
      sigmaScale,
    };
    await Deno.writeTextFile(statePath, JSON.stringify(state, null, 2) + "\n");
    // Under a curriculum the running mean has been graded WITH oracle help, so
    // it is a checkpoint and not yet a champion: it goes to mean.json, and
    // best.json is written only after the ε=1 re-score below.
    const running = opt.curriculum ? `${opt.out}/mean.json` : bestPath;
    await writeCandidate(mean, running, opt);
  }

  // ---- the curriculum's final exam ----------------------------------------
  let champion = mean;
  if (opt.curriculum) {
    const lane = acceptLaneOf(opt.gens);
    console.log("");
    console.log(`== ε=1 最終試験 (オラクル無し、素の計算席)  牌山レーン ${lane} ==`);
    // ONE measurement, on the same instrument the search used: the learned
    // curves as arm A, the INIT curves as the control arm, no oracle anywhere.
    // A consumer that learned to lean on the oracle cannot beat where it started
    // once the help is gone — and in that case it is the init that gets written.
    const initFile = `${opt.out}/final-init.json`;
    await writeCandidate(init, initFile, opt);
    console.log(
      `   子: deno ${pairedArgs(opt, "<候補>", lane, undefined, initFile).slice(5).join(" ")}`,
    );
    const fin = await evaluate({
      vec: mean,
      idx: 0,
      lane,
      file: `${opt.out}/final-learned.json`,
      incumbent: initFile,
    }, opt);
    if (!fin.stats) {
      console.log(`  ⚠ 再採点に失敗しました (${fin.err})。best.json には初期値を書きます。`);
      champion = init;
    } else {
      const d = fin.stats.dRankDojo;
      console.log(
        `  学習後 対 初期値  道場順位差 ${signed(d.mean)} ±${d.ci.toFixed(4)}  SD ${
          d.sd.toFixed(3)
        }   素点差 ${fin.stats.dScore.mean.toFixed(0).padStart(7)}`,
      );
      await Deno.writeTextFile(
        historyPath,
        JSON.stringify({
          space: opt.space,
          gen: opt.gens,
          idx: "final",
          lane,
          eps: 1,
          against: initFile,
          curves: vecToConsumer(mean).curves,
          fitness: fitnessOf(fin.stats, opt.marginWeight),
          dRankDojo: fin.stats.dRankDojo,
          dRank: fin.stats.dRank,
          dScore: fin.stats.dScore,
          vioA0Total: vioTotal(fin.stats.vioA0),
          vioB0Total: vioTotal(fin.stats.vioB0),
          vioA0: fin.stats.vioA0,
          flagged: false,
        }) + "\n",
        { append: true },
      );
      if (d.mean > 0) {
        console.log(
          "  ⚠ オラクル抜きでは初期値に及びません。カリキュラムに寄りかかった学習です。\n" +
            "     best.json には初期値を書きます (助けが要る champion は残しません)。",
        );
        champion = init;
      } else {
        console.log("  → ε=1 でも初期値以上。best.json に学習後の曲線を書きます。");
      }
    }
  }
  await writeCandidate(champion, bestPath, opt);

  // ---- final report -------------------------------------------------------
  console.log("");
  console.log("== champion ==");
  if (consumer) {
    // The interpretability deliverable: how far each curve actually moved, in
    // SCORE units at its own knots. A curve with max|Δ| ≈ 0 was not used by the
    // search and can be read as "the hand-written shape was already right".
    const p0 = vecToConsumer(init);
    const p1 = vecToConsumer(champion);
    console.log(`${padName("曲線", 16)}  歩幅      最大|Δ値|   節点ごとの値 (初期 → 学習後)`);
    ALL_SPECS.forEach((spec) => {
      const a = curveValues(p0.curves[spec.key]);
      const b = curveValues(p1.curves[spec.key]);
      const dmax = Math.max(...a.map((y, i) => Math.abs(b[i] - y)));
      const pts = a.map((y, i) => `${fmt(y)}→${fmt(b[i])}`).join("  ");
      console.log(
        `${padName(spec.key, 16)}${fmt(CURVE_STEP[spec.key]).padStart(8)}` +
          `${fmt(dmax).padStart(12)}   ${pts}`,
      );
    });
    console.log("");
    console.log(`最良曲線: ${bestPath}`);
    console.log("1000シードで検証 (探索に使っていないレーン):");
    console.log(
      "  deno run --allow-read --allow-write --allow-ffi --allow-env=MJGAME_NATIVE \\\n" +
        `    src/main.ts paired --games=1000 --seed=${VALIDATE_SEED} --seats=khhh \\\n` +
        `    --ktune=${opt.baseKtune} --consumer=${bestPath}`,
    );
    console.log("同じレーンの下敷きのみ (比較対象):");
    console.log(
      "  deno run --allow-read --allow-write --allow-ffi --allow-env=MJGAME_NATIVE \\\n" +
        `    src/main.ts paired --games=1000 --seed=${VALIDATE_SEED} --seats=khhh \\\n` +
        `    --ktune=${opt.baseKtune}`,
    );
    return;
  }

  // Hand-spaced: the columns below are display-width aligned, which `padStart`
  // cannot do for a CJK header.
  console.log(`${padName("パラメータ", 30)}      既定      調整後     倍率`);
  opt.params.forEach((p, i) => {
    const r = champion[i] / p.def;
    console.log(
      `${padName(p.path, 30)}${fmt(p.def).padStart(10)}${fmt(champion[i]).padStart(12)}` +
        `${("×" + r.toFixed(2)).padStart(9)}`,
    );
  });
  console.log(
    `${padName("heuristic.shanten (基準単位・固定)", 30)}${fmt(1000).padStart(10)}` +
      `${fmt(1000).padStart(12)}${"×1.00".padStart(9)}`,
  );
  // Whatever `--params` left out is not in the table above and was not searched;
  // it sits at its default (or at the base's value) in every file this run wrote.
  for (const p of PARAMS) {
    if (opt.params.includes(p)) continue;
    console.log(
      `${padName(p.path + " (探索外)", 30)}${fmt(p.def).padStart(10)}` +
        `${"—".padStart(12)}${"—".padStart(9)}`,
    );
  }
  if (opt.baseKtune) console.log(`下敷き (--base-ktune): ${opt.baseKtune}`);
  console.log("");
  console.log(`最良ベクトル: ${bestPath}`);
  console.log("1000シードで検証 (探索に使っていないレーン):");
  console.log(
    "  deno run --allow-read --allow-write --allow-ffi --allow-env=MJGAME_NATIVE \\\n" +
      `    src/main.ts paired --games=1000 --seed=${VALIDATE_SEED} --seats=khhh \\\n` +
      `    --ktune=${bestPath}`,
  );
  console.log("同じレーンの無調整 k席 (比較対象):");
  console.log(
    "  deno run --allow-read --allow-write --allow-ffi --allow-env=MJGAME_NATIVE \\\n" +
      `    src/main.ts paired --games=1000 --seed=${VALIDATE_SEED} --seats=khhh`,
  );
}

if (import.meta.main) await main();
