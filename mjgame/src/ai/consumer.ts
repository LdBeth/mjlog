// 感性の消費 — the learned-but-interpretable consumer of the evidence vector.
//
// M9's thesis, second half. `evidence.ts` keeps the 計算 exactly as it was; this
// file replaces the hand-written arithmetic that CONSUMES it with a small map
// that can be fitted, and can still be read off a sheet of paper:
//
//     score = M_atk(context) · Σᵢ fᵢ(xᵢ)  −  M_def(context) · g(danger)  +  Σⱼ hⱼ(bonusⱼ)
//
// Every f, g, h and every factor of M is ONE piecewise-linear curve over ONE
// named feature, with four knots at fixed positions. Seventeen curves, four
// parameters each: sixty-eight numbers, each of which can be plotted against its own
// axis and argued about in Japanese.
//
// MONOTONE BY CONSTRUCTION. A curve stores raw θ and reads its knot VALUES as
// y₀ = θ₀, yᵢ = yᵢ₋₁ + |θᵢ|. The absolute value is the whole trick: no
// perturbation of θ, from a gradient step or a random search, can make a curve
// non-monotone, so the fit never has to be constrained and never has to be
// checked. Curves are NONDECREASING, always; the direction a feature pushes the
// score is fixed STRUCTURAL metadata (`sign`), never a fitted quantity. That is
// what keeps the thing interpretable under training: "more ukeire is never
// worse" and "more danger is never cheaper" are properties of the file, not
// observations about the current weights.
//
// EXTRAPOLATION. Features whose init term is an unbounded linear function
// (shanten, ukeire, dora, risk, the bonus hooks, and the two context scales)
// carry `mode: "linear"`: outside the knot range the end segment's slope
// continues, so an init built from a linear weight is EXACTLY that linear
// function on the whole real line and no clamp can truncate it. Features that
// are bounded by construction (the danger ordinal, the genbutsu flag, junme,
// pressure) carry `mode: "clamp"` and hold their end values. This is the choice
// the spec left open, made in favour of exact init-equivalence.
//
// WHAT STAYS OUTSIDE. `dojoCost` (the compliance fallthrough pricing), the
// compliance FILTER itself, and the riichi decision are not here and must not
// be: the first is a price on a 禁じ手 and is deliberately immune to every scale
// in this file, and the second is a veto no score may reach around. The consumer
// replaces the score CORE — `ctx.eff·eff − ctx.def·risk + drawBonus − keepBonus`
// — and nothing else.
//
// DEVIATION, deliberate and reported. The two hook bonuses (`drawBonus`,
// `keepBonus`) form a THIRD, unscaled group rather than living inside
// `M_atk · Σ f`. That is not a liberty: the base policy adds them outside
// `ctx.eff` and `ctx.def` on purpose ("they are already in score units and
// already know whether the policy is folding" — heuristic.ts), so folding them
// into M_atk would multiply them by 0.05 while folding and init-equivalence
// would fail on exactly the decisions the fold multiplier exists for.

import type { HeuristicWeights } from "./heuristic.ts";
import type { CandidateEvidence, ContextEvidence, EvidenceVector } from "./evidence.ts";

// ---------------------------------------------------------------------------
// the curve
// ---------------------------------------------------------------------------

/** Raw parameters of one curve: θ₀ is a level, θ₁..θ₃ are |increments|. */
export type CurveParams = [number, number, number, number];

/** Four knot positions. Fixed metadata — never fitted. */
export type Knots = readonly [number, number, number, number];

export type CurveMode = "clamp" | "linear";

/** Every curve in the model, in the order the score visits them. */
export type CurveKey =
  // per-candidate, attack side (scaled by M_atk)
  | "shanten"
  | "ukeire"
  | "ukeireType"
  | "dora"
  | "yakuhaiPair"
  | "isolatedHonor"
  // per-candidate, defence side (scaled by M_def)
  | "risk"
  | "dangerLevel"
  | "safe"
  // per-candidate, unscaled
  | "drawBonus"
  | "keepBonus"
  // per-decision context, the two multipliers
  | "atkEff"
  | "atkPressure"
  | "atkJunme"
  | "defScale"
  | "defPressure"
  | "defJunme";

export interface CurveSpec {
  key: CurveKey;
  /** Which named evidence field this curve reads. */
  field: keyof CandidateEvidence | keyof ContextEvidence;
  knots: Knots;
  /** Structural direction of the effect. Never fitted. */
  sign: 1 | -1;
  mode: CurveMode;
  /** One line, for the plot's caption. */
  about: string;
}

/** Knot VALUES from raw θ. Nondecreasing whatever θ is — that is the point. */
export function curveValues(theta: CurveParams): [number, number, number, number] {
  const y0 = theta[0];
  const y1 = y0 + Math.abs(theta[1]);
  const y2 = y1 + Math.abs(theta[2]);
  const y3 = y2 + Math.abs(theta[3]);
  return [y0, y1, y2, y3];
}

/**
 * The curve at `x`. Piecewise linear between the knots; at a knot, the knot's
 * own value EXACTLY (the equality short-circuits exist for that: recovering
 * y₁ as y₀ + slope·(k₁−k₀) is only exact when the division and the
 * multiplication round-trip, and they do not always).
 */
export function evalCurve(spec: CurveSpec, theta: CurveParams, x: number): number {
  const y = curveValues(theta);
  const k = spec.knots;
  const slope = (i: number) => (y[i + 1] - y[i]) / (k[i + 1] - k[i]);
  if (x === k[0]) return y[0];
  if (x === k[1]) return y[1];
  if (x === k[2]) return y[2];
  if (x === k[3]) return y[3];
  if (x < k[0]) return spec.mode === "linear" ? y[0] + slope(0) * (x - k[0]) : y[0];
  if (x > k[3]) return spec.mode === "linear" ? y[3] + slope(2) * (x - k[3]) : y[3];
  const i = x < k[1] ? 0 : x < k[2] ? 1 : 2;
  return y[i] + slope(i) * (x - k[i]);
}

/** θ for the straight line y = slope·x through the knots. `slope` must be ≥ 0. */
export function linearTheta(knots: Knots, slope: number): CurveParams {
  if (!(slope >= 0) || !Number.isFinite(slope)) {
    throw new RangeError(
      `consumer: 単調曲線は減少できません (slope=${slope})。` +
        `負の重みは構造的な符号 (sign) で表現してください。`,
    );
  }
  return [
    slope * knots[0],
    slope * (knots[1] - knots[0]),
    slope * (knots[2] - knots[1]),
    slope * (knots[3] - knots[2]),
  ];
}

/** θ for the constant c. */
export function constTheta(c: number): CurveParams {
  return [c, 0, 0, 0];
}

// ---------------------------------------------------------------------------
// the model
// ---------------------------------------------------------------------------

/**
 * The attack terms, grouped the way `HeuristicPolicy.scoreDiscard` adds them:
 * the two ukeire terms are summed together before joining the running total,
 * because that is what `eff += live * w.ukeire + types.length * w.ukeireType`
 * does, and float addition is not associative. Init-equivalence is asserted to
 * the last bit, so the ORDER of the additions is part of the specification.
 */
export const ATK_GROUPS: readonly (readonly CurveSpec[])[] = [
  [{
    key: "shanten",
    field: "shantenAfter",
    knots: [-1, 1, 3, 6],
    sign: -1,
    mode: "linear",
    about: "向聴数 — 残る手の遠さ。一歩が他の全項を支配する",
  }],
  [
    {
      key: "ukeire",
      field: "ukeireLive",
      knots: [0, 8, 16, 32],
      sign: 1,
      mode: "linear",
      about: "受け入れ枚数 (山に残る有効牌)",
    },
    {
      key: "ukeireType",
      field: "ukeireTypeCount",
      knots: [0, 4, 8, 16],
      sign: 1,
      mode: "linear",
      about: "受け入れ種類数 — 広さ",
    },
  ],
  [{
    key: "dora",
    field: "doraKept",
    knots: [0, 2, 4, 8],
    sign: 1,
    mode: "linear",
    about: "手中に残るドラ (赤含む)",
  }],
  [{
    key: "yakuhaiPair",
    field: "yakuhaiPairs",
    knots: [0, 1, 2, 4],
    sign: 1,
    mode: "linear",
    about: "門前で持つ役牌対子 — 役の種であり安全牌でもある",
  }],
  [{
    key: "isolatedHonor",
    field: "isolatedHonorLate",
    knots: [0, 12, 24, 48],
    sign: -1,
    mode: "linear",
    about: "孤立字牌 × min(巡目,12) — 遅いほど重い",
  }],
];

export const DEF_SPECS: readonly CurveSpec[] = [
  {
    key: "risk",
    field: "risk",
    knots: [0, 100, 200, 400],
    sign: 1,
    mode: "linear",
    about: "riskOf — 放銃の値段 (規則の梯子、または P×失点)",
  },
  {
    key: "dangerLevel",
    field: "dangerLevel",
    knots: [0, 1, 2, 3],
    sign: 1,
    mode: "clamp",
    about: "危険度の段位 0..3 — riskOf を分解した規則側の読み",
  },
  {
    key: "safe",
    field: "explicitSafe",
    knots: [0, 1, 2, 3],
    sign: -1,
    mode: "clamp",
    about: "現物の明示 (安全) — 推定ではなく証明",
  },
];

export const FREE_SPECS: readonly CurveSpec[] = [
  {
    key: "drawBonus",
    field: "drawBonus",
    knots: [0, 200, 400, 1000],
    sign: 1,
    mode: "linear",
    about: "drawBonus — 一巡先読みの加点 (既に点数単位)",
  },
  {
    key: "keepBonus",
    field: "keepBonus",
    knots: [0, 1000, 2500, 5000],
    sign: -1,
    mode: "linear",
    about: "keepBonus — 温存の減点 (立案の keep 集合を含む)",
  },
];

export const CTX_ATK_SPECS: readonly CurveSpec[] = [
  {
    key: "atkEff",
    field: "eff",
    knots: [0, 0.25, 0.5, 1],
    sign: 1,
    mode: "linear",
    about: "M_atk: ctx.eff — ベタ降り中の効率倍率をそのまま含む",
  },
  {
    key: "atkPressure",
    field: "pressure",
    knots: [0, 1, 2, 3],
    sign: 1,
    mode: "clamp",
    about: "M_atk: 場の圧 — 初期値は定数1",
  },
  {
    key: "atkJunme",
    field: "junme",
    knots: [0, 6, 12, 18],
    sign: 1,
    mode: "clamp",
    about: "M_atk: 巡目 — 初期値は定数1",
  },
];

export const CTX_DEF_SPECS: readonly CurveSpec[] = [
  {
    key: "defScale",
    field: "def",
    knots: [0, 1, 2, 4],
    sign: 1,
    mode: "linear",
    about: "M_def: ctx.def — ベタ降り倍率 × 順位効用の risk",
  },
  {
    key: "defPressure",
    field: "pressure",
    knots: [0, 1, 2, 3],
    sign: 1,
    mode: "clamp",
    about: "M_def: 場の圧 — 初期値は定数1",
  },
  {
    key: "defJunme",
    field: "junme",
    knots: [0, 6, 12, 18],
    sign: 1,
    mode: "clamp",
    about: "M_def: 巡目 — 初期値は定数1",
  },
];

/** Every curve, in score-evaluation order. */
export const ALL_SPECS: readonly CurveSpec[] = [
  ...ATK_GROUPS.flat(),
  ...DEF_SPECS,
  ...FREE_SPECS,
  ...CTX_ATK_SPECS,
  ...CTX_DEF_SPECS,
];

export const SPEC_BY_KEY: ReadonlyMap<CurveKey, CurveSpec> = new Map(
  ALL_SPECS.map((s) => [s.key, s]),
);

/** The fitted half: one θ per curve. Metadata (knots, signs) is not stored. */
export interface ConsumerParams {
  version: 1;
  curves: Record<CurveKey, CurveParams>;
}

// ---------------------------------------------------------------------------
// scoring
// ---------------------------------------------------------------------------

function term(spec: CurveSpec, ev: Record<string, number>, p: ConsumerParams): number {
  return spec.sign * evalCurve(spec, p.curves[spec.key], ev[spec.field]);
}

function multiplier(
  specs: readonly CurveSpec[],
  ctx: ContextEvidence,
  p: ConsumerParams,
): number {
  let m = 1;
  for (const spec of specs) m *= term(spec, ctx as unknown as Record<string, number>, p);
  return m;
}

/**
 * The score CORE for one discard candidate: everything the hand-written
 * `ctx.eff · eff − ctx.def · riskOf + drawBonus − keepBonus` used to compute,
 * and nothing else. `dojoCost` is subtracted by the caller, on both paths.
 *
 * The accumulation order mirrors the arithmetic it replaces exactly — see
 * `ATK_GROUPS` — so that `initFromWeights` is bit-identical and not merely
 * close.
 */
export function scoreDiscard(ev: EvidenceVector, p: ConsumerParams): number {
  const cand = ev.candidate as unknown as Record<string, number>;
  const mAtk = multiplier(CTX_ATK_SPECS, ev.context, p);
  const mDef = multiplier(CTX_DEF_SPECS, ev.context, p);

  let atk = 0;
  for (const group of ATK_GROUPS) {
    let g = 0;
    for (const spec of group) g += term(spec, cand, p);
    atk += g;
  }

  let def = 0;
  for (const spec of DEF_SPECS) def += term(spec, cand, p);

  let s = mAtk * atk - mDef * def;
  for (const spec of FREE_SPECS) s += term(spec, cand, p);
  return s;
}

// ---------------------------------------------------------------------------
// init: the hand-written score, exactly
// ---------------------------------------------------------------------------

/**
 * The consumer that reproduces the CURRENT hand-written discard score, term for
 * term and bit for bit.
 *
 * Pass the MERGED weights — what `HeuristicPolicy` actually holds after
 * `DEFAULT_WEIGHTS` and any `--ktune` section have been folded together — since
 * those are the numbers the score is made of. `w.danger`, `w.foldEfficiency`,
 * `w.foldDanger` and the 順位効用 block are deliberately NOT copied here: they
 * live upstream of the seam, inside `riskOf` and `Ctx`, and reach the consumer
 * as the `risk`, `eff` and `def` evidence fields instead.
 */
export function initFromWeights(w: HeuristicWeights): ConsumerParams {
  const lin = (key: CurveKey, slope: number): CurveParams =>
    linearTheta(SPEC_BY_KEY.get(key)!.knots, slope);
  return {
    version: 1,
    curves: {
      shanten: lin("shanten", w.shanten),
      ukeire: lin("ukeire", w.ukeire),
      ukeireType: lin("ukeireType", w.ukeireType),
      dora: lin("dora", w.dora),
      yakuhaiPair: lin("yakuhaiPair", w.yakuhaiPair),
      isolatedHonor: lin("isolatedHonor", w.isolatedHonor),
      // The danger price itself is `riskOf`'s business, on both paths: at init
      // the consumer passes it through untouched, and the two decomposition
      // curves beside it start at exactly zero.
      risk: lin("risk", 1),
      dangerLevel: constTheta(0),
      safe: constTheta(0),
      drawBonus: lin("drawBonus", 1),
      keepBonus: lin("keepBonus", 1),
      // M_atk = ctx.eff × 1 × 1, M_def = ctx.def × 1 × 1.
      atkEff: lin("atkEff", 1),
      atkPressure: constTheta(1),
      atkJunme: constTheta(1),
      defScale: lin("defScale", 1),
      defPressure: constTheta(1),
      defJunme: constTheta(1),
    },
  };
}

// ---------------------------------------------------------------------------
// serialization
// ---------------------------------------------------------------------------

/**
 * JSON, in curve order, with the fixed metadata alongside as a COMMENT-LIKE
 * block: `knots`, `sign` and `mode` are written so the file can be read and
 * plotted on its own, and ignored on load so that editing them cannot silently
 * change what the engine computes.
 */
export function serializeConsumer(p: ConsumerParams): string {
  const curves: Record<string, unknown> = {};
  for (const spec of ALL_SPECS) {
    curves[spec.key] = {
      theta: p.curves[spec.key],
      values: curveValues(p.curves[spec.key]),
      knots: spec.knots,
      sign: spec.sign,
      mode: spec.mode,
      about: spec.about,
    };
  }
  return JSON.stringify({ version: 1, curves }, null, 2) + "\n";
}

/**
 * Read a serialized consumer back. Strict where it matters: every curve must be
 * present and must carry four finite numbers. Anything else — a stray key, a
 * `knots` array that disagrees with the code — is IGNORED, because the metadata
 * is the code's, not the file's.
 */
export function parseConsumerParams(json: unknown): ConsumerParams {
  if (typeof json !== "object" || json === null || Array.isArray(json)) {
    throw new Error("consumer: オブジェクトである必要があります");
  }
  const root = json as { version?: unknown; curves?: unknown };
  if (root.version !== 1) {
    throw new Error(`consumer: version は 1 である必要があります: ${String(root.version)}`);
  }
  if (typeof root.curves !== "object" || root.curves === null || Array.isArray(root.curves)) {
    throw new Error("consumer: curves オブジェクトがありません");
  }
  const src = root.curves as Record<string, unknown>;
  const curves = {} as Record<CurveKey, CurveParams>;
  for (const spec of ALL_SPECS) {
    const entry = src[spec.key];
    if (entry === undefined) throw new Error(`consumer: 曲線がありません: ${spec.key}`);
    const raw = Array.isArray(entry) ? entry : (entry as { theta?: unknown } | null)?.theta;
    if (!Array.isArray(raw) || raw.length !== 4) {
      throw new Error(`consumer: ${spec.key} の theta は長さ4の配列である必要があります`);
    }
    const t = raw.map((v) => {
      if (typeof v !== "number" || !Number.isFinite(v)) {
        throw new Error(`consumer: ${spec.key} の theta に数値でない値があります: ${String(v)}`);
      }
      return v;
    });
    curves[spec.key] = [t[0], t[1], t[2], t[3]];
  }
  return { version: 1, curves };
}

/** Points for a plot: the four knots and the curve's value at each. */
export function curvePoints(
  key: CurveKey,
  p: ConsumerParams,
): { x: number; y: number }[] {
  const spec = SPEC_BY_KEY.get(key)!;
  const y = curveValues(p.curves[key]);
  return spec.knots.map((x, i) => ({ x, y: spec.sign * y[i] }));
}
