// M15 — the `ev` ktune block: population scalars, search bounds and the three
// sub-switches of the expected-value core, plus the rule constants the C++
// scorer needs. Pure data: no FFI here (that is `ev.ts`).
//
// WHAT IS A PARAMETER AND WHAT IS NOT. Everything the DP does with the hand is
// counting over public facts and the rules — none of it is tunable. What IS
// tunable is the handful of population averages every self-hand model needs
// (how wide the next rung's acceptance turns out to be, how fast the table
// readies, how a ron compares to a tsumo) — the same scalars `handvalue.ts`
// carries, with the same defaults, so the incumbent and the DP price the same
// table the same way where the DP is silent (the closed-form tail).
//
// SWITCH SEMANTICS. Absent block ⇒ the seat never touches `libmjev` and plays
// bit-for-bit its prior game. Present ⇒ the dylib is REQUIRED (`ev.ts` throws
// without it). `{}` ⇒ every default below, all three sub-switches on.

import { EV_PARAM_ORDER, EV_PARAMS_LEN } from "./evlayout.ts";

export interface EvParams {
  /** Expected live ukeire at shanten [≥3, 2, 1, tenpai] for rungs below the current one (the tail). */
  meanUkeire: readonly [number, number, number, number];
  /** Per-opponent ron hazard relative to the tsumo hazard for a ronnable tenpai. */
  ronFactor: number;
  /** Per own-turn P(a tenpai opponent ends the hand), × Σ tenpaiP. */
  oppHazard: number;
  /** Growth of Σ tenpaiP per future turn. */
  oppGrowth: number;
  /** Static value model for the tail (DEFAULT_HAND ≡ DEFAULT_COMPUTED). */
  valueRiichi: number;
  valueDamaten: number;
  valueOpen: number;
  valueHonitsu: number;
  valuePerDora: number;
  valueYakuhai: number;
  valueDealer: number;
  valueCap: number;
  /** Deal-in rate of a FUTURE (unknown) discard per tenpai opponent per turn. */
  dealinRate: number;
  /** Share of an opponent's win value we pay when they win off someone else / tsumo (0.5 when dealer). */
  tsumoShare: number;
  /** Residual per-turn deal-in hazard of a defended (folded) hand. */
  foldHazard: number;
  /** Multiplier on the future deal-in rate after our own riichi (tsumogiri only). */
  riichiDealinMult: number;
  /** P(一発) on the first turn after declaration. */
  ippatsuP: number;
  /** Fraction of the riichi stick treated as lost at 流局 (1 = lost). */
  stickAtDraw: number;
  /** Value of dealer 連荘 on a dealer win/tenpai (0 = not valued). */
  dealerRenchan: number;
  /** Points a call must gain over passing (calls). */
  callMargin: number;
  /** Points riichi must gain over dama (riichi). */
  riichiMargin: number;
  /** Points per hand-written score unit (dojoCost/senseLineTax/planner bonuses). */
  pointsPerScore: number;
  /**
   * Exact enumeration up to this shanten; deeper roots take the tail. It is a
   * MAXIMUM, not a setting: `mjev_eval_discard` prices the whole field at the
   * deepest level EVERY candidate can afford, so the level actually used is
   * the worst candidate's 向聴 when that is at or below this, and the tail
   * otherwise (see native/README.md).
   *
   * STILL 3 on 2026-08-31, and now for a measured reason rather than a
   * historical one. 60 real discard roots from `runs/ev/lane-800000.jsonl`,
   * after the search was widened and the shape geometry shared, at
   * `maxNodes` 1,200,000:
   *
   *   | 設定 | 決定/秒 | 平均ms | p95ms | 3向聴根の厳密到達 |
   *   | 3    |    3.6  |   278  |  920  | 1/9               |
   *   | 4    |    2.0  |   506  | 2122  | 5/9               |
   *   | 5    |    2.0  |   506  | 2122  | 5/9 (4 と同一)    |
   *
   * Level 4 buys the exact model for about half the 3向聴 roots and costs the
   * OTHER half the entire budget for nothing: an attempt that escapes is
   * abandoned and the field is answered from the tail anyway, so an
   * unaffordable level is strictly worse than one that is never offered. At
   * `maxNodes` 2,500,000 five of six 3向聴 roots do complete, but p95 is then
   * 3.5 s. 3 is the deepest level whose p95 stays inside a second.
   * `exactShanten` 5 is indistinguishable from 4 on real hands: a root whose
   * worst candidate is 5向聴 is rare enough not to appear in 60 draws.
   */
  exactShanten: number;
  /**
   * Same-shanten upgrades (待ち替え) enumerated at shanten ≤ this; clamped to 1
   * inside the engine. 1: the oracle-mode diff puts the whole remaining
   * pruning loss on these lines (6.0% mean / 22.8% worst on 1向聴 rests at
   * T = 2), so switching them off is not on offer — what is tunable is the
   * acceptance-mass gate they must clear, and that is measured in `mjev.cc`.
   */
  sameShantenRungs: number;
  /**
   * Value-state budget for ONE evaluation, spent by the root candidates in
   * order; an attempt that runs out is abandoned whole, so the number is a
   * FIELD budget and not a per-candidate one.
   *
   * 1,200,000 since 2026-08-31 (was 60,000), on the owner's directive that
   * speed is spendable for decision quality. 60,000 was measured before the
   * budget-slicing repair and is now far below the completion size: on 60 real
   * lane roots at `exactShanten` 3 it left 14 of 17 2向聴 fields and every
   * 3向聴 field priced by the closed-form tail. At 1,200,000 the mean field
   * costs 278 ms (3.6 決定/秒), p95 920 ms, and only 2 of 21 2向聴 fields
   * still truncate. Truncation is NOT graceful — the whole field drops a
   * model — so the budget must stay above the completion size, not near it.
   */
  maxNodes: number;
  /** Sub-switches — which decisions the core serves. */
  discard: boolean;
  riichi: boolean;
  calls: boolean;
  /** Rule constants for the scorer (JANKI defaults). */
  kuitan: boolean;
  kazoeYakuman: boolean;
  kiriageMangan: boolean;
  doubleWindFu: 2 | 4;
  notenPenaltyTotal: number;
}

export const DEFAULT_EV: EvParams = {
  meanUkeire: [24, 20, 14, 6],
  ronFactor: 0.5,
  oppHazard: 0.12,
  oppGrowth: 0.04,
  valueRiichi: 7000,
  valueDamaten: 4200,
  valueOpen: 3900,
  valueHonitsu: 7700,
  valuePerDora: 1600,
  valueYakuhai: 1000,
  valueDealer: 1.5,
  valueCap: 16000,
  dealinRate: 0.05,
  tsumoShare: 0.3,
  foldHazard: 0.01,
  riichiDealinMult: 1.5,
  ippatsuP: 0.1,
  stickAtDraw: 1,
  dealerRenchan: 0,
  callMargin: 0,
  riichiMargin: 0,
  pointsPerScore: 4,
  exactShanten: 3,
  sameShantenRungs: 1,
  maxNodes: 1200000,
  discard: true,
  riichi: true,
  calls: true,
  kuitan: true,
  kazoeYakuman: false,
  kiriageMangan: false,
  doubleWindFu: 2,
  notenPenaltyTotal: 3000,
};

/**
 * Fill a partial override against the defaults; `meanUkeire` element-wise
 * (the `mergeHand` reasoning). Dies on a non-finite number or a bad enum, so a
 * typo in a JSON vector is a refusal, not a NaN that "discards the first tile".
 */
export function mergeEv(w?: Partial<EvParams>): EvParams {
  const d = DEFAULT_EV.meanUkeire;
  const o = w?.meanUkeire;
  const mu: [number, number, number, number] = [
    o?.[0] ?? d[0],
    o?.[1] ?? d[1],
    o?.[2] ?? d[2],
    o?.[3] ?? d[3],
  ];
  const m: EvParams = { ...DEFAULT_EV, ...w, meanUkeire: mu };
  for (const [k, v] of Object.entries(m)) {
    if (typeof v === "number" && !Number.isFinite(v)) throw new Error(`ev.${k} が有限でない: ${v}`);
    if (Array.isArray(v) && v.some((x) => !Number.isFinite(x))) {
      throw new Error(`ev.${k} が有限でない`);
    }
  }
  if (m.doubleWindFu !== 2 && m.doubleWindFu !== 4) {
    throw new Error(`ev.doubleWindFu は 2 か 4: ${m.doubleWindFu}`);
  }
  if (m.exactShanten < 0 || m.exactShanten > 6) {
    throw new Error(`ev.exactShanten は 0..6: ${m.exactShanten}`);
  }
  if (m.maxNodes < 1) throw new Error(`ev.maxNodes は 1 以上: ${m.maxNodes}`);
  if (m.pointsPerScore <= 0) throw new Error(`ev.pointsPerScore は正: ${m.pointsPerScore}`);
  return m;
}

/** The `mjev_create` vector, in `EV_PARAM_ORDER`. Booleans are 0/1. */
export function packEvParams(p: EvParams): Float64Array {
  const out = new Float64Array(EV_PARAMS_LEN);
  const rec = p as unknown as Record<string, unknown>;
  EV_PARAM_ORDER.forEach((name, i) => {
    let v: unknown;
    if (name.startsWith("meanUkeire")) v = p.meanUkeire[Number(name.slice(-1))];
    else v = rec[name];
    out[i] = typeof v === "boolean" ? (v ? 1 : 0) : (v as number);
  });
  return out;
}
