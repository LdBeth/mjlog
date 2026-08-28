// 色読み — the 感性 layer's field sense: トイツ場 and 染め場 (2026-08-28).
//
// NOT a river read. The doctrine (owner, 2026-08-28) distinguishes the two:
// a river read infers ONE opponent's concealed hand from THEIR discards and is
// forbidden to this project's agents; 色読み senses what kind of 場 the whole
// board has become — a field flowing toward pairs (トイツ場) or toward a single
// color (染め場) — and prices the FIELD, not a hand. The facts below are
// therefore exposed only at field level (per-suit heat, one pairing scalar),
// never as "seat N holds X".
//
// Why it exists: the 2026-08-27 arena batch fed 10 mangan+ deal-ins in 30
// games, and the owner's replay review found the cluster was 染め手 — mostly
// CLOSED flush hands. The danger assessor (`mjrender/danger.ts`) only runs when
// a riichi or a real furo threat stands, and prices suji/kabe — exactly the
// wrong ruler against a flush, where the live suit and every honor is hot and
// suji means nothing. A closed dyer on a quiet table was invisible to both the
// risk ladder and the fold gate. The second finding — false 七対子 commitment —
// is the same blindness on the offense side: `kernel.shanten` is the MIN over
// standard/chiitoi, so four early pairs silently flip the whole discard chooser
// onto the pairs line with no judgment about whether the field pairs at all.
// 七対子 is the トイツ場 hand; committing to it in a 順子場 is the trap.
//
// M11's lesson governs the split: the FACTS here are fixed arithmetic (their
// internal constants are definition, like the assessor's thresholds, and are
// never fitted); every behavioural consequence flows through the three
// CONSUMPTION weights, which init to 0 — an absent `sense` ktune block plays
// bit-for-bit the game played before this file existed.

import { tileType } from "mjrender/tiles.ts";
import type { Observation } from "../observe.ts";

/** The field, sensed. All values in [0, 1]. */
export interface FieldSense {
  /**
   * 染め場 heat per suit (m, p, s): how strongly the field has committed to
   * this color. Driven by an opponent river void of the suit while long in the
   * others, reinforced by same-suit melds.
   */
  someba: [number, number, number];
  /** `max(someba)` — how dyed the field is at its hottest. */
  hot: number;
  /**
   * Per suit, the tile types the field's own dye evidence proves out: types the
   * most-committed opponent for that suit has themselves discarded. A field
   * price must not land on a tile the dye's own source let go.
   */
  safe: [Set<number>, Set<number>, Set<number>];
  /**
   * トイツ場: the field flowing toward pairs — pon-heavy table, early discards
   * duplicating across rivers, own draws pairing up.
   */
  toitsuba: number;
}

/** The three consumption weights. All-zero ⇒ the sense changes nothing. */
export interface SenseWeights {
  /**
   * Defence: risk points (the `w.danger` currency — 危険度高 is 200) added to a
   * tile of a dyed suit, scaled by that suit's heat; honors take the hottest
   * suit's heat, since the dyer holds them. Fires precisely where the assessor
   * does not look: a quiet table with a silent flush growing on it.
   */
  someRisk: number;
  /**
   * Fold gate: pressure added as `somePressure × hot` (a declared riichi is
   * 1.0). Un-zeros `computeFold`'s quiet-table early-out against a field the
   * assessor has no threat entry for.
   */
  somePressure: number;
  /**
   * Offense: score tax on a discard whose kept shape rides the chiitoi line
   * (min-shanten = chiitoi < standard) in a field that does not pair —
   * `chiitoiTax × (standard − chiitoi shanten) × (1 − toitsuba)`. In a true
   * トイツ場 the tax vanishes; in a 順子場 the min-shanten flip stops being
   * free. Never taxes a hand already at 6 pairs (see `senseLineTax`).
   */
  chiitoiTax: number;
}

export const INIT_SENSE: SenseWeights = {
  someRisk: 0,
  somePressure: 0,
  chiitoiTax: 0,
};

/** A partial over the identity (flat record). */
export function mergeSense(w?: Partial<SenseWeights>): SenseWeights {
  return { ...INIT_SENSE, ...w };
}

/** Whether any weight is live — the lazy guard; zero weights must cost zero work. */
export function senseActive(w: SenseWeights): boolean {
  return w.someRisk !== 0 || w.somePressure !== 0 || w.chiitoiTax !== 0;
}

// ---------------------------------------------------------------------------
// fact constants — definition, not tuning surface
// ---------------------------------------------------------------------------

/**
 * The void score is the DEFICIT below the uniform expectation, in tiles:
 * `nNum/3 − inX`, minus a dead zone, over a scale. Short rivers void a suit by
 * pure chance constantly (at 6 discards a given suit is ≤1 nearly half the
 * time, and there are nine opponent×suit chances per decision) — the 0827
 * replay measured both a share-based and a raw-void score calling 40%+ of
 * mid-game decisions hot. Deficit demands length before it grants confidence:
 * zero of a suit is worth 0.17 at 6 discards, 0.5 at 9, 1.0 at 13.
 */
const VOID_DEAD = 1.5;
const VOID_SCALE = 3;
/** One whole meld inside the suit is loud — nearly half the certainty alone. */
const MELD_HEAT = 0.35;
/** Pon count that saturates the トイツ場 pon component. */
const PON_SAT = 4;
/** Early junme window for the duplicate-discard component. */
const DUP_JUNME = 8;

const suitOf = (ty: number): number => ty < 9 ? 0 : ty < 18 ? 1 : ty < 27 ? 2 : 3;

/**
 * The consumption bar: heat prices only above this, rescaled to [0, 1]. The
 * home paired sweep (2026-08-28) measured the LINEAR consumption of heat at
 * +0.08 道場順位 per arm even at weak weights — the cost was the constant
 * drizzle of 0.2–0.4 readings taxing ordinary decisions, not the strong reads.
 * A 染め場 is called when the evidence clears the bar, not on every murmur.
 */
const HEAT_BAR = 0.35;

/** `max(0, heat − bar) / (1 − bar)` — what the consumers price. */
export function dyeEff(heat: number): number {
  return Math.max(0, heat - HEAT_BAR) / (1 - HEAT_BAR);
}

/** One opponent's dye evidence, per suit — the lane records these raw. */
export interface OppDyeEvidence {
  /** Deficit-based void score per suit, before meld reinforcement. */
  voidScore: [number, number, number];
  /** Meld reinforcement per suit (`MELD_HEAT` × same-suit melds). */
  meldBoost: [number, number, number];
  /** Number-tile discards in the river. */
  nNum: number;
  /** Honor discards in the river — a dyer keeps honors; recorded for the fit. */
  honors: number;
  /** Every type this opponent has discarded (the dye's own proof set). */
  seen: Set<number>;
}

/**
 * The sense's working parts, exposed raw so the oracle lane can record them
 * and the fit can evaluate ALTERNATIVE fact definitions offline (a different
 * dead zone, an honor-retention corroborator) without replaying anything.
 */
export interface FieldSenseDetail {
  /** Relative opponents 1..3 → entries 0..2. */
  opps: [OppDyeEvidence, OppDyeEvidence, OppDyeEvidence];
  /** トイツ場 components — FIELD evidence only (see the doctrine note below). */
  tPon: number;
  tDup: number;
  /** Own concealed pair types — recorded as ALIGNMENT, never field evidence. */
  ownPairs: number;
}

/** Compute the raw evidence. Pure in the Observation. */
export function fieldSenseDetail(obs: Observation): FieldSenseDetail {
  const opps = [] as unknown as FieldSenseDetail["opps"];
  // A called-away tile was still discarded — it stays in the distribution.
  for (let o = 1; o < 4; o++) {
    const bySuit = [0, 0, 0];
    let nNum = 0;
    let honors = 0;
    const seen = new Set<number>();
    for (const e of obs.rivers[o]) {
      const ty = tileType(e.tile);
      seen.add(ty);
      const s = suitOf(ty);
      if (s < 3) {
        bySuit[s]++;
        nNum++;
      } else honors++;
    }
    const voidScore: [number, number, number] = [0, 0, 0];
    const meldBoost: [number, number, number] = [0, 0, 0];
    for (let s = 0; s < 3; s++) {
      // The deficit-below-expectation void score — see VOID_DEAD/VOID_SCALE.
      voidScore[s] = Math.min(1, Math.max(0, (nNum / 3 - bySuit[s] - VOID_DEAD) / VOID_SCALE));
      for (const m of obs.melds[o]) {
        if (m.tiles.every((t) => suitOf(tileType(t)) === s)) meldBoost[s] += MELD_HEAT;
      }
    }
    opps.push({ voidScore, meldBoost, nNum, honors, seen });
  }

  // トイツ場 components. (1) pon on the table, anyone's: tiles are clumping.
  let pons = 0;
  for (const melds of obs.melds) {
    for (const m of melds) if (m.kind === "pon") pons++;
  }
  const tPon = Math.min(1, pons / PON_SAT);
  // (2) the same types discarded more than once across the early rivers.
  const early = new Map<number, number>();
  for (const river of obs.rivers) {
    for (const e of river) {
      if (e.junme > DUP_JUNME) continue;
      const ty = tileType(e.tile);
      early.set(ty, (early.get(ty) ?? 0) + 1);
    }
  }
  let dup = 0;
  for (const n of early.values()) if (n >= 2) dup++;
  const tDup = early.size >= 6
    ? Math.min(1, Math.max(0, (dup / early.size - 0.15) / 0.35))
    : 0;
  // Own pairing — measured, but NOT folded into the field (doctrine below).
  const counts = new Map<number, number>();
  for (const t of obs.hand) {
    const ty = tileType(t);
    counts.set(ty, (counts.get(ty) ?? 0) + 1);
  }
  let ownPairs = 0;
  for (const n of counts.values()) if (n >= 2) ownPairs++;

  return { opps, tPon, tDup, ownPairs };
}

/**
 * Sense the field. Pure in the Observation; a policy memoizes it per decision.
 *
 * DOCTRINE (owner, 2026-08-28): the 場 is not the same for everyone at the
 * table — usually ONE player is out of the 場 the other three share. Two
 * consequences are built in. For 染め場, the max-over-opponents reduction IS
 * the outlier detection: the field is dyed when one river has left the field's
 * flow, and the price lands on the suit, not the seat. For トイツ場, our OWN
 * hand is excluded from the field evidence (`tPon`/`tDup` only): four pairs in
 * our hand are the TEMPTATION to commit to 七対子, never proof that pairs are
 * flowing — we may be the misaligned one, pairing in a field that runs. The
 * alignment reading comes out of the consumption instead: the chiitoi tax
 * scales by `1 − toitsuba`, so own pairing in a genuinely pairing field is
 * untaxed (aligned), and the same hand in a running field pays full price.
 */
export function fieldSense(obs: Observation): FieldSense {
  const d = fieldSenseDetail(obs);
  const someba: [number, number, number] = [0, 0, 0];
  const safe: [Set<number>, Set<number>, Set<number>] = [new Set(), new Set(), new Set()];
  for (let i = 0; i < 3; i++) {
    const o = d.opps[i];
    for (let s = 0; s < 3; s++) {
      const c = Math.min(1, o.voidScore[s] + o.meldBoost[s]);
      if (c > someba[s]) {
        someba[s] = c;
        safe[s] = o.seen;
      }
    }
  }
  const hot = Math.max(someba[0], someba[1], someba[2]);
  const toitsuba = Math.min(1, 0.55 * d.tPon + 0.45 * d.tDup);
  return { someba, hot, safe, toitsuba };
}

/** 七対子 shanten of a 13-tile counts vector: 6 − pairs, held back under 7 kinds. */
export function chiitoiShanten(counts: number[]): number {
  let pairs = 0;
  let kinds = 0;
  for (let ty = 0; ty < 34; ty++) {
    if (counts[ty] >= 1) kinds++;
    if (counts[ty] >= 2) pairs++;
  }
  return 6 - pairs + Math.max(0, 7 - kinds);
}
