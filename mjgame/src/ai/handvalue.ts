// 手牌価値 — what OUR OWN hand is worth, as a counting chain over public facts.
//
// 計算 already prices the other side of every decision: `computed.ts` says what
// dealing a tile in costs. This file says what NOT folding is worth, so the two
// halves of a push/fold judgment finally live on the same scale — points.
//
// THE COUNTING CHAIN. Everything below is arithmetic over facts the seat may
// legally count (its own thirteen tiles, the unseen pool, the wall, the melds
// and riichi sticks on the table). Nothing here reads a river.
//
//   1. LEVELS. A hand at shanten s is s advances away from tenpai. The chain
//      has at most four rungs — shanten ≥3, 2, 1, tenpai — because beyond three
//      away the distinction stops paying for itself and the wall runs out first.
//   2. ADVANCE. At a rung holding `u` live acceptances out of `U` unseen tiles,
//      one own draw advances with p = u/U. `u` is EXACT for the rung the hand is
//      standing on (the caller counted it); for the rungs below it is not known
//      yet — the hand has not chosen those shapes — so it comes from a fitted
//      mean, `meanUkeire`.
//   3. WIN. A tenpai hand wins on its own draw with w/U, and on somebody else's
//      discard some multiple of that; `ronFactor` is that multiple, per
//      opponent, and it is zero when the wait is not ronnable or is furiten.
//   4. SURVIVAL. Every own turn the table gets a chance to end the hand first:
//      `oppHazard` per unit of Σ P(opponent tenpai), with `oppGrowth` letting
//      that sum drift up as the hand gets late. This is the ONLY place opponents
//      enter, and they enter as a base rate — never as a per-seat read.
//   5. VALUE. A separate, static price for the hand if it does land: menzen vs
//      open vs 染め手 base, plus dora and 役牌, ×1.5 dealer, capped, and then the
//      exact 本場/供託 add-ons on top of the cap because those are surcharges on
//      the PAYMENT and not part of the hand (same order as `score.ts#ronValue`).
//
// WHY THE SCALARS ARE FITTED, AND WHY ONLY THE SCALARS (the M10 pattern). The
// structure above is not a model of anything — it is bookkeeping, and it is
// either right or it is a bug. What is genuinely unknown is a handful of
// population averages: how wide a 2-shanten hand's acceptance turns out to be
// once it is chosen, how much a ron adds to a tsumo, how fast the table readies.
// Those are named in `HandWeights`, defaulted from measurement, and refitted
// offline against recorded ground truth (`handcalib.ts` records the FACTS, the
// fit re-enters THIS function). So the seat that plays and the fit that tunes it
// run the same code over the same numbers, and the model can never fork into two
// versions of itself.
//
// The file is deliberately dependency-free — types only, no `Observation`, no
// engine. Facts are built by the policy, which has the context to count them.

/**
 * Everything `handOutlook` is allowed to know, counted by the caller.
 *
 * These are the fields the calibration recorder stores verbatim, so a change
 * here is a change to a recorded file format: add, do not repurpose.
 */
export interface HandFacts {
  /** shanten of the 13-tile resting shape, 0 = tenpai (clamp ≥0). */
  shanten: number;
  /** live copies the shape accepts now (Σ unseen over ukeire types); at tenpai: live wait copies. */
  ukeire: number;
  ukeireTypes: number;
  /** Σ unseen over all 34 types (the draw denominator). */
  unseenTotal: number;
  /** own draws remaining ≈ floor(wallRemaining / 4). */
  turnsLeft: number;
  junme: number;
  /** dora in hand + melds, aka included. */
  dora: number;
  /** open (non-ankan) melds. */
  open: number;
  /** menzen. */
  closed: boolean;
  /** already declared. */
  riichi: boolean;
  /** value-honor triplets/kans (melded or concealed). */
  yakuhaiTriplets: number;
  /** concealed value-honor pairs. */
  yakuhaiPairs: number;
  /** ≤2 strays outside the dominant suit (honors welcome). */
  honitsu: boolean;
  /** tenpai (or would-be) hand has ≥1 ronnable wait. Closed & not furiten ⇒ true (riichi cures). */
  ronnable: boolean;
  furiten: boolean;
  dealer: boolean;
  /** P(tenpai) of the three opponents, relative order. */
  oppTenpai: readonly number[];
  honba: number;
  kyotaku: number;
}

/**
 * The fitting surface. Every number the chain cannot count is here and nowhere
 * else — there is no magic constant in `handOutlook` below except the exact
 * 300/1000 of the 本場 and 供託 rules, which are the rules and not a model.
 */
export interface HandWeights {
  /**
   * Expected live ukeire at shanten [≥3, 2, 1] after an advance, and live wait
   * copies at tenpai — used for the levels BELOW the current one, whose real
   * count is unknown because the hand has not committed to those shapes yet.
   */
  meanUkeire: readonly [number, number, number, number];
  /** Per-opponent ron hazard relative to the tsumo hazard for a ronnable tenpai. */
  ronFactor: number;
  /** Per own-turn P(a tenpai opponent ends the hand), × Σ oppTenpai. */
  oppHazard: number;
  /** Growth of Σ oppTenpai per future turn (the table gets readier). */
  oppGrowth: number;
  /** Own-hand value model (initialised from `DEFAULT_COMPUTED`'s numbers). */
  valueRiichi: number;
  valueDamaten: number;
  valueOpen: number;
  valueHonitsu: number;
  valuePerDora: number;
  valueYakuhai: number;
  valueDealer: number;
  valueCap: number;
  /**
   * CONSUMPTION scalars — how the policy SPENDS the outlook, not how the model
   * predicts. They are tuned by paired runs, never by the offline fit, because
   * no recorded ground truth exists for "how hard should this seat push".
   */
  pushScale: number;
  /** discard-score units per point of EV. */
  evWeight: number;
}

/**
 * Measured starting point. The value block is `DEFAULT_COMPUTED`'s own value
 * model verbatim — the price we put on an opponent's hand and the price we put
 * on ours are the same published averages, and if a fit moves one it should be
 * asked why it did not move the other.
 */
export const DEFAULT_HAND: HandWeights = {
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
  pushScale: 3000,
  evWeight: 0.1,
};

/**
 * Fill a partial override against the defaults.
 *
 * `meanUkeire` is merged ELEMENT BY ELEMENT, for the same reason `computed.ts`
 * merges its tenpai table cell by cell: the four rungs are four independent
 * parameters, a fit may move one and leave the others, and a whole-array
 * override read out of a JSON `--ktune` file would otherwise let a dropped entry
 * arrive silently as `undefined`. The result is always a fresh tuple, so a
 * caller may not reach back through it into the defaults.
 */
export function mergeHand(w?: Partial<HandWeights>): HandWeights {
  const d = DEFAULT_HAND.meanUkeire;
  const o = w?.meanUkeire;
  const mu: [number, number, number, number] = [
    o?.[0] ?? d[0],
    o?.[1] ?? d[1],
    o?.[2] ?? d[2],
    o?.[3] ?? d[3],
  ];
  return { ...DEFAULT_HAND, ...w, meanUkeire: mu };
}

/** What the chain concluded: the two halves and their product. */
export interface HandOutlook {
  /** P(this hand is the one that wins), over the remaining own draws. */
  pwin: number;
  /** Points it collects if it does, 本場/供託 included. */
  value: number;
  /** `pwin * value` — the figure the fold gate and the discard score spend. */
  ev: number;
}

/** Longest DP the wall can ever justify: 136 live tiles ÷ 4 seats, with slack. */
const MAX_TURNS = 20;

/**
 * The whole model, in closed form: a ≤20×4 forward sweep, no allocation beyond
 * one four-slot row, no I/O, no randomness. Pure — the same facts and weights
 * give the same bits on the play path and inside the offline fit.
 */
export function handOutlook(f: HandFacts, w: HandWeights): HandOutlook {
  const pwin = handPwin(f, w);
  const value = handValue(f, w);
  return { pwin, value, ev: pwin * value };
}

/**
 * P(win), by walking the level chain forward one own draw at a time.
 *
 * Turn order inside a step is the real one: we draw (a tenpai hand wins here, a
 * lesser hand advances here), then the other three seats act (the hazard). A
 * hand that reaches tenpai on THIS draw cannot also win on it — it has just now
 * become tenpai — so arrivals are folded in after the win is taken, and their
 * first chance is the next turn.
 */
function handPwin(f: HandFacts, w: HandWeights): number {
  const turns = Math.min(MAX_TURNS, Math.max(0, Math.floor(f.turnsLeft)));
  if (turns <= 0) return 0;
  const U = Math.max(1, f.unseenTotal);
  const shanten = Math.max(0, Math.floor(f.shanten));
  // ≥3 away → rung 0, 2 → 1, 1 → 2, tenpai → 3.
  const cur = 3 - Math.min(shanten, 3);

  // Live acceptances per rung: exact where we stand, the fitted mean below it.
  const adv: number[] = [0, 0, 0, 0];
  for (let i = cur; i < 3; i++) {
    const u = i === cur ? Math.max(0, f.ukeire) : Math.max(0, w.meanUkeire[i]);
    adv[i] = Math.min(1, u / U);
  }

  // The tenpai rung's own hazard. A ron is `ronFactor` of a tsumo, three times
  // over — one per opponent — and a furiten wait collects none of it. An already
  // declared riichi is ronnable by construction.
  const waits = shanten === 0 ? Math.max(0, f.ukeire) : Math.max(0, w.meanUkeire[3]);
  const rons = (f.ronnable || f.riichi) && !f.furiten ? 3 * w.ronFactor : 0;
  const h = Math.min(1, Math.max(0, (waits / U) * (1 + rons)));

  const sumOpp = f.oppTenpai.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);

  const mass = [0, 0, 0, 0];
  mass[cur] = 1;
  let pwin = 0;
  for (let t = 0; t < turns; t++) {
    const won = mass[3] * h;
    pwin += won;
    mass[3] -= won;
    // Downward, so no rung climbs two steps in one draw.
    for (let i = 2; i >= cur; i--) {
      const moved = mass[i] * adv[i];
      mass[i] -= moved;
      mass[i + 1] += moved;
    }
    const q = Math.min(0.95, Math.max(0, w.oppHazard * (sumOpp + w.oppGrowth * t)));
    const survive = 1 - q;
    for (let i = cur; i <= 3; i++) mass[i] *= survive;
  }
  return Math.min(1, Math.max(0, pwin));
}

/**
 * What the hand pays if it lands.
 *
 * Menzen splits on whether the win can actually be COLLECTED off a discard: a
 * closed hand that is riichi or not furiten prices as a riichi hand (the
 * declaration is still available to it), a furiten damaten as a damaten. Open
 * hands split on 染め手 only. 本場 and 供託 are added AFTER the cap, because
 * they are surcharges on the payment rather than part of the hand's value —
 * the same order `score.ts#ronValue` settles them in.
 */
function handValue(f: HandFacts, w: HandWeights): number {
  const base = f.closed
    ? (f.riichi || !f.furiten ? w.valueRiichi : w.valueDamaten)
    : (f.honitsu ? w.valueHonitsu : w.valueOpen);
  let v = base +
    w.valuePerDora * f.dora +
    w.valueYakuhai * (f.yakuhaiTriplets + 0.5 * f.yakuhaiPairs);
  if (f.dealer) v *= w.valueDealer;
  v = Math.min(w.valueCap, v);
  return v + 300 * f.honba + 1000 * f.kyotaku;
}
