// The riichi head (M12) — the first learned DECISION model: declare or damaten.
//
// `wantRiichi` has always been four booleans: waits exist, a wait is live, not
// furiten, four tiles of wall. Inside the region where all four pass it has
// exactly one answer — declare — and that is the gap this head fills: riichi
// versus damaten is a judgement about wait quality, hand value, how late it is
// and who else has already declared, none of which the booleans read.
//
// WHAT STAYS OUTSIDE, permanently. The four gates themselves, `riichiBanned`
// (地獄単騎 / 即引っかけ are 禁じ手 vetoes, not scores) and the `mustCure`
// override (片和了り forces the declaration) are not judgements the model may
// buy its way past — the head is consulted strictly INSIDE the gated-in region
// and decides nothing else. Same principle `consumer.ts` states for the
// discard core.
//
// ABSENT BY DEFAULT, the M11 discipline: no `riichi` block in the `--ktune`
// file means `wantRiichi` behaves bit-for-bit as it always has. And the INIT
// weights below are the identity in the other direction: bias +1, every
// feature weight 0, so a head initialised from them declares unconditionally —
// the pre-head behaviour reproduced exactly, which is what makes the swap
// measurable rather than merely plausible.
//
// Every feature is FINITE BY CONSTRUCTION — counts, flags, bounded model
// outputs; no ratios that can divide by zero. A NaN anywhere would poison the
// init sum (0·NaN is NaN) and silently break the equivalence the tests pin.
//
// The feature set is deliberately confined to the population-safe axis: the
// seat's own thirteen tiles, public counts and the scoring rules. Whether to
// declare barely depends on how the opposition plays, so a head fitted against
// a synthetic field transfers in a way the opponent-model fits do not.

/** What the head reads at one gated-in riichi decision. */
export interface RiichiFeatures {
  /**
   * The M11 outlook of the post-discard tenpai shape, in thousands of points:
   * `ev/1000`, so a typical value lands O(1..8) and its weight is O(1) like the
   * others. Priced by `handValue` under its declared/closed branch — the model
   * has no damaten price (see the trap note in `heuristic.ts`) — so this is
   * "what declaring is playing for", and the damaten side of the comparison
   * lives entirely in the weights.
   */
  ev: number;
  /** P(this hand wins) over the remaining draws, 0..1 — the outlook's other half. */
  pwin: number;
  /** Outlook value in thousands of points (the `ev` above is `pwin·value`). */
  value: number;
  /** Live copies of the waits (facts.ukeire at tenpai) — wait QUALITY, 0..~30ish. */
  liveWaits: number;
  /** Distinct wait types — wait BREADTH. */
  waitTypes: number;
  /** 巡目 — a 2巡 declaration and a 15巡 one are different bets. */
  junme: number;
  /** Own draws remaining, ≈ wallRemaining/4 — how long the lock-in lasts. */
  turnsLeft: number;
  /** Dora in the resting shape (aka included) — value already worth protecting. */
  dora: number;
  /** 1 when dealer, else 0 — the 1.5× multiplier and the 連荘 stake. */
  dealer: number;
  /** Opponents already in riichi, 0..3 — the 追っかけ question. */
  oppRiichi: number;
  /** Riichi sticks already on the table — declaring buys a shot at them. */
  kyotaku: number;
}

/**
 * One weight per feature plus the bias: declare iff
 * `bias + Σ wᵢ·xᵢ ≥ 0`. Linear on purpose — a decision boundary over a dozen
 * meaningful numbers is fittable from a small corpus and every coefficient is
 * readable as a policy statement ("each opponent riichi is worth −0.4 of a
 * declaration").
 */
export interface RiichiWeights {
  bias: number;
  ev: number;
  pwin: number;
  value: number;
  liveWaits: number;
  waitTypes: number;
  junme: number;
  turnsLeft: number;
  dora: number;
  dealer: number;
  oppRiichi: number;
  kyotaku: number;
}

/**
 * The identity head: always declare, exactly what the code did before a head
 * existed. `bias` 1 and nothing else, so `decideRiichi` returns true for every
 * finite feature vector — the init-equivalence test drives real hanchan
 * through both paths and requires the same bits.
 */
export const INIT_RIICHI: RiichiWeights = {
  bias: 1,
  ev: 0,
  pwin: 0,
  value: 0,
  liveWaits: 0,
  waitTypes: 0,
  junme: 0,
  turnsLeft: 0,
  dora: 0,
  dealer: 0,
  oppRiichi: 0,
  kyotaku: 0,
};

/**
 * Fill a partial override against the identity. Flat scalars only, so unlike
 * `mergeHand` there is no element-wise array to guard.
 */
export function mergeRiichi(w?: Partial<RiichiWeights>): RiichiWeights {
  return { ...INIT_RIICHI, ...w };
}

/** The verdict: declare (true) or stay damaten (false). Pure; no I/O, no rng. */
export function decideRiichi(f: RiichiFeatures, w: RiichiWeights): boolean {
  return w.bias +
      w.ev * f.ev +
      w.pwin * f.pwin +
      w.value * f.value +
      w.liveWaits * f.liveWaits +
      w.waitTypes * f.waitTypes +
      w.junme * f.junme +
      w.turnsLeft * f.turnsLeft +
      w.dora * f.dora +
      w.dealer * f.dealer +
      w.oppRiichi * f.oppRiichi +
      w.kyotaku * f.kyotaku >= 0;
}
