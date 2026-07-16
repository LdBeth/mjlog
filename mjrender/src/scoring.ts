// Score arithmetic: placements (起家 tie-break) and the オーラス
// "minimum win to overtake the leader" search. Pure functions; scores are in
// units of 100 points throughout (matching the log's `ten`/`sc` encoding).

/** Seat priority for equal scores: closer to the initial East (起家) ranks higher. */
function prio(seat: number, initialEast: number): number {
  return (seat - initialEast + 4) % 4;
}

/** Placement (1..4) per seat. */
export function placements(scores: number[], initialEast = 0): number[] {
  const order = [0, 1, 2, 3].sort(
    (a, b) => scores[b] - scores[a] || prio(a, initialEast) - prio(b, initialEast),
  );
  const rank = new Array<number>(4).fill(0);
  order.forEach((seat, i) => rank[seat] = i + 1);
  return rank;
}

// Candidate hand values for the overtake search: base points of the common
// han/fu grid (fu 30/40/50 × han 1-4, sub-mangan) plus the limit hands.
interface Cand {
  base: number;
  limit?: string; // named limit hand, when the value is one
}

const CANDS: Cand[] = (() => {
  const out: Cand[] = [];
  const seen = new Set<number>();
  for (const fu of [30, 40, 50]) {
    for (let han = 1; han <= 4; han++) {
      const base = fu * (1 << (2 + han));
      if (base >= 2000 || seen.has(base)) continue;
      seen.add(base);
      out.push({ base });
    }
  }
  out.push(
    { base: 2000, limit: "満貫" },
    { base: 3000, limit: "跳満" },
    { base: 4000, limit: "倍満" },
    { base: 6000, limit: "三倍満" },
    { base: 8000, limit: "役満" },
  );
  return out.sort((a, b) => a.base - b.base);
})();

const ru = (points: number) => Math.ceil(points / 100) * 100;

/** Ron payment for a hand of base points. */
function ronPay(dealer: boolean, base: number): number {
  return ru(base * (dealer ? 6 : 4));
}

export interface OvertakeNeeds {
  /** Ron off a non-leader, e.g. "ロン5200" / "ロン満貫". */
  ron?: string;
  /** Ron off the leader (direct hit), e.g. "直撃2600". */
  direct?: string;
  /** Tsumo, e.g. "ツモ1300-2600" (non-dealer), "ツモ2600オール" (dealer), "ツモ満貫". */
  tsumo?: string;
  /** True when even a yakuman direct hit does not overtake. */
  impossible: boolean;
}

/**
 * Minimum winning hand for `seat` to take 1st place this hand, per win mode.
 * Includes honba (+300 ron / +100 each tsumo) and the kyotaku on the table.
 * Returns null when `seat` already holds 1st.
 */
export function overtakeNeeds(args: {
  scores: number[]; // units of 100
  seat: number;
  dealer: number; // this round's oya
  honba: number;
  kyotaku: number;
  initialEast: number;
}): OvertakeNeeds | null {
  const { scores, seat, dealer, honba, kyotaku, initialEast } = args;
  const rank = placements(scores, initialEast);
  if (rank[seat] === 1) return null;
  const leader = rank.indexOf(1);

  const my = scores[seat] * 100;
  const ld = scores[leader] * 100;
  const isDealer = seat === dealer;
  const bonusRon = 300 * honba + 1000 * kyotaku;
  const beats = (a: number, b: number) =>
    a > b || (a === b && prio(seat, initialEast) < prio(leader, initialEast));

  // CANDS is sorted ascending, so the minimum winning hand per mode is the
  // FIRST candidate whose payment overtakes the leader.
  const ronLabel = (c: Cand) => c.limit ?? `${ronPay(isDealer, c.base)}`;
  const tsumoOf = (c: Cand) => {
    if (isDealer) {
      const each = ru(c.base * 2);
      return { gain: each * 3, leaderPays: each, label: c.limit ?? `${each}オール` };
    }
    const fromDealer = ru(c.base * 2);
    const fromOther = ru(c.base);
    return {
      gain: fromDealer + fromOther * 2,
      leaderPays: leader === dealer ? fromDealer : fromOther,
      label: c.limit ?? `${fromOther}-${fromDealer}`,
    };
  };

  const ron = CANDS.find((c) => beats(my + ronPay(isDealer, c.base) + bonusRon, ld));
  const direct = CANDS.find((c) => {
    const pay = ronPay(isDealer, c.base);
    return beats(my + pay + bonusRon, ld - pay - 300 * honba);
  });
  const tsumo = CANDS.find((c) => {
    const t = tsumoOf(c);
    return beats(
      my + t.gain + 100 * honba * 3 + 1000 * kyotaku,
      ld - t.leaderPays - 100 * honba,
    );
  });

  return {
    ron: ron && `ロン${ronLabel(ron)}`,
    direct: direct && `直撃${ronLabel(direct)}`,
    tsumo: tsumo && `ツモ${tsumoOf(tsumo).label}`,
    impossible: !ron && !direct && !tsumo,
  };
}

/**
 * Decode the log's `owari` pairs (score, placement points) into per-seat rows,
 * sorted 1st→4th (score descending; stable, so seat order breaks ties).
 * Scores are converted to points (×100).
 */
export function owariRows(
  owari: number[],
): Array<{ seat: number; score: number; points: number }> {
  const rows = [];
  for (let s = 0; s < 4 && s * 2 + 1 < owari.length; s++) {
    rows.push({ seat: s, score: owari[s * 2] * 100, points: owari[s * 2 + 1] });
  }
  return rows.sort((a, b) => b.score - a.score);
}
