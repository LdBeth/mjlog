// Discard-danger assessment against riichi opponents.
//
// Returns BOTH a summary level (a labelled heuristic, NOT a wait solver) and
// the raw evidence per threat (suji status, kabe/chance, copies visible,
// yakuhai) so the consuming LLM can do real push/fold reasoning from features
// instead of arguing with a black-box verdict.

import { rankOfType, suitOfType } from "./tiles.ts";

export type DangerLevel = "安全" | "危険度低" | "危険度中" | "危険度高";

export interface RiichiThreat {
  seat: number;
  /** Tile *types* that are 100% safe vs this seat (their discards + everything
   *  discarded by anyone after their riichi declaration). */
  safeTypes: Set<number>;
  /** Honor *types* that are yakuhai for this seat (round wind + this seat's wind
   *  + the three dragons). Value honors are held for their han, so they carry a
   *  higher shanpon/tanki deal-in risk than guest honors. */
  valueHonors: Set<number>;
}

/** Evidence for one non-genbutsu threat. */
export interface ThreatDetail {
  seat: number;
  level: DangerLevel;
  /** Human-readable feature notes, e.g. ["無スジ", "ワンチャンス", "場に1枚"]. */
  notes: string[];
}

export interface DangerAssessment {
  level: DangerLevel;
  seats: number[]; // riichi seats this tile is not genbutsu against
  details: ThreatDetail[]; // evidence per seat, same order as `seats`
}

const RANK: Record<DangerLevel, number> = {
  "安全": 0,
  "危険度低": 1,
  "危険度中": 2,
  "危険度高": 3,
};

/**
 * Danger of discarding `tileType` given the active riichi threats.
 * `visibleCounts` is public information (rivers/melds/indicators);
 * `ownCounts`, when given, adds the discarder's concealed tiles so the
 * kabe/chance evidence matches what the discarder can actually see.
 */
export function assessDanger(
  tileType: number,
  threats: RiichiThreat[],
  visibleCounts: number[],
  ownCounts?: number[],
): DangerAssessment | null {
  if (threats.length === 0) return null;

  let worst: DangerLevel = "安全";
  const seats: number[] = [];
  const details: ThreatDetail[] = [];

  for (const th of threats) {
    if (th.safeTypes.has(tileType)) continue; // genbutsu vs this seat
    seats.push(th.seat);
    const d = tileDetail(tileType, th, visibleCounts, ownCounts);
    details.push(d);
    if (RANK[d.level] > RANK[worst]) worst = d.level;
  }

  // when every threat was genbutsu, worst is still 安全 and the lists are empty
  return { level: worst, seats, details };
}

function tileDetail(
  type: number,
  th: RiichiThreat,
  visible: number[],
  ownCounts?: number[],
): ThreatDetail {
  const notes: string[] = [];
  let level: DangerLevel;

  if (suitOfType(type) === "z") {
    // honor: can only deal in as shanpon/tanki, so live copies dominate
    const isValue = th.valueHonors.has(type);
    notes.push(isValue ? "役牌" : "客風");
    if (visible[type] >= 3) {
      level = "危険度低"; // 3 already out ⇒ at most a tanki
    } else if (isValue) {
      // Yakuhai are kept for their value, so a live one is more likely held as
      // a pair (shanpon) than a guest honor: raise it when ≤1 copy is public.
      level = visible[type] <= 1 ? "危険度高" : "危険度中";
    } else {
      level = "危険度中";
    }
    notes.push(visible[type] === 0 ? "生牌" : `場に${visible[type]}枚`);
    return { seat: th.seat, level, notes };
  }

  const r = rankOfType(type); // 1..9
  const needLower = r >= 4; // guarded by the (r-3) suji
  const needUpper = r <= 6; // guarded by the (r+3) suji
  const coveredLower = needLower && th.safeTypes.has(type - 3);
  const coveredUpper = needUpper && th.safeTypes.has(type + 3);
  const need = (needLower ? 1 : 0) + (needUpper ? 1 : 0);
  const covered = (coveredLower ? 1 : 0) + (coveredUpper ? 1 : 0);

  if (need > 0 && covered === need) {
    level = "危険度低"; // full suji
    notes.push("スジ");
  } else if (covered > 0) {
    level = "危険度中"; // half-suji
    notes.push("半スジ");
  } else {
    // no suji: middle tiles are most dangerous
    level = r >= 3 && r <= 7 ? "危険度高" : "危険度中";
    notes.push("無スジ");
  }

  // Kabe / chance: a ryanmen waiting on this tile needs both bridging ranks
  // live in the opponent's hand. Count what the discarder can see (public +
  // their own concealed tiles): min live copies over the viable side(s).
  const live = (t: number) => Math.max(0, 4 - (visible[t] + (ownCounts?.[t] ?? 0)));
  const sides: number[] = [];
  if (r >= 3) sides.push(Math.min(live(type - 1), live(type - 2)));
  if (r <= 7) sides.push(Math.min(live(type + 1), live(type + 2)));
  if (sides.length > 0) {
    const chance = Math.max(...sides);
    if (chance === 0) notes.push("ノーチャンス");
    else if (chance === 1) notes.push("ワンチャンス");
  }

  notes.push(visible[type] === 0 ? "生牌" : `場に${visible[type]}枚`);
  return { seat: th.seat, level, notes };
}
