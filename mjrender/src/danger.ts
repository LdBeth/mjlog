// Rough discard-danger heuristic against riichi opponents.
// This is a labelled heuristic (genbutsu / suji / kind), NOT a wait solver —
// the consuming LLM is expected to add real push/fold judgement.

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

export interface DangerAssessment {
  level: DangerLevel;
  seats: number[]; // riichi seats this tile is not genbutsu against
}

const RANK: Record<DangerLevel, number> = {
  "安全": 0,
  "危険度低": 1,
  "危険度中": 2,
  "危険度高": 3,
};

/** Danger of discarding `tileType` given the active riichi threats. */
export function assessDanger(
  tileType: number,
  threats: RiichiThreat[],
  visibleCounts: number[],
): DangerAssessment | null {
  if (threats.length === 0) return null;

  let worst: DangerLevel = "安全";
  const seats: number[] = [];

  for (const th of threats) {
    if (th.safeTypes.has(tileType)) continue; // genbutsu vs this seat
    seats.push(th.seat);
    const level = tileLevel(tileType, th.safeTypes, visibleCounts, th.valueHonors);
    if (RANK[level] > RANK[worst]) worst = level;
  }

  if (seats.length === 0) return { level: "安全", seats: [] };
  return { level: worst, seats };
}

function tileLevel(
  type: number,
  safe: Set<number>,
  visible: number[],
  valueHonors: Set<number>,
): DangerLevel {
  if (suitOfType(type) === "z") {
    // honor: fewer live copies ⇒ safer (can only be shanpon/tanki)
    if (visible[type] >= 3) return "危険度低"; // 3 already out ⇒ at most a tanki
    // Yakuhai are kept for their value, so a live one is more likely held as a
    // pair (shanpon) than a guest honor: raise it when ≤1 copy is public.
    if (valueHonors.has(type)) return visible[type] <= 1 ? "危険度高" : "危険度中";
    return "危険度中";
  }
  const r = rankOfType(type); // 1..9
  const needLower = r >= 4; // guarded by the (r-3) suji
  const needUpper = r <= 6; // guarded by the (r+3) suji
  const coveredLower = needLower && safe.has(type - 3);
  const coveredUpper = needUpper && safe.has(type + 3);
  const need = (needLower ? 1 : 0) + (needUpper ? 1 : 0);
  const covered = (coveredLower ? 1 : 0) + (coveredUpper ? 1 : 0);

  if (need > 0 && covered === need) return "危険度低"; // full suji
  if (covered > 0) return "危険度中"; // half-suji
  // no suji: middle tiles are most dangerous
  return r >= 3 && r <= 7 ? "危険度高" : "危険度中";
}
