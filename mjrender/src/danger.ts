// Discard-danger assessment against riichi opponents AND threatening open hands.
//
// Returns BOTH a summary level (a labelled heuristic, NOT a wait solver) and
// the raw evidence per threat (suji status, kabe/chance, surviving wait shapes,
// copies visible, yakuhai, plus the meld read for open hands) so the consuming
// LLM can do real push/fold reasoning from features instead of arguing with a
// black-box verdict.
//
// Everything here is deterministic and computed from public information only
// (rivers, melds, dora indicators) plus the discarder's own concealed tiles,
// which only sharpen the wall counts they can personally see.

import { rankOfType, suitOfType, typeGlyph } from "./tiles.ts";

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

/**
 * An opened (non-riichi) hand worth defending against. Tenpai is *suspected*,
 * never known, so the level is damped unless the melds themselves shout value.
 */
export interface FuroThreat {
  seat: number;
  /** Tile types in this seat's own river (furiten-safe vs them).
   *  NOTE: only their OWN river — passed-tile (通り筋) temporary safety after
   *  someone else's discard is deliberately NOT claimed, since an open hand can
   *  ron the very next tile it just declined to call. */
  safeTypes: Set<number>;
  valueHonors: Set<number>;
  /** Open melds only (chi/pon/daiminkan/shouminkan) — drives activation & display.
   *  An ankan leaves the hand closed, so it is excluded here. */
  openMeldCount: number;
  /** Number suit of a flush read ("m"|"p"|"s"), or null. */
  honitsuSuit: "m" | "p" | "s" | null;
  toitoi: boolean;
  /** Yakuhai triplet melds, as honor type numbers. */
  yakuhaiMelds: Set<number>;
  /** Dora tiles visible inside their melds (indicator-derived + aka). */
  meldDora: number;
}

/** Evidence for one non-genbutsu threat. */
export interface ThreatDetail {
  seat: number;
  level: DangerLevel;
  kind: "riichi" | "furo";
  /** Open meld count, for furo details only (the display suffix P#副露N). */
  openMeldCount?: number;
  /** Human-readable feature notes, e.g. ["無スジ", "ワンチャンス", "場に1枚"]. */
  notes: string[];
}

export interface DangerAssessment {
  level: DangerLevel;
  seats: number[]; // threat seats this tile is not genbutsu against
  details: ThreatDetail[]; // evidence per seat, same order as `seats`
}

const RANK: Record<DangerLevel, number> = {
  "安全": 0,
  "危険度低": 1,
  "危険度中": 2,
  "危険度高": 3,
};

const LEVELS: DangerLevel[] = ["安全", "危険度低", "危険度中", "危険度高"];

const minLevel = (a: DangerLevel, b: DangerLevel): DangerLevel => RANK[a] <= RANK[b] ? a : b;

/** The five primitive wait shapes a deal-in can go through. */
type WaitShape = "リャンメン" | "カンチャン" | "ペンチャン" | "シャンポン" | "タンキ";

/** Fixed display order — the note always lists shapes in this sequence. */
const SHAPE_ORDER: readonly WaitShape[] = [
  "リャンメン",
  "カンチャン",
  "ペンチャン",
  "シャンポン",
  "タンキ",
];

const SUIT_LABEL = { m: "萬子", p: "筒子", s: "索子" } as const;

/**
 * Danger of discarding `tileType` given the active riichi threats and the
 * opened hands worth defending against.
 * `visibleCounts` is public information (rivers/melds/indicators);
 * `ownCounts`, when given, adds the discarder's concealed tiles so the
 * kabe/chance/wait-shape evidence matches what the discarder can actually see.
 */
export function assessDanger(
  tileType: number,
  threats: RiichiThreat[],
  furo: FuroThreat[],
  visibleCounts: number[],
  ownCounts?: number[],
): DangerAssessment | null {
  if (threats.length === 0 && furo.length === 0) return null;

  let worst: DangerLevel = "安全";
  const seats: number[] = [];
  const details: ThreatDetail[] = [];
  const add = (d: ThreatDetail) => {
    seats.push(d.seat);
    details.push(d);
    if (RANK[d.level] > RANK[worst]) worst = d.level;
  };

  for (const th of threats) {
    if (th.safeTypes.has(tileType)) continue; // genbutsu vs this seat
    const base = tileDetail(tileType, th, visibleCounts, ownCounts);
    add({
      seat: th.seat,
      level: minLevel(base.level, base.cap),
      kind: "riichi",
      notes: base.notes,
    });
  }

  for (const f of furo) {
    if (f.safeTypes.has(tileType)) continue; // genbutsu vs this seat
    add(furoDetail(tileType, f, visibleCounts, ownCounts));
  }

  // when every threat was genbutsu, worst is still 安全 and the lists are empty
  return { level: worst, seats, details };
}

/** A per-threat reading before the shape cap is folded in. */
interface BaseDetail {
  level: DangerLevel;
  /** Ceiling implied by the surviving wait shapes (never raises the level). */
  cap: DangerLevel;
  notes: string[];
}

/**
 * Which primitive wait shapes could still be waiting on `type` for this threat.
 *
 * A shape survives only if the opponent could physically be holding it: the
 * bridging tiles must still be live (kabe), and for a ryanmen the *other* end
 * of the wait must not be in their genbutsu (suji furiten). Kanchan / penchan /
 * tanki wait on `type` alone, which is already known not to be genbutsu here,
 * so no furiten test applies to them.
 */
function waitShapes(
  type: number,
  safeTypes: Set<number>,
  live: (t: number) => number,
): Set<WaitShape> {
  const out = new Set<WaitShape>();
  if (live(type) >= 1) out.add("タンキ");
  if (live(type) >= 2) out.add("シャンポン");
  if (suitOfType(type) === "z") return out; // honors: only shanpon/tanki exist

  const r = rankOfType(type); // 1..9
  // ryanmen, upper holding (X+1,X+2 → waits X / X+3) and lower (X-2,X-1 → X-3 / X)
  const upper = r <= 6 && !safeTypes.has(type + 3) && live(type + 1) > 0 && live(type + 2) > 0;
  const lower = r >= 4 && !safeTypes.has(type - 3) && live(type - 1) > 0 && live(type - 2) > 0;
  if (upper || lower) out.add("リャンメン");
  // penchan exists only at rank 3 (holding 1,2) and rank 7 (holding 8,9)
  if (r === 3 && live(type - 1) > 0 && live(type - 2) > 0) out.add("ペンチャン");
  if (r === 7 && live(type + 1) > 0 && live(type + 2) > 0) out.add("ペンチャン");
  if (r >= 2 && r <= 8 && live(type - 1) > 0 && live(type + 1) > 0) out.add("カンチャン");
  return out;
}

function shapeNote(shapes: Set<WaitShape>): string {
  const list = SHAPE_ORDER.filter((s) => shapes.has(s));
  return `当たり形:${list.length ? list.join("・") : "なし"}`;
}

/**
 * Ceiling implied by the surviving shapes. No surviving shape ⇒ this tile
 * cannot deal in at all. Only run-shaped waits justify the top levels, so a
 * tile that can only be caught by a shanpon/tanki is capped low — EXCEPT for
 * honors, where shanpon/tanki *is* the whole mechanism and the copies-visible
 * rule below already prices it (capping there would flatten every honor to 低).
 */
function shapeCap(shapes: Set<WaitShape>, isHonor: boolean): DangerLevel {
  if (shapes.size === 0) return "安全";
  if (isHonor) return shapes.has("シャンポン") ? "危険度高" : "危険度低";
  const runless = !shapes.has("リャンメン") && !shapes.has("カンチャン") &&
    !shapes.has("ペンチャン");
  return runless ? "危険度低" : "危険度高";
}

function tileDetail(
  type: number,
  th: { safeTypes: Set<number>; valueHonors: Set<number> },
  visible: number[],
  ownCounts?: number[],
): BaseDetail {
  const notes: string[] = [];
  let level: DangerLevel;
  // Live copies as the discarder can see them (public + their own concealed).
  const live = (t: number) => Math.max(0, 4 - (visible[t] + (ownCounts?.[t] ?? 0)));
  const shapes = waitShapes(type, th.safeTypes, live);
  // Note order is fixed: [suji|honor kind] → [kabe] → 当たり形 → copies visible.
  const cap = shapeCap(shapes, suitOfType(type) === "z");

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
    notes.push(shapeNote(shapes));
    notes.push(visible[type] === 0 ? "生牌" : `場に${visible[type]}枚`);
    return { level, cap, notes };
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
  const sides: number[] = [];
  if (r >= 3) sides.push(Math.min(live(type - 1), live(type - 2)));
  if (r <= 7) sides.push(Math.min(live(type + 1), live(type + 2)));
  if (sides.length > 0) {
    const chance = Math.max(...sides);
    if (chance === 0) notes.push("ノーチャンス");
    else if (chance === 1) notes.push("ワンチャンス");
  }

  notes.push(shapeNote(shapes));
  notes.push(visible[type] === 0 ? "生牌" : `場に${visible[type]}枚`);
  return { level, cap, notes };
}

/** One step down the ladder (低 is the floor for a hand that might be tenpai). */
function downgrade(l: DangerLevel): DangerLevel {
  return LEVELS[Math.max(1, RANK[l] - 1)];
}

/** One step up the ladder, capped at 危険度高. */
function upgrade(l: DangerLevel): DangerLevel {
  return LEVELS[Math.min(3, RANK[l] + 1)];
}

/**
 * An opened hand's reading: the same suji/kabe/shape evidence as a riichi, then
 * damped because tenpai is only suspected (< 3 open melds), then raised once
 * when the melds themselves advertise value (a flush read covering this tile,
 * or a yakuhai meld backed by visible dora).
 */
function furoDetail(
  type: number,
  f: FuroThreat,
  visible: number[],
  ownCounts?: number[],
): ThreatDetail {
  const base = tileDetail(type, f, visible, ownCounts);
  let level = base.level;

  // 1. fewer than 3 open melds ⇒ tenpai is a suspicion, not a reading
  if (f.openMeldCount < 3) level = downgrade(level);

  // 2. at most ONE upgrade: a flush read makes its suit + all honors hot;
  //    a yakuhai meld with 2+ visible dora is a hand worth fearing.
  const honitsuHit = f.honitsuSuit !== null &&
    (suitOfType(type) === f.honitsuSuit || suitOfType(type) === "z");
  if (honitsuHit || (f.yakuhaiMelds.size >= 1 && f.meldDora >= 2)) level = upgrade(level);

  // 3. the surviving wait shapes still cap everything
  level = minLevel(level, base.cap);

  // furo-specific evidence goes last, after the standard notes
  const notes = [...base.notes];
  if (f.honitsuSuit !== null) notes.push(`染め手模様(${SUIT_LABEL[f.honitsuSuit]})`);
  if (f.toitoi) notes.push("トイトイ模様");
  for (const t of [...f.yakuhaiMelds].sort((a, b) => a - b)) {
    notes.push(`役牌副露(${typeGlyph(t)})`);
  }
  if (f.meldDora >= 1) notes.push(`ドラ${f.meldDora}(副露内)`);

  return { seat: f.seat, level, kind: "furo", openMeldCount: f.openMeldCount, notes };
}
