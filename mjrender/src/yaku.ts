// Tenhou yaku / yakuman name table (ids 0..54) and the open-hand yaku read
// (deterministic 役 outlook for a 3-meld hand, used as 副露判断 evidence).
// Regular wins list `(id, han)` pairs in AGARI `yaku`; yakuman wins list ids in
// AGARI `yakuman`. ids 37..51 are yakuman; 52..54 are dora / ura / aka.

import { rankOfType, suitOfType, tileType, typeGlyph } from "./tiles.ts";
import type { Meld, Tile } from "./model.ts";

export const YAKU_NAMES: readonly string[] = [
  "門前清自摸和", // 0
  "立直", // 1
  "一発", // 2
  "槍槓", // 3
  "嶺上開花", // 4
  "海底摸月", // 5
  "河底撈魚", // 6
  "平和", // 7
  "断幺九", // 8
  "一盃口", // 9
  "自風 東", // 10
  "自風 南", // 11
  "自風 西", // 12
  "自風 北", // 13
  "場風 東", // 14
  "場風 南", // 15
  "場風 西", // 16
  "場風 北", // 17
  "役牌 白", // 18
  "役牌 發", // 19
  "役牌 中", // 20
  "両立直", // 21
  "七対子", // 22
  "混全帯幺九", // 23
  "一気通貫", // 24
  "三色同順", // 25
  "三色同刻", // 26
  "三槓子", // 27
  "対々和", // 28
  "三暗刻", // 29
  "小三元", // 30
  "混老頭", // 31
  "二盃口", // 32
  "純全帯幺九", // 33
  "混一色", // 34
  "清一色", // 35
  "人和", // 36
  "天和", // 37 (yakuman from here)
  "地和", // 38
  "大三元", // 39
  "四暗刻", // 40
  "四暗刻単騎", // 41
  "字一色", // 42
  "緑一色", // 43
  "清老頭", // 44
  "九蓮宝燈", // 45
  "純正九蓮宝燈", // 46
  "国士無双", // 47
  "国士無双13面", // 48
  "大四喜", // 49
  "小四喜", // 50
  "四槓子", // 51
  "ドラ", // 52
  "裏ドラ", // 53
  "赤ドラ", // 54
];

export function yakuName(id: number): string {
  return YAKU_NAMES[id] ?? `役${id}`;
}

export function isYakumanId(id: number): boolean {
  return id >= 37 && id <= 51;
}

/** Parse an AGARI `yaku="id,han,id,han,..."` attribute. */
export function parseYaku(attr: string | undefined): Array<{ id: number; han: number }> {
  if (!attr) return [];
  const nums = attr.split(/[\s,]+/).filter((s) => s.length).map(Number);
  const out: Array<{ id: number; han: number }> = [];
  for (let i = 0; i + 1 < nums.length; i += 2) {
    out.push({ id: nums[i], han: nums[i + 1] });
  }
  return out;
}

/** Parse an AGARI `yakuman="id,id,..."` attribute. */
export function parseYakuman(attr: string | undefined): number[] {
  if (!attr) return [];
  return attr.split(/[\s,]+/).filter((s) => s.length).map(Number);
}

const LIMIT_NAMES = ["", "満貫", "跳満", "倍満", "役満", "数え役満"];

export function limitName(limit: number): string {
  return LIMIT_NAMES[limit] ?? "";
}

// ---- open-hand yaku read (副露判断 anchor evidence) ----

const yaochuu = (t: number): boolean =>
  suitOfType(t) === "z" || rankOfType(t) === 1 || rankOfType(t) === 9;
const SUIT_LABEL = { m: "萬", p: "筒", s: "索" } as const;
const SUITS = ["m", "p", "s"] as const;
const SUIT_BASE = { m: 0, p: 9, s: 18 } as const;

/**
 * Deterministic yaku outlook for an opened hand: which yaku the melds still
 * admit, judged from the melds plus the reconstructed concealed tiles at that
 * moment. Vocabulary: 確定 = already melded, 可 = nothing blocks it,
 * 後付け可 = a concealed yakuhai pair keeps the back-up open, 見込み = ≥2 of
 * the needed run blocks are realized (melded or complete in hand). Falls back
 * to 役なし懸念 when nothing applies — the classic 3副露 hazard.
 */
export function openYakuRead(
  melds: Meld[],
  concealed: Tile[],
  valueHonors: ReadonlySet<number>,
  kuitan: boolean,
): string[] {
  const reads: string[] = [];
  const meldTypes = melds.flatMap((m) => m.tiles.map(tileType));
  const handTypes = concealed.map(tileType);
  const handCount = new Array<number>(34).fill(0);
  for (const t of handTypes) handCount[t]++;

  // 役牌: a melded triplet locks the yaku in; a concealed pair keeps the
  // back-up (後付け) open.
  const tripletTypes = new Set(
    melds.filter((m) => m.kind !== "chi").map((m) => tileType(m.tiles[0])),
  );
  for (const m of melds) {
    if (m.kind === "chi") continue;
    const t = tileType(m.tiles[0]);
    if (valueHonors.has(t)) reads.push(`役牌${typeGlyph(t)}確定`);
  }
  for (const t of [...valueHonors].sort((a, b) => a - b)) {
    if (!tripletTypes.has(t) && handCount[t] >= 2) reads.push(`役牌${typeGlyph(t)}後付け可`);
  }

  // 断幺九: the melds must be clean; concealed offenders can still be cut.
  if (kuitan && !meldTypes.some(yaochuu)) {
    const n = handTypes.filter(yaochuu).length;
    reads.push(`断幺九可${n ? `（手内幺九${n}枚）` : ""}`);
  }

  // 混一色/清一色: melds committed to one suit (or all honors). The target
  // suit is the meld suit, else the concealed majority suit; only claimed
  // while ≤3 concealed tiles would have to go.
  const meldSuits = new Set(
    meldTypes.filter((t) => suitOfType(t) !== "z").map((t) => suitOfType(t)),
  );
  if (meldSuits.size <= 1) {
    const bySuit = (s: string) => handTypes.filter((t) => suitOfType(t) === s).length;
    const suit = meldSuits.size === 1
      ? [...meldSuits][0]
      : SUITS.reduce((a, b) => (bySuit(b) > bySuit(a) ? b : a));
    if (bySuit(suit) > 0 || meldSuits.size === 1) {
      const honors = meldTypes.some((t) => suitOfType(t) === "z") ||
        handTypes.some((t) => suitOfType(t) === "z");
      const off = handTypes.length -
        bySuit(suit) - (honors ? handTypes.filter((t) => suitOfType(t) === "z").length : 0);
      if (off <= 3) {
        const kind = honors ? "混一色" : "清一色";
        reads.push(`${kind}(${SUIT_LABEL[suit as keyof typeof SUIT_LABEL]})可${
          off ? `（他色手内${off}枚）` : ""
        }`);
      }
    }
  }

  // 対々和: every meld is a triplet/kan.
  if (melds.every((m) => m.kind !== "chi")) reads.push("対々和可");

  // チャンタ系: every meld carries a terminal/honor.
  if (melds.every((m) => m.tiles.some((tile) => yaochuu(tileType(tile))))) {
    reads.push(meldTypes.some((t) => suitOfType(t) === "z") ? "混全帯可" : "純全帯可");
  }

  // Run blocks realized so far: melded chi, or all three ranks in hand.
  const realized = new Set<string>(); // "suit:startRank"
  for (const m of melds) {
    if (m.kind !== "chi") continue;
    const start = Math.min(...m.tiles.map((tile) => rankOfType(tileType(tile))));
    realized.add(`${suitOfType(tileType(m.tiles[0]))}:${start}`);
  }
  for (const s of SUITS) {
    for (let n = 1; n <= 7; n++) {
      const b = SUIT_BASE[s] + n - 1;
      if (handCount[b] && handCount[b + 1] && handCount[b + 2]) realized.add(`${s}:${n}`);
    }
  }
  for (let n = 1; n <= 7; n++) {
    const suits = SUITS.filter((s) => realized.has(`${s}:${n}`));
    if (suits.length >= 2) reads.push(`三色(${n}${n + 1}${n + 2})見込み`);
  }
  for (const s of SUITS) {
    const blocks = [1, 4, 7].filter((n) => realized.has(`${s}:${n}`)).length;
    if (blocks >= 2) reads.push(`一通(${SUIT_LABEL[s]})見込み`);
  }

  return reads.length ? reads : ["役なし懸念（現状、確定役なし）"];
}
