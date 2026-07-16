// Tile encoding, Japanese notation, red-fives, and dora successor logic.

import type { Meld, Tile } from "./model.ts";

/** Suit of a tile type (0..33): 'm' | 'p' | 's' | 'z' (honors). */
export type Suit = "m" | "p" | "s" | "z";

export function tileType(id: Tile): number {
  return id >> 2;
}

export function suitOfType(type: number): Suit {
  if (type < 9) return "m";
  if (type < 18) return "p";
  if (type < 27) return "s";
  return "z";
}

/** Rank within suit: 1..9 for m/p/s, 1..7 for honors (E,S,W,N,白,發,中). */
export function rankOfType(type: number): number {
  if (type < 27) return (type % 9) + 1;
  return type - 26; // 27→1 ... 33→7
}

/** The three red-five tile ids (the `%4==0` copy of 5m/5p/5s). */
const AKA_IDS = new Set<number>([16, 52, 88]);

export function isAka(id: Tile): boolean {
  return AKA_IDS.has(id);
}

const MAN = ["一", "二", "三", "四", "五", "六", "七", "八", "九"];
const PIN = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"];
const SOU = ["１", "２", "３", "４", "５", "６", "７", "８", "９"];
const HONOR = ["東", "南", "西", "北", "白", "發", "中"];

export const WIND = ["東", "南", "西", "北"];

/** Round name from a kyoku index, e.g. 0→"東1局", 4→"南1局". */
export function roundName(kyoku: number): string {
  return `${WIND[Math.floor(kyoku / 4) % 4]}${(kyoku % 4) + 1}局`;
}

/** Glyph for a tile *type* (no aka distinction). */
export function typeGlyph(type: number): string {
  const suit = suitOfType(type);
  const r = rankOfType(type);
  switch (suit) {
    case "m":
      return MAN[r - 1];
    case "p":
      return PIN[r - 1];
    case "s":
      return SOU[r - 1];
    case "z":
      return HONOR[r - 1];
  }
}

/** Glyph for a tile *id*, marking aka fives with a 赤 prefix when rules enable aka. */
export function tileGlyph(id: Tile, aka = true): string {
  const g = typeGlyph(tileType(id));
  return aka && isAka(id) ? "赤" + g : g;
}

/** Short ascii notation, e.g. "5m", "3p", "東", "0p" for aka. */
export function tileAscii(id: Tile, aka = true): string {
  const type = tileType(id);
  const suit = suitOfType(type);
  if (suit === "z") return typeGlyph(type);
  const r = aka && isAka(id) ? 0 : rankOfType(type);
  return `${r}${suit}`;
}

/**
 * Dora tile *type* pointed to by an indicator *type*.
 * Suits wrap 9→1; winds cycle E→S→W→N→E; dragons 白→發→中→白.
 */
export function doraFromIndicatorType(indType: number): number {
  const suit = suitOfType(indType);
  if (suit !== "z") {
    const base = suit === "m" ? 0 : suit === "p" ? 9 : 18;
    const r = rankOfType(indType); // 1..9
    return base + (r % 9); // 9→0 offset ⇒ rank 1
  }
  // honors
  if (indType <= 30) {
    // winds 27..30
    return indType === 30 ? 27 : indType + 1;
  }
  // dragons 31..33
  return indType === 33 ? 31 : indType + 1;
}

/**
 * Render a concealed hand (list of tile ids) as suit-grouped Japanese notation,
 * e.g. "一二三 ④⑤⑥ ３４５ 東東 白白白". Sorted within each suit.
 */
export function renderHand(concealed: Tile[], melds: Meld[] = [], aka = true): string {
  // Sorting by id already orders by (type, id) — type = id >> 2 is monotone in
  // id — and puts the suits in m/p/s/z order, so one global sort replaces the
  // per-suit grouping: each consecutive same-suit run IS a suit group.
  const sorted = [...concealed].sort((a, b) => a - b);
  const parts: string[] = [];
  for (let i = 0; i < sorted.length;) {
    const suit = suitOfType(tileType(sorted[i]));
    let body = "";
    while (i < sorted.length && suitOfType(tileType(sorted[i])) === suit) {
      body += tileGlyph(sorted[i++], aka);
    }
    parts.push(body);
  }
  for (const m of melds) parts.push(renderMeld(m, aka));
  return parts.join(" ");
}

const MELD_LABEL: Record<Meld["kind"], string> = {
  chi: "チー",
  pon: "ポン",
  daiminkan: "大明槓",
  shouminkan: "加槓",
  ankan: "暗槓",
  nuki: "抜き",
};

export function renderMeld(m: Meld, aka = true): string {
  // m.tiles is sorted ascending (Meld invariant), which is the (type, id) order.
  const body = m.tiles.map((id) => tileGlyph(id, aka)).join("");
  return `[${MELD_LABEL[m.kind]}${body}]`;
}
