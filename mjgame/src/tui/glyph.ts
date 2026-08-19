// Tile glyphs and colors. Every tile cell is EXACTLY two terminal columns.
//
// The pin table is the load-bearing decision here. mjrender's `typeGlyph` uses
// ①..⑨ (U+2460), which is East-Asian **Ambiguous**: one column in most
// terminals, two when "treat ambiguous as wide" is on. Either way it disagrees
// with somebody, and a one-column pin tile shears the whole board. ㈠..㈨
// (U+3220 PARENTHESIZED IDEOGRAPH ONE) is unambiguously Wide, so it is used
// here instead. Man/sou/honors are Wide in both tables.
//
// Aka fives are shown by COLOR (bright red bold), never by a 赤 prefix, which
// would make the cell four columns wide.

import type { Tile } from "mjrender/model.ts";
import { rankOfType, suitOfType, tileType } from "mjrender/tiles.ts";
import type { Span } from "./screen.ts";
import { sp } from "./screen.ts";
import { SGR } from "./term.ts";

export type GlyphMode = "kanji" | "ascii";

const MAN = ["一", "二", "三", "四", "五", "六", "七", "八", "九"];
const PIN = ["㈠", "㈡", "㈢", "㈣", "㈤", "㈥", "㈦", "㈧", "㈨"]; // U+3220..U+3228, Wide
const SOU = ["１", "２", "３", "４", "５", "６", "７", "８", "９"]; // fullwidth digits, Wide
const HONOR = ["東", "南", "西", "北", "白", "發", "中"];

/** Suit colors. Honors are bright white so they stand out against the rivers. */
const SUIT_SGR = {
  m: SGR.yellow,
  p: SGR.cyan,
  s: SGR.green,
  z: SGR.brightWhite,
} as const;

const AKA_SGR = "1;91"; // bold bright red

export interface GlyphOpts {
  mode: GlyphMode;
  /** Tile ids that are red fives under the active ruleset (雀鬼会: 52, 53). */
  aka: ReadonlySet<Tile>;
}

/** Two-column text for a tile *type* (no aka distinction). */
export function typeText(type: number, mode: GlyphMode): string {
  const suit = suitOfType(type);
  const r = rankOfType(type);
  if (suit === "z") return HONOR[r - 1]; // honors stay kanji in both modes
  if (mode === "ascii") return `${r}${suit}`;
  return suit === "m" ? MAN[r - 1] : suit === "p" ? PIN[r - 1] : SOU[r - 1];
}

/** Two-column text for a tile *id*; aka fives read `0p` in ascii mode. */
export function tileText(id: Tile, o: GlyphOpts): string {
  const type = tileType(id);
  if (o.mode === "ascii" && o.aka.has(id)) return `0${suitOfType(type)}`;
  return typeText(type, o.mode);
}

export function typeSgr(type: number, extra = ""): string {
  const base = SUIT_SGR[suitOfType(type)];
  return extra === "" ? base : `${extra};${base}`;
}

export function tileSgr(id: Tile, o: GlyphOpts, extra = ""): string {
  if (o.aka.has(id)) return extra === "" ? AKA_SGR : `${extra};${AKA_SGR}`;
  return typeSgr(tileType(id), extra);
}

/** A styled two-column span for one tile id. */
export function tileSpan(id: Tile, o: GlyphOpts, extra = ""): Span {
  return sp(tileText(id, o), tileSgr(id, o, extra));
}

/** A styled two-column span for one tile type. */
export function typeSpan(type: number, o: GlyphOpts, extra = ""): Span {
  return sp(typeText(type, o.mode), typeSgr(type, extra));
}

// ---------------------------------------------------------------------------
// River markers
// ---------------------------------------------------------------------------

/**
 * One-column suffix markers. ASCII on purpose: ▽ (U+25BD) and → (U+2192) are
 * East-Asian *Ambiguous*, so a river built from them is one column wider on
 * some terminals than on others — exactly the tearing this module exists to
 * prevent. Meaning is documented in the help overlay.
 */
export const MARK = {
  none: " ",
  tsumogiri: "'", // ツモ切り
  riichi: "*", // リーチ宣言牌
  called: "^", // 鳴かれた
} as const;

export const MARK_SGR = {
  none: "",
  tsumogiri: SGR.gray,
  riichi: SGR.brightRed,
  called: SGR.gray,
} as const;

/** Wind name for a wind tile type (27..30). */
export const WINDS = ["東", "南", "西", "北"];

/** Relative-seat labels used across the panels. */
export const REL_LABEL = ["自分", "下家", "対面", "上家"];
