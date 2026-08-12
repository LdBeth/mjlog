// Rule configuration and the 雀鬼流 / 雀鬼会 (dojo) preset.
//
// Sources for the preset:
//   https://ja.wikipedia.org/wiki/雀鬼流
//   https://m4complex.com/paioku/archives/1357
//   https://scrapbox.io/kayato/雀鬼流のルール
//
// Where the sources are silent (一発/裏ドラ, ウマ magnitudes, 評価点 schedule),
// the value here is a documented default, not a sourced fact — see the plan's
// "Open items" section.

import type { Tile } from "mjrender/model.ts";
import { AKA_5P } from "./tiles.ts";

export interface RuleConfig {
  // --- match shape ---
  hanchan: boolean; // 東南戦
  startScore: number; // 30000持ち
  returnScore: number; // 30000返し
  uma: [number, number, number, number];
  truncateSub1000: boolean; // 1000点未満切り捨て (at match end)
  westEntry: boolean; // 西入
  dealerRepeatOnTenpai: boolean; // 親テンパイ連荘
  agariYame: boolean; // オーラス親和了やめ
  boxUnderContinues: boolean; // 箱下続行

  // --- dora ---
  akaIds: ReadonlySet<Tile>;
  ippatsu: boolean;
  uradora: boolean;
  kanDora: boolean;
  kanUra: boolean;

  // --- yaku policy ---
  kuitan: boolean; // 喰いタン
  kazoeYakuman: boolean; // 数え役満 (dojo: false ⇒ 13han+ = 三倍満)
  ninhou: boolean; // 人和 (dojo: false)
  tenhouChiihou: boolean; // 天和/地和
  nagashiMangan: boolean; // 流し満貫
  kiriageMangan: boolean; // 切り上げ満貫
  doubleWindFu: 2 | 4; // 連風牌の雀頭符

  // --- draws & multi-ron ---
  kyuushuKyuuhai: boolean; // 九種九牌 (dojo: false)
  suufonRenda: boolean; // 四風連打 (dojo: false)
  sanchahouDraw: boolean; // 三家和 → 流局
  suukaikanDraw: boolean; // 四開槓 → 流局
  suuchaRiichiDraw: boolean; // 四人立直 → 流局
  doubleRon: boolean; // false ⇒ 頭ハネ
  notenPenaltyTotal: number; // 罰符 total (3000)
}

/** The 雀鬼会 preset. */
export const JANKI: RuleConfig = {
  hanchan: true,
  startScore: 30000,
  returnScore: 30000,
  uma: [20, 10, 0, 0], // bonus only — no minus uma
  truncateSub1000: true,
  westEntry: false,
  dealerRepeatOnTenpai: true,
  agariYame: true,
  boxUnderContinues: true,

  akaIds: new Set(AKA_5P), // 赤5筒 ×2
  ippatsu: true,
  uradora: true,
  kanDora: true,
  kanUra: true,

  kuitan: true,
  kazoeYakuman: false,
  ninhou: false,
  tenhouChiihou: true,
  nagashiMangan: true,
  kiriageMangan: false,
  doubleWindFu: 2,

  kyuushuKyuuhai: false,
  suufonRenda: false,
  sanchahouDraw: true,
  suukaikanDraw: true,
  suuchaRiichiDraw: true,
  doubleRon: false, // 頭ハネ
  notenPenaltyTotal: 3000,
};

// ---------------------------------------------------------------------------
// Dojo penalty configuration
// ---------------------------------------------------------------------------

/** 評価点マイナス weight classes. The real dojo's schedule is not published. */
export const PENALTY = {
  light: 1, // 長考, 第一打字牌切り
  medium: 3, // 不聴時ドラ切り, 後付け, 明槓
  heavy: 5, // 立直禁止違反, 役満関連牌切り
  forfeit: 10, // 和了放棄
} as const;

export interface DojoConfig {
  enabled: boolean;
  /** Run the approximate (judgment-requiring) rules. Off for RL training. */
  tierB: boolean;
  /** 打牌 norm; over this is 長考. */
  thinkLimitMs: number;
  /** Over this is 大長考. */
  thinkHardMs: number;
  /** Hesitating this long over a call prompt counts as 腰. */
  koshiMs: number;
  /** Whether the CPU policies try to play dojo-clean. */
  aiObeysRules: boolean;
  /** Per-rule 評価点 overrides, keyed by RuleId. */
  weights: Record<string, number>;
}

export const DOJO_DEFAULT: DojoConfig = {
  enabled: true,
  tierB: true,
  thinkLimitMs: 3000,
  thinkHardMs: 4000,
  koshiMs: 1200,
  aiObeysRules: true,
  weights: {},
};

/** Config for headless self-play: deterministic rules only, no timing signals. */
export const DOJO_HEADLESS: DojoConfig = {
  ...DOJO_DEFAULT,
  tierB: false,
  aiObeysRules: false,
};
