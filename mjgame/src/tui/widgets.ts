// The individual panels. Every widget is a pure function of `Ctx` returning
// `Line[]` (or one `Line`); nothing here touches the Screen or the game engine,
// which is what makes the layout testable and the app loop small.

import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import type { DangerLevel } from "mjrender/danger.ts";
import { roundName } from "mjrender/tiles.ts";
import { doraFromIndicatorType, tileType } from "mjrender/tiles.ts";
import type { Observation } from "../observe.ts";
import { anyFuriten } from "../table.ts";
import type { Action, Seat, Violation } from "../types.ts";
import type { Line, Span } from "./screen.ts";
import { lineWidth, sp } from "./screen.ts";
import type { GlyphOpts } from "./glyph.ts";
import { MARK, MARK_SGR, REL_LABEL, tileSpan, typeSpan, typeText } from "./glyph.ts";
import { padEnd, SGR, truncate, width } from "./term.ts";

export type Phase = "idle" | "turn" | "claim" | "over";

export type Overlay =
  | { kind: "help" }
  | { kind: "danger" }
  | { kind: "kan"; options: Action[] }
  /** Disambiguation between equally-shaped calls (chi shapes, or pon pairs
   *  that differ only in whether they spend an aka five). */
  | { kind: "chi"; options: Action[]; title: string }
  | { kind: "call"; openedAt: number }
  | { kind: "quit" }
  | { kind: "text"; title: string; body: Line[]; footer: string };

export interface Ctx {
  obs: Observation | null;
  glyph: GlyphOpts;
  /** Absolute seat → display name. */
  names: string[];
  /** Hand tiles in display order; the drawn tile is last and set apart. */
  slots: Tile[];
  drawnIndex: number;
  cursor: number;
  /** Per slot: is discarding it legal right now (respecting armed riichi)? */
  selectable: boolean[];
  riichiArmed: boolean;
  phase: Phase;
  timer: TimerState;
  ledger: Violation[];
  log: string[];
  message: string;
  /** How many 6-tile rows each river gets at the current terminal height. */
  riverRows: number;
  claim: { tile: Tile; from: Seat } | null;
}

/** Countdown state: a per-turn allowance, then the one match-long bank. */
export interface TimerState {
  turnMs: number;
  bankMs: number;
  turnLeftMs: number;
  bankLeftMs: number;
  /** Time spent past base+bank. The clock never stops and never forces a play. */
  overMs: number;
}

function timerText(t: TimerState): Line {
  if (t.turnLeftMs > 0) {
    return [sp(`${Math.ceil(t.turnLeftMs / 1000)}s `, SGR.bold)];
  }
  if (t.bankLeftMs > 0) {
    return [sp("0s ", SGR.bold), sp(`+${(t.bankLeftMs / 1000).toFixed(1)}s `, "1;91")];
  }
  // Overtime. Take as long as you like — it only costs 評価点.
  return [sp(`-${(t.overMs / 1000).toFixed(1)}s `, "1;91")];
}

const DIM = SGR.gray;
const SEP = sp(" │ ", DIM);

const LEVEL_SGR: Record<DangerLevel, string> = {
  "安全": SGR.green,
  "危険度低": SGR.brightWhite,
  "危険度中": SGR.yellow,
  "危険度高": "1;91",
};

const MELD_LABEL: Record<Meld["kind"], string> = {
  chi: "チー",
  pon: "ポン",
  daiminkan: "大明槓",
  shouminkan: "加槓",
  ankan: "暗槓",
  nuki: "抜き",
};

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

/** 1-based placements for a relative score array; ties break by absolute seat. */
export function placements(obs: Observation): number[] {
  const order = obs.scores
    .map((s, rel) => ({ rel, s, abs: (obs.seat + rel) % 4 }))
    .sort((a, b) => b.s - a.s || a.abs - b.abs);
  const out = [0, 0, 0, 0];
  order.forEach((o, i) => (out[o.rel] = i + 1));
  return out;
}

export function absOf(obs: Observation, rel: number): Seat {
  return ((obs.seat + rel) % 4) as Seat;
}

/** Pad a line out to `w` columns so panel backgrounds stay rectangular. */
export function padLine(l: Line, w: number): Line {
  const gap = w - lineWidth(l);
  return gap > 0 ? [...l, sp(" ".repeat(gap))] : l;
}

/** A box-drawn frame around pre-rendered lines, sized by display width. */
export function boxed(lines: Line[], sgr = DIM): Line[] {
  const inner = Math.max(...lines.map(lineWidth), 0);
  const bar = "─".repeat(inner + 2);
  return [
    [sp("┌" + bar + "┐", sgr)],
    ...lines.map((l) => [sp("│ ", sgr), ...padLine(l, inner), sp(" │", sgr)]),
    [sp("└" + bar + "┘", sgr)],
  ];
}

export function meldSpans(melds: readonly Meld[], g: GlyphOpts): Span[] {
  if (melds.length === 0) return [sp("―", DIM)];
  const out: Span[] = [];
  melds.forEach((m, i) => {
    if (i > 0) out.push(sp(" "));
    out.push(sp("[" + MELD_LABEL[m.kind], DIM));
    // An ankan is face-down at the ends; show it dimmed rather than hidden,
    // since the player already knows their own and CPU ankan are public info
    // the moment the dora indicator flips.
    const extra = m.kind === "ankan" ? SGR.dim : "";
    for (const t of m.tiles) out.push(tileSpan(t, g, extra));
    out.push(sp("]", DIM));
  });
  return out;
}

/**
 * River rows, 6 tiles per row like a real table. Each cell is 3 columns:
 * a 2-column tile plus a 1-column marker (see `MARK` for why it is ASCII).
 */
export function riverLines(
  entries: readonly RiverEntry[],
  g: GlyphOpts,
  maxRows: number,
  perRow = 6,
): Line[] {
  const cap = maxRows * perRow;
  // When the river outgrows the space we have, drop the oldest rows: the recent
  // discards are what the player is reading for danger.
  const shown = entries.length > cap ? entries.slice(entries.length - cap) : entries;
  const rows: Line[] = [];
  for (let i = 0; i < shown.length; i += perRow) {
    const line: Line = [];
    for (const e of shown.slice(i, i + perRow)) {
      const called = e.calledBy !== undefined;
      const mark = called
        ? MARK.called
        : e.riichiDeclare
        ? MARK.riichi
        : e.tsumogiri
        ? MARK.tsumogiri
        : MARK.none;
      const markSgr = called
        ? MARK_SGR.called
        : e.riichiDeclare
        ? MARK_SGR.riichi
        : e.tsumogiri
        ? MARK_SGR.tsumogiri
        : "";
      line.push(tileSpan(e.tile, g, called ? SGR.dim : ""));
      line.push(sp(mark, markSgr));
    }
    rows.push(line);
  }
  while (rows.length < maxRows) rows.push([]);
  return rows.slice(0, maxRows);
}

function scoreSgr(delta: number): string {
  return delta >= 30000 ? SGR.brightGreen : delta < 10000 ? SGR.brightRed : SGR.brightWhite;
}

/**
 * A draining countdown: this turn's allowance empties first, then the match
 * bank. Both share one bar so the moment you start eating into the bank is
 * unmissable — that is the only part you cannot get back.
 */
function timerBar(t: TimerState, cells = 8): Span {
  const frac = Math.max(0, Math.min(1, t.turnLeftMs / Math.max(1, t.turnMs)));
  const on = Math.round(frac * cells);
  if (t.turnLeftMs > 0) {
    const sgr = frac < 0.25 ? "1;91" : frac < 0.5 ? SGR.yellow : SGR.green;
    return sp("[" + "#".repeat(on) + "-".repeat(cells - on) + "]", sgr);
  }
  if (t.bankLeftMs > 0) {
    const bankFrac = Math.max(0, Math.min(1, t.bankLeftMs / Math.max(1, t.bankMs)));
    const bankOn = Math.round(bankFrac * cells);
    return sp("[" + "!".repeat(bankOn) + "-".repeat(cells - bankOn) + "]", "1;91");
  }
  // ASCII only: U+203E and friends are East-Asian Ambiguous and would tear the
  // header on terminals that render them wide.
  return sp("[" + "=".repeat(cells) + "]", "1;91");
}

// ---------------------------------------------------------------------------
// header
// ---------------------------------------------------------------------------

export function headerBar(ctx: Ctx, w: number): Line {
  const o = ctx.obs;
  if (!o) return [sp(padEnd(" mjgame ", w), SGR.bold)];

  const doraTypes = o.doraIndicators.map((id) => doraFromIndicatorType(tileType(id)));
  const left: Line = [
    sp(" "),
    sp(roundName(o.kyoku), SGR.bold),
    sp(o.honba > 0 ? `${o.honba}本場` : "", SGR.bold),
    SEP,
    sp(`${o.junme}巡目`),
    SEP,
    sp(`残り${o.wallRemaining}`),
    SEP,
    sp("ドラ "),
    ...doraTypes.map((t) => typeSpan(t, ctx.glyph)),
    SEP,
    sp(`供託 ${o.kyotaku * 1000} `),
  ];

  const right: Line = ctx.phase === "turn" || ctx.phase === "claim"
    ? [
      ...timerText(ctx.timer),
      timerBar(ctx.timer),
      sp(" "),
    ]
    : [];

  // The header sits inside the top border rule, so the slack between the two
  // halves is filled with the rule itself rather than blanked out.
  const gap = w - lineWidth(left) - lineWidth(right);
  return gap > 0 ? [...left, sp("─".repeat(gap), DIM), ...right] : left;
}

// ---------------------------------------------------------------------------
// opponent seats
// ---------------------------------------------------------------------------

/** Header + river + melds for one relative seat (1 = 下家, 2 = 対面, 3 = 上家). */
export function seatPanel(ctx: Ctx, rel: number, w: number): Line[] {
  const o = ctx.obs;
  if (!o) return [];
  const ranks = placements(o);
  const abs = absOf(o, rel);
  const name = ctx.names[abs] ?? `P${abs}`;
  const riichi = o.riichi[rel];

  const head: Line = [
    sp(REL_LABEL[rel] + " ", SGR.bold),
    sp(`P${abs} `, DIM),
    sp(truncate(name, 10) + "  "),
    sp(String(o.scores[rel]), scoreSgr(o.scores[rel])),
    sp(`  ${ranks[rel]}位  `, DIM),
    riichi ? sp(`*リーチ(${o.riichiJunme[rel]})`, "1;91") : sp(""),
  ];

  const rows = riverLines(o.rivers[rel], ctx.glyph, ctx.riverRows);
  const river: Line[] = rows.map((r, i) => [sp(i === 0 ? "河 " : "   ", DIM), ...r]);
  const melds: Line = [sp("副露 ", DIM), ...meldSpans(o.melds[rel], ctx.glyph)];

  return [head, ...river, melds].map((l) => padLine(l, w));
}

// ---------------------------------------------------------------------------
// centre of the table
// ---------------------------------------------------------------------------

export function centerBlock(ctx: Ctx): Line[] {
  const o = ctx.obs;
  if (!o) return [];
  const roundWind = typeText(o.roundWind, "kanji");
  const l1: Line = [sp(`${roundWind} ${o.honba}本場`, SGR.bold)];
  // Riichi sticks: one marker per seat, in relative order starting from self.
  const sticks: Line = [sp("立直 ", DIM)];
  for (let r = 0; r < 4; r++) sticks.push(sp(o.riichi[r] ? "o" : ".", o.riichi[r] ? "1;91" : DIM));
  const l3: Line = [sp(`供託 ${o.kyotaku * 1000}`, DIM)];
  return boxed([l1, sticks, l3]);
}

// ---------------------------------------------------------------------------
// own seat
// ---------------------------------------------------------------------------

/** Name/score/melds line, own river, hand line, and the cursor caret line. */
export function ownPanel(ctx: Ctx, w: number): Line[] {
  const o = ctx.obs;
  if (!o) return [];
  const ranks = placements(o);
  const abs = o.seat;
  const winds = ["東", "南", "西", "北"];
  const head: Line = [
    sp("自分 ", SGR.bold),
    sp(`P${abs} `, DIM),
    sp(`${winds[o.seatWind - 27] ?? "?"}家 `),
    sp(String(o.scores[0]), scoreSgr(o.scores[0])),
    sp(`  ${ranks[0]}位   `, DIM),
    sp("副露 ", DIM),
    ...meldSpans(o.melds[0], ctx.glyph),
  ];

  // Own river, on one long row so it sits between the table centre and the
  // hand. 24 cells × 3 columns + the 6-column label fits the 80-column minimum.
  const river: Line = [
    sp("河    ", DIM),
    ...riverLines(o.rivers[0], ctx.glyph, 1, 24)[0],
  ];

  const hand: Line = [sp("手牌  ", DIM)];
  const caret: Line = [sp("      ")];
  const active = ctx.phase === "turn";
  ctx.slots.forEach((id, i) => {
    if (i === ctx.drawnIndex) {
      hand.push(sp("│ ", DIM));
      caret.push(sp("  "));
    }
    const dim = active && !ctx.selectable[i] ? SGR.dim : "";
    const sel = active && i === ctx.cursor;
    hand.push(tileSpan(id, ctx.glyph, sel ? SGR.reverse : dim));
    hand.push(sp(" "));
    caret.push(sp(sel ? "^^" : "  ", sel ? (ctx.riichiArmed ? "1;91" : SGR.brightCyan) : ""));
    caret.push(sp(" "));
  });
  if (ctx.drawnIndex >= 0 && ctx.drawnIndex === ctx.slots.length) hand.push(sp("│ ", DIM));

  const label: Line = ctx.drawnIndex >= 0
    ? [sp("   ツモ", DIM)]
    : ctx.phase === "claim"
    ? [sp("   (鳴き判断)", SGR.yellow)]
    : [];

  return [head, river, [...hand, ...label], caret].map((l) => padLine(l, w));
}

export function metricsLine(ctx: Ctx, w: number): Line {
  const o = ctx.obs;
  if (!o) return [];
  const live = new Map(o.ukeire.map((u) => [u.type, u.live]));
  const line: Line = [sp("〔", DIM)];
  if (o.shanten <= 0) {
    line.push(sp("聴牌", SGR.brightGreen), sp(" 待ち "));
    for (const t of o.waits) {
      line.push(typeSpan(t, ctx.glyph));
      line.push(sp(`${live.get(t) ?? 0} `, DIM));
    }
  } else {
    line.push(sp(`${o.shanten}向聴`, o.shanten <= 1 ? SGR.yellow : ""), sp(" 受入 "));
    const top = [...o.ukeire].sort((a, b) => b.live - a.live).slice(0, 6);
    for (const u of top) {
      line.push(typeSpan(u.type, ctx.glyph));
      line.push(sp(`${u.live} `, DIM));
    }
    const kinds = o.ukeire.length;
    const total = o.ukeire.reduce((a, b) => a + b.live, 0);
    line.push(sp(`(${kinds}種${total}枚) `, DIM));
  }
  line.push(sp(`ドラ${o.doraCount}`, o.doraCount > 0 ? SGR.brightYellow : DIM));
  line.push(sp("〕", DIM));

  const f = o.furiten;
  const fText = !anyFuriten(f) ? "振聴なし" : "振聴" +
    [f.permanent ? "(永続)" : "", f.temporary ? "(同巡)" : "", f.riichi ? "(リーチ後)" : ""]
      .filter((s) => s).join("");
  line.push(sp("  " + fText, anyFuriten(f) ? "1;91" : DIM));
  if (ctx.message) line.push(sp("  " + ctx.message, SGR.brightCyan));
  return padLine(line, w);
}

export function dangerRow(ctx: Ctx, w: number): Line {
  const o = ctx.obs;
  if (!o) return [];
  if (o.danger.size === 0) {
    return padLine([sp("危険  ", DIM), sp("リーチ・副露なし — 全牌安全", SGR.green)], w);
  }
  const order = [...o.danger.entries()].sort((a, b) =>
    rankLevel(b[1].level) - rankLevel(a[1].level) || a[0] - b[0]
  );
  const line: Line = [sp("危険  ", DIM)];
  for (const [type, d] of order) {
    const note = d.details[0]?.notes[0];
    const cell: Line = [
      typeSpan(type, ctx.glyph),
      sp(":", DIM),
      sp(note ? `${note}・` : "", DIM),
      sp(shortLevel(d.level), LEVEL_SGR[d.level]),
      sp("  "),
    ];
    if (lineWidth(line) + lineWidth(cell) > w - 4) {
      line.push(sp("…", DIM));
      break;
    }
    line.push(...cell);
  }
  return padLine(line, w);
}

function rankLevel(l: DangerLevel): number {
  return l === "危険度高" ? 3 : l === "危険度中" ? 2 : l === "危険度低" ? 1 : 0;
}

function shortLevel(l: DangerLevel): string {
  return l === "安全" ? "安全" : l.replace("危険度", "");
}

// ---------------------------------------------------------------------------
// penalty ledger
// ---------------------------------------------------------------------------

export function ledgerPanel(ctx: Ctx, w: number): Line[] {
  const line: Line = [];
  for (let s = 0; s < 4; s++) {
    const mine = ctx.ledger.filter((v) => v.seat === s);
    const pts = mine.reduce((a, v) => a + v.points, 0);
    line.push(sp(`P${s} `, DIM));
    line.push(sp(pts === 0 ? "0" : `-${pts}`, pts === 0 ? DIM : "1;91"));
    if (mine.length > 0) {
      const byLabel = new Map<string, number>();
      for (const v of mine) byLabel.set(v.label, (byLabel.get(v.label) ?? 0) + 1);
      const parts = [...byLabel].map(([l, n]) => (n > 1 ? `${l}×${n}` : l));
      line.push(sp(` [${parts.join(", ")}]`, SGR.yellow));
    }
    line.push(sp("  "));
  }
  const l = lineWidth(line) > w ? clipLine(line, w) : line;
  return [padLine(l, w)];
}

function clipLine(l: Line, w: number): Line {
  const out: Line = [];
  let n = 0;
  for (const s of l) {
    const room = w - 1 - n;
    if (room <= 0) break;
    const t = truncate(s.text, room);
    out.push(sp(t, s.sgr));
    n += width(t);
  }
  out.push(sp("…", DIM));
  return out;
}

// ---------------------------------------------------------------------------
// action bar
// ---------------------------------------------------------------------------

export function actionBar(ctx: Ctx, w: number): Line {
  if (ctx.riichiArmed) {
    const t = " リーチ宣言中 — 切る牌を選んで Enter   Esc 解除 ";
    return [sp(padEnd(t, w), "1;97;41")];
  }
  if (ctx.phase === "claim") {
    const has = (k: Action["t"]) => ctx.obs?.legal.some((a) => a.t === k) ?? false;
    const parts = [
      has("ron") ? "y/Enter 和了" : "",
      has("pon") ? "p ポン" : "",
      has("chi") ? "c チー" : "",
      has("daiminkan") ? "n カン" : "",
      "Esc 見送り",
      "d 危険",
      "? ヘルプ",
    ].filter((s) => s);
    return [sp(padEnd(" " + parts.join("  "), w), "1;30;103")];
  }
  if (ctx.phase !== "turn") {
    return [sp(padEnd(" 待機中…   ? ヘルプ  q 終了", w), DIM)];
  }
  const has = (k: Action["t"]) => ctx.obs?.legal.some((a) => a.t === k) ?? false;
  const canRiichi = ctx.obs?.legal.some((a) => a.t === "discard" && a.riichi) ?? false;
  const canKan = has("ankan") || has("kakan");
  const parts = [
    "←→ 選択",
    "Enter 打牌",
    "t ツモ切り",
    canRiichi ? "r リーチ" : "",
    canKan ? "k カン" : "",
    has("tsumo") ? "a ツモ和了" : "",
    "d 危険",
    "s 記録",
    "? ヘルプ",
  ].filter((s) => s);
  return [sp(padEnd(" " + parts.join("  "), w), SGR.onGray)];
}

// ---------------------------------------------------------------------------
// overlays
// ---------------------------------------------------------------------------

export interface OverlayView {
  title: string;
  body: Line[];
  footer: string;
}

export function overlay(ctx: Ctx, o: Overlay): OverlayView {
  switch (o.kind) {
    case "help":
      return { title: "ヘルプ", body: helpBody(), footer: "任意のキーで閉じる" };
    case "danger":
      return { title: "危険牌の根拠", body: dangerBody(ctx), footer: "任意のキーで閉じる" };
    case "kan":
      return {
        title: "カン",
        body: o.options.map((a, i) => [
          sp(`${i + 1}. `, SGR.brightCyan),
          ...kanLabel(a, ctx),
        ]),
        footer: "1-9 選択   Esc 取消",
      };
    case "chi":
      return {
        title: o.title,
        body: o.options.map((a, i) => [
          sp(`${i + 1}. `, SGR.brightCyan),
          ...(a.t === "chi" || a.t === "pon" ? a.tiles.map((t) => tileSpan(t, ctx.glyph)) : []),
          sp(" + ", DIM),
          ...(a.t === "chi" || a.t === "pon" ? [tileSpan(a.called, ctx.glyph, SGR.reverse)] : []),
        ]),
        footer: "1-9 選択   Esc 取消",
      };
    case "call":
      return { title: "鳴きますか", body: callBody(ctx), footer: barText(ctx) };
    case "quit":
      return {
        title: "終了しますか",
        body: [[sp("対局を破棄して終了します。")]],
        footer: "y 終了   Esc 続行",
      };
    case "text":
      return { title: o.title, body: o.body, footer: o.footer };
  }
}

function barText(ctx: Ctx): string {
  const has = (k: Action["t"]) => ctx.obs?.legal.some((a) => a.t === k) ?? false;
  return [
    has("ron") ? "y 和了" : "",
    has("pon") ? "p ポン" : "",
    has("chi") ? "c チー" : "",
    has("daiminkan") ? "n カン" : "",
    "Esc 見送り",
  ].filter((s) => s).join("   ");
}

function callBody(ctx: Ctx): Line[] {
  const o = ctx.obs;
  const body: Line[] = [];
  if (ctx.claim && o) {
    const abs = ctx.claim.from;
    const rel = (abs - o.seat + 4) % 4;
    body.push([
      sp(`${REL_LABEL[rel]} P${abs} の打 `),
      tileSpan(ctx.claim.tile, ctx.glyph, SGR.reverse),
    ]);
  }
  body.push([]);
  const has = (k: Action["t"]) => o?.legal.some((a) => a.t === k) ?? false;
  if (has("ron")) body.push([sp("和了できます (y)", SGR.brightGreen)]);
  if (has("pon")) body.push([sp("ポン可 (p)")]);
  if (has("chi")) body.push([sp("チー可 (c)")]);
  if (has("daiminkan")) body.push([sp("大明槓可 (n) — 道場ルールでは禁じ手", SGR.yellow)]);
  return body;
}

function kanLabel(a: Action, ctx: Ctx): Line {
  if (a.t === "ankan") return [sp("暗槓 "), typeSpan(a.type, ctx.glyph)];
  if (a.t === "kakan") return [sp("加槓 "), tileSpan(a.tile, ctx.glyph)];
  if (a.t === "daiminkan") return [sp("大明槓 "), tileSpan(a.called, ctx.glyph)];
  return [sp(a.t)];
}

function dangerBody(ctx: Ctx): Line[] {
  const o = ctx.obs;
  if (!o || o.danger.size === 0) return [[sp("現在、警戒すべき相手はいません。", SGR.green)]];
  const body: Line[] = [];
  const order = [...o.danger.entries()].sort((a, b) =>
    rankLevel(b[1].level) - rankLevel(a[1].level) || a[0] - b[0]
  );
  for (const [type, d] of order) {
    body.push([
      typeSpan(type, ctx.glyph),
      sp("  "),
      sp(d.level, LEVEL_SGR[d.level]),
      sp(`   対象 ${d.seats.map((s) => "P" + s).join(",")}`, DIM),
    ]);
    for (const det of d.details) {
      const tag = det.kind === "furo"
        ? `P${det.seat}副露${det.openMeldCount ?? 0}`
        : `P${det.seat}`;
      body.push([
        sp("   ┗ ", DIM),
        sp(tag + " ", SGR.brightCyan),
        sp(det.level + " ", LEVEL_SGR[det.level]),
        sp(det.notes.join("・"), DIM),
      ]);
    }
  }
  return body;
}

function helpBody(): Line[] {
  const rows: Array<[string, string]> = [
    ["← → / h l", "手牌カーソル移動"],
    ["1-9 0", "位置を直接指定 (0 = ツモ牌)"],
    ["Enter / Space", "カーソルの牌を打牌"],
    ["t", "ツモ切り"],
    ["r", "リーチ宣言を準備 (Esc で解除)"],
    ["k", "カンメニュー"],
    ["a", "ツモ和了 / ロン"],
    ["d", "危険牌の根拠を表示"],
    ["s", "現局面をログ欄にダンプ"],
    ["?", "このヘルプ"],
    ["q / Ctrl-C", "終了"],
    ["", ""],
    ["鳴き選択", "p ポン / c チー / n カン / y 和了 / Esc 見送り"],
    ["", ""],
    [
      "河の記号",
      `${MARK.tsumogiri} ツモ切り  ${MARK.riichi} リーチ宣言牌  ${MARK.called} 鳴かれた`,
    ],
    ["牌の色", "萬=黄 筒=シアン 索=緑 字=白 赤5筒=赤"],
  ];
  return rows.map(([k, v]) => k === "" ? [] : [sp(padEnd(k, 16), SGR.brightCyan), sp(v)]);
}
