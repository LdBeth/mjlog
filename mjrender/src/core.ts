// Reusable core: the full query API over a parsed game. The CLI subcommands
// and the MCP server are both thin wrappers over THESE functions.

import { loadXml } from "./load.ts";
import { parseGame } from "./parse.ts";
import { formatInstruction, renderGameAnnotated } from "./render.ts";
import { owariRows, placements } from "./scoring.ts";
import { renderSnapshot } from "./snapshot.ts";
import { replayTo } from "./state.ts";
import { roundName, tileGlyph, typeGlyph, WIND } from "./tiles.ts";
import type { AgariResult, Beat, Game, RenderOptions } from "./model.ts";
import { limitName, yakuName } from "./yaku.ts";

export async function loadGame(path: string): Promise<Game> {
  const game = parseGame(await loadXml(path));
  if (game.rules.sanma) {
    throw new Error("三人打ち（サンマ）はこのバージョンでは未対応です。");
  }
  return game;
}

function fullOpts(opts: Partial<RenderOptions>): RenderOptions {
  return { hands: opts.hands ?? "key", snapshots: opts.snapshots ?? "none" };
}

/** Path → full transcript (back-compat entry point). */
export async function render(path: string, opts: Partial<RenderOptions> = {}): Promise<string> {
  return renderGame(await loadGame(path), opts);
}

export function renderGame(game: Game, opts: Partial<RenderOptions> = {}): string {
  return renderGameAnnotated(game, fullOpts(opts)).text;
}

/**
 * One kyoku, self-contained: format preamble + game header + that round.
 * Anchor IDs inside are game-global (the whole game is rendered and sliced),
 * so they agree with list_anchors / get_snapshot.
 */
export function renderKyoku(
  game: Game,
  selector: string,
  opts: Partial<RenderOptions> = {},
): string {
  const indices = resolveKyoku(game, selector);
  const r = renderGameAnnotated(game, fullOpts(opts));
  return [r.sections.header, ...indices.map((i) => r.sections.rounds[i])].join("\n");
}

// Appended to the outline's format legend: what this view omits and how the
// commentator is expected to drill down (kyoku render, snapshots, draft tools).
const OUTLINE_NOTE = `■アウトライン表示（この出力について）
これは対局全体のアウトライン: 局ヘッダ・結果・解説ポイント一覧のみで、打牌の記録は含まない。
・各局の全打牌・★比較・危険度根拠は局単位で取得する（MCPツール mj_render_kyoku / CLI kyoku コマンド）。
  牌表記などの凡例（この牌譜の読み方）も局の出力の冒頭に付属する
・〔解説ポイント#N: …〕（N巡・P#）は各局の解説対象一覧。mj_get_snapshot にアンカーID（または
  局+巡目）を渡すと、その時点の全員の河・点数・手牌つき盤面を再現できる
・特にリーチ宣言・聴牌が絡む局面（リーチ判断/押し引きアンカー）は、解説を書く前に必ず
  mj_get_snapshot で盤面を確認すること（盤面確認なしで評価しない）
・解説は mj_add_comment で保存し（1呼び出しで複数アンカーまとめて可）、最後に
  mj_weave_commentary で完成稿を書き出す
・局の出力中の★行（注目の打牌・鳴き）には、mj_add_note（局+巡目+席で指定）で任意の一言を
  1件ずつ添えられる
`;

// Outline keeps: kyoku header, 点況/逆転条件, and the condensed result block
// (◆和了/◆流局 with hand, yaku/points, score movements).
const OUTLINE_KEEP =
  /^(【|◆和了|◆流局)|^\s+(点況:|和了手:|役:|ドラ表示:|点棒:|P\d\(\d位\) 逆転条件:)/;

/**
 * Crude whole-game outline: per kyoku only the header (with 点況/逆転条件),
 * the condensed result block, and the round's commentary-anchor index — no
 * per-turn lines (those come from renderKyoku, one round at a time). Kept
 * lines are sliced verbatim from the full render, so they are byte-identical
 * with renderKyoku output and anchor ids stay game-global.
 */
export function renderOutline(game: Game): string {
  const r = renderGameAnnotated(game, fullOpts({}));
  // Skip the format legend (everything before the ==== game/players block) —
  // renderKyoku carries it, and the outline is always read alongside those.
  const gameBlock = r.sections.header.slice(r.sections.header.indexOf("=".repeat(48)));
  const out: string[] = [gameBlock, OUTLINE_NOTE];
  r.sections.rounds.forEach((section, i) => {
    out.push(...section.split("\n").filter((l) => OUTLINE_KEEP.test(l)));
    // 終局総括 is anchored to the last round but rendered inline in the owari
    // section (kept verbatim below) — listing it here would duplicate it.
    for (const b of r.beats.filter((b) => b.round === i && b.kind !== "終局総括")) {
      out.push(
        `〔解説ポイント#${b.id}: ${b.kind}｜${b.topic}〕` +
          `（${b.junme}巡${b.seat !== undefined ? `・P${b.seat}` : ""}）`,
      );
    }
    out.push("―".repeat(20), "");
  });
  if (r.sections.owari) out.push(r.sections.owari);
  return out.join("\n");
}

export function roundLabel(game: Game, roundIndex: number): string {
  const round = game.rounds[roundIndex];
  return `${roundName(round.kyoku)}${round.honba}本場`;
}

/**
 * Resolve a kyoku selector to round indices. Accepts a 0-based round index
 * ("6"), or wind+number with optional honba: "S3", "南3", "E1.2" / "東1.2"
 * (= 東1局2本場). Wind letters: E/S/W/N or 東/南/西/北.
 */
export function resolveKyoku(game: Game, selector: string): number[] {
  const sel = selector.trim();
  if (/^\d+$/.test(sel)) {
    const i = Number(sel);
    if (i < 0 || i >= game.rounds.length) {
      throw new Error(`round index ${i} out of range (0..${game.rounds.length - 1})`);
    }
    return [i];
  }
  const m = sel.match(/^([ESWNeswn東南西北])(\d)(?:\.(\d+))?$/);
  if (!m) throw new Error(`invalid kyoku selector "${selector}" (e.g. "S3", "東1", "E1.2", "6")`);
  const windIdx = "ESWN".indexOf(m[1].toUpperCase()) >= 0
    ? "ESWN".indexOf(m[1].toUpperCase())
    : WIND.indexOf(m[1]);
  const kyoku = windIdx * 4 + (Number(m[2]) - 1);
  const honba = m[3] === undefined ? undefined : Number(m[3]);
  const hits = game.rounds.flatMap((r, i) =>
    r.kyoku === kyoku && (honba === undefined || r.honba === honba) ? [i] : []
  );
  if (hits.length === 0) throw new Error(`no round matches "${selector}"`);
  return hits;
}

const beatCache = new WeakMap<Game, Beat[]>();

/** All commentary beats (anchor IDs) of the game, in transcript order. */
export function listAnchors(game: Game): Beat[] {
  let beats = beatCache.get(game);
  if (!beats) {
    beats = renderGameAnnotated(game, fullOpts({})).beats;
    beatCache.set(game, beats);
  }
  return beats;
}

/** Human/LLM-readable one-line-per-anchor table. */
export function anchorTable(game: Game): string {
  return listAnchors(game).map((b) =>
    `#${b.id}\t${b.kind}\t${roundLabel(game, b.round)}\t${b.junme}巡` +
    `${b.seat !== undefined ? `\tP${b.seat}` : "\t"}\t${b.topic}`
  ).join("\n");
}

// ---- commentary weaving ----

/** One commentary entry, keyed by transcript anchor id. */
export interface WeaveComment {
  anchor: number;
  text: string;
}

/** An optional one-liner for a ★ line, addressed by game position. */
export interface StarNote {
  kyoku: string; // round selector, same grammar as everywhere else
  junme: number;
  seat: number;
  text: string;
}

export type AnchorComments = Record<string, string> | WeaveComment[];

/** Full weave input: anchor comments plus optional ★-line notes. */
export interface WeaveInput {
  anchors?: AnchorComments;
  notes?: StarNote[];
}

export interface WeaveOptions extends Partial<RenderOptions> {
  /** Anchors with no comment: keep the placeholder line (default) or strip it. */
  missing?: "keep" | "strip";
}

export interface WeaveResult {
  text: string;
  filled: number[];
  missing: number[];
  notesApplied: number;
}

const ANCHOR_LINE = /^〔解説ポイント#(\d+): ([^｜]+)｜/;

/**
 * Splice anchor-keyed commentary (and optional ★-line notes) into a freshly
 * rendered transcript. This is the harness side of the commentary loop: the
 * LLM never reproduces fact lines (verbatim copying is where tile facts get
 * corrupted) — it hands over only {anchor id → comment} plus notes addressed
 * by kyoku + junme + seat, and every other line comes from the renderer.
 * Accepts a {"12": "…"} map, a [{anchor, text}] list, or {anchors, notes}.
 */
export function weaveCommentary(
  game: Game,
  input: AnchorComments | WeaveInput,
  opts: WeaveOptions = {},
): WeaveResult {
  const structured = !Array.isArray(input) && ("anchors" in input || "notes" in input);
  const comments = (structured ? (input as WeaveInput).anchors : input as AnchorComments) ?? {};
  const notes = (structured ? (input as WeaveInput).notes : undefined) ?? [];

  const entries = Array.isArray(comments)
    ? comments.map((c) => [c.anchor, c.text] as const)
    : Object.entries(comments).map(([k, v]) => [Number(k), v] as const);
  const byId = new Map<number, string>();
  for (const [id, text] of entries) {
    if (!Number.isInteger(id) || id < 1) throw new Error(`invalid anchor id "${id}"`);
    if (byId.has(id)) throw new Error(`duplicate comment for anchor #${id}`);
    if (typeof text !== "string" || !text.trim()) {
      throw new Error(`empty comment for anchor #${id}`);
    }
    byId.set(id, text.trim());
  }
  if (byId.size === 0 && notes.length === 0) {
    throw new Error("nothing to weave: provide anchor comments and/or ★ notes");
  }

  const r = renderGameAnnotated(game, fullOpts(opts));
  const unknown = [...byId.keys()].filter((id) => id > r.beats.length);
  if (unknown.length) {
    throw new Error(
      `unknown anchor id(s): ${unknown.map((i) => `#${i}`).join(" ")} — ` +
        `this game has #1..#${r.beats.length} (see anchors / mj_list_anchors)`,
    );
  }

  // Resolve each ★ note to a rendered line. When a seat acted twice in one
  // go-around (call + discard), the note goes after the LAST ★ line — the
  // discard that closed the turn.
  const noteAt = new Map<number, string[]>();
  for (const n of notes) {
    if (typeof n.text !== "string" || !n.text.trim()) {
      throw new Error(`empty ★ note for ${n.kyoku} ${n.junme}巡 P${n.seat}`);
    }
    const rIdx = uniqueRound(game, n.kyoku);
    const hits = r.stars.filter((s) =>
      s.round === rIdx && s.junme === n.junme && s.seat === n.seat
    );
    if (hits.length === 0) {
      const here = r.stars.filter((s) => s.round === rIdx)
        .map((s) => `${s.junme}巡P${s.seat}`).join(" ");
      throw new Error(
        `no ★ line for P${n.seat} at ${roundLabel(game, rIdx)} ${n.junme}巡` +
          (here ? ` — ★ sites in this kyoku: ${here}` : " — this kyoku has no ★ lines"),
      );
    }
    const site = hits[hits.length - 1];
    noteAt.set(site.line, [...noteAt.get(site.line) ?? [], n.text.trim()]);
  }

  const filled: number[] = [];
  const missing: number[] = [];
  const out: string[] = [];
  const lines = r.text.split("\n");
  // A ★ line owns its `┗ …` continuation lines; a pending note waits them out.
  let pending: string[] = [];
  const flush = (): void => {
    for (const n of pending) {
      const [first, ...rest] = n.split("\n");
      out.push(`  ◆一言: ${first}`, ...rest.map((l) => (l ? `    ${l}` : l)));
    }
    pending = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (pending.length && !line.startsWith("  ┗")) flush();
    const m = line.match(ANCHOR_LINE);
    if (!m) {
      out.push(line);
      pending.push(...noteAt.get(i) ?? []);
      continue;
    }
    const id = Number(m[1]);
    const text = byId.get(id);
    if (text === undefined) {
      missing.push(id);
      if ((opts.missing ?? "keep") === "keep") out.push(line);
      continue;
    }
    filled.push(id);
    const [first, ...rest] = text.split("\n");
    out.push(`◆解説（${m[2]}）: ${first}`, ...rest.map((l) => (l ? `  ${l}` : l)));
  }
  flush();
  // The woven document is for readers, not the commentator LLM — swap the
  // fill-mode instructions for the reader legend.
  const text = out.join("\n").replace(formatInstruction("fill"), () => formatInstruction("final"));
  return { text, filled, missing, notesApplied: notes.length };
}

/** Every ★ site of the game, in transcript order (for tooling and tests). */
export function listStarSites(game: Game): Array<{ round: number; junme: number; seat: number }> {
  return renderGameAnnotated(game, fullOpts({})).stars
    .map(({ round, junme, seat }) => ({ round, junme, seat }));
}

/** One-line confirmation of a weave (what was filled, what is still open). */
export function weaveSummary(r: WeaveResult, out?: string): string {
  const total = r.filled.length + r.missing.length;
  const notes = r.notesApplied ? ` / ★一言 ${r.notesApplied}件` : "";
  const miss = r.missing.length ? ` / 未記入: ${r.missing.map((i) => `#${i}`).join(" ")}` : "";
  return `${
    out ? `wrote ${out} — ` : ""
  }解説 ${r.filled.length}/${total} 箇所を織り込み${notes}${miss}`;
}

// ---- structured fact queries (each backs an MCP tool) ----

/** One round's start conditions. Scores are in points (×100 applied). */
export interface KyokuStart {
  label: string;
  roundIndex: number;
  dealer: number;
  honba: number;
  kyotaku: number;
  doraIndicator: string;
  seats: Array<{ seat: number; name: string; score: number; place: number }>;
}

export function kyokuStart(game: Game, selector: string): KyokuStart {
  const i = uniqueRound(game, selector);
  const round = game.rounds[i];
  const place = placements(round.startScores, game.rounds[0].dealer);
  return {
    label: roundLabel(game, i),
    roundIndex: i,
    dealer: round.dealer,
    honba: round.honba,
    kyotaku: round.kyotaku,
    doraIndicator: tileGlyph(round.firstDora, game.rules.aka),
    seats: [0, 1, 2, 3].map((s) => ({
      seat: s,
      name: game.players[s].name,
      score: round.startScores[s] * 100,
      place: place[s],
    })),
  };
}

/** One round's outcome(s) — multiple entries = double/triple ron. */
export type KyokuResult =
  | {
    type: "agari";
    who: number;
    fromWho: number;
    tsumo: boolean;
    winningTile: string;
    points: number; // 素点 (before honba adjustments)
    fu: number;
    limit: string | null; // 満貫/跳満/… or null below limit
    yaku: string[];
  }
  | { type: "ryuukyoku"; reason: string; tenpaiSeats: number[] };

export function kyokuResults(game: Game, selector: string): KyokuResult[] {
  const round = game.rounds[uniqueRound(game, selector)];
  return round.results.map((res) => {
    if (res.kind === "agari") {
      const a = res as AgariResult;
      return {
        type: "agari" as const,
        who: a.who,
        fromWho: a.fromWho,
        tsumo: a.who === a.fromWho,
        winningTile: tileGlyph(a.machi, game.rules.aka),
        points: a.points,
        fu: a.fu,
        limit: limitName(a.limit) || null,
        yaku: [
          ...a.yaku.map((y) => `${yakuName(y.id)}(${y.han})`),
          ...a.yakuman.map((id) => `${yakuName(id)}(役満)`),
        ],
      };
    }
    return {
      type: "ryuukyoku" as const,
      reason: res.type ?? "荒牌平局",
      tenpaiSeats: res.tenpaiHands.map((t) => t.who),
    };
  });
}

/** Every riichi declaration of the game (optionally one kyoku), with waits. */
export interface RiichiDeclaration {
  kyoku: string;
  roundIndex: number;
  seat: number;
  junme: number;
  anchor: number; // the リーチ判断 anchor id at this declaration
  waits: string; // wait tile glyphs, e.g. "４７"
  waitCount: number; // live (unseen) copies of the waits at declaration time
}

export function riichiDeclarations(game: Game, selector?: string): RiichiDeclaration[] {
  const only = selector === undefined ? null : new Set(resolveKyoku(game, selector));
  const out: RiichiDeclaration[] = [];
  for (const b of listAnchors(game)) {
    if (b.kind !== "リーチ判断" || b.seat === undefined) continue;
    if (only && !only.has(b.round)) continue;
    const st = replayTo(game, game.rounds[b.round], { eventIndex: b.eventIndex });
    const info = st.restInfo(b.seat);
    out.push({
      kyoku: roundLabel(game, b.round),
      roundIndex: b.round,
      seat: b.seat,
      junme: b.junme,
      anchor: b.id,
      waits: info.types.map((t) => typeGlyph(t)).join(""),
      waitCount: info.count,
    });
  }
  return out;
}

/** Final standings, or null when the log has no 終局 record. */
export function finalStandings(
  game: Game,
): Array<{ place: number; seat: number; name: string; score: number; points: number }> | null {
  if (!game.owari) return null;
  return owariRows(game.owari).map((r, i) => ({
    place: i + 1,
    seat: r.seat,
    name: game.players[r.seat].name,
    score: r.score,
    points: r.points,
  }));
}

/** Resolve a kyoku selector that must name exactly one round. */
export function uniqueRound(game: Game, selector: string): number {
  const indices = resolveKyoku(game, selector);
  if (indices.length > 1) {
    const opts = indices.map((i) => `${selector}.${game.rounds[i].honba}`).join(" / ");
    throw new Error(`"${selector}" matches ${indices.length} rounds — disambiguate: ${opts}`);
  }
  return indices[0];
}

export type SnapshotQuery =
  | { anchor: number }
  | { kyoku: string; junme: number };

/**
 * Board snapshot at an addressable position: a transcript anchor (`#N`), or an
 * explicit kyoku + junme (state at the END of that go-around). The kyoku form
 * requires a unique round — add ".<honba>" when a kyoku repeats.
 */
export function getSnapshot(game: Game, q: SnapshotQuery): string {
  if ("anchor" in q) {
    const beat = listAnchors(game).find((b) => b.id === q.anchor);
    if (!beat) throw new Error(`no anchor #${q.anchor} (use mj_list_anchors / anchors)`);
    const round = game.rounds[beat.round];
    const st = replayTo(game, round, { eventIndex: beat.eventIndex });
    return renderSnapshot(game, round, st, `#${beat.id} ${beat.kind}｜${beat.topic}`);
  }
  const round = game.rounds[uniqueRound(game, q.kyoku)];
  const st = replayTo(game, round, { junme: q.junme });
  return renderSnapshot(game, round, st, `${q.junme}巡目終了時点`);
}
