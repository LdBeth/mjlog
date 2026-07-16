// Reusable core: the full query API over a parsed game. The CLI subcommands
// and the MCP server are both thin wrappers over THESE functions.

import { loadXml } from "./load.ts";
import { parseGame } from "./parse.ts";
import { renderGameAnnotated } from "./render.ts";
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

function uniqueRound(game: Game, selector: string): number {
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
