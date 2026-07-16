// Reusable core: the full query API over a parsed game. The CLI subcommands
// and the MCP server are both thin wrappers over THESE functions.

import { loadXml } from "./load.ts";
import { parseGame } from "./parse.ts";
import { renderGameAnnotated } from "./render.ts";
import { renderSnapshot } from "./snapshot.ts";
import { replayTo } from "./state.ts";
import type { Beat, Game, RenderOptions } from "./model.ts";

const WIND = ["東", "南", "西", "北"];

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
  const k = round.kyoku;
  return `${WIND[Math.floor(k / 4) % 4]}${(k % 4) + 1}局${round.honba}本場`;
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
  const hits = game.rounds
    .map((r, i) => ({ r, i }))
    .filter(({ r }) => r.kyoku === kyoku && (honba === undefined || r.honba === honba))
    .map(({ i }) => i);
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
    if (!beat) throw new Error(`no anchor #${q.anchor} (use list_anchors / anchors)`);
    const round = game.rounds[beat.round];
    const st = replayTo(game, round, { eventIndex: beat.eventIndex });
    return renderSnapshot(game, round, st, `#${beat.id} ${beat.kind}｜${beat.topic}`);
  }
  const indices = resolveKyoku(game, q.kyoku);
  if (indices.length > 1) {
    const opts = indices.map((i) => `${q.kyoku}.${game.rounds[i].honba}`).join(" / ");
    throw new Error(`"${q.kyoku}" matches ${indices.length} rounds — disambiguate: ${opts}`);
  }
  const round = game.rounds[indices[0]];
  const st = replayTo(game, round, { junme: q.junme });
  return renderSnapshot(game, round, st, `${q.junme}巡目終了時点`);
}
