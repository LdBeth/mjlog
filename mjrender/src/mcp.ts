// MCP server (stdio): exposes the mjrender query core to LLM agents.
//
//   deno task mcp        (equivalent: deno run --allow-read src/mcp.ts)
//
// All tools are thin wrappers over core.ts and share the mj_ name prefix.
// The server is STATEFUL: mj_open_log is the only tool that takes a path — it
// parses the log once into session state, and every other tool operates on
// that state (erroring until a log is opened). The commentary draft also
// lives server-side: the agent fills one anchor at a time with mj_add_comment
// (plus mj_add_note for ★ lines), checks progress with mj_draft_status, and
// mj_weave_commentary splices the accumulated draft into a re-rendered
// transcript written to a file. The agent never reproduces fact lines and the
// woven document never passes through the model's context.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import {
  anchorTable,
  finalStandings,
  getSnapshot,
  kyokuResults,
  kyokuStart,
  listAnchors,
  listStarSites,
  loadGame,
  renderKyoku,
  renderOutline,
  riichiDeclarations,
  roundLabel,
  type StarNote,
  uniqueRound,
  weaveCommentary,
  weaveSummary,
} from "./core.ts";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { isUrl } from "./load.ts";
import type { Game } from "./model.ts";

// ---- session state ----
// One log at a time: the parsed game plus the commentary draft being built
// against it. mj_open_log replaces it; reopening the same (unchanged) log
// keeps the draft so an accidental re-open loses nothing.
interface Session {
  path: string;
  mtime: number; // 0 for URL sources (a finished game's log is immutable)
  game: Game;
  comments: Map<number, string>; // anchor id → commentary text
  notes: Map<string, StarNote>; // "round:junme:seat" → ★-line note
}
let session: Session | undefined;

function current(): Session {
  if (!session) throw new Error("no log loaded — call mj_open_log with the mjlog path/URL first");
  return session;
}

function draftLine(s: Session): string {
  return `draft: ${s.comments.size}/${listAnchors(s.game).length} comments, ${s.notes.size} notes`;
}

type ToolResult = { content: Array<{ type: "text"; text: string }>; isError?: boolean };

// Handler arg shapes, stated explicitly: the SDK's zod-based inference degrades
// to `any` when `zod` resolves to a different npm instance than the SDK's own.
interface KyokuArgs {
  kyoku: string;
  hands?: "key" | "all";
  snapshots?: "none" | "inline";
}
interface SnapshotArgs {
  anchor?: number;
  kyoku?: string;
  junme?: number;
}
interface CommentArgs {
  comments: Array<{ anchor: number; text: string }>;
}
interface NoteArgs {
  notes: Array<{ kyoku: string; junme: number; seat: number; text: string }>;
}
interface WeaveArgs {
  out: string;
  missing?: "keep" | "strip";
  hands?: "key" | "all";
}

function ok(text: string): ToolResult {
  return { content: [{ type: "text", text }] };
}

async function run(fn: () => Promise<string> | string): Promise<ToolResult> {
  try {
    return ok(await fn());
  } catch (e) {
    return {
      content: [{ type: "text", text: `error: ${e instanceof Error ? e.message : String(e)}` }],
      isError: true,
    };
  }
}

const KYOKU = z.string().describe(
  'Round selector: wind+number like "S3" / "東1" (optionally ".honba", e.g. "E1.2" when a kyoku repeats), or a 0-based round index like "6"',
);

const server = new McpServer({ name: "mjrender", version: "0.4.4" });

server.registerTool(
  "mj_open_log",
  {
    description:
      "Open a Tenhou game log and parse it into the session — the ONLY tool that takes a path; " +
      "every other mj_ tool operates on the opened log. Also starts an empty commentary draft " +
      "(reopening the same unchanged log keeps the draft; fresh=true discards it). Workflow: " +
      "open → mj_render_game for the game outline → mj_render_kyoku one round at a time for " +
      "per-turn detail → mj_get_snapshot at riichi/tenpai moments → save anchor comments with " +
      "mj_add_comment (several per call is fine) → mj_weave_commentary writes the finished " +
      "document.",
    inputSchema: {
      path: z.string().describe(
        "Tenhou mjlog source: local file path (gzipped .mjlog or plain .xml), or a tenhou.net " +
          "URL — a replay link like https://tenhou.net/0/?log=<id>&tw=1 or the raw log endpoint",
      ),
      fresh: z.boolean().optional()
        .describe("Discard the existing commentary draft when reopening the same log"),
    },
  },
  ({ path, fresh }: { path: string; fresh?: boolean }) =>
    run(async () => {
      const mtime = isUrl(path) ? 0 : (await Deno.stat(path)).mtime?.getTime() ?? 0;
      const keep = !fresh && session !== undefined && session.path === path &&
        session.mtime === mtime;
      session = {
        path,
        mtime,
        game: keep ? session!.game : await loadGame(path),
        comments: keep ? session!.comments : new Map(),
        notes: keep ? session!.notes : new Map(),
      };
      const g = session.game;
      const players = g.players.map((p) => `P${p.seat} ${p.name}`).join(" / ");
      return `opened ${path}\n${players}\n` +
        `kyoku: ${g.rounds.length} / anchors: ${listAnchors(g).length} (mj_list_anchors)\n` +
        draftLine(session);
    }),
);

server.registerTool(
  "mj_render_game",
  {
    description:
      "OUTLINE of the opened game (crude, cheap): the players block, then per kyoku only the " +
      "header with start scores, the condensed result (winner/yaku/points/score movements), and " +
      "the 〔解説ポイント#N〕 anchor index with junme+seat — NO per-turn lines (the notation " +
      "legend arrives with each mj_render_kyoku). Read this first to " +
      "orient, then fetch full per-turn detail ONE ROUND AT A TIME with mj_render_kyoku. At " +
      "riichi declarations and tenpai moments, check mj_get_snapshot BEFORE writing the comment. " +
      "Commentary goes through mj_add_comment — do NOT reproduce transcript lines yourself.",
    inputSchema: {},
  },
  () => run(() => renderOutline(current().game)),
);

server.registerTool(
  "mj_render_kyoku",
  {
    description:
      "Render ONE round (kyoku) of the opened game in full per-turn detail, self-contained " +
      "(format preamble included). Anchor ids inside are game-global, so they agree with " +
      "mj_list_anchors/mj_get_snapshot. When the round has riichi declarations or tenpai, pull " +
      "mj_get_snapshot for those moments before commenting on them.",
    inputSchema: {
      kyoku: KYOKU,
      hands: z.enum(["key", "all"]).optional(),
      snapshots: z.enum(["none", "inline"]).optional(),
    },
  },
  ({ kyoku, hands, snapshots }: KyokuArgs) =>
    run(() => renderKyoku(current().game, kyoku, { hands, snapshots })),
);

server.registerTool(
  "mj_list_anchors",
  {
    description: "List every commentary anchor of the opened game, one per line: " +
      "#id, kind (配牌評価/リーチ判断/押し引き/副露判断/局総括/流局評価/終局総括), kyoku, junme, " +
      "seat, topic.",
    inputSchema: {},
  },
  () => run(() => anchorTable(current().game)),
);

server.registerTool(
  "mj_get_snapshot",
  {
    description:
      "Recall the full board state at a position: all four rivers (▽=tsumogiri, *=riichi tile, " +
      "(→Pn)=called away), melds, live scores + placements, riichi states, dora, remaining wall, " +
      "and each seat's concealed hand with shanten/ukeire. Address by anchor id (from the " +
      "transcript's 〔解説ポイント#N〕 / mj_list_anchors), or by kyoku + junme (state at the end of " +
      "that go-around). ALWAYS check this at riichi declarations and tenpai moments before " +
      "writing commentary (リーチ判断/押し引き anchors) — do not judge them from the outline alone.",
    inputSchema: {
      anchor: z.number().int().positive().optional().describe("Anchor id #N"),
      kyoku: KYOKU.optional(),
      junme: z.number().int().nonnegative().optional()
        .describe("Go-around number (requires kyoku)"),
    },
  },
  ({ anchor, kyoku, junme }: SnapshotArgs) =>
    run(() => {
      const g = current().game;
      if (anchor !== undefined) return getSnapshot(g, { anchor });
      if (kyoku !== undefined && junme !== undefined) return getSnapshot(g, { kyoku, junme });
      throw new Error("provide either `anchor`, or both `kyoku` and `junme`");
    }),
);

// ---- commentary draft (server-side state, filled one entry at a time) ----

server.registerTool(
  "mj_add_comment",
  {
    description:
      "Save commentary for one or MORE anchors into the session draft — batch several per call " +
      "(e.g. a finished kyoku's worth, max 10) to conserve tool calls; saving an anchor again " +
      "replaces it. Fill anchors in any order; nothing is written to disk until " +
      "mj_weave_commentary. The " +
      "batch is atomic (one bad entry saves nothing). Returns draft progress and which anchors " +
      "are still unfilled. ★-marked lines you meet in kyoku renders can optionally get a " +
      "one-liner via mj_add_note.",
    inputSchema: {
      comments: z.array(z.object({
        anchor: z.number().int().positive().describe("Anchor id #N"),
        text: z.string().min(1).describe(
          "Commentary for this anchor (plain text, may be multiline)",
        ),
      })).min(1).max(10).describe("Anchor comments to save, one entry per anchor (max 10)"),
    },
  },
  ({ comments }: CommentArgs) =>
    run(() => {
      const s = current();
      const beats = listAnchors(s.game);
      // validate the whole batch before touching the draft
      const seen = new Set<number>();
      for (const { anchor, text } of comments) {
        if (anchor > beats.length) {
          throw new Error(
            `unknown anchor #${anchor} — this game has #1..#${beats.length} (mj_list_anchors)`,
          );
        }
        if (seen.has(anchor)) throw new Error(`anchor #${anchor} appears twice in this batch`);
        seen.add(anchor);
        if (!text.trim()) throw new Error(`empty comment for anchor #${anchor}`);
      }
      const replaced: number[] = [];
      for (const { anchor, text } of comments) {
        if (s.comments.has(anchor)) replaced.push(anchor);
        s.comments.set(anchor, text.trim());
      }
      const ids = comments.map((c) => `#${c.anchor}`).join(" ");
      const open = beats.filter((x) => !s.comments.has(x.id)).map((x) => `#${x.id}`);
      const rest = open.length === 0
        ? " — all anchors filled; mj_weave_commentary writes the document"
        : ` / 未記入: ${open.slice(0, 16).join(" ")}${open.length > 16 ? ` …(+${open.length - 16})` : ""}`;
      return `saved ${ids}${
        replaced.length ? ` (replaced ${replaced.map((i) => `#${i}`).join(" ")})` : ""
      } — ${s.comments.size}/${beats.length}${rest}`;
    }),
);

server.registerTool(
  "mj_add_note",
  {
    description:
      "Save optional one-liners for ★-marked lines (notable discards/calls) into the session " +
      "draft, addressed by kyoku + junme + seat — one entry per ★ site, batched up to 10 per " +
      "call; saving the same site again replaces it. The batch is atomic (one bad entry saves " +
      "nothing). If the seat has several ★ lines in one go-around (e.g. call then discard), the " +
      "note lands after the last one.",
    inputSchema: {
      notes: z.array(z.object({
        kyoku: KYOKU,
        junme: z.number().int().nonnegative().describe("Go-around number of the ★ line"),
        seat: z.number().int().min(0).max(3).describe("Acting seat 0-3 (P0-P3)"),
        text: z.string().min(1).describe("Short one-liner for that ★ moment"),
      })).min(1).max(10).describe("★-line notes to save, one entry per ★ site (max 10)"),
    },
  },
  ({ notes }: NoteArgs) =>
    run(() => {
      const s = current();
      const sites = listStarSites(s.game);
      // validate the whole batch before touching the draft
      const staged: Array<{ key: string; note: StarNote; label: string }> = [];
      const seen = new Set<string>();
      for (const n of notes) {
        const t = n.text.trim();
        if (!t) throw new Error(`empty ★ note for ${n.kyoku} ${n.junme}巡 P${n.seat}`);
        const round = uniqueRound(s.game, n.kyoku);
        const label = `${roundLabel(s.game, round)} ${n.junme}巡 P${n.seat}`;
        const here = sites.filter((x) => x.round === round);
        if (!here.some((x) => x.junme === n.junme && x.seat === n.seat)) {
          const list = here.map((x) => `${x.junme}巡P${x.seat}`).join(" ");
          throw new Error(
            `no ★ line for P${n.seat} at ${roundLabel(s.game, round)} ${n.junme}巡` +
              (list ? ` — ★ sites in this kyoku: ${list}` : " — this kyoku has no ★ lines"),
          );
        }
        const key = `${round}:${n.junme}:${n.seat}`;
        if (seen.has(key)) throw new Error(`duplicate ★ note in this batch: ${label}`);
        seen.add(key);
        staged.push({ key, note: { kyoku: String(round), junme: n.junme, seat: n.seat, text: t }, label });
      }
      const replaced = staged.filter((x) => s.notes.has(x.key)).map((x) => x.label);
      for (const x of staged) s.notes.set(x.key, x.note);
      return `★ note saved: ${staged.map((x) => x.label).join(" / ")}` +
        `${replaced.length ? ` (replaced: ${replaced.join(" / ")})` : ""} — ` +
        `${s.notes.size} note(s) in draft`;
    }),
);

server.registerTool(
  "mj_draft_status",
  {
    description:
      "Progress of the session's commentary draft: every anchor as a checklist line " +
      "(✓ filled / ・ unfilled), plus the saved ★ notes.",
    inputSchema: {},
  },
  () =>
    run(() => {
      const s = current();
      const list = listAnchors(s.game).map((b) =>
        `${s.comments.has(b.id) ? "✓" : "・"} #${b.id}\t${b.kind}\t` +
        `${roundLabel(s.game, b.round)}\t${b.junme}巡` +
        `${b.seat !== undefined ? `\tP${b.seat}` : "\t"}\t${b.topic}`
      );
      const notes = [...s.notes.values()].map((n) =>
        `★ ${roundLabel(s.game, uniqueRound(s.game, n.kyoku))} ${n.junme}巡 P${n.seat}: ${n.text}`
      );
      return [`${s.path} — ${draftLine(s)}`, ...list, ...notes].join("\n");
    }),
);

server.registerTool(
  "mj_weave_commentary",
  {
    description:
      "Produce the finished commentary document: deterministically splice the session draft " +
      "(everything accumulated via mj_add_comment / mj_add_note) into a re-rendered transcript " +
      "and WRITE IT TO `out` — never copy transcript lines yourself; every fact line comes from " +
      "the renderer verbatim. Returns only a one-line summary (filled/missing counts) — the " +
      "document itself never enters the conversation; unfilled anchors stay as placeholders " +
      "(missing=keep) so partial drafts are valid and you can weave again after more " +
      "mj_add_comment calls.",
    inputSchema: {
      out: z.string().describe(
        "Where to write the woven document (UTF-8). This server runs on the USER'S machine — " +
          "paths from your own sandbox/workspace do not exist here. Best: a bare filename like " +
          "'commentary.txt', which lands next to the log file (for URL sources: in the user's " +
          "home directory). The summary reports the absolute path — relay it to the user",
      ),
      missing: z.enum(["keep", "strip"]).optional()
        .describe("Anchors you did not fill: keep their placeholder lines (default) or strip them"),
      hands: z.enum(["key", "all"]).optional()
        .describe("Reconstructed-hand verbosity of the woven transcript (default key)"),
    },
  },
  ({ out, missing, hands }: WeaveArgs) =>
    run(async () => {
      const s = current();
      if (s.comments.size === 0 && s.notes.size === 0) {
        throw new Error("draft is empty — save commentary first with mj_add_comment / mj_add_note");
      }
      const r = weaveCommentary(s.game, {
        anchors: [...s.comments].map(([anchor, text]) => ({ anchor, text })),
        notes: [...s.notes.values()],
      }, { missing, hands });
      const dest = resolveOut(out, s.path);
      try {
        await Deno.writeTextFile(dest, r.text + "\n");
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        throw new Error(
          `cannot write ${dest}: ${msg}\nThis MCP server runs on the user's machine — paths from ` +
            `your own environment (e.g. /mnt/…) don't exist here. Pass a bare filename to write ` +
            `next to the log file, or an absolute path that exists on the user's machine.`,
        );
      }
      return weaveSummary(r, dest);
    }),
);

// The caller may live in a different filesystem than this server (an agent
// sandbox vs the user's machine), so a relative `out` must resolve somewhere
// predictable from the input: beside the log file, or under $HOME for URL
// sources. The summary always reports the resulting absolute path.
function resolveOut(out: string, srcPath: string): string {
  if (isAbsolute(out)) return out;
  const base = isUrl(srcPath) ? Deno.env.get("HOME") ?? Deno.cwd() : dirname(resolve(srcPath));
  return join(base, out);
}

// ---- structured fact tools (JSON responses; same source as the transcript) ----

const json = (v: unknown) => JSON.stringify(v, null, 1);

server.registerTool(
  "mj_get_kyoku_start",
  {
    description:
      "Start conditions of one round: dealer, honba, kyotaku, dora indicator, and per-seat " +
      "start scores with placements. JSON.",
    inputSchema: { kyoku: KYOKU },
  },
  ({ kyoku }: { kyoku: string }) => run(() => json(kyokuStart(current().game, kyoku))),
);

server.registerTool(
  "mj_get_kyoku_result",
  {
    description:
      "Outcome(s) of one round: winner, tsumo/ron + discarder, winning tile, points/fu/limit " +
      "and yaku — or draw reason + tenpai seats. Multiple entries = double/triple ron. JSON.",
    inputSchema: { kyoku: KYOKU },
  },
  ({ kyoku }: { kyoku: string }) => run(() => json(kyokuResults(current().game, kyoku))),
);

server.registerTool(
  "mj_get_riichi_declarations",
  {
    description:
      "Every riichi declaration (whole game, or one kyoku): seat, junme, wait tiles, live " +
      "(unseen) wait count at declaration time, and the リーチ判断 anchor id. JSON.",
    inputSchema: { kyoku: KYOKU.optional() },
  },
  ({ kyoku }: { kyoku?: string }) => run(() => json(riichiDeclarations(current().game, kyoku))),
);

server.registerTool(
  "mj_get_final_standings",
  {
    description: "Final standings: place, seat, name, score, placement points. JSON.",
    inputSchema: {},
  },
  () =>
    run(() => {
      const s = finalStandings(current().game);
      if (!s) throw new Error("log has no 終局 record (game did not finish?)");
      return json(s);
    }),
);

await server.connect(new StdioServerTransport());
