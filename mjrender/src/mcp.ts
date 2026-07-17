// MCP server (stdio): exposes the mjrender query core to LLM agents, PACED.
//
// Run via the bundle (the SDK's server/* imports are deliberately extensionless
// so `deno check` sees real types; `deno run src/mcp.ts` fails at import — build
// `deno bundle -o mcp.mjs src/mcp.ts` and run that instead).
//
// All tools are thin wrappers over core.ts and share the mj_ name prefix. The
// server is STATEFUL: mj_open_log is the only tool that takes a path — it parses
// the log once into session state, and every other tool operates on that state.
//
// The session also carries a FOCUS cursor: the highest unlocked round index
// (starts at 0). The intended chat loop is one kyoku of detail per turn —
//   mj_render_kyoku(<focus>)  → study that one round in full
//   mj_get_snapshot(...)      → confirm boards at riichi/tenpai moments
//   mj_add_comment / mj_add_note → save the focus kyoku's anchors/★ notes
//   mj_next_kyoku             → unlock the next round and END THE CHAT TURN
// Future rounds are LOCKED for per-turn renders, snapshots, and per-kyoku fact
// tools — the gate paces reading, it is NOT a spoiler shield: the ungated
// mj_render_game outline (results, ◆終局 included) and mj_open_log are read once
// at open to orient. The format legend is emitted once per process, appended to
// the first mj_open_log reply. ★ notes take no kyoku argument — they address the
// note window: the focus kyoku, or right after mj_next_kyoku still the finished
// kyoku until mj_render_kyoku opens the new focus (re-save replaces; empty text
// deletes). mj_weave_commentary splices the accumulated draft into a
// re-rendered transcript written to a file; the model never copies fact lines and
// the woven document never enters its context.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio";
import { z } from "zod";
import {
  getSnapshot,
  kyokuResults,
  kyokuStart,
  listAnchors,
  listStarSites,
  loadGame,
  renderKyoku,
  renderOutline,
  resolveKyoku,
  riichiDeclarations,
  roundLabel,
  standingsLine,
  type StarNote,
  uniqueRound,
  weaveCommentary,
  weaveSummary,
} from "./core.ts";
import { formatInstruction } from "./render.ts";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { isUrl } from "./load.ts";
import type { Beat, Game } from "./model.ts";

// ---- session state ----
// One log at a time: the parsed game, the commentary draft being built against
// it, and the focus cursor. mj_open_log replaces it; reopening the same
// (unchanged) log keeps the draft AND the focus so an accidental re-open loses
// nothing (fresh=true resets both).
interface Session {
  path: string;
  mtime: number; // 0 for URL sources (a finished game's log is immutable)
  game: Game;
  comments: Map<number, string>; // anchor id → commentary text
  notes: Map<string, StarNote>; // "round:junme:seat" → ★-line note
  focus: number; // highest unlocked round index (starts 0)
  // The round ★ notes currently address. Trails the focus across mj_next_kyoku:
  // the finished kyoku stays notable until mj_render_kyoku of the NEW focus
  // runs — that render is the moment the previous round's notes lock.
  noteRound: number;
}
let session: Session | undefined;

// The ~2KB notation legend goes out ONCE per server process, appended to the
// first mj_open_log reply. Later opens (or reopens) omit it.
let legendSent = false;

function current(): Session {
  if (!session) throw new Error("no log loaded — call mj_open_log with the mjlog path/URL first");
  return session;
}

// Hard read-gate: a round beyond the focus cursor is not yet available for
// per-turn renders, snapshots, or per-kyoku fact queries. This paces reading —
// it is NOT a spoiler shield (results stay visible in the ungated outline).
function assertUnlocked(s: Session, round: number, what: string): void {
  if (round > s.focus) {
    throw new Error(
      `locked: ${what}（${roundLabel(s.game, round)}）is beyond the current focus ` +
        `${roundLabel(s.game, s.focus)} — finish the focus kyoku's anchors with ` +
        `mj_add_comment, then advance with mj_next_kyoku`,
    );
  }
}

function draftLine(s: Session): string {
  const unlocked = listAnchors(s.game).filter((b) => b.round <= s.focus);
  const filled = unlocked.filter((b) => s.comments.has(b.id)).length;
  return `draft: ${filled}/${unlocked.length} comments ` +
    `(kyoku ${s.focus + 1}/${s.game.rounds.length} unlocked), ${s.notes.size} notes`;
}

// Trailer appended to the working checklists (mj_list_anchors / mj_draft_status)
// so an unfilled-but-short list is not read as "the whole game".
function lockedTrailer(s: Session): string[] {
  const remaining = s.game.rounds.length - (s.focus + 1);
  return remaining > 0 ? [`（未開放: 残り${remaining}局 — mj_next_kyoku で進行）`] : [];
}

const fmtBeat = (b: Beat): string =>
  `#${b.id}(${b.kind}・${b.junme}巡${b.seat !== undefined ? `P${b.seat}` : ""})`;

// ★-note nudge for the round about to leave focus. Never blocks; returns null
// when the round already has enough notes (or has no ★ sites).
function starHint(s: Session, round: number): string | null {
  const sites = listStarSites(s.game).filter((x) => x.round === round);
  const noted = [...s.notes.values()]
    .filter((n) => uniqueRound(s.game, n.kyoku) === round).length;
  if (noted >= Math.min(2, sites.length)) return null;
  const cands = sites.slice(0, 8).map((x) => `${x.junme}巡P${x.seat}`).join(" ");
  return `HINT: ${roundLabel(s.game, round)} has ${sites.length} ★ sites but only ${noted} ` +
    `note(s) — LAST CHANCE: its notes lock when mj_render_kyoku opens the next kyoku; ` +
    `consider mj_add_note: ${cands}`;
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
  notes: Array<{ junme: number; seat: number; text: string }>;
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

const server = new McpServer({ name: "mjrender", version: "0.5.0" });

server.registerTool(
  "mj_open_log",
  {
    description:
      "Open a Tenhou game log and parse it into the session — the ONLY tool that takes a path; " +
      "every other mj_ tool operates on the opened log. Starts an empty commentary draft and sets " +
      "the FOCUS cursor to kyoku 0 (reopening the same unchanged log keeps the draft AND focus; " +
      "fresh=true resets both). The reply carries the notation legend the FIRST time per process. " +
      "PACED FLOW: read mj_render_game once to orient (results are visible, not spoilers), then " +
      "handle ONE kyoku per chat turn — mj_render_kyoku(<focus>) for detail, mj_get_snapshot at " +
      "riichi/tenpai, mj_add_comment / mj_add_note to fill that kyoku's anchors, then " +
      "mj_next_kyoku to unlock the next round and END YOUR TURN. Future rounds are locked until " +
      "you advance. mj_weave_commentary writes the finished document once all anchors are filled.",
    inputSchema: {
      path: z.string().describe(
        "Tenhou mjlog source: local file path (gzipped .mjlog or plain .xml), or a tenhou.net " +
          "URL — a replay link like https://tenhou.net/0/?log=<id>&tw=1 or the raw log endpoint",
      ),
      fresh: z.boolean().optional()
        .describe("Discard the existing draft AND reset the focus cursor when reopening the same log"),
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
        focus: keep ? session!.focus : 0,
        noteRound: keep ? session!.noteRound : 0,
      };
      const g = session.game;
      const players = g.players.map((p) => `P${p.seat} ${p.name}`).join(" / ");
      const lines = [
        `opened ${path}`,
        players,
        `kyoku: ${g.rounds.length} / anchors: ${listAnchors(g).length} (mj_list_anchors)`,
        `focus: ${roundLabel(g, session.focus)} (round ${session.focus})`,
        "流れ: mj_render_game で全体を俯瞰 → mj_render_kyoku で担当の1局を精読 → その局のアンカーを" +
        "mj_add_comment で埋める → mj_next_kyoku で次局を開放しチャットターンを終える" +
        "（未開放局の詳細レンダ/スナップショット/facts はロック）",
        draftLine(session),
      ];
      if (!legendSent) {
        legendSent = true;
        lines.push("", formatInstruction("fill"));
      }
      return lines.join("\n");
    }),
);

server.registerTool(
  "mj_render_game",
  {
    description:
      "OUTLINE of the whole opened game (crude, cheap, UNGATED): the players block, then per kyoku " +
      "only the header with start scores, the condensed result (winner/yaku/points/score " +
      "movements), and the 〔解説ポイント#N〕 anchor index — NO per-turn lines. Read this ONCE at " +
      "open to orient (how each player fared — results are NOT spoilers here), and return to it for " +
      "recaps at the 中間総括 (wind boundary) and 終局総括 (game end). Per-turn detail stays gated: " +
      "fetch it ONE ROUND AT A TIME with mj_render_kyoku on the current focus kyoku. Commentary " +
      "goes through mj_add_comment — do NOT reproduce transcript lines yourself.",
    inputSchema: {},
  },
  () => run(() => renderOutline(current().game)),
);

server.registerTool(
  "mj_render_kyoku",
  {
    description:
      "Render ONE round (kyoku) of the opened game in full per-turn detail — GATED to the focus " +
      "cursor (a future round errors with `locked`; call mj_next_kyoku to advance). Board " +
      "snapshots are embedded INLINE by default. Anchor ids inside are game-global, so they agree " +
      "with mj_list_anchors / mj_get_snapshot. This is the one round you comment this chat turn: " +
      "study it, fill its anchors with mj_add_comment, then mj_next_kyoku and end the turn.",
    inputSchema: {
      kyoku: KYOKU,
      hands: z.enum(["key", "all"]).optional(),
      snapshots: z.enum(["none", "inline"]).optional()
        .describe(
          "Inline board snapshots above each anchor (default inline; the 配牌評価 anchor carries " +
            "none — the deal block above it already shows every hand — and the end-of-hand " +
            "◇結果時点の各家手牌 block is folded into the final snapshot, 振聴 marks included) " +
            "or omit them (none)",
        ),
    },
  },
  ({ kyoku, hands, snapshots }: KyokuArgs) =>
    run(() => {
      const s = current();
      const indices = resolveKyoku(s.game, kyoku);
      const anyLocked = indices.some((i) => i > s.focus);
      if (anyLocked) {
        const anyUnlocked = indices.some((i) => i <= s.focus);
        if (anyUnlocked) {
          const opts = indices.map((i) => `"${kyoku}.${s.game.rounds[i].honba}"`).join(" / ");
          throw new Error(
            `"${kyoku}" matches both an unlocked and a locked repeat — specify one: ${opts}`,
          );
        }
        assertUnlocked(s, Math.min(...indices), "mj_render_kyoku");
      }
      // Rendering the focus round moves the ★-note window onto it, locking the
      // previous kyoku's notes (re-reading past rounds does not move it).
      if (indices.includes(s.focus)) s.noteRound = s.focus;
      return renderKyoku(s.game, kyoku, {
        hands,
        snapshots: snapshots ?? "inline",
        header: "none",
        owari: true,
      });
    }),
);

server.registerTool(
  "mj_list_anchors",
  {
    description: "List the UNLOCKED commentary anchors of the opened game (up to the focus kyoku), " +
      "one per line: #id, kind (配牌評価/リーチ判断/押し引き/副露判断/局総括/流局評価/中間総括/終局総括), " +
      "kyoku, junme, seat, topic. This is the working checklist for the focus kyoku; future rounds " +
      "are hidden until mj_next_kyoku advances (the full index lives in the ungated mj_render_game outline).",
    inputSchema: {},
  },
  () =>
    run(() => {
      const s = current();
      const list = listAnchors(s.game).filter((b) => b.round <= s.focus).map((b) =>
        `#${b.id}\t${b.kind}\t${roundLabel(s.game, b.round)}\t${b.junme}巡` +
        `${b.seat !== undefined ? `\tP${b.seat}` : "\t"}\t${b.topic}`
      );
      return [...list, ...lockedTrailer(s)].join("\n");
    }),
);

server.registerTool(
  "mj_get_snapshot",
  {
    description:
      "Recall the full board state at a position: all four rivers (▽=tsumogiri, *=riichi tile, " +
      "(→Pn)=called away), melds, live scores + placements, riichi states, dora, remaining wall, " +
      "and each seat's concealed hand with shanten/ukeire. Address by anchor id (from the " +
      "transcript's 〔解説ポイント#N〕 / mj_list_anchors), or by kyoku + junme (state at the end of " +
      "that go-around). GATED: the target round must be at or before the focus cursor. ALWAYS " +
      "check this at riichi declarations and tenpai moments before writing commentary " +
      "(リーチ判断/押し引き anchors) — do not judge them from the outline alone.",
    inputSchema: {
      anchor: z.number().int().positive().optional().describe("Anchor id #N"),
      kyoku: KYOKU.optional(),
      junme: z.number().int().nonnegative().optional()
        .describe("Go-around number (requires kyoku)"),
    },
  },
  ({ anchor, kyoku, junme }: SnapshotArgs) =>
    run(() => {
      const s = current();
      const g = s.game;
      if (anchor !== undefined) {
        const beat = listAnchors(g).find((b) => b.id === anchor);
        if (!beat) throw new Error(`no anchor #${anchor} (use mj_list_anchors)`);
        assertUnlocked(s, beat.round, `mj_get_snapshot #${anchor}`);
        return getSnapshot(g, { anchor });
      }
      if (kyoku !== undefined && junme !== undefined) {
        assertUnlocked(s, uniqueRound(g, kyoku), "mj_get_snapshot");
        return getSnapshot(g, { kyoku, junme });
      }
      throw new Error("provide either `anchor`, or both `kyoku` and `junme`");
    }),
);

// ---- commentary draft (server-side state, filled one entry at a time) ----

server.registerTool(
  "mj_add_comment",
  {
    description:
      "Save commentary for one or MORE anchors into the session draft — batch the focus kyoku's " +
      "worth per call (max 10) to conserve tool calls. Anchors of the FOCUS kyoku are fillable; a " +
      "future kyoku's anchor errors `locked` (advance first with mj_next_kyoku); a PAST kyoku's " +
      "anchor is replace-only (revise an existing comment at 中間総括/終局総括 time, but no new " +
      "fills). Nothing is written to disk until mj_weave_commentary. The batch is atomic (one bad " +
      "entry saves nothing). Returns draft progress and which UNLOCKED anchors are still unfilled. " +
      "★-marked lines you meet in kyoku renders can optionally get a one-liner via mj_add_note.",
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
        const beat = beats.find((b) => b.id === anchor)!;
        if (beat.round > s.focus) {
          throw new Error(
            `locked: anchor #${anchor}（${roundLabel(s.game, beat.round)}）is beyond the current ` +
              `focus ${roundLabel(s.game, s.focus)} — comment it after mj_next_kyoku opens that kyoku`,
          );
        }
        if (beat.round < s.focus && !s.comments.has(anchor)) {
          throw new Error(
            `anchor #${anchor}（${roundLabel(s.game, beat.round)}）is a past kyoku — past anchors ` +
              `are replace-only; only its existing comment can be revised, not newly filled`,
          );
        }
      }
      const replaced: number[] = [];
      for (const { anchor, text } of comments) {
        if (s.comments.has(anchor)) replaced.push(anchor);
        s.comments.set(anchor, text.trim());
      }
      const ids = comments.map((c) => `#${c.anchor}`).join(" ");
      const unlocked = beats.filter((b) => b.round <= s.focus);
      const open = unlocked.filter((x) => !s.comments.has(x.id)).map((x) => `#${x.id}`);
      const last = s.focus >= s.game.rounds.length - 1;
      const rest = open.length === 0
        ? (last
          ? " — all anchors filled; mj_weave_commentary writes the document"
          : " — 開放局のアンカーは全て記入済み; mj_next_kyoku で次局へ進みターンを終える")
        : ` / 未記入: ${open.slice(0, 16).join(" ")}${open.length > 16 ? ` …(+${open.length - 16})` : ""}`;
      return `saved ${ids}${
        replaced.length ? ` (replaced ${replaced.map((i) => `#${i}`).join(" ")})` : ""
      } — ${s.comments.size}/${unlocked.length}${rest}`;
    }),
);

server.registerTool(
  "mj_add_note",
  {
    description:
      "Save optional one-liners for ★-marked lines (notable discards/calls) of the kyoku being " +
      "commented — no kyoku argument: notes always address the current note window, i.e. the " +
      "focus kyoku (right after mj_next_kyoku, still the just-finished kyoku — the window moves " +
      "only when mj_render_kyoku opens the new focus, and the previous kyoku's notes lock then). " +
      "One entry per ★ site (junme + seat), batched up to 10 per call. Saving the same site " +
      "again replaces it; EMPTY/blank text DELETES the saved note at that site. The batch is " +
      "atomic (one bad entry saves nothing). If the seat has several ★ lines in one go-around " +
      "(call then discard), the note lands after the last one.",
    inputSchema: {
      notes: z.array(z.object({
        junme: z.number().int().nonnegative().describe("Go-around number of the ★ line"),
        seat: z.number().int().min(0).max(3).describe("Acting seat 0-3 (P0-P3)"),
        text: z.string().describe("Short one-liner for that ★ moment; empty/blank deletes the saved note"),
      })).min(1).max(10).describe(
        "★-line notes to save or delete in the current kyoku, one entry per ★ site (max 10)",
      ),
    },
  },
  ({ notes }: NoteArgs) =>
    run(() => {
      const s = current();
      const round = s.noteRound;
      const here = listStarSites(s.game).filter((x) => x.round === round);
      // validate the whole batch before touching the draft
      const staged: Array<{ key: string; note?: StarNote; del: boolean; label: string }> = [];
      const seen = new Set<string>();
      for (const n of notes) {
        const label = `${roundLabel(s.game, round)} ${n.junme}巡 P${n.seat}`;
        if (!here.some((x) => x.junme === n.junme && x.seat === n.seat)) {
          const list = here.map((x) => `${x.junme}巡P${x.seat}`).join(" ");
          throw new Error(
            `no ★ line for P${n.seat} at ${n.junme}巡 — notes address ${
              roundLabel(s.game, round)
            }` + (list ? `（★: ${list}）` : "（この局に★行なし）"),
          );
        }
        const key = `${round}:${n.junme}:${n.seat}`;
        if (seen.has(key)) throw new Error(`duplicate ★ note in this batch: ${label}`);
        seen.add(key);
        const t = n.text.trim();
        if (!t) {
          if (!s.notes.has(key)) throw new Error(`no saved ★ note to delete at ${label}`);
          staged.push({ key, del: true, label });
        } else {
          staged.push({
            key,
            del: false,
            note: { kyoku: String(round), junme: n.junme, seat: n.seat, text: t },
            label,
          });
        }
      }
      const saved: string[] = [], deleted: string[] = [], replaced: string[] = [];
      for (const x of staged) {
        if (x.del) {
          s.notes.delete(x.key);
          deleted.push(x.label);
        } else {
          if (s.notes.has(x.key)) replaced.push(x.label);
          s.notes.set(x.key, x.note!);
          saved.push(x.label);
        }
      }
      const parts: string[] = [];
      if (saved.length) {
        parts.push(
          `saved ${saved.length}: ${saved.join(" / ")}` +
            (replaced.length ? ` (replaced: ${replaced.join(" / ")})` : ""),
        );
      }
      if (deleted.length) parts.push(`deleted ${deleted.length}: ${deleted.join(" / ")}`);
      return `★ ${parts.join(" ; ")} — ${s.notes.size} note(s) in draft`;
    }),
);

server.registerTool(
  "mj_draft_status",
  {
    description:
      "Progress of the session's commentary draft: every UNLOCKED anchor (up to the focus kyoku) " +
      "as a checklist line (✓ filled / ・ unfilled), plus the saved ★ notes and how many kyoku " +
      "remain locked.",
    inputSchema: {},
  },
  () =>
    run(() => {
      const s = current();
      const list = listAnchors(s.game).filter((b) => b.round <= s.focus).map((b) =>
        `${s.comments.has(b.id) ? "✓" : "・"} #${b.id}\t${b.kind}\t` +
        `${roundLabel(s.game, b.round)}\t${b.junme}巡` +
        `${b.seat !== undefined ? `\tP${b.seat}` : "\t"}\t${b.topic}`
      );
      const notes = [...s.notes.values()].map((n) =>
        `★ ${roundLabel(s.game, uniqueRound(s.game, n.kyoku))} ${n.junme}巡 P${n.seat}: ${n.text}`
      );
      return [`${s.path} — ${draftLine(s)}`, ...list, ...notes, ...lockedTrailer(s)].join("\n");
    }),
);

server.registerTool(
  "mj_next_kyoku",
  {
    description:
      "Advance the focus cursor to the next kyoku — the pacing step. FIRST fill every anchor of " +
      "the current focus kyoku with mj_add_comment; this errors and lists any that are unfilled " +
      "(at a wind boundary it prints the 中間総括 checkpoint with current standings). On success " +
      "it unlocks the next round, reports progress + a ★-note hint for the finished " +
      "kyoku + (at a wind crossing) the standings block, then tells you to END THIS CHAT TURN. At " +
      "the final kyoku it does not advance — it tells you to write the document with " +
      "mj_weave_commentary and STOP.",
    inputSchema: {},
  },
  () =>
    run(() => {
      const s = current();
      const g = s.game;
      const focus = s.focus;
      const lastRound = g.rounds.length - 1;
      const focusBeats = listAnchors(g).filter((b) => b.round === focus);
      const unfilled = focusBeats.filter((b) => !s.comments.has(b.id));

      // 1. Guard: the focus kyoku must be fully commented before advancing.
      if (unfilled.length > 0) {
        const lines = [
          `未記入のアンカーが ${unfilled.length} 件あります（${roundLabel(g, focus)}）:`,
          ...unfilled.map(fmtBeat),
          "save them with mj_add_comment, then call mj_next_kyoku again",
        ];
        const chukan = unfilled.find((b) => b.kind === "中間総括");
        if (chukan && focus < lastRound) {
          const nextWind = roundLabel(g, focus + 1).charAt(0);
          const prevWind = roundLabel(g, focus).charAt(0);
          lines.push(
            "",
            `〔${nextWind}入前チェックポイント〕${prevWind}場終了時の点況:`,
            standingsLine(g.rounds[focus + 1].startScores, g.rounds[0].dealer),
            `#${chukan.id}（中間総括）を mj_add_comment で埋めてから mj_next_kyoku を再実行: ` +
              `順位状況と次場の各家の狙いを整理する`,
          );
        }
        throw new Error(lines.join("\n"));
      }

      // 2. ★-note hint for the round about to leave focus (its notes stay open
      //    until the next focus render locks them).
      const hint = starHint(s, focus);

      // 3. Final kyoku: no advance.
      if (focus >= lastRound) {
        return `all ${listAnchors(g).length} anchors filled — write the document with ` +
          `mj_weave_commentary, report the result to the user, and STOP.`;
      }

      // 4. Advance.
      const prevLabel = roundLabel(g, focus);
      s.focus = focus + 1;
      const newFocus = s.focus;
      const newLabel = roundLabel(g, newFocus);
      const unlocked = listAnchors(g).filter((b) => b.round <= newFocus);
      const filled = unlocked.filter((b) => s.comments.has(b.id)).length;

      const out: string[] = [
        `advanced: ${prevLabel} ✓ → focus ${newLabel}`,
        `(${filled}/${unlocked.length} anchors, ${s.notes.size} ★notes)`,
      ];
      if (hint) out.push(hint);
      if ((g.rounds[newFocus].kyoku >> 2) !== (g.rounds[focus].kyoku >> 2)) {
        out.push(
          `== ${newLabel.charAt(0)}入 ==`,
          standingsLine(g.rounds[newFocus].startScores, g.rounds[0].dealer),
        );
      }
      if (newFocus === lastRound) {
        const owari = listAnchors(g).find((b) => b.kind === "終局総括");
        out.push(
          `これが最終局（オーラス）で 終局総括${owari ? ` #${owari.id}` : ""} も focus に含まれる`,
        );
      }
      const ks = kyokuStart(g, String(newFocus));
      out.push(
        `次局 ${ks.label}: 親P${ks.dealer} / ドラ表示 ${ks.doraIndicator}` +
          `${ks.kyotaku ? ` / 供託 ${ks.kyotaku}` : ""}${ks.honba ? ` / ${ks.honba}本場` : ""}`,
      );
      out.push(
        `STOP: end this chat turn NOW — report progress on the finished kyoku to the user. ` +
          `Next turn, start with mj_render_kyoku("${newFocus}")`,
      );
      return out.join("\n");
    }),
);

server.registerTool(
  "mj_weave_commentary",
  {
    description:
      "Produce the finished commentary document: deterministically splice the session draft " +
      "(everything accumulated via mj_add_comment / mj_add_note) into a re-rendered transcript " +
      "and WRITE IT TO `out` — never copy transcript lines yourself; every fact line comes from " +
      "the renderer verbatim. UNGATED (weaves the whole game so far). Returns only a one-line " +
      "summary (filled/missing counts) — the document itself never enters the conversation; " +
      "unfilled anchors stay as placeholders (missing=keep). When anchors are still unfilled the " +
      "summary is prefixed with a loud `warning: partial weave` line — best run it once all kyoku " +
      "are commented.",
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
      const summary = weaveSummary(r, dest);
      if (r.missing.length > 0) {
        const remaining = s.game.rounds.length - (s.focus + 1);
        return `warning: partial weave — ${r.missing.length} anchors unfilled` +
          `（focus ${roundLabel(s.game, s.focus)}・残り${remaining}局）\n${summary}`;
      }
      return summary;
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
      "start scores with placements. GATED to the focus cursor. JSON.",
    inputSchema: { kyoku: KYOKU },
  },
  ({ kyoku }: { kyoku: string }) =>
    run(() => {
      const s = current();
      assertUnlocked(s, uniqueRound(s.game, kyoku), "mj_get_kyoku_start");
      return json(kyokuStart(s.game, kyoku));
    }),
);

server.registerTool(
  "mj_get_kyoku_result",
  {
    description:
      "Outcome(s) of one round: winner, tsumo/ron + discarder, winning tile, points/fu/limit " +
      "and yaku — or draw reason + tenpai seats. Multiple entries = double/triple ron. GATED to " +
      "the focus cursor. JSON.",
    inputSchema: { kyoku: KYOKU },
  },
  ({ kyoku }: { kyoku: string }) =>
    run(() => {
      const s = current();
      assertUnlocked(s, uniqueRound(s.game, kyoku), "mj_get_kyoku_result");
      return json(kyokuResults(s.game, kyoku));
    }),
);

server.registerTool(
  "mj_get_riichi_declarations",
  {
    description:
      "Riichi declarations with seat, junme, wait tiles, live (unseen) wait count at declaration " +
      "time, and the リーチ判断 anchor id. With `kyoku` → that round (GATED to the focus cursor). " +
      "Without `kyoku` → every declaration UP TO the focus kyoku, with a `（未開放局は含まず）` note " +
      "so absence isn't read as 'no more riichi'. JSON.",
    inputSchema: { kyoku: KYOKU.optional() },
  },
  ({ kyoku }: { kyoku?: string }) =>
    run(() => {
      const s = current();
      if (kyoku !== undefined) {
        assertUnlocked(s, uniqueRound(s.game, kyoku), "mj_get_riichi_declarations");
        return json(riichiDeclarations(s.game, kyoku));
      }
      const shown = riichiDeclarations(s.game).filter((d) => d.roundIndex <= s.focus);
      return `${json(shown)}\n（未開放局は含まず）`;
    }),
);

await server.connect(new StdioServerTransport());
