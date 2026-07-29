// MCP server (stdio): exposes the mjrender query core to LLM agents — STATELESS.
//
// Redesigned for the 2026-07-28 MCP spec: the server holds no session. Every
// tool takes `log` (a local path or tenhou.net URL) — the explicit handle the
// model passes back on each call. The commentary draft being built against a
// log lives on DISK (draft.ts: ~/.mjrender/drafts/<sha256-of-xml>.json), so it
// survives restarts and reconnects with no restore step; concurrent writers
// are last-write-wins at whole-file granularity. A parsed-game cache keyed by
// path+mtime makes the per-call re-parse cheap — a transparent cache, not
// protocol state (a fresh process recomputes identically).
//
// Pacing is ADVISORY, not enforced: nothing is locked and any round is
// readable or commentable at any time. Tool descriptions and mj_draft_status
// recommend the loop — mj_outline once to orient, then per reply ONE kyoku:
// mj_render_kyoku → mj_get_snapshot at riichi/tenpai → mj_add_comment /
// mj_add_note → mj_draft_status for coverage. mj_weave_commentary splices the
// disk draft into a re-rendered transcript written to a file: the model never
// copies fact lines and the woven document never enters its context.

import { McpServer } from "@modelcontextprotocol/server";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";
import {
  getSnapshot,
  kyokuResults,
  kyokuStart,
  listAnchors,
  listStarSites,
  loadGameKeyed,
  renderKyoku,
  renderOutline,
  riichiDeclarations,
  roundLabel,
  type StarNote,
  uniqueRound,
  weaveCommentary,
  weaveSummary,
} from "./core.ts";
import { deleteDraft, draftPath, loadDraft, saveDraft } from "./draft.ts";
import { formatInstruction } from "./render.ts";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { isUrl } from "./load.ts";
import type { Beat, Game } from "./model.ts";

// ---- parsed-game cache (transparent; keyed by the log string) ----

interface Opened {
  mtime: number; // 0 for URL sources (a finished game's log is immutable)
  game: Game;
  key: string; // sha256 of the decoded XML — the draft key
}
const CACHE_MAX = 4;
const cache = new Map<string, Opened>();

async function openLog(log: string): Promise<Opened> {
  const mtime = isUrl(log) ? 0 : (await Deno.stat(log)).mtime?.getTime() ?? 0;
  const hit = cache.get(log);
  if (hit && hit.mtime === mtime) {
    cache.delete(log); // refresh LRU position
    cache.set(log, hit);
    return hit;
  }
  const { game, key } = await loadGameKeyed(log);
  const entry = { mtime, game, key };
  cache.delete(log);
  cache.set(log, entry);
  if (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value!);
  return entry;
}

// ---- disk-draft glue ----

// The draft file's {anchors, notes} lists, as maps for editing. Note keys are
// "round:junme:seat" with the round resolved against THIS game.
interface DraftMaps {
  byId: Map<number, string>;
  notes: Map<string, StarNote>;
}

const noteKey = (round: number, junme: number, seat: number): string => `${round}:${junme}:${seat}`;

async function openDraft(game: Game, key: string, log: string): Promise<DraftMaps> {
  const d = await loadDraft(key, log);
  const byId = new Map<number, string>();
  for (const { anchor, text } of d.anchors) byId.set(anchor, text);
  const notes = new Map<string, StarNote>();
  for (const n of d.notes) notes.set(noteKey(uniqueRound(game, n.kyoku), n.junme, n.seat), n);
  return { byId, notes };
}

/** Atomically replace the draft file; returns its path. */
function persist(key: string, log: string, d: DraftMaps): Promise<string> {
  return saveDraft(key, {
    version: 1,
    log,
    savedAt: "",
    anchors: [...d.byId].map(([anchor, text]) => ({ anchor, text })),
    notes: [...d.notes.values()],
  });
}

function coverage(game: Game, d: DraftMaps): string {
  return `draft: ${d.byId.size}/${listAnchors(game).length} comments, ${d.notes.size} notes`;
}

// Advisory ★-note nudge: rounds that have ★ sites but no saved note yet.
// Never blocks anything.
function starHint(game: Game, d: DraftMaps): string | null {
  const noted = new Set([...d.notes.values()].map((n) => uniqueRound(game, n.kyoku)));
  const bare = [...new Set(listStarSites(game).map((x) => x.round))].filter((r) => !noted.has(r));
  if (bare.length === 0) return null;
  const labels = bare.slice(0, 8).map((r) => roundLabel(game, r)).join(" ");
  return `HINT: ★注記のない局: ${labels}${bare.length > 8 ? ` …(+${bare.length - 8})` : ""} — ` +
    "印象的な★行（鳴き・危険打牌など）には mj_add_note で一言を添えられる（任意）";
}

const fmtBeat = (b: Beat): string =>
  `#${b.id}(${b.kind}・${b.junme}巡${b.seat !== undefined ? `P${b.seat}` : ""})`;

const elide = (items: string[], n = 16): string =>
  items.slice(0, n).join(" ") + (items.length > n ? ` …(+${items.length - n})` : "");

const checklist = (game: Game, d: DraftMaps): string[] =>
  listAnchors(game).map((b) =>
    `${d.byId.has(b.id) ? "✓" : "・"} #${b.id}\t${b.kind}\t` +
    `${roundLabel(game, b.round)}\t${b.junme}巡` +
    `${b.seat !== undefined ? `\tP${b.seat}` : "\t"}\t${b.topic}`
  );

// Mechanical batch checks for mj_add_comment — unknown id, duplicate in
// batch, empty text.
function stageComments(
  beats: Beat[],
  comments: Array<{ anchor: number; text: string }>,
): Array<{ anchor: number; text: string; beat: Beat }> {
  const staged: Array<{ anchor: number; text: string; beat: Beat }> = [];
  const seen = new Set<number>();
  for (const { anchor, text } of comments) {
    const beat = beats.find((b) => b.id === anchor);
    if (!beat) {
      throw new Error(
        `unknown anchor #${anchor} — this game has #1..#${beats.length} (mj_list_anchors)`,
      );
    }
    if (seen.has(anchor)) throw new Error(`anchor #${anchor} appears twice in this batch`);
    seen.add(anchor);
    if (!text.trim()) throw new Error(`empty comment for anchor #${anchor}`);
    staged.push({ anchor, text: text.trim(), beat });
  }
  return staged;
}

// ★-site existence check; `where` names the addressed round.
function assertStarSite(
  here: ReturnType<typeof listStarSites>,
  junme: number,
  seat: number,
  where: string,
): void {
  if (!here.some((x) => x.junme === junme && x.seat === seat)) {
    const list = here.map((x) => `${x.junme}巡P${x.seat}`).join(" ");
    throw new Error(
      `no ★ line for P${seat} at ${junme}巡 — ${where}` +
        (list ? `（★: ${list}）` : "（この局に★行なし）"),
    );
  }
}

type ToolResult = { content: Array<{ type: "text"; text: string }>; isError?: boolean };

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

// Draft-touching handlers run FIFO-serialized within this process: a client
// that pipelines requests (or an LLM batching tool calls) must never
// interleave two read-modify-write cycles on the draft file. Cross-process
// writers remain last-write-wins by design (see draft.ts).
let chain: Promise<unknown> = Promise.resolve();
function runLocked(fn: () => Promise<string> | string): Promise<ToolResult> {
  const next = chain.then(() => run(fn));
  chain = next;
  return next;
}

const LOG = z.string().describe(
  "Tenhou mjlog source: local file path (gzipped .mjlog or plain .xml), or a tenhou.net URL — " +
    "a replay link like https://tenhou.net/0/?log=<id>&tw=1 or the raw log endpoint. The server " +
    "is STATELESS: pass the SAME value to every mj_ tool; it is the handle that keys the " +
    "on-disk commentary draft",
);

const KYOKU = z.string().describe(
  'Round selector: wind+number like "S3" / "東1" (optionally ".honba", e.g. "E1.2" when a kyoku repeats), or a 0-based round index like "6"',
);

const readOnly = { readOnlyHint: true, idempotentHint: true };

function buildServer(): McpServer {
  const server = new McpServer({ name: "mjrender", version: "0.7.0" });

  server.registerTool(
    "mj_outline",
    {
      description:
        "OUTLINE of the whole game (crude, cheap): notation legend, players block, then per kyoku " +
        "only the header with start scores, the condensed result (winner/yaku/points/score " +
        "movements), and the 〔解説ポイント#N〕 anchor index — NO per-turn lines. Read this ONCE " +
        "per conversation to orient (results are NOT spoilers here) and return to it for recaps " +
        "at the 中間総括 (wind boundary) and 終局総括 (game end). The reply also reports the " +
        "on-disk draft's coverage, so a fresh process resumes exactly where commentary left off. " +
        "RECOMMENDED PACE: per-turn detail one round at a time — mj_render_kyoku, comment it via " +
        "mj_add_comment before moving on. Do NOT reproduce transcript lines yourself.",
      inputSchema: z.object({ log: LOG }),
      annotations: readOnly,
    },
    ({ log }) =>
      runLocked(async () => {
        const { game, key } = await openLog(log);
        const d = await openDraft(game, key, log);
        return [
          `${log} — ${coverage(game, d)}`,
          "",
          formatInstruction("fill"),
          "",
          renderOutline(game),
        ].join("\n");
      }),
  );

  server.registerTool(
    "mj_render_kyoku",
    {
      description:
        "Render ONE round (kyoku) in full per-turn detail. Board snapshots are embedded INLINE by " +
        "default. Anchor ids inside are game-global, so they agree with mj_list_anchors / " +
        "mj_get_snapshot. RECOMMENDED PACE: one kyoku per reply — study it and fill its anchors " +
        "with mj_add_comment before rendering the next (batch-reading rounds degrades commentary).",
      inputSchema: z.object({
        log: LOG,
        kyoku: KYOKU,
        hands: z.enum(["key", "all"]).optional(),
        snapshots: z.enum(["none", "inline"]).optional()
          .describe(
            "Inline board snapshots above each anchor (default inline; the 配牌評価 anchor " +
              "carries none — the deal block above it already shows every hand — and the " +
              "end-of-hand ◇結果時点の各家手牌 block is folded into the final snapshot, 振聴 " +
              "marks included) or omit them (none)",
          ),
      }),
      annotations: readOnly,
    },
    ({ log, kyoku, hands, snapshots }) =>
      run(async () => {
        const { game } = await openLog(log);
        return renderKyoku(game, kyoku, {
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
      description:
        "The full commentary checklist: every anchor of the game, one per line — ✓ (commented in " +
        "the on-disk draft) or ・ (unfilled), #id, kind (配牌評価/リーチ判断/押し引き/副露判断/" +
        "局総括/流局評価/中間総括/終局総括), kyoku, junme, seat, topic — plus the coverage line.",
      inputSchema: z.object({ log: LOG }),
      annotations: readOnly,
    },
    ({ log }) =>
      runLocked(async () => {
        const { game, key } = await openLog(log);
        const d = await openDraft(game, key, log);
        return [...checklist(game, d), coverage(game, d)].join("\n");
      }),
  );

  server.registerTool(
    "mj_get_snapshot",
    {
      description:
        "Recall the full board state at a position: all four rivers (▽=tsumogiri, *=riichi tile, " +
        "(→Pn)=called away), melds, live scores + placements, riichi states, dora, remaining " +
        "wall, and each seat's concealed hand with shanten/ukeire. Address by anchor id (from " +
        "the transcript's 〔解説ポイント#N〕 / mj_list_anchors), or by kyoku + junme (state at " +
        "the end of that go-around). ALWAYS check this at riichi declarations and tenpai moments " +
        "before writing commentary (リーチ判断/押し引き anchors) — do not judge them from the " +
        "outline alone.",
      inputSchema: z.object({
        log: LOG,
        anchor: z.number().int().positive().optional().describe("Anchor id #N"),
        kyoku: KYOKU.optional(),
        junme: z.number().int().nonnegative().optional()
          .describe("Go-around number (requires kyoku)"),
      }),
      annotations: readOnly,
    },
    ({ log, anchor, kyoku, junme }) =>
      run(async () => {
        const { game } = await openLog(log);
        if (anchor !== undefined) {
          if (!listAnchors(game).some((b) => b.id === anchor)) {
            throw new Error(`no anchor #${anchor} (use mj_list_anchors)`);
          }
          return getSnapshot(game, { anchor });
        }
        if (kyoku !== undefined && junme !== undefined) return getSnapshot(game, { kyoku, junme });
        throw new Error("provide either `anchor`, or both `kyoku` and `junme`");
      }),
  );

  server.registerTool(
    "mj_add_comment",
    {
      description:
        "Save commentary for one or MORE anchors into the on-disk draft — batch a kyoku's worth " +
        "per call (max 10) to conserve tool calls. Any anchor of any round is fillable at any " +
        "time; saving an already-commented anchor REPLACES it (revising past rounds at " +
        "中間総括/終局総括 time is encouraged). The batch is atomic (one bad entry saves " +
        "nothing) and the draft file is replaced atomically, so a crash never loses saved work. " +
        "Returns coverage and which anchors are still unfilled. ★-marked lines you meet in " +
        "kyoku renders can optionally get a one-liner via mj_add_note.",
      inputSchema: z.object({
        log: LOG,
        comments: z.array(z.object({
          anchor: z.number().int().positive().describe("Anchor id #N"),
          text: z.string().min(1).describe(
            "Commentary for this anchor (plain text, may be multiline)",
          ),
        })).min(1).max(10).describe("Anchor comments to save, one entry per anchor (max 10)"),
      }),
      annotations: { idempotentHint: true },
    },
    ({ log, comments }) =>
      runLocked(async () => {
        const { game, key } = await openLog(log);
        const d = await openDraft(game, key, log);
        const beats = listAnchors(game);
        const staged = stageComments(beats, comments);
        const replaced: number[] = [];
        for (const { anchor, text } of staged) {
          if (d.byId.has(anchor)) replaced.push(anchor);
          d.byId.set(anchor, text);
        }
        await persist(key, log, d);
        const ids = staged.map((c) => `#${c.anchor}`).join(" ");
        const open = beats.filter((b) => !d.byId.has(b.id)).map((b) => `#${b.id}`);
        const rest = open.length === 0
          ? " — all anchors filled; mj_weave_commentary writes the document"
          : ` / 未記入: ${elide(open)}`;
        return `saved ${ids}${
          replaced.length ? ` (replaced ${replaced.map((i) => `#${i}`).join(" ")})` : ""
        } — ${d.byId.size}/${beats.length}${rest}`;
      }),
  );

  server.registerTool(
    "mj_add_note",
    {
      description:
        "Save optional one-liners for ★-marked lines (notable discards/calls), addressed by " +
        "kyoku + junme + seat. One entry per ★ site, batched up to 10 per call. Saving the same " +
        "site again replaces it; EMPTY/blank text DELETES the saved note at that site. The batch " +
        "is atomic (one bad entry saves nothing) and lands in the on-disk draft. If the seat has " +
        "several ★ lines in one go-around (call then discard), the note lands after the last one.",
      inputSchema: z.object({
        log: LOG,
        notes: z.array(z.object({
          kyoku: KYOKU,
          junme: z.number().int().nonnegative().describe("Go-around number of the ★ line"),
          seat: z.number().int().min(0).max(3).describe("Acting seat 0-3 (P0-P3)"),
          text: z.string().describe(
            "Short one-liner for that ★ moment; empty/blank deletes the saved note",
          ),
        })).min(1).max(10).describe(
          "★-line notes to save or delete, one entry per ★ site (max 10)",
        ),
      }),
      annotations: { idempotentHint: true },
    },
    ({ log, notes }) =>
      runLocked(async () => {
        const { game, key } = await openLog(log);
        const d = await openDraft(game, key, log);
        const sites = listStarSites(game);
        // validate the whole batch before touching the draft
        const staged: Array<{ key: string; note?: StarNote; del: boolean; label: string }> = [];
        const seen = new Set<string>();
        for (const n of notes) {
          const round = uniqueRound(game, n.kyoku);
          const label = `${roundLabel(game, round)} ${n.junme}巡 P${n.seat}`;
          assertStarSite(
            sites.filter((x) => x.round === round),
            n.junme,
            n.seat,
            roundLabel(game, round),
          );
          const k = noteKey(round, n.junme, n.seat);
          if (seen.has(k)) throw new Error(`duplicate ★ note in this batch: ${label}`);
          seen.add(k);
          const t = n.text.trim();
          if (!t) {
            if (!d.notes.has(k)) throw new Error(`no saved ★ note to delete at ${label}`);
            staged.push({ key: k, del: true, label });
          } else {
            staged.push({
              key: k,
              del: false,
              note: { kyoku: String(round), junme: n.junme, seat: n.seat, text: t },
              label,
            });
          }
        }
        const saved: string[] = [], deleted: string[] = [], replaced: string[] = [];
        for (const x of staged) {
          if (x.del) {
            d.notes.delete(x.key);
            deleted.push(x.label);
          } else {
            if (d.notes.has(x.key)) replaced.push(x.label);
            d.notes.set(x.key, x.note!);
            saved.push(x.label);
          }
        }
        await persist(key, log, d);
        const parts: string[] = [];
        if (saved.length) {
          parts.push(
            `saved ${saved.length}: ${saved.join(" / ")}` +
              (replaced.length ? ` (replaced: ${replaced.join(" / ")})` : ""),
          );
        }
        if (deleted.length) parts.push(`deleted ${deleted.length}: ${deleted.join(" / ")}`);
        return `★ ${parts.join(" ; ")} — ${d.notes.size} note(s) in draft`;
      }),
  );

  server.registerTool(
    "mj_draft_status",
    {
      description:
        "Progress of the on-disk commentary draft for a log: every anchor as a checklist line " +
        "(✓ filled / ・ unfilled), the saved ★ notes, a ★-coverage hint, and the draft file's " +
        "path. Nothing is ever locked — this is the advisory view of what remains. RECOMMENDED " +
        "PACE: one kyoku per reply; revise any round any time.",
      inputSchema: z.object({ log: LOG }),
      annotations: readOnly,
    },
    ({ log }) =>
      runLocked(async () => {
        const { game, key } = await openLog(log);
        const d = await openDraft(game, key, log);
        const notes = [...d.notes.values()].map((n) =>
          `★ ${roundLabel(game, uniqueRound(game, n.kyoku))} ${n.junme}巡 P${n.seat}: ${n.text}`
        );
        const hint = starHint(game, d);
        return [
          `${log} — ${coverage(game, d)}`,
          `draft file: ${draftPath(key)}`,
          ...checklist(game, d),
          ...notes,
          ...(hint ? [hint] : []),
        ].join("\n");
      }),
  );

  server.registerTool(
    "mj_clear_draft",
    {
      description:
        "Delete the on-disk commentary draft for a log (all anchor comments and ★ notes). " +
        "Irreversible — the next mj_add_comment starts from scratch.",
      inputSchema: z.object({ log: LOG }),
      annotations: { destructiveHint: true, idempotentHint: true },
    },
    ({ log }) =>
      runLocked(async () => {
        const { key } = await openLog(log);
        return await deleteDraft(key)
          ? `deleted ${draftPath(key)}`
          : `no draft on disk for this log (${draftPath(key)})`;
      }),
  );

  server.registerTool(
    "mj_weave_commentary",
    {
      description:
        "Produce the finished commentary document: deterministically splice the on-disk draft " +
        "(everything saved via mj_add_comment / mj_add_note) into a re-rendered transcript and " +
        "WRITE IT TO `out` — never copy transcript lines yourself; every fact line comes from " +
        "the renderer verbatim. Returns only a one-line summary (filled/missing counts) — the " +
        "document itself never enters the conversation; unfilled anchors stay as placeholders " +
        "(missing=keep). When anchors are still unfilled the summary is prefixed with a loud " +
        "`warning: partial weave` line — best run it once every kyoku is commented.",
      inputSchema: z.object({
        log: LOG,
        out: z.string().describe(
          "Where to write the woven document (UTF-8). This server runs on the USER'S machine — " +
            "paths from your own sandbox/workspace do not exist here. Best: a bare filename like " +
            "'commentary.txt', which lands next to the log file (for URL sources: in the user's " +
            "home directory). The summary reports the absolute path — relay it to the user",
        ),
        missing: z.enum(["keep", "strip"]).optional()
          .describe(
            "Anchors you did not fill: keep their placeholder lines (default) or strip them",
          ),
        hands: z.enum(["key", "all"]).optional()
          .describe("Reconstructed-hand verbosity of the woven transcript (default key)"),
      }),
      annotations: { idempotentHint: true },
    },
    ({ log, out, missing, hands }) =>
      runLocked(async () => {
        const { game, key } = await openLog(log);
        const d = await openDraft(game, key, log);
        if (d.byId.size === 0 && d.notes.size === 0) {
          throw new Error(
            "draft is empty — save commentary first with mj_add_comment / mj_add_note",
          );
        }
        const r = weaveCommentary(game, {
          anchors: [...d.byId].map(([anchor, text]) => ({ anchor, text })),
          notes: [...d.notes.values()],
        }, { missing, hands });
        const dest = resolveOut(out, log);
        try {
          await Deno.writeTextFile(dest, r.text + "\n");
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          throw new Error(
            `cannot write ${dest}: ${msg}\nThis MCP server runs on the user's machine — paths ` +
              `from your own environment (e.g. /mnt/…) don't exist here. Pass a bare filename ` +
              `to write next to the log file, or an absolute path that exists on the user's ` +
              `machine.`,
          );
        }
        const summary = weaveSummary(r, dest);
        return r.missing.length > 0
          ? `warning: partial weave — ${r.missing.length} anchors unfilled: ${
            elide(r.missing.map((id) => `#${id}`))
          }\n${summary}`
          : summary;
      }),
  );

  server.registerTool(
    "mj_get_kyoku_start",
    {
      description:
        "Start conditions of one round: dealer, honba, kyotaku, dora indicator, and per-seat " +
        "start scores with placements. JSON.",
      inputSchema: z.object({ log: LOG, kyoku: KYOKU }),
      annotations: readOnly,
    },
    ({ log, kyoku }) => run(async () => json(kyokuStart((await openLog(log)).game, kyoku))),
  );

  server.registerTool(
    "mj_get_kyoku_result",
    {
      description:
        "Outcome(s) of one round: winner, tsumo/ron + discarder, winning tile, points/fu/limit " +
        "and yaku — or draw reason + tenpai seats. Multiple entries = double/triple ron. JSON.",
      inputSchema: z.object({ log: LOG, kyoku: KYOKU }),
      annotations: readOnly,
    },
    ({ log, kyoku }) => run(async () => json(kyokuResults((await openLog(log)).game, kyoku))),
  );

  server.registerTool(
    "mj_get_riichi_declarations",
    {
      description:
        "Riichi declarations with seat, junme, wait tiles, live (unseen) wait count at " +
        "declaration time, and the リーチ判断 anchor id. With `kyoku` → that round; without → " +
        "every declaration in the game. JSON.",
      inputSchema: z.object({ log: LOG, kyoku: KYOKU.optional() }),
      annotations: readOnly,
    },
    ({ log, kyoku }) => run(async () => json(riichiDeclarations((await openLog(log)).game, kyoku))),
  );

  return server;
}

// The caller may live in a different filesystem than this server (an agent
// sandbox vs the user's machine), so a relative `out` must resolve somewhere
// predictable from the input: beside the log file, or under $HOME for URL
// sources. The summary always reports the resulting absolute path.
function resolveOut(out: string, srcPath: string): string {
  if (isAbsolute(out)) return out;
  const base = isUrl(srcPath) ? Deno.env.get("HOME") ?? Deno.cwd() : dirname(resolve(srcPath));
  return join(base, out);
}

const json = (v: unknown) => JSON.stringify(v, null, 1);

serveStdio(buildServer);
