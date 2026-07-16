// MCP server (stdio): exposes the mjrender query core to LLM agents.
//
//   deno task mcp        (equivalent: deno run --allow-read src/mcp.ts)
//
// All tools are thin wrappers over core.ts and share the mj_ name prefix —
// mj_render_game, mj_render_kyoku, mj_list_anchors, mj_get_snapshot, plus the
// structured fact tools. The intended flow: the agent renders the (lean)
// transcript once, then recalls full board state for any 〔解説ポイント#N〕
// anchor — or any explicit kyoku+junme — while writing commentary.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import {
  anchorTable,
  finalStandings,
  getSnapshot,
  kyokuResults,
  kyokuStart,
  loadGame,
  renderGame,
  renderKyoku,
  riichiDeclarations,
} from "./core.ts";
import { isUrl } from "./load.ts";
import type { Game } from "./model.ts";

// Parsed-game cache keyed by path, invalidated when the file's mtime changes.
// URL sources have no mtime; a finished game's log is immutable, so 0 pins
// them in the cache for the process lifetime.
const cache = new Map<string, { mtime: number; game: Game }>();
async function game(path: string): Promise<Game> {
  const mtime = isUrl(path) ? 0 : (await Deno.stat(path)).mtime?.getTime() ?? 0;
  const hit = cache.get(path);
  if (hit && hit.mtime === mtime) return hit.game;
  const g = await loadGame(path);
  cache.set(path, { mtime, game: g });
  return g;
}

type ToolResult = { content: Array<{ type: "text"; text: string }>; isError?: boolean };

// Handler arg shapes, stated explicitly: the SDK's zod-based inference degrades
// to `any` when `zod` resolves to a different npm instance than the SDK's own.
interface RenderArgs {
  path: string;
  hands?: "key" | "all";
  snapshots?: "none" | "inline";
}
interface KyokuArgs extends RenderArgs {
  kyoku: string;
}
interface SnapshotArgs {
  path: string;
  anchor?: number;
  kyoku?: string;
  junme?: number;
}

function ok(text: string): ToolResult {
  return { content: [{ type: "text", text }] };
}

async function run(fn: () => Promise<string>): Promise<ToolResult> {
  try {
    return ok(await fn());
  } catch (e) {
    return {
      content: [{ type: "text", text: `error: ${e instanceof Error ? e.message : String(e)}` }],
      isError: true,
    };
  }
}

const PATH = z.string().describe(
  "Tenhou mjlog source: local file path (gzipped .mjlog or plain .xml), or a tenhou.net URL — " +
    "a replay link like https://tenhou.net/0/?log=<id>&tw=1 or the raw log endpoint",
);
const KYOKU = z.string().describe(
  'Round selector: wind+number like "S3" / "東1" (optionally ".honba", e.g. "E1.2" when a kyoku repeats), or a 0-based round index like "6"',
);

const server = new McpServer({ name: "mjrender", version: "0.2.0" });

server.registerTool(
  "mj_render_game",
  {
    description: "Render a full Tenhou game log as an LLM-ready Japanese commentary transcript. " +
      "Fact lines + computed metrics (shanten/ukeire/waits/dora/danger) + 〔解説ポイント#N〕 " +
      "commentary anchors whose #N ids are addressable via mj_get_snapshot.",
    inputSchema: {
      path: PATH,
      hands: z.enum(["key", "all"]).optional()
        .describe("Reconstructed-hand verbosity: key beats only (default) or every turn"),
      snapshots: z.enum(["none", "inline"]).optional()
        .describe("inline = embed a full board snapshot above every anchor (token-heavy)"),
    },
  },
  ({ path, hands, snapshots }: RenderArgs) =>
    run(async () => renderGame(await game(path), { hands, snapshots })),
);

server.registerTool(
  "mj_render_kyoku",
  {
    description:
      "Render ONE round (kyoku) of the game, self-contained (format preamble included). " +
      "Anchor ids inside are game-global, so they agree with mj_list_anchors/mj_get_snapshot.",
    inputSchema: {
      path: PATH,
      kyoku: KYOKU,
      hands: z.enum(["key", "all"]).optional(),
      snapshots: z.enum(["none", "inline"]).optional(),
    },
  },
  ({ path, kyoku, hands, snapshots }: KyokuArgs) =>
    run(async () => renderKyoku(await game(path), kyoku, { hands, snapshots })),
);

server.registerTool(
  "mj_list_anchors",
  {
    description: "List every commentary anchor of the game, one per line: " +
      "#id, kind (配牌評価/リーチ判断/押し引き/局総括/流局評価/終局総括), kyoku, junme, seat, topic.",
    inputSchema: { path: PATH },
  },
  ({ path }: { path: string }) => run(async () => anchorTable(await game(path))),
);

server.registerTool(
  "mj_get_snapshot",
  {
    description:
      "Recall the full board state at a position: all four rivers (▽=tsumogiri, *=riichi tile, " +
      "(→Pn)=called away), melds, live scores + placements, riichi states, dora, remaining wall, " +
      "and each seat's concealed hand with shanten/ukeire. Address by anchor id (from the " +
      "transcript's 〔解説ポイント#N〕 / mj_list_anchors), or by kyoku + junme (state at the end of " +
      "that go-around).",
    inputSchema: {
      path: PATH,
      anchor: z.number().int().positive().optional().describe("Anchor id #N"),
      kyoku: KYOKU.optional(),
      junme: z.number().int().nonnegative().optional()
        .describe("Go-around number (requires kyoku)"),
    },
  },
  ({ path, anchor, kyoku, junme }: SnapshotArgs) =>
    run(async () => {
      const g = await game(path);
      if (anchor !== undefined) return getSnapshot(g, { anchor });
      if (kyoku !== undefined && junme !== undefined) return getSnapshot(g, { kyoku, junme });
      throw new Error("provide either `anchor`, or both `kyoku` and `junme`");
    }),
);

// ---- structured fact tools (JSON responses; same source as the transcript) ----

const json = (v: unknown) => JSON.stringify(v, null, 1);

server.registerTool(
  "mj_get_kyoku_start",
  {
    description:
      "Start conditions of one round: dealer, honba, kyotaku, dora indicator, and per-seat " +
      "start scores with placements. JSON.",
    inputSchema: { path: PATH, kyoku: KYOKU },
  },
  ({ path, kyoku }: { path: string; kyoku: string }) =>
    run(async () => json(kyokuStart(await game(path), kyoku))),
);

server.registerTool(
  "mj_get_kyoku_result",
  {
    description:
      "Outcome(s) of one round: winner, tsumo/ron + discarder, winning tile, points/fu/limit " +
      "and yaku — or draw reason + tenpai seats. Multiple entries = double/triple ron. JSON.",
    inputSchema: { path: PATH, kyoku: KYOKU },
  },
  ({ path, kyoku }: { path: string; kyoku: string }) =>
    run(async () => json(kyokuResults(await game(path), kyoku))),
);

server.registerTool(
  "mj_get_riichi_declarations",
  {
    description:
      "Every riichi declaration (whole game, or one kyoku): seat, junme, wait tiles, live " +
      "(unseen) wait count at declaration time, and the リーチ判断 anchor id. JSON.",
    inputSchema: { path: PATH, kyoku: KYOKU.optional() },
  },
  ({ path, kyoku }: { path: string; kyoku?: string }) =>
    run(async () => json(riichiDeclarations(await game(path), kyoku))),
);

server.registerTool(
  "mj_get_final_standings",
  {
    description: "Final standings: place, seat, name, score, placement points. JSON.",
    inputSchema: { path: PATH },
  },
  ({ path }: { path: string }) =>
    run(async () => {
      const s = finalStandings(await game(path));
      if (!s) throw new Error("log has no 終局 record (game did not finish?)");
      return json(s);
    }),
);

await server.connect(new StdioServerTransport());
