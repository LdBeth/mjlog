// Reusable core: path → LLM-ready Japanese transcript.
// A future MCP server wraps THIS function unchanged.

import { loadXml } from "./load.ts";
import { parseGame } from "./parse.ts";
import { renderGame } from "./render.ts";
import type { RenderOptions } from "./model.ts";

export async function render(path: string, opts: Partial<RenderOptions> = {}): Promise<string> {
  const xml = await loadXml(path);
  const game = parseGame(xml);
  if (game.rules.sanma) {
    throw new Error("三人打ち（サンマ）はこのバージョンでは未対応です。");
  }
  const options: RenderOptions = { player: opts.player, hands: opts.hands ?? "key" };
  return renderGame(game, options);
}
