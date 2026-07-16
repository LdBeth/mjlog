// CLI: render a Tenhou mjlog to an LLM-ready Japanese commentary transcript.
//
//   deno run --allow-read src/cli.ts [--hands key|all] <file.mjlog|xml>

import { render } from "./core.ts";
import type { RenderOptions } from "./model.ts";

function usage(): never {
  console.error(
    "usage: deno run --allow-read src/cli.ts [--hands key|all] <file.mjlog|xml>",
  );
  Deno.exit(2);
}

function parseArgs(argv: string[]): { file: string; opts: Partial<RenderOptions> } {
  let hands: "key" | "all" = "key";
  let file: string | undefined;

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--hands") hands = argv[++i] === "all" ? "all" : "key";
    else if (a.startsWith("--hands=")) hands = a.slice("--hands=".length) === "all" ? "all" : "key";
    else if (a === "-h" || a === "--help") usage();
    else if (!a.startsWith("-")) file = a;
    else usage();
  }
  if (!file) usage();
  return { file, opts: { hands } };
}

if (import.meta.main) {
  const { file, opts } = parseArgs(Deno.args);
  try {
    console.log(await render(file, opts));
  } catch (err) {
    console.error("error:", err instanceof Error ? err.message : String(err));
    Deno.exit(1);
  }
}
