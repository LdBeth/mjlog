// CLI over the core query API. An LLM agent (or a human) can call every core
// capability from the shell; the MCP server exposes the same four verbs.
//
//   deno run --allow-read src/cli.ts [render] [--hands key|all] [--snapshots inline] <file>
//   deno run --allow-read src/cli.ts kyoku <sel> [--hands key|all] <file>
//   deno run --allow-read src/cli.ts anchors <file>
//   deno run --allow-read src/cli.ts snapshot (--anchor N | --kyoku <sel> --junme N) <file>
//
// <sel> = 0-based round index, or wind+number(+".honba"): "S3", "東1", "E1.2".

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
  type SnapshotQuery,
} from "./core.ts";
import type { RenderOptions } from "./model.ts";

function usage(): never {
  console.error(
    [
      "usage: cli.ts [command] [options] <file.mjlog|xml | tenhou.net URL>",
      "  render (default)  --hands key|all   --snapshots inline",
      "  kyoku <sel>       one round, self-contained (sel: S3 / 東1 / E1.2 / round index)",
      "  anchors           list commentary anchors (#id kind kyoku junme seat topic)",
      "  snapshot          --anchor N | --kyoku <sel> --junme N",
      "  facts <kind>      structured JSON: start <sel> | result <sel> | riichi [sel] | standings",
    ].join("\n"),
  );
  Deno.exit(2);
}

const COMMANDS = ["render", "kyoku", "anchors", "snapshot", "facts"] as const;

interface Args {
  cmd: (typeof COMMANDS)[number];
  file: string;
  sel?: string; // kyoku positional
  factKind?: string; // facts positional
  anchor?: number;
  kyoku?: string;
  junme?: number;
  opts: Partial<RenderOptions>;
}

function parseArgs(argv: string[]): Args {
  const args: Args = { cmd: "render", file: "", opts: {} };
  let i = 0;
  if ((COMMANDS as readonly string[]).includes(argv[0])) {
    args.cmd = argv[0] as Args["cmd"];
    i = 1;
  }
  const positional: string[] = [];
  const next = (flag: string): string => {
    const v = argv[++i];
    if (v === undefined) {
      console.error(`missing value for ${flag}`);
      usage();
    }
    return v;
  };
  for (; i < argv.length; i++) {
    const a = argv[i];
    const val = (flag: string) => a.startsWith(flag + "=") ? a.slice(flag.length + 1) : next(flag);
    if (a === "--hands" || a.startsWith("--hands=")) {
      args.opts.hands = val("--hands") === "all" ? "all" : "key";
    } else if (a === "--snapshots" || a.startsWith("--snapshots=")) {
      args.opts.snapshots = val("--snapshots") === "inline" ? "inline" : "none";
    } else if (a === "--anchor" || a.startsWith("--anchor=")) {
      args.anchor = Number(val("--anchor"));
    } else if (a === "--kyoku" || a.startsWith("--kyoku=")) {
      args.kyoku = val("--kyoku");
    } else if (a === "--junme" || a.startsWith("--junme=")) {
      args.junme = Number(val("--junme"));
    } else if (a === "-h" || a === "--help") {
      usage();
    } else if (!a.startsWith("-")) {
      positional.push(a);
    } else {
      usage();
    }
  }

  if (args.cmd === "kyoku") {
    if (positional.length !== 2) usage();
    [args.sel, args.file] = positional;
  } else if (args.cmd === "facts") {
    // facts <kind> [sel] <file>
    if (positional.length === 3) [args.factKind, args.sel, args.file] = positional;
    else if (positional.length === 2) [args.factKind, args.file] = positional;
    else usage();
  } else {
    if (positional.length !== 1) usage();
    args.file = positional[0];
  }
  return args;
}

function snapshotQuery(a: Args): SnapshotQuery {
  if (a.anchor !== undefined) {
    if (!Number.isInteger(a.anchor) || a.anchor < 1) {
      console.error("--anchor must be a positive anchor id");
      usage();
    }
    return { anchor: a.anchor };
  }
  if (a.kyoku !== undefined && a.junme !== undefined) {
    if (!Number.isInteger(a.junme) || a.junme < 0) {
      console.error("--junme must be a non-negative integer");
      usage();
    }
    return { kyoku: a.kyoku, junme: a.junme };
  }
  console.error("snapshot needs --anchor N, or --kyoku <sel> --junme N");
  usage();
}

if (import.meta.main) {
  const a = parseArgs(Deno.args);
  try {
    const game = await loadGame(a.file);
    switch (a.cmd) {
      case "render":
        console.log(renderGame(game, a.opts));
        break;
      case "kyoku":
        console.log(renderKyoku(game, a.sel!, a.opts));
        break;
      case "anchors":
        console.log(anchorTable(game));
        break;
      case "snapshot":
        console.log(getSnapshot(game, snapshotQuery(a)));
        break;
      case "facts": {
        const json = (v: unknown) => console.log(JSON.stringify(v, null, 1));
        const need = (): string => {
          if (a.sel === undefined) {
            console.error(`facts ${a.factKind} needs a kyoku selector`);
            usage();
          }
          return a.sel;
        };
        if (a.factKind === "start") json(kyokuStart(game, need()));
        else if (a.factKind === "result") json(kyokuResults(game, need()));
        else if (a.factKind === "riichi") json(riichiDeclarations(game, a.sel));
        else if (a.factKind === "standings") json(finalStandings(game));
        else usage();
        break;
      }
    }
  } catch (err) {
    console.error("error:", err instanceof Error ? err.message : String(err));
    Deno.exit(1);
  }
}
