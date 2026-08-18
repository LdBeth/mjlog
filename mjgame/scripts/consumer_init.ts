// Write the INIT consumer: the curve set that reproduces the current
// hand-written discard score exactly.
//
//   deno run --allow-read --allow-write scripts/consumer_init.ts \
//     [--ktune=runs/tune/best.json] --out=weights/consumer-init.json
//
// With no `--ktune` the curves are built from `DEFAULT_WEIGHTS`; with one, from
// the SAME merge the policy constructor performs (defaults, then the file's
// `heuristic` section, with `danger` merged level by level so a partial
// override cannot drop a level). That matters: a fit that starts from a tuned
// baseline must start from the tuned baseline's score, not the default one.
//
// The file this writes is the fixed point of M9a — feeding it back through
// `--consumer` must leave every decision, and therefore every game, untouched.

import { DEFAULT_WEIGHTS } from "../src/ai/heuristic.ts";
import type { HeuristicWeights } from "../src/ai/heuristic.ts";
import { initFromWeights, serializeConsumer } from "../src/ai/consumer.ts";

function die(msg: string): never {
  console.error(msg);
  Deno.exit(2);
}

function main(argv: string[]): void {
  let out = "weights/consumer-init.json";
  let ktunePath = "";
  for (const arg of argv) {
    if (arg.startsWith("--out=")) out = arg.slice(6);
    else if (arg.startsWith("--ktune=")) ktunePath = arg.slice(8);
    else die(`不明なオプション: ${arg}\n使い方: consumer_init.ts [--ktune=PATH] [--out=PATH]`);
  }

  let w: HeuristicWeights = { ...DEFAULT_WEIGHTS, danger: { ...DEFAULT_WEIGHTS.danger } };
  if (ktunePath) {
    let json: unknown;
    try {
      json = JSON.parse(Deno.readTextFileSync(ktunePath));
    } catch (e) {
      die(`--ktune が読めません: ${ktunePath}\n${e instanceof Error ? e.message : e}`);
    }
    const k = json as { heuristic?: Partial<HeuristicWeights> };
    w = {
      ...DEFAULT_WEIGHTS,
      ...k.heuristic,
      danger: { ...DEFAULT_WEIGHTS.danger, ...k.heuristic?.danger },
    };
  }

  const params = initFromWeights(w);
  Deno.writeTextFileSync(out, serializeConsumer(params));
  console.log(
    `${out} を書きました (${ktunePath || "DEFAULT_WEIGHTS"} 由来、曲線17本・パラメータ68個)`,
  );
}

if (import.meta.main) main(Deno.args);
