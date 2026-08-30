// League freeze — snapshot the CURRENT champion configuration as a complete,
// self-contained ktune JSON under `weights/league/`.
//
// The league practice (adopted 2026-08-25, see CLAUDE.md Decisions): at each
// meaningful improvement of the "k" seat, the champion configuration is frozen
// as `weights/league/frozen-<label>.json`, and future candidates are graded
// against MIXED fields of past snapshots (a `--table` file per field) rather
// than against a monoculture of the current self — the standard defense
// against style-specific self-play overfitting when there is no human corpus.
//
// What "complete" means: every section is dumped FULLY RESOLVED through the
// same merge functions the constructors use (`mergeHeuristic`/`mergeAugmented`/
// `mergeComputed`, and `mergeHand`/`mergeRiichi` when the source vector
// switches those models on). A later change to a DEFAULT_* constant therefore
// cannot reach back into a snapshot: every field the seat reads is written in
// the file. The exceptions are deliberate: `hand`/`riichi` are switches, so a
// source vector WITHOUT them yields a snapshot without them — dumping their
// defaults would switch the models ON.
//
// Self-check: before writing, the script plays verification hanchan with the
// ORIGINAL partial vector and with the RESOLVED dump on all four seats and
// requires bit-identical results — so a merge-semantics bug (a nested record
// the resolver spreads differently from a constructor) kills the freeze
// instead of producing a snapshot that silently differs from the seat it
// claims to be.
//
// Snapshots are pinned by `test/league_test.ts` and THOSE PINS NEVER
// REGENERATE. When a later default-field addition changes a snapshot's
// behaviour (a new field the old JSON does not carry), the fix is adding the
// explicit old value to the snapshot JSON — never re-pinning.
//
// Usage:
//   deno task freeze --label=0825                        # freeze the defaults
//   deno task freeze --label=1103 --ktune=weights/champion.json [--plan]

import { mergeAugmented } from "../src/ai/augmented.ts";
import { mergeComputed } from "../src/ai/computed.ts";
import { mergeDealin } from "../src/ai/dealin.ts";
import { mergeFold } from "../src/ai/fold.ts";
import { mergeHand } from "../src/ai/handvalue.ts";
import { mergeHeuristic } from "../src/ai/heuristic.ts";
import { mergeRiichi } from "../src/ai/riichi.ts";
import { mergeSense } from "../src/ai/sense.ts";
import { die } from "../src/cli/die.ts";
import { headless } from "../src/harness.ts";
import type { KTune, TableSpec } from "../src/spec.ts";
import { loadKtune } from "../src/spec.ts";

/** Same fingerprint as `test/frozen_test.ts` / `test/league_test.ts` — the
 * printed lines paste directly into the league pin table. */
function fingerprint(seed: number, table: TableSpec): string {
  const r = headless(1, seed, table).results[0];
  const body = JSON.stringify({
    scores: r.scores,
    outcomes: r.outcomes,
    ledger: r.ledger.map((v) => `${v.seat}:${v.label}`),
    riichis: r.riichis,
  });
  let x = 0x811c9dc5;
  for (let i = 0; i < body.length; i++) x = Math.imul(x ^ body.charCodeAt(i), 0x01000193) >>> 0;
  return `${r.scores.join("/")}#${x.toString(16).padStart(8, "0")}`;
}

/** The pin seeds — deliberately the epoch-pin seeds of `test/frozen_test.ts`,
 * so the first snapshot's fingerprints are comparable to the epoch's by eye. */
const PIN_SEEDS = [101, 505, 909];

function kTable(ktune: KTune | undefined, plan: boolean): TableSpec {
  const seat = { kind: "k" as const, ...(ktune ? { ktune } : {}), ...(plan ? { plan } : {}) };
  return [{ ...seat }, { ...seat }, { ...seat }, { ...seat }];
}

function main() {
  let label: string | undefined;
  let ktunePath: string | undefined;
  let plan = false;
  for (const a of Deno.args) {
    if (a.startsWith("--label=")) label = a.slice(8);
    else if (a.startsWith("--ktune=")) ktunePath = a.slice(8);
    else if (a === "--plan") plan = true;
    else die(`freeze が知らない引数です: ${a}\n使い方: --label=MMDD [--ktune=FILE] [--plan]`);
  }
  if (!label || !/^[0-9A-Za-z_-]+$/.test(label)) {
    die("--label=<英数字> は必須です (例: --label=0825 → frozen-0825.json)");
  }

  const src: KTune = ktunePath ? loadKtune(ktunePath) : {};
  const resolved: KTune = {
    heuristic: mergeHeuristic(src.heuristic),
    augment: mergeAugmented(src.augment),
    // `--plan` enters exactly where `makePolicy` injects it: under the vector,
    // so a source file that names `planner` outranks the flag.
    computed: mergeComputed({ planner: plan, ...src.computed }),
    ...(src.hand ? { hand: mergeHand(src.hand) } : {}),
    ...(src.riichi ? { riichi: mergeRiichi(src.riichi) } : {}),
    // 色読み is a switch like the two above: absent stays absent (identity),
    // present is resolved through its merge so the dump is complete.
    ...(src.sense ? { sense: mergeSense(src.sense) } : {}),
    // M13's fold head, likewise: absent stays absent (the incumbent gate),
    // present is resolved through `mergeFold` so the snapshot is complete —
    // and `mergeFold` VALIDATES, so a snapshot can never carry a block the
    // live seat would refuse.
    ...(src.fold ? { fold: mergeFold(src.fold) } : {}),
    // M14 likewise. `mergeDealin` VALIDATES (and refuses `{}`, which has no
    // identity here), so a snapshot can never carry a block the live seat
    // would reject.
    ...(src.dealin ? { dealin: mergeDealin(src.dealin) } : {}),
  };

  // The self-check: the resolved dump and the original configuration must be
  // the same seat, bit for bit, TODAY (that is the definition of "resolved").
  for (const seed of PIN_SEEDS) {
    const a = headless(1, seed, kTable(ktunePath ? src : undefined, plan));
    const b = headless(1, seed, kTable(resolved, false));
    if (JSON.stringify(a.results) !== JSON.stringify(b.results)) {
      die(
        `freeze 自己検証に失敗 (種${seed}): 解決済みダンプが元の席と一致しません。\n` +
          `マージ関数とコンストラクタがずれています — スナップショットは書きません。`,
      );
    }
  }

  const out = new URL(`../weights/league/frozen-${label}.json`, import.meta.url).pathname;
  Deno.mkdirSync(new URL("../weights/league", import.meta.url).pathname, { recursive: true });
  try {
    Deno.lstatSync(out);
    die(`${out} は既に在ります — 凍結は一度きりです (別の --label を)`);
  } catch (e) {
    if (!(e instanceof Deno.errors.NotFound)) throw e;
  }
  Deno.writeTextFileSync(out, JSON.stringify(resolved, null, 2) + "\n");

  console.log(`凍結: ${out}`);
  console.log(`元: ${ktunePath ?? "(既定値)"}${plan ? " --plan" : ""}`);
  console.log("test/league_test.ts の LEAGUE_PIN に貼る行:");
  console.log(`  "frozen-${label}.json": {`);
  const snap = loadKtune(out, "freeze 検証");
  for (const seed of PIN_SEEDS) {
    console.log(`    ${seed}: "${fingerprint(seed, kTable(snap, false))}",`);
  }
  console.log("  },");
}

if (import.meta.main) main();
