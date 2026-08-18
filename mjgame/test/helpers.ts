// Test helpers. The `tiles()` notation parser is adapted from
// mjrender/test/units_test.ts:20, extended for the 雀鬼会 aka set (赤5筒 ×2).

import type { Tile } from "mjrender/model.ts";
import { dojoHooks } from "../src/dojo.ts";
import type { MatchResult } from "../src/match.ts";
import { runMatchSync } from "../src/match.ts";
import type { SyncPolicy } from "../src/policy.ts";
import { DOJO_HEADLESS, JANKI } from "../src/rules.ts";
import { scorer } from "../src/score.ts";
import type { Table } from "../src/table.ts";
import type { Seat } from "../src/types.ts";
import { SEATS } from "../src/types.ts";

const SUIT_BASE: Record<string, number> = { m: 0, p: 9, s: 18 };
const HONOR = "東南西北白發中";

/**
 * Parse compact notation into tile ids, allocating distinct copies of a type
 * in ascending id order: `tiles("123m0p55p東東")`.
 *
 * `0m` / `0p` / `0s` request an aka five. Under this project's ruleset only
 * pin has aka (ids 52 and 53), so `0p` twice yields both red 5-pin.
 */
export function tiles(spec: string): Tile[] {
  const used = new Map<number, number>();
  const out: Tile[] = [];

  // Copies of a type are handed out in ascending id order. That is also what
  // makes `0p` work: the aka 5-pin are ids 52 and 53, the two lowest copies.
  const take = (type: number): Tile => {
    const n = used.get(type) ?? 0;
    used.set(type, n + 1);
    return type * 4 + n;
  };

  let pending = "";
  for (const ch of spec) {
    if (ch === " ") continue;
    const h = HONOR.indexOf(ch);
    if (h >= 0) {
      out.push(take(27 + h));
      continue;
    }
    if (ch >= "0" && ch <= "9") {
      pending += ch;
      continue;
    }
    const base = SUIT_BASE[ch];
    if (base === undefined) throw new Error(`bad tile spec char: ${ch}`);
    for (const d of pending) {
      const r = Number(d);
      if (r === 0) out.push(take(base + 4)); // aka five
      else out.push(take(base + r - 1));
    }
    pending = "";
  }
  if (pending) throw new Error(`trailing digits in tile spec: ${pending}`);
  return out;
}

/**
 * One headless hanchan under the dojo's headless ruleset, with the driver's own
 * wiring: the penalty hooks (without which the ledger is always empty and
 * `tsumogiriLock` never arms) and the live-Table tap an oracle seat reads
 * through. `make` builds the policy for each seat and is called exactly once per
 * seat, in seat order, so a stateful provider may be constructed inside it.
 *
 * Shared rather than copied: `augmented_test.ts` measures oracle arms with it
 * and `computed_test.ts` measures 計算 arms, and a difference between the two
 * harnesses would be a difference in the measurement, not in the policy.
 */
export function playHanchan(
  seed: number,
  make: (seat: Seat, ref: { t: Table | null }) => SyncPolicy,
): MatchResult {
  const ref: { t: Table | null } = { t: null };
  const policies = SEATS.map((s) => make(s, ref));
  return runMatchSync(policies, {
    seed,
    cfg: JANKI,
    dojo: DOJO_HEADLESS,
    scorer,
    tableRef: ref,
    ...dojoHooks({ dojo: DOJO_HEADLESS, oracle: scorer }),
  });
}

/** Read a file from the repo root (one level above mjgame/). */
export function repoPath(name: string): string {
  return new URL(`../../${name}`, import.meta.url).pathname;
}
