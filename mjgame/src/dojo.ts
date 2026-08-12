// Wiring between the game master and the penalty registry: map each committed
// action onto the hooks it should fire, run them, and file the results in the
// table's ledger.

import type { Tile } from "mjrender/model.ts";
import type { WinOracle } from "./legal.ts";
import { ANY_WIN } from "./legal.ts";
import type { Hook, RuleCtx } from "./penalty/mod.ts";
import { runHook } from "./penalty/rules.ts";
import type { DojoConfig } from "./rules.ts";
import type { Table } from "./table.ts";
import type { Action, RoundOutcome, Seat } from "./types.ts";
import { SEATS } from "./types.ts";

export interface Timing {
  elapsedMs: number;
  /** How long a call prompt sat open before being dismissed — the 腰 signal. */
  callPromptMs?: number;
}

export interface DojoHooksOptions {
  dojo: DojoConfig;
  oracle?: WinOracle;
  /** Supplied by the TUI; headless play has no timing signal, by design. */
  timing?: (seat: Seat) => Timing | undefined;
}

function hooksFor(action: Action): Hook[] {
  switch (action.t) {
    case "discard":
      return action.riichi ? ["post-discard", "on-riichi"] : ["post-discard"];
    case "pon":
    case "chi":
    case "daiminkan":
      return ["on-call"];
    case "ankan":
    case "kakan":
      return ["on-kan"];
    case "tsumo":
    case "ron":
      return ["on-win"];
    default:
      return [];
  }
}

export function dojoHooks(opts: DojoHooksOptions) {
  const oracle = opts.oracle ?? ANY_WIN;

  const ctxFor = (t: Table, seat: Seat, action: Action, drawn: Tile | null): RuleCtx => ({
    t,
    seat,
    action,
    drawn,
    cfg: t.cfg,
    dojo: opts.dojo,
    oracle,
    timing: opts.timing?.(seat),
  });

  return {
    onAction(t: Table, seat: Seat, action: Action, drawn: Tile | null): void {
      const ctx = ctxFor(t, seat, action, drawn);
      for (const h of hooksFor(action)) {
        for (const v of runHook(h, ctx)) t.addViolation(v);
      }
    },

    onRoundEnd(t: Table, _outcome: RoundOutcome): void {
      for (const seat of SEATS) {
        const ctx = ctxFor(t, seat, { t: "pass" }, null);
        for (const v of runHook("on-round-end", ctx)) t.addViolation(v);
      }
    },
  };
}
