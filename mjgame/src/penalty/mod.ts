// The dojo penalty registry.
//
// 雀鬼流 禁じ手 are NOT enforced here: every prohibited move is legal in
// `legal.ts` and goes through unchanged. This module watches actions after they
// commit and writes `Violation`s to a ledger, which `score.ts::finalStandings`
// then applies — a penalized player ranks below every clean player regardless of
// score, which is how the real 雀鬼会 works.
//
// Each rule is a pure predicate over a snapshot, so it is unit-testable from a
// fixture and individually disableable. Tier A rules are exact; Tier B rules
// approximate a judgment call and carry a `confidence` below 1 plus an evidence
// string, so a disputed call can be audited rather than silently trusted.

import type { Tile } from "mjrender/model.ts";
import type { WinOracle } from "../legal.ts";
import type { DojoConfig, RuleConfig } from "../rules.ts";
import type { Table } from "../table.ts";
import type { Action, Seat, Violation } from "../types.ts";

export type Hook =
  | "post-draw"
  | "post-discard"
  | "on-call"
  | "on-kan"
  | "on-riichi"
  | "on-win"
  | "on-round-end";

export interface RuleCtx {
  t: Table;
  seat: Seat;
  action: Action;
  /** The tile in hand from the draw, when the action followed one. */
  drawn: Tile | null;
  cfg: RuleConfig;
  dojo: DojoConfig;
  oracle: WinOracle;
  /** Wall-clock the seat took to decide; only present for a human at a TUI. */
  timing?: { elapsedMs: number; callPromptMs?: number };
}

export interface DojoRule {
  id: string;
  label: string; // Japanese, shown in the ledger
  tier: "A" | "B";
  points: number; // 評価点マイナス
  hooks: Hook[];
  check(ctx: RuleCtx): Array<Partial<Violation>> | null;
}

/** Fill in the fields every rule shares, so `check` only returns the specifics. */
function complete(rule: DojoRule, ctx: RuleCtx, part: Partial<Violation>): Violation {
  return {
    rule: rule.id,
    label: rule.label,
    seat: ctx.seat,
    kyoku: ctx.t.kyoku,
    junme: ctx.t.junme,
    points: ctx.dojo.weights[rule.id] ?? rule.points,
    tier: rule.tier,
    confidence: rule.tier === "A" ? 1 : 0.8,
    detail: "",
    ...part,
  };
}

export function makeRunner(rules: DojoRule[]) {
  const byHook = new Map<Hook, DojoRule[]>();
  for (const r of rules) {
    for (const h of r.hooks) {
      const list = byHook.get(h) ?? [];
      list.push(r);
      byHook.set(h, list);
    }
  }

  /**
   * `only` narrows the run to a single rule id. It is a pure optimization for
   * callers that would discard the rest anyway (`preview.ts::previewSkipKan`):
   * the result is exactly what filtering the full run by `v.rule === only`
   * would give, minus the predicate calls whose answers were thrown away.
   */
  return function runHook(hook: Hook, ctx: RuleCtx, only?: string): Violation[] {
    if (!ctx.dojo.enabled) return [];
    const out: Violation[] = [];
    for (const rule of byHook.get(hook) ?? []) {
      if (only !== undefined && rule.id !== only) continue;
      if (rule.tier === "B" && !ctx.dojo.tierB) continue;
      let parts: Array<Partial<Violation>> | null = null;
      try {
        parts = rule.check(ctx);
      } catch (err) {
        // A buggy rule must never take the game down with it.
        parts = [{
          detail: `rule ${rule.id} threw: ${err instanceof Error ? err.message : err}`,
          confidence: 0,
          points: 0,
        }];
      }
      for (const p of parts ?? []) out.push(complete(rule, ctx, p));
    }
    return out;
  };
}
