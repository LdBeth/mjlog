// Legal-uniform baseline. Its only jobs are to exercise every branch of the
// game master in self-play smoke tests and to be the bar `ai/heuristic.ts`
// must clear.

import type { Observation } from "../observe.ts";
import type { SyncPolicy } from "../policy.ts";
import type { Rng } from "../rng.ts";
import { sfc32 } from "../rng.ts";
import type { Action } from "../types.ts";

export interface RandomOptions {
  /** Probability of taking a win when one is offered. 1 = always. */
  winRate?: number;
  /** Probability of accepting a call (pon/chi/kan) when offered. */
  callRate?: number;
  /** Probability of declaring riichi when a riichi discard is available. */
  riichiRate?: number;
}

export class RandomPolicy implements SyncPolicy {
  readonly name: string;
  readonly sync = true;
  private rng: Rng;
  private opts: Required<RandomOptions>;

  constructor(name: string, seed: number, opts: RandomOptions = {}) {
    this.name = name;
    this.rng = sfc32(seed);
    this.opts = {
      winRate: opts.winRate ?? 1,
      callRate: opts.callRate ?? 0.25,
      riichiRate: opts.riichiRate ?? 0.5,
    };
  }

  reset(seed: number): void {
    this.rng = sfc32(seed);
  }

  decide(obs: Observation): Action {
    const { legal } = obs;
    const pick = <T>(xs: T[]): T => xs[this.rng.int(xs.length)];

    // Always take a win when offered (a policy that passes on wins makes the
    // smoke test's round-termination invariant much harder to reason about).
    const wins = legal.filter((a) => a.t === "tsumo" || a.t === "ron");
    if (wins.length && this.rng.float() < this.opts.winRate) return pick(wins);

    const calls = legal.filter(
      (a) =>
        a.t === "pon" || a.t === "chi" || a.t === "daiminkan" ||
        a.t === "ankan" || a.t === "kakan",
    );
    if (calls.length && this.rng.float() < this.opts.callRate) return pick(calls);

    const discards = legal.filter((a) => a.t === "discard");
    if (discards.length) {
      const riichi = discards.filter((a) => a.t === "discard" && a.riichi);
      if (riichi.length && this.rng.float() < this.opts.riichiRate) return pick(riichi);
      const plain = discards.filter((a) => a.t === "discard" && !a.riichi);
      return pick(plain.length ? plain : discards);
    }

    const pass = legal.find((a) => a.t === "pass");
    if (pass) return pass;
    return legal[0];
  }
}
