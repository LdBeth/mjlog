// The single seat interface. Human and CPU go through exactly the same path —
// only the driver differs (round.ts's generator is synchronous; the TUI wraps it
// in an async loop that awaits `decide`).

import type { Observation } from "./observe.ts";
import type { Action, PublicEvent } from "./types.ts";

export interface Policy {
  readonly name: string;
  /** Choose one of `obs.legal`. Returning something not in `legal` is an error. */
  decide(obs: Observation): Action | Promise<Action>;
  /** Optional: observe the table without being asked to act. */
  notify?(e: PublicEvent): void;
  /** Optional: reset per-match RNG or caches. */
  reset?(seed: number): void;
}

/** A policy that never awaits — required for the fast self-play driver. */
export interface SyncPolicy extends Policy {
  decide(obs: Observation): Action;
}
