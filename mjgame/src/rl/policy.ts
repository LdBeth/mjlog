// The learned seat: encode → forward → mask → argmax → back to an engine action.
//
// v1 is greedy and therefore fully deterministic; the rng is still constructed
// and reset per match so that switching to sampling (or ε-greedy exploration
// for self-play data) is a change of two lines and not of the interface.

import type { Observation } from "../observe.ts";
import type { SyncPolicy } from "../policy.ts";
import type { Rng } from "../rng.ts";
import { sfc32 } from "../rng.ts";
import type { Action } from "../types.ts";
import { actionIndex, ACTIONS, resolve } from "./actionspace.ts";
import { encode, flatten } from "./features.ts";
import type { Net } from "./net.ts";
import { forward, loadNet } from "./net.ts";

export class NeuralPolicy implements SyncPolicy {
  readonly name: string;
  readonly sync = true;
  readonly net: Net;
  private rng: Rng;

  /** Loads the weights eagerly: a missing manifest must fail at startup. */
  constructor(name: string, seed: number, manifestPath: string) {
    this.name = name;
    this.rng = sfc32(seed);
    this.net = loadNet(manifestPath);
  }

  reset(seed: number): void {
    this.rng = sfc32(seed);
  }

  /** Reserved for sampling / ε-greedy self-play; v1's `decide` is greedy. */
  random(): number {
    return this.rng.float();
  }

  decide(obs: Observation): Action {
    const logits = forward(this.net, flatten(encode(obs)));

    // Legality is enforced by the mask, not by the network: an illegal slot is
    // driven to −∞ so it can never win the argmax however confident it was.
    const masked = new Float32Array(ACTIONS).fill(-Infinity);
    for (const a of obs.legal) {
      const i = actionIndex(a, obs.akaIds);
      if (i >= 0 && i < ACTIONS) masked[i] = logits[i];
    }
    // Ascending scan with a strict `>` ⇒ ties go to the lowest index.
    let best = -1;
    let bestVal = -Infinity;
    for (let i = 0; i < ACTIONS; i++) {
      if (masked[i] > bestVal) {
        bestVal = masked[i];
        best = i;
      }
    }
    if (best < 0) return obs.legal[0];

    const picked = resolve(best, obs.legal, { drawn: obs.drawn, akaIds: obs.akaIds });
    // `resolve` only returns null for an unmasked slot, which `best` never is.
    return picked ?? obs.legal[0];
  }
}
