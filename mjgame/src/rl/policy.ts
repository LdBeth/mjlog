// The learned seat: encode → forward → mask → pick → back to an engine action.
//
// The pick is greedy by default (`temperature` 0), and therefore fully
// deterministic: evaluation, benchmarks and the interactive game all run that
// way. With a positive temperature the same masked logits are turned into a
// softmax and SAMPLED from the seat's own rng — that is how PPO self-play
// collects data whose actions carry entropy for the trainer to learn from.
// The rng is constructed and reset per match either way, so a sampling seat
// reproduces exactly like a greedy one given the same seed.

import type { Observation } from "../observe.ts";
import type { SyncPolicy } from "../policy.ts";
import type { Rng } from "../rng.ts";
import { sfc32 } from "../rng.ts";
import type { Action } from "../types.ts";
import { actionIndex, ACTIONS, resolve } from "./actionspace.ts";
import { encode, encodeSeq, flatten } from "./features.ts";
import type { Net } from "./net.ts";
import { forward, loadNet, seqInput } from "./net.ts";

export interface PolicyOptions {
  /** 0 (default) = argmax; >0 = softmax sampling, larger being flatter. */
  temperature?: number;
}

export class NeuralPolicy implements SyncPolicy {
  readonly name: string;
  readonly sync = true;
  readonly net: Net;
  readonly temperature: number;
  private rng: Rng;

  /**
   * Loads the weights eagerly: a missing manifest must fail at startup. An
   * already-loaded `Net` is accepted too, for callers (tests, batched rollout
   * workers) that share one net across seats instead of re-reading the blob.
   *
   * A path gives this seat its OWN net, and so its own native context when the
   * Accelerate shim is in play; a shared `Net` shares that one context, which
   * is fine because seats decide one at a time inside a match — but a net must
   * never be handed across Workers, whose calls would genuinely overlap.
   */
  constructor(name: string, seed: number, net: string | Net, opts: PolicyOptions = {}) {
    this.name = name;
    this.rng = sfc32(seed);
    this.net = typeof net === "string" ? loadNet(net) : net;
    this.temperature = opts.temperature ?? 0;
  }

  reset(seed: number): void {
    this.rng = sfc32(seed);
  }

  /** The seat's own stream — `decide` draws from it only when sampling. */
  random(): number {
    return this.rng.float();
  }

  /**
   * One softmax draw over the legal slots. The shift by the legal maximum is
   * what keeps `exp` finite; illegal slots are −∞ before the shift and so
   * contribute an exact 0 weight, never a probability to be renormalised away.
   */
  private sample(masked: Float32Array): number {
    let max = -Infinity;
    for (let i = 0; i < ACTIONS; i++) if (masked[i] > max) max = masked[i];

    const weights = new Float64Array(ACTIONS);
    let total = 0;
    for (let i = 0; i < ACTIONS; i++) {
      if (masked[i] === -Infinity) continue;
      const w = Math.exp((masked[i] - max) / this.temperature);
      weights[i] = w;
      total += w;
    }
    if (!(total > 0)) return -1; // no legal slot at all — the caller falls back

    const target = this.rng.float() * total;
    let acc = 0;
    let last = -1;
    for (let i = 0; i < ACTIONS; i++) {
      if (weights[i] === 0) continue;
      last = i;
      acc += weights[i];
      if (target < acc) return i;
    }
    // Only reachable when float32 residue leaves `target` above the whole sum.
    return last;
  }

  decide(obs: Observation): Action {
    // A v4 net also reads the river token stream; `seqInput` returns `flat`
    // untouched for a v3 one, so an old snapshot still plays and benches
    // through exactly this line. The seq is only built when it will be read.
    const flat = flatten(encode(obs));
    const input = this.net.attn ? seqInput(this.net, flat, encodeSeq(obs)) : flat;
    const logits = forward(this.net, input);

    // Legality is enforced by the mask, not by the network: an illegal slot is
    // driven to −∞ so it can never win the argmax however confident it was.
    const masked = new Float32Array(ACTIONS).fill(-Infinity);
    for (const a of obs.legal) {
      const i = actionIndex(a, obs.akaIds);
      if (i >= 0 && i < ACTIONS) masked[i] = logits[i];
    }
    let best = -1;
    if (this.temperature > 0) {
      best = this.sample(masked);
    } else {
      // Ascending scan with a strict `>` ⇒ ties go to the lowest index.
      let bestVal = -Infinity;
      for (let i = 0; i < ACTIONS; i++) {
        if (masked[i] > bestVal) {
          bestVal = masked[i];
          best = i;
        }
      }
    }
    if (best < 0) return obs.legal[0];

    const picked = resolve(best, obs.legal, { drawn: obs.drawn, akaIds: obs.akaIds });
    // `resolve` only returns null for an unmasked slot, which `best` never is.
    return picked ?? obs.legal[0];
  }
}
