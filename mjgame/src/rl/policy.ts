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
import { pickLesserEvil, violationPoints } from "../penalty/preview.ts";
import type { SyncPolicy } from "../policy.ts";
import type { Rng } from "../rng.ts";
import { sfc32 } from "../rng.ts";
import type { Action, Violation } from "../types.ts";
import { actionIndex, ACTIONS, resolve } from "./actionspace.ts";
import type { CachedEncoding, EncodingCache } from "./features.ts";
import { encode, encodeSeq, flatten, INPUT_LEN } from "./features.ts";
import type { Net } from "./net.ts";
import { closeNet, forward, loadNet, SEQ_INPUT_LEN, seqInput } from "./net.ts";

export interface PolicyOptions {
  /** 0 (default) = argmax; >0 = softmax sampling, larger being flatter. */
  temperature?: number;
  /**
   * Whether the dojo's veto applies to this seat's action set (default true).
   * Mirrors `HeuristicOptions.dojo`: off, the seat plays straight from
   * `obs.legal` and pays for whatever the ledger charges.
   */
  dojo?: boolean;
}

/** What a v3 net's `seqInput` is handed and ignores. */
const NO_SEQ = new Int8Array(0);

/**
 * The actions the dojo referee would let this seat make — `obs.legal` minus
 * everything the ledger would charge for.
 *
 * This is the veto `HeuristicPolicy` has always had (`compliant`,
 * `mandatoryKan`, `compliantDiscards`), given to the learned seat: the net
 * learns what SCORES, and roughly 60% of its violations were fouls the
 * speculative referee could have named before the action was taken. Pricing
 * them is not an option here — a net has no ledger term to outbid — so the
 * foul is simply removed from the support.
 *
 * TWO INVARIANTS the caller depends on. The list is never empty: when every
 * legal action is charged the full `obs.legal` comes back, because a filter
 * must never leave a seat unable to act (the heuristic's fallthrough pricing
 * is its analogue). And the list this returns is the SUPPORT the decision is
 * made over, which is what the trajectory's `mask` records — PPO's importance
 * ratio is only a ratio if the recorded mask is the distribution the action
 * was sampled from.
 *
 * Pure in the Observation: the preview reads the live Table, but only ever
 * hypothetically (mutate-and-rollback under `guarded`).
 */
export function compliantActions(obs: Observation): Action[] {
  const pv = obs.preview;
  // No dojo wired, or nothing to choose between: the filter cannot change the
  // decision, and asking the referee costs a table mutation apiece.
  if (!pv || obs.legal.length < 2) return obs.legal;

  // 立直後カン見送り is the one rule that fires on an OMISSION: in riichi,
  // passing up a kan that leaves the wait alone is itself the foul. So when
  // declining is charged, the only question is whether accepting is charged
  // more — the cheaper option wins, ties going to declining, which is
  // `pickLesserEvil` called with the decline first. Same reading as
  // `HeuristicPolicy.mandatoryKan`.
  if (obs.riichi[0] && obs.drawn !== null) {
    const skip = pv.skipKan(obs.drawn);
    if (skip.length > 0) {
      let best: { a: Action; vs: Violation[] } | null = null;
      for (const a of obs.legal) {
        if (a.t !== "ankan") continue;
        const vs = pv.kan(a, obs.drawn);
        if (!best || violationPoints(vs) < violationPoints(best.vs)) best = { a, vs };
      }
      // A single-element support, deliberately: it flows through the ordinary
      // pick path below, so a sampling seat still draws exactly one rng float.
      if (best && pickLesserEvil(skip, best.vs) === "b") return [best.a];
    }
  }

  const closed = obs.melds[0].every((m) => m.kind === "ankan");
  const ok: Action[] = [];
  for (const a of obs.legal) {
    // Everything the switch does not name — tsumo, ron, pass — is kept: nothing
    // previewable charges for taking a win or for standing pat, and declining a
    // win would only be 見逃し, which the engine's own rules punish harder.
    switch (a.t) {
      case "discard": {
        if (pv.discard(a, obs.drawn).length !== 0) continue;
        const info = obs.discardInfo.get(a.tile);
        // 片和了り and 後付け are charged at WIN time, so the preview cannot
        // see them — but the discard that walks into them is right here.
        // A split wait must not be left damaten: with riichi on offer the
        // declaring variant survives and the plain one does not (the
        // heuristic spells the same thing as `mustCure`).
        if (info?.katagari && !a.riichi) continue;
        // 後付け: only an OPEN hand is stuck with a yakuless tenpai; a closed
        // one can still cure the same shape by declaring.
        if (info?.yakuless && !closed) continue;
        break;
      }
      case "pon":
      case "chi":
        if (pv.call(a).length !== 0) continue;
        break;
      case "daiminkan":
        if (pv.call(a).length !== 0 || pv.kan(a, obs.drawn).length !== 0) continue;
        break;
      case "ankan":
      case "kakan":
        if (pv.kan(a, obs.drawn).length !== 0) continue;
        break;
    }
    ok.push(a);
  }
  return ok.length > 0 ? ok : obs.legal;
}

export class NeuralPolicy implements SyncPolicy, EncodingCache {
  readonly name: string;
  readonly sync = true;
  readonly net: Net;
  readonly temperature: number;
  /** Whether `decide` picks from `compliantActions(obs)` or from `obs.legal`. */
  readonly dojo: boolean;
  private rng: Rng;

  /**
   * The input vector, ONE buffer for the whole match: `flatten` refills
   * 0..1673 and the river encoder writes `z` into 1674..1737. Reuse is safe
   * because nothing downstream keeps it — `forward` reads it and returns a
   * fresh result — and it is invisible to the trajectory writer, which reads
   * `lastEncoding` (per-call arrays) and serialises by value on the spot.
   */
  private buf = new Float32Array(SEQ_INPUT_LEN);
  /** The 1674-wide prefix VIEW of `buf`: exactly what a v3 first layer takes. */
  private flat = this.buf.subarray(0, INPUT_LEN);
  /** The distinct legal slots of one decision, ascending, and their weights. */
  private slots = new Int32Array(ACTIONS);
  private weights = new Float64Array(ACTIONS);

  /**
   * The encoding of the decision just made, offered to a wrapper that would
   * otherwise encode the very same Observation a second time (`RecordingPolicy`
   * does exactly that). The arrays are freshly allocated per decision, so a
   * reader may hold them; the Observation is carried alongside so the reader can
   * check it is being offered the encoding of ITS decision and not a stale one.
   */
  lastEncoding: CachedEncoding | null = null;

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
    this.dojo = opts.dojo ?? true;
  }

  reset(seed: number): void {
    this.rng = sfc32(seed);
  }

  /**
   * Release the native contexts this seat's net holds — the one thing about a
   * seat that the garbage collector cannot clean up, and the reason a driver
   * that built one policy per match used to grow by ~4MB of native memory a
   * hanchan. Idempotent (`closeNet` clears what it frees), and the net keeps
   * working on the TS path afterwards.
   *
   * Only safe on a net this policy OWNS. A `Net` handed in by the caller may be
   * shared with other seats, so its owner closes it, not this.
   */
  close(): void {
    closeNet(this.net);
  }

  /** The seat's own stream — `decide` draws from it only when sampling. */
  random(): number {
    return this.rng.float();
  }

  /**
   * The DISTINCT slots `legal` occupies, ascending, into `this.slots`; returns
   * how many there are. Insertion into a sorted run, because `legal` is a
   * handful of actions and the sorted order is what makes the sampler below
   * accumulate its weights in exactly the slot order the 78-wide scan it
   * replaces did — the same sum, hence the same draw.
   */
  private collectSlots(legal: Action[]): number {
    const slots = this.slots;
    let n = 0;
    for (const a of legal) {
      const i = actionIndex(a);
      if (i < 0 || i >= ACTIONS) continue;
      let p = n;
      while (p > 0 && slots[p - 1] > i) p--;
      if (p > 0 && slots[p - 1] === i) continue; // two actions, one slot
      for (let q = n; q > p; q--) slots[q] = slots[q - 1];
      slots[p] = i;
      n++;
    }
    return n;
  }

  /**
   * One softmax draw over the `n` legal slots in `this.slots`. The shift by the
   * legal maximum is what keeps `exp` finite; an illegal slot is not in the list
   * at all and so contributes nothing, never a probability to be renormalised
   * away — as does a legal slot the net drove to −∞ itself.
   */
  private sample(logits: Float32Array, n: number): number {
    const slots = this.slots, weights = this.weights;
    let max = -Infinity;
    for (let j = 0; j < n; j++) {
      const v = logits[slots[j]];
      if (v > max) max = v;
    }

    let total = 0;
    for (let j = 0; j < n; j++) {
      const v = logits[slots[j]];
      if (v === -Infinity) {
        weights[j] = 0;
        continue;
      }
      const w = Math.exp((v - max) / this.temperature);
      weights[j] = w;
      total += w;
    }
    if (!(total > 0)) return -1; // no legal slot at all — the caller falls back

    const target = this.rng.float() * total;
    let acc = 0;
    let last = -1;
    for (let j = 0; j < n; j++) {
      if (weights[j] === 0) continue;
      last = slots[j];
      acc += weights[j];
      if (target < acc) return slots[j];
    }
    // Only reachable when float32 residue leaves `target` above the whole sum.
    return last;
  }

  decide(obs: Observation): Action {
    // A v4 net also reads the river token stream; `seqInput` hands back the
    // 1674-wide prefix untouched for a v3 one, so an old snapshot still plays
    // and benches through exactly this line. The seq is only built when it
    // will be read — and both halves are written into `this.buf` in place.
    const enc = encode(obs);
    const seq = this.net.attn ? encodeSeq(obs) : null;
    flatten(enc, this.buf);
    const input = seqInput(this.net, this.flat, seq ?? NO_SEQ, this.buf);
    const logits = forward(this.net, input);

    // The dojo's veto, applied to the SUPPORT rather than to the choice: a
    // 禁じ手 the referee would name is not offered to the argmax or to the
    // sampler at all. `legal` — not `obs.legal` — is therefore what every line
    // below reads, `resolve` included: two actions can share one slot and
    // differ in compliance (tedashi vs tsumogiri of the same type, aka vs
    // plain copy), and resolving against the unfiltered list would quietly
    // hand back the foul variant of the slot the net just picked. It is also
    // what `lastEncoding` offers the recorder, because PPO's importance ratio
    // needs the recorded mask to be the support the action was sampled from.
    const legal = this.dojo ? compliantActions(obs) : obs.legal;

    // Offered to a recording wrapper; see `lastEncoding`. `enc`/`seq` are this
    // call's own arrays — `this.buf` is not part of the offer.
    this.lastEncoding = { obs, planes: enc.planes, scalars: enc.scalars, seq, legal };

    // Legality is enforced by the mask, not by the network — and the mask is
    // `legal` itself, walked once. A slot no legal action names can never
    // win, which is what the −∞ scatter into a 78-wide vector used to say; a
    // slot the NET drove to −∞ cannot win either, exactly as before (the old
    // ascending scan started at −∞ with a strict `>`).
    let best = -1;
    if (this.temperature > 0) {
      best = this.sample(logits, this.collectSlots(legal));
    } else {
      // Ties go to the LOWEST slot: the ascending scan spelled that as a strict
      // `>`, and out of order it is `>` on the logit and `<` on the index.
      let bestVal = -Infinity;
      for (const a of legal) {
        const i = actionIndex(a);
        if (i < 0 || i >= ACTIONS) continue;
        const v = logits[i];
        if (v > bestVal || (v === bestVal && i < best)) {
          bestVal = v;
          best = i;
        }
      }
    }
    if (best < 0) return legal[0];

    const picked = resolve(best, legal, { drawn: obs.drawn, akaIds: obs.akaIds });
    // `resolve` only returns null for an unmasked slot, which `best` never is.
    return picked ?? legal[0];
  }
}
