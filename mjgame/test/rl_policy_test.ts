// `NeuralPolicy`'s two decision rules: greedy argmax (temperature 0, what the
// interactive game and every evaluation run use) and softmax sampling (positive
// temperature, what PPO self-play collects rollouts with).
//
// The net here is built IN MEMORY with a zero weight matrix, so every output is
// exactly its bias: the logits are chosen by the test rather than inferred from
// an observation, which is what makes "this is the argmax" and "these two are
// close enough that both must appear" statements about the policy and not about
// the encoder.

import { assert, assertEquals } from "@std/assert";
import type { Meld } from "mjrender/model.ts";
import type { Observation } from "../src/observe.ts";
import type { Action } from "../src/types.ts";
import { actionIndex, ACTIONS } from "../src/rl/actionspace.ts";
import { FEATURES, INPUT_LEN } from "../src/rl/features.ts";
import type { LayerSpec, Net } from "../src/rl/net.ts";
import { NeuralPolicy } from "../src/rl/policy.ts";
import { JANKI } from "../src/rules.ts";
import { tiles } from "./helpers.ts";

// ---------------------------------------------------------------------------
// a net whose logits the test dictates
// ---------------------------------------------------------------------------

/**
 * One `1263→79` layer with an all-zero weight matrix: `forward` then returns the
 * bias vector for ANY input. `logits[i]` is output i; the value head (78) is
 * left at 0 and must stay ignored.
 */
function constantNet(logits: Record<number, number>): Net {
  const OUT = FEATURES.actions + 1;
  const spec: LayerSpec = { in: INPUT_LEN, out: OUT, act: "none" };
  const b = new Float32Array(OUT);
  for (const [i, v] of Object.entries(logits)) b[Number(i)] = v;
  return {
    manifest: {
      version: 1,
      arch: "mlp",
      features: { planes: FEATURES.planes, scalars: FEATURES.scalars },
      actions: FEATURES.actions,
      layers: [spec],
      blob: "(in-memory)",
    },
    path: "(in-memory)",
    layers: [{ ...spec, w: new Float32Array(OUT * INPUT_LEN), b }],
    outputs: OUT,
  };
}

// ---------------------------------------------------------------------------
// the observation
// ---------------------------------------------------------------------------

// Same fixture shape as rl_features_test.ts, trimmed to what a discard decision
// needs. The hand holds one copy each of 1m/2m/3m/4m, so the four legal discards
// land on action slots 0–3 and nothing else in the space is reachable.
const HAND = tiles("1234m56789m1234p");
const PON = tiles("白白白");
const INDICATOR = tiles("9s");

function discard(tile: number): Action {
  return { t: "discard", tile, riichi: false, tsumogiri: false };
}

function baseObs(over: Partial<Observation> = {}): Observation {
  const pon: Meld = { kind: "pon", who: 2, fromWho: 1, tiles: PON, calledTile: PON[2] };
  return {
    seat: 0,
    kyoku: 1,
    honba: 0,
    kyotaku: 0,
    junme: 6,
    wallRemaining: 42,
    hand: HAND,
    drawn: null,
    melds: [[], [], [pon], []],
    rivers: [[], [], [], []],
    scores: [25000, 25000, 25000, 25000],
    riichi: [false, false, false, false],
    riichiJunme: [-1, -1, -1, -1],
    doraIndicators: INDICATOR,
    seatWind: 27,
    roundWind: 27,
    akaIds: JANKI.akaIds,
    shanten: 2,
    waits: [],
    ronnable: [],
    katagari: false,
    ukeire: [],
    doraCount: 0,
    furiten: { permanent: false, temporary: false, riichi: false },
    discardInfo: new Map(),
    tsumogiriLock: false,
    danger: new Map(),
    violations: [0, 0, 0, 0],
    legal: HAND.slice(0, 4).map(discard),
    ...over,
  };
}

/** The masked argmax, recomputed here the way the greedy path defines it. */
function greedyIndex(obs: Observation, logits: Record<number, number>): number {
  const masked = new Float32Array(ACTIONS).fill(-Infinity);
  for (const a of obs.legal) {
    const i = actionIndex(a);
    if (i >= 0 && i < ACTIONS) masked[i] = logits[i] ?? 0;
  }
  let best = -1;
  let bestVal = -Infinity;
  for (let i = 0; i < ACTIONS; i++) {
    if (masked[i] > bestVal) {
      bestVal = masked[i];
      best = i;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// 1. temperature 0 is the greedy rule, unchanged
// ---------------------------------------------------------------------------

Deno.test("policy: 温度0は合法手のargmax、既定も温度0", () => {
  // Slot 2 (discard 3m) wins; slot 40 is an illegal riichi discard with a much
  // larger logit, and must lose anyway.
  const logits = { 0: 0.5, 1: 1.5, 2: 2.5, 3: -1, 40: 99 };
  const obs = baseObs();
  const want = greedyIndex(obs, logits);
  assertEquals(want, 2, "the fixture's argmax");

  const explicit = new NeuralPolicy("N", 7, constantNet(logits), { temperature: 0 });
  const implicit = new NeuralPolicy("N", 7, constantNet(logits));
  assertEquals(implicit.temperature, 0, "temperature defaults to 0");

  const picked = explicit.decide(obs);
  assertEquals(picked, obs.legal[2], "by identity — the engine compares by reference");
  assertEquals(implicit.decide(obs), picked, "omitting opts is the same decision");
  // Greedy stays greedy however often it is asked, and never touches the rng.
  for (let i = 0; i < 5; i++) assertEquals(explicit.decide(obs), picked);
});

Deno.test("policy: 温度0の同点は最小の行動番号に落ちる", () => {
  // Three-way tie across slots 0–2: the ascending scan with a strict `>` keeps
  // the lowest, and slot 3 sits just below so the tie is the only thing tested.
  const logits = { 0: 1, 1: 1, 2: 1, 3: 0.9 };
  const obs = baseObs();
  assertEquals(greedyIndex(obs, logits), 0);
  const p = new NeuralPolicy("N", 1, constantNet(logits));
  assertEquals(p.decide(obs), obs.legal[0]);
});

// ---------------------------------------------------------------------------
// 2. sampling stays inside the mask
// ---------------------------------------------------------------------------

Deno.test("policy: 温度>0でも選ばれるのは必ず合法手", () => {
  // Every illegal slot is louder than every legal one: only the mask can keep
  // the sampler honest.
  const logits: Record<number, number> = { 0: 0.1, 1: 0.2, 2: 0.15, 3: 0.05 };
  for (let i = 4; i < ACTIONS; i++) logits[i] = 5;
  const obs = baseObs();
  const p = new NeuralPolicy("N", 20260814, constantNet(logits), { temperature: 1 });
  const seen = new Set<Action>();
  for (let i = 0; i < 200; i++) {
    const a = p.decide(obs);
    assert(obs.legal.includes(a), `sampled an action outside legal: ${JSON.stringify(a)}`);
    seen.add(a);
  }
  assertEquals(seen.size, 4, "all four legal discards are reachable");
});

// ---------------------------------------------------------------------------
// 3. determinism
// ---------------------------------------------------------------------------

Deno.test("policy: 同じシード・同じ温度なら同じ手順を再現する", () => {
  const logits = { 0: 1, 1: 0.9, 2: 0.2, 3: -0.5 };
  const obs = baseObs();
  const run = (seed: number): number[] => {
    const p = new NeuralPolicy("N", seed, constantNet(logits), { temperature: 0.8 });
    return Array.from({ length: 40 }, () => obs.legal.indexOf(p.decide(obs)));
  };
  const a = run(1234);
  assertEquals(run(1234), a, "same seed ⇒ identical sequence");
  assert(a.every((i) => i >= 0), "every step resolved to a legal action");

  // `reset` re-seeds the same stream, so a reused policy repeats itself too.
  const p = new NeuralPolicy("N", 1234, constantNet(logits), { temperature: 0.8 });
  const first = Array.from({ length: 40 }, () => obs.legal.indexOf(p.decide(obs)));
  p.reset(1234);
  assertEquals(Array.from({ length: 40 }, () => obs.legal.indexOf(p.decide(obs))), first);
  assertEquals(first, a);

  // A different seed walks a different stream. (Not a law — 4 actions × 40 draws
  // makes a collision ~4^-40 — but a broken rng hookup would show up here.)
  assert(run(999).join() !== a.join(), "a different seed gave the identical sequence");
});

// ---------------------------------------------------------------------------
// 4. sampling actually explores
// ---------------------------------------------------------------------------

Deno.test("policy: 温度>0は僅差の二手を両方引く", () => {
  // Two near-equal front-runners, the other two far enough back that the test is
  // about the top pair; greedy would return slot 0 every single time.
  const logits = { 0: 1.0, 1: 0.98, 2: -6, 3: -6 };
  const obs = baseObs();
  assertEquals(greedyIndex(obs, logits), 0, "greedy sees only one of them");

  const p = new NeuralPolicy("N", 31337, constantNet(logits), { temperature: 1 });
  const counts = [0, 0, 0, 0];
  for (let i = 0; i < 300; i++) counts[obs.legal.indexOf(p.decide(obs))]++;
  assert(counts[0] > 0 && counts[1] > 0, `both near-ties must appear: ${counts}`);
  // ~e^-7 apiece: seeing one is possible, seeing a third of the draws is not.
  assert(counts[2] + counts[3] < 30, `the far-behind pair took over: ${counts}`);

  // Temperature flattens: at 10 the 0.02 logit gap is nothing, so the split is
  // close to even, while at 0.02 the same gap is decisive.
  const hot = new NeuralPolicy("N", 31337, constantNet(logits), { temperature: 10 });
  const hotCounts = [0, 0, 0, 0];
  for (let i = 0; i < 300; i++) hotCounts[obs.legal.indexOf(hot.decide(obs))]++;
  assert(hotCounts[2] > 10 && hotCounts[3] > 10, `温度10でも均されない: ${hotCounts}`);

  const cold = new NeuralPolicy("N", 31337, constantNet(logits), { temperature: 0.001 });
  const coldCounts = [0, 0, 0, 0];
  for (let i = 0; i < 200; i++) coldCounts[obs.legal.indexOf(cold.decide(obs))]++;
  assertEquals(coldCounts[0], 200, "a near-zero temperature is greedy in the limit");
});
