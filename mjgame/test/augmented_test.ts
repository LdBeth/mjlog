// The oracle-augmented heuristic (M8-zero ablation harness).
//
// Two halves, tested separately on purpose: `oracleReads` is a PREDICATE over a
// live Table (does this seat's ron fire on this type, right now, for how much),
// and `AugmentedHeuristic` is a set of consumption terms over whatever Reads it
// is handed. The predicate is checked against a stacked table with a known
// wait; the terms are checked by handing the policy Reads directly.

import {
  assert,
  assertEquals,
  assertNotEquals,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import type { DangerAssessment, DangerLevel } from "mjrender/danger.ts";
import type { PlayerInfo, Tile } from "mjrender/model.ts";
import { tileType } from "mjrender/tiles.ts";
import type { Ctx } from "../src/ai/heuristic.ts";
import { HeuristicPolicy } from "../src/ai/heuristic.ts";
import type { OracleChannel, Reads, ReadsProvider } from "../src/ai/augmented.ts";
import {
  AugmentedHeuristic,
  curriculumReads,
  noisyReads,
  oracleReads,
  parseChannels,
} from "../src/ai/augmented.ts";
import { pairedRun } from "../src/paired.ts";
import { sfc32 } from "../src/rng.ts";
import { observe } from "../src/observe.ts";
import type { Observation } from "../src/observe.ts";
import { JANKI } from "../src/rules.ts";
import { ronValue, scorer } from "../src/score.ts";
import { Table } from "../src/table.ts";
import type { Seat } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { Wall } from "../src/wall.ts";
import { playHanchan, tiles } from "./helpers.ts";

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

/** Distinct copies across several `tiles()` calls (furiten_test.ts:31). */
function deck(): (spec: string) => Tile[] {
  const taken = new Set<Tile>();
  return (spec: string) =>
    tiles(spec).map((want) => {
      const ty = tileType(want);
      let id = ty * 4;
      while (taken.has(id)) id++;
      if (id >= ty * 4 + 4) throw new Error(`fifth copy of tile type ${ty}`);
      taken.add(id);
      return id;
    });
}

/** A wall that deals exactly `hands` (furiten_test.ts:50, minus the draw list). */
function stackedWall(hands: Tile[][], indicator: Tile, dealer: Seat = 0): Wall {
  const t: Tile[] = new Array(136).fill(-1);
  let taken = 0;
  const put = (id: Tile) => {
    t[135 - taken++] = id;
  };
  for (let block = 0; block < 3; block++) {
    for (let k = 0; k < 4; k++) {
      const seat = (dealer + k) % 4;
      for (let n = 0; n < 4; n++) put(hands[seat][block * 4 + n]);
    }
  }
  for (let k = 0; k < 4; k++) put(hands[(dealer + k) % 4][12]);
  t[5] = indicator;

  const used = new Set(t.filter((id) => id >= 0));
  const spare: Tile[] = [];
  for (let id = 0; id < 136; id++) if (!used.has(id)) spare.push(id);
  for (let i = 0; i < 136; i++) if (t[i] < 0) t[i] = spare.pop()!;
  return new Wall(t);
}

const PLAYERS: PlayerInfo[] = SEATS.map((seat) => ({ seat, name: `P${seat}` }));

const SIX_S = 23; // 6索
const NINE_S = 26; // 9索

/**
 * Seat 1 is riichi on 234m 567m 234p 55p 78s — 両面 6s/9s, and riichi itself
 * supplies the yaku, so BOTH waits are ronnable and the predicate must light up
 * on exactly those two types.
 */
function riichiTable(seat1 = "234m567m234p55p78s", declare = true): Table {
  const d = deck();
  const junk = () => d("149m149p149s東南西北");
  const hands = [junk(), d(seat1), junk(), junk()];
  const t = new Table(
    {
      kyoku: 0,
      honba: 0,
      kyotaku: 0,
      dealer: 0,
      scores: [25000, 25000, 25000, 25000],
      wall: stackedWall(hands, d("北")[0]),
      dice: [0, 0],
    },
    JANKI,
    PLAYERS,
  );
  t.riichi[1] = declare;
  return t;
}

const CH = (...cs: OracleChannel[]) => new Set<OracleChannel>(cs);

function readsAt(t: Table, channels: Set<OracleChannel>, seat: Seat = 0): Reads {
  const obs = observe(t, seat, [], null, scorer);
  const r = oracleReads(() => t, scorer, channels)(obs);
  assert(r !== null, "the oracle had a table and channels; it must produce reads");
  return r;
}

/** Which types the deal-in channel flagged for the opponent at relative index. */
function litTypes(p: Float32Array): number[] {
  const out: number[] = [];
  for (let ty = 0; ty < 34; ty++) if (p[ty] > 0) out.push(ty);
  return out;
}

// ---------------------------------------------------------------------------
// the oracle predicate
// ---------------------------------------------------------------------------

Deno.test("oracle: dealinP lights up on exactly the ronnable waits", () => {
  const t = riichiTable();
  const r = readsAt(t, CH("C1", "C2", "C3"));

  assertEquals(r.tenpaiP, [1, 0, 0], "only seat 1 (relative 0) is tenpai");
  assertEquals(litTypes(r.dealinP![0]), [SIX_S, NINE_S]);
  assertEquals(litTypes(r.dealinP![1]), []);
  assertEquals(litTypes(r.dealinP![2]), []);

  // The value channel prices exactly those types, and nothing else.
  const six = ronValue(t, 1, 0, SIX_S * 4);
  const nine = ronValue(t, 1, 0, NINE_S * 4);
  assert(six !== null && nine !== null, "a riichi tenpai hand always scores");
  assertEquals(r.dealinValue![0][SIX_S], six);
  assertEquals(r.dealinValue![0][NINE_S], nine);
  assertEquals(r.expLoss![0], (six + nine) / 2);
  assertEquals(r.expLoss![1], 0);
});

Deno.test("oracle: furiten kills every deal-in bit but not the tenpai bit", () => {
  const t = riichiTable();
  t.furiten[1].permanent = true;
  const r = readsAt(t, CH("C1", "C2", "C3"));

  assertEquals(r.tenpaiP, [1, 0, 0], "a furiten hand is still tenpai");
  assertEquals(litTypes(r.dealinP![0]), [], "…but nothing deals in to it");
  assertEquals(r.expLoss![0], 0);
});

Deno.test("oracle: 見せ牌 (ronBlocked) and 和了放棄 (sanctioned) narrow the bits", () => {
  const blocked = riichiTable();
  blocked.ronBlocked[1].add(SIX_S);
  assertEquals(litTypes(readsAt(blocked, CH("C1")).dealinP![0]), [NINE_S]);

  const out = riichiTable();
  out.sanctioned[1] = true;
  assertEquals(litTypes(readsAt(out, CH("C1")).dealinP![0]), []);
});

Deno.test("oracle: a tenpai with no yaku deals in to nobody", () => {
  // 123m 456m 789p + 88s/22s シャンポン, damaten: no 平和 (shanpon), no 断幺九
  // (terminals), no 役牌, no 一気通貫, no 三色. A ron simply cannot be declared.
  const t = riichiTable("123m456m789p88s22s", false);
  const r = readsAt(t, CH("C1", "C2", "C3"));
  assertEquals(r.tenpaiP, [1, 0, 0], "the shape is tenpai");
  assertEquals(litTypes(r.dealinP![0]), [], "…but no wait carries a yaku");
  assertEquals(r.expLoss![0], 0);

  // The control: declare riichi on the same shape and both waits light up.
  const declared = riichiTable("123m456m789p88s22s", true);
  assertEquals(litTypes(readsAt(declared, CH("C1")).dealinP![0]), [19, 25]); // 2s, 8s
});

Deno.test("oracle: channels gate what is computed", () => {
  const t = riichiTable();

  const only2 = readsAt(t, CH("C2"));
  assertEquals(only2.tenpaiP, [1, 0, 0]);
  assertEquals(only2.dealinP, undefined, "C1 off ⇒ no deal-in field at all");
  assertEquals(only2.expLoss, undefined);

  const only1 = readsAt(t, CH("C1"));
  assertEquals(only1.tenpaiP, undefined);
  assert(only1.dealinP !== undefined);
  assertEquals(only1.expLoss, undefined);

  // An empty set is the control arm: no reads, so the base policy is untouched.
  const obs = observe(t, 0, [], null, scorer);
  assertEquals(oracleReads(() => t, scorer, new Set())(obs), null);
  // And so is a null table (outside a round).
  assertEquals(oracleReads(() => null, scorer, CH("C1"))(obs), null);
});

Deno.test("oracle: wall channels read the wall, and the seat arithmetic holds", () => {
  const t = riichiTable();
  // A turn decision: seat 0 has drawn, so the next live tile is seat 1's.
  const drawn = t.wall.draw();
  t.hands[0].push(drawn);
  const obs = observe(t, 0, [], drawn, scorer);
  const r = oracleReads(() => t, scorer, CH("C4", "C5", "C6"))(obs)!;

  assertEquals(r.nextDraw, tileType(t.wall.peekLive(0)!), "C4: next off the wall");
  assertEquals(r.ownNextDraw, tileType(t.wall.peekLive(3)!), "…ours is three later");
  assertEquals(r.nextDora !== null, true, "C5: a second indicator is still available");
  // C6 fires only for declared riichi seats; seat 1 is relative index 0.
  assertEquals(r.riichiNextDraw![0], tileType(t.wall.peekLive(0)!));
  assertEquals(r.riichiNextDraw![1], null);
  assertEquals(r.riichiNextDraw![2], null);
});

Deno.test("oracle: parseChannels validates", () => {
  assertEquals([...parseChannels("C1,C2,C3")!], ["C1", "C2", "C3"]);
  assertEquals([...parseChannels("c6")!], ["C6"]);
  assertEquals([...parseChannels("none")!], []);
  assertEquals([...parseChannels("c7o,C7P")!], ["C7O", "C7P"]);
  assertEquals(parseChannels("C7"), null);
  assertEquals(parseChannels("C1,x"), null);
});

Deno.test("oracle: C7O hands the planner the wall and the hands; C7P hands it nothing", () => {
  const t = riichiTable();
  const oracle = readsAt(t, CH("C7O"));
  assertEquals(oracle.planner, true);

  // The live wall, exactly: 122 tiles minus the 52 dealt.
  const wall = oracle.wallComposition!;
  let live = 0;
  for (let ty = 0; ty < 34; ty++) live += wall[ty];
  assertEquals(live, t.wall.remaining);
  assertEquals(wall[tileType(t.wall.peekLive(0)!)] > 0, true, "the next tile is in there");

  // Three opponents, thirteen tiles each, in relative order.
  const opp = oracle.oppConcealed!;
  assertEquals(opp.length, 3);
  for (let i = 0; i < 3; i++) {
    let n = 0;
    for (let ty = 0; ty < 34; ty++) n += opp[i][ty];
    assertEquals(n, 13);
  }
  // Seat 1 (relative 0) is the riichi hand: 234m567m234p55p78s.
  assertEquals(opp[0][SIX_S + 1], 1, "the 7索 of its 78索 両面");

  // The public arm switches the same machinery on with no truth attached.
  const pub = readsAt(t, CH("C7P"));
  assertEquals(pub.planner, true);
  assertEquals(pub.wallComposition, undefined);
  assertEquals(pub.oppConcealed, undefined);
});

// ---------------------------------------------------------------------------
// the consumption terms
// ---------------------------------------------------------------------------

/** Reaches the protected hooks the way `scoreDiscard` does. */
class Probe extends AugmentedHeuristic {
  riskWith(ctx: Ctx, tile: Tile, reads: Reads | null): number {
    this.reads = reads;
    try {
      return this.riskOf(ctx, tile);
    } finally {
      this.reads = null;
    }
  }
  pressureWith(obs: Observation, reads: Reads | null): number {
    this.reads = reads;
    try {
      return this.pressureOf(obs);
    } finally {
      this.reads = null;
    }
  }
}

function danger(level: DangerLevel, seat = 1): DangerAssessment {
  return { level, seats: [seat], details: [{ seat, level, kind: "riichi", notes: [] }] };
}

/** Just enough Observation/Ctx for the risk hook: a danger map and a hand. */
function riskCtx(map: Map<number, DangerAssessment>): Ctx {
  const obs = { danger: map, riichi: [false, true, false, false] } as unknown as Observation;
  return {
    obs,
    open: 0,
    closed: true,
    doraTypes: new Set(),
    valueHonors: new Set(),
    unseen: new Array<number>(34).fill(4),
    folding: false,
    canRiichi: false,
    eff: 1,
    def: 1,
  };
}

function dealin(rel: number, ty: number, value: number): Reads {
  const p = [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
  const v = [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
  p[rel][ty] = 1;
  v[rel][ty] = value;
  return { dealinP: p, dealinValue: v };
}

Deno.test("augmented: 安全 is a proof — no estimate may price it", () => {
  const safe = 5; // 6m, genbutsu
  const hot = 12; // 4p, no-suji
  const probe = new Probe("probe", 1, () => null);
  const ctx = riskCtx(new Map([[safe, danger("安全")], [hot, danger("危険度高")]]));

  // A confident, expensive deal-in claim on a tile the assessor calls genbutsu.
  const lie = dealin(0, safe, 12000);
  assertEquals(probe.riskWith(ctx, safe * 4, lie), 0, "RULE FLOOR: 安全 stays free");

  // The same claim on a live tile does get through, scaled by λ.
  assertEquals(probe.riskWith(ctx, hot * 4, dealin(0, hot, 12000)), 0.25 * 12000);
});

Deno.test("augmented: an absent assessment is not a proof — the estimate prices quiet tables", () => {
  const silent = 12; // 4p; nobody has declared anything, the assessor is idle
  const probe = new Probe("probe", 1, () => null);
  const ctx = riskCtx(new Map()); // quiet table: EMPTY danger map

  // The base policy has nothing to say here...
  assertEquals(probe.riskWith(ctx, silent * 4, null), 0);
  // ...but a deal-in estimate must still reach the discard choice: an absent
  // entry means "not assessed", and only an explicit 安全 entry is a proof.
  const est = dealin(0, silent, 8000);
  assertEquals(probe.riskWith(ctx, silent * 4, est), 0.25 * 8000);
});

Deno.test("augmented: the rule floor bounds a quiet estimate from below", () => {
  const hot = 12;
  const probe = new Probe("probe", 1, () => null);
  const ctx = riskCtx(new Map([[hot, danger("危険度高")]]));
  const base = probe.riskWith(ctx, hot * 4, null); // the plain heuristic's own figure
  assertEquals(base, 200);

  // Nobody can ron it — but 危険度高 is not a proof of safety, so half stands.
  const quiet: Reads = {
    dealinP: [new Float32Array(34), new Float32Array(34), new Float32Array(34)],
  };
  assertEquals(probe.riskWith(ctx, hot * 4, quiet), 0.5 * base);
});

Deno.test("augmented: pressure is priced by who is tenpai and for how much", () => {
  const probe = new Probe("probe", 1, () => null);
  const obs = {
    riichi: [false, true, false, false],
    danger: new Map<number, DangerAssessment>(),
  } as unknown as Observation;

  // Base reading: one declared riichi and nothing else ⇒ 1.0.
  assertEquals(probe.pressureWith(obs, null), 1);

  // Truth: seat 1 is tenpai for a mangan, seat 2 is tenpai cheap, seat 3 noten.
  const priced = probe.pressureWith(obs, { tenpaiP: [1, 1, 0], expLoss: [12000, 3000, 0] });
  assertEquals(priced, 2 + 0.5, "12000/6000 capped at 2, plus 3000/6000");

  // A declared riichi the reader missed still counts full.
  assertEquals(probe.pressureWith(obs, { tenpaiP: [0, 0, 0] }), 1);
});

// ---------------------------------------------------------------------------
// oracle fading (noisyReads)
// ---------------------------------------------------------------------------

/** Every droppable group filled at once — the worst case for the ε sweep. */
function fullReads(): Reads {
  const f3 = () => [new Float32Array(34), new Float32Array(34), new Float32Array(34)];
  return {
    tenpaiP: [1, 0, 0],
    dealinP: f3(),
    dealinValue: f3(),
    expLoss: [3900, 0, 0],
    nextDraw: 5,
    ownNextDraw: 6,
    nextDora: 7,
    riichiNextDraw: [8, null, null],
    planner: true,
    wallComposition: new Float32Array(34),
    oppConcealed: f3(),
  };
}

/** The groups, in the order `noisyReads` documents. */
const NOISE_GROUPS: readonly (readonly (keyof Reads)[])[] = [
  ["dealinP", "dealinValue"],
  ["tenpaiP"],
  ["expLoss"],
  ["nextDraw", "ownNextDraw"],
  ["nextDora"],
  ["riichiNextDraw"],
  ["wallComposition", "oppConcealed"],
];

/** One character per group: "1" survived this call, "0" was dropped. */
function presence(r: Reads): string {
  return NOISE_GROUPS.map((g) => {
    const kept = g.filter((f) => r[f] !== undefined).length;
    assert(kept === 0 || kept === g.length, "a group is dropped whole or not at all");
    return kept === 0 ? "0" : "1";
  }).join("");
}

const OBS = {} as unknown as Observation; // the wrapper never inspects it

/** The presence sequence a given (ε, seed) produces over `n` decisions. */
function trace(eps: number, seed: number, n = 50): string {
  const p = noisyReads(() => fullReads(), eps, seed);
  let s = "";
  for (let i = 0; i < n; i++) s += presence(p(OBS)!) + " ";
  return s;
}

Deno.test("noise: ε=0 is the provider itself, not a copy of it", () => {
  const inner: ReadsProvider = () => fullReads();
  assertStrictEquals(noisyReads(inner, 0), inner, "the un-noised arm must be bit-identical");
});

Deno.test("noise: ε=1 drops every group — but never the planner flag", () => {
  const faded = noisyReads(() => fullReads(), 1);
  for (let i = 0; i < 25; i++) {
    const r = faded(OBS)!;
    assertEquals(presence(r), "0000000");
    assertEquals(r.planner, true, "C7O degrades to C7P, it does not switch off");
  }
  // …and a provider that never set it does not acquire one.
  const noPlan = noisyReads(() => ({ tenpaiP: [1, 1, 1] }), 1);
  assertEquals(noPlan(OBS), {});
});

Deno.test("noise: ε outside [0,1] is refused", () => {
  const inner: ReadsProvider = () => fullReads();
  assertThrows(() => noisyReads(inner, -0.01), RangeError);
  assertThrows(() => noisyReads(inner, 1.01), RangeError);
  assertThrows(() => noisyReads(inner, NaN), RangeError);
});

Deno.test("noise: the dropout schedule is deterministic per (ε, seed)", () => {
  assertEquals(trace(0.3, 1), trace(0.3, 1), "same seed ⇒ same losses, so paired runs replay");
  assertNotEquals(trace(0.3, 1), trace(0.3, 2), "a different seed ⇒ a different schedule");
  assertNotEquals(trace(0.3, 1), trace(0.6, 1), "ε is folded into the seed");
});

Deno.test("noise: each group survives ≈1−ε of the time, independently", () => {
  const n = 2000;
  const eps = 0.3;
  const p = noisyReads(() => fullReads(), eps, 0xABCDEF);
  const kept = NOISE_GROUPS.map(() => 0);
  for (let i = 0; i < n; i++) {
    const s = presence(p(OBS)!);
    for (let g = 0; g < kept.length; g++) if (s[g] === "1") kept[g]++;
  }
  for (let g = 0; g < kept.length; g++) {
    const rate = kept[g] / n;
    assert(
      Math.abs(rate - (1 - eps)) < 0.05,
      `group ${g}: 残存率 ${rate.toFixed(3)} が 0.7 から離れすぎ`,
    );
  }
});

Deno.test("noise: a null read passes through, and consumes no randomness", () => {
  const pure = trace(0.4, 7, 20);

  // The same wrapper, but every third call has nothing to degrade. If a null
  // burned a draw the surviving calls would fall out of step immediately.
  let k = 0;
  const gappy = noisyReads(() => (k++ % 3 === 0 ? null : fullReads()), 0.4, 7);
  let seen = "";
  let n = 0;
  for (let i = 0; n < 20; i++) {
    const r = gappy(OBS);
    if (r === null) continue;
    seen += presence(r) + " ";
    n++;
  }
  assert(k > 20, "the gappy provider really did return nulls");
  assertEquals(seen, pure);
});

Deno.test("noise: one draw per PRESENT group — the stream tracks the channel set", () => {
  // Two providers filling ONE group each, different groups. Both consume
  // exactly one uniform per call, so their loss sequences must coincide.
  const value = noisyReads(() => ({ expLoss: [1, 2, 3] }), 0.5, 11);
  const dora = noisyReads(() => ({ nextDora: 4 }), 0.5, 11);
  const seq = (p: ReadsProvider, f: keyof Reads) => {
    let s = "";
    for (let i = 0; i < 40; i++) s += p(OBS)![f] === undefined ? "0" : "1";
    return s;
  };
  const a = seq(value, "expLoss");
  assertEquals(a, seq(dora, "nextDora"));
  assert(a.includes("0") && a.includes("1"), "…and the sequence is not degenerate");
});

// ---------------------------------------------------------------------------
// the curriculum (curriculumReads)
// ---------------------------------------------------------------------------

/**
 * What a COUNTING reader can produce, filled with values no oracle fixture uses
 * so that the source of every surviving group is identifiable. Deliberately
 * missing the three order groups (`nextDraw`, `nextDora`, `riichiNextDraw`):
 * `computedReads` refuses to invent an ordering of unseen tiles, so a dropped
 * order group has nothing to fall back TO — and must degrade to absence, exactly
 * as it does under `noisyReads`.
 */
function computedFixture(): Reads {
  const f3 = (v: number) => [0, 1, 2].map(() => new Float32Array(34).fill(v));
  return {
    tenpaiP: [0.5, 0.5, 0.5],
    dealinP: f3(9),
    dealinValue: f3(9),
    expLoss: [1234, 1234, 1234],
    wallComposition: new Float32Array(34).fill(2),
    oppConcealed: f3(9),
  };
}

/** "o" = this group came from the oracle, "c" = from the counting reader. */
function source(r: Reads): string {
  const at = (
    got: boolean,
    oracleSide: boolean,
  ) => (!got ? "-" : oracleSide ? "o" : "c");
  return [
    at(r.dealinP !== undefined, r.dealinP?.[0][0] === 0),
    at(r.tenpaiP !== undefined, r.tenpaiP?.[0] === 1),
    at(r.expLoss !== undefined, r.expLoss?.[0] === 3900),
    at(r.nextDraw !== undefined, true),
    at(r.nextDora !== undefined, true),
    at(r.riichiNextDraw !== undefined, true),
    at(r.wallComposition !== undefined, r.wallComposition?.[0] === 0),
  ].join("");
}

const ORACLE_P: ReadsProvider = () => fullReads();
const COMPUTED_P: ReadsProvider = () => computedFixture();

Deno.test("curriculum: ε=0 is the oracle provider itself, not a copy", () => {
  assertStrictEquals(
    curriculumReads(ORACLE_P, COMPUTED_P, 0),
    ORACLE_P,
    "ε=0 の腕は純オラクルとビット単位で同一でなければならない",
  );
});

Deno.test("curriculum: ε=1 is the computed provider itself, not a copy", () => {
  // The contract the M9c champion re-score depends on: at ε=1 there is no
  // oracle machinery left in the path, so "graded without help" is a fact about
  // the wiring and not a statistical claim.
  assertStrictEquals(curriculumReads(ORACLE_P, COMPUTED_P, 1), COMPUTED_P);
});

Deno.test("curriculum: a dropped group falls back to the computed answer", () => {
  const p = curriculumReads(ORACLE_P, COMPUTED_P, 0.5, 4242);
  let sawComputed = false;
  let sawOracle = false;
  for (let i = 0; i < 200; i++) {
    const s = source(p(OBS)!);
    // The four groups the counting reader can answer are never absent…
    assert(/^[oc][oc][oc]/.test(s), `落ちた組が undefined になった: ${s}`);
    assert(s[6] === "o" || s[6] === "c", `残り枚数の組が消えた: ${s}`);
    // …and the three ORDER groups are oracle-or-nothing, since counting cannot
    // produce an order.
    assert(/[o-]{3}$/.test(s.slice(3, 6)), `数えられない情報が湧いた: ${s}`);
    if (s.includes("c")) sawComputed = true;
    if (s.includes("o")) sawOracle = true;
  }
  assert(sawComputed && sawOracle, "ε=0.5 なのに片側しか出ていない");
});

Deno.test("curriculum: each group takes the computed answer ≈ε of the time", () => {
  const n = 2000;
  const eps = 0.3;
  const p = curriculumReads(ORACLE_P, COMPUTED_P, eps, 0xABCDEF);
  const fell = [0, 0, 0, 0, 0, 0, 0];
  for (let i = 0; i < n; i++) {
    const s = source(p(OBS)!);
    for (let g = 0; g < fell.length; g++) if (s[g] !== "o") fell[g]++;
  }
  for (let g = 0; g < fell.length; g++) {
    const rate = fell[g] / n;
    assert(Math.abs(rate - eps) < 0.05, `組 ${g}: 脱落率 ${rate.toFixed(3)} が 0.3 から離れすぎ`);
  }
});

Deno.test("curriculum: the schedule is deterministic per (ε, seed)", () => {
  const trace = (eps: number, seed: number) => {
    const p = curriculumReads(ORACLE_P, COMPUTED_P, eps, seed);
    let s = "";
    for (let i = 0; i < 40; i++) s += source(p(OBS)!) + " ";
    return s;
  };
  assertEquals(trace(0.3, 1), trace(0.3, 1), "同じ種で同じ脱落 — paired が再現できる条件");
  assertNotEquals(trace(0.3, 1), trace(0.3, 2));
  assertNotEquals(trace(0.3, 1), trace(0.6, 1), "ε は種に畳み込まれている");
});

Deno.test("curriculum: a caller-supplied Rng is used as given", () => {
  const trace = (r: ReturnType<typeof sfc32>) => {
    const p = curriculumReads(ORACLE_P, COMPUTED_P, 0.4, r);
    let s = "";
    for (let i = 0; i < 30; i++) s += source(p(OBS)!) + " ";
    return s;
  };
  assertEquals(trace(sfc32(99)), trace(sfc32(99)));
  assertNotEquals(trace(sfc32(99)), trace(sfc32(100)));
});

Deno.test("curriculum: ε outside [0,1] is refused", () => {
  assertThrows(() => curriculumReads(ORACLE_P, COMPUTED_P, -0.01), RangeError);
  assertThrows(() => curriculumReads(ORACLE_P, COMPUTED_P, 1.01), RangeError);
  assertThrows(() => curriculumReads(ORACLE_P, COMPUTED_P, NaN), RangeError);
});

Deno.test("curriculum: a silent oracle leaves the counting reader in charge", () => {
  const p = curriculumReads(() => null, COMPUTED_P, 0.5, 7);
  for (let i = 0; i < 20; i++) {
    const r = p(OBS)!;
    // Every group either the computed answer or nothing — never an oracle value,
    // and never a crash.
    assert(r.nextDraw === undefined, "オラクルが黙っているのに順序が湧いた");
    const s = source(r);
    assert(/^[c-][c-][c-]/.test(s), `オラクル由来の値が出た: ${s}`);
  }
  // Both silent is silence.
  assertEquals(curriculumReads(() => null, () => null, 0.5, 7)(OBS), null);
});

Deno.test("curriculum: the planner flag survives every dropout", () => {
  const p = curriculumReads(ORACLE_P, COMPUTED_P, 0.9, 3);
  for (let i = 0; i < 30; i++) assertEquals(p(OBS)!.planner, true);
  // …and is taken from the counting reader when only it declared one.
  const q = curriculumReads(() => ({ tenpaiP: [1, 0, 0] }), () => ({ planner: true }), 0.5, 3);
  for (let i = 0; i < 10; i++) assertEquals(q(OBS)!.planner, true);
});

// ---------------------------------------------------------------------------
// end to end
// ---------------------------------------------------------------------------

function playOne(seed: number, channels: Set<OracleChannel>, noise = 0) {
  // `make` runs once per seat, so the (stateful) noise wrapper is built exactly
  // once — for seat 0, the only seat that reads it.
  return playHanchan(seed, (s, ref) =>
    s === 0
      ? new AugmentedHeuristic(
        `O${s}`,
        seed * 4 + s,
        noisyReads(oracleReads(() => ref.t, scorer, channels), noise),
      )
      : new HeuristicPolicy(`H${s}`, seed * 4 + s));
}

Deno.test("augmented: an all-channels seat plays a legal hanchan, deterministically", () => {
  const all = CH("C1", "C2", "C3", "C4", "C5", "C6");
  const a = playOne(101, all);
  const b = playOne(101, all);
  assert(a.rounds.length > 0);
  assertEquals(a.scores, b.scores, "same seed ⇒ same match");
  // 供託 is paid out when the match finishes, so the points are conserved.
  assertEquals(a.scores.reduce((x, y) => x + y, 0), 4 * JANKI.startScore);
});

Deno.test("augmented: a C7 planner seat plays a full legal hanchan, both arms", () => {
  // The control: what the ledger looks like when nobody plans.
  const plain = playOne(303, CH("C1", "C2", "C3"));

  for (const ch of [CH("C1", "C2", "C3", "C7O"), CH("C1", "C2", "C3", "C7P")]) {
    const a = playOne(303, ch);
    const b = playOne(303, ch);
    const name = [...ch].join(",");
    assert(a.rounds.length > 0, `${name}: the match was played`);
    assertEquals(a.scores, b.scores, `${name}: same seed ⇒ same match`);
    assertEquals(
      a.scores.reduce((x, y) => x + y, 0),
      4 * JANKI.startScore,
      `${name}: 点棒は保存される`,
    );
    // Plan discipline changes which hands get opened, so the ledger moves — but
    // a planner that walked the seat into 禁じ手 would show up as a blow-up, not
    // as a wobble.
    assert(
      a.ledger.length <= plain.ledger.length + 8,
      `${name}: 違反 ${a.ledger.length} vs 対照 ${plain.ledger.length}`,
    );
  }
});

/**
 * The rules a compliance-filtered seat may still be charged for, each because
 * the preview cannot veto it (`penalty/preview.ts` header):
 *
 *   under-8000       fires on the PAYMENTS at a round's end, not on an action.
 *   katagari         fires on taking a win. Declining one is 見逃し — furiten by
 *   atozuke          the engine's own rules — so wins are always taken, and the
 *                    discard that builds the wait is vetoed (and priced) instead.
 *                    The veto has a fallthrough, so these can still land.
 *   riichi-kan-skip  the ankan dilemma, now confined to kans that genuinely move
 *                    the wait: the ledger charges for passing one up AND for
 *                    taking it. Both cost 中; the tie falls to declining.
 */
const UNPREVIEWABLE = new Set(["under-8000", "katagari", "atozuke", "riichi-kan-skip"]);

Deno.test("augmented: the C7 planner cannot buy its way past the dojo filter", () => {
  // The planner's `planKeep` malus (5000) outbids every `dojoCost` price, which
  // is exactly why compliance cannot be a price. It is enforced at action
  // SELECTION instead — the base class narrows the candidate set before any plan
  // term is scored — so a planning seat's ledger holds nothing an action could
  // have avoided.
  for (const seed of [303, 404, 606]) {
    const r = playOne(seed, CH("C1", "C2", "C3", "C7O"));
    const filed = r.ledger.filter((v) => v.seat === 0);
    const stray = filed.filter((v) => !UNPREVIEWABLE.has(v.rule));
    assertEquals(
      stray.map((v) => `${v.rule}: ${v.detail}`),
      [],
      `seed ${seed}: 席0 filed an avoidable violation`,
    );
  }
});

Deno.test("augmented: a faded oracle seat plays a legal hanchan, reproducibly", () => {
  const all = CH("C1", "C2", "C3", "C4", "C5", "C6", "C7O");
  const a = playOne(505, all, 0.5);
  const b = playOne(505, all, 0.5);
  assert(a.rounds.length > 0);
  assertEquals(a.scores, b.scores, "same seed ⇒ same match, dropouts included");
  assertEquals(a.scores.reduce((x, y) => x + y, 0), 4 * JANKI.startScore, "点棒は保存される");

  // ε=0 is the un-noised arm exactly, and half-truth is not that arm — losing
  // half the reads must actually change the play, or the sweep measures nothing.
  const full = playOne(505, all).scores;
  assertEquals(playOne(505, all, 0).scores, full, "ε=0 は無ノイズと完全に同一");
  assertNotEquals(a.scores, full, "ε=0.5 が真値と同じ打牌 — 劣化が効いていない");
});

Deno.test("augmented: with no channels the seat is the plain heuristic, bit for bit", () => {
  const st = pairedRun(1, 909, "ohhh", { oracle: new Set() });
  assertEquals(st.dRank.mean, 0);
  assertEquals(st.dScore.mean, 0);
  assertEquals(st.vioA, st.vioB);
});

Deno.test("paired: reports a paired difference over identical walls", () => {
  const st = pairedRun(2, 42, "ohhh", { oracle: parseChannels("C1,C2,C3")! });
  assertEquals(st.games, 2);
  assertEquals(st.better + st.tie + st.worse, 2);
  assert(st.rankA >= 1 && st.rankA <= 4);
  assert(st.rankB >= 1 && st.rankB <= 4);
  assertEquals(st.dRank.mean, st.rankA - st.rankB);
  assert(st.ms > 0);
});
