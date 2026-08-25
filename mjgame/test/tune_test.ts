// The outcome-tuning harness: the `--ktune` plumbing, `paired --json`, and the
// pure search operators of `scripts/tune.ts`.
//
// The tuner itself is NOT run here. One generation of the real thing is minutes
// of self-play in subprocesses; what a test can pin down is that the vector
// reaches exactly one arm, that the machine-readable output says what the search
// loop reads, and that the operators are the deterministic functions the resume
// logic assumes they are.

import {
  assert,
  assertAlmostEquals,
  assertEquals,
  assertNotEquals,
  assertThrows,
} from "@std/assert";
import { argError } from "../src/cli/args.ts";
import { pairedJson, pairedRun } from "../src/paired.ts";
import type { KTune } from "../src/harness.ts";
import { initFromWeights } from "../src/ai/consumer.ts";
import type { ConsumerParams } from "../src/ai/consumer.ts";
import { sfc32 } from "../src/rng.ts";
import {
  candidateKtune,
  clampVec,
  deepMerge,
  defaults,
  gauss,
  laneOf,
  logMean,
  mergedWeights,
  pairedArgs,
  PARAMS,
  parseOpts,
  perturb,
  select,
  selectParams,
  setPath,
  sigmaAt,
  spacePaths,
  toKtune,
  vioTotal,
} from "../scripts/tune.ts";
import type { Candidate, Nested, PairedJson, State } from "../scripts/tune.ts";

// ---------------------------------------------------------------------------
// --ktune plumbing: the vector reaches arm A and nothing else
// ---------------------------------------------------------------------------

/**
 * A vector nobody could mistake for a tuning candidate: at 100000 per dora the
 * seat will hold every dora against any danger the table can show, which is a
 * different player. The point is not that it is a GOOD player — it is that arm A
 * must visibly move while arm B does not budge by a single point.
 */
const EXTREME: KTune = { heuristic: { dora: 100_000 } };

Deno.test("ktune moves arm A and never reaches the control arm", () => {
  const base = pairedRun(2, 777, "khhh");
  const tuned = pairedRun(2, 777, "khhh", { ktune: EXTREME });

  // B is the same policy on the same walls in both runs — bit for bit.
  assertEquals(tuned.rankB, base.rankB, "B腕の順位が動いた");
  assertEquals(tuned.scoreB, base.scoreB, "B腕の点数が動いた");
  assertEquals(tuned.rankDojoB, base.rankDojoB, "B腕の道場順位が動いた");
  assertEquals(tuned.vioB, base.vioB, "B腕の違反数が動いた");
  assertEquals(
    [...tuned.vioB0.entries()].sort(),
    [...base.vioB0.entries()].sort(),
    "B腕の違反内訳が動いた",
  );

  // ...and A did move, or the file was plumbed nowhere.
  assertNotEquals(tuned.scoreA, base.scoreA, "A腕が感性ベクトルに反応していない");
});

Deno.test("ktune is inert on a seat kind that is not k", () => {
  // "hhhh" has no computing seat, so the vector has nothing to configure and
  // both runs must be identical — including arm A.
  const base = pairedRun(1, 4242, "hhhh");
  const tuned = pairedRun(1, 4242, "hhhh", { ktune: EXTREME });
  assertEquals(tuned.scoreA, base.scoreA);
  assertEquals(tuned.rankA, base.rankA);
});

// ---------------------------------------------------------------------------
// --ktune-b: the scalar-space variance-reduction instrument (M10d)
// ---------------------------------------------------------------------------

Deno.test("--ktune-b は paired 専用", () => {
  const base = { seats: "khhh", calibrate: "" };
  assertEquals(argError({ ...base, cmd: "paired", ktuneBPath: "x.json" }), null);
  for (const cmd of ["selfplay", "bench", "play"]) {
    assertEquals(
      argError({ ...base, cmd, ktuneBPath: "x.json" }),
      "--ktune-b は paired 専用です",
      `${cmd} が --ktune-b を受け付けてしまった`,
    );
  }
  // …and an absent flag is never refused.
  assertEquals(argError({ ...base, cmd: "selfplay" }), null);
});

/**
 * THE NULL CHECK of the scalar instrument. Both arms are the same seat holding
 * the SAME vector, so every game must be bit-identical to the end and every
 * difference exactly 0 — not "small", 0. A非対称 anywhere in the wiring (a seed
 * that reaches one arm only, an option the control arm silently drops) shows up
 * here as a nonzero SD, which is precisely what `scripts/tune.ts` prints as
 * "器が非対称です".
 */
Deno.test("ktune-b: 同じベクトルを両腕に渡すと差は厳密に0", () => {
  const same: KTune = { heuristic: { dora: 75, ukeire: 14 }, augment: { lambda: 0.3 } };
  const st = pairedRun(6, 900_000, "khhh", { ktune: same, ktuneB: same });
  for (const d of [st.dRank, st.dRankDojo, st.dScore]) {
    assertEquals(d.mean, 0);
    assertEquals(d.sd, 0);
    assertEquals(d.ci, 0);
  }
  assertEquals(st.rankA, st.rankB);
  assertEquals(st.rankDojoA, st.rankDojoB);
  assertEquals(st.scoreA, st.scoreB);
  assertEquals(st.vioA, st.vioB);
  assertEquals([...st.vioA0.entries()].sort(), [...st.vioB0.entries()].sort());
  assertEquals(st.tie, 6);
});

Deno.test("ktune-b は対照腕だけを差し替え、A腕には触れない", () => {
  const A: KTune = { heuristic: { dora: 100_000 } };
  const vsBaseline = pairedRun(4, 777, "khhh", { ktune: A });
  // The incumbent must be VISIBLY not the baseline. (Before the 2026-08-25
  // epoch an empty vector sufficed — a bare "k" seat differed from the old
  // hand-written "h". Now frozen-h IS a copy of the bare "k", so the two arms
  // would coincide bit for bit; the control needs a vector of its own.)
  const B: KTune = { heuristic: { dora: 50_000 } };
  const vsIncumbent = pairedRun(4, 777, "khhh", { ktune: A, ktuneB: B });
  assertEquals(vsIncumbent.rankA, vsBaseline.rankA, "対照の指定がA腕を動かした");
  assertEquals(vsIncumbent.scoreA, vsBaseline.scoreA, "対照の指定がA腕を動かした");
  assertNotEquals(vsIncumbent.scoreB, vsBaseline.scoreB, "B腕が hhhh のままになっている");
});

/** A curve set that is emphatically not the hand-written score. */
function wildConsumer(): ConsumerParams {
  const init = initFromWeights(mergedWeights(""));
  return { version: 1, curves: { ...init.curves, dora: [0, 5000, 5000, 10000] } };
}

Deno.test("ktune-b と --consumer は合成できる (B腕はA腕の曲線を引き継ぐ)", () => {
  const A: KTune = { heuristic: { dora: 100_000 } };
  const wild = wildConsumer();
  const inherited = pairedRun(4, 777, "khhh", { ktune: A, ktuneB: {}, consumer: wild });
  const spelled = pairedRun(4, 777, "khhh", {
    ktune: A,
    ktuneB: {},
    consumer: wild,
    consumerB: wild,
  });
  // Naming the same curves for the control arm changes nothing: absent
  // `consumerB`, arm B already holds arm A's curves and the two arms differ in
  // the 感性 vector alone.
  assertEquals(spelled.scoreA, inherited.scoreA);
  assertEquals(spelled.scoreB, inherited.scoreB);
  assertEquals(spelled.rankB, inherited.rankB);
  // …and the inheritance is real: without the curves the control arm plays
  // differently.
  const noCurves = pairedRun(4, 777, "khhh", { ktune: A, ktuneB: {} });
  assertNotEquals(inherited.scoreB, noCurves.scoreB, "曲線がB腕に届いていない");
});

// ---------------------------------------------------------------------------
// paired --json
// ---------------------------------------------------------------------------

Deno.test("pairedJson round-trips the documented keys", () => {
  const st = pairedRun(2, 777, "khhh");
  const parsed = JSON.parse(pairedJson(st)) as PairedJson;

  assertEquals(
    Object.keys(parsed).sort(),
    [
      "dRank",
      "dRankDojo",
      "dScore",
      "games",
      "ms",
      "rankA",
      "rankB",
      "seats",
      "seed",
      "vioA0",
      "vioB0",
    ],
  );
  assertEquals(parsed.games, 2);
  assertEquals(parsed.seed, 777);
  assertEquals(parsed.seats, "khhh");
  assertEquals(parsed.rankA, st.rankA);
  assertEquals(parsed.rankB, st.rankB);
  for (const k of ["mean", "sd", "ci"] as const) {
    assertEquals(parsed.dRank[k], st.dRank[k]);
    assertEquals(parsed.dRankDojo[k], st.dRankDojo[k]);
    assertEquals(parsed.dScore[k], st.dScore[k]);
  }
  // The Maps must have become plain objects, not `{}` from a naive stringify.
  assertEquals(parsed.vioA0, Object.fromEntries(st.vioA0));
  assertEquals(parsed.vioB0, Object.fromEntries(st.vioB0));
  assert(typeof parsed.ms === "number");
});

Deno.test("pairedJson serializes a non-empty violation map", () => {
  // pairedRun's own maps may be empty on a short clean run, so the serializer is
  // also checked against a stats object with entries planted in it.
  const st = pairedRun(1, 777, "khhh");
  st.vioA0.set("第一打字牌切り", 3);
  st.vioB0.set("不聴時ドラ切り", 1);
  const parsed = JSON.parse(pairedJson(st)) as PairedJson;
  assertEquals(parsed.vioA0["第一打字牌切り"], 3);
  assertEquals(parsed.vioB0["不聴時ドラ切り"], 1);
  assertEquals(vioTotal(parsed.vioA0), 3);
  assertEquals(vioTotal(parsed.vioB0), 1);
});

// ---------------------------------------------------------------------------
// the parameter table
// ---------------------------------------------------------------------------

Deno.test("the parameter table is well formed", () => {
  assertEquals(PARAMS.length, 12);
  for (const p of PARAMS) {
    assert(p.min > 0, `${p.path}: 対数空間の探索なので下限は正 (${p.min})`);
    assert(p.min <= p.def && p.def <= p.max, `${p.path}: 既定値が境界の外`);
    assert(p.path.split(".").length >= 2, `${p.path}: 節名が要る`);
    assert(
      ["heuristic", "augment", "computed"].includes(p.path.split(".")[0]),
      `${p.path}: 未知の節`,
    );
  }
  // The numeraire is deliberately absent.
  assert(!PARAMS.some((p) => p.path === "heuristic.shanten"));
  // Fractions stay fractions.
  for (const path of ["augment.floor", "computed.tenpaiFloor"]) {
    const p = PARAMS.find((q) => q.path === path)!;
    assert(p.max <= 1, `${path}: 割合なので上限は1以下`);
  }
});

Deno.test("clampVec respects the bounds and repairs non-finite entries", () => {
  const lo = clampVec(PARAMS.map(() => -99));
  const hi = clampVec(PARAMS.map(() => 1e9));
  PARAMS.forEach((p, i) => {
    assertEquals(lo[i], p.min);
    assertEquals(hi[i], p.max);
  });
  const nan = clampVec(PARAMS.map(() => NaN));
  assertEquals(nan, defaults());
  // In-bounds vectors pass through untouched.
  assertEquals(clampVec(defaults()), defaults());
});

Deno.test("toKtune nests by path and names only the tuned keys", () => {
  const k = toKtune(defaults()) as {
    heuristic: Record<string, unknown>;
    augment: Record<string, number>;
    computed: Record<string, number>;
  };
  assertEquals(Object.keys(k).sort(), ["augment", "computed", "heuristic"]);
  assertEquals(k.heuristic.ukeire, 12);
  assertEquals(k.heuristic.dora, 60);
  assertEquals(k.heuristic.danger, { "危険度低": 30, "危険度中": 90, "危険度高": 200 });
  assertEquals(k.augment.lambda, 0.25);
  assertEquals(k.computed.dealinScale, 0.065);
  // Untouched terms must keep tracking their source defaults.
  assertEquals(k.heuristic.shanten, undefined);
  assertEquals(k.heuristic.firstHonor, undefined);
  assertEquals(k.computed.tenpaiPrior, undefined);
  // ...and the file is what `loadKtune` accepts: a plain JSON object.
  assertEquals(JSON.parse(JSON.stringify(k)), k);
});

Deno.test("setPath creates intermediate objects and overwrites leaves", () => {
  const o: Record<string, unknown> = {};
  setPath(o, "a.b.c", 1);
  setPath(o, "a.b.d", 2);
  setPath(o, "a.b.c", 3);
  setPath(o, "top", 4);
  assertEquals(o, { a: { b: { c: 3, d: 2 } }, top: 4 });
});

// ---------------------------------------------------------------------------
// the search operators
// ---------------------------------------------------------------------------

Deno.test("logMean of identical vectors is the identity", () => {
  const v = defaults();
  const m = logMean([v, v, v, v]);
  m.forEach((x, i) => assertAlmostEquals(x, v[i], 1e-9));
  assertEquals(logMean([v]).length, v.length);
  assertEquals(logMean([]), []);
});

Deno.test("logMean is geometric, not arithmetic", () => {
  // The whole reason for log space: {x/4, 4x} must average back to x.
  assertAlmostEquals(logMean([[4], [0.25]])[0], 1, 1e-12);
  assertAlmostEquals(logMean([[2], [8]])[0], 4, 1e-12);
});

Deno.test("perturb is deterministic given the seed and stays in bounds", () => {
  const mean = defaults();
  const a = perturb(mean, 0.25, sfc32(1).fork(0));
  const b = perturb(mean, 0.25, sfc32(1).fork(0));
  assertEquals(a, b, "同じシードで違う候補が出た");
  const c = perturb(mean, 0.25, sfc32(2).fork(0));
  assertNotEquals(a, c, "シードを変えても同じ候補が出た");

  // A wild sigma must still produce a legal vector — that is what the clamp is
  // for, and it is what keeps a negative danger cost (a seat that WANTS to deal
  // in) out of the population.
  for (let s = 0; s < 20; s++) {
    const v = perturb(mean, 5, sfc32(s));
    PARAMS.forEach((p, i) => {
      assert(v[i] >= p.min && v[i] <= p.max, `${p.path}: ${v[i]} が境界外`);
      assert(v[i] > 0);
    });
  }
});

Deno.test("perturb at sigma 0 is the mean itself", () => {
  const mean = defaults();
  const v = perturb(mean, 0, sfc32(9));
  v.forEach((x, i) => assertAlmostEquals(x, mean[i], 1e-9));
});

Deno.test("gauss is centred and scaled", () => {
  const rng = sfc32(31337);
  let s = 0, s2 = 0;
  const n = 4000;
  for (let i = 0; i < n; i++) {
    const g = gauss(rng);
    assert(Number.isFinite(g), "Box–Muller が非有限を返した");
    s += g;
    s2 += g * g;
  }
  assertAlmostEquals(s / n, 0, 0.1);
  assertAlmostEquals(Math.sqrt(s2 / n), 1, 0.1);
});

Deno.test("sigma anneals geometrically from start to finish", () => {
  assertAlmostEquals(sigmaAt(0, 16, 0.25, 0.1), 0.25, 1e-12);
  assertAlmostEquals(sigmaAt(15, 16, 0.25, 0.1), 0.1, 1e-12);
  // Monotone decreasing in between.
  for (let g = 1; g < 16; g++) assert(sigmaAt(g, 16, 0.25, 0.1) < sigmaAt(g - 1, 16, 0.25, 0.1));
  // A one-generation run has nothing to anneal.
  assertEquals(sigmaAt(0, 1, 0.25, 0.1), 0.25);
});

Deno.test("lanes rotate and never touch the validation lane", () => {
  const lanes = new Set<number>();
  for (let g = 0; g < 64; g++) {
    const lane = laneOf(g);
    assert(!lanes.has(lane), "世代が牌山レーンを再利用した");
    lanes.add(lane);
    // 50000 plus a thousand-seed validation run must stay clear of the tuner.
    assert(lane >= 200_000, `検証レーンに近すぎる: ${lane}`);
  }
});

// ---------------------------------------------------------------------------
// selection
// ---------------------------------------------------------------------------

function cand(idx: number, dojo: number | null): Candidate {
  const diff = (mean: number) => ({ mean, sd: 1, ci: 0.1 });
  return {
    idx,
    vec: defaults(),
    stats: dojo === null ? null : {
      games: 250,
      seed: 200_000,
      seats: "khhh",
      rankA: 2.5,
      rankB: 2.5,
      dRank: diff(dojo),
      dRankDojo: diff(dojo),
      dScore: diff(0),
      vioA0: {},
      vioB0: {},
      ms: 1,
    },
  };
}

Deno.test("select takes the most-negative dRankDojo", () => {
  const cands = [cand(0, 0.02), cand(1, -0.11), cand(2, 0.5), cand(3, -0.30), cand(4, -0.05)];
  const elite = select(cands, 3);
  assertEquals(elite.map((c) => c.idx), [3, 1, 4]);
  assertEquals(select(cands, 1).map((c) => c.idx), [3]);
});

Deno.test("select drops failed candidates rather than scoring them zero", () => {
  // A crashed run is not an average result: on this objective 0 would beat every
  // candidate that merely failed to improve.
  const cands = [cand(0, 0.4), cand(1, null), cand(2, 0.9), cand(3, null)];
  const elite = select(cands, 3);
  assertEquals(elite.map((c) => c.idx), [0, 2]);
  assertEquals(select([cand(0, null)], 2), []);
});

Deno.test("select breaks ties by index, so a generation is reproducible", () => {
  const cands = [cand(2, -0.1), cand(0, -0.1), cand(1, -0.1)];
  assertEquals(select(cands, 2).map((c) => c.idx), [0, 1]);
  // ...and always returns at least one when asked for fewer than one.
  assertEquals(select(cands, 0).length, 1);
});

Deno.test("vioTotal sums a label breakdown", () => {
  assertEquals(vioTotal({ a: 2, b: 3 }), 5);
  assertEquals(vioTotal({}), 0);
  assertEquals(vioTotal(undefined), 0);
});

// ---------------------------------------------------------------------------
// the scalar space's incumbent pairing, base merge and --params subset (M10d)
// ---------------------------------------------------------------------------

Deno.test("pairedArgs: scalar 空間は候補を --ktune、現行を --ktune-b で渡す", () => {
  const opt = parseOpts(["--space=scalar", "--seeds=250"]);
  assertEquals(
    pairedArgs(opt, "/o/cand.json", 200_000, undefined, "/o/inc.json"),
    [
      "run",
      "--allow-read",
      "--allow-write",
      "--allow-ffi",
      "--allow-env=MJGAME_NATIVE",
      "src/main.ts",
      "paired",
      "--games=250",
      "--seed=200000",
      "--seats=khhh",
      "--ktune=/o/cand.json",
      "--ktune-b=/o/inc.json",
      "--json",
    ],
  );
  // The absolute anchor row: no incumbent, so the control is plain hhhh again.
  assertEquals(
    pairedArgs(opt, "/o/anchor.json", 200_000).filter((s) => s.startsWith("--ktune")),
    ["--ktune=/o/anchor.json"],
  );
});

Deno.test("pairedArgs: scalar 空間の下敷きはフラグではなく候補ファイルに入る", () => {
  // `--base-ktune` in the scalar space is merged INTO the candidate file (see
  // `candidateKtune`), so exactly one --ktune flag describes the whole seat.
  const opt = parseOpts(["--space=scalar", "--base-ktune=/b/base.json"]);
  const args = pairedArgs(opt, "/o/cand.json", 201_000, undefined, "/o/inc.json");
  assertEquals(args.filter((s) => s.startsWith("--ktune")), [
    "--ktune=/o/cand.json",
    "--ktune-b=/o/inc.json",
  ]);
  assert(!args.some((s) => s.includes("base.json")), "下敷きが子の argv に漏れている");
});

Deno.test("pairedArgs: consumer 空間の argv は M9c のまま", () => {
  const opt = parseOpts(["--space=consumer", "--base-ktune=/b/best.json"]);
  assertEquals(
    pairedArgs(opt, "/o/cand.json", 400_000, undefined, "/o/inc.json"),
    [
      "run",
      "--allow-read",
      "--allow-write",
      "--allow-ffi",
      "--allow-env=MJGAME_NATIVE",
      "src/main.ts",
      "paired",
      "--games=1000",
      "--seed=400000",
      "--seats=khhh",
      "--ktune=/b/best.json",
      "--consumer=/o/cand.json",
      "--consumer-b=/o/inc.json",
      "--json",
    ],
  );
  // …including the curriculum's two flags, in their old order.
  assertEquals(
    pairedArgs(opt, "/o/cand.json", 400_000, 0.5, "/o/inc.json").slice(-3),
    ["--oracle=C1,C2,C3", "--curriculum=0.5", "--json"],
  );
});

Deno.test("scalar 空間は --base-ktune を継承しない", () => {
  // The consumer space defaults to the scalar champion; the scalar space must
  // NOT pick that up, or a plain `tune.ts` run would silently change meaning.
  assertEquals(parseOpts([]).baseKtune, "");
  assertEquals(parseOpts(["--space=scalar"]).baseKtune, "");
  assert(parseOpts(["--space=consumer"]).baseKtune.endsWith("runs/tune/best.json"));
});

Deno.test("deepMerge: 候補は下敷きの上に載り、配列は丸ごと生き残る", () => {
  const base: Nested = {
    heuristic: { dora: 55, firstHonor: 7 },
    computed: {
      dealinScale: 0.09,
      waitNormalize: true,
      tenpaiPrior: [[0.1, 0.2], [0.3, 0.4]],
      shapePrior: { "リャンメン": 0.45, "タンキ": 0.08 },
    },
  };
  const ov: Nested = {
    heuristic: { dora: 60 },
    computed: { dealinScale: 0.065, tenpaiFloor: 0.25 },
  };
  const m = deepMerge(base, ov) as {
    heuristic: Record<string, number>;
    computed: Record<string, unknown>;
  };
  // The overlay wins where it speaks…
  assertEquals(m.heuristic.dora, 60);
  assertEquals(m.computed.dealinScale, 0.065);
  assertEquals(m.computed.tenpaiFloor, 0.25);
  // …and the base survives everywhere it does not, including the fitted table
  // (the whole point: `mergeComputed` merges it cell by cell in the child).
  assertEquals(m.heuristic.firstHonor, 7);
  assertEquals(m.computed.waitNormalize, true);
  assertEquals(m.computed.tenpaiPrior, [[0.1, 0.2], [0.3, 0.4]]);
  assertEquals(m.computed.shapePrior, { "リャンメン": 0.45, "タンキ": 0.08 });
  // Neither argument is mutated.
  assertEquals((base.heuristic as Record<string, number>).dora, 55);
  assertEquals(Object.keys(ov.computed as Nested).sort(), ["dealinScale", "tenpaiFloor"]);
});

Deno.test("candidateKtune: 下敷きなしは toKtune そのもの、ありは deepMerge", () => {
  const v = defaults();
  assertEquals(candidateKtune(v), toKtune(v));
  const base: Nested = { computed: { tenpaiPrior: [[0.5]], waitNormalize: true } };
  const merged = candidateKtune(v, PARAMS, base) as {
    computed: Record<string, unknown>;
    heuristic: Record<string, unknown>;
  };
  // The twelve ride on top; the base's fitted cells reach the seat untouched.
  assertEquals(merged.computed.tenpaiPrior, [[0.5]]);
  assertEquals(merged.computed.waitNormalize, true);
  assertEquals(merged.computed.dealinScale, 0.065);
  assertEquals(merged.heuristic.dora, 60);
  // …and the file is still a plain JSON object `loadKtune` accepts.
  assertEquals(JSON.parse(JSON.stringify(merged)), merged);
});

Deno.test("selectParams: 表の部分集合を表の順で返す", () => {
  const sel = selectParams("heuristic.dora,heuristic.ukeire");
  // Typing order is irrelevant: the coordinate layout is a function of the SET.
  assertEquals(sel.map((p) => p.path), ["heuristic.ukeire", "heuristic.dora"]);
  assertEquals(selectParams("heuristic.ukeire,heuristic.dora").map((p) => p.path), [
    "heuristic.ukeire",
    "heuristic.dora",
  ]);
  assertEquals(selectParams(" computed.tenpaiFloor ").map((p) => p.path), [
    "computed.tenpaiFloor",
  ]);
  assertEquals(selectParams(PARAMS.map((p) => p.path).join(",")).length, PARAMS.length);
});

Deno.test("selectParams: 未知・重複・空はエラー (表つきで)", () => {
  const e = assertThrows(() => selectParams("heuristic.dora,heuristic.nope")) as Error;
  assert(e.message.includes("heuristic.nope"));
  // The message has to say what IS allowed, or a typo costs a source read.
  for (const p of PARAMS) assert(e.message.includes(p.path), `${p.path} が一覧にない`);
  assertThrows(() => selectParams("heuristic.dora,heuristic.dora"));
  assertThrows(() => selectParams(""));
  assertThrows(() => selectParams(",,"));
  // The numeraire is not a searchable coordinate here either.
  assertThrows(() => selectParams("heuristic.shanten"));
});

Deno.test("--params は探索ベクトルの長さと候補ファイルの中身を絞る", () => {
  const opt = parseOpts(["--params=heuristic.dora,computed.tenpaiFloor"]);
  const sel = opt.params;
  assertEquals(sel.map((p) => p.path), ["heuristic.dora", "computed.tenpaiFloor"]);

  const init = defaults(sel);
  assertEquals(init, [60, 0.25]);
  assertEquals(clampVec([1e9, -1], sel), [240, 0.0625]);
  const step = perturb(init, 0.3, sfc32(9), sel);
  assertEquals(step.length, 2);
  sel.forEach((p, i) => assert(step[i] >= p.min && step[i] <= p.max));

  // The candidate file names those two and nothing else — every untouched
  // parameter keeps tracking its default, in the candidate AND in the incumbent.
  const k = candidateKtune(step, sel) as {
    heuristic: Record<string, unknown>;
    computed: Record<string, unknown>;
  };
  assertEquals(Object.keys(k).sort(), ["computed", "heuristic"]);
  assertEquals(Object.keys(k.heuristic), ["dora"]);
  assertEquals(Object.keys(k.computed), ["tenpaiFloor"]);
  assertEquals(k.heuristic.dora, step[0]);
  assertEquals(k.computed.tenpaiFloor, step[1]);
});

Deno.test("state.json は探索した座標を記録する (--params 違いの --resume は拒否)", () => {
  const opt = parseOpts(["--params=heuristic.dora,heuristic.ukeire"]);
  const state: State = {
    space: opt.space,
    gen: 3,
    mean: defaults(opt.params),
    sigma: 0.2,
    seed: 1,
    paths: spacePaths(opt),
    baseKtune: opt.baseKtune,
    curriculum: false,
    incumbentPath: "/o/incumbent-g2.json",
    sigmaScale: 0.85,
  };
  const back = JSON.parse(JSON.stringify(state)) as State;
  assertEquals(back.paths, ["heuristic.ukeire", "heuristic.dora"]);
  assertEquals(back.mean.length, back.paths.length);
  assertEquals(back.paths, spacePaths(opt));
  // A resume under a different subset — or under the full table — must not match.
  assertNotEquals(back.paths, spacePaths(parseOpts(["--params=heuristic.dora"])));
  assertNotEquals(back.paths, spacePaths(parseOpts([])));
  assertEquals(spacePaths(parseOpts([])), PARAMS.map((p) => p.path));
  // The consumer space's coordinates are its curves, not the scalar table.
  assertEquals(spacePaths(parseOpts(["--space=consumer"])).length, 17);
});
