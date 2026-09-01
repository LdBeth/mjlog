// M15 — the WIRING of the `ev` block: loader → seat → CLI → freeze.
//
// `test/ev_native_test.ts` owns the engine (scorer parity, the DP's analytic
// cases, determinism) and `test/evpack_test.ts` owns the wire layout. This file
// owns the SWITCH, and it makes the same two claims `dealin_wiring_test` does,
// plus the two that are new to a block which REQUIRES a dylib:
//
//   ABSENT is bit-identical, and touches no FFI. No `ev` section ⇒ no core is
//   built, nothing is dlopened, and the seat plays the game it has always
//   played. Every test in section 1 must pass on a machine where
//   `deno task build-ev` was never run — that is the claim, not a convenience.
//
//   PRESENT reaches the seat (unit B). `chooseDiscard` prices every candidate
//   with the DP and `computeFold` compares the two root lines, so a seat
//   carrying the block plays a DIFFERENT game — on `kkkk` and on the champion
//   vector against the frozen field.
//
//   AND THE INTEGRATION IS STRUCTURAL, not incidental. Section 7 states the
//   fold derivation against a STUBBED core (no dylib, no FFI): the verdict is
//   `bestFold > bestPush` even on a quiet table, where the incumbent gate's
//   `pressure === 0` early-out would have returned "push" without looking.
//   Section 9 states the other half of the contract — `dealinCostPts` is
//   `riskOf` in points, exit for exit — and section 10 the hidden-information
//   mapping the oracle fills.
//
//   IT REFUSES WHAT IT SUPERSEDES (D3), and refuses it WITHOUT the dylib: the
//   conflict checks in `makePolicy` run before `buildEv`, so a bad table is
//   diagnosed as a bad table rather than as a missing library.
//
//   AND IT HAS NO FALLBACK: `ev` present with no usable `libmjev` THROWS.

import {
  assert,
  assertAlmostEquals,
  assertEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import type { DangerAssessment, DangerLevel } from "mjrender/danger.ts";
import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import { tileType } from "mjrender/tiles.ts";
import { argError } from "../src/cli/args.ts";
import type { ConsumerParams } from "../src/ai/consumer.ts";
import { AugmentedHeuristic } from "../src/ai/augmented.ts";
import type { Reads, ReadsProvider } from "../src/ai/augmented.ts";
import type { EvCore } from "../src/ai/ev.ts";
import { buildEv, closeEv, closeEvLib, evNative } from "../src/ai/ev.ts";
import { DBLS_LEN, INTS_LEN, OUT_LEN, REST_META_LEN } from "../src/ai/evlayout.ts";
import { indicatorOfDora } from "../src/ai/evpack.ts";
import { DEFAULT_EV, mergeEv } from "../src/ai/evparams.ts";
import type { Ctx, EvResult, HeuristicWeights } from "../src/ai/heuristic.ts";
import { doraTypesOf, publicUnseen, valueHonorsOf } from "../src/ai/planner.ts";
import type { Observation } from "../src/observe.ts";
import { JANKI } from "../src/rules.ts";
import { closeArm, headless, loadKtune, makePolicy, openArm } from "../src/harness.ts";
import type { KTune, TableSpec } from "../src/harness.ts";
import type { Action } from "../src/types.ts";
import { tiles } from "./helpers.ts";

const SEED = 8191;
const GAMES = 3;

/** The shipped vector — the block has to compose with everything already on it. */
const CHAMPION: KTune = loadKtune(
  new URL("../weights/champion.json", import.meta.url).pathname,
);
/**
 * The champion with its M12 `riichi` head stripped. The champion CARRIES that
 * head, and `ev.riichi` refuses it (D3) — so every rule that is NOT about the
 * riichi collision has to be stated against a vector the collision does not
 * already reject, or it would pass for the wrong reason.
 */
const NO_RIICHI: KTune = { ...CHAMPION, riichi: undefined };

/**
 * Can this machine actually build a core? Decided ONCE, at load, by trying —
 * `evNative()` answers whether the FFI gate is open, but only `buildEv` knows
 * whether the dylib is there and its ABI matches.
 */
const SKIP = (() => {
  try {
    const c = buildEv(mergeEv({}));
    closeEv(c);
    return false;
  } catch (e) {
    console.log(
      `[M15] libmjev が使えないので EV 配線テストの一部を飛ばします ` +
        `(evNative=${evNative()}): ${e instanceof Error ? e.message.split("\n")[0] : e}`,
    );
    return true;
  }
})();

// ---------------------------------------------------------------------------
// 1. absent ≡ identical (and dylib-free)
// ---------------------------------------------------------------------------

Deno.test("M15 wiring: ev ブロックが無い席はビット単位で従来どおり — kkkk", () => {
  const plain = headless(GAMES, SEED, "kkkk", {});
  const stripped = headless(GAMES, SEED, "kkkk", { ktune: { ev: undefined } });
  assertEquals(stripped.results, plain.results);
});

Deno.test("M15 wiring: ev ブロックが無い席はビット単位で従来どおり — khhh + champion", () => {
  // The champion carries no `ev` block (`champion_test` pins that), so the
  // stripped arm is the same vector spelled explicitly — the identity is
  // stated the way `dealin_wiring_test` states its own.
  const plain = headless(GAMES, SEED, "khhh", { ktune: CHAMPION });
  const stripped = headless(GAMES, SEED, "khhh", {
    ktune: { ...CHAMPION, ev: undefined },
  });
  assertEquals(stripped.results, plain.results);
});

// ---------------------------------------------------------------------------
// 2. present reaches the seat — DEFERRED to unit B
// ---------------------------------------------------------------------------

Deno.test({
  name: "M15 wiring: EV核 を載せた席は別の対局を打つ — kkkk と khhh",
  // Enabled with unit B: `chooseDiscard` and `computeFold` now read the core,
  // so a seat carrying the block MUST play a different game. Until unit B this
  // test was `ignore: true` on purpose — the identity was the correct
  // behaviour for unit A, and would be the silent drop this test exists to
  // catch the moment the integration landed.
  ignore: SKIP,
  fn: () => {
    // `maxNodes` 250 on every ev arm, for the reason the unit C/D aggregates
    // give: at the default 60,000 these four `headless` runs are MINUTES of
    // search each and the suite stops being runnable (2026-08-30 review —
    // measured after the 河底 crash was fixed, which is what used to end these
    // arms early). The claim is that the block REACHES the seat, and a
    // truncated DP is still the DP: it prices with the core's own tail rather
    // than with the linear surrogate, which is exactly the difference under
    // test.
    const plain = headless(GAMES, SEED, "kkkk", {});
    const ev = headless(GAMES, SEED, "kkkk", { ktune: { ev: { maxNodes: 250 } } });
    assert(
      JSON.stringify(ev.results) !== JSON.stringify(plain.results),
      "ev ブロックが席に届いていない (kkkk)",
    );

    // …and on the shipped vector against the frozen field, which is the arm a
    // paired grade actually runs. `ev.riichi` is off because the champion
    // carries the M12 head (D3); unit B's candidate is spelled exactly so.
    const champ = headless(GAMES, SEED, "khhh", { ktune: CHAMPION });
    const withEv = headless(GAMES, SEED, "khhh", {
      ktune: { ...CHAMPION, ev: { riichi: false, maxNodes: 250 } },
    });
    assert(
      JSON.stringify(withEv.results) !== JSON.stringify(champ.results),
      "ev ブロックが席に届いていない (khhh + champion)",
    );
  },
});

// ---------------------------------------------------------------------------
// 3. ownership: built once per seat, freed with the seat
// ---------------------------------------------------------------------------

Deno.test({
  name: "M15 wiring: EV核 は席ごとに一度だけ作られ、close で解放される",
  ignore: SKIP,
  fn: () => {
    const table: TableSpec = [
      { kind: "k", ktune: { ev: {} } },
      { kind: "h" },
      { kind: "h" },
      { kind: "h" },
    ];
    const arm = openArm(table);
    try {
      assertEquals(arm.built.length, 4);
      // `reset` rebuilds the provider chain (that is what `withReads` is for);
      // the core must survive it — a core built inside `build` would allocate a
      // native context per hanchan and free none of them.
      arm.built[0].reset(SEED);
      arm.built[0].reset(SEED + 1);
    } finally {
      // Idempotent by contract — a double close would be a native double-free.
      closeArm(arm);
      closeArm(arm);
    }
  },
});

// ---------------------------------------------------------------------------
// 4. D3 refusals — decided BEFORE the dylib is ever asked for
// ---------------------------------------------------------------------------

/** A "k" seat, built directly: no driver, no oracle tap, no dylib needed. */
function seatWith(ktune: KTune, extra: Record<string, unknown> = {}) {
  return () => makePolicy({ kind: "k", name: "K0", seed: 0, ktune, ...extra });
}

Deno.test("M15 wiring: ev.discard は自分が置き換える節を拒否する (dylib 不要)", () => {
  // Each message must name BOTH sides: the point of the refusal is to say what
  // would have been silently ignored, not merely that something is wrong.
  const names = (fn: () => unknown, other: string) => {
    const e = assertThrows(fn, Error) as Error;
    assertStringIncludes(e.message, "ev");
    assertStringIncludes(e.message, other);
  };
  names(seatWith({ ev: {} }, { consumer: {} as ConsumerParams }), "consumer");
  names(seatWith({ ev: {}, hand: {} }), "hand");
  names(seatWith({ ev: {}, fold: {} }), "fold");
  names(seatWith({ ev: {} }, { foldSink: () => {} }), "foldcalib");

  // …and with the sub-switch off the same vector is legal (the D3 rule is about
  // what the core actually SERVES, not about the block being present).
  const off: KTune = { ev: { discard: false }, hand: {}, fold: {} };
  noConflict(off);
});

/**
 * The layout is accepted: with a dylib it builds (and is freed here); without
 * one it still fails, but on the MISSING LIBRARY and never on a conflict. That
 * order — refusals first, `buildEv` second — is what lets a bad table be
 * diagnosed as a bad table on a machine that never ran `deno task build-ev`.
 */
function noConflict(ktune: KTune) {
  if (SKIP) {
    const e = assertThrows(seatWith(ktune), Error) as Error;
    assert(
      !e.message.includes("併用できません"),
      `衝突しないはずの構成が拒否された: ${e.message}`,
    );
    return;
  }
  seatWith(ktune)().close();
}

Deno.test("M15 wiring: ev.riichi は M12 のリーチヘッドを拒否する (dylib 不要)", () => {
  const e = assertThrows(seatWith({ ev: {}, riichi: {} }), Error) as Error;
  assertStringIncludes(e.message, "ev");
  assertStringIncludes(e.message, "riichi");
  // `ev: {riichi: false}` is how unit B/D carry the block beside the M12 head.
  noConflict({ ev: { riichi: false }, riichi: {} });
});

// ---------------------------------------------------------------------------
// 5. no fallback: present without a usable dylib THROWS
// ---------------------------------------------------------------------------

Deno.test("M15 wiring: ev を載せて libmjev が無ければ makePolicy が落ちる", () => {
  const prev = Deno.env.get("MJGAME_NATIVE");
  Deno.env.set("MJGAME_NATIVE", "0");
  // The gate is resolved on the FIRST open and the handle is then cached for
  // the process, so a test that flips the variable has to re-arm it — that is
  // exactly what `closeEvLib` is for. (Any core built earlier in this file was
  // already freed by its own `close()`.)
  closeEvLib();
  try {
    const e = assertThrows(seatWith({ ev: {} }), Error) as Error;
    // The hint has to name the way out — a seat that silently fell back to the
    // linear surrogate would be graded as the DP.
    assertStringIncludes(e.message, "build-ev");
  } finally {
    if (prev === undefined) Deno.env.delete("MJGAME_NATIVE");
    else Deno.env.set("MJGAME_NATIVE", prev);
    // Left CLOSED, not reopened: the next `buildEv` opens it again under the
    // restored environment, which is the state every other test expects.
    closeEvLib();
  }
});

// ---------------------------------------------------------------------------
// 6. the CLI refusals (the same rules, as messages)
// ---------------------------------------------------------------------------

Deno.test("M15 wiring: --foldcalib は ev.discard を載せた席を断る", () => {
  const base = { cmd: "selfplay", seats: "khhh", calibrate: "" };
  const lane = { ...base, foldcalib: "lane.jsonl" };
  // The plain lane is what the fit needs, and it is still legal.
  assertEquals(argError({ ...lane, ktune: CHAMPION }), null);

  const withEv: KTune = { ...NO_RIICHI, ev: {} };
  const viaKtune = argError({ ...lane, ktune: withEv });
  assert(viaKtune?.includes("ev"), viaKtune ?? "(拒否されなかった)");
  assert(viaKtune?.includes("--foldcalib"), viaKtune ?? "(拒否されなかった)");

  // …and the same vector spelled as a `--table` seat 0.
  const table: TableSpec = [
    { kind: "k", ktune: withEv },
    { kind: "h" },
    { kind: "h" },
    { kind: "h" },
  ];
  const viaTable = argError({ ...lane, table });
  assert(viaTable?.includes("ev"), viaTable ?? "(拒否されなかった)");

  // `ev: {discard: false}` opts out — the lane and the core are then disjoint.
  assertEquals(argError({ ...lane, ktune: { ...NO_RIICHI, ev: { discard: false } } }), null);
  // Without `--foldcalib` the block is nobody's business.
  assertEquals(argError({ ...base, ktune: withEv }), null);
  assertEquals(argError({ ...base, table }), null);
});

Deno.test("M15 wiring: ev.riichi と riichi 節の同居を argError が断る", () => {
  const base = { cmd: "selfplay", seats: "khhh", calibrate: "" };
  // The champion carries a `riichi` block, so this is the realistic collision.
  assert(CHAMPION.riichi, "champion に riichi 節が無い — この検査の前提が崩れている");
  const clash = argError({ ...base, ktune: { ...CHAMPION, ev: {} } });
  assert(clash?.includes("ev"), clash ?? "(拒否されなかった)");
  assert(clash?.includes("riichi"), clash ?? "(拒否されなかった)");
  // Unit B/D's spelling: the core serves discard/calls, the M12 head serves riichi.
  assertEquals(
    argError({ ...base, ktune: { ...CHAMPION, ev: { riichi: false } } }),
    null,
  );
  // And a vector with no `riichi` block at all composes freely.
  assertEquals(argError({ ...base, ktune: { ...CHAMPION, riichi: undefined, ev: {} } }), null);
  // Via `--table` seat 0, too.
  const table: TableSpec = [
    { kind: "k", ktune: { ...CHAMPION, ev: {} } },
    { kind: "h" },
    { kind: "h" },
    { kind: "h" },
  ];
  assert(argError({ ...base, table })?.includes("riichi"));
});

// ---------------------------------------------------------------------------
// 7. the fold derivation, against a STUBBED core (no dylib, no FFI)
// ---------------------------------------------------------------------------
//
// The claim is about the WIRING, so the engine is replaced by a constant: the
// verdict must be `bestFold > bestPush` and nothing else. Stating it this way
// (rather than by playing games with a real core) is what makes the two
// interesting consequences assertable — the quiet-table early-out is bypassed,
// and the own-riichi early-out is NOT.

/** The two reused buffers with no native context behind them (`evpack_test`'s). */
function stubCore(params = mergeEv({})): EvCore {
  return {
    handle: 0,
    params,
    ints: new Int32Array(INTS_LEN),
    dbls: new Float64Array(DBLS_LEN),
    out: new Float64Array(OUT_LEN),
    meta: new Float64Array(REST_META_LEN),
  };
}

function evResult(bestPush: number, bestFold: number): EvResult {
  return {
    total: new Float64Array(34),
    dama: new Float64Array(34),
    riichi: new Float64Array(34),
    foldLine: new Float64Array(34),
    bestPush,
    bestFold,
    nodes: 0,
    truncated: false,
  };
}

/**
 * The seat, with its protected seams opened.
 *
 * `stub` short-circuits `evOf`, which is the only thing that ever touches the
 * dylib; everything else — `evDiscard`'s root test, `computeFold`'s branch,
 * the two pricing hooks — runs exactly as it does in a match.
 */
class Probe extends AugmentedHeuristic {
  stub: EvResult | null = null;
  private readonly prov: ReadsProvider;

  constructor(
    provider: ReadsProvider,
    core: EvCore | undefined,
    weights?: Partial<HeuristicWeights>,
  ) {
    super("P0", 1, provider, { ...(core ? { ev: core } : {}), ...(weights ? { weights } : {}) });
    this.prov = provider;
  }

  protected override evOf(obs: Observation): EvResult {
    return this.stub ?? super.evOf(obs);
  }

  /** `decide` is what normally arms the reads; these probes call the hooks directly. */
  arm(obs: Observation): void {
    this.reads = this.prov(obs);
  }

  verdict(obs: Observation): boolean {
    return this.shouldFold(obs);
  }
  value(obs: Observation): EvResult {
    return this.evOf(obs);
  }
  costPts(ctx: Ctx, tile: Tile): number {
    return this.dealinCostPts(ctx, tile);
  }
  scoreRisk(ctx: Ctx, tile: Tile): number {
    return this.riskOf(ctx, tile);
  }
  probIn(ctx: Ctx, tile: Tile): number {
    return this.dealinProbOf(ctx, tile);
  }
  hidden(obs: Observation): ReturnType<AugmentedHeuristic["hiddenInfoOf"]> {
    return this.hiddenInfoOf(obs);
  }
  /** Unit D: the two call hooks, asked directly (`decide` would want a whole turn). */
  callChoice(ctx: Ctx, legal: Action[]): Action | null {
    return this.chooseCall(ctx, legal);
  }
  kanChoice(ctx: Ctx, legal: Action[]): Action | null {
    return this.chooseKan(ctx, legal);
  }
}

// --- board scaffolding (hand-built, as `evpack_test` builds its own) --------

function river(ts: Tile[]): RiverEntry[] {
  return ts.map((tile, i) => ({ tile, junme: i + 1, tsumogiri: false, riichiDeclare: false }));
}

function threat(level: DangerLevel, seat = 1): DangerAssessment {
  return { level, seats: [seat], details: [{ seat, level, kind: "riichi", notes: [] }] };
}

function board(over: Partial<Observation> = {}): Observation {
  const hand = over.hand ?? tiles("123456789m1122p東");
  return {
    seat: 0,
    kyoku: 0,
    honba: 0,
    kyotaku: 0,
    junme: 8,
    wallRemaining: 40,
    hand,
    drawn: "drawn" in over ? over.drawn! : hand[hand.length - 1],
    melds: [[], [], [], []],
    rivers: [river(tiles("南")), river(tiles("北")), river(tiles("西")), river(tiles("發"))],
    scores: [25000, 25000, 25000, 25000],
    riichi: [false, false, false, false],
    riichiJunme: [-1, -1, -1, -1],
    doraIndicators: tiles("9s"),
    seatWind: 27,
    roundWind: 27,
    akaIds: JANKI.akaIds,
    shanten: 0,
    waits: [],
    ronnable: [],
    katagari: false,
    discardInfo: new Map(),
    tsumogiriLock: false,
    ukeire: [],
    doraCount: 0,
    furiten: { permanent: false, temporary: false, riichi: false },
    danger: new Map(),
    violations: [0, 0, 0, 0],
    legal: [],
    ...over,
  };
}

function ctxOf(obs: Observation): Ctx {
  return {
    obs,
    open: obs.melds[0].length,
    closed: obs.melds[0].every((m) => m.kind === "ankan"),
    doraTypes: doraTypesOf(obs),
    valueHonors: valueHonorsOf(obs.roundWind, obs.seatWind),
    unseen: publicUnseen(obs),
    folding: false,
    canRiichi: false,
    eff: 1,
    def: 1,
  };
}

const noReads: ReadsProvider = () => null;

Deno.test("M15 wiring: 押し引きは EV核 の二本の線の比較 — 静かな卓でも早期脱出しない", () => {
  const p = new Probe(noReads, stubCore());

  // A QUIET table: no riichi, no furo threat, no danger entry ⇒ `pressureOf`
  // is 0 and the incumbent gate returns "push" without ever pricing anything.
  // With the core on, the DP's own hazard sweep is what decides.
  p.stub = evResult(-1200, 800);
  assertEquals(p.verdict(board()), true, "bestFold > bestPush なら降り");

  p.stub = evResult(2400, 800);
  assertEquals(p.verdict(board()), false, "bestPush ≥ bestFold なら押し");

  // The tie goes to pushing — `>` and not `>=`, the same way every other
  // comparison in this file breaks a tie toward doing nothing special.
  p.stub = evResult(500, 500);
  assertEquals(p.verdict(board()), false, "同値は押し");
});

Deno.test("M15 wiring: 自リーチ後の早期脱出は EV核 の外に残る", () => {
  const p = new Probe(noReads, stubCore());
  p.stub = evResult(-9000, 9000);
  // After our own declaration the only legal discard is the drawn tile: a
  // "fold" verdict would be a statement with no action behind it.
  assertEquals(p.verdict(board({ riichi: [true, false, false, false] })), false);
});

Deno.test("M15 wiring: 鳴き判断 (13枚) には打牌根が無く、従来の門を通る", () => {
  const p = new Probe(noReads, stubCore());
  p.stub = evResult(-9000, 9000);
  // A claim decision rests on 3n+1 tiles. Unit B serves the discard and the
  // fold verdict only; the call comparisons are unit D.
  const claim = board({ hand: tiles("123456789m1122p"), drawn: null });
  assertEquals(claim.hand.length % 3, 1);
  assertEquals(p.verdict(claim), false, "静かな卓の従来の門は押し");
});

Deno.test("M15 wiring: ev.discard を切れば核は載っていても判断に入らない", () => {
  const p = new Probe(noReads, stubCore(mergeEv({ discard: false })));
  p.stub = evResult(-9000, 9000);
  assertEquals(p.verdict(board()), false, "サブスイッチが切れていれば従来の門");
});

// ---------------------------------------------------------------------------
// 8. …and against the real core
// ---------------------------------------------------------------------------

Deno.test({
  name: "M15 wiring: 実核の押し引きも bestFold > bestPush と一致する",
  ignore: SKIP,
  fn: () => {
    const core = buildEv(mergeEv({}));
    try {
      const p = new Probe(noReads, core);
      const danger = new Map<number, DangerAssessment>();
      for (let ty = 0; ty < 34; ty++) danger.set(ty, threat("危険度高"));
      const obs = board({
        riichi: [false, true, false, false],
        riichiJunme: [-1, 5, -1, -1],
        danger,
      });
      const verdict = p.verdict(obs);
      const e = p.value(obs);
      assertEquals(verdict, e.bestFold > e.bestPush);
      assert(p.evStats.calls > 0, "EV核 が一度も呼ばれていない");
      assert(Number.isFinite(e.bestPush) && Number.isFinite(e.bestFold));
      // The memo is per Observation identity: two questions, one native call.
      p.verdict(obs);
      p.value(obs);
      assertEquals(p.evStats.calls, 1, "同じ盤面で核が二度呼ばれた");
    } finally {
      closeEv(core);
    }
  },
});

// ---------------------------------------------------------------------------
// 9. the pricing hooks: `dealinCostPts` IS `riskOf`, in points
// ---------------------------------------------------------------------------

/** A stub reader: whatever the test wants the seat to believe about the table. */
function readsOf(r: Reads): ReadsProvider {
  return () => r;
}

function f34(fill = 0): Float32Array {
  return new Float32Array(34).fill(fill);
}

Deno.test("M15 hooks: dealinCostPts は riskOf × pointsPerScore に一致する", () => {
  const pps = DEFAULT_EV.pointsPerScore;
  const danger = new Map<number, DangerAssessment>();
  danger.set(0, threat("安全")); // 1m: a proof
  danger.set(1, threat("危険度低")); // 2m: the ladder's cheap rung
  danger.set(4, threat("危険度高")); // 5m: where the estimate should dominate
  // 9m (type 8) is deliberately left UNASSESSED: absence of assessment is not
  // absence of danger, and both sides have to price it the same way.
  const obs = board({ riichi: [false, true, false, false], danger });
  const ctx = ctxOf(obs);

  // A read that is hot on 5m and cold elsewhere, with a payment to go with it.
  const dealinP = [f34(), f34(), f34()];
  const dealinValue = [f34(), f34(), f34()];
  dealinP[0][4] = 0.08;
  dealinValue[0][4] = 8000;
  dealinP[0][1] = 0.01;
  dealinValue[0][1] = 3900;
  dealinP[0][0] = 0.5; // …on a tile the assessor proved safe: must stay free.
  dealinValue[0][0] = 12000;
  const reads: Reads = { dealinP, dealinValue, expLoss: [5200, 0, 0], tenpaiP: [1, 0, 0] };

  const p = new Probe(readsOf(reads), stubCore());
  p.arm(obs);
  for (const tile of obs.hand) {
    assertAlmostEquals(
      p.costPts(ctx, tile),
      p.scoreRisk(ctx, tile) * pps,
      1e-9,
      `型 ${tileType(tile)}`,
    );
  }
  // The proof is honoured: 安全 costs nothing however hot the estimate is.
  assertEquals(p.costPts(ctx, tiles("1m")[0]), 0);
  // …and the estimate is genuinely reaching the price on the hot tile.
  assert(p.costPts(ctx, tiles("5m")[0]) > 0);
  // `dealinP` rows are Float32Array, so the comparison is at float32 precision.
  assertAlmostEquals(p.probIn(ctx, tiles("5m")[0]), 0.08, 1e-7);

  // With NO read at all the hook is the rule ladder in points, and still equal
  // to `riskOf` — the base policy's contract, through the augmented class.
  const blind = new Probe(noReads, stubCore());
  blind.arm(obs);
  for (const tile of obs.hand) {
    assertAlmostEquals(blind.costPts(ctx, tile), blind.scoreRisk(ctx, tile) * pps, 1e-9);
  }
  assertEquals(
    blind.probIn(ctx, tiles("5m")[0]),
    0,
    "読みが無ければ確率は 0 (推定が無いという意味)",
  );
});

Deno.test("M15 hooks: 感性の割増も points に換算されて乗る", () => {
  // `liveYakuhai` fires on a value honor nobody has shown, from 6巡目, with an
  // opponent's call on the table and no danger entry for the tile — the one
  // surcharge that prices where the assessor does not look.
  const melds: Observation["melds"] = [[], [], [], []];
  melds[1] = [{
    kind: "pon",
    who: 1,
    fromWho: 0,
    tiles: tiles("111s"),
    calledTile: tiles("1s")[0],
  }];
  const obs = board({ hand: tiles("123456789m112p白"), melds });
  const ctx = ctxOf(obs);
  const p = new Probe(noReads, stubCore(), { liveYakuhai: 200 });
  p.arm(obs);
  const haku = tiles("白")[0];
  assertEquals(p.scoreRisk(ctx, haku), 200, "生牌の役牌に割増が乗っている");
  assertEquals(p.costPts(ctx, haku), 200 * DEFAULT_EV.pointsPerScore);
});

// ---------------------------------------------------------------------------
// 10. the hidden-information mapping (plan D7)
// ---------------------------------------------------------------------------

Deno.test("M15 hooks: hiddenInfoOf は読みを分布チャネルに写す", () => {
  const obs = board();
  // 4m as the next own draw; 3p as the next kan-DORA (so the INDICATOR is 2p).
  const wall = f34();
  wall[0] = 3;
  wall[27] = 1;
  const p = new Probe(readsOf({ ownNextDraw: 3, nextDora: 11, wallComposition: wall }), stubCore());
  p.arm(obs);
  const h = p.hidden(obs);
  assert(h, "チャネルが埋まっていない");
  assertEquals(h.drawDist?.length, 1, "K=1: 次の自摸ひとつだけ");
  assertEquals(Array.from(h.drawDist![0]), Array.from({ length: 34 }, (_, i) => (i === 3 ? 1 : 0)));
  assertEquals(indicatorOfDora(11), 10, "3p のドラ表示牌は 2p");
  assertEquals(Array.from(h.nextDora!), Array.from({ length: 34 }, (_, i) => (i === 10 ? 1 : 0)));
  // The pool is a COMPOSITION (counts), not a normalised distribution: it
  // replaces `publicUnseen`, which is counted the same way.
  assertEquals(h.pool![0], 3);
  assertEquals(h.pool![27], 1);
  assertEquals(h.ura, undefined, "ウラは読みが持たない (将来の学習モジュールの席)");

  // Absent channels ⇒ null, and a read carrying none of the three ⇒ null too:
  // the computed seat's posterior stays the uniform unseen pool.
  const blind = new Probe(noReads, stubCore());
  blind.arm(obs);
  assertEquals(blind.hidden(obs), null);
  const partial = new Probe(readsOf({ tenpaiP: [1, 0, 0] }), stubCore());
  partial.arm(obs);
  assertEquals(partial.hidden(obs), null);

  // `riichiNextDraw` is per-opponent SEQUENTIAL information and is NOT consumed
  // (plan D7) — the constitution forbids the 計算 seat to read it.
  const seq = new Probe(readsOf({ riichiNextDraw: [4, null, null] }), stubCore());
  seq.arm(obs);
  assertEquals(seq.hidden(obs), null, "riichiNextDraw はチャネルに入らない");
});

// ---------------------------------------------------------------------------
// 11. UNIT C — the riichi question
// ---------------------------------------------------------------------------
//
// The four gates of `wantRiichi` are untouched doctrine; what unit C replaces
// is the JUDGEMENT inside them — `ev.riichi[ty] > ev.dama[ty] + riichiMargin`
// for the chosen tile, in place of the M12 head's verdict (D3 refuses a vector
// carrying both). Two kinds of claim below: the decision-level one, where a
// hand-built tenpai board makes the comparison the only moving part, and the
// aggregate one, where the margin has to move the declaration RATE.

/** Riichi declarations summed over the whole table (every kkkk seat is an ev seat). */
function totalOf(run: ReturnType<typeof headless>, key: "riichis" | "furoRounds"): number {
  return run.results.reduce(
    (s, m) => s + (m[key] ?? [0, 0, 0, 0]).reduce((a, b) => a + b, 0),
    0,
  );
}

/** …and the same for seat 0 alone, which is the only seat a `khhh` claim is about. */
function seat0Of(run: ReturnType<typeof headless>, key: "riichis" | "furoRounds"): number {
  return run.results.reduce((s, m) => s + (m[key] ?? [0, 0, 0, 0])[0], 0);
}

/**
 * A closed tenpai with EXACTLY ONE discard on offer, plain or riichi: 123m
 * 456m 789m 12p 55p, cutting the 9s. With the choice of tile settled by the
 * legal list, the only thing left to observe is the declaration — which is
 * precisely the question unit C answers.
 */
function riichiBoard(): { obs: Observation; cut: Tile } {
  const hand = tiles("123456789m1255p9s");
  const cut = hand[hand.length - 1];
  const legal: Action[] = [
    { t: "discard", tile: cut, riichi: false, tsumogiri: true },
    { t: "discard", tile: cut, riichi: true, tsumogiri: true },
  ];
  // 3p (type 11) is the wait the 12p ryanmen leaves; 4 copies are live.
  return {
    obs: board({
      hand,
      drawn: cut,
      shanten: 0,
      waits: [11],
      ukeire: [{ type: 11, live: 4 }],
      legal,
    }),
    cut,
  };
}

Deno.test({
  name: "M15 unit C: リーチ判断は ev.riichi[ty] > ev.dama[ty] + riichiMargin",
  ignore: SKIP,
  fn: () => {
    const { obs } = riichiBoard();

    // The margin is the whole comparison here: the same board, the same tile,
    // three verdicts. `mustCure` is false throughout, so nothing but the price
    // is deciding.
    const verdicts = new Map<number, boolean>();
    for (const riichiMargin of [0, -1e9, 1e9]) {
      const core = buildEv(mergeEv({ riichiMargin }));
      try {
        const p = new Probe(noReads, core);
        const a = p.decide(obs);
        assertEquals(a.t, "discard");
        const declared = a.t === "discard" && a.riichi;
        assertEquals(p.lastTrace?.riichi, declared, "trace と行動が食い違っている");
        assertEquals(p.lastTrace?.mustCure, false, "片和了りの強制ではない盤面のはず");
        // The trace carries the two numbers the decision was made on (B2's
        // fields, unit C's inputs) whether or not it declared.
        assert(Number.isFinite(p.lastTrace?.riichiValue ?? NaN), "riichiValue が記録されていない");
        assert(Number.isFinite(p.lastTrace?.foldValue ?? NaN), "foldValue が記録されていない");
        verdicts.set(riichiMargin, declared);
      } finally {
        closeEv(core);
      }
    }
    // The two claims that hold of ANY engine: an infinitely generous margin
    // declares wherever the four gates admit, and an infinitely mean one never
    // does. The default margin's verdict on this board is the DP's own opinion
    // and is deliberately NOT pinned here — that is the engine's test file,
    // not the switch's.
    assertEquals(verdicts.get(-1e9), true, "margin −1e9 なら門が通す限り宣言する");
    assertEquals(verdicts.get(1e9), false, "margin +1e9 なら決して宣言しない");

    // With no core the same board is the pre-M15 answer: the four gates admit
    // it and, with no M12 head, declaring is unconditional.
    const plain = new Probe(noReads, undefined);
    const a = plain.decide(obs);
    assert(a.t === "discard" && a.riichi, "核が無ければ従来どおり無条件で宣言する");
  },
});

Deno.test({
  name: "M15 unit C: ev.riichi の入切で対局も宣言数も変わる — kkkk",
  ignore: SKIP,
  fn: () => {
    // `calls: false` on BOTH arms: unit D is a separate switch, and a test that
    // let it move too would not be about the riichi question. The ON arm
    // carries a hostile margin: on a bare seat (no M12 head) the OFF path
    // declares unconditionally and the DP's own judgement can agree with it
    // game for game, so "on vs off with the default margin" is not a claim
    // about the SWITCH — the margin is what only the switched-on path reads.
    // `maxNodes` 250 on BOTH arms (2026-08-30 review): the default 60,000 makes
    // each of these `headless` runs minutes of search, and the claim is about
    // the SWITCH, which a truncated DP flips just as visibly.
    const on = headless(GAMES, SEED, "kkkk", {
      ktune: { ev: { calls: false, riichiMargin: 1e9, maxNodes: 250 } },
    });
    const off = headless(GAMES, SEED, "kkkk", {
      ktune: { ev: { riichi: false, calls: false, maxNodes: 250 } },
    });
    assert(
      JSON.stringify(on.results) !== JSON.stringify(off.results),
      "ev.riichi が席に届いていない",
    );
    assert(
      totalOf(on, "riichis") !== totalOf(off, "riichis"),
      `宣言数が同じ (${totalOf(on, "riichis")}) — 判断が入れ替わっていない`,
    );
  },
});

Deno.test({
  name: "M15 unit C: riichiMargin が宣言率を単調に動かす — kkkk",
  ignore: SKIP,
  fn: () => {
    // A longer arm than the identity tests: a RATE needs events, and the four
    // ev seats of a `kkkk` table supply four times as many as a `khhh` one.
    // `maxNodes` is cut to keep the four arms inside a test suite's budget; the
    // claim is about the ORDER of three rates, which the node cap does not
    // reorder (D5: truncation is deterministic and applies to every arm alike).
    const games = 6;
    const arm = (ev: Record<string, unknown>) =>
      totalOf(
        headless(games, SEED, "kkkk", { ktune: { ev: { calls: false, maxNodes: 250, ...ev } } }),
        "riichis",
      );
    const mid = arm({});
    const lo = arm({ riichiMargin: -1e9 });
    const hi = arm({ riichiMargin: 1e9 });
    // The unconditional-declare path: `ev.riichi` off, and no M12 head on a
    // bare `kkkk` seat, so every opportunity the four gates admit is taken.
    const uncond = totalOf(
      headless(games, SEED, "kkkk", {
        ktune: { ev: { riichi: false, calls: false, maxNodes: 250 } },
      }),
      "riichis",
    );

    // NOTE: the unconditional-declare path (`ev.riichi` off, no M12 head) is NOT
    // a ceiling for the −1e9 arm: the DP's dama line is push-only (the fold
    // option lives in O_BEST_FOLD), so a priced margin can declare on shapes the
    // unconditional path never reaches tenpai with, and vice versa. Only the
    // ordering claims above are the contract; `uncond` is kept for the log.
    console.log(`riichi 宣言数 kkkk: +1e9 ${hi} / 既定 ${mid} / −1e9 ${lo} / 無条件 ${uncond}`);
    // What survives at +1e9 is the dojo's own 片和了り cure (`mustCure`), which
    // is a prescription and not a judgement: no margin can veto it.
    assert(hi >= 0);
  },
});

Deno.test({
  name: "M15 unit C: ev.riichi を切れば riichiMargin は読まれない (M12 の道は不変)",
  ignore: SKIP,
  fn: () => {
    // Unit B's vector — the champion's M12 head deciding riichi, the core
    // deciding the discard — must be untouched by unit C. The observable form
    // of "untouched" is that the parameter unit C reads is INERT here: if the
    // riichi branch had leaked outside its switch, a margin of ±1e9 would
    // rewrite every declaration on the table.
    // Same budget on every arm, for `別の対局を打つ`'s reason.
    const unitB: KTune = { ...CHAMPION, ev: { riichi: false, calls: false, maxNodes: 250 } };
    const base = headless(GAMES, SEED, "khhh", { ktune: unitB });
    for (const riichiMargin of [1e9, -1e9]) {
      const same = headless(GAMES, SEED, "khhh", {
        ktune: { ...CHAMPION, ev: { riichi: false, calls: false, maxNodes: 250, riichiMargin } },
      });
      assertEquals(same.results, base.results, `riichiMargin=${riichiMargin} が漏れている`);
    }
  },
});

// ---------------------------------------------------------------------------
// 12. UNIT D — the call and kan questions
// ---------------------------------------------------------------------------
//
// The vetoes stay (`hasYakuProspect`, the referee's compliance test, the dojo's
// kan rules); what changes is the ACCEPTANCE RULE. "Only a call that buys a
// shanten step" is a rule about speed; the DP prices the whole hand, so a call
// that buys no step can still be worth taking and a step can still be worth
// declining.

/**
 * A 役牌 pon offered at 2向聴 that buys NO shanten step: 23m 4p 888p 23s 67s 東
 * with a 白 pair, ponning the third 白. `shanten(hand) === shanten(rest, 1
 * meld) === 2`, so the incumbent's acceptance rule declines it by construction
 * — and the pon is exactly the kind of call the rule was wrong about, since it
 * turns a yakuless shape into a scoring one.
 */
function valuePonBoard(): { obs: Observation; legal: Action[] } {
  const hand = tiles("23m4p888p23s67s東白白");
  const pair: [Tile, Tile] = [hand[hand.length - 2], hand[hand.length - 1]];
  const legal: Action[] = [
    { t: "pon", tiles: pair, called: (31 * 4 + 2) as Tile },
    { t: "pass" },
  ];
  return { obs: board({ hand, drawn: null, shanten: 2, waits: [], ukeire: [], legal }), legal };
}

/** 1111m 234p 567p 99s 56s: closed tenpai on 4s/7s, and the ankan leaves it alone. */
function ankanBoard(): { obs: Observation; legal: Action[] } {
  const hand = tiles("1111m234567p99s56s");
  const legal: Action[] = [
    { t: "ankan", type: 0 },
    { t: "discard", tile: hand[0], riichi: false, tsumogiri: false },
  ];
  return {
    obs: board({
      hand,
      drawn: hand[0],
      shanten: 0,
      waits: [21, 24],
      ukeire: [{ type: 21, live: 4 }, { type: 24, live: 4 }],
      legal,
    }),
    legal,
  };
}

Deno.test({
  name: "M15 unit D: 向聴が進まない役牌ポンを、価格が受け入れる",
  ignore: SKIP,
  fn: () => {
    const { obs, legal } = valuePonBoard();
    const ctx = ctxOf(obs);

    // The incumbent declines: the call buys no step, and that was the whole
    // acceptance rule.
    assertEquals(new Probe(noReads, undefined).callChoice(ctx, legal), null);

    // The priced seat accepts it when the margin lets anything through — which
    // is what proves the acceptance rule MOVED rather than merely gained a veto.
    const cheap = buildEv(mergeEv({ callMargin: -1e9 }));
    try {
      const a = new Probe(noReads, cheap).callChoice(ctx, legal);
      assertEquals(a?.t, "pon", "価格の門を全開にしても鳴かない — 受理規則が動いていない");
    } finally {
      closeEv(cheap);
    }

    // …and declines it when the margin closes, so the comparison is a real one
    // in both directions.
    const dear = buildEv(mergeEv({ callMargin: 1e9 }));
    try {
      assertEquals(new Probe(noReads, dear).callChoice(ctx, legal), null);
    } finally {
      closeEv(dear);
    }
  },
});

Deno.test({
  name: "M15 unit D: 暗槓も価格で決まる (道場の門は先に通る)",
  ignore: SKIP,
  fn: () => {
    const { obs, legal } = ankanBoard();
    const ctx = ctxOf(obs);

    // The incumbent takes any ankan the dojo admits.
    assertEquals(new Probe(noReads, undefined).kanChoice(ctx, legal)?.t, "ankan");

    const dear = buildEv(mergeEv({ callMargin: 1e9 }));
    try {
      assertEquals(new Probe(noReads, dear).kanChoice(ctx, legal), null, "価格で断れていない");
    } finally {
      closeEv(dear);
    }

    const cheap = buildEv(mergeEv({ callMargin: -1e9 }));
    try {
      assertEquals(new Probe(noReads, cheap).kanChoice(ctx, legal)?.t, "ankan");
    } finally {
      closeEv(cheap);
    }

    // The dojo's vetoes run BEFORE the price and the price cannot buy past
    // them: the same hand at 1向聴 is not a kan whatever it is worth.
    const notTenpai = board({ ...obs, shanten: 1 });
    const free = buildEv(mergeEv({ callMargin: -1e9 }));
    try {
      assertEquals(new Probe(noReads, free).kanChoice(ctxOf(notTenpai), legal), null);
    } finally {
      closeEv(free);
    }
  },
});

Deno.test({
  name: "M15 unit D: ev.calls の入切で対局も副露数も変わる — kkkk",
  ignore: SKIP,
  fn: () => {
    const games = 6;
    const on = headless(games, SEED, "kkkk", { ktune: { ev: { maxNodes: 250 } } });
    const off = headless(games, SEED, "kkkk", { ktune: { ev: { calls: false, maxNodes: 250 } } });
    assert(
      JSON.stringify(on.results) !== JSON.stringify(off.results),
      "ev.calls が席に届いていない",
    );
    assert(
      totalOf(on, "furoRounds") !== totalOf(off, "furoRounds"),
      `副露数が同じ (${totalOf(on, "furoRounds")}) — 受理規則が入れ替わっていない`,
    );
  },
});

Deno.test({
  name: "M15 unit D: callMargin +1e9 なら席0は一度も鳴かない",
  ignore: SKIP,
  fn: () => {
    const run = headless(6, SEED, "kkkk", {
      ktune: { ev: { maxNodes: 250, callMargin: 1e9 } },
    });
    assertEquals(seat0Of(run, "furoRounds"), 0, "無限大の閾値を越えて鳴いた");
    // The kan half of the same claim is stated at the decision level above
    // (`furoRounds` counts OPEN melds, and an 暗槓 leaves the hand closed).
  },
});

Deno.test({
  name: "M15 unit D: ev.calls を切れば callMargin は読まれない (従来の鳴きは不変)",
  ignore: SKIP,
  fn: () => {
    // Same budget on every arm, for `別の対局を打つ`'s reason.
    const unitB: KTune = { ...CHAMPION, ev: { riichi: false, calls: false, maxNodes: 250 } };
    const base = headless(GAMES, SEED, "khhh", { ktune: unitB });
    for (const callMargin of [1e9, -1e9]) {
      const same = headless(GAMES, SEED, "khhh", {
        ktune: { ...CHAMPION, ev: { riichi: false, calls: false, maxNodes: 250, callMargin } },
      });
      assertEquals(same.results, base.results, `callMargin=${callMargin} が漏れている`);
    }
  },
});

Deno.test({
  name: "M15 units C/D: 三つのスイッチを全部切れば ev ブロックが無いのと同じ対局",
  ignore: SKIP,
  fn: () => {
    // The strongest statement available about the REFACTOR the two units
    // needed (`callShape` out of `shantenAfterCall`, the pricing facts out of
    // `evOf`): with every sub-switch off, the seat is the incumbent, tile for
    // tile, across a whole field.
    const plain = headless(GAMES, SEED, "khhh", { ktune: NO_RIICHI });
    const allOff = headless(GAMES, SEED, "khhh", {
      ktune: { ...NO_RIICHI, ev: { discard: false, riichi: false, calls: false } },
    });
    assertEquals(allOff.results, plain.results);
  },
});

// ---------------------------------------------------------------------------
// 13. THE 2026-08-30 REVIEW — what the units left wired wrong
// ---------------------------------------------------------------------------
//
// Every test below is the witness for one finding of the M15 integration
// review. They are stated at the DECISION level, mostly against a stubbed core,
// because each bug was a wiring bug: the engine answered correctly and the seat
// read the wrong number, or asked about the wrong hand.

/** A probe whose two planner hooks count their own calls. */
class Counting extends Probe {
  keeps = 0;
  draws = 0;
  protected override keepBonus(ctx: Ctx, tile: Tile): number {
    this.keeps++;
    return super.keepBonus(ctx, tile);
  }
  protected override drawBonus(ctx: Ctx, tile: Tile): number {
    this.draws++;
    return super.drawBonus(ctx, tile);
  }
}

/** Discard actions for `ts`, plus a riichi variant for whatever `withRiichi` holds. */
function discards(ts: Tile[], withRiichi: Set<Tile> = new Set()): Action[] {
  const out: Action[] = [];
  for (const tile of ts) {
    out.push({ t: "discard", tile, riichi: false, tsumogiri: false });
    if (withRiichi.has(tile)) out.push({ t: "discard", tile, riichi: true, tsumogiri: false });
  }
  return out;
}

/** An `EvResult` with per-type numbers set by hand. */
function priced(
  rows: Record<number, { total?: number; dama?: number; riichi?: number; fold?: number }>,
  bestPush: number,
  bestFold: number,
): EvResult {
  const e = evResult(bestPush, bestFold);
  e.total.fill(-Infinity);
  e.dama.fill(-Infinity);
  e.riichi.fill(-Infinity);
  e.foldLine.fill(-Infinity);
  for (const [k, v] of Object.entries(rows)) {
    const ty = Number(k);
    if (v.total !== undefined) e.total[ty] = v.total;
    if (v.dama !== undefined) e.dama[ty] = v.dama;
    if (v.riichi !== undefined) e.riichi[ty] = v.riichi;
    if (v.fold !== undefined) e.foldLine[ty] = v.fold;
  }
  return e;
}

Deno.test("M15 review: 降りる席は fold 線の argmax を切る (push 線ではない)", () => {
  // THE FATAL ONE. `computeFold` returns `bestFold > bestPush`, and then every
  // candidate was ranked by `O_TOTAL` — the PUSH line, which contains no fold
  // option at all. The two argmaxes are different tiles by construction, so the
  // seat announced a fold and threw the tile that best advanced the hand it had
  // just abandoned.
  const hand = tiles("123456789m1122p東");
  const [ichiman] = hand; // 1m, type 0 — the fold line's tile (安全)
  const kyuman = hand[8]; // 9m, type 8 — the push line's tile
  const ton = hand[hand.length - 1]; // 東, type 27
  const legal = discards([ichiman, kyuman, ton]);
  const obs = board({ hand, legal, riichi: [false, true, false, false] });

  const rows = {
    0: { total: -5000, dama: -5000, fold: 2000 },
    8: { total: 9000, dama: 9000, fold: -9000 },
    27: { total: -5000, dama: -5000, fold: -1000 },
  };

  // FOLDING: `bestFold > bestPush`, so the fold line is the one being played.
  const folder = new Probe(noReads, stubCore());
  folder.stub = priced(rows, -1000, 2000);
  const fold = folder.decide(obs);
  assert(fold.t === "discard");
  assertEquals(folder.verdict(obs), true, "この盤面は降りと判定されているはず");
  assertEquals(tileType(fold.tile), 0, "降りているのに push 線の argmax を切った");
  assertEquals(folder.lastTrace?.folding, true);

  // PUSHING: the same four vectors, the other verdict — and the other tile.
  const pusher = new Probe(noReads, stubCore());
  pusher.stub = priced(rows, 9000, 2000);
  const push = pusher.decide(obs);
  assert(push.t === "discard");
  assertEquals(pusher.verdict(obs), false);
  assertEquals(tileType(push.tile), 8, "押しているのに fold 線で選んだ");
});

Deno.test("M15 review: EV 経路では立案者の keepBonus/drawBonus を呼ばない", () => {
  // `planKeep` is 5000 SCORE units; at `pointsPerScore` 4 that is a 20,000
  // point thumb on a scale where a whole hand is usually worth less. The DP
  // prices the shape, so the plan's steering terms are not a correction to it —
  // they are a different agent's opinion, loud enough to be the only one heard.
  const hand = tiles("123456789m1122p東");
  const legal = discards([hand[0], hand[8], hand[hand.length - 1]]);
  const obs = board({ hand, legal });

  const ev = new Counting(noReads, stubCore());
  ev.stub = priced(
    { 0: { total: 10, dama: 10 }, 8: { total: 5, dama: 5 }, 27: { total: 1, dama: 1 } },
    10,
    0,
  );
  ev.decide(obs);
  assertEquals([ev.keeps, ev.draws], [0, 0], "EV 経路が立案者の項を読んでいる");

  // …and the incumbent still does: the hooks were not deleted, they were taken
  // off the path the DP owns.
  const plain = new Counting(noReads, undefined);
  plain.decide(obs);
  assert(plain.keeps > 0 && plain.draws > 0, "従来経路から立案者の項が消えている");
});

Deno.test("M15 review: 禁じ手のリーチは dama 線で値付けする", () => {
  // `O_TOTAL` is `max(dama, riichi)`, but the declaration can still be refused
  // — 地獄単騎 and 即引っかけ are the dojo's, and the DP knows neither. Pricing
  // a candidate by a riichi it will not be allowed to declare picks the tile
  // for a line the seat then cannot play.
  const hand = tiles("123456789m1122p東");
  const ichiman = hand[0]; // 1m
  const nipin = hand[hand.length - 2]; // 2p (1122p の二枚目)
  const legal = discards([ichiman, nipin], new Set([ichiman]));
  const rows = {
    0: { total: 5000, dama: 0, riichi: 5000 },
    10: { total: 1000, dama: 1000 },
  };

  // 即引っかけ: the wait (4m, type 3) is one suji from the tile being cut (1m).
  const banned = new Probe(noReads, stubCore());
  banned.stub = priced(rows, 5000, 0);
  const a = banned.decide(board({ hand, legal, waits: [3], ukeire: [{ type: 3, live: 3 }] }));
  assert(a.t === "discard");
  assertEquals(tileType(a.tile), 10, "宣言できないリーチの値段で牌を選んだ");
  assertEquals(a.riichi, false);

  // The control: the same prices with a wait the dojo has no objection to.
  const clean = new Probe(noReads, stubCore());
  clean.stub = priced(rows, 5000, 0);
  const b = clean.decide(board({ hand, legal, waits: [10], ukeire: [{ type: 10, live: 2 }] }));
  assert(b.t === "discard");
  assertEquals(tileType(b.tile), 0, "禁じ手でないのに dama 線に落とした");
  assertEquals(b.riichi, true);
});

Deno.test("M15 review: pointsPerScore × augment.lambda ≠ 1 を積む前に断る (dylib 不要)", () => {
  // `dealinCostPts` IS `riskOf` in points — the equality the DP's whole defence
  // input rests on — and it holds only while λ = 1/pointsPerScore.
  const e = assertThrows(
    seatWith({ ev: { pointsPerScore: 8 } }),
    Error,
  ) as Error;
  assertStringIncludes(e.message, "pointsPerScore");
  assertStringIncludes(e.message, "lambda");
  const f = assertThrows(
    seatWith({ ev: {}, augment: { lambda: 0.5 } }),
    Error,
  ) as Error;
  assertStringIncludes(f.message, "lambda");
  // Moved TOGETHER, the pair is legal: it is the product that is the contract.
  noConflict({ ev: { pointsPerScore: 8 }, augment: { lambda: 0.125 } });
});

Deno.test("M15 review: 安全 の段は dealinCostPts でも riskOf と同じ値段", () => {
  // Both methods used to price 安全 differently the moment a vector priced that
  // rung at all: `riskOf` charged `w.danger["安全"]`, `dealinCostPts` charged 0.
  const danger = new Map<number, DangerAssessment>();
  danger.set(0, threat("安全"));
  danger.set(1, threat("危険度高"));
  const obs = board({ riichi: [false, true, false, false], danger });
  const ctx = ctxOf(obs);
  const pps = DEFAULT_EV.pointsPerScore;

  const p = new Probe(noReads, stubCore(), {
    danger: { "安全": 7, "危険度低": 30, "危険度中": 90, "危険度高": 200 },
  });
  p.arm(obs);
  for (const tile of obs.hand) {
    assertAlmostEquals(
      p.costPts(ctx, tile),
      p.scoreRisk(ctx, tile) * pps,
      1e-9,
      `型 ${tileType(tile)}`,
    );
  }
  assertEquals(p.costPts(ctx, tiles("1m")[0]), 7 * pps, "安全 の段が points 側で落ちている");
});

Deno.test({
  name: "M15 review: 槓のある手も EV 経路に入る (根は 14 − 3·面子数、槓は引かない)",
  ignore: SKIP,
  fn: () => {
    // A kan's fourth tile is paid for by the rinshan draw, so a post-暗槓
    // discard root holds 11 concealed tiles beside one meld — `14 − 3·1`, the
    // same as a pon's. The root test used to be `hand.length % 3 === 2`, which
    // admits this shape; it is now the exact equality, which is `mjev.cc`'s own
    // (`parseEval`). If either side ever subtracts the kan count, this test
    // fails LOUDLY — as a refusal from the DP, not as a seat that quietly went
    // back to the linear surrogate.
    const kanTiles = tiles("5555s");
    const kan: Meld = {
      kind: "ankan",
      who: 0,
      fromWho: 0,
      tiles: kanTiles,
      calledTile: kanTiles[0],
    };
    const melds: Observation["melds"] = [[kan], [], [], []];
    const hand = tiles("12345678m11p1s");
    assertEquals(hand.length, 14 - 3 * 1);
    const obs = board({
      hand,
      drawn: hand[hand.length - 1],
      melds,
      shanten: 1,
      legal: discards([hand[0], hand[8], hand[hand.length - 1]]),
    });

    const core = buildEv(mergeEv({}));
    try {
      const p = new Probe(noReads, core);
      const a = p.decide(obs);
      assertEquals(a.t, "discard");
      assertEquals(p.lastTrace?.units, "points", "槓のある手で EV 経路に入っていない");
      assert(p.evStats.calls > 0, "核が呼ばれていない");
      const e = p.value(obs);
      assert(Number.isFinite(e.bestPush) && Number.isFinite(e.bestFold), "槓根の値が有限でない");
      // …and the rest root of the same hand (the PASS side of a call) is the
      // mirror image: 13 − 3·1.
      const rest = board({ ...obs, hand: hand.slice(0, 10), drawn: null });
      assertEquals(rest.hand.length, 13 - 3 * 1);
    } finally {
      closeEv(core);
    }
  },
});

Deno.test({
  name: "M15 review: 暗槓の値段も打牌と同じ通行料を払う (危険な手が槓を買えない)",
  ignore: SKIP,
  fn: () => {
    // `bestPush` is a DISCARD root — every candidate has already paid `−costIn`
    // — while `mjev_eval_rest` is a 13-tile hold and pays nothing. The kan line
    // does not skip that discard, it defers it by one draw, so comparing the
    // two raw made every kan cheaper than every push by the price of a deal-in.
    //
    // THE CLAIM: with every tile priced identically dangerous, both lines pay
    // the same toll and the verdict must not move. Before the fix the loud
    // board took a kan at any margin below the toll — a kan bought with the
    // danger of the tiles it was not discarding.
    const { obs, legal } = ankanBoard();
    const loud = new Map<number, DangerAssessment>();
    for (let ty = 0; ty < 34; ty++) loud.set(ty, threat("危険度高"));
    const dangerous = board({ ...obs, danger: loud });
    // 危険度高 at 1,000,000 score units ⇒ a 4,000,000 point toll at
    // `pointsPerScore` 4: far above anything a hand is worth, so a leak of it
    // into the comparison cannot hide inside the engine's own numbers.
    const w: Partial<HeuristicWeights> = {
      danger: { "安全": 0, "危険度低": 30, "危険度中": 90, "危険度高": 1_000_000 },
    };

    const quietVerdicts: (string | null)[] = [];
    for (const callMargin of [-1e9, 0, 1e3, 1e5, 1e6, 1e9]) {
      const core = buildEv(mergeEv({ callMargin }));
      try {
        const quiet = new Probe(noReads, core, w).kanChoice(ctxOf(obs), legal);
        const threatened = new Probe(noReads, core, w).kanChoice(ctxOf(dangerous), legal);
        quietVerdicts.push(quiet?.t ?? null);
        assertEquals(
          threatened?.t ?? null,
          quiet?.t ?? null,
          `callMargin=${callMargin}: 危険度が槓の判断を動かした (通行料が片側にしか乗っていない)`,
        );
      } finally {
        closeEv(core);
      }
    }
    // Not vacuous: the sweep contains both verdicts, so "the two boards agree"
    // is a statement about the toll and not about a constant answer.
    assert(quietVerdicts.includes("ankan"), `静かな卓で一度も槓していない: ${quietVerdicts}`);
    assert(quietVerdicts.includes(null), `静かな卓で一度も断っていない: ${quietVerdicts}`);
  },
});

Deno.test("M15 review: 1枚に満たない生牌山は pool チャネルに載せない (核が拒否する)", () => {
  // 計算's `wallComposition` is `unseen × wallRemaining/unseenTotal`, so on the
  // last discards of every hand that gets there the vector is all zeros.
  // Handing that over as a REPLACEMENT posterior tells the DP the tiles come
  // from nowhere; `mjev.cc` refuses it (`Nroot < 1`) and `evEvalDiscard` turns
  // the refusal into a throw in the middle of a match. This was a real crash
  // (`headless(3, 8191, "kkkk", {ev:{}})`), not a hypothetical.
  const obs = board();
  const empty = new Probe(readsOf({ wallComposition: f34(0) }), stubCore());
  empty.arm(obs);
  assertEquals(empty.hidden(obs), null, "空の山が pool チャネルに載っている");

  // ONE tile left is the same refusal in disguise: 計算's shares are float32
  // fractions, and a single live tile sums to 0.99999998 — under the DP's
  // `Nroot >= 1`. The test sums it the way `parseEval` does, so the two sides
  // agree on the boundary rather than on a rounded idea of it.
  const one = f34(0);
  one[0] = 0.5;
  one[1] = 0.25;
  one[2] = 0.25 - 1e-8;
  const single = new Probe(readsOf({ wallComposition: one }), stubCore());
  single.arm(obs);
  assertEquals(single.hidden(obs), null, "1枚未満の山が pool に載っている");

  // A wall with real tiles left still reaches the channel, fractions and all.
  const part = f34(0);
  part[0] = 1.5;
  part[1] = 0.75;
  const some = new Probe(readsOf({ wallComposition: part }), stubCore());
  some.arm(obs);
  assertEquals(some.hidden(obs)?.pool?.[0], 1.5);
});
