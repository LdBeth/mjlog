// The baseline CPU. Every observation here is hand-built (the pattern comes
// from timer_test.ts) so each test states exactly one situation and varies
// nothing else — no table, no wall, no round driver.

import { assert, assertEquals } from "@std/assert";
import type { DangerAssessment, DangerLevel } from "mjrender/danger.ts";
import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import { tileType } from "mjrender/tiles.ts";
import { HeuristicPolicy } from "../src/ai/heuristic.ts";
import { mergeRiichi } from "../src/ai/riichi.ts";
import type { Observation } from "../src/observe.ts";
import type { ActionPreview } from "../src/penalty/preview.ts";
import { JANKI } from "../src/rules.ts";
import type { Action, Violation } from "../src/types.ts";
import { tiles } from "./helpers.ts";

/** One discard action per tile in hand, riichi never offered. */
function discardsOf(hand: Tile[], drawn: Tile | null): Action[] {
  return hand.map((tile) => ({
    t: "discard",
    tile,
    riichi: false,
    tsumogiri: drawn !== null && tile === drawn,
  }));
}

/**
 * Two 巡 already played, one honor cut per seat. The rivers must not be empty:
 * the policy prices a 第一打字牌切り at 4000, which would otherwise swamp every
 * other term whenever an honor is the discard under test.
 */
function openingRivers(): RiverEntry[][] {
  const pool = tiles("北北北北西西西西"); // one distinct copy per seat, twice
  return [0, 1, 2, 3].map((s) =>
    [pool[s], pool[4 + s]].map((tile, i): RiverEntry => ({
      tile,
      junme: i + 1,
      tsumogiri: false,
      riichiDeclare: false,
    }))
  );
}

/**
 * A complete, internally consistent Observation. `legal` defaults to "discard
 * anything"; pass it explicitly for claim situations.
 */
function baseObs(over: Partial<Observation> = {}): Observation {
  const hand = over.hand ?? tiles("123456789m1122p東");
  const drawn = "drawn" in over ? over.drawn! : hand[hand.length - 1];
  const obs: Observation = {
    seat: 0,
    kyoku: 0,
    honba: 0,
    kyotaku: 0,
    junme: 3,
    wallRemaining: 58,
    hand,
    drawn,
    melds: [[], [], [], []],
    rivers: openingRivers(),
    scores: [25000, 25000, 25000, 25000],
    riichi: [false, false, false, false],
    riichiJunme: [-1, -1, -1, -1],
    doraIndicators: tiles("9s"), // ⇒ dora is 1s, which no hand below holds
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
  if (!over.legal) obs.legal = discardsOf(obs.hand, obs.drawn);
  return obs;
}

/** A single-seat threat entry, shaped the way `assessDanger` returns them. */
function threat(
  level: DangerLevel,
  seat = 1,
  kind: "riichi" | "furo" = "riichi",
): DangerAssessment {
  return { level, seats: [seat], details: [{ seat, level, kind, notes: [] }] };
}

const TON = 27; // 東
const HAKU = 31; // 白

Deno.test("heuristic: a win is taken over any discard", () => {
  const hand = tiles("123456789m11122p"); // 123m 456m 789m 111p 22p
  const obs = baseObs({
    hand,
    drawn: hand[hand.length - 1],
    shanten: -1,
    legal: [...discardsOf(hand, hand[hand.length - 1]), { t: "tsumo" }],
  });
  const p = new HeuristicPolicy("cpu", 1);
  assertEquals(p.decide(obs).t, "tsumo");
});

Deno.test("heuristic: cuts the isolated honor out of a finished shape", () => {
  const obs = baseObs(); // 123456789m 1122p + lone 東
  const p = new HeuristicPolicy("cpu", 1);
  const a = p.decide(obs);
  assertEquals(a.t, "discard");
  assert(a.t === "discard");
  assertEquals(tileType(a.tile), TON, "the lone honor is the only free tile");
});

Deno.test("heuristic: folds a far hand against a riichi", () => {
  const hand = tiles("1346m2479p2466s東白");
  const danger = new Map<number, DangerAssessment>([[HAKU, threat("危険度高")]]);
  const obs = baseObs({
    hand,
    drawn: hand[hand.length - 1], // the 白 was just drawn
    shanten: 3,
    riichi: [false, true, false, false],
    riichiJunme: [-1, 2, -1, -1],
    danger,
  });
  const p = new HeuristicPolicy("cpu", 1);
  const a = p.decide(obs);
  assertEquals(a.t, "discard");
  assert(a.t === "discard");
  assert(tileType(a.tile) !== HAKU, "must not fire the 危険度高 tile while folding");
  assertEquals(tileType(a.tile), TON, "the genbutsu honor is the fold");
});

Deno.test("heuristic: pushes the dangerous tile when tenpai with value", () => {
  const hand = tiles("123456789m1122p東");
  const danger = new Map<number, DangerAssessment>([[TON, threat("危険度高")]]);
  const obs = baseObs({
    hand,
    shanten: 0,
    waits: [tileType(tiles("1p")[0]), tileType(tiles("2p")[0])],
    doraIndicators: tiles("9p"), // ⇒ dora 1p, of which the hand holds two
    doraCount: 2,
    riichi: [false, true, false, false],
    riichiJunme: [-1, 2, -1, -1],
    danger,
  });
  const p = new HeuristicPolicy("cpu", 1);
  const a = p.decide(obs);
  assertEquals(a.t, "discard");
  assert(a.t === "discard");
  assertEquals(tileType(a.tile), TON, "tenpai with two dora pushes through the threat");
});

// ---------------------------------------------------------------------------
// 生牌の役牌 — the `liveYakuhai` surcharge (2026-08-28)
// ---------------------------------------------------------------------------

/** One opponent pon, so the 副露 clause of the surcharge is satisfied. */
function oppPon(): Meld[][] {
  const ts = tiles("333s");
  return [[], [{ kind: "pon", who: 1, fromWho: 2, tiles: ts, calledTile: ts[0] }], [], []];
}

/**
 * 中盤の生牌の役牌. No threat is assessed (nobody has declared, one meld is
 * below the assessor's activation), so the whole difference between the two
 * seats below is `liveYakuhai`: the default 0 keeps the old game, and a live
 * weight buys the honor back out of the discard.
 */
function liveHakuObs() {
  const hand = tiles("123456789m1122p白");
  return baseObs({
    hand,
    drawn: hand[hand.length - 1],
    junme: 6,
    melds: oppPon(),
  });
}

Deno.test("heuristic: liveYakuhai 0 still cuts the 生牌の白 (the default game)", () => {
  const p = new HeuristicPolicy("cpu", 1, { weights: { liveYakuhai: 0 } });
  const a = p.decide(liveHakuObs());
  assert(a.t === "discard");
  assertEquals(tileType(a.tile), HAKU);
});

Deno.test("heuristic: liveYakuhai holds a 生牌の役牌 against an open hand", () => {
  const p = new HeuristicPolicy("cpu", 1, { weights: { liveYakuhai: 3000 } });
  const a = p.decide(liveHakuObs());
  assert(a.t === "discard");
  assert(tileType(a.tile) !== HAKU, "誰も切っていない白を副露相手に放つのは高い");
});

Deno.test("heuristic: liveYakuhai stays silent when a clause is missing", () => {
  const p = new HeuristicPolicy("cpu", 1, { weights: { liveYakuhai: 3000 } });
  // 序盤 (巡目 < 6): the honor has not been HELD yet, it is just late to be cut.
  {
    const obs = { ...liveHakuObs(), junme: 5 };
    const a = p.decide(obs);
    assert(a.t === "discard");
    assertEquals(tileType(a.tile), HAKU, "序盤は課金しない");
  }
  // 門前だけの卓: no call has been made, so a 役牌 wait is not on the table.
  {
    const obs = { ...liveHakuObs(), melds: [[], [], [], []] };
    const a = p.decide(obs);
    assert(a.t === "discard");
    assertEquals(tileType(a.tile), HAKU, "副露がなければ課金しない");
  }
  // 自風の東 while 東場 would be a value honor, but 南 is nobody's here: seat 0
  // sits 東, so 南/西/北 belong to the other three — 南 IS someone's seat wind.
  // Use our OWN seat wind instead, in a 南場, where it is worth nothing to the
  // three seats that could ron us.
  {
    const hand = tiles("123456789m1122p南");
    const obs = baseObs({
      hand,
      drawn: hand[hand.length - 1],
      junme: 6,
      melds: oppPon(),
      seatWind: 28, // 南家
      roundWind: 27, // 東場 ⇒ 南 is only OUR wind
    });
    const a = p.decide(obs);
    assert(a.t === "discard");
    assertEquals(tileType(a.tile), 28, "自風のみの牌は他家にとって役牌ではない");
  }
});

Deno.test("heuristic: liveYakuhai fires only where the assessor is not looking", () => {
  const p = new HeuristicPolicy("cpu", 1, { weights: { liveYakuhai: 3000 } });
  // An assessed tile is already priced by the ladder (a live 役牌 under a
  // riichi reads 危険度高 there); stacking the surcharge on top made the arena
  // replay swap a 中-rated honor for a 高-rated number tile INTO the riichi.
  const base = liveHakuObs();
  const assessed: DangerAssessment = {
    level: "危険度中",
    seats: [1],
    details: [{ seat: 1, level: "危険度中", kind: "furo", openMeldCount: 1, notes: ["役牌"] }],
  };
  const danger = new Map(base.danger);
  danger.set(HAKU, assessed);
  const a = p.decide({ ...base, danger });
  assert(a.t === "discard");
  assertEquals(tileType(a.tile), HAKU, "査定済みの牌には重ねて課金しない");
});

Deno.test("heuristic: liveYakuhai counts open melds only — an 暗槓 is not a call", () => {
  const p = new HeuristicPolicy("cpu", 1, { weights: { liveYakuhai: 3000 } });
  const ts = tiles("3333s");
  const ankanOnly: Meld[][] = [
    [],
    [{ kind: "ankan", who: 1, fromWho: 1, tiles: ts, calledTile: ts[0] }],
    [],
    [],
  ];
  const a = p.decide({ ...liveHakuObs(), melds: ankanOnly });
  assert(a.t === "discard");
  assertEquals(tileType(a.tile), HAKU, "暗槓だけの卓は門前扱い");
});

Deno.test("heuristic: never calls 明槓", () => {
  const hand = tiles("111m2345678p234s");
  const called = tiles("1111m")[3];
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 1,
    legal: [{ t: "daiminkan", called }, { t: "pass" }],
  });
  const p = new HeuristicPolicy("cpu", 1);
  assertEquals(p.decide(obs).t, "pass");
});

Deno.test("heuristic: refuses a chi with no route to any yaku", () => {
  // 111m 77m 45p 23s 56s 89s — the chi improves the shape but the prospect
  // screen finds no yaku left to aim at: no 役牌 melded or paired, three 1m
  // put 断幺九 out of reach, the concealed part spans m+s against a p meld so
  // 混一色 is gone, and the chi itself rules out 対々和.
  const hand = tiles("11177m45p235689s");
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 2,
    legal: [
      { t: "pass" },
      { t: "chi", tiles: [hand[5], hand[6]], called: tiles("6p")[0] },
    ],
  });
  const p = new HeuristicPolicy("cpu", 1);
  assertEquals(p.decide(obs).t, "pass");
});

Deno.test("heuristic: calls toward a 混一色 build", () => {
  // 12s 456s 789s 33s 白白 ＋ 一枚の 5m. The old confirmed-yaku filter refused
  // this chi (a 白 pair with nothing "confirmed" read as バック); the screen
  // lets it through, and so does the ledger — the crime would be finishing on
  // a yakuless wait, which this build never does.
  const all = tiles("12s456s789s33s白白5m3s");
  const hand = all.slice(0, 13);
  const called = all[13]; // a third 3s, distinct from the pair in hand
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 1, // 456s + 789s + 33s/白白/12s の三ブロック
    legal: [
      { t: "pass" },
      { t: "chi", tiles: [hand[0], hand[1]], called },
    ],
  });
  const p = new HeuristicPolicy("cpu", 1);
  assertEquals(p.decide(obs).t, "chi");
});

Deno.test("heuristic: avoids the discard that fixes a yakuless open wait", () => {
  // An open hand (one chi), two discards that both reach tenpai — one of them
  // on a wait nothing scores on, which is a Tier A 後付け the moment it lands.
  const meldTiles = tiles("123m");
  const meld: Meld = {
    kind: "chi",
    who: 0,
    fromWho: 3,
    tiles: meldTiles,
    calledTile: meldTiles[0],
  };
  const hand = tiles("456m678p55s78s5m");
  const [dirty, clean] = [hand[hand.length - 1], hand[0]];
  const obs = baseObs({
    hand,
    drawn: null,
    melds: [[meld], [], [], []],
    discardInfo: new Map([
      [dirty, { shanten: 0, katagari: false, yakuless: true }],
      [clean, { shanten: 0, katagari: false, yakuless: false }],
    ]),
    legal: [
      { t: "discard", tile: dirty, riichi: false, tsumogiri: false },
      { t: "discard", tile: clean, riichi: false, tsumogiri: false },
    ],
  });
  const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
  assertEquals(chosen.t, "discard");
  if (chosen.t === "discard") assertEquals(chosen.tile, clean);
});

Deno.test("heuristic: a pair-rich hand may still pon toward 対々和", () => {
  // 22m 55m 88p 22s 66s 999p — after the 2m pon the rest still holds four
  // pairs behind the 999p set, which is what an actual 対々和 build looks
  // like. The 2026-08-27 tightening must not refuse THIS; it exists to refuse
  // the pair-poor pon (see planner_test's plan-discipline case).
  const hand = tiles("22m55m88p22s66s999p");
  const called = tiles("222m")[2];
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 2,
    legal: [
      { t: "pass" },
      { t: "pon", tiles: [hand[0], hand[1]], called },
    ],
  });
  const p = new HeuristicPolicy("cpu", 1);
  assertEquals(p.decide(obs).t, "pon");
});

Deno.test("heuristic: takes a yakuhai pon that gains a shanten", () => {
  const hand = tiles("11199m22p4578s白白");
  const called = tiles("白白白")[2];
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 2,
    legal: [
      { t: "pass" },
      { t: "pon", tiles: [hand[11], hand[12]], called },
    ],
  });
  const p = new HeuristicPolicy("cpu", 1);
  assertEquals(p.decide(obs).t, "pon");
});

Deno.test("heuristic: the same seed and observation give the same action", () => {
  const obs = baseObs();
  const a = new HeuristicPolicy("a", 7);
  const b = new HeuristicPolicy("b", 7);
  assertEquals(a.decide(obs), b.decide(obs), "two policies, one seed");
  assertEquals(a.decide(obs), a.decide(obs), "and no hidden per-call state");
});

Deno.test("heuristic: dojo:false lifts the 禁じ手 filters", () => {
  // 明槓: deliberately loose. `chooseKan` only ever *returns* an ankan, so the
  // dojo flag cannot change this one either way — all that is asserted is that
  // lifting the filter still yields a legal action.
  const kanHand = tiles("111m2345678p234s");
  const kanObs = baseObs({
    hand: kanHand,
    drawn: null,
    shanten: 1,
    legal: [{ t: "daiminkan", called: tiles("1111m")[3] }, { t: "pass" }],
  });
  const kan = new HeuristicPolicy("cpu", 1, { dojo: false }).decide(kanObs);
  assert(kanObs.legal.includes(kan), "whatever it picks must be a legal action");

  // The yaku-prospect screen, where the flag does bite: the very chi refused
  // above is now taken.
  const hand = tiles("11177m45p235689s");
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 2,
    legal: [
      { t: "pass" },
      { t: "chi", tiles: [hand[5], hand[6]], called: tiles("6p")[0] },
    ],
  });
  assertEquals(new HeuristicPolicy("cpu", 1, { dojo: false }).decide(obs).t, "chi");
});

// --------------------------------------------------------------------------
// 片和了り: prevention first, abandonment only when there is nothing to shift to
// --------------------------------------------------------------------------

Deno.test("heuristic: prefers the clean tenpai over the 片和了り one", () => {
  const hand = tiles("123456789m1122p東");
  const [a, b] = [hand[9], hand[11]]; // one tile from each pair of 1p/2p
  const obs = baseObs({
    hand,
    drawn: hand[hand.length - 1],
    // Both discards reach tenpai; only one of them splits the wait.
    discardInfo: new Map([
      [hand[hand.length - 1], { shanten: 0, katagari: true, yakuless: false }],
      [a, { shanten: 0, katagari: false, yakuless: false }],
      [b, { shanten: 0, katagari: true, yakuless: false }],
    ]),
  });
  const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
  assertEquals(chosen.t, "discard");
  if (chosen.t === "discard") assertEquals(chosen.tile, a);
});

Deno.test("heuristic: gives the hand up when every tenpai is 片和了り and it cannot riichi", () => {
  const hand = tiles("123456789m1122p東");
  const drawn = hand[hand.length - 1];
  // An open hand: no riichi is on offer, so the split wait has no cure.
  const discardInfo = new Map(
    hand.map((t) =>
      [t, { shanten: t === drawn ? 1 : 0, katagari: t !== drawn, yakuless: false }] as const
    ),
  );
  const obs = baseObs({ hand, drawn, discardInfo });
  const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
  assertEquals(chosen.t, "discard");
  // The one tile that breaks tenpai is taken over any of the split-wait shapes.
  if (chosen.t === "discard") assertEquals(chosen.tile, drawn);
});

Deno.test("heuristic: riichi cures 片和了り rather than abandoning the hand", () => {
  const hand = tiles("123456789m1122p東");
  const drawn = hand[hand.length - 1];
  const keep = hand[9];
  const obs = baseObs({
    hand,
    drawn,
    waits: [tileType(tiles("1p")[0])],
    ukeire: [{ type: tileType(tiles("1p")[0]), live: 3 }],
    discardInfo: new Map([[keep, { shanten: 0, katagari: true, yakuless: false }]]),
    legal: [
      { t: "discard", tile: keep, riichi: false, tsumogiri: false },
      { t: "discard", tile: keep, riichi: true, tsumogiri: false },
    ],
  });
  const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
  assertEquals(chosen.t, "discard");
  if (chosen.t === "discard") assertEquals(chosen.riichi, true);
});

// The shipped doctrine boundary (see CLAUDE.md 最終形リーチ): immediate
// declaration unless `holdShape` says the hand is cheap-and-narrow; two held
// turns release it.
const DOCTRINE = mergeRiichi({ bias: 0.1, holdShape: -1, tenpaiHeld: 0.5 });

function riichiPick(p: HeuristicPolicy, over: Partial<Observation>): boolean {
  const hand = over.hand!;
  const drawn = hand[hand.length - 1];
  const obs = baseObs({
    drawn,
    discardInfo: new Map([[drawn, { shanten: 0, katagari: false, yakuless: false }]]),
    legal: [
      { t: "discard", tile: drawn, riichi: false, tsumogiri: true },
      { t: "discard", tile: drawn, riichi: true, tsumogiri: false },
    ],
    ...over,
  });
  const chosen = p.decide(obs);
  assertEquals(chosen.t, "discard");
  return chosen.t === "discard" ? chosen.riichi : false;
}

Deno.test("heuristic: 最終形 doctrine — a cheap hand holds, then declares after two held turns", () => {
  // 123456789m1122p + 東: cutting the 東 leaves the 1p/2p シャンポン — 4 live,
  // but the hand is riichi(+平和)のみ with no dora, which is exactly the
  // cheap-and-unremarkable shape the doctrine holds. Two held turns later the
  // same observation declares. The head must be IN the vector for any of this
  // — the default policy still declares immediately (M12 discipline).
  const hand = tiles("123456789m1122p東");
  const p = new HeuristicPolicy("cpu", 1, { riichi: DOCTRINE });
  const w = [tileType(tiles("1p")[0]), tileType(tiles("2p")[0])];
  const shared: Partial<Observation> = {
    hand,
    waits: w,
    ukeire: [{ type: w[0], live: 2 }, { type: w[1], live: 2 }],
  };
  assertEquals(riichiPick(p, { ...shared, junme: 4 }), false, "安手は即リーチしない");
  assertEquals(riichiPick(p, { ...shared, junme: 5 }), false, "1巡待ってもまだ宣言しない");
  assertEquals(riichiPick(p, { ...shared, junme: 6 }), true, "2巡待てば宣言してよい");
});

Deno.test("heuristic: 最終形 doctrine — value or a sanctioned 単騎 declares immediately", () => {
  // Same シャンポン but the 9m indicator makes the held 1m a dora: the value
  // model prices it above the riichi-only baseline, and >2 acceptance plus a
  // real hand is an immediate declaration.
  const hand = tiles("123456789m1122p東");
  const w = [tileType(tiles("1p")[0]), tileType(tiles("2p")[0])];
  const dora = new HeuristicPolicy("cpu", 1, { riichi: DOCTRINE });
  assertEquals(
    riichiPick(dora, {
      hand,
      junme: 4,
      doraIndicators: tiles("9m"),
      waits: w,
      ukeire: [{ type: w[0], live: 2 }, { type: w[1], live: 2 }],
    }),
    true,
    "ドラ1の実のある手は即リーチ",
  );
  // 七対子の単騎 is sanctioned regardless of its price.
  const chiitoi = new HeuristicPolicy("cpu", 1, { riichi: DOCTRINE });
  const hand7 = tiles("1133m5577p2244s9s東");
  const w9s = tileType(tiles("9s")[0]);
  assertEquals(
    riichiPick(chiitoi, {
      hand: hand7,
      junme: 4,
      waits: [w9s],
      ukeire: [{ type: w9s, live: 3 }],
    }),
    true,
    "七対子単騎は即リーチしてよい",
  );
});

Deno.test("heuristic: a short stack folds in the South round, pushes in the East", () => {
  const hand = tiles("123456789m1122p東");
  const drawn = hand[hand.length - 1];
  const shared = {
    hand,
    drawn,
    shanten: 1,
    riichi: [false, true, false, false],
    danger: new Map([[TON, threat("危険度高")]]),
  };
  const pick = (score: number, roundWind = 28) => {
    const obs = baseObs({ ...shared, roundWind, scores: [score, 25000, 25000, 25000] });
    const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
    return chosen.t === "discard" ? chosen.tile : -1;
  };
  // 南場, 1-shanten with a dangerous 東: comfortable stack pushes it, short one
  // does not — the 8000-line buffer is live in the closing stretch.
  assertEquals(pick(25000), drawn);
  assert(pick(9000) !== drawn, "a 9000-point stack should not fire the 危険度高 tile");
  // 東場, same short stack: the rule is judged at 終局 (2026-08-27 ruling), and
  // with a whole hanchan left to recover in the buffer stays out of the gate.
  assertEquals(pick(9000, 27), drawn, "東場の 9000点 is not yet a buffer problem");
});

Deno.test("heuristic: tedashi stops once ドラ切り has been ponned off us", () => {
  const hand = tiles("123456789m1122p東");
  const drawn = hand[hand.length - 1]; // 東, the tile a free hand would cut
  const pick = (tsumogiriLock: boolean) => {
    const obs = baseObs({ hand, drawn, tsumogiriLock });
    const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
    return chosen.t === "discard" ? chosen.tile : -1;
  };
  // Unlocked, the lone honor goes. Locked, every tedashi is a violation, so the
  // drawn tile goes back out instead — which here happens to be the same tile.
  assertEquals(pick(false), drawn);
  assertEquals(pick(true), drawn, "a locked hand must tsumogiri");

  // With a drawn tile that efficiency would rather keep, the lock still wins.
  const keeper = tiles("123456789m1122p1p")[15];
  const locked = baseObs({
    hand: [...hand.slice(0, 13), keeper],
    drawn: keeper,
    tsumogiriLock: true,
  });
  const chosen = new HeuristicPolicy("cpu", 1).decide(locked);
  assertEquals(chosen.t, "discard");
  if (chosen.t === "discard") assertEquals(chosen.tile, keeper);
});

// --------------------------------------------------------------------------
// The compliance filter. These Observations carry a `preview` — the referee,
// asked hypothetically — which is what every real driver supplies and no
// hand-built Observation above does. That asymmetry is the point: without a
// preview the policy keeps the older priced behaviour every test above pins,
// and with one it stops CHOOSING the charged action at all.
// --------------------------------------------------------------------------

/** One Tier A entry, enough for the policy to see the action is charged. */
function charge(rule: string): Violation[] {
  return [{
    rule,
    label: rule,
    seat: 0,
    kyoku: 0,
    junme: 0,
    points: 3,
    tier: "A",
    confidence: 1,
    detail: "stub",
  }];
}

/**
 * A referee stub. Each hook says whether it would charge; anything unset stays
 * silent, so a test states exactly the one veto it is about.
 */
function stubPreview(o: {
  discard?: (a: Extract<Action, { t: "discard" }>) => string | null;
  call?: (a: Action) => string | null;
  kan?: (a: Action) => string | null;
  skipKan?: boolean;
}): ActionPreview {
  const vs = (id: string | null | undefined) => (id ? charge(id) : []);
  return {
    discard: (a) => vs(o.discard?.(a)),
    call: (a) => vs(o.call?.(a)),
    kan: (a) => vs(o.kan?.(a)),
    skipKan: () => (o.skipKan ? charge("riichi-kan-skip") : []),
  };
}

Deno.test("filter: a discard the referee would charge is not chosen", () => {
  const hand = tiles("123456789m1122p東");
  const free = hand[hand.length - 1]; // the lone 東 — what an unfiltered policy cuts
  const obs = baseObs({
    hand,
    preview: stubPreview({ discard: (a) => (a.tile === free ? "first-honor" : null) }),
  });
  const a = new HeuristicPolicy("cpu", 1).decide(obs);
  assertEquals(a.t, "discard");
  assert(a.t === "discard");
  assert(a.tile !== free, "the charged tile must be out of the choice set entirely");
});

Deno.test("filter: when every discard is charged the prices decide again", () => {
  // The fallthrough. Something has to be thrown, so the filter stands aside and
  // `dojoCost` ranks the damage — which here is the same tile the unfiltered
  // policy picks, because nothing else has changed.
  const hand = tiles("123456789m1122p東");
  const free = hand[hand.length - 1];
  const obs = baseObs({ hand, preview: stubPreview({ discard: () => "noten-dora" }) });
  const a = new HeuristicPolicy("cpu", 1).decide(obs);
  assertEquals(a.t, "discard");
  assert(a.t === "discard");
  assertEquals(a.tile, free);
});

Deno.test("filter: a riichi declaration the referee would charge is declined", () => {
  const hand = tiles("123456789m1122p東");
  const keep = hand[9];
  const build = (preview?: ActionPreview) =>
    baseObs({
      hand,
      drawn: hand[hand.length - 1],
      waits: [tileType(tiles("1p")[0])],
      ukeire: [{ type: tileType(tiles("1p")[0]), live: 3 }],
      discardInfo: new Map([[keep, { shanten: 0, katagari: true, yakuless: false }]]),
      legal: [
        { t: "discard", tile: keep, riichi: false, tsumogiri: false },
        { t: "discard", tile: keep, riichi: true, tsumogiri: false },
      ],
      preview,
    });
  // Control: with no referee to ask, the split wait is cured by declaring.
  const cure = new HeuristicPolicy("cpu", 1).decide(build());
  assert(cure.t === "discard" && cure.riichi, "control: 片和了り is cured by riichi");

  // The declaration is 地獄単騎 — a 禁じ手 outranks the cure.
  const banned = stubPreview({ discard: (a) => (a.riichi ? "jigoku-tanki" : null) });
  const chosen = new HeuristicPolicy("cpu", 1).decide(build(banned));
  assert(chosen.t === "discard" && !chosen.riichi, "a charged declaration is not made");
});

Deno.test("filter: a call the referee would charge is not taken", () => {
  const hand = tiles("11199m22p4578s白白");
  const called = tiles("白白白")[2];
  const legal: Action[] = [
    { t: "pass" },
    { t: "pon", tiles: [hand[11], hand[12]], called },
  ];
  const obs = baseObs({
    hand,
    drawn: null,
    shanten: 2,
    legal,
    preview: stubPreview({ call: () => "hadaka-tanki" }),
  });
  assertEquals(new HeuristicPolicy("cpu", 1).decide(obs).t, "pass");
});

Deno.test("filter: 立直後カン見送り forces the kan the policy would have skipped", () => {
  // `chooseKan` declines this one on its own terms (the concealed four are not
  // all in hand, so it cannot prove the wait survives). Declining is the
  // violation here, so the referee's verdict on the OMISSION overrides.
  const hand = tiles("111m456p78p111s99s");
  const drawn = hand[0];
  const legal: Action[] = [
    { t: "discard", tile: drawn, riichi: false, tsumogiri: true },
    { t: "ankan", type: 0 },
  ];
  const shared = { hand, drawn, shanten: 0, riichi: [true, false, false, false], legal };
  assertEquals(
    new HeuristicPolicy("cpu", 1).decide(baseObs(shared)).t,
    "discard",
    "control: with nobody asking, the kan is passed up",
  );
  const obs = baseObs({ ...shared, preview: stubPreview({ skipKan: true }) });
  assertEquals(new HeuristicPolicy("cpu", 1).decide(obs).t, "ankan");
});

Deno.test("filter: it is off with dojo:false, prices and all", () => {
  const hand = tiles("123456789m1122p東");
  const free = hand[hand.length - 1];
  const obs = baseObs({
    hand,
    preview: stubPreview({ discard: (a) => (a.tile === free ? "first-honor" : null) }),
  });
  const a = new HeuristicPolicy("cpu", 1, { dojo: false }).decide(obs);
  assert(a.t === "discard" && a.tile === free, "an unleashed policy ignores the referee");
});

// --------------------------------------------------------------------------
// 片和了り / 後付け are vetoed too, though no preview can see them: they are
// charged at WIN time (declining a win is 見逃し, so there is no action to
// veto there) while the only prevention is the discard that builds the wait.
// Left as prices they lose — a shanten step is 1000 and the katagari price is
// 1500, so a two-step gap already buys the violation, and the C7 planner's
// planKeep malus (5000) buys it outright.
// --------------------------------------------------------------------------

/** All-but-one tile flagged, with the flagged one the only route to tenpai. */
function splitWaitObs(flag: "katagari" | "yakuless", over: Partial<Observation> = {}) {
  const hand = tiles("123456789m1122p東");
  const split = hand[9]; // a 1p: cutting it reaches the split tenpai
  const discardInfo = new Map(
    hand.map((t) =>
      [t, {
        shanten: t === split ? 0 : 2,
        katagari: flag === "katagari" && t === split,
        yakuless: flag === "yakuless" && t === split,
      }] as const
    ),
  );
  return { split, obs: baseObs({ hand, discardInfo, preview: stubPreview({}), ...over }) };
}

Deno.test("filter: a 片和了り discard is vetoed, not merely outbid", () => {
  const { split, obs } = splitWaitObs("katagari");
  // Control: the price alone loses. Tenpai vs 2向聴 is worth 2000 and 片和了り
  // costs 1500, so an unrefereed policy takes the split wait.
  const priced = new HeuristicPolicy("cpu", 1).decide(baseObs({ ...obs, preview: undefined }));
  assert(priced.t === "discard" && priced.tile === split, "control: the price is outbid");

  const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
  assert(chosen.t === "discard" && chosen.tile !== split, "the split wait is out of the set");
});

Deno.test("filter: a 後付け discard is vetoed for an open hand", () => {
  const meld: Meld[] = [{
    kind: "pon",
    who: 0,
    fromWho: 1,
    tiles: tiles("白白白"),
    calledTile: tiles("白白白")[0],
  }];
  const open = { melds: [meld, [], [], []] as Meld[][] };
  const { split, obs } = splitWaitObs("yakuless", open);
  const chosen = new HeuristicPolicy("cpu", 1).decide(obs);
  assert(chosen.t === "discard" && chosen.tile !== split, "a yakuless open tenpai is refused");

  // 門前 is exempt: the same shape can still be cured by declaring riichi, so
  // the rule cannot fire and the veto must not either.
  const closed = splitWaitObs("yakuless");
  const kept = new HeuristicPolicy("cpu", 1).decide(closed.obs);
  assert(kept.t === "discard" && kept.tile === closed.split, "門前 keeps the tenpai");
});

Deno.test("filter: when every tenpai is 片和了り the prices decide again", () => {
  // The fallthrough contract, shared with the previewable rules: something has
  // to be thrown, so the veto stands aside and `dojoCost` ranks the damage —
  // which is exactly what the unrefereed policy already does.
  const hand = tiles("123456789m1122p東");
  const drawn = hand[hand.length - 1];
  const discardInfo = new Map(
    hand.map((t) => [t, { shanten: 0, katagari: true, yakuless: false }] as const),
  );
  const build = (preview?: ActionPreview) => baseObs({ hand, drawn, discardInfo, preview });
  const a = new HeuristicPolicy("cpu", 1).decide(build(stubPreview({})));
  const b = new HeuristicPolicy("cpu", 1).decide(build());
  assert(a.t === "discard" && b.t === "discard");
  assertEquals(a.tile, b.tile, "an empty candidate set falls back to the priced ranking");
});

Deno.test("filter: the 片和了り veto lifts when riichi is on offer", () => {
  const hand = tiles("123456789m1122p東");
  const keep = hand[9]; // a 1p — the split tenpai
  const alt = hand[hand.length - 1]; // the 東 — 1向聴, but clean
  const build = (canRiichi: boolean) =>
    baseObs({
      hand,
      drawn: alt,
      waits: [tileType(tiles("1p")[0])],
      ukeire: [{ type: tileType(tiles("1p")[0]), live: 3 }],
      discardInfo: new Map([
        [keep, { shanten: 0, katagari: true, yakuless: false }],
        [alt, { shanten: 1, katagari: false, yakuless: false }],
      ]),
      legal: [
        { t: "discard", tile: keep, riichi: false, tsumogiri: false },
        ...(canRiichi
          ? [{ t: "discard", tile: keep, riichi: true, tsumogiri: false } as Action]
          : []),
        { t: "discard", tile: alt, riichi: false, tsumogiri: true },
      ],
      preview: stubPreview({}),
    });
  // 立直 is itself a yaku: it makes every wait scoring, so the shape stops being
  // split and there is nothing to veto — the same clause `dojoCost` prices on.
  const cure = new HeuristicPolicy("cpu", 1).decide(build(true));
  assert(cure.t === "discard" && cure.tile === keep && cure.riichi, "riichi cures it instead");

  // With no declaration available the exemption lapses and the veto bites.
  const folded = new HeuristicPolicy("cpu", 1).decide(build(false));
  assert(folded.t === "discard" && folded.tile === alt, "no cure ⇒ the split wait is refused");
});
