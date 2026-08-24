// The dojo veto the learned seat plays under: `compliantActions` narrows
// `obs.legal` to what the referee would let pass, and `NeuralPolicy.decide`
// picks — and records its mask — from that support alone.
//
// The referee here is a STUB, not a real `ActionPreview` over a live Table:
// `penalty_preview_test.ts` already pins what each rule says about a real
// board, and repeating that machinery would make these tests statements about
// the rules rather than about the filter. What the filter needs is only "this
// action is charged, that one is not", so the stub answers exactly that, per
// action BY REFERENCE — which is also how the same-slot case (two discards of
// one tile type) can be posed at all.

import { assert, assertEquals } from "@std/assert";
import type { Meld, Tile } from "mjrender/model.ts";
import type { ActionPreview } from "../src/penalty/preview.ts";
import type { DiscardInfo, Observation } from "../src/observe.ts";
import type { Action, Violation } from "../src/types.ts";
import { compliantActions } from "../src/rl/policy.ts";
import { JANKI } from "../src/rules.ts";
import { tiles } from "./helpers.ts";

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

const HAND = tiles("1234m56789m1234p");
const INDICATOR = tiles("9s");

function discard(tile: Tile, over: Partial<Extract<Action, { t: "discard" }>> = {}): Action {
  return { t: "discard", tile, riichi: false, tsumogiri: false, ...over };
}

/** One made-up ledger entry — only `points` is ever read (by `violationPoints`). */
function violation(points: number, rule = "test-rule"): Violation {
  return {
    rule,
    label: "テスト",
    seat: 0,
    kyoku: 1,
    junme: 6,
    points,
    tier: "A",
    confidence: 1,
    detail: "",
  };
}

/**
 * A referee that charges exactly the actions handed to it, identified by
 * reference. `skip` is what declining a kan costs (empty = nothing).
 */
function stubPreview(charged: Iterable<Action>, skip: Violation[] = []): ActionPreview {
  const set = new Set<Action>(charged);
  const judge = (a: Action): Violation[] => (set.has(a) ? [violation(10)] : []);
  return {
    discard: (a) => judge(a),
    call: (a) => judge(a),
    kan: (a) => judge(a),
    skipKan: () => skip,
  };
}

function baseObs(over: Partial<Observation> = {}): Observation {
  const pon: Meld = {
    kind: "pon",
    who: 2,
    fromWho: 1,
    tiles: tiles("白白白"),
    calledTile: tiles("白白白")[2],
  };
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
    discardInfo: new Map<Tile, DiscardInfo>(),
    tsumogiriLock: false,
    danger: new Map(),
    violations: [0, 0, 0, 0],
    legal: HAND.slice(0, 4).map((t) => discard(t)),
    ...over,
  };
}

// ---------------------------------------------------------------------------
// 1. no referee, nothing to choose between
// ---------------------------------------------------------------------------

Deno.test("compliance: 審判が居なければ合法手そのものを返す", () => {
  // A driver that wired no DojoConfig leaves `preview` undefined, and every
  // hand-built Observation in the other tests is in exactly that position.
  const obs = baseObs();
  assert(obs.preview === undefined, "the fixture has no preview");
  assertEquals(compliantActions(obs), obs.legal);
  assert(compliantActions(obs) === obs.legal, "the very same array, not a copy");
});

Deno.test("compliance: 選択肢が一つなら審判に訊かない", () => {
  // One legal action cannot be filtered into anything else, so the referee —
  // whose every answer costs a table mutation — is never asked. The stub would
  // charge this discard if it were.
  const only = discard(HAND[0]);
  let asked = 0;
  const pv: ActionPreview = {
    discard: (_a) => {
      asked++;
      return [violation(10)];
    },
    call: () => [],
    kan: () => [],
    skipKan: () => [],
  };
  const obs = baseObs({ legal: [only], preview: pv });
  assertEquals(compliantActions(obs), [only]);
  assertEquals(asked, 0, "a single-action decision must not run a preview");
});

// ---------------------------------------------------------------------------
// 2. the ordinary filter
// ---------------------------------------------------------------------------

Deno.test("compliance: 咎められる打牌だけが支持集合から落ちる", () => {
  const obs = baseObs();
  const [a0, a1, a2, a3] = obs.legal;
  const filtered = compliantActions(baseObs({
    legal: obs.legal,
    preview: stubPreview([a1, a3]),
  }));
  assertEquals(filtered, [a0, a2]);
  // By identity: the game master compares the returned action by reference.
  assert(filtered[0] === a0 && filtered[1] === a2);
});

Deno.test("compliance: 同じ牌種の二手でも咎められない方だけが残る", () => {
  // Two copies of 1m: the drawn one (tsumogiri) and one already in hand
  // (tedashi). They share action slot 0, so the net cannot tell them apart —
  // only `resolve` picks between them, and it must be handed the filtered list
  // or the charged variant walks back in through the same slot.
  const pair = tiles("11m");
  const tsumogiri = discard(pair[0], { tsumogiri: true });
  const tedashi = discard(pair[1]);
  const obs = baseObs({
    hand: [...HAND.slice(1), ...pair],
    drawn: pair[0],
    legal: [tsumogiri, tedashi],
    preview: stubPreview([tedashi]),
  });
  const filtered = compliantActions(obs);
  assertEquals(filtered.length, 1);
  assert(filtered[0] === tsumogiri, "the clean variant of the shared slot survives");
});

Deno.test("compliance: 片和了りは立直宣言でだけ通り、後付けは鳴き手で落ちる", () => {
  // Neither rule can be previewed — both are charged at WIN time — so the
  // filter reads them off `discardInfo`, exactly where `compliantDiscards`
  // reads them. 片和了り: declaring is itself the cure, so the riichi variant
  // of the same tile stays. 後付け: a closed hand can still declare, an open
  // one is stuck, so the veto is on the open hand only.
  const split = HAND[0], yakuless = HAND[1], clean = HAND[2];
  const info = new Map<Tile, DiscardInfo>([
    [split, { shanten: 0, katagari: true, yakuless: false }],
    [yakuless, { shanten: 0, katagari: false, yakuless: true }],
  ]);
  const plain = discard(split);
  const declare = discard(split, { riichi: true });
  const legal = [plain, declare, discard(yakuless), discard(clean)];
  const open: Meld = {
    kind: "pon",
    who: 0,
    fromWho: 1,
    tiles: tiles("發發發"),
    calledTile: tiles("發發發")[2],
  };

  const asOpen = compliantActions(baseObs({
    legal,
    discardInfo: info,
    melds: [[open], [], [], []],
    preview: stubPreview([]),
  }));
  assertEquals(asOpen, [declare, legal[3]], "片和了りの素切りも後付けも落ちる");

  // The same hand closed: 後付け is curable by declaring, so its plain discard
  // is not vetoed here (only 片和了り's is).
  const asClosed = compliantActions(baseObs({
    legal,
    discardInfo: info,
    preview: stubPreview([]),
  }));
  assertEquals(asClosed, [declare, legal[2], legal[3]]);
});

// ---------------------------------------------------------------------------
// 3. 立直後カン見送り — the rule that fires on the omission
// ---------------------------------------------------------------------------

Deno.test("compliance: 立直中の見送りが咎められるなら暗槓一手だけを返す", () => {
  const drawn = HAND[0];
  const kan: Action = { t: "ankan", type: 0 };
  const tsumogiri = discard(drawn, { tsumogiri: true });
  const obs = baseObs({
    drawn,
    riichi: [true, false, false, false],
    legal: [tsumogiri, kan],
    // Declining costs 10; the kan itself is clean, so accepting wins.
    preview: stubPreview([], [violation(10)]),
  });
  const filtered = compliantActions(obs);
  assertEquals(filtered.length, 1, "a single-element support, chosen through the normal path");
  assert(filtered[0] === kan);
});

Deno.test("compliance: 見送りの方が安ければ通常の絞り込みに戻る", () => {
  // Both options are charged and declining is the cheaper one (`pickLesserEvil`
  // gives the tie to declining too). The forced-kan branch then declines to
  // force anything, the ordinary filter drops both charged actions, and the
  // empty-result fallback hands the full legal list back.
  const drawn = HAND[0];
  const kan: Action = { t: "ankan", type: 0 };
  const tsumogiri = discard(drawn, { tsumogiri: true });
  const legal = [tsumogiri, kan];
  const obs = baseObs({
    drawn,
    riichi: [true, false, false, false],
    legal,
    preview: {
      discard: () => [violation(10)],
      call: () => [],
      kan: () => [violation(50)],
      skipKan: () => [violation(10)],
    },
  });
  assert(compliantActions(obs) === legal, "everything charged ⇒ the full legal list");
});

// ---------------------------------------------------------------------------
// 4. the seat must always be able to act
// ---------------------------------------------------------------------------

Deno.test("compliance: 全ての手が咎められるなら合法手全体に戻す", () => {
  // A mask must never be empty. When every option is a 禁じ手 the ledger simply
  // gets its entry — which is what the heuristic's fallthrough pricing does at
  // the same fork.
  const obs = baseObs();
  const all = compliantActions(baseObs({
    legal: obs.legal,
    preview: stubPreview(obs.legal),
  }));
  assertEquals(all, obs.legal);
});

Deno.test("compliance: 和了・見逃しは常に支持集合に残る", () => {
  // Nothing previewable charges for taking a win or for standing pat, and the
  // stub charges everything it is handed — these two are never handed to it.
  const tsumo: Action = { t: "tsumo" };
  const pass: Action = { t: "pass" };
  const bad = discard(HAND[0]);
  const obs = baseObs({ legal: [tsumo, pass, bad], preview: stubPreview([bad]) });
  assertEquals(compliantActions(obs), [tsumo, pass]);
});
