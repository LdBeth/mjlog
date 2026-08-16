// The preview must agree with the ledger.
//
// `penalty/preview.ts` exists so a policy can decline a 禁じ手 instead of paying
// for it, and it is only worth anything if its answer is the ledger's answer.
// So every test here builds the SAME fixture twice: once to ask the preview what
// would happen, once to actually take the action through `dojo.ts`'s hooks (the
// real wiring `round.ts` uses) and read `Table.ledger`. The two rule-id multisets
// must match — no false positives, no false negatives, both directions.
//
// The fixture idioms (makeTable / setHand / makeDora / doDiscard) are lifted
// from penalty_test.ts on purpose: the rules are pinned there, so a preview
// agreeing with the ledger on those shapes is agreeing with tested behaviour.

import { assert, assertEquals } from "@std/assert";
import type { Meld, Tile } from "mjrender/model.ts";
import { dojoHooks } from "../src/dojo.ts";
import { buildMeld } from "../src/legal.ts";
import { previewCall, previewDiscard, previewKan, previewSkipKan } from "../src/penalty/preview.ts";
import { DOJO_DEFAULT, JANKI } from "../src/rules.ts";
import { sfc32 } from "../src/rng.ts";
import { scorer } from "../src/score.ts";
import { Table } from "../src/table.ts";
import type { Action, Seat, Violation } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { Wall } from "../src/wall.ts";
import { tiles } from "./helpers.ts";

function makeTable(o: { kyoku?: number; scores?: number[]; seed?: number } = {}): Table {
  const rng = sfc32(o.seed ?? 1);
  return new Table(
    {
      kyoku: o.kyoku ?? 0,
      honba: 0,
      kyotaku: 0,
      dealer: 0,
      scores: o.scores ?? [30000, 30000, 30000, 30000],
      wall: Wall.shuffled(rng),
      dice: [0, 0],
    },
    JANKI,
    SEATS.map((seat) => ({ seat, name: `P${seat}` })),
  );
}

function setHand(t: Table, seat: Seat, ids: Tile[]): void {
  t.hands[seat].length = 0;
  t.hands[seat].push(...ids);
}

/** Make `type` a dora by revealing an indicator that points at it. */
function makeDora(t: Table, type: number): void {
  const ind = type < 27
    ? (type % 9 === 0 ? type + 8 : type - 1)
    : type === 27
    ? 30
    : type === 31
    ? 33
    : type - 1;
  t.emit({ t: "dora", indicator: ind * 4 }, { e: "dora", indicator: ind * 4 });
}

/** A discard that has already happened — scenery, not the action under test. */
function doDiscard(t: Table, seat: Seat, tile: Tile, riichi = false): void {
  t.emit(
    { t: "discard", who: seat, tile, tsumogiri: false, riichi },
    { e: "discard", who: seat, tile, tsumogiri: false, riichi },
  );
  if (riichi) t.riichi[seat] = true;
}

/**
 * A riichi already accepted, both halves of it: `Table.riichi` is the engine's
 * flag and `BoardState.riichiActive` is what `round.ts` re-reads it from after
 * every discard, so a fixture that sets only the first would have its riichi
 * quietly cleared by the next committed discard.
 */
function declareRiichi(t: Table, seat: Seat): void {
  t.riichi[seat] = true;
  t.board.riichiActive[seat] = true;
}

/** A pon already on the table for `seat`, from three explicit ids. */
function pon(t: Table, seat: Seat, ids: Tile[]): void {
  const meld: Meld = {
    kind: "pon",
    who: seat,
    fromWho: ((seat + 1) % 4) as Seat,
    tiles: [...ids].sort((a, b) => a - b),
    calledTile: ids[0],
  };
  t.emit({ t: "call", meld }, { e: "call", meld });
}

// ---------------------------------------------------------------------------
// the agreement harness
// ---------------------------------------------------------------------------

interface Fixture {
  t: Table;
  seat: Seat;
  action: Action;
  drawn?: Tile | null;
  /** The discarder a call takes from. */
  from?: Seat;
  /**
   * A discard forced by the action under test: after a call the seat MUST throw
   * something, with no drawn tile to fall back on, which is where 裸単騎 and
   * ドラ切り後の手出し actually land. The preview folds that turn into its answer
   * (see `previewCall`), so the ledger side has to play it out.
   */
  then?: (t: Table) => Action;
}

const ruleIds = (vs: readonly Violation[]): string[] => vs.map((v) => v.rule).sort();

/** What the preview says, before anything is committed. */
function askPreview(f: Fixture): string[] {
  const { t, seat, action } = f;
  const drawn = f.drawn ?? null;
  switch (action.t) {
    case "discard":
      return ruleIds(previewDiscard(t, seat, action, DOJO_DEFAULT, scorer, drawn));
    case "pon":
    case "chi":
    case "daiminkan":
      return ruleIds(previewCall(t, seat, action, DOJO_DEFAULT, scorer));
    case "ankan":
    case "kakan":
      return ruleIds(previewKan(t, seat, action, drawn, DOJO_DEFAULT, scorer));
    default:
      return [];
  }
}

/** Take the action for real, exactly as `round.ts` does, and file the result. */
function commit(f: Fixture): string[] {
  const { t, seat, action } = f;
  const hooks = dojoHooks({ dojo: DOJO_DEFAULT, oracle: scorer });
  const play = (a: Action, drawn: Tile | null) => {
    if (a.t === "discard") {
      if (a.riichi) {
        t.emit({ t: "reach", who: seat, step: 1 }, { e: "riichi", who: seat, step: 1 });
        if (t.firstTurnIntact) t.doubleRiichi[seat] = true;
      }
      t.emit(
        { t: "discard", who: seat, tile: a.tile, tsumogiri: a.tsumogiri, riichi: a.riichi },
        { e: "discard", who: seat, tile: a.tile, tsumogiri: a.tsumogiri, riichi: a.riichi },
      );
      t.riichi[seat] = t.board.riichiActive[seat];
      t.turnIndex++;
    } else if (a.t !== "pass" && a.t !== "ron" && a.t !== "tsumo") {
      const from = a.t === "ankan" || a.t === "kakan" ? seat : (f.from ?? seat);
      const meld = buildMeld(seat, from, a, t);
      t.emit({ t: "call", meld }, { e: "call", meld });
      t.firstTurnIntact = false;
      if (a.t !== "pon" && a.t !== "chi") t.kanTotal++;
    }
    hooks.onAction(t, seat, a, drawn);
  };
  play(action, f.drawn ?? null);
  if (f.then) play(f.then(t), null);
  return ruleIds(t.ledger);
}

/**
 * Build the fixture twice — the preview reads the table, and the commit changes
 * it — and assert the two verdicts are the same set of rules.
 */
function agrees(build: () => Fixture, expected: string[]): void {
  const previewed = askPreview(build());
  const ledgered = commit(build());
  assertEquals(
    previewed,
    ledgered,
    `preview [${previewed.join(",")}] vs ledger [${ledgered.join(",")}]`,
  );
  for (const rule of expected) {
    assert(previewed.includes(rule), `expected ${rule}, got [${previewed.join(",")}]`);
  }
  if (expected.length === 0) assertEquals(previewed, [], "a clean action must preview clean");
}

// ---------------------------------------------------------------------------
// post-discard rules
// ---------------------------------------------------------------------------

Deno.test("preview 第一打字牌切り", () => {
  agrees(() => {
    const t = makeTable();
    const honor = tiles("東")[0];
    setHand(t, 0, [...tiles("123456789m123p"), honor]);
    return { t, seat: 0, action: { t: "discard", tile: honor, riichi: false, tsumogiri: true } };
  }, ["first-honor"]);
});

Deno.test("preview 不聴時ドラ切り", () => {
  agrees(() => {
    const t = makeTable();
    makeDora(t, 0); // 1m
    const hand = () => [...tiles("47m258p369s東南西北中"), ...tiles("1m")];
    setHand(t, 0, hand());
    doDiscard(t, 0, tiles("4m")[0]); // close the 第一打 window
    setHand(t, 0, hand());
    return {
      t,
      seat: 0,
      action: { t: "discard", tile: tiles("1m")[0], riichi: false, tsumogiri: true },
    };
  }, ["noten-dora"]);
});

Deno.test("preview ドラ切り後の手出し", () => {
  agrees(() => {
    const t = makeTable();
    const hand = [...tiles("123456789m123p"), tiles("東")[0]];
    setHand(t, 0, hand);
    doDiscard(t, 0, tiles("9m")[0]);
    setHand(t, 0, hand);
    t.tsumogiriLock[0] = true; // as the ドラ切り rule's arming half would
    // A tedashi: the drawn tile is something else entirely.
    return {
      t,
      seat: 0,
      action: { t: "discard", tile: tiles("東")[0], riichi: false, tsumogiri: false },
      drawn: tiles("1p")[0],
    };
  }, ["dora-pon-lock"]);
});

Deno.test("preview 役満関連牌切り", () => {
  agrees(() => {
    const t = makeTable();
    for (const base of [tiles("白白白")[0], tiles("發發發")[0]]) {
      pon(t, 1, [base, base + 1, base + 2] as Tile[]);
    }
    const chun = tiles("中")[0];
    setHand(t, 0, [...tiles("19m19p19s東南西2m"), chun]);
    doDiscard(t, 0, tiles("2m")[0]);
    return { t, seat: 0, action: { t: "discard", tile: chun, riichi: false, tsumogiri: true } };
  }, ["yakuman-related"]);
});

Deno.test("preview 裸単騎", () => {
  agrees(() => {
    const t = makeTable();
    for (const spec of ["111m", "222m", "333m", "444p"]) {
      const ids = tiles(spec);
      pon(t, 0, ids);
    }
    const hand = tiles("99s");
    setHand(t, 0, hand);
    doDiscard(t, 1, tiles("1s")[0]); // scenery, so no river is empty
    return { t, seat: 0, action: { t: "discard", tile: hand[0], riichi: false, tsumogiri: true } };
  }, ["hadaka-tanki"]);
});

Deno.test("preview 立直後カン見送り, both as an action and as an omission", () => {
  const build = (): Fixture => {
    const t = makeTable();
    const hand = tiles("111m456p78p111s99s");
    const drawn = tiles("1111m")[3];
    setHand(t, 0, [...hand, drawn]);
    declareRiichi(t, 0);
    t.firstTurnIntact = false;
    return {
      t,
      seat: 0,
      action: { t: "discard", tile: drawn, riichi: false, tsumogiri: true },
      drawn,
    };
  };
  agrees(build, ["riichi-kan-skip"]);

  // The omission side asks the same question of the same predicate: "what does
  // declining the kan cost?" — which is precisely the tsumogiri above.
  const f = build();
  assertEquals(
    ruleIds(previewSkipKan(f.t, 0, f.drawn!, DOJO_DEFAULT, scorer)),
    ["riichi-kan-skip"],
  );
  // ...and it must stay silent when there is no kan to pass up.
  const g = makeTable();
  setHand(g, 0, tiles("123m456p78p111s99s9p"));
  declareRiichi(g, 0);
  assertEquals(previewSkipKan(g, 0, tiles("9p")[0], DOJO_DEFAULT, scorer), []);
});

Deno.test("preview: an ordinary discard is clean on both sides", () => {
  agrees(() => {
    const t = makeTable();
    const hand = [...tiles("123456789m123p"), tiles("東")[0]];
    setHand(t, 0, hand);
    doDiscard(t, 0, tiles("9m")[0]);
    setHand(t, 0, hand);
    return {
      t,
      seat: 0,
      action: { t: "discard", tile: tiles("東")[0], riichi: false, tsumogiri: true },
    };
  }, []);
});

// ---------------------------------------------------------------------------
// on-riichi rules — the declaration and its discard, judged together
// ---------------------------------------------------------------------------

/** A table where a riichi is a mid-game declaration, not a ダブリー. */
function riichiTable(): Table {
  const t = makeTable();
  doDiscard(t, 1, tiles("1s")[0]);
  t.firstTurnIntact = false;
  return t;
}

Deno.test("preview 即引っかけ立直", () => {
  agrees(() => {
    const t = riichiTable();
    setHand(t, 0, tiles("1123m56m123p789p99s"));
    return {
      t,
      seat: 0,
      action: { t: "discard", tile: tiles("1m")[0], riichi: true, tsumogiri: false },
    };
  }, ["hikkake"]);
});

Deno.test("preview 地獄単騎立直", () => {
  agrees(() => {
    const t = riichiTable();
    const ton = tiles("東東東東");
    setHand(t, 1, [ton[1], ton[2]]);
    doDiscard(t, 1, ton[1]);
    doDiscard(t, 1, ton[2]); // two 東 gone: the tanki is a graveyard
    setHand(t, 0, [...tiles("123456789m123p"), ton[0], tiles("9s")[0]]);
    return {
      t,
      seat: 0,
      action: { t: "discard", tile: tiles("9s")[0], riichi: true, tsumogiri: true },
    };
  }, ["jigoku-tanki"]);
});

Deno.test("preview 一手変わり四暗刻での立直", () => {
  agrees(() => {
    const t = riichiTable();
    setHand(t, 0, [...tiles("111m222m333m456p9s"), tiles("1p")[0]]);
    return {
      t,
      seat: 0,
      action: { t: "discard", tile: tiles("1p")[0], riichi: true, tsumogiri: true },
    };
  }, ["suuankou-riichi"]);
});

Deno.test("preview 役満模様への立直", () => {
  agrees(() => {
    const t = riichiTable();
    for (const spec of ["東東東", "南南南"]) pon(t, 1, tiles(spec));
    setHand(t, 0, [...tiles("123456789m123p"), tiles("9s")[0]]);
    return {
      t,
      seat: 0,
      action: { t: "discard", tile: tiles("9s")[0], riichi: true, tsumogiri: true },
    };
  }, ["yakuman-threat-riichi"]);
});

Deno.test("preview: a plain riichi is clean on both sides", () => {
  agrees(() => {
    const t = riichiTable();
    setHand(t, 0, [...tiles("123456789m123p"), tiles("9s")[0]]);
    return {
      t,
      seat: 0,
      action: { t: "discard", tile: tiles("9s")[0], riichi: true, tsumogiri: true },
    };
  }, []);
});

// ---------------------------------------------------------------------------
// calls and kans
// ---------------------------------------------------------------------------

Deno.test("preview 明槓: 大明槓", () => {
  agrees(() => {
    const t = makeTable();
    const s5 = tiles("5555s");
    setHand(t, 0, [...tiles("123m456m789m12p"), s5[0], s5[1], s5[2]]);
    setHand(t, 1, [s5[3]]);
    doDiscard(t, 1, s5[3]);
    return { t, seat: 0, action: { t: "daiminkan", called: s5[3] }, from: 1 };
  }, ["minkan"]);
});

Deno.test("preview 明槓: 加槓", () => {
  agrees(() => {
    const t = makeTable();
    const s5 = tiles("5555s");
    pon(t, 0, [s5[0], s5[1], s5[2]]);
    setHand(t, 0, [...tiles("123m456m789m12p"), s5[3]]);
    return { t, seat: 0, action: { t: "kakan", tile: s5[3] } };
  }, ["minkan"]);
});

Deno.test("preview 暗槓条件違反: a wait-shifting ankan is charged on both sides", () => {
  // `previewKan` has to lay the meld before it asks, because `round.ts` does:
  // the `on-kan` hook always sees the kan already on the table. This fixture
  // pins that the two agree on a kan that IS a violation — 11123m456m789m99p
  // waits 1m/4m/7m and 9p, and kanning the 1m kills the 9p half.
  agrees(() => {
    const t = makeTable();
    const m1 = tiles("1111m");
    const drawn = m1[3];
    setHand(t, 0, [...m1.slice(0, 3), ...tiles("23m456m789m99p"), drawn]);
    declareRiichi(t, 0);
    t.firstTurnIntact = false;
    return { t, seat: 0, action: { t: "ankan", type: 0 }, drawn };
  }, ["ankan-form"]);
});

Deno.test("preview 暗槓条件違反: the wait-preserving control previews clean", () => {
  // The other half of the same contract, and the one the riichi 4th-copy
  // dilemma turns on: the 3p6p9p wait lives in 456p78p and the 111m kan does
  // not touch it, so nothing is charged and `mandatoryKan` can take the kan
  // instead of eating 立直後カン見送り for declining it.
  agrees(() => {
    const t = makeTable();
    const m1 = tiles("1111m");
    const drawn = m1[3];
    setHand(t, 0, [...m1.slice(0, 3), ...tiles("456p78p111s99s"), drawn]);
    declareRiichi(t, 0);
    t.firstTurnIntact = false;
    return { t, seat: 0, action: { t: "ankan", type: 0 }, drawn };
  }, []);
});

Deno.test("preview: a pon that forces a 裸単騎 discard is charged at the CALL", () => {
  // The lookahead. Nothing is wrong with the pon itself — the violation lands on
  // the discard the caller is then forced to make, with no drawn tile to hide
  // behind, which is why the call is the last point at which it is avoidable.
  agrees(() => {
    const t = makeTable();
    for (const spec of ["111m", "222m", "333m"]) pon(t, 0, tiles(spec));
    const p4 = tiles("4444p");
    setHand(t, 0, [p4[0], p4[1], tiles("9s")[0], tiles("9s")[1]]);
    setHand(t, 1, [p4[2]]);
    doDiscard(t, 1, p4[2]);
    return {
      t,
      seat: 0,
      action: { t: "pon", tiles: [p4[0], p4[1]], called: p4[2] },
      from: 1,
      then: (tb) => ({
        t: "discard",
        tile: tb.hands[0][0],
        riichi: false,
        tsumogiri: false,
      }),
    };
  }, ["hadaka-tanki"]);
});

Deno.test("preview: an ordinary pon is clean on both sides", () => {
  agrees(() => {
    const t = makeTable();
    const w = tiles("白白白");
    setHand(t, 0, [...tiles("123m456m78m12p"), w[0], w[1]]);
    setHand(t, 1, [w[2]]);
    doDiscard(t, 1, w[2]);
    return {
      t,
      seat: 0,
      action: { t: "pon", tiles: [w[0], w[1]], called: w[2] },
      from: 1,
      then: (tb) => ({ t: "discard", tile: tb.hands[0][0], riichi: false, tsumogiri: false }),
    };
  }, []);
});

// ---------------------------------------------------------------------------
// the preview must not leave a trace
// ---------------------------------------------------------------------------

Deno.test("preview: asking the question changes nothing", () => {
  const t = makeTable();
  const hand = [...tiles("123456789m123p"), tiles("東")[0]];
  setHand(t, 0, hand);
  const w = tiles("白白白");
  setHand(t, 1, [w[2]]);
  doDiscard(t, 1, w[2]);
  const snapshot = {
    hand: [...t.hands[0]],
    melds: t.melds[0].length,
    river: t.board.rivers[1].length,
    called: t.board.rivers[1][0].calledBy,
    riichi: [...t.riichi],
    lock: [...t.tsumogiriLock],
    ledger: t.ledger.length,
  };
  previewDiscard(
    t,
    0,
    { t: "discard", tile: hand[0], riichi: true, tsumogiri: false },
    DOJO_DEFAULT,
    scorer,
  );
  previewCall(t, 0, { t: "pon", tiles: [hand[0], hand[1]], called: w[2] }, DOJO_DEFAULT, scorer);
  previewKan(t, 0, { t: "kakan", tile: hand[0] }, null, DOJO_DEFAULT, scorer);
  assertEquals([...t.hands[0]], snapshot.hand);
  assertEquals(t.melds[0].length, snapshot.melds);
  assertEquals(t.board.rivers[1].length, snapshot.river);
  assertEquals(t.board.rivers[1][0].calledBy, snapshot.called);
  assertEquals([...t.riichi], snapshot.riichi);
  assertEquals([...t.tsumogiriLock], snapshot.lock);
  assertEquals(t.ledger.length, snapshot.ledger, "a hypothetical never files anything");
});
