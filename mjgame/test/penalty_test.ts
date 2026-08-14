// Penalty-rule fixtures: every rule gets a positive case and a negative case,
// and every documented exception clause gets its own negative case.

import { assert, assertEquals } from "@std/assert";
import type { Tile } from "mjrender/model.ts";
import type { Meld } from "mjrender/model.ts";
import type { WinOracle } from "../src/legal.ts";
import { ANY_WIN } from "../src/legal.ts";
import { runHook } from "../src/penalty/rules.ts";
import type { Hook, RuleCtx } from "../src/penalty/mod.ts";
import { DOJO_DEFAULT, JANKI } from "../src/rules.ts";
import { scorer } from "../src/score.ts";
import { sfc32 } from "../src/rng.ts";
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
  // Indicator -> dora is +1 within the suit, so point at the predecessor.
  const ind = type < 27
    ? (type % 9 === 0 ? type + 8 : type - 1)
    : type === 27
    ? 30
    : type === 31
    ? 33
    : type - 1;
  t.emit({ t: "dora", indicator: ind * 4 }, { e: "dora", indicator: ind * 4 });
}

function ctx(
  t: Table,
  seat: Seat,
  action: Action,
  drawn: Tile | null = null,
  oracle: WinOracle = ANY_WIN,
): RuleCtx {
  return { t, seat, action, drawn, cfg: t.cfg, dojo: DOJO_DEFAULT, oracle };
}

/** Apply a chi meld for `seat`, built from explicit ids so nothing collides. */
function chi(t: Table, seat: Seat, meldTiles: Tile[]): void {
  const meld: Meld = {
    kind: "chi",
    who: seat,
    fromWho: ((seat + 3) % 4) as Seat,
    tiles: [...meldTiles].sort((a, b) => a - b),
    calledTile: meldTiles[0],
  };
  t.emit({ t: "call", meld }, { e: "call", meld });
}

function fire(hook: Hook, c: RuleCtx): Violation[] {
  return runHook(hook, c);
}

function ids(rule: string, vs: Violation[]): boolean {
  return vs.some((v) => v.rule === rule);
}

function doDiscard(t: Table, seat: Seat, tile: Tile, riichi = false): Action {
  const a: Action = { t: "discard", tile, riichi, tsumogiri: false };
  t.emit(
    { t: "discard", who: seat, tile, tsumogiri: false, riichi },
    { e: "discard", who: seat, tile, tsumogiri: false, riichi },
  );
  if (riichi) t.riichi[seat] = true;
  return a;
}

// ---------------------------------------------------------------------------

Deno.test("第一打字牌切り: fires on an opening honor, not on a number tile", () => {
  const honor = tiles("東")[0];
  {
    const t = makeTable();
    setHand(t, 0, [...tiles("123456789m123p"), honor]);
    const a = doDiscard(t, 0, honor);
    assert(ids("first-honor", fire("post-discard", ctx(t, 0, a))));
  }
  {
    const t = makeTable();
    const num = tiles("9m")[0];
    setHand(t, 0, [...tiles("12345678m123p"), num, honor]);
    const a = doDiscard(t, 0, num);
    assert(!ids("first-honor", fire("post-discard", ctx(t, 0, a))));
  }
});

Deno.test("第一打字牌切り: exempt when it is a double-riichi declaration", () => {
  const t = makeTable();
  const honor = tiles("南")[0];
  setHand(t, 0, [...tiles("123456789m11122p"), honor]);
  t.doubleRiichi[0] = true;
  const a = doDiscard(t, 0, honor, true);
  assert(!ids("first-honor", fire("post-discard", ctx(t, 0, a))));
});

Deno.test("明槓: 大明槓 and 加槓 both fire, 暗槓 does not", () => {
  const t = makeTable();
  const called = tiles("5s")[0];
  assert(ids("minkan", fire("on-call", ctx(t, 1, { t: "daiminkan", called }))));
  assert(ids("minkan", fire("on-kan", ctx(t, 1, { t: "kakan", tile: called }))));
  setHand(t, 1, tiles("5555s123m456p12s"));
  assert(!ids("minkan", fire("on-kan", ctx(t, 1, { t: "ankan", type: 22 }))));
});

Deno.test("不聴時ドラ切り: fires when not tenpai, exempt at tenpai and for 赤5筒", () => {
  // A deliberately scattered hand, so the shape stays far from tenpai.
  const scattered = () => tiles("147m258p369s東南西北中");
  {
    const t = makeTable();
    makeDora(t, 0); // 1m
    setHand(t, 0, scattered());
    doDiscard(t, 0, tiles("4m")[0]); // burn the first-discard window
    setHand(t, 0, scattered());
    const a = doDiscard(t, 0, tiles("1m")[0]);
    assert(ids("noten-dora", fire("post-discard", ctx(t, 0, a))));
  }
  {
    // 赤5筒 is explicitly cuttable before tenpai when it is not indicator dora.
    const t = makeTable();
    const aka = 52 as Tile;
    const hand = () => [...tiles("147m369s東南西北白發中"), aka];
    setHand(t, 0, hand());
    doDiscard(t, 0, tiles("4m")[0]);
    setHand(t, 0, hand());
    const a = doDiscard(t, 0, aka);
    assert(!ids("noten-dora", fire("post-discard", ctx(t, 0, a))));
  }
});

Deno.test("即引っかけ立直: wait suji-trapped by the declaration tile", () => {
  const t = makeTable();
  // After cutting 1m: 123m 56m 123p 789p 99s — a clean 4m/7m ryanmen. The 1m we
  // just cut is the suji of 4m, which is exactly the trap the rule bans.
  setHand(t, 0, tiles("1123m56m123p789p99s"));
  const a = doDiscard(t, 0, tiles("1m")[0], true);
  const vs = fire("on-riichi", ctx(t, 0, a));
  const v = vs.find((x) => x.rule === "hikkake");
  assert(v !== undefined, `expected hikkake, got [${vs.map((x) => x.rule).join(",")}]`);
  assert(v.detail.includes("3"), `evidence should name the trapped wait: ${v.detail}`);
});

Deno.test("即引っかけ立直: a wait with no suji relation to the cut is clean", () => {
  const t = makeTable();
  // Same 4m/7m wait, but declared on 9s — no suji relation at all.
  setHand(t, 0, tiles("123m56m123p789p999s"));
  const a = doDiscard(t, 0, tiles("9s")[0], true);
  assert(!ids("hikkake", fire("on-riichi", ctx(t, 0, a))));
});

Deno.test("役満関連牌切り: 大三元 fires from 2 melds, others need 3", () => {
  const t = makeTable();
  const white = tiles("白白白")[0];
  const green = tiles("發發發")[0];
  // P1 pons 白 and 發 — two dragon triplets is already the 大三元 threshold.
  for (const base of [white, green]) {
    const m = {
      kind: "pon" as const,
      who: 1,
      fromWho: 2,
      tiles: [base, base + 1, base + 2],
      calledTile: base,
    };
    t.emit({ t: "call", meld: m }, { e: "call", meld: m });
  }
  const chun = tiles("中")[0];
  setHand(t, 0, [...tiles("19m19p19s東南西"), chun]);
  doDiscard(t, 0, tiles("2m")[0]);
  const a = doDiscard(t, 0, chun);
  assert(ids("yakuman-related", fire("post-discard", ctx(t, 0, a))));
});

Deno.test("持ち点8000点未満: judged at round end", () => {
  const lo = makeTable({ scores: [4000, 42000, 37000, 37000] });
  assert(ids("under-8000", fire("on-round-end", ctx(lo, 0, { t: "pass" }))));
  const ok = makeTable();
  assert(!ids("under-8000", fire("on-round-end", ctx(ok, 0, { t: "pass" }))));
});

// ---------------------------------------------------------------------------
// 後付け. Judged at the waiting hand, by the real scorer — so every fixture
// below is built against `scorer`, not the `ANY_WIN` placeholder (under which
// every wait is ronnable and the rule can never fire).
// ---------------------------------------------------------------------------

/**
 * The canonical yakuless open tenpai:
 *
 *   副露 チー123m ＋ 手牌 456m 678p 55s 78s   (待ち 6s/9s)
 *
 * 6s finishes 123m 456m 678p 678s 55s, 9s finishes …789s 55s. Neither scores:
 * the 1m in the meld kills 断幺九 (喰いタン is ON in JANKI), 456m kills 混全帯幺九,
 * three suits kill 混一色, the chi kills 対々和, and there is no 役牌 anywhere.
 *
 * Index 3 of the returned ids is a stray 1p: the tile the fixture discards to
 * arrive at the shape.
 */
function yakulessTenpai(t: Table): Tile[] {
  const all = tiles("123m1p456m678p55s78s");
  chi(t, 0, all.slice(0, 3));
  setHand(t, 0, all.slice(3));
  return all;
}

Deno.test("後付け: open tenpai where no wait scores at all", () => {
  const t = makeTable();
  const all = yakulessTenpai(t);
  const a = doDiscard(t, 0, all[3]); // the stray 1p
  const vs = fire("post-discard", ctx(t, 0, a, null, scorer));
  const v = vs.find((x) => x.rule === "atozuke");
  assert(v !== undefined, `expected atozuke, got [${vs.map((x) => x.rule).join(",")}]`);
  assertEquals(v.tier, "A");
  assertEquals(v.confidence, 1, "the win oracle answers exactly");
  assert(v.detail.includes("23") && v.detail.includes("26"), `evidence names 6s/9s: ${v.detail}`);
  assert(!ids("katagari", vs), "no wait scores, so this is not 片和了り");
});

Deno.test("後付け: a 混一色 build is not a violation (neither noten nor tenpai)", () => {
  // The false positive the old on-call rule produced: a 白 back pair inside an
  // obvious honitsu build read as バック, though every completion carries 混一色.
  {
    const t = makeTable();
    // チー123s ＋ 456s 789s 3s 白白 5m 6m — still 1向聴 after cutting the 6m.
    const all = tiles("123s6m456789s3s白白5m");
    chi(t, 0, all.slice(0, 3));
    setHand(t, 0, all.slice(3));
    const a = doDiscard(t, 0, all[3]); // the 6m
    assert(!ids("atozuke", fire("post-discard", ctx(t, 0, a, null, scorer))));
  }
  {
    const t = makeTable();
    // The same build, now tenpai: シャンポン 3s/白, and both waits carry 混一色.
    const all = tiles("123s5m456789s33s白白");
    chi(t, 0, all.slice(0, 3));
    setHand(t, 0, all.slice(3));
    const a = doDiscard(t, 0, all[3]); // the stray 5m
    assert(!ids("atozuke", fire("post-discard", ctx(t, 0, a, null, scorer))));
  }
});

Deno.test("後付け: a mixed wait belongs to 片和了り, not here", () => {
  const t = makeTable();
  // チー123m ＋ 456m 678p 22s 白白: シャンポン 2s/白. 白 scores (役牌), 2s does not.
  const all = tiles("123m1p456m678p22s白白");
  chi(t, 0, all.slice(0, 3));
  setHand(t, 0, all.slice(3));
  const a = doDiscard(t, 0, all[3]);
  const vs = fire("post-discard", ctx(t, 0, a, null, scorer));
  assert(ids("katagari", vs), "the split wait is 片和了り");
  assert(!ids("atozuke", vs), "and must not be charged twice");
});

Deno.test("後付け: charged once per round, not on every later discard", () => {
  const t = makeTable();
  const all = yakulessTenpai(t);
  const first = doDiscard(t, 0, all[3]); // the stray 1p
  const vs = fire("post-discard", ctx(t, 0, first, null, scorer));
  const v = vs.find((x) => x.rule === "atozuke");
  assert(v !== undefined, "expected the first discard to be charged");
  t.addViolation(v); // runHook does not write the ledger; the driver does

  // A later 巡, same hand, same yakuless tenpai (the drawn tile goes straight
  // back out, so the concealed shape is untouched).
  const second = doDiscard(t, 0, tiles("9p")[0], false);
  assert(!ids("atozuke", fire("post-discard", ctx(t, 0, second, null, scorer))));
});

Deno.test("後付け: exempt when the yakuless wait is 純カラ", () => {
  const t = makeTable();
  // チー123m ＋ 456m 678p 55s 79s: a lone 8s カンチャン, and all four 8s are gone.
  const all = tiles("123m1p456m678p55s79s");
  chi(t, 0, all.slice(0, 3));
  setHand(t, 0, all.slice(3));
  for (const eight of tiles("8888s")) doDiscard(t, 1, eight);
  const a = doDiscard(t, 0, all[3]);
  assert(!ids("atozuke", fire("post-discard", ctx(t, 0, a, null, scorer))));
});

Deno.test("後付け: a closed hand is exempt — it can cure the shape with 立直", () => {
  const t = makeTable();
  // The very same 14 tiles, none of them melded: 門前 so 立直 puts a yaku on
  // every wait, and the dojo's objection disappears.
  const all = tiles("123m1p456m678p55s78s");
  setHand(t, 0, all);
  const a = doDiscard(t, 0, all[3]);
  assert(!ids("atozuke", fire("post-discard", ctx(t, 0, a, null, scorer))));
});

Deno.test("長考: fires past the 3-second norm, escalates past 4", () => {
  const t = makeTable();
  const base = ctx(t, 0, { t: "discard", tile: 0, riichi: false, tsumogiri: true });
  assertEquals(fire("post-discard", { ...base, timing: { elapsedMs: 1500 } }).length, 0);
  const slow = fire("post-discard", { ...base, timing: { elapsedMs: 3500 } });
  assert(ids("chouko", slow));
  assertEquals(slow.find((v) => v.rule === "chouko")?.label, "長考");
  const verySlow = fire("post-discard", { ...base, timing: { elapsedMs: 6000 } });
  assertEquals(verySlow.find((v) => v.rule === "chouko")?.label, "大長考");
});

Deno.test("Tier B can be switched off wholesale (for RL training)", () => {
  const t = makeTable();
  const c = ctx(t, 0, { t: "discard", tile: 0, riichi: false, tsumogiri: true });
  const off = { ...c, dojo: { ...DOJO_DEFAULT, tierB: false }, timing: { elapsedMs: 9000 } };
  assertEquals(fire("post-discard", off).filter((v) => v.tier === "B").length, 0);
});

Deno.test("a rule that throws is contained, not fatal", () => {
  const t = makeTable();
  // A hand of the wrong size makes several analyses throw internally; the
  // runner must swallow it and keep the game alive.
  setHand(t, 0, tiles("1m"));
  const a: Action = { t: "discard", tile: tiles("1m")[0], riichi: false, tsumogiri: true };
  const vs = fire("post-discard", ctx(t, 0, a));
  for (const v of vs) {
    if (v.confidence === 0) assertEquals(v.points, 0, "a crashed rule must not score");
  }
});

Deno.test("立直後カン見送り: fires on a riichi tsumogiri that passed up a kan", () => {
  const t = makeTable();
  // 111m as a concealed triplet inside a tenpai hand; the 4th 1m is drawn and
  // thrown away rather than kanned. The kan does not touch the 456p78p wait.
  const hand = tiles("111m456p78p111s99s");
  const drawn = tiles("1111m")[3]; // the fourth copy, id 3
  setHand(t, 0, hand);
  t.riichi[0] = true;
  const a: Action = { t: "discard", tile: drawn, riichi: false, tsumogiri: true };
  t.emit(
    { t: "discard", who: 0, tile: drawn, tsumogiri: true, riichi: false },
    { e: "discard", who: 0, tile: drawn, tsumogiri: true, riichi: false },
  );
  assert(ids("riichi-kan-skip", fire("post-discard", ctx(t, 0, a, drawn))));
});

Deno.test("立直後カン見送り: the declaring discard is exempt", () => {
  // Regression: `t.riichi` is set before the post-discard hook runs, so the
  // declaring tedashi used to look like a riichi turn. Worse, reconstructing
  // the hand as `[...hand, drawn]` counts the still-held drawn tile twice and
  // invents a fourth copy — the rule fired on a kan that never existed.
  const t = makeTable();
  const hand = tiles("111m456p78p111s99s");
  const drawn = tiles("1111m")[3];
  setHand(t, 0, [...hand.slice(0, 12), drawn]);
  const cut = hand[12]; // some other tile: a tedashi, as a declaration must be
  const a: Action = { t: "discard", tile: cut, riichi: true, tsumogiri: false };
  t.emit(
    { t: "discard", who: 0, tile: cut, tsumogiri: false, riichi: true },
    { e: "discard", who: 0, tile: cut, tsumogiri: false, riichi: true },
  );
  t.riichi[0] = true;
  assert(!ids("riichi-kan-skip", fire("post-discard", ctx(t, 0, a, drawn))));
});

Deno.test("立直後カン見送り: a declaration that cuts the drawn 4th copy is exempt too", () => {
  // Regression: the exemption used to test only `!tsumogiri`, so a declaration
  // made by cutting the drawn tile still reached the rule — and the tile it cut
  // was the very 4th copy the rule looks for, ledgering a kan-skip on the turn
  // riichi was declared. Same fixture as the positive case, but `riichi: true`.
  const t = makeTable();
  const hand = tiles("111m456p78p111s99s");
  const drawn = tiles("1111m")[3];
  setHand(t, 0, hand);
  const a: Action = { t: "discard", tile: drawn, riichi: true, tsumogiri: true };
  t.emit(
    { t: "discard", who: 0, tile: drawn, tsumogiri: true, riichi: true },
    { e: "discard", who: 0, tile: drawn, tsumogiri: true, riichi: true },
  );
  t.riichi[0] = true; // as round.ts does, before the hook fires
  assert(!ids("riichi-kan-skip", fire("post-discard", ctx(t, 0, a, drawn))));
});
