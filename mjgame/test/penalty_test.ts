// Penalty-rule fixtures: every rule gets a positive case and a negative case,
// and every documented exception clause gets its own negative case.

import { assert, assertEquals } from "@std/assert";
import type { Tile } from "mjrender/model.ts";
import type { Meld } from "mjrender/model.ts";
import type { WinOracle } from "../src/legal.ts";
import { ANY_WIN } from "../src/legal.ts";
import { buildMeld } from "../src/legal.ts";
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

/**
 * Declare an ankan the way `round.ts` does — meld on the table first, hook
 * after — and report whether 暗槓条件違反 charged for it. `spec` is the 14-tile
 * hand including all four copies of `type`; the fourth copy is the draw.
 */
function ankanCharged(spec: string, type: number, riichi: boolean): boolean {
  const t = makeTable();
  const hand = tiles(spec);
  setHand(t, 0, hand);
  const drawn = hand.filter((id) => Math.floor(id / 4) === type)[3];
  if (riichi) t.riichi[0] = true;
  const action: Action = { t: "ankan", type };
  const meld = buildMeld(0, 0, action, t);
  t.emit({ t: "call", meld }, { e: "call", meld });
  return ids("ankan-form", fire("on-kan", ctx(t, 0, action, drawn)));
}

Deno.test("暗槓条件違反: a wait-preserving ankan is clean, in riichi and out", () => {
  // 111m is a plain concealed triplet; the 3p6p9p wait lives entirely in
  // 456p78p and does not notice the kan. Regression: the hook fires AFTER the
  // meld is on the table, and the predicate used to add the kan a second time —
  // judging 10 tiles against two melds' worth of slots — so it answered
  // "the wait changed" for every ankan ever declared.
  assert(!ankanCharged("1111m456p78p111s99s", 0, true));
  assert(!ankanCharged("1111m456p78p111s99s", 0, false));
});

Deno.test("暗槓条件違反: an ankan that shifts the wait is charged, in riichi and out", () => {
  // 11123m456m789m99p waits 1m/4m/7m (111m + 23m) AND 9p (11m pair + 123m,
  // shanpon-style on the 99p). Kanning the 1m destroys the second reading, so
  // the 9p half of the wait vanishes — テンパイが変わるカン.
  assert(ankanCharged("111123m456m789m99p", 0, true));
  assert(ankanCharged("111123m456m789m99p", 0, false));
});

/**
 * Cut a dora 1m out of a hand whose remaining 13 tiles have a known shanten,
 * and report whether 不聴時ドラ切り charged for it. The first discard is burnt
 * on an unrelated tile so the opening-honor window is closed either way.
 */
function cutDoraFrom(spec: string, o: { kyoku?: number } = {}): boolean {
  const t = makeTable({ kyoku: o.kyoku ?? 0 });
  makeDora(t, 0); // 1m
  const hand = () => [...tiles(spec), ...tiles("1m")];
  setHand(t, 0, hand());
  doDiscard(t, 0, tiles(spec)[0]);
  setHand(t, 0, hand());
  const a = doDiscard(t, 0, tiles("1m")[0]);
  return ids("noten-dora", fire("post-discard", ctx(t, 0, a)));
}

Deno.test("不聴時ドラ切り: fires from 3向聴 out, exempt at 2向聴以内", () => {
  // Each spec is the 13-tile shape LEFT BEHIND once the dora 1m is cut, labelled
  // with the shanten it holds. The line is drawn between 3向聴 and 2向聴.
  const shapes: [number, string][] = [
    [6, "47m258p369s東南西北中"],
    [3, "234m78m45p12p9s5s東南"],
    [2, "234m456m78m12p9s5s東"],
    [1, "234m456m78m12p1p1p9s"],
    [0, "234m456m789m123p1p"],
  ];
  for (const [sh, spec] of shapes) {
    assertEquals(cutDoraFrom(spec), sh >= 3, `向聴${sh} (${spec})`);
  }
});

Deno.test("不聴時ドラ切り: exempt for 赤5筒, spent honor dora, and オーラス", () => {
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
  {
    // 例外: an honor dora already twice on the table is spent. The cut itself is
    // the first copy, so one earlier 中 in another river reaches the threshold —
    // with none, the same 6向聴 hand is charged.
    const chun = () => tiles("中中");
    const cutChun = (prior: boolean): boolean => {
      const t = makeTable();
      makeDora(t, 33); // 中
      if (prior) {
        setHand(t, 1, [...tiles("147m258p369s東南西北"), chun()[1]]);
        doDiscard(t, 1, chun()[1]);
      }
      const hand = () => [...tiles("147m258p369s東南西北"), chun()[0]];
      setHand(t, 0, hand());
      doDiscard(t, 0, tiles("4m")[0]);
      setHand(t, 0, hand());
      const a = doDiscard(t, 0, chun()[0]);
      return ids("noten-dora", fire("post-discard", ctx(t, 0, a)));
    };
    assert(cutChun(false), "a live honor dora is still charged");
    assert(!cutChun(true), "a 中 already once in the rivers is spent");
  }
  {
    // 例外: オーラス conditions can justify it — same hand, last kyoku.
    assert(cutDoraFrom("47m258p369s東南西北中"), "control: 東1局 charges");
    assert(!cutDoraFrom("47m258p369s東南西北中", { kyoku: 7 }));
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
// 後付け / 片和了り. Both are judged at the WIN, never at the waiting hand:
// sitting in an open 形式聴牌 or on a split wait is a shape, not a 禁じ手 — the
// foul is cashing it. Both read the pre-win hand through the real scorer, so
// every fixture below is built against `scorer`, not the `ANY_WIN` placeholder
// (under which every wait is ronnable and neither rule can fire).
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

/**
 * The canonical split wait: 456m 678p 22s 白白 — シャンポン 2s/白, either open
 * (チー123m in front of it) or 門前 (123m held). 白 completes a 役牌 triplet and
 * scores; 2s completes nothing that does — the 1m kills 断幺九 and 456m kills
 * 混全帯幺九, and a 白 PAIR is not a yaku.
 *
 * The spare copies come back with it: the third 白 is the winning tile, and the
 * two loose 2s are what a 純カラ fixture buries.
 */
function splitWait(t: Table, open = true): { haku: Tile; loose2s: [Tile, Tile] } {
  const body = tiles("123m1p456m678p"); // 123m, stray 1p, 456m, 678p
  const s2 = tiles("2222s");
  const w = tiles("白白白");
  if (open) chi(t, 0, body.slice(0, 3));
  setHand(t, 0, [
    ...(open ? [] : body.slice(0, 3)),
    ...body.slice(4),
    s2[0],
    s2[1],
    w[0],
    w[1],
  ]);
  return { haku: w[2], loose2s: [s2[2], s2[3]] };
}

Deno.test("片和了り: ロン on the scoring side of a live split wait is charged", () => {
  const t = makeTable();
  const { haku } = splitWait(t);
  doDiscard(t, 1, haku); // P1 lets the 白 go and P0 takes it
  const vs = fire("on-win", ctx(t, 0, { t: "ron" }, null, scorer));
  const v = vs.find((x) => x.rule === "katagari");
  assert(v !== undefined, `expected katagari, got [${vs.map((x) => x.rule).join(",")}]`);
  assertEquals(v.tier, "A");
  assertEquals(v.confidence, 1, "the win oracle answers exactly");
  assert(v.detail.includes("19") && v.detail.includes("31"), `evidence names 2s/白: ${v.detail}`);
  assert(!ids("atozuke", vs), "a wait that scores is not 後付け");
});

Deno.test("片和了り: 門前ツモ is exempt — 門前清自摸和 scores on every tile", () => {
  const t = makeTable();
  const { haku } = splitWait(t, false);
  t.hands[0].push(haku); // the drawn 白, still in hand at the on-win hook
  assert(!ids("katagari", fire("on-win", ctx(t, 0, { t: "tsumo" }, haku, scorer))));
});

Deno.test("片和了り: a 門前 RON on the scoring side still charges", () => {
  // The exemption above is 自摸 only. A closed hand that waits on a tile it
  // cannot use and rons the one it can has done exactly what the rule forbids —
  // and unlike a 立直 hand it never certified the other side.
  const t = makeTable();
  const { haku } = splitWait(t, false);
  doDiscard(t, 1, haku);
  const vs = fire("on-win", ctx(t, 0, { t: "ron" }, null, scorer));
  assert(ids("katagari", vs), `expected katagari, got [${vs.map((x) => x.rule).join(",")}]`);
});

Deno.test("片和了り: 立直 dissolves the split — 立直 itself scores on every wait", () => {
  // Pinning down how `analyze` resolves this rather than legislating over it:
  // `hasYaku` runs with `t.riichi[seat]`, so after a declaration EVERY wait is
  // ronnable, `katagari` is false, and the rule never reaches its test. A
  // 片和了り therefore cannot survive a 立直 — which is why the 門前 clause above
  // needs no riichi carve-out.
  const t = makeTable();
  const { haku } = splitWait(t, false);
  t.riichi[0] = true;
  doDiscard(t, 1, haku);
  assert(!ids("katagari", fire("on-win", ctx(t, 0, { t: "ron" }, null, scorer))));
});

Deno.test("片和了り: an OPEN 自摸 on the split wait is charged", () => {
  const t = makeTable();
  const { haku } = splitWait(t);
  t.hands[0].push(haku); // drawn, and still in hand when the hook fires
  const vs = fire("on-win", ctx(t, 0, { t: "tsumo" }, haku, scorer));
  const v = vs.find((x) => x.rule === "katagari");
  assert(v !== undefined, `expected katagari, got [${vs.map((x) => x.rule).join(",")}]`);
  assert(v.detail.includes("ツモ"), `evidence should name the win: ${v.detail}`);
});

Deno.test("片和了り: exempt when the yakuless side is 純カラ", () => {
  const t = makeTable();
  const { haku, loose2s } = splitWait(t);
  for (const two of loose2s) doDiscard(t, 1, two); // both remaining 2s are gone
  doDiscard(t, 1, haku);
  assert(!ids("katagari", fire("on-win", ctx(t, 0, { t: "ron" }, null, scorer))));
});

Deno.test("後付け: a win with no scoring wait at all", () => {
  const t = makeTable();
  const all = yakulessTenpai(t);
  doDiscard(t, 0, all[3]); // the stray 1p — now waiting 6s/9s, neither scoring
  const vs = fire("on-win", ctx(t, 0, { t: "ron" }, null, scorer));
  const v = vs.find((x) => x.rule === "atozuke");
  assert(v !== undefined, `expected atozuke, got [${vs.map((x) => x.rule).join(",")}]`);
  assertEquals(v.tier, "A");
  assertEquals(v.confidence, 1, "the win oracle answers exactly");
  assert(v.detail.includes("23") && v.detail.includes("26"), `evidence names 6s/9s: ${v.detail}`);
  assert(!ids("katagari", vs), "no wait scores, so this is not 片和了り");
});

Deno.test("後付け: the same win judged by ANY_WIN charges nothing", () => {
  // The documented degradation. With the placeholder oracle every wait is
  // ronnable, so `ronnable.length === 0` is never true: with no scorer wired in
  // there is no evidence, and the ledger does not guess.
  const t = makeTable();
  const all = yakulessTenpai(t);
  doDiscard(t, 0, all[3]);
  const vs = fire("on-win", ctx(t, 0, { t: "ron" }, null, ANY_WIN));
  assert(!ids("atozuke", vs));
  assert(!ids("katagari", vs));
});

Deno.test("後付け: an open 自摸 bailed out by a lucky yaku is still 後付け", () => {
  // The shape the shipped wiring actually catches: `legal.ts` never offers a ron
  // on a yakuless wait, but it does offer the 自摸 whose only yaku is 海底/嶺上 —
  // a yaku the WAIT never carried. The hand won without ever having one.
  const t = makeTable();
  const all = yakulessTenpai(t);
  doDiscard(t, 0, all[3]);
  const six = tiles("6s")[0]; // completes 678s; no copy of it is in the fixture
  t.hands[0].push(six);
  assert(ids("atozuke", fire("on-win", ctx(t, 0, { t: "tsumo" }, six, scorer))));
});

Deno.test("後付け: a 門前 win is exempt — 立直 certifies, 門前ツモ scores", () => {
  const t = makeTable();
  // The very same 13 tiles, none of them melded.
  const all = tiles("123m1p456m678p55s78s");
  setHand(t, 0, all);
  doDiscard(t, 0, all[3]);
  assert(!ids("atozuke", fire("on-win", ctx(t, 0, { t: "ron" }, null, scorer))));
  const six = tiles("6s")[0];
  t.hands[0].push(six);
  assert(!ids("atozuke", fire("on-win", ctx(t, 0, { t: "tsumo" }, six, scorer))));
});

Deno.test("後付け: a 混一色 build is not a violation — every completion scores", () => {
  // The false positive the original on-call rule produced: a 白 back pair inside
  // an obvious honitsu build read as バック, though every completion carries
  // 混一色. チー123s ＋ 456s 789s 33s 白白: シャンポン 3s/白, both sides scoring.
  const t = makeTable();
  const all = tiles("123s5m456789s33s白白");
  chi(t, 0, all.slice(0, 3));
  setHand(t, 0, all.slice(3));
  doDiscard(t, 0, all[3]); // the stray 5m
  const vs = fire("on-win", ctx(t, 0, { t: "ron" }, null, scorer));
  assert(!ids("atozuke", vs));
  assert(!ids("katagari", vs), "both waits score, so the wait was never split");
});

Deno.test("後付け/片和了り: nothing is charged while the hand merely waits", () => {
  // The inversion this reworking is about. An open yakuless tenpai is a legal
  // shape — it takes 聴牌料 at a draw — so no discard, and no 立直, may ledger it.
  const t = makeTable();
  const all = yakulessTenpai(t);
  const first = doDiscard(t, 0, all[3]); // into the yakuless tenpai
  assertEquals(fire("post-discard", ctx(t, 0, first, null, scorer)), []);
  assertEquals(fire("on-riichi", ctx(t, 0, first, null, scorer)), []);
  // Several 巡 later, same shape (the drawn tile goes straight back out).
  for (const spare of tiles("9p9p9p")) {
    const later = doDiscard(t, 0, spare);
    assertEquals(
      fire("post-discard", ctx(t, 0, later, null, scorer)),
      [],
      "the waiting hand must stay clean however long it waits",
    );
  }
  assertEquals(t.ledger, []);
});

Deno.test("片和了り: a split wait charges only once — the seat wins once", () => {
  // The state-time rule needed a ledger-dedupe clause to avoid charging on every
  // later discard. At the win there is nothing to dedupe: one win, one charge.
  const t = makeTable();
  const { haku } = splitWait(t);
  doDiscard(t, 1, haku);
  const vs = fire("on-win", ctx(t, 0, { t: "ron" }, null, scorer));
  assertEquals(vs.filter((v) => v.rule === "katagari").length, 1);
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
