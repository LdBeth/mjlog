// 振聴 fixtures, driven through the real round generator.
//
// Furiten is the one win restriction that cannot be checked from a static
// Table: it is a property of the seat's 13-tile hand against its own river, and
// `round.ts` is the only place that knows when that hand last changed. So these
// tests stack a wall, run `runRound` to the point of interest and assert on the
// action lists the game master actually offered — not on a hand-built fixture.

import { assert, assertEquals, assertFalse } from "@std/assert";
import type { Tile } from "mjrender/model.ts";
import { tileType } from "mjrender/tiles.ts";
import { observe } from "../src/observe.ts";
import { type RoundResult, runRound } from "../src/round.ts";
import { DOJO_DEFAULT, JANKI } from "../src/rules.ts";
import { scorer } from "../src/score.ts";
import type { Furiten, Table } from "../src/table.ts";
import type { Action, Request, Seat } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { Wall } from "../src/wall.ts";
import { tiles } from "./helpers.ts";

// ---------------------------------------------------------------------------
// Fixture plumbing
// ---------------------------------------------------------------------------

/**
 * A per-round tile allocator. `tiles()` restarts its copy counter on every
 * call, so a round assembled from several specs would hand the same id to two
 * seats; this hands out the lowest still-free copy of each requested type.
 */
function deck(): (spec: string) => Tile[] {
  const taken = new Set<Tile>();
  return (spec: string) => {
    return tiles(spec).map((want) => {
      const ty = tileType(want);
      let id = ty * 4;
      while (taken.has(id)) id++;
      if (id >= ty * 4 + 4) throw new Error(`fifth copy of tile type ${ty}`);
      taken.add(id);
      return id;
    });
  };
}

/**
 * A wall that deals exactly `hands` and then yields exactly `draws`, mirroring
 * `Wall.deal`'s consumption order (three blocks of four per seat from the
 * dealer, then one each). `indicator` lands in the dora slot; everything left
 * over fills the rest of the wall, so the round can still run to exhaustion.
 */
function stackedWall(hands: Tile[][], draws: Tile[], indicator: Tile, dealer: Seat = 0): Wall {
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
  for (const d of draws) put(d);
  t[5] = indicator;

  const used = new Set(t.filter((id) => id >= 0));
  const spare: Tile[] = [];
  for (let id = 0; id < 136; id++) if (!used.has(id)) spare.push(id);
  for (let i = 0; i < 136; i++) if (t[i] < 0) t[i] = spare.pop()!;
  return new Wall(t);
}

interface Step {
  req: Request;
  /** Every seat's three furiten flags at the instant the request was yielded. */
  furiten: Furiten[];
}

type Decide = (req: Request, t: Table) => Action;

interface Played {
  result: RoundResult;
  table: Table;
  trace: Step[];
}

function playRound(o: {
  hands: Tile[][];
  draws: Tile[];
  indicator: Tile;
  decide: Decide;
}): Played {
  let table!: Table;
  const trace: Step[] = [];
  const gen = runRound(
    {
      kyoku: 0,
      honba: 0,
      kyotaku: 0,
      dealer: 0,
      scores: [30000, 30000, 30000, 30000],
      wall: stackedWall(o.hands, o.draws, o.indicator),
      dice: [0, 0],
    },
    { cfg: JANKI, dojo: DOJO_DEFAULT, scorer, onTable: (t) => (table = t) },
  );

  let r = gen.next();
  for (;;) {
    if (r.done) return { result: r.value, table, trace };
    trace.push({ req: r.value, furiten: SEATS.map((s) => ({ ...table.furiten[s] })) });
    r = gen.next(o.decide(r.value, table));
  }
}

/** Tsumogiri every turn, pass every claim — the inert baseline for a script. */
function passive(req: Request): Action {
  if (req.k === "claim") return { t: "pass" };
  const d = req.legal.find((a) => a.t === "discard" && a.tsumogiri && !a.riichi) ??
    req.legal.find((a) => a.t === "discard" && !a.riichi);
  if (!d) throw new Error(`seat ${req.seat} has no discard`);
  return d;
}

const hasRon = (as: Action[]) => as.some((a) => a.t === "ron");
const hasTsumo = (as: Action[]) => as.some((a) => a.t === "tsumo");

type ClaimStep = Step & { req: Extract<Request, { k: "claim" }> };
type TurnStep = Step & { req: Extract<Request, { k: "turn" }> };

/** Every claim window `seat` was offered on a copy of `tile`. */
function claimsOn(trace: Step[], seat: Seat, tile: Tile): ClaimStep[] {
  return trace.filter((s): s is ClaimStep =>
    s.req.k === "claim" && s.req.seat === seat && tileType(s.req.tile) === tileType(tile)
  );
}

function turnsOf(trace: Step[], seat: Seat): TurnStep[] {
  return trace.filter((s): s is TurnStep => s.req.k === "turn" && s.req.seat === seat);
}

/** Three junk hands, three shanten from anything and frozen by tsumogiri. */
function junkHands(d: (spec: string) => Tile[]): [Tile[], Tile[], Tile[]] {
  const one = () => d("149m149p149s東南西北");
  return [one(), one(), one()];
}

// ---------------------------------------------------------------------------
// 1 / 2 / 6 — 河に待ちがあれば待ち全体がロン不可、ただしツモは可
// ---------------------------------------------------------------------------

/**
 * Seat 1 is 両面 6s/9s (234m 567m 234p 55p 78s) and lets a 9s go through its
 * own river. `banked` switches that 9s for an unrelated tile, giving the
 * control run in which the ron IS offered — without it, "no ron" would prove
 * nothing about furiten (it could just as well be a missing yaku).
 */
function ryanmenRound(furitenTrap: boolean) {
  const d = deck();
  const seat1 = d("234m567m234p55p78s");
  const [h0, h2, h3] = junkHands(d);
  const nineS = d("9s")[0];
  const decoy = d("8m")[0];
  const fill = d("8m8m8m");
  const sixS = d("6s6s");
  const observed: Furiten[] = [];

  const played = playRound({
    hands: [h0, seat1, h2, h3],
    // d0 seat0, d1 seat1, d2 seat2, d3 seat3, d4 seat0, d5 seat1.
    draws: [fill[0], furitenTrap ? nineS : decoy, fill[1], fill[2], sixS[0], sixS[1]],
    indicator: d("北")[0],
    decide: (req, t) => {
      if (req.k === "claim" && req.seat === 1) {
        // Seat 1 also gets chi windows on the 8m fillers; only the 6s matters.
        if (tileType(req.tile) === tileType(sixS[0])) {
          observed.push({ ...observe(t, 1, req.legal, null, scorer, req.tile).furiten });
        }
        // The control run takes the ron; the trap run cannot be offered one.
        const ron = req.legal.find((a) => a.t === "ron");
        if (ron) return ron;
      }
      if (req.k === "turn" && req.seat === 1 && req.drawn === sixS[1]) {
        const tsumo = req.legal.find((a) => a.t === "tsumo");
        if (tsumo) return tsumo;
      }
      return passive(req);
    },
  });
  return { ...played, seat1Waits: { nineS, sixS }, observed };
}

Deno.test("振聴: a wait sitting in one's own river blocks ron on the WHOLE wait set", () => {
  const { trace, observed, seat1Waits } = ryanmenRound(true);

  // The flag is set by the discard itself, so it is already true when the next
  // seat is asked to act.
  assert(turnsOf(trace, 2)[0].furiten[1].permanent, "9s 切り after tenpai must set 振聴");

  // Seat 0 then cuts a 6s — a tile seat 1 never discarded, and the other half
  // of the same 両面. Furiten covers the whole wait, not just the 9s.
  const claims = claimsOn(trace, 1, seat1Waits.sixS[0]);
  assertEquals(claims.length, 1, "seat 1 should still be polled (it can chi the 6s)");
  assertFalse(hasRon(claims[0].req.legal), "振聴ロン must not be offered");
  assert(claims[0].req.legal.some((a) => a.t === "chi"), "the 6s really is claimable");

  // 6 — the same truth reaches a policy through `observe()`.
  assertEquals(observed.length, 1);
  assert(observed[0].permanent, "observe() must report the permanent flag");
});

Deno.test("振聴: the same seat may still ツモ (furiten never gates a tsumo)", () => {
  const { result, trace } = ryanmenRound(true);

  const drawn6s = turnsOf(trace, 1)[1];
  assert(drawn6s.furiten[1].permanent, "still 振聴 on the draw");
  assert(hasTsumo(drawn6s.req.legal), "ツモ must be offered while 振聴");

  assertEquals(result.outcome.kind, "agari");
  if (result.outcome.kind !== "agari") return;
  const [win] = result.outcome.wins;
  assertEquals(win.who, 1);
  assertEquals(win.fromWho, 1, "a tsumo, not a ron");
});

Deno.test("振聴: control — without the 9s in the river the very same ron IS offered", () => {
  const { result, trace, observed, seat1Waits } = ryanmenRound(false);

  assertFalse(turnsOf(trace, 2)[0].furiten[1].permanent);
  const claims = claimsOn(trace, 1, seat1Waits.sixS[0]);
  assertEquals(claims.length, 1);
  assert(hasRon(claims[0].req.legal), "a clean 両面 must be ronnable");
  assertFalse(observed[0].permanent);

  assertEquals(result.outcome.kind, "agari");
  if (result.outcome.kind !== "agari") return;
  assertEquals(result.outcome.wins[0].who, 1);
  assertEquals(result.outcome.wins[0].fromWho, 0);
});

// ---------------------------------------------------------------------------
// 3 — 手替わりで待ちが河から外れれば振聴は消える
// ---------------------------------------------------------------------------

Deno.test("振聴: a hand that shifts off the discarded wait is no longer 振聴", () => {
  const d = deck();
  const seat1 = d("234m567m234p55p78s");
  const [h0, h2, h3] = junkHands(d);
  const nineS = d("9s")[0];
  const fill = d("8m8m8m8m");
  const fiveS = d("5s")[0];
  const filler2 = d("6m6m");
  const sixS = d("6s")[0];

  // 78s is the tail of the hand spec, so the 8s is the last id in it (type 25).
  const heldEightS = seat1[seat1.length - 1];
  assertEquals(tileType(heldEightS), 25, "fixture: the last hand tile is the 8s");

  const { result, trace } = playRound({
    hands: [h0, seat1, h2, h3],
    // d0..d3 open the go-around, d5 hands seat 1 the 5s it reshapes on, and d8
    // is the 6s seat 0 finally lets go.
    draws: [fill[0], nineS, fill[1], fill[2], fill[3], fiveS, filler2[0], filler2[1], sixS],
    indicator: d("北")[0],
    decide: (req) => {
      if (req.k === "turn" && req.seat === 1 && req.drawn === fiveS) {
        // 8s 切り: 5s7s カンチャン, which the 9s in the river does not touch.
        const cut = req.legal.find((a) => a.t === "discard" && a.tile === heldEightS);
        assert(cut, "the held 8s must be discardable");
        return cut;
      }
      if (req.k === "claim" && req.seat === 1) {
        const ron = req.legal.find((a) => a.t === "ron");
        if (ron) return ron;
      }
      return passive(req);
    },
  });

  // Before the reshape the seat is 振聴; after it, clean again.
  assert(turnsOf(trace, 2)[0].furiten[1].permanent, "9s 切り made it 振聴");
  assertFalse(turnsOf(trace, 2)[1].furiten[1].permanent, "8s 切り must clear it");

  const claims = claimsOn(trace, 1, sixS);
  assertEquals(claims.length, 1);
  assert(hasRon(claims[0].req.legal), "the reshaped wait is ronnable again");

  assertEquals(result.outcome.kind, "agari");
  if (result.outcome.kind !== "agari") return;
  assertEquals(result.outcome.wins[0].who, 1);
  assertEquals(result.outcome.wins[0].fromWho, 0);
});

// ---------------------------------------------------------------------------
// 4 — 振聴立直: 待ちは一局中ロンできない
// ---------------------------------------------------------------------------

Deno.test("振聴: a riichi declared over one's own discard is never offered a ron", () => {
  const d = deck();
  const seat1 = d("234m567m234p55p78s");
  const [h0, h2, h3] = junkHands(d);
  const nineS = d("9s")[0];
  const fill = d("8m8m8m8m");
  const filler2 = d("6m6m");
  const sixS = d("6s6s6s");

  const { result, trace } = playRound({
    hands: [h0, seat1, h2, h3],
    // seat 1 riichis on the 9s (d1); seat 0 then cuts a 6s twice (d4, d8) before
    // seat 1 draws the third itself (d9).
    draws: [
      fill[0],
      nineS,
      fill[1],
      fill[2],
      sixS[0],
      fill[3],
      filler2[0],
      filler2[1],
      sixS[1],
      sixS[2],
    ],
    indicator: d("北")[0],
    decide: (req) => {
      if (req.k === "turn" && req.seat === 1 && req.drawn === nineS) {
        const riichi = req.legal.find((a) => a.t === "discard" && a.riichi && a.tile === nineS);
        assert(riichi, "a 振聴立直 is legal in the dojo, only penalised");
        return riichi;
      }
      if (req.k === "turn" && req.seat === 1) {
        const tsumo = req.legal.find((a) => a.t === "tsumo");
        if (tsumo) return tsumo;
      }
      return passive(req);
    },
  });

  // Riichi bars calls, so a barred seat is not polled at all — the assertion is
  // that seat 1 is never asked about a 6s, on either of seat 0's two cuts.
  assertEquals(claimsOn(trace, 1, sixS[0]).length, 0, "no ron window for a 振聴立直");
  for (const s of turnsOf(trace, 1).slice(1)) {
    assert(s.furiten[1].permanent, "the 振聴 sticks for the whole hand");
  }

  // It can still be completed by ツモ.
  assertEquals(result.outcome.kind, "agari");
  if (result.outcome.kind !== "agari") return;
  assertEquals(result.outcome.wins[0].who, 1);
  assertEquals(result.outcome.wins[0].fromWho, 1);
  assert(
    result.outcome.wins[0].yaku.some((y) => y.han > 0),
    "立直+ツモ scores",
  );
});

// ---------------------------------------------------------------------------
// 5 — 見逃し: ロンを提示されなくても同巡振聴になる
// ---------------------------------------------------------------------------

Deno.test("振聴: passing a completing tile is 見逃し even when no ron was offered", () => {
  const d = deck();
  // 111m 345m 678p + 22s/白白 シャンポン. Winning on 白 is 役牌; winning on the
  // 2s is a bare open-ended shape with no yaku at all, so no ron is offered
  // there — but the tile still completes the hand, so letting it pass is 見逃し.
  const seat1 = d("111m345m678p22s白白");
  const junk = () => d("9m9p1p4p9s1s4s東南西北發中");
  const [h0, h2, h3] = [junk(), junk(), junk()];
  const twoS = d("2s")[0];
  const haku = d("白白");
  const fill = d("7s8s6s5s");

  const { result, trace } = playRound({
    hands: [h0, seat1, h2, h3],
    // d3 seat 3 cuts the 2s (yakuless ⇒ no ron), d4 seat 0 cuts a 白 in the same
    // go-around (blocked by the fresh 同巡振聴), d5 is seat 1's own draw (which
    // clears it), d6 seat 2 cuts the last 白.
    draws: [fill[0], fill[1], fill[2], twoS, haku[0], fill[3], haku[1]],
    indicator: d("北")[0],
    decide: (req) => {
      if (req.k === "claim" && req.seat === 1) {
        const ron = req.legal.find((a) => a.t === "ron");
        if (ron) return ron;
      }
      return passive(req);
    },
  });

  const onTwoS = claimsOn(trace, 1, twoS);
  assertEquals(onTwoS.length, 1, "seat 1 is polled — it can pon the 2s");
  assertFalse(hasRon(onTwoS[0].req.legal), "fixture: the 2s shape is yakuless");
  assertFalse(onTwoS[0].furiten[1].temporary, "not yet 見逃し when the tile is offered");

  // The pass is what arms it, even though a ron was never on the table.
  const onHaku = claimsOn(trace, 1, haku[0]);
  assertEquals(onHaku.length, 2, "both 白 windows reach seat 1");
  assert(onHaku[0].furiten[1].temporary, "見逃し must be marked without an offered ron");
  assertFalse(hasRon(onHaku[0].req.legal), "同巡振聴 blocks the 役牌 ron");

  // …and it is only same-turn: seat 1's next draw clears it, so the second 白
  // is ronnable.
  assertFalse(turnsOf(trace, 1)[1].furiten[1].temporary, "the draw clears 同巡振聴");
  assertFalse(turnsOf(trace, 1)[1].furiten[1].permanent, "nothing landed in its own river");
  assert(hasRon(onHaku[1].req.legal), "after the draw the same ron is legal");

  assertEquals(result.outcome.kind, "agari");
  if (result.outcome.kind !== "agari") return;
  assertEquals(result.outcome.wins[0].who, 1);
  assertEquals(result.outcome.wins[0].fromWho, 2);
});
