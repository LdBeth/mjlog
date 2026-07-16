// scoring.ts: placement tie-breaks and the オーラス overtake search.

import { overtakeNeeds, placements } from "../src/scoring.ts";

function eq<T>(a: T, b: T, msg: string): void {
  const as = JSON.stringify(a), bs = JSON.stringify(b);
  if (as !== bs) throw new Error(`${msg}: got ${as}, want ${bs}`);
}

Deno.test("scoring: placements break ties toward the initial East", () => {
  eq(placements([250, 250, 250, 250], 0), [1, 2, 3, 4], "all tied, 起家 first");
  eq(placements([250, 250, 250, 250], 2), [3, 4, 1, 2], "起家=seat2 rotates priority");
  eq(placements([100, 400, 200, 300], 0), [4, 1, 3, 2], "plain ordering");
});

Deno.test("scoring: overtake needs, non-dealer (sample South-4 values)", () => {
  // P1 32500 vs leader P0 42100, P3 dealer, no honba/kyotaku (verified by hand):
  // ron must exceed 9600 → haneman(12000); direct: 2p > 9600 → 5200;
  // tsumo: 4han30fu (2000-3900, total 7900, leader pays 2000) is the minimum.
  const needs = overtakeNeeds({
    scores: [421, 325, 138, 116],
    seat: 1,
    dealer: 3,
    honba: 0,
    kyotaku: 0,
    initialEast: 0,
  })!;
  eq(needs.ron, "ロン跳満", "ron threshold");
  eq(needs.direct, "直撃5200", "direct-hit threshold");
  eq(needs.tsumo, "ツモ2000-3900", "tsumo threshold");
  eq(needs.impossible, false, "reachable");
});

Deno.test("scoring: overtake needs, dealer values use 6x base / all-pay", () => {
  // P3 dealer, 11600 vs 42100 (gap 30500): dealer sanbaiman ron = 36000 wins;
  // dealer baiman tsumo = 8000オール (leader pays 8000).
  const needs = overtakeNeeds({
    scores: [421, 325, 138, 116],
    seat: 3,
    dealer: 3,
    honba: 0,
    kyotaku: 0,
    initialEast: 0,
  })!;
  eq(needs.ron, "ロン三倍満", "dealer ron threshold");
  eq(needs.direct, "直撃跳満", "dealer direct threshold");
  eq(needs.tsumo, "ツモ倍満", "dealer tsumo threshold");
});

Deno.test("scoring: honba and kyotaku count toward the gain", () => {
  // gap exactly 2000: plain ron needs 2000... with 1 kyotaku +1000 and 1 honba
  // +300, a 1000-point ron (1300 total +1000 stick) already overtakes.
  const with_ = overtakeNeeds({
    scores: [270, 250, 240, 240],
    seat: 1,
    dealer: 0,
    honba: 1,
    kyotaku: 1,
    initialEast: 0,
  })!;
  eq(with_.ron, "ロン1000", "sticks push a 1000 ron over the line");
  const without = overtakeNeeds({
    scores: [270, 250, 240, 240],
    seat: 1,
    dealer: 0,
    honba: 0,
    kyotaku: 0,
    initialEast: 0,
  })!;
  eq(without.ron, "ロン2600", "bare ron must exceed 2000 (tie loses to 起家 side)");
});

Deno.test("scoring: leader gets null, hopeless last gets impossible", () => {
  eq(
    overtakeNeeds({
      scores: [400, 200, 200, 200],
      seat: 0,
      dealer: 0,
      honba: 0,
      kyotaku: 0,
      initialEast: 0,
    }),
    null,
    "leader has no needs line",
  );
  const hopeless = overtakeNeeds({
    scores: [900, 50, 25, 25],
    seat: 2,
    dealer: 0,
    honba: 0,
    kyotaku: 0,
    initialEast: 0,
  })!;
  eq(hopeless.impossible, true, "even yakuman direct cannot close 87500");
});
