// The danger map's visibility accounting.
//
// `assessDanger` (mjrender/src/danger.ts) counts a tile's remaining copies as
// `4 − (visibleCounts + ownCounts)`: the first argument must be PUBLIC evidence
// only. `Table.visibleCounts(seat)` deliberately folds the seat's own concealed
// hand in — the ukeire `live` field wants exactly that — so passing it straight
// through counted every held copy TWICE. Every tile the seat held 2+ copies of
// was under-counted, and the honor branch's `visible[type] >= 3` rule read the
// wrong number too.
//
// The live consequence (2026-08-28 arena wire log): a seat holding 東東 with a
// single 東 in someone's river read 場に3枚 / 当たり形:なし ⇒ 安全 against both
// a riichi and a 3副露 hand, and 安全 is a PROOF — `AugmentedHeuristic.riskOf`
// returns on it without consulting its own deal-in estimate. The 東 went out
// into a 12000 単騎 ron.

import { assert, assertEquals } from "@std/assert";
import type { Tile } from "mjrender/model.ts";
import { observe } from "../src/observe.ts";
import { JANKI } from "../src/rules.ts";
import { Table } from "../src/table.ts";
import type { Seat } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { Wall } from "../src/wall.ts";
import { tiles } from "./helpers.ts";

/**
 * An UNSHUFFLED wall, so the round's dora indicator is a known constant
 * (`tiles[5]` = id 5 = 2萬) and cannot wander onto a type under test. Every
 * hand is overwritten below, so what the deal handed out never matters.
 */
function makeTable(): Table {
  const ids: Tile[] = Array.from({ length: 136 }, (_, i) => i);
  return new Table(
    {
      kyoku: 0,
      honba: 0,
      kyotaku: 0,
      dealer: 0,
      scores: [25000, 25000, 25000, 25000],
      wall: new Wall(ids),
      dice: [0, 0],
    },
    JANKI,
    SEATS.map((seat) => ({ seat, name: `P${seat}` })),
  );
}

function setHand(t: Table, seat: Seat, hand: Tile[]): void {
  t.hands[seat].length = 0;
  t.hands[seat].push(...hand);
}

function discard(t: Table, seat: Seat, tile: Tile, riichi = false): void {
  t.emit(
    { t: "discard", who: seat, tile, tsumogiri: false, riichi },
    { e: "discard", who: seat, tile, tsumogiri: false, riichi },
  );
  if (riichi) t.riichi[seat] = true;
}

/**
 * 下家 has cut one 東 early; 対面 then declares riichi on a 6筒. So the 東 is
 * NOT genbutsu against the riichi seat — its safe set is its OWN discards plus
 * everything cut after the declaration (`BoardState.discard`), and the 東 is
 * neither. Note the shape of this fixture: the East MUST come from a third
 * seat. A riichi seat's whole river, pre-declaration discards included, is
 * genbutsu against it by furiten, so "in the riichi seat's own early river"
 * could never produce a live 東.
 *
 * 起家 (seat 0) holds 東東 and 1筒1筒 — the two-copy holdings the double count
 * used to erase.
 */
function tableUnderRiichi(): Table {
  // ONE `tiles()` call per disjoint type set: copies of a type are handed out
  // in ascending id order, so two calls would hand out the same 東.
  const easts = tiles("東東東");
  const t = makeTable();
  setHand(t, 0, [...easts.slice(0, 2), ...tiles("123456789m11p")]);
  setHand(t, 1, [easts[2], ...tiles("23456789s99s34p")]);
  setHand(t, 2, tiles("222333444555m6p"));
  setHand(t, 3, tiles("666777888999m2p"));

  discard(t, 1, easts[2]);
  discard(t, 2, tiles("222333444555m6p")[12], true); // 6筒 でリーチ
  return t;
}

const EAST = 27;
const P1 = 9;

Deno.test("observe: the danger map counts our own copies ONCE (公開情報 + 手牌)", () => {
  const t = tableUnderRiichi();
  const obs = observe(t, 0 as Seat, [], null);

  const d = obs.danger.get(EAST);
  assert(d, "リーチが立っている以上 東 の評価は存在する");
  // Under the double count: 場に3枚 → 残り0枚 → 当たり形:なし → 安全 (a false
  // proof). Correctly: one 東 is public, two are ours, one is live ⇒ 単騎のみ.
  assert(d.level !== "安全", `東 が 安全 と読まれた: ${JSON.stringify(d)}`);
  assertEquals(d.level, "危険度低"); // 役牌(場風東) 危険度高 を 単騎 cap が抑える
  const notes = d.details[0].notes;
  assert(notes.includes("場に1枚"), `notes=${notes.join("/")} — 場に1枚 のはず`);
  assert(
    notes.some((n) => n.startsWith("当たり形:") && n.includes("タンキ")),
    `notes=${notes.join("/")} — 残り1枚なら タンキ は生きている`,
  );
  assert(
    notes.some((n) => n === "当たり形:タンキ"),
    `notes=${notes.join("/")} — 残り1枚なので シャンポン は消えている`,
  );
});

Deno.test("observe: a number tile we hold two copies of is not capped away", () => {
  const t = tableUnderRiichi();
  const obs = observe(t, 0 as Seat, [], null);

  const d = obs.danger.get(P1);
  assert(d, "1筒 の評価は存在する");
  // 1筒: nothing public, two in hand ⇒ two live. The double count made it four
  // and erased every wait shape, so the tile read 安全.
  assert(d.level !== "安全", `1筒 が 安全 と読まれた: ${JSON.stringify(d)}`);
  const notes = d.details[0].notes;
  assert(
    notes.some((n) => n.startsWith("当たり形:") && n.includes("シャンポン")),
    `notes=${notes.join("/")} — 残り2枚なら シャンポン は生きている`,
  );
});
