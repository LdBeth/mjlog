// Differential test: our wall layout vs a real Tenhou log.
//
// This is `checkMlogRounds` (tenhou.cc:49) reimplemented in TypeScript. It
// validates, in one shot, the shuffle direction, the deal interleaving, the
// live-wall draw direction (tiles[135 - i]), the dead-wall dora/ura slots, and
// the rinshan order {1,0,3,2} — i.e. everything `Wall` asserts about layout.

import { assertEquals } from "@std/assert";
import { parseGame } from "mjrender/parse.ts";
import type { Round, Tile } from "mjrender/model.ts";
import { sfc32 } from "../src/rng.ts";
import { Wall } from "../src/wall.ts";
import { tenhouWalls } from "../src/tenhou_wall.ts";
import { repoPath } from "./helpers.ts";

const RINSHAN_ORD = [1, 0, 3, 2];

function seedAttr(xml: string): string {
  const m = xml.match(/<SHUFFLE[^>]*seed="([^"]*)"/);
  if (!m) throw new Error("no SHUFFLE seed in log");
  return m[1];
}

/** Rebuild the 52-tile deal sequence from per-seat hands (mjlog.cc:112-119). */
function dealSequence(round: Round): Tile[] {
  const out: Tile[] = [];
  for (let block = 0; block < 3; block++) {
    for (let j = 0; j < 4; j++) {
      const hand = round.startHands[(round.dealer + j) % 4];
      out.push(...hand.slice(block * 4, block * 4 + 4));
    }
  }
  for (let j = 0; j < 4; j++) {
    out.push(round.startHands[(round.dealer + j) % 4][12]);
  }
  return out;
}

Deno.test("wall layout matches a real Tenhou log", async () => {
  const xml = await Deno.readTextFile(repoPath("1.xml"));
  const game = parseGame(xml);
  const walls = tenhouWalls(seedAttr(xml), game.rounds.length);

  assertEquals(walls.length, game.rounds.length);

  for (let r = 0; r < game.rounds.length; r++) {
    const round = game.rounds[r];
    const { tiles, dice } = walls[r];
    const where = `round ${r} (kyoku ${round.kyoku})`;

    // Dice, as recorded in INIT seed[3],seed[4].
    assertEquals(dice, round.dice, `${where}: dice`);

    // First dora indicator lives at dead-wall slot 5.
    assertEquals(tiles[5], round.firstDora, `${where}: first dora`);

    // Live wall: the deal, then every non-rinshan draw, taken from the top.
    const live = dealSequence(round);
    for (const e of round.events) {
      if (e.t === "draw" && !e.rinshan) live.push(e.tile);
    }
    for (let i = 0; i < live.length; i++) {
      assertEquals(live[i], tiles[135 - i], `${where}: live draw ${i}`);
    }

    // Rinshan draws come from dead-wall slots 1,0,3,2 in that order.
    const rinshan = round.events.filter((e) => e.t === "draw" && e.rinshan);
    for (let k = 0; k < rinshan.length; k++) {
      const e = rinshan[k] as { tile: Tile };
      assertEquals(e.tile, tiles[RINSHAN_ORD[k]], `${where}: rinshan ${k}`);
    }

    // Subsequent dora indicators occupy slots 7, 9, 11, 13.
    const doras = round.events.filter((e) => e.t === "dora");
    for (let k = 0; k < doras.length; k++) {
      const e = doras[k] as { indicator: Tile };
      assertEquals(e.indicator, tiles[5 + 2 * (k + 1)], `${where}: dora ${k + 1}`);
    }

    // Capacity invariant asserted by tenhou.cc:117.
    const kans = rinshan.length;
    const drawn = live.length - 52 + kans;
    assertEquals(drawn <= 70, true, `${where}: ${drawn} draws exceeds 70`);
  }
});

Deno.test("Wall.shuffled is a permutation and reproducible", () => {
  const a = Wall.shuffled(sfc32(42));
  const b = Wall.shuffled(sfc32(42));
  const c = Wall.shuffled(sfc32(43));

  assertEquals(a.tiles, b.tiles, "same seed ⇒ same wall");
  assertEquals(a.tiles.length, 136);
  assertEquals([...a.tiles].sort((x, y) => x - y), Array.from({ length: 136 }, (_, i) => i));
  assertEquals(a.tiles.every((t, i) => t === c.tiles[i]), false, "different seed ⇒ different wall");
});

Deno.test("Wall accounting: 52 dealt, 70 post-deal draws, kans trade live for rinshan", () => {
  const w = Wall.shuffled(sfc32(7));
  const hands = w.deal(0);
  assertEquals(hands.map((h) => h.length), [13, 13, 13, 13]);
  assertEquals(w.remaining, 70);

  const seen = new Set<Tile>(hands.flat());
  assertEquals(seen.size, 52, "deal produces 52 distinct tiles");

  w.revealIndicator();
  for (let i = 0; i < 68; i++) w.draw();
  w.drawRinshan();
  assertEquals(w.kanCount, 1);
  assertEquals(w.remaining, 1);
  w.draw();
  assertEquals(w.exhausted, true);
});

Deno.test("Wall dora/ura indicators come from interleaved dead-wall slots", () => {
  const w = Wall.shuffled(sfc32(9));
  w.deal(0);
  const d0 = w.revealIndicator();
  const d1 = w.revealIndicator();
  assertEquals(d0, w.tiles[5]);
  assertEquals(d1, w.tiles[7]);
  assertEquals(w.doraIndicators(), [w.tiles[5], w.tiles[7]]);
  assertEquals(w.uraIndicators(), [w.tiles[4], w.tiles[6]]);
});

Deno.test("Wall.peekLive follows the LIVE sequence and a kan does not consume it", () => {
  const w = Wall.shuffled(sfc32(11));
  w.deal(0);
  w.revealIndicator();

  // The next six live draws, read before any of them are taken.
  const peeked = [0, 1, 2, 3, 4, 5].map((k) => w.peekLive(k));
  assertEquals(peeked.filter((t) => t !== null).length, 6);

  assertEquals(w.draw(), peeked[0]);
  assertEquals(w.draw(), peeked[1]);
  // Peeks are relative to what is left: after two draws, peek 0 is the third.
  assertEquals(w.peekLive(0), peeked[2]);

  // A kan draws from the dead wall (RINSHAN_ORD), so it must not advance the
  // live sequence — only shorten what is left of the round.
  const before = w.remaining;
  const rinshan = w.drawRinshan();
  assertEquals(rinshan, w.tiles[1], "first rinshan is dead-wall slot 1");
  assertEquals(peeked.includes(rinshan), false, "rinshan is not a live-wall tile");
  assertEquals(w.remaining, before - 1);
  assertEquals(w.peekLive(0), peeked[2], "the kan did not eat a live tile");

  assertEquals(w.draw(), peeked[2]);
  assertEquals(w.draw(), peeked[3]);
});

Deno.test("Wall.peekLive is null past the end of the round", () => {
  const w = Wall.shuffled(sfc32(12));
  w.deal(0);
  w.revealIndicator();
  assertEquals(w.remaining, 70);
  assertEquals(w.peekLive(69) !== null, true);
  assertEquals(w.peekLive(70), null, "one past the last drawable tile");
  assertEquals(w.peekLive(-1), null);

  for (let i = 0; i < 70; i++) w.draw();
  assertEquals(w.peekLive(0), null, "exhausted wall peeks at nothing");
});

Deno.test("Wall.peekNextIndicator is what the next reveal turns up", () => {
  const w = Wall.shuffled(sfc32(13));
  w.deal(0);
  for (let k = 0; k < 5; k++) {
    const peek = w.peekNextIndicator();
    assertEquals(peek, w.revealIndicator(), `indicator ${k}`);
  }
  assertEquals(w.peekNextIndicator(), null, "no sixth indicator");
});
