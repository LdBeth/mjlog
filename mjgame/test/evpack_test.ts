// M15 — the packer, on a machine with no `libmjev`.
//
// `evpack.ts` is pure TypeScript by design, and this file is why: the wire
// layout IS the specification of what the DP sees (the engine has no TS twin),
// so every offset has to be checkable without FFI, without `--allow-ffi`, and
// without anyone having run `deno task build-ev`. Nothing here dlopens
// anything — `EvCore` is imported as a TYPE (erased) and the two buffers are
// plain typed arrays.
//
// Two kinds of claim:
//
//   THE OFFSETS ARE FROZEN. `evlayout.ts` says "a change to any offset is an
//   ABI bump on BOTH sides"; the literal table below is the third witness, so
//   an accidental renumber fails here even if `mjev.cc` was edited in lockstep.
//
//   THE FACTS ARE THE RIGHT FACTS. `evFactsFromObservation` is the only place
//   the Observation is read, so aka accounting, the own-river bag, the horizon
//   clamp, the candidate mask, the meld encoding and the hidden-info flags are
//   asserted against hand-built boards.

import { assertEquals, assertThrows } from "@std/assert";
import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import type { EvCore } from "../src/ai/ev.ts";
import {
  D_COSTIN,
  D_DRAW,
  D_EXPLOSS,
  D_GAIN,
  D_NEXTDORA,
  D_PIN,
  D_POOL,
  D_RISK,
  D_TENPAI,
  D_URA,
  DBLS_LEN,
  I_AKA_HELD,
  I_AKA_UNSEEN,
  I_CAND,
  I_CLOSED,
  I_DEALER,
  I_DORA,
  I_FURITEN_PERM,
  I_FURITEN_TEMP,
  I_HAND,
  I_HAS_DRAW,
  I_HAS_NEXTDORA,
  I_HAS_POOL,
  I_HAS_URA,
  I_HONBA,
  I_JUNME,
  I_K,
  I_KANDORA_ON,
  I_KYOTAKU,
  I_MELDS,
  I_MODE,
  I_NMELDS,
  I_OWN_RIICHI,
  I_RIVER,
  I_ROUND_WIND,
  I_SEAT_WIND,
  I_T,
  I_UNSEEN,
  INTS_LEN,
  KMAX,
  OUT_LEN,
  REST_META_LEN,
} from "../src/ai/evlayout.ts";
import type { EvFacts } from "../src/ai/evpack.ts";
import { evFactsFromObservation, indicatorOfDora, packEvInputs, T_CAP } from "../src/ai/evpack.ts";
import { DEFAULT_EV } from "../src/ai/evparams.ts";
import type { Observation } from "../src/observe.ts";
import { doraFromIndicatorType } from "mjrender/tiles.ts";
import { JANKI } from "../src/rules.ts";
import { tiles } from "./helpers.ts";

// ---------------------------------------------------------------------------
// scaffolding
// ---------------------------------------------------------------------------

/** The two reused buffers, with no context behind them. */
function scratch(): EvCore {
  return {
    handle: 0,
    params: DEFAULT_EV,
    ints: new Int32Array(INTS_LEN),
    dbls: new Float64Array(DBLS_LEN),
    out: new Float64Array(OUT_LEN),
    meta: new Float64Array(REST_META_LEN),
  } as unknown as EvCore;
}

function river(ts: Tile[], calledBy?: number): RiverEntry[] {
  return ts.map((tile, i) => ({
    tile,
    junme: i + 1,
    tsumogiri: false,
    riichiDeclare: false,
    ...(calledBy === undefined ? {} : { calledBy }),
  }));
}

/** One distinct copy of 北/西 per seat, the way `heuristic_test` opens a board. */
function openingRivers(): RiverEntry[][] {
  const pool = tiles("北北北北西西西西");
  return [0, 1, 2, 3].map((s) => river([pool[s], pool[4 + s]]));
}

const KITA = 30; // 北
const SHA = 29; // 西
const TON = 27; // 東

function baseObs(over: Partial<Observation> = {}): Observation {
  const hand = over.hand ?? tiles("123456789m1122p東");
  const drawn = "drawn" in over ? over.drawn! : hand[hand.length - 1];
  return {
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

function meld(kind: Meld["kind"], spec: string, who = 0, fromWho = 1): Meld {
  const ts = tiles(spec);
  return { kind, who, fromWho: kind === "ankan" ? who : fromWho, tiles: ts, calledTile: ts[0] };
}

/** The policy half, all zeros — the packer copies it, it never invents it. */
function policyHalf() {
  return {
    pIn: new Array<number>(34).fill(0),
    costIn: new Array<number>(34).fill(0),
    tenpaiP: [0, 0, 0],
    expLoss: [0, 0, 0],
    gain: 1,
    risk: 1,
  };
}

// ---------------------------------------------------------------------------
// 1. the offsets are frozen
// ---------------------------------------------------------------------------

Deno.test("evpack: 配置は凍結されている (ABI 1) — 変えるなら mjev.cc と同時に", () => {
  assertEquals(
    [
      I_HAND,
      I_NMELDS,
      I_MELDS,
      I_SEAT_WIND,
      I_ROUND_WIND,
      I_DEALER,
      I_HONBA,
      I_KYOTAKU,
      I_OWN_RIICHI,
      I_FURITEN_PERM,
      I_FURITEN_TEMP,
      I_JUNME,
      I_T,
      I_AKA_HELD,
      I_AKA_UNSEEN,
      I_CLOSED,
      I_CAND,
      I_UNSEEN,
      I_DORA,
      I_RIVER,
      I_KANDORA_ON,
      I_HAS_DRAW,
      I_K,
      I_HAS_POOL,
      I_HAS_URA,
      I_HAS_NEXTDORA,
      I_MODE,
      INTS_LEN,
    ],
    [
      0,
      34,
      35,
      47,
      48,
      49,
      50,
      51,
      52,
      53,
      54,
      55,
      56,
      57,
      58,
      59,
      60,
      94,
      128,
      162,
      196,
      197,
      198,
      199,
      200,
      201,
      202,
      208,
    ],
  );
  assertEquals(
    [
      D_TENPAI,
      D_EXPLOSS,
      D_PIN,
      D_COSTIN,
      D_GAIN,
      D_RISK,
      D_DRAW,
      D_POOL,
      D_URA,
      D_NEXTDORA,
      DBLS_LEN,
      KMAX,
    ],
    [0, 3, 6, 40, 74, 75, 76, 212, 246, 280, 320, 4],
  );
});

// ---------------------------------------------------------------------------
// 2. the round trip: every ints/dbls offset, from one hand-built EvFacts
// ---------------------------------------------------------------------------

Deno.test("evpack: 手で組んだ EvFacts の全オフセットが往復する", () => {
  // Distinct sentinels everywhere, so a misplaced field lands on a value that
  // belongs to somebody else and the comparison names it.
  const hand = Array.from({ length: 34 }, (_, i) => (i % 5 === 0 ? 1 : 0));
  const candMask = Array.from({ length: 34 }, (_, i) => (i < 9 ? 1 : 0));
  const unseen = Array.from({ length: 34 }, (_, i) => (i % 4) + 1);
  const doraCount = Array.from({ length: 34 }, (_, i) => (i === 7 ? 2 : 0));
  const ownRiverBag = Array.from({ length: 34 }, (_, i) => (i === 33 ? 3 : 0));
  const pIn = Array.from({ length: 34 }, (_, i) => i / 100);
  const costIn = Array.from({ length: 34 }, (_, i) => 1000 + i);

  const f: EvFacts = {
    hand,
    mode: 0,
    melds: [
      meld("chi", "345p"), // kind 0, lowest type 11 (3p), open
      meld("pon", "777s"), // kind 1, type 24, open
      meld("ankan", "1111m"), // kind 2, type 0, concealed
      meld("shouminkan", "999m"), // kind 2, type 8, open
    ],
    seatWind: 29,
    roundWind: 28,
    dealer: false,
    honba: 5,
    kyotaku: 2,
    ownRiichi: true,
    furitenPerm: true,
    furitenTemp: false,
    junme: 11,
    T: 13,
    akaHeld: 1,
    akaUnseen: 1,
    closed: false,
    candMask,
    unseen,
    doraCount,
    ownRiverBag,
    kanDoraOn: true,
    tenpaiP: [0.1, 0.2, 0.3],
    expLoss: [4000, 5000, 6000],
    pIn,
    costIn,
    gain: 1.25,
    risk: 0.75,
  };

  const core = scratch();
  packEvInputs(core, f);

  const wi = new Int32Array(INTS_LEN);
  for (let i = 0; i < 34; i++) {
    wi[I_HAND + i] = hand[i];
    wi[I_CAND + i] = candMask[i];
    wi[I_UNSEEN + i] = unseen[i];
    wi[I_DORA + i] = doraCount[i];
    wi[I_RIVER + i] = ownRiverBag[i];
  }
  wi[I_NMELDS] = 4;
  wi.set([0, 11, 0, 1, 24, 0, 2, 0, 1, 2, 8, 0], I_MELDS);
  wi[I_SEAT_WIND] = 29;
  wi[I_ROUND_WIND] = 28;
  wi[I_DEALER] = 0;
  wi[I_HONBA] = 5;
  wi[I_KYOTAKU] = 2;
  wi[I_OWN_RIICHI] = 1;
  wi[I_FURITEN_PERM] = 1;
  wi[I_FURITEN_TEMP] = 0;
  wi[I_JUNME] = 11;
  wi[I_T] = 13;
  wi[I_AKA_HELD] = 1;
  wi[I_AKA_UNSEEN] = 1;
  wi[I_CLOSED] = 0;
  wi[I_KANDORA_ON] = 1;
  wi[I_MODE] = 0;
  assertEquals(core.ints, wi);

  const wd = new Float64Array(DBLS_LEN);
  wd.set([0.1, 0.2, 0.3], D_TENPAI);
  wd.set([4000, 5000, 6000], D_EXPLOSS);
  wd.set(pIn, D_PIN);
  wd.set(costIn, D_COSTIN);
  wd[D_GAIN] = 1.25;
  wd[D_RISK] = 0.75;
  assertEquals(core.dbls, wd);
});

// ---------------------------------------------------------------------------
// 3. the hidden-information channels: absent ≠ uniform
// ---------------------------------------------------------------------------

Deno.test("evpack: 隠蔽情報チャネルは旗で有無を区別し、K 行だけ載る", () => {
  const base = evFactsFromObservation(baseObs(), { mode: 0, ...policyHalf() });
  const core = scratch();

  // Absent: every flag 0, every row zero.
  packEvInputs(core, base);
  assertEquals(core.ints[I_HAS_DRAW], 0);
  assertEquals(core.ints[I_K], 0);
  assertEquals(core.ints[I_HAS_POOL], 0);
  assertEquals(core.ints[I_HAS_URA], 0);
  assertEquals(core.ints[I_HAS_NEXTDORA], 0);
  assertEquals(core.dbls.slice(D_DRAW, DBLS_LEN), new Float64Array(DBLS_LEN - D_DRAW));

  // Present: two draw rows, a pool, an ura and a next-dora indicator.
  const onehot = (ty: number) => {
    const v = new Float64Array(34);
    v[ty] = 1;
    return v;
  };
  const pool = new Float64Array(34).fill(1 / 34);
  packEvInputs(core, {
    ...base,
    hidden: { drawDist: [onehot(4), onehot(5)], pool, ura: onehot(9), nextDora: onehot(12) },
  });
  assertEquals(core.ints[I_HAS_DRAW], 1);
  assertEquals(core.ints[I_K], 2);
  assertEquals(core.ints[I_HAS_POOL], 1);
  assertEquals(core.ints[I_HAS_URA], 1);
  assertEquals(core.ints[I_HAS_NEXTDORA], 1);
  assertEquals(core.dbls[D_DRAW + 4], 1);
  assertEquals(core.dbls[D_DRAW + 34 + 5], 1);
  // The third row was never supplied and must stay zero, not stale.
  assertEquals(core.dbls.slice(D_DRAW + 68, D_DRAW + 102), new Float64Array(34));
  assertEquals(core.dbls[D_POOL + 0], 1 / 34);
  assertEquals(core.dbls[D_URA + 9], 1);
  assertEquals(core.dbls[D_NEXTDORA + 12], 1);

  // …and the SAME buffer packed again without them is clean: the whole point of
  // reusing the buffers is that they are the only per-decision allocation, so a
  // stale flag would be an invisible bug in a deterministic engine.
  packEvInputs(core, base);
  assertEquals(core.ints[I_HAS_DRAW], 0);
  assertEquals(core.ints[I_K], 0);
  assertEquals(core.ints[I_HAS_URA], 0);
  assertEquals(core.dbls.slice(D_DRAW, DBLS_LEN), new Float64Array(DBLS_LEN - D_DRAW));
});

Deno.test("evpack: 壊れた入力は拒否する (黙って詰めない)", () => {
  const f = evFactsFromObservation(baseObs(), { mode: 0, ...policyHalf() });
  const core = scratch();
  assertThrows(() => packEvInputs(core, { ...f, hand: [1, 2, 3] }), Error, "hand");
  assertThrows(
    () =>
      packEvInputs(core, {
        ...f,
        melds: [
          meld("pon", "111m"),
          meld("pon", "222m"),
          meld("pon", "333m"),
          meld("pon", "444m"),
          meld("pon", "555m"),
        ],
      }),
    Error,
    "面子は4つまで",
  );
  assertThrows(
    () =>
      packEvInputs(core, {
        ...f,
        hidden: { drawDist: Array.from({ length: KMAX + 1 }, () => new Float64Array(34)) },
      }),
    Error,
    "drawDist",
  );
  assertThrows(
    () => packEvInputs(core, { ...f, hidden: { pool: new Float64Array(33) } }),
    Error,
    "pool",
  );
});

// ---------------------------------------------------------------------------
// 4. the Observation-side facts
// ---------------------------------------------------------------------------

Deno.test("evpack: 赤5筒は ID で数える — 手の中と、見えている場所すべて", () => {
  const half = policyHalf();
  // Neither aka is anywhere: held 0, unseen 2.
  const none = evFactsFromObservation(baseObs(), { mode: 0, ...half });
  assertEquals([none.akaHeld, none.akaUnseen], [0, 2]);

  // Both in hand: held 2, and NOTHING is unseen — "unseen" is unseen BY US.
  const both = evFactsFromObservation(
    baseObs({ hand: tiles("123456789m00p東東東") }),
    { mode: 0, ...half },
  );
  assertEquals([both.akaHeld, both.akaUnseen], [2, 0]);

  // One held, one in an opponent's river.
  const aka = tiles("0p0p"); // ids 52, 53
  const rivers = openingRivers();
  rivers[2] = river([aka[1]]);
  const split = evFactsFromObservation(
    baseObs({ hand: tiles("123456789m0p2p東東"), rivers }),
    { mode: 0, ...half },
  );
  assertEquals([split.akaHeld, split.akaUnseen], [1, 0]);

  // One in an opponent's MELD, none held.
  const melds: Meld[][] = [[], [], [], []];
  melds[1] = [{ kind: "pon", who: 1, fromWho: 0, tiles: [aka[0], 54, 55], calledTile: aka[0] }];
  const inMeld = evFactsFromObservation(baseObs({ melds }), { mode: 0, ...half });
  assertEquals([inMeld.akaHeld, inMeld.akaUnseen], [0, 1]);

  // …and as a dora INDICATOR.
  const asInd = evFactsFromObservation(
    baseObs({ doraIndicators: [aka[0]] }),
    { mode: 0, ...half },
  );
  assertEquals([asInd.akaHeld, asInd.akaUnseen], [0, 1]);
});

Deno.test("evpack: 赤5筒は自分の副露の中でも「持っている」 (score.ts と同じ勘定)", () => {
  const half = policyHalf();
  const aka = tiles("0p0p"); // ids 52, 53
  // A 5p pon of ours holding one aka, the other still in hand. `score.ts`
  // counts hand + melds, so this hand is +2 han of aka whatever it discards —
  // and a call comparison that dropped the melded one would price the call
  // below the pass for a reason that is not true. (2026-08-30 review.)
  const melds: Meld[][] = [[], [], [], []];
  melds[0] = [{ kind: "pon", who: 0, fromWho: 1, tiles: [aka[0], 54, 55], calledTile: 55 }];
  const f = evFactsFromObservation(
    baseObs({ hand: tiles("123456789m0p東"), melds }),
    { mode: 0, ...half },
  );
  assertEquals(f.akaHeld, 2, "副露の中の赤が数えられていない");
  // Both are visible to us, so neither is unseen — and the melded copy is
  // counted ONCE there, not twice.
  assertEquals(f.akaUnseen, 0);
});

Deno.test("evpack: 根が手牌を差し替えたら赤も根の牌で数える", () => {
  const half = policyHalf();
  const aka = tiles("0p0p");
  const hand = [...tiles("123456789m"), aka[0], aka[1], ...tiles("東")];
  const obs = baseObs({ hand });
  // The real root holds both.
  assertEquals(evFactsFromObservation(obs, { mode: 0, ...half }).akaHeld, 2);

  // A post-call root: the two aka went into the meld, the rest is what is left.
  // Aka are IDS, so the counts vector cannot answer this — `tiles` is what does.
  const rest = tiles("123456789m");
  const called: Meld = {
    kind: "pon",
    who: 0,
    fromWho: 1,
    tiles: [aka[0], aka[1], 54],
    calledTile: 54,
  };
  const post = evFactsFromObservation(obs, {
    mode: 0,
    hand: new Array<number>(34).fill(0),
    tiles: rest,
    melds: [called],
    ...half,
  });
  assertEquals(post.akaHeld, 2, "鳴いた先の赤が数えられていない");
  // SEEN is a fact about the table and is read off the Observation alone: the
  // hypothetical root must not make a tile it double-counted go unseen.
  assertEquals(post.akaUnseen, 0);
});

Deno.test("evpack: 槓は「一面子」— 根の枚数は 14/13 − 3·面子数、槓の数は引かない", () => {
  const half = policyHalf();
  const core = scratch();
  const sum = (f: EvFacts) => {
    packEvInputs(core, f);
    let s = 0;
    for (let ty = 0; ty < 34; ty++) s += core.ints[I_HAND + ty];
    return s;
  };
  // THE CONVENTION `native/mjev.cc` VALIDATES (`parseEval`: `sum != (mode == 0
  // ? 14 : 13) - 3 * nMelds`), pinned from the packer's side so a change to
  // either file has to face this test. A kan's fourth tile is paid for by the
  // rinshan draw, so the concealed count after an 暗槓 is what it is after a
  // pon — 11 at a discard root, 10 at rest — and subtracting `kans` on top
  // would refuse every kan hand the engine actually produces. (2026-08-30
  // review: the TypeScript root test now states this equality exactly, in
  // place of `hand.length % 3 === 2`.)
  const melds: Meld[][] = [[], [], [], []];
  melds[0] = [meld("ankan", "5555s", 0, 0)];
  const discardRoot = baseObs({ hand: tiles("123456789m11p"), melds });
  assertEquals(discardRoot.hand.length, 11, "暗槓後の打牌根は 14 − 3·1 枚");
  assertEquals(sum(evFactsFromObservation(discardRoot, { mode: 0, ...half })), 14 - 3 * 1);

  const restRoot = baseObs({ hand: tiles("123456789m1p"), melds });
  assertEquals(sum(evFactsFromObservation(restRoot, { mode: 1, ...half })), 13 - 3 * 1);

  // Two kans, same rule.
  melds[0] = [meld("ankan", "5555s", 0, 0), meld("daiminkan", "2222p")];
  const two = baseObs({ hand: tiles("12345678m"), melds });
  assertEquals(sum(evFactsFromObservation(two, { mode: 0, ...half })), 14 - 3 * 2);
});

Deno.test("evpack: 自分の河は袋 — 鳴かれた牌も振聴の証拠として残る", () => {
  const half = policyHalf();
  const rivers = openingRivers();
  // A 東 of ours that somebody pon'd away. `publicUnseen` skips it (the meld
  // counted the copy); the furiten bag must NOT.
  rivers[0] = [...rivers[0], ...river(tiles("東"), 2)];
  const melds: Meld[][] = [[], [], [], []];
  melds[2] = [meld("pon", "東東東", 2, 0)];
  const f = evFactsFromObservation(baseObs({ rivers, melds }), { mode: 0, ...half });
  assertEquals(f.ownRiverBag[KITA], 1);
  assertEquals(f.ownRiverBag[SHA], 1);
  assertEquals(f.ownRiverBag[TON], 1, "鳴かれた自分の打牌も自河に残る");
  // Nobody else's river reaches the bag.
  assertEquals(Array.from(f.ownRiverBag).reduce((a, b) => a + b, 0), 3);
});

Deno.test("evpack: 残り自摸 T は wallRemaining/4 の切り捨て、20 で頭打ち", () => {
  const half = policyHalf();
  const at = (wall: number) =>
    evFactsFromObservation(baseObs({ wallRemaining: wall }), { mode: 0, ...half }).T;
  assertEquals(at(58), 14);
  assertEquals(at(7), 1);
  assertEquals(at(3), 0);
  assertEquals(at(0), 0);
  assertEquals(at(200), T_CAP, "手番の地平は 20 で頭打ち");
  // An explicit override wins (the call comparisons price the post-call hand at
  // the SAME horizon).
  assertEquals(
    evFactsFromObservation(baseObs(), { mode: 1, T: 3, ...half }).T,
    3,
  );
});

Deno.test("evpack: candMask は根が持つ型すべて — 絞り込みは argmax 側の仕事", () => {
  const half = policyHalf();
  const obs = baseObs({ hand: tiles("123456789m1122p東") });
  const f = evFactsFromObservation(obs, { mode: 0, ...half });
  const held = new Set<number>();
  for (let ty = 0; ty < 34; ty++) if (f.hand[ty] > 0) held.add(ty);
  for (let ty = 0; ty < 34; ty++) {
    assertEquals(f.candMask[ty], held.has(ty) ? 1 : 0, `型 ${ty}`);
  }
  // 12 distinct types in 123456789m 1122p 東.
  assertEquals(held.size, 12);
  // A rest root prices no candidate at all.
  const rest = evFactsFromObservation(obs, { mode: 1, ...half });
  assertEquals(Array.from(rest.candMask), new Array(34).fill(0));
});

Deno.test("evpack: 面子は {種別, 最小型, 暗刻フラグ} に符号化される", () => {
  const core = scratch();
  const half = policyHalf();
  const melds: Meld[][] = [[], [], [], []];
  melds[0] = [meld("chi", "678s"), meld("daiminkan", "222p")];
  const f = evFactsFromObservation(baseObs({ melds }), { mode: 1, ...half });
  packEvInputs(core, f);
  assertEquals(core.ints[I_NMELDS], 2);
  // 678s: run, lowest type 6s = 18+5 = 23, open.
  assertEquals([...core.ints.slice(I_MELDS, I_MELDS + 3)], [0, 23, 0]);
  // 222p: kan, type 2p = 9+1 = 10, open (a 大明槓 is not concealed).
  assertEquals([...core.ints.slice(I_MELDS + 3, I_MELDS + 6)], [2, 10, 0]);
  // The unused slots stay zero.
  assertEquals([...core.ints.slice(I_MELDS + 6, I_MELDS + 12)], [0, 0, 0, 0, 0, 0]);
  assertEquals(core.ints[I_CLOSED], 0, "チーは門前を壊す");

  // An 暗槓 alone keeps the hand menzen and sets the concealed flag.
  melds[0] = [meld("ankan", "5555s")];
  const ankan = evFactsFromObservation(baseObs({ melds }), { mode: 1, ...half });
  packEvInputs(core, ankan);
  assertEquals([...core.ints.slice(I_MELDS, I_MELDS + 3)], [2, 22, 1]);
  assertEquals(core.ints[I_CLOSED], 1);
});

Deno.test("evpack: 場況 — ドラは重複度、不可視数は publicUnseen、親と振聴", () => {
  const half = policyHalf();
  const core = scratch();
  const obs = baseObs({
    hand: tiles("123456789m1122p東"),
    doraIndicators: tiles("9s9s"), // both name 1s (type 18)
    honba: 3,
    kyotaku: 1,
    junme: 7,
    seatWind: 28, // 南 ⇒ not the dealer
    roundWind: 27,
    riichi: [true, false, false, false],
    furiten: { permanent: false, temporary: false, riichi: true },
  });
  const f = evFactsFromObservation(obs, { mode: 0, ...half });
  packEvInputs(core, f);
  assertEquals(core.ints[I_DORA + 18], 2, "同じ型を指す表示牌が二枚なら ドラ2");
  assertEquals(core.ints[I_DORA + 26], 0, "表示牌そのものはドラではない");
  // 北 ×4 sit in the four opening rivers; the 9s indicators are visible too.
  assertEquals(core.ints[I_UNSEEN + KITA], 0);
  assertEquals(core.ints[I_UNSEEN + 26], 2, "9s: 4 − 表示牌2");
  assertEquals(core.ints[I_UNSEEN + TON], 3, "東: 4 − 手の1枚");
  assertEquals(core.ints[I_DEALER], 0);
  assertEquals(core.ints[I_SEAT_WIND], 28);
  assertEquals(core.ints[I_ROUND_WIND], 27);
  assertEquals(core.ints[I_HONBA], 3);
  assertEquals(core.ints[I_KYOTAKU], 1);
  assertEquals(core.ints[I_JUNME], 7);
  assertEquals(core.ints[I_OWN_RIICHI], 1);
  assertEquals(core.ints[I_FURITEN_PERM], 0);
  assertEquals(core.ints[I_FURITEN_TEMP], 1, "リーチ後の見逃しは一時振聴に畳む");
  assertEquals(core.ints[I_MODE], 0);
  assertEquals(core.ints[I_KANDORA_ON], 0);

  // 東 as the seat wind IS the dealer.
  const oya = evFactsFromObservation(baseObs({ seatWind: 27 }), { mode: 0, ...half });
  assertEquals(oya.dealer, true);
});

Deno.test("evpack: 政策側の値はそのまま渡る — hand 上書きと kanDoraOn を含む", () => {
  const core = scratch();
  const pIn = Array.from({ length: 34 }, (_, i) => (i === 30 ? 0.4 : 0));
  const costIn = Array.from({ length: 34 }, (_, i) => (i === 30 ? 6400 : 0));
  // The post-ankan rest root: a hand the Observation does not hold.
  const post = new Array<number>(34).fill(0);
  post[0] = 3;
  post[9] = 2;
  const f = evFactsFromObservation(baseObs(), {
    mode: 1,
    hand: post,
    kanDoraOn: true,
    pIn,
    costIn,
    tenpaiP: [0.5, 0.2, 0.05],
    expLoss: [5200, 3900, 1000],
    gain: 1.4,
    risk: 0.6,
  });
  packEvInputs(core, f);
  assertEquals(core.ints[I_HAND + 0], 3);
  assertEquals(core.ints[I_HAND + 9], 2);
  assertEquals(core.ints[I_MODE], 1);
  assertEquals(core.ints[I_KANDORA_ON], 1);
  assertEquals([...core.ints.slice(I_CAND, I_CAND + 34)], new Array(34).fill(0));
  assertEquals([...core.dbls.slice(D_TENPAI, D_TENPAI + 3)], [0.5, 0.2, 0.05]);
  assertEquals([...core.dbls.slice(D_EXPLOSS, D_EXPLOSS + 3)], [5200, 3900, 1000]);
  assertEquals(core.dbls[D_PIN + 30], 0.4);
  assertEquals(core.dbls[D_COSTIN + 30], 6400);
  assertEquals(core.dbls[D_GAIN], 1.4);
  assertEquals(core.dbls[D_RISK], 0.6);
});

// ---------------------------------------------------------------------------
// 5. the indicator ↔ dora inverse
// ---------------------------------------------------------------------------

Deno.test("evpack: indicatorOfDora は doraFromIndicatorType の逆写像", () => {
  // Stated as a round trip over ALL 34 types rather than as a table: the two
  // functions live in different repositories, and a wrap-around edge (9→1,
  // 北→東, 中→白) is exactly the kind of thing a hand-written table gets right
  // in one direction only.
  for (let dora = 0; dora < 34; dora++) {
    const ind = indicatorOfDora(dora);
    assertEquals(doraFromIndicatorType(ind), dora, `ドラ ${dora} の表示牌 ${ind}`);
  }
  // The wraps, spelled out — a round trip alone would survive a consistent
  // off-by-one in both directions.
  assertEquals(indicatorOfDora(0), 8, "1m のドラ表示牌は 9m");
  assertEquals(indicatorOfDora(9), 17, "1p のドラ表示牌は 9p");
  assertEquals(indicatorOfDora(18), 26, "1s のドラ表示牌は 9s");
  assertEquals(indicatorOfDora(27), 30, "東のドラ表示牌は 北");
  assertEquals(indicatorOfDora(31), 33, "白のドラ表示牌は 中");
  assertEquals(indicatorOfDora(4), 3, "5m のドラ表示牌は 4m");
});
