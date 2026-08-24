// Feature encoding v4: an Observation → (48 tile planes, 42 scalars, token seq).
//
// The layout is a FROZEN contract shared with the Python/MLX trainer: the same
// bytes this module writes into a trajectory are what the trainer reshapes, so
// nothing here may be reordered without bumping `FEATURES.version` on both
// sides.
//
// Every plane is a 34-long stretch indexed by tile TYPE (0..33, aka fives fold
// onto their type — plane 5 is what keeps the red information). Per-seat planes
// and scalars keep the Observation's RELATIVE ordering (0 = self), which is
// what lets one network play all four seats.
//
// v2 over v1: planes 0–21 and scalars 0–32 are BYTE-IDENTICAL to v1; what is
// new is (a) the tile a claim decision is being offered (v1 could not tell pon
// from pass because it never saw the tile), (b) the oracle-derived per-discard
// facts the teacher heuristic consults — keeps-best shanten, 片和了り, yakuless
// — and (c) danger levels, the other seats' last discards, waits/ukeire, the
// aka outside our hand, the dora count and the dealer.
//
// v3 over v2: planes 0–35 and scalars 0–38 are BYTE-IDENTICAL to v2 — the v2
// vector is an exact PREFIX of the v3 one, which is what lets the trainer's
// surgery tool widen an existing first layer by copying the old columns and
// zero-filling the new ones ([1224..1631] and [1671..1673]). What is appended
// is the per-OPPONENT reading a defensive policy needs and v2 could only see
// aggregated: their meld composition, how much of their river was 手出し, the
// danger they individually pose, and the tile each one declared riichi on
// (plus that declaration's 巡目 as a scalar).
//
// NOTE ON CELL RANGE: v2 planes are strictly 0/1 indicator bits. Three of the
// v3 groups (p36–p44) are small COUNTS/LEVELS instead — still Int8, still in
// 0..4 — so a consumer that normalises must not assume bits beyond p35.
//
// v4 over v3: the planes and the scalars DO NOT MOVE — byte for byte, cell for
// cell, `encode` emits exactly what v3 emitted, and the frozen v3 digests are
// asserted unchanged. What v4 adds is a SECOND, differently-shaped view of the
// same table: `encodeSeq`, a token stream over the four rivers (see below) that
// an attention encoder folds into a 64-wide vector `z`, concatenated after the
// 1674 plane/scalar dims. Rivers reach the planes only as per-type counts, so
// the ORDER tiles were let go in — the single most-read signal at a real table
// — was invisible to v3 by construction; that order is what the token stream
// carries. The version bump exists for the seq alone.

import type { Tile } from "mjrender/model.ts";
import type { DangerLevel } from "mjrender/danger.ts";
import { doraFromIndicatorType, tileType } from "mjrender/tiles.ts";
import { countsFromTiles, shanten } from "../kernel.ts";
import type { Observation } from "../observe.ts";
import type { Table } from "../table.ts";
import type { Action, Seat } from "../types.ts";
import { SEATS } from "../types.ts";

export const FEATURES = {
  version: 4,
  planes: 48,
  scalars: 42,
  actions: 78,
} as const;

/** 34 tile types per plane. */
export const TYPES = 34;

/** Length of the flattened plane buffer (48 × 34). */
export const PLANE_LEN = FEATURES.planes * TYPES;

/** Flattened input width the network takes: planes ++ scalars. */
export const INPUT_LEN = PLANE_LEN + FEATURES.scalars;

// ---------------------------------------------------------------------------
// v4 token stream — the shapes `train/V4_SPEC.md` freezes
// ---------------------------------------------------------------------------

/** How many LEADING entries of each river become tokens. */
export const SEQ_RIVER_MAX = 24;

/** Bytes in one packed token: `[type, seatRel, idx, flags]`. */
export const SEQ_TOKEN_BYTES = 4;

/** Hard cap on the token count: 4 seats × 24 entries. */
export const SEQ_MAX = 4 * SEQ_RIVER_MAX;

/**
 * Width of one token after the dense expansion every forward does:
 * onehot34(type) ++ onehot4(seatRel) ++ idx/24 ++ tsumogiri ++ riichiDecl ++
 * calledAway = 34 + 4 + 1 + 3 = 42, in exactly that order.
 */
export const SEQ_DENSE = TYPES + 4 + 1 + 3;

/** Bit positions of the `flags` byte. */
export const SEQ_FLAG_TSUMOGIRI = 1;
export const SEQ_FLAG_RIICHI = 2;
export const SEQ_FLAG_CALLED = 4;

export interface Encoded {
  planes: Int8Array;
  scalars: Float32Array;
}

/**
 * One encoding, kept by a policy that already did the work — see
 * `EncodingCache`. `seq` is null when the policy had no use for the token
 * stream (a v3 net), never when it built one.
 */
export interface CachedEncoding {
  /** The very Observation these bytes were encoded from, for an identity check. */
  obs: Observation;
  planes: Int8Array;
  scalars: Float32Array;
  seq: Int8Array | null;
  /**
   * The support `decide` actually chose from — `obs.legal` after the dojo
   * compliance filter (`compliantActions`). This, not `obs.legal`, is what the
   * recorded `mask` must equal: the trainer's masked softmax has to reproduce
   * the distribution the action was sampled from, or the importance ratio it
   * computes is a ratio between two different distributions.
   */
  legal: Action[];
}

/**
 * A policy that offers its last encoding to whoever wraps it. `RecordingPolicy`
 * takes the offer when the Observation matches BY REFERENCE and re-encodes
 * otherwise, so wrapping a heuristic seat (which implements nothing here) keeps
 * working and the recorded bytes are identical either way.
 */
export interface EncodingCache {
  readonly lastEncoding: CachedEncoding | null;
}

/**
 * Shared 34-wide counter, refilled per call — `encode` is synchronous and each
 * of its six count passes is fully consumed before the next one starts, so one
 * buffer serves them all (the same scratch-reuse pattern `kernel.ts` documents).
 * The array is module-private for exactly that reason: a caller that held on to
 * a returned count would see it change under them.
 */
const COUNTS = new Uint8Array(TYPES);

function countTypes(ids: Iterable<Tile>): Uint8Array {
  COUNTS.fill(0);
  for (const id of ids) COUNTS[tileType(id)]++;
  return COUNTS;
}

/** Which of p26–p28 a danger level lights. 安全 lights none. */
const DANGER_PLANE: Partial<Record<DangerLevel, number>> = {
  "危険度低": 26,
  "危険度中": 27,
  "危険度高": 28,
};

/**
 * The ORDINAL a danger level takes in p42–p44 — the one place in the encoding
 * where a level is a magnitude rather than a bucket. Same order `danger.ts`
 * ranks them in: 安全 0 / 危険度低 1 / 危険度中 2 / 危険度高 3.
 */
const DANGER_RANK: Record<DangerLevel, number> = {
  "安全": 0,
  "危険度低": 1,
  "危険度中": 2,
  "危険度高": 3,
};

/**
 * The offered tile recovered from the call actions, for Observations that
 * predate `claimTile`. Every pon/chi/daiminkan on the table names the same
 * discarded tile in `called`, so the first one found is the answer; ron and
 * pass carry nothing, which is why the field itself is the primary source.
 */
function claimTileFromLegal(obs: Observation): Tile | null {
  for (const a of obs.legal) {
    if (a.t === "pon" || a.t === "chi" || a.t === "daiminkan") return a.called;
  }
  return null;
}

/**
 * Encode one observation.
 *
 * Plane map (each entry is one 34-long stretch):
 *   0–3   own hand (drawn tile included) count ≥1 / ≥2 / ≥3 / ≥4
 *   4     the drawn tile's type (all zero when it is not our draw)
 *   5     types of the aka fives we hold
 *   6–13  rivers, two planes per relative seat r: count ≥1 then ≥2
 *   14–17 tile types appearing in relative seat r's melds
 *   18    dora types;  19  dora indicator types
 *   20    visible-to-me count ≥3;  21  visible count = 4
 *   22    the tile a claim is being offered (all zero on a turn decision)
 *   23–25 last river entry of relative seat 1 / 2 / 3
 *   26–28 danger level 低 / 中 / 高 (安全 lights nothing)
 *   29    types we may legally discard
 *   30    discards that leave the BEST shanten of all our discards
 *   31    discards that leave 片和了り;  32  discards that leave a yakuless tenpai
 *   33    our waits;  34  our ukeire types
 *   35    aka fives visible OUTSIDE our concealed hand (rivers + every meld)
 *   ---- v3 (everything below is per-OPPONENT, relative seats 1 / 2 / 3) ----
 *   36–38 their meld composition: COUNT 0–4 of each type over all meld tiles
 *   39–41 their 手出し river: COUNT 0–4 of each type discarded from hand
 *   42–44 the danger THEY pose per type, as a LEVEL 0–3 (安全/低/中/高)
 *   45–47 the tile they declared riichi on, one-hot (all zero when not riichi)
 *
 * "Visible" is the POLICY'S OWN view and deliberately approximate: own hand +
 * every river entry + every meld tile + the indicators, each tile id counted
 * once. It therefore misses tiles buried in other seats' hands and in the dead
 * wall, which is exactly the information the seat is not allowed to have.
 *
 * Planes 0–35 and scalars 0–38 are byte-identical to v2 — see the file header.
 */
export function encode(obs: Observation): Encoded {
  const planes = new Int8Array(PLANE_LEN);
  const set = (p: number, ty: number) => {
    planes[p * TYPES + ty] = 1;
  };
  /** v3 counting planes: one more copy, never past 4. */
  const inc = (p: number, ty: number) => {
    const i = p * TYPES + ty;
    if (planes[i] < 4) planes[i]++;
  };

  // --- p0–p3: own hand ---
  const hand = countTypes(obs.hand);
  for (let ty = 0; ty < TYPES; ty++) {
    for (let k = 0; k < 4; k++) if (hand[ty] > k) set(k, ty);
  }

  // --- p4: the drawn tile ---
  if (obs.drawn !== null) set(4, tileType(obs.drawn));

  // --- p5: aka held ---
  for (const id of obs.hand) if (obs.akaIds.has(id)) set(5, tileType(id));

  // --- p6–p13: rivers ---
  // Counted straight off the entries: a `.map(e => e.tile)` here would build
  // four throwaway arrays per encode, one per river, for nothing.
  for (let r = 0; r < 4; r++) {
    COUNTS.fill(0);
    for (const e of obs.rivers[r] ?? []) COUNTS[tileType(e.tile)]++;
    for (let ty = 0; ty < TYPES; ty++) {
      if (COUNTS[ty] >= 1) set(6 + 2 * r, ty);
      if (COUNTS[ty] >= 2) set(7 + 2 * r, ty);
    }
  }

  // --- p14–p17: melds ---
  for (let r = 0; r < 4; r++) {
    for (const m of obs.melds[r] ?? []) {
      for (const id of m.tiles) set(14 + r, tileType(id));
    }
  }

  // --- p18/p19: dora ---
  for (const ind of obs.doraIndicators) {
    const ity = tileType(ind);
    set(18, doraFromIndicatorType(ity));
    set(19, ity);
  }

  // --- p20/p21: visible counts ---
  const seen = new Set<Tile>();
  for (const id of obs.hand) seen.add(id);
  for (let r = 0; r < 4; r++) {
    for (const e of obs.rivers[r] ?? []) seen.add(e.tile);
    for (const m of obs.melds[r] ?? []) for (const id of m.tiles) seen.add(id);
  }
  for (const id of obs.doraIndicators) seen.add(id);
  const vis = countTypes(seen);
  for (let ty = 0; ty < TYPES; ty++) {
    if (vis[ty] >= 3) set(20, ty);
    if (vis[ty] >= 4) set(21, ty);
  }

  // --- p22: the tile a claim is offered on ---
  // `claimTile` is what the driver hands us; the fallback reconstructs it from
  // the call actions themselves, so an Observation built without the field
  // (older fixtures, hand-built tests) still encodes the claim correctly.
  const claimed = obs.claimTile ?? claimTileFromLegal(obs);
  if (claimed !== null) set(22, tileType(claimed));

  // --- p23–p25: the other seats' last discards ---
  for (let r = 1; r < 4; r++) {
    const river = obs.rivers[r] ?? [];
    if (river.length > 0) set(22 + r, tileType(river[river.length - 1].tile));
  }

  // --- p26–p28: danger buckets (安全 lights nothing) ---
  for (const [ty, d] of obs.danger) {
    const p = DANGER_PLANE[d.level];
    if (p !== undefined) set(p, ty);
  }

  // --- p29: types we may legally discard ---
  for (const a of obs.legal) if (a.t === "discard") set(29, tileType(a.tile));

  // --- p30–p32: what each discard leaves behind ---
  let best = Infinity;
  for (const info of obs.discardInfo.values()) best = Math.min(best, info.shanten);
  for (const [tile, info] of obs.discardInfo) {
    const ty = tileType(tile);
    if (info.shanten === best) set(30, ty);
    if (info.katagari) set(31, ty);
    if (info.yakuless) set(32, ty);
  }

  // --- p33/p34: waits and ukeire ---
  for (const ty of obs.waits) set(33, ty);
  for (const u of obs.ukeire) set(34, u.type);

  // --- p35: aka seen outside our own concealed hand ---
  // Our own melds count as "outside": a red five in an ankan is public and is
  // no longer a tile we can choose to hold or cut.
  for (let r = 0; r < 4; r++) {
    for (const e of obs.rivers[r] ?? []) if (obs.akaIds.has(e.tile)) set(35, tileType(e.tile));
    for (const m of obs.melds[r] ?? []) {
      for (const id of m.tiles) if (obs.akaIds.has(id)) set(35, tileType(id));
    }
  }

  // -------------------------------------------------------------------------
  // v3: the per-opponent reading (relative seats 1/2/3 ⇒ planes +0/+1/+2)
  // -------------------------------------------------------------------------

  // --- p36–p38: what their melds are MADE OF ---
  // Counts, not the bits p14–p17 carry: a pon of 5m and a kan of 5m are the
  // same plane cell in v2 and 3 vs 4 here. The called tile is included (it is
  // theirs now), and so is an ankan — its composition is announced even though
  // the hand stays closed.
  for (let r = 1; r < 4; r++) {
    for (const m of obs.melds[r] ?? []) {
      for (const id of m.tiles) inc(35 + r, tileType(id));
    }
  }

  // --- p39–p41: how much of their river was 手出し ---
  // A tedashi says the hand moved; a tsumogiri usually says it did not. Counted
  // per type and capped at 4, so the plane doubles as "they keep cutting these".
  for (let r = 1; r < 4; r++) {
    for (const e of obs.rivers[r] ?? []) if (!e.tsumogiri) inc(38 + r, tileType(e.tile));
  }

  // --- p42–p44: the danger each of them poses, per type ---
  // `DangerAssessment.details` carries one entry per threat seat that the tile
  // is NOT genbutsu against, with that seat's own level — genbutsu seats are
  // absent, which is exactly the 安全 (0) these planes default to. `seat` there
  // is ABSOLUTE (it comes from `Table.threats`/`furoThreats`), so it is folded
  // back into the relative order everything else here uses.
  for (const [ty, d] of obs.danger) {
    for (const det of d.details) {
      const rel = ((det.seat - obs.seat) % 4 + 4) % 4;
      if (rel === 0) continue; // a threat is never the observing seat
      const i = (41 + rel) * TYPES + ty;
      const rank = DANGER_RANK[det.level] ?? 0;
      if (rank > planes[i]) planes[i] = rank;
    }
  }

  // --- p45–p47: the tile each of them declared riichi on ---
  // The declaration is marked on the river entry itself (`riichiDeclare`, the
  // sideways tile), so the plane stays dark for a seat that never declared.
  for (let r = 1; r < 4; r++) {
    for (const e of obs.rivers[r] ?? []) {
      if (e.riichiDeclare) {
        set(44 + r, tileType(e.tile));
        break;
      }
    }
  }

  // -------------------------------------------------------------------------
  // scalars
  // -------------------------------------------------------------------------
  const s = new Float32Array(FEATURES.scalars);
  for (let r = 0; r < 4; r++) {
    s[0 + r] = (obs.scores[r] ?? 0) / 25000;
    s[4 + r] = obs.riichi[r] ? 1 : 0;
    const rj = obs.riichiJunme[r] ?? -1;
    s[8 + r] = rj < 0 ? 0 : rj / 18;
  }
  s[12] = obs.junme / 18;
  s[13] = obs.wallRemaining / 70;
  s[14] = obs.kyoku / 8;
  s[15] = Math.min(obs.honba, 8) / 8;
  s[16] = Math.min(obs.kyotaku, 4) / 4;
  s[17] = obs.roundWind === 27 ? 1 : 0;
  for (let w = 0; w < 4; w++) s[18 + w] = obs.seatWind === 27 + w ? 1 : 0;
  s[22] = Math.max(0, obs.shanten) / 8;
  s[23] = obs.furiten.permanent ? 1 : 0;
  s[24] = obs.furiten.temporary ? 1 : 0;
  s[25] = obs.furiten.riichi ? 1 : 0;
  s[26] = obs.tsumogiriLock ? 1 : 0;
  for (let r = 0; r < 4; r++) s[27 + r] = Math.min(obs.violations[r] ?? 0, 4) / 4;
  // 門前: an ankan does not open the hand, so a hand of nothing but ankan is
  // still menzen — and so is a hand with no melds at all (vacuously true).
  s[31] = obs.melds[0].every((m) => m.kind === "ankan") ? 1 : 0;
  s[32] = obs.melds[0].length / 4;
  s[33] = Math.min(obs.doraCount, 8) / 8;
  // 親 as a RELATIVE seat, like everything else here: 起家 rotates with kyoku.
  s[34 + ((obs.kyoku % 4) - obs.seat + 4) % 4] = 1;
  s[38] = obs.legal.some((a) => a.t === "pass") ? 1 : 0;

  // v3 s39–s41: WHEN each opponent declared, normalised like every other 巡目.
  // This repeats s9–s11 by design — the v3 block is appended, never a reorder
  // of the v2 prefix, so the per-opponent group is contiguous with p45–p47 it
  // pairs with (the tile) rather than being spliced into the old seat block.
  for (let r = 1; r < 4; r++) {
    const rj = obs.riichiJunme[r] ?? -1;
    s[38 + r] = rj < 0 ? 0 : rj / 18;
  }

  return { planes, scalars: s };
}

// ---------------------------------------------------------------------------
// v4: the river token stream
// ---------------------------------------------------------------------------

/**
 * The four rivers as one PACKED token stream — the "seq" half of feature v4.
 *
 * One token per discard, ALL FOUR seats including our own, `SEQ_TOKEN_BYTES`
 * Int8 each:
 *
 *     [type, seatRel, idx, flags]
 *
 * Order is seatRel 0 (self) first, then 1, 2, 3 — the same relative ordering
 * every plane and scalar uses — and chronological inside each river, which is
 * the whole point: the planes see a river as a bag of types, and this sees it
 * as the sequence it actually was. Each river contributes its FIRST
 * `SEQ_RIVER_MAX` (24) entries, so the stream is at most `SEQ_MAX` (96) tokens
 * long and the encoder never has to pad. Truncation drops the LATEST discards,
 * not the earliest — a river that long is deep into 終盤, where the opening
 * shape is what still reads.
 *
 * `idx` is the 0-based position inside that seat's own river (0..23), which is
 * what tells 第一打 from 十打目 once the tokens are shuffled together by
 * attention; it is expanded as `idx / 24`, never as a one-hot.
 *
 * The three flags are read straight off the `RiverEntry` the board keeps —
 * nothing here is re-derived:
 *   bit0 `tsumogiri`     — the entry's own 手出し/ツモ切り mark
 *   bit1 `riichiDeclare` — the sideways declaration tile (planes 45–47 read the
 *                          same field, so the two can never disagree)
 *   bit2 `calledBy !== undefined` — the tile was claimed into someone's meld.
 *
 * The called-away bit deserves a word: `BoardState.applyMeld` writes `calledBy`
 * onto the discarder's last river entry when a chi/pon/daiminkan takes it, so
 * the mark is already on the entry and does NOT have to be recovered by
 * matching meld `calledTile` ids against rivers. It matters because a called
 * tile is a discard that never sat in the river — reading the stream without it
 * would count a tile as passed-safe that was in fact scooped up immediately.
 *
 * Returns an Int8Array of exactly `4 × L` bytes for `L` tokens, and a
 * ZERO-length array before anyone has discarded (the `L = 0` case every forward
 * implementation must special-case as `z = bz`).
 *
 * It is a VIEW onto a buffer allocated by this call — `subarray`, not `slice`,
 * so the tail is not copied — and that buffer belongs to the returned view
 * alone. Nothing here is shared between calls, so a caller may keep the result
 * for as long as it likes; it only has to respect `byteOffset`/`byteLength`
 * (`length` and the FFI's pointer-of-view both already do).
 */
export function encodeSeq(obs: Observation): Int8Array {
  const buf = new Int8Array(SEQ_MAX * SEQ_TOKEN_BYTES);
  let n = 0;
  for (let r = 0; r < 4; r++) {
    const river = obs.rivers[r] ?? [];
    const take = Math.min(river.length, SEQ_RIVER_MAX);
    for (let i = 0; i < take; i++) {
      const e = river[i];
      let flags = 0;
      if (e.tsumogiri) flags |= SEQ_FLAG_TSUMOGIRI;
      if (e.riichiDeclare) flags |= SEQ_FLAG_RIICHI;
      if (e.calledBy !== undefined) flags |= SEQ_FLAG_CALLED;
      buf[n++] = tileType(e.tile);
      buf[n++] = r;
      buf[n++] = i;
      buf[n++] = flags;
    }
  }
  return buf.subarray(0, n);
}

/**
 * One packed token expanded to the `SEQ_DENSE` floats the encoder multiplies.
 * Shared by the TS attention forward and by the tests that pin the layout, so
 * there is exactly one place the field order lives.
 *
 * A field outside its documented range sets NO bit and contributes nothing —
 * the spec's rule, and the same one `rlnet_attn_encode` follows. `encodeSeq`
 * cannot produce such a token; a corrupt "seq" replayed from a trajectory file
 * can, and it must degrade rather than write past its own slot (an unguarded
 * `out[type] = 1` with type = 40 would land on the flag dims).
 *
 * `idx / SEQ_RIVER_MAX` is computed in double and rounded once, by the store
 * into the Float32Array — which is exactly what the C side spells out.
 */
export function expandToken(tokens: Int8Array, i: number, out: Float32Array): void {
  out.fill(0);
  const b = i * SEQ_TOKEN_BYTES;
  const type = tokens[b], seatRel = tokens[b + 1], idx = tokens[b + 2], flags = tokens[b + 3];
  if (type >= 0 && type < TYPES) out[type] = 1; // onehot34(type)
  if (seatRel >= 0 && seatRel < 4) out[TYPES + seatRel] = 1; // onehot4(seatRel)
  if (idx >= 0 && idx < SEQ_RIVER_MAX) out[TYPES + 4] = idx / SEQ_RIVER_MAX;
  out[TYPES + 5] = flags & SEQ_FLAG_TSUMOGIRI ? 1 : 0;
  out[TYPES + 6] = flags & SEQ_FLAG_RIICHI ? 1 : 0;
  out[TYPES + 7] = flags & SEQ_FLAG_CALLED ? 1 : 0;
}

// ---------------------------------------------------------------------------
// oracle (hidden-state) features — NOT part of the policy input
// ---------------------------------------------------------------------------

/** Oracle plane count (5 × 34 = 170 Int8 cells). */
export const ORACLE_PLANES = 5;
/** Length of the flattened oracle plane buffer. */
export const ORACLE_LEN = ORACLE_PLANES * TYPES;

export interface EncodedOracle {
  oplanes: Int8Array;
  /** Standard shanten of the three opponents, same relative order as the planes. */
  oppShanten: number[];
}

/**
 * The hidden state, for an ASYMMETRIC critic: the value head may read this at
 * training time, the policy never does. Nothing here touches `Observation`, so
 * `encode`'s frozen v2 contract is untouched — a trainer that ignores these
 * bytes trains exactly the network it trained before.
 *
 * Oracle plane map (each entry is one 34-long stretch of COUNTS, 0–4, not the
 * one-hot thresholds `encode` uses):
 *   0–2  concealed hand of relative seat 1 / 2 / 3 (seat+1, seat+2, seat+3).
 *        Concealed only — melds are public and already in the policy's p14–p17.
 *   3    hidden remainder: 4 minus every copy the oracle can place somewhere,
 *        i.e. what is still in the live wall or the unseen dead wall (ura
 *        indicators included, since they are dead-wall tiles nobody has seen).
 *   4    ura-dora indicator counts (all of them; the trainer decides which of
 *        them matter, e.g. only the ones a riichi could ever cash).
 *
 * The zones summed for plane 3 are all four concealed hands, every UNCALLED
 * river entry (a called tile lives in the meld that took it, and counting both
 * would double-count it), every meld tile, and the revealed dora indicators —
 * the same partition `Table.visibleCounts` uses, minus its per-seat blindness.
 */
export function encodeOracle(t: Table, seat: Seat): EncodedOracle {
  const oplanes = new Int8Array(ORACLE_LEN);
  const oppShanten: number[] = [];

  // --- o0–o2: the opponents' concealed hands, in relative order ---
  for (let r = 1; r < 4; r++) {
    const s = ((seat + r) % 4) as Seat;
    const counts = countsFromTiles(t.hands[s]);
    for (let ty = 0; ty < TYPES; ty++) oplanes[(r - 1) * TYPES + ty] = counts[ty];
    // Exactly observe.ts's convention: called melds fill set slots, an ankan
    // keeps the hand closed (so chiitoi/kokushi stay reachable).
    oppShanten.push(shanten(counts, t.melds[s].length, t.isMenzen(s)));
  }

  // --- o3: what neither the hands nor the table can account for ---
  const placed: Tile[] = [];
  for (const s of SEATS) {
    placed.push(...t.hands[s]);
    for (const e of t.board.rivers[s]) if (e.calledBy === undefined) placed.push(e.tile);
    for (const m of t.melds[s]) placed.push(...m.tiles);
  }
  placed.push(...t.indicators);
  const known = countsFromTiles(placed);
  for (let ty = 0; ty < TYPES; ty++) oplanes[3 * TYPES + ty] = 4 - known[ty];

  // --- o4: the ura indicators ---
  const ura = countsFromTiles(t.wall.uraIndicators());
  for (let ty = 0; ty < TYPES; ty++) oplanes[4 * TYPES + ty] = ura[ty];

  return { oplanes, oppShanten };
}

/**
 * planes ++ scalars as one Float32Array — the network's input vector.
 *
 * `out`, when given, is filled IN PLACE and returned: it must be at least
 * `INPUT_LEN` long and may be LONGER, in which case the tail past `INPUT_LEN`
 * is left exactly as it was. That is what lets `NeuralPolicy` keep one
 * `SEQ_INPUT_LEN` buffer for every decision and have the river encoder write
 * `z` into the tail. Without `out` a fresh `INPUT_LEN` vector is returned, as
 * always.
 *
 * The plane half is an Int8→Float32 `set`, i.e. the per-element numeric
 * conversion the spec defines — same values as the old cell-by-cell loop, since
 * every Int8 is exact in float32.
 */
export function flatten(e: Encoded, out?: Float32Array): Float32Array {
  const dst = out ?? new Float32Array(INPUT_LEN);
  dst.set(e.planes, 0);
  dst.set(e.scalars, PLANE_LEN);
  return dst;
}
