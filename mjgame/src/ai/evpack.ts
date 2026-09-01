// M15 — the packer: TypeScript facts → the `ints`/`dbls` wire of `libmjev`.
//
// PURE TS, NO FFI. Nothing here dlopens anything — it does not even import
// `ev.ts`: what it writes into is `EvWire`, the two reused buffers, which an
// `EvCore` satisfies structurally. That is deliberate: the wire layout is the
// specification of what the DP sees (`evlayout.ts`), so it has to be checkable
// on a machine with no dylib built —
// `test/evpack_test.ts` asserts every offset against hand-built Observations.
//
// TWO LAYERS, on purpose:
//
//   `EvFacts` is plain data. It names exactly what crosses, in TS terms, and
//   knows nothing about an Observation, a policy or a read. A unit test can
//   build one by hand and compare the packed arrays element by element.
//
//   `evFactsFromObservation` derives the OBSERVATION half of that record —
//   the counting facts, which are the same for every seat that looks at the
//   same table. The POLICY half (`pIn`/`costIn`/`tenpaiP`/`expLoss`/
//   `gain`/`risk`, and the hidden-information channels) is passed in, because
//   it is doctrine: `costIn` is the `riskOf` arithmetic in points and stays in
//   TypeScript where the dojo rulings live (plan §1, "packed by TS so the
//   doctrine stays in TS").
//
// `candMask` is every type the root HOLDS — not every type it is allowed to
// let go. The compliance filter, `guardTriplets` and the 禁じ手 vetoes apply to
// the ARGMAX over the returned prices, not to the facts: a price the seat then
// refuses to take is still information (it is what the fold line is compared
// against), and a fact vector that quietly encoded a doctrine filter would make
// the C++ side a second, invisible home for the rules.

import type { Meld, Tile } from "mjrender/model.ts";
import { doraFromIndicatorType, tileType } from "mjrender/tiles.ts";
import type { Observation } from "../observe.ts";
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
  KMAX,
} from "./evlayout.ts";
import { publicUnseen } from "./planner.ts";

/** The DP's horizon cap (`planner.ts:108`, `handvalue.ts:179`). */
export const T_CAP = 20;

/**
 * The dora INDICATOR whose flip names `type` as a dora — the exact inverse of
 * mjrender's `doraFromIndicatorType`.
 *
 * The hidden-information channels speak in INDICATOR types (that is what a
 * wall slot holds and what the counting expectation over `unseen` is written
 * against), while a read that knows the future speaks in DORA types
 * (`Reads.nextDora` is the tile that becomes valuable). One of the two has to
 * convert, and it is this side: the wire layout is the specification, and the
 * specification says indicator.
 */
export function indicatorOfDora(type: number): number {
  if (type < 27) {
    const base = type < 9 ? 0 : type < 18 ? 9 : 18;
    const r = type - base; // 0..8 ⇒ rank 1..9
    return base + (r === 0 ? 8 : r - 1);
  }
  // 東南西北 cycle among 27..30, 白發中 among 31..33.
  if (type <= 30) return type === 27 ? 30 : type - 1;
  return type === 31 ? 33 : type - 1;
}

/**
 * The optional hidden-information overrides (plan D7). Every channel absent is
 * the computed seat: the DP's posterior is the uniform unseen pool. A channel
 * PRESENT replaces that posterior exactly where it enters, and its flag is set
 * so absence stays distinguishable from a uniform vector.
 *
 * The engine never learns which producer filled these — the oracle's one-hots
 * and a future learned module's soft distributions arrive through the same
 * fields.
 */
export interface EvHidden {
  /** The next `K ≤ KMAX` own draws, each a distribution over 34 types. */
  drawDist?: readonly Float64Array[];
  /** Live-wall composition, in place of `publicUnseen`. */
  pool?: Float64Array;
  /** Ura INDICATOR type distribution. */
  ura?: Float64Array;
  /** Next kan-dora INDICATOR type distribution. */
  nextDora?: Float64Array;
}

/**
 * One evaluation's inputs, as plain data. Field-for-field the `ints`/`dbls`
 * layout of `evlayout.ts`, in TypeScript spelling.
 *
 * Arrays are read, never retained: `packEvInputs` copies into the core's own
 * buffers, so a caller may reuse its scratch vectors freely.
 */
export interface EvFacts {
  /** counts[34] of the ROOT hand — 14 tiles when `mode` 0, 13 when `mode` 1; melds excluded. */
  hand: ArrayLike<number>;
  /** 0 = discard root (`mjev_eval_discard`), 1 = rest root (`mjev_eval_rest`). */
  mode: 0 | 1;
  /** The seat's own melds, in mjrender spelling; encoded to {kind, type, concealed}. */
  melds: readonly Meld[];
  seatWind: number;
  roundWind: number;
  dealer: boolean;
  honba: number;
  kyotaku: number;
  ownRiichi: boolean;
  furitenPerm: boolean;
  /** Temporary furiten — the riichi lock folds in here (both forbid a ron this go-around). */
  furitenTemp: boolean;
  junme: number;
  /** Own draws remaining, already clamped (`T_CAP`). */
  T: number;
  /** Red fives THIS ROOT holds — concealed part and own melds, as `score.ts` counts them. */
  akaHeld: number;
  /** Red fives nobody can see, off the OBSERVATION (never off a hypothetical root). */
  akaUnseen: number;
  /** Menzen: every meld (if any) is an 暗槓. */
  closed: boolean;
  /** 1 per type the root may be asked to price (discard roots). */
  candMask: ArrayLike<number>;
  /** The draw pool unless `hidden.pool` overrides it. */
  unseen: ArrayLike<number>;
  /** Dora MULTIPLICITY per type from the indicators (aka are ids, counted separately). */
  doraCount: ArrayLike<number>;
  /** Types in the seat's OWN river — the furiten proof. */
  ownRiverBag: ArrayLike<number>;
  /** A post-暗槓 rest root with a kan-dora reveal pending. */
  kanDoraOn: boolean;
  /** Per opponent, relative order (1..3 ⇒ index 0..2). */
  tenpaiP: ArrayLike<number>;
  expLoss: ArrayLike<number>;
  /** Σ_i P(opponent i rons this type NOW), per candidate type. */
  pIn: ArrayLike<number>;
  /** What letting this type go now costs in POINTS (the `riskOf` arithmetic). */
  costIn: ArrayLike<number>;
  /** 順位効用 multipliers; 1 when the layer is off. */
  gain: number;
  risk: number;
  hidden?: EvHidden | null;
}

function copy34(dst: Int32Array | Float64Array, at: number, src: ArrayLike<number>, what: string) {
  if (src.length < 34) throw new Error(`evpack: ${what} は34要素です: ${src.length}`);
  for (let i = 0; i < 34; i++) dst[at + i] = src[i];
}

/** A supplied hidden channel has to be a real 34-vector; the C++ side re-checks the mass. */
function copyDist(dst: Float64Array, at: number, src: ArrayLike<number>, what: string) {
  if (src.length !== 34) throw new Error(`evpack: ${what} は34要素の分布です: ${src.length}`);
  for (let i = 0; i < 34; i++) {
    const v = src[i];
    if (!Number.isFinite(v)) throw new Error(`evpack: ${what}[${i}] が有限でない: ${v}`);
    dst[at + i] = v;
  }
}

/** mjrender `MeldKind` → the wire's {0 run, 1 triplet, 2 kan}. */
function meldKindCode(m: Meld): number {
  switch (m.kind) {
    case "chi":
      return 0;
    case "pon":
      return 1;
    case "ankan":
    case "daiminkan":
    case "shouminkan":
      return 2;
    default:
      // `nuki` is sanma only and cannot occur at this table.
      throw new Error(`evpack: 面子の種類が扱えません: ${m.kind}`);
  }
}

/**
 * The two buffers `packEvInputs` writes into — everything it needs of an
 * `EvCore` and nothing else.
 *
 * NAMED SEPARATELY so a caller that has no core can still pack: M15b's
 * `--evcalib` lane records the wire of a seat that carries no `ev` block at
 * all (the lane must be the champion's own continuation), and the alternative
 * — building a core just to borrow its arrays — would dlopen the dylib inside
 * a seat whose whole claim is that it never touches the FFI. `EvCore`
 * satisfies this structurally, so nothing at the call sites changes.
 */
export interface EvWire {
  ints: Int32Array;
  dbls: Float64Array;
}

/**
 * Fill `core.ints` / `core.dbls` from `f`. Everything is zeroed first, so a
 * buffer reused across decisions can never leak a stale flag or a stale row of
 * a hidden channel into the next evaluation — the whole reason the buffers are
 * reused is that they are the ONLY per-decision allocation, and a partial fill
 * would be an invisible bug in a deterministic engine.
 */
export function packEvInputs(core: EvWire, f: EvFacts): void {
  const ints = core.ints;
  const dbls = core.dbls;
  ints.fill(0);
  dbls.fill(0);

  copy34(ints, I_HAND, f.hand, "hand");
  if (f.melds.length > 4) throw new Error(`evpack: 面子は4つまでです: ${f.melds.length}`);
  ints[I_NMELDS] = f.melds.length;
  f.melds.forEach((m, i) => {
    const at = I_MELDS + i * 3;
    ints[at] = meldKindCode(m);
    // `tiles` is sorted ascending, so the lowest TYPE names a run; for a
    // triplet/kan every tile is that type anyway.
    ints[at + 1] = Math.min(...m.tiles.map(tileType));
    ints[at + 2] = m.kind === "ankan" ? 1 : 0;
  });
  ints[I_SEAT_WIND] = f.seatWind;
  ints[I_ROUND_WIND] = f.roundWind;
  ints[I_DEALER] = f.dealer ? 1 : 0;
  ints[I_HONBA] = f.honba;
  ints[I_KYOTAKU] = f.kyotaku;
  ints[I_OWN_RIICHI] = f.ownRiichi ? 1 : 0;
  ints[I_FURITEN_PERM] = f.furitenPerm ? 1 : 0;
  ints[I_FURITEN_TEMP] = f.furitenTemp ? 1 : 0;
  ints[I_JUNME] = f.junme;
  ints[I_T] = f.T;
  ints[I_AKA_HELD] = f.akaHeld;
  ints[I_AKA_UNSEEN] = f.akaUnseen;
  ints[I_CLOSED] = f.closed ? 1 : 0;
  copy34(ints, I_CAND, f.candMask, "candMask");
  copy34(ints, I_UNSEEN, f.unseen, "unseen");
  copy34(ints, I_DORA, f.doraCount, "doraCount");
  copy34(ints, I_RIVER, f.ownRiverBag, "ownRiverBag");
  ints[I_KANDORA_ON] = f.kanDoraOn ? 1 : 0;
  ints[I_MODE] = f.mode;

  for (let i = 0; i < 3; i++) {
    dbls[D_TENPAI + i] = f.tenpaiP[i] ?? 0;
    dbls[D_EXPLOSS + i] = f.expLoss[i] ?? 0;
  }
  copy34(dbls, D_PIN, f.pIn, "pIn");
  copy34(dbls, D_COSTIN, f.costIn, "costIn");
  dbls[D_GAIN] = f.gain;
  dbls[D_RISK] = f.risk;

  const h = f.hidden;
  if (!h) return;
  if (h.drawDist && h.drawDist.length > 0) {
    const rows = h.drawDist;
    if (rows.length > KMAX) throw new Error(`evpack: drawDist は${KMAX}行までです: ${rows.length}`);
    rows.forEach((row, j) => copyDist(dbls, D_DRAW + j * 34, row, `drawDist[${j}]`));
    ints[I_HAS_DRAW] = 1;
    ints[I_K] = rows.length;
  }
  if (h.pool) {
    copyDist(dbls, D_POOL, h.pool, "pool");
    ints[I_HAS_POOL] = 1;
  }
  if (h.ura) {
    copyDist(dbls, D_URA, h.ura, "ura");
    ints[I_HAS_URA] = 1;
  }
  if (h.nextDora) {
    copyDist(dbls, D_NEXTDORA, h.nextDora, "nextDora");
    ints[I_HAS_NEXTDORA] = 1;
  }
}

/** The policy half of the facts — everything an Observation cannot answer. */
export interface EvFactOpts {
  mode: 0 | 1;
  /** Defaults to the seat's own melds (`obs.melds[0]`). */
  melds?: readonly Meld[];
  /** counts[34] override — the post-call / post-kan shape a call comparison prices. */
  hand?: ArrayLike<number>;
  /**
   * The root's own concealed TILE IDS, when `hand` overrides the Observation's.
   *
   * Aka are IDS, not types, so a counts vector cannot answer "how many red
   * fives does this root hold" — and a call spends tiles, so the answer for a
   * post-call root is genuinely different from the Observation's. Omitted ⇒
   * `obs.hand`, which is the right answer for the real root and the only one
   * available to a synthetic counts-only root.
   */
  tiles?: readonly Tile[];
  /** Horizon override; defaults to `min(T_CAP, floor(wallRemaining/4))`. */
  T?: number;
  /** A post-暗槓 rest root with a kan-dora reveal pending. */
  kanDoraOn?: boolean;
  hidden?: EvHidden | null;
  pIn: ArrayLike<number>;
  costIn: ArrayLike<number>;
  tenpaiP: ArrayLike<number>;
  expLoss: ArrayLike<number>;
  gain: number;
  risk: number;
}

/** Type counts of a tile list. */
function countsOf(ts: Iterable<number>): number[] {
  const c = new Array<number>(34).fill(0);
  for (const t of ts) c[tileType(t)]++;
  return c;
}

/**
 * The counting facts, straight off the Observation — the same accounting every
 * other consumer of "what is still out there" already goes through
 * (`publicUnseen`), so a type the planner, the reader and the DP all price is
 * priced off ONE unseen vector.
 */
export function evFactsFromObservation(obs: Observation, opts: EvFactOpts): EvFacts {
  const hand = opts.hand ?? countsOf(obs.hand);
  const melds = opts.melds ?? obs.melds[0];

  // Dora MULTIPLICITY, not the type set: two indicators naming the same type
  // are two dora. (`doraTypesOf` answers the set question; the DP wants the
  // count, exactly as `computed.ts:1284` builds it.)
  const doraCount = new Array<number>(34).fill(0);
  for (const ind of obs.doraIndicators) doraCount[doraFromIndicatorType(tileType(ind))]++;

  // The seat's OWN river, as a bag. A tile someone CALLED away still sits in
  // our discards for furiten purposes, so `calledBy` is not filtered here —
  // unlike `publicUnseen`, where the meld already counted the copy.
  const ownRiverBag = new Array<number>(34).fill(0);
  for (const e of obs.rivers[0]) ownRiverBag[tileType(e.tile)]++;

  // The candidates are the types the ROOT holds. Filters run on the argmax.
  const candMask = new Array<number>(34).fill(0);
  if (opts.mode === 0) { for (let ty = 0; ty < 34; ty++) if (hand[ty] > 0) candMask[ty] = 1; }

  // Aka are tile IDS, so they are counted by identity and never by type.
  //
  // HELD is the ROOT's holding, concealed part AND melds: an aka inside our own
  // 副露 is a han this hand keeps for the rest of the 局, exactly as
  // `score.ts` counts it (hand + melds), and a call comparison that dropped the
  // aka it just melded would price the call below the pass for a reason that is
  // not true. `opts.tiles` is what makes the post-call answer differ from the
  // Observation's; `melds` is already the ROOT's meld list (the hypothetical
  // one at a call comparison).
  let akaHeld = 0;
  for (const t of opts.tiles ?? obs.hand) if (obs.akaIds.has(t)) akaHeld++;
  for (const m of melds) for (const t of m.tiles) if (obs.akaIds.has(t)) akaHeld++;

  // SEEN is a fact about the TABLE and so is read off the Observation alone —
  // never off the root. (A hypothetical root double-counts the tile it called;
  // the unseen pool must not inherit that.)
  let akaSeen = 0;
  for (const t of obs.hand) if (obs.akaIds.has(t)) akaSeen++;
  for (const river of obs.rivers) for (const e of river) if (obs.akaIds.has(e.tile)) akaSeen++;
  for (const ms of obs.melds) {
    for (const m of ms) for (const t of m.tiles) if (obs.akaIds.has(t)) akaSeen++;
  }
  for (const t of obs.doraIndicators) if (obs.akaIds.has(t)) akaSeen++;
  const akaUnseen = Math.max(0, Math.min(2, obs.akaIds.size - akaSeen));

  return {
    hand,
    mode: opts.mode,
    melds,
    seatWind: obs.seatWind,
    roundWind: obs.roundWind,
    // 東 (type 27) as the seat wind IS the dealer's seat, by definition of the
    // relative-wind assignment.
    dealer: obs.seatWind === 27,
    honba: obs.honba,
    kyotaku: obs.kyotaku,
    ownRiichi: obs.riichi[0],
    furitenPerm: obs.furiten.permanent,
    // The riichi lock and the temporary pass forbid the same thing for the
    // rest of this go-around; the DP only asks "may we ron right now".
    furitenTemp: obs.furiten.temporary || obs.furiten.riichi,
    junme: obs.junme,
    T: opts.T ?? Math.min(T_CAP, Math.floor(obs.wallRemaining / 4)),
    akaHeld,
    akaUnseen,
    // An 暗槓 keeps the hand menzen; anything else opens it.
    closed: melds.every((m) => m.kind === "ankan"),
    candMask,
    unseen: publicUnseen(obs),
    doraCount,
    ownRiverBag,
    kanDoraOn: opts.kanDoraOn ?? false,
    tenpaiP: opts.tenpaiP,
    expLoss: opts.expLoss,
    pIn: opts.pIn,
    costIn: opts.costIn,
    gain: opts.gain,
    risk: opts.risk,
    hidden: opts.hidden ?? null,
  };
}
