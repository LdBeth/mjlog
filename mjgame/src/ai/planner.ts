// C7: the max-profit lock-on planner — predictions consumed as FUTURES, not as
// per-tile averages.
//
// The rest of the augmented policy prices single tiles (how dangerous, how
// useful) and lets an argmax over discards fall out. That is not how a human
// reads a hand. A human decides what the hand is going to BE — 「789三色、678p
// をポンして」 — and from then on the discards are bookkeeping. This module
// makes that explicit:
//
//   1. enumerate the concrete futures: every way the current tiles can be read
//      as 4 sets + a head, with the tiles still missing spelled out;
//   2. price each one — P(assembling the missing tiles | availability) × what
//      the finished hand pays;
//   3. hand back the ranked list. `AugmentedHeuristic` LOCKS ON to the maximum
//      and re-locks only when a challenger beats it by a clear margin.
//
// Two things make this more than efficiency with extra steps. First, CALL
// futures: a tile the opponents do not want is available through pon/chi, so a
// plan can price 副露 as a completion route — with the completed OPEN hand run
// through the real scorer, because an open future with no yaku is not a future,
// it is a 後付け trap. Second, availability is where hidden information enters:
// the same enumeration reads a true wall composition (C7O) or the public unseen
// counts (C7P), and nothing else about the planner changes. That is what makes
// the C7O/C7P pair an ablation of INFORMATION rather than of architecture.
//
// What this module does NOT do: it never looks at danger, never touches
// `dojoCost`, and never returns an action. It ranks futures. The safety
// machinery keeps its veto — a keep-set tile is expensive to cut, not illegal.

import type { Meld, Tile } from "mjrender/model.ts";
import { doraFromIndicatorType, rankOfType, suitOfType, tileType } from "mjrender/tiles.ts";
import type { Observation } from "../observe.ts";
import type { RuleConfig } from "../rules.ts";
import { JANKI } from "../rules.ts";
import { basePoints, ronPayment } from "../score.ts";
import { scoreWin } from "../yaku.ts";
import type { Reads } from "./augmented.ts";

// ---------------------------------------------------------------------------
// the public shape
// ---------------------------------------------------------------------------

/** One concrete completable final hand, priced. */
export interface TargetPlan {
  /**
   * Identity of the FINAL SHAPE — the sorted (kind, type) list of the four sets
   * and the head, called melds included. Deliberately blind to how far along the
   * hand is and to `open`, so that drawing a planned tile (or calling one) does
   * not change the plan's identity and silently break the lock: this is the key
   * the hysteresis in `relock` matches on.
   *
   * NOT in the spec's field list; the lock is unimplementable without it.
   */
  key: string;
  /** Tiles of the CURRENT hand this plan keeps. Everything else is spendable. */
  keep: Set<Tile>;
  /** Tile TYPES still needed, with multiplicity. The last one is the win tile. */
  required: number[];
  /** The subset of `required` a pon/chi could take (never the winning tile). */
  callable: number[];
  /** Does the plan involve (further) calls? Equivalently `callable.length > 0`. */
  open: boolean;
  /** Estimated points the finished hand pays on a ron, 本場 included. */
  value: number;
  /** P(assembling `required` in the draws this round still has). */
  pComplete: number;
  /** `pComplete * value` — the quantity the lock maximises. */
  profit: number;
  /** Human-readable, for the TUI and for debugging. */
  label: string;
}

/**
 * Where the missing tiles could come from. Both channels answer the same three
 * questions; only the answers differ, which is the whole point of the C7O/C7P
 * ablation pair.
 */
export interface Availability {
  /** Copies reachable by self-draw. */
  draw(ty: number): number;
  /** Copies likely to be shed by an opponent, i.e. claimable. */
  call(ty: number): number;
  /** Size of the pool `draw`/`call` are counted against. */
  poolSize(): number;
}

export interface PlannerOptions {
  /** Structures the block search may emit before it stops. */
  maxStructures?: number;
  /** Concrete futures that get the (expensive) real scorer run on them. */
  maxPlans?: number;
  /**
   * How far off the efficiency frontier a plan may sit: a future needing more
   * than `shanten + 1 + slack` tiles is a different hand, not a plan for this
   * one.
   */
  slack?: number;
  /** Ruleset for the hypothetical scorer (喰いタン, aka set, 数え役満). */
  cfg?: RuleConfig;
  /** Own draws left; defaults to `floor(wallRemaining / 4)`. */
  draws?: number;
}

/** Hysteresis margin: a challenger must beat the lock by this factor. */
export const RELOCK_MARGIN = 1.18;

/** Own draws still coming, by the crudest possible split of what is left. */
export function ownDraws(obs: Observation): number {
  return Math.floor(obs.wallRemaining / 4);
}

// ---------------------------------------------------------------------------
// availability
// ---------------------------------------------------------------------------

/**
 * Copies of each type NOT visible to this seat: 4 minus rivers, melds,
 * indicators and own hand.
 *
 * THE ONE UNSEEN COUNT. Every consumer of "how many of this type are still out
 * there" goes through this vector — the planner's availability model, the 計算
 * reader's wait survival, and the base policy's ukeire pricing (`Ctx.unseen`,
 * computed once per decision). It is the same accounting `Observation.ukeire`
 * carries per accepted type (`4 − visibleCounts`), so a type that appears in
 * both is priced identically whichever side asks.
 */
export function publicUnseen(obs: Observation): number[] {
  const c = new Array<number>(34).fill(4);
  for (const river of obs.rivers) {
    // A called tile is counted once, in the meld that took it.
    for (const e of river) if (e.calledBy === undefined) c[tileType(e.tile)]--;
  }
  for (const melds of obs.melds) {
    for (const m of melds) for (const t of m.tiles) c[tileType(t)]--;
  }
  for (const t of obs.doraIndicators) c[tileType(t)]--;
  for (const t of obs.hand) c[tileType(t)]--;
  for (let ty = 0; ty < 34; ty++) if (c[ty] < 0) c[ty] = 0;
  return c;
}

/**
 * The dora TYPES the revealed indicators name (aka are tile ids, not types, so
 * they are counted elsewhere). One definition, because three modules used to
 * spell the same `map`+`Set` out for themselves.
 */
export function doraTypesOf(obs: Observation): Set<number> {
  return new Set(obs.doraIndicators.map((t) => doraFromIndicatorType(tileType(t))));
}

/**
 * The honor types worth a yaku to a seat sitting at `seatWind` in `roundWind`:
 * the three dragons plus the two winds. Taken as loose numbers rather than an
 * `Observation` because the 計算 reader builds one of these per OPPONENT, from
 * a seat wind it derives itself.
 */
export function valueHonorsOf(roundWind: number, seatWind: number): Set<number> {
  return new Set([31, 32, 33, seatWind, roundWind]);
}

/**
 * C7O when `reads` carries a wall composition, C7P when it does not.
 *
 * C7O — `wallComposition` IS the draw distribution, so `draw` is exact and the
 * pool is the live wall itself. `call` is the crude half: an opponent holding a
 * copy might be building with it or about to shed it, and the planner has no
 * model of which, so it counts half of every held copy. That number is a
 * placeholder for the reader's eventual "will they discard this" head; it is
 * documented as crude rather than tuned, because tuning a proxy teaches the
 * planner to exploit the proxy.
 *
 * C7P — no hidden information at all: `draw` is the public unseen count (which
 * mixes the wall with the other three hands, exactly the ignorance the oracle
 * removes) and `call` is a quarter of it, on the same "someone holds it and
 * eventually lets it go" hand-wave. This arm exists to separate the planner's
 * value as an ARCHITECTURE from the value of the information it is fed.
 *
 * NOT modelled, deliberately: `call` does not distinguish pon from chi, so the
 * kamicha-only restriction on chi (which would cost roughly a factor of three)
 * is absent. A factor of three sits well inside the error bar of a 0.5/0.25
 * shed-probability guess, and pretending otherwise would dress the guess up as
 * a model.
 */
export function availabilityFrom(obs: Observation, reads: Reads | null): Availability {
  const wall = reads?.wallComposition;
  if (wall) {
    const opp = reads?.oppConcealed;
    let pool = 0;
    for (let ty = 0; ty < 34; ty++) pool += wall[ty];
    return {
      draw: (ty) => wall[ty],
      call: (ty) => {
        if (!opp) return 0;
        let held = 0;
        for (const o of opp) held += o[ty];
        return 0.5 * held;
      },
      poolSize: () => pool,
    };
  }
  const unseen = publicUnseen(obs);
  let pool = 0;
  for (let ty = 0; ty < 34; ty++) pool += unseen[ty];
  return {
    draw: (ty) => unseen[ty],
    call: (ty) => 0.25 * unseen[ty],
    poolSize: () => pool,
  };
}

/**
 * P(every required tile arrives), under the simplest model that ranks futures
 * consistently: each of `draws` own draws is one independent sample from a pool
 * of `poolSize()`, so a type with `a` copies reachable lands with probability
 * `1 - (1 - a/U)^draws`; the slots are treated as independent and the product is
 * damped by 0.85 per extra tile to pay for that independence being a lie (real
 * multi-tile completions compete for the same turns).
 *
 * Absolute calibration is not claimed and not needed — only the ORDER of the
 * futures is consumed.
 */
export function pCompleteOf(
  required: readonly number[],
  callable: Iterable<number>,
  avail: Availability,
  draws: number,
): number {
  if (required.length === 0) return 1;
  if (draws <= 0) return 0;
  const can = callable instanceof Set ? callable : new Set(callable);
  const u = Math.max(1, avail.poolSize());
  let p = 1;
  for (const ty of required) {
    let a = avail.draw(ty);
    if (can.has(ty)) a += avail.call(ty);
    if (a <= 0) return 0;
    p *= 1 - Math.pow(1 - Math.min(1, a / u), draws);
  }
  return p * Math.pow(0.85, required.length - 1);
}

// ---------------------------------------------------------------------------
// block structures
// ---------------------------------------------------------------------------

interface FinalBlock {
  kind: "run" | "triplet" | "pair";
  /** For a run, the LOWEST type of the three — the `decompose.ts` convention. */
  type: number;
}

interface NeedOption {
  /** The type that finishes this block. */
  type: number;
  block: FinalBlock;
  /** How an opponent's discard could supply it, or null (head/tanki). */
  call: "pon" | "chi" | null;
}

interface NeedSlot {
  /** Concrete tiles already held toward this block. */
  held: Tile[];
  options: NeedOption[];
}

interface Structure {
  /** Tiles of the blocks that are already complete. */
  blockTiles: Tile[];
  done: FinalBlock[];
  needs: NeedSlot[];
}

/**
 * Every reading of the concealed tiles as (4 − melds) sets + a head, where each
 * unfinished block is one tile short. A block two or more tiles short is not
 * enumerated: that is a different hand, and `slack` on the required count is the
 * knob for "a slightly slower but richer shape", not this.
 *
 * The search peels blocks off the lowest live type, exactly like
 * `decompose.ts:peelSets` and `mjrender/shanten.ts` — block options are tried
 * before the "leave it as a floater" branch, so the tile-efficient structures
 * are the ones emitted first and the emit cap therefore truncates the tail, not
 * the head.
 */
function enumerateStructures(
  obs: Observation,
  maxStructures: number,
  maxNeeds: number,
): Structure[] {
  const slots = 4 - obs.melds[0].length;
  const out: Structure[] = [];
  if (slots < 0) return out;

  // Tiles of each type, aka first: a block that can be built from either copy
  // should be built from the one that is worth a han.
  const byType: Tile[][] = Array.from({ length: 34 }, () => []);
  for (const t of obs.hand) byType[tileType(t)].push(t);
  for (const list of byType) {
    list.sort((a, b) => Number(obs.akaIds.has(b)) - Number(obs.akaIds.has(a)) || a - b);
  }
  // `lo` hands out tiles to blocks (aka first); `hi` retires floaters off the
  // tail, so discarding a spare copy never spends the red one.
  const lo = new Array<number>(34).fill(0);
  const hi = byType.map((l) => l.length);
  const count = (ty: number) => hi[ty] - lo[ty];

  const blockTiles: Tile[] = [];
  const done: FinalBlock[] = [];
  const needs: NeedSlot[] = [];
  let sets = 0;
  // Set slots only: `needs` also holds the head when it is still a 単騎, and the
  // head does not compete for a set slot.
  let partials = 0;
  let head = false;
  let nodes = 0;

  const takeBlock = (ty: number, n: number, into: Tile[]): void => {
    for (let k = 0; k < n; k++) into.push(byType[ty][lo[ty]++]);
  };
  const undoBlock = (ty: number, n: number, from: Tile[]): void => {
    for (let k = 0; k < n; k++) {
      from.pop();
      lo[ty]--;
    }
  };

  const emit = (): void => {
    out.push({
      blockTiles: blockTiles.slice(),
      done: done.slice(),
      needs: needs.map((n) => ({ held: n.held.slice(), options: n.options })),
    });
  };

  const dfs = (start: number): void => {
    if (out.length >= maxStructures || ++nodes > 6000) return;
    let i = start;
    while (i < 34 && count(i) === 0) i++;

    if (sets + partials === slots && head) {
      emit();
      return;
    }
    if (i >= 34) return;
    if (needs.length >= maxNeeds) return;

    const rank = i < 27 ? i % 9 : -1;
    const room = slots - sets - partials;

    // --- complete sets ---
    if (room > 0 && count(i) >= 3) {
      takeBlock(i, 3, blockTiles);
      done.push({ kind: "triplet", type: i });
      sets++;
      dfs(i);
      sets--;
      done.pop();
      undoBlock(i, 3, blockTiles);
    }
    if (room > 0 && rank >= 0 && rank <= 6 && count(i + 1) > 0 && count(i + 2) > 0) {
      takeBlock(i, 1, blockTiles);
      takeBlock(i + 1, 1, blockTiles);
      takeBlock(i + 2, 1, blockTiles);
      done.push({ kind: "run", type: i });
      sets++;
      dfs(i);
      sets--;
      done.pop();
      undoBlock(i + 2, 1, blockTiles);
      undoBlock(i + 1, 1, blockTiles);
      undoBlock(i, 1, blockTiles);
    }

    // --- the head, complete ---
    if (!head && count(i) >= 2) {
      takeBlock(i, 2, blockTiles);
      done.push({ kind: "pair", type: i });
      head = true;
      dfs(i);
      head = false;
      done.pop();
      undoBlock(i, 2, blockTiles);
    }

    // --- unfinished blocks, one tile short ---
    if (room > 0 && count(i) >= 2) {
      // A pair that wants to be a triplet: pon-able.
      const held: Tile[] = [];
      takeBlock(i, 2, held);
      needs.push({
        held,
        options: [{ type: i, block: { kind: "triplet", type: i }, call: "pon" }],
      });
      partials++;
      dfs(i);
      partials--;
      needs.pop();
      undoBlock(i, 2, held);
    }
    if (room > 0 && rank >= 0 && rank <= 7 && count(i + 1) > 0) {
      // 両面 / 辺張: chi-able from either end that exists.
      const options: NeedOption[] = [];
      if (rank >= 1) {
        options.push({ type: i - 1, block: { kind: "run", type: i - 1 }, call: "chi" });
      }
      if (rank <= 6) options.push({ type: i + 2, block: { kind: "run", type: i }, call: "chi" });
      if (options.length > 0) {
        const held: Tile[] = [];
        takeBlock(i, 1, held);
        takeBlock(i + 1, 1, held);
        needs.push({ held, options });
        partials++;
        dfs(i);
        partials--;
        needs.pop();
        undoBlock(i + 1, 1, held);
        undoBlock(i, 1, held);
      }
    }
    if (room > 0 && rank >= 0 && rank <= 6 && count(i + 2) > 0) {
      // 嵌張: chi-able on the middle tile.
      const held: Tile[] = [];
      takeBlock(i, 1, held);
      takeBlock(i + 2, 1, held);
      needs.push({
        held,
        options: [{ type: i + 1, block: { kind: "run", type: i }, call: "chi" }],
      });
      partials++;
      dfs(i);
      partials--;
      needs.pop();
      undoBlock(i + 2, 1, held);
      undoBlock(i, 1, held);
    }

    // --- the head, still single (単騎). Never callable: a pon would make it a
    // triplet, which is a different structure, and no call makes a head.
    if (!head && needs.length < maxNeeds) {
      const held: Tile[] = [];
      takeBlock(i, 1, held);
      needs.push({ held, options: [{ type: i, block: { kind: "pair", type: i }, call: null }] });
      head = true;
      dfs(i);
      head = false;
      needs.pop();
      undoBlock(i, 1, held);
    }

    // --- floater: spend this copy, off the tail so the aka survives ---
    hi[i]--;
    dfs(i);
    hi[i]++;
  };

  dfs(0);
  return out;
}

// ---------------------------------------------------------------------------
// concrete futures
// ---------------------------------------------------------------------------

interface Candidate {
  structure: Structure;
  chosen: NeedOption[];
  /** Index into `chosen` of the tile the hand wins on — never melded. */
  winIdx: number;
  /** Indices a call may serve: the plan's `callable`. Empty ⇒ a closed future. */
  callIdx: number[];
  /** Indices the VALUATION melds — `callIdx` minus the winning slot. */
  meldIdx: number[];
  finalBlocks: FinalBlock[];
  key: string;
  open: boolean;
}

const blockKey = (b: FinalBlock): string => `${b.kind[0]}${b.type}`;

/**
 * Identity of the finished hand: called melds + finished blocks, sorted. Blind
 * to `open` on purpose — an open and a closed route to the SAME final shape are
 * one plan as far as the lock is concerned, so calling a planned tile does not
 * knock the hand off its own plan.
 */
function finalKey(melds: readonly Meld[], blocks: readonly FinalBlock[]): string {
  const parts = melds.map((m) => {
    const lowest = Math.min(...m.tiles.map(tileType));
    const kind = m.kind === "chi" ? "run" : "triplet";
    return blockKey({ kind, type: lowest });
  });
  for (const b of blocks) parts.push(blockKey(b));
  return parts.sort().join(",");
}

/** Product over the needs' options, bounded; falls back to one greedy pick. */
function assignments(needs: NeedSlot[], avail: Availability, cap: number): NeedOption[][] {
  let total = 1;
  for (const n of needs) total *= n.options.length;
  if (total > cap) {
    // Too wide to enumerate: take the end with the most copies left, which is
    // the assignment the hand would actually aim at.
    return [
      needs.map((n) =>
        n.options.reduce((best, o) => avail.draw(o.type) > avail.draw(best.type) ? o : best)
      ),
    ];
  }
  let out: NeedOption[][] = [[]];
  for (const n of needs) {
    const next: NeedOption[][] = [];
    for (const prefix of out) for (const o of n.options) next.push([...prefix, o]);
    out = next;
  }
  return out;
}

/** A concrete id for a type we do not hold. Aka-blind: 赤5筒 are ids 52/53. */
const idOf = (ty: number): Tile => ty * 4 + 3;

function makeCandidates(
  obs: Observation,
  structures: Structure[],
  avail: Availability,
): Candidate[] {
  const melds = obs.melds[0];
  const out: Candidate[] = [];
  const seen = new Set<string>();

  for (const s of structures) {
    if (s.needs.length === 0) continue; // already a winning shape — nothing to plan
    for (const chosen of assignments(s.needs, avail, 16)) {
      const finalBlocks = [...s.done, ...chosen.map((c) => c.block)];
      const key = finalKey(melds, finalBlocks);

      // For VALUATION one slot has to be the tile the hand wins on, and it
      // cannot also be a meld — claiming the last missing tile is a ron, not a
      // completion. A slot no call could serve (a 単騎 head) is the natural
      // choice; failing that the last one is taken, arbitrarily and only for
      // pricing.
      let winIdx = chosen.findIndex((c) => c.call === null);
      if (winIdx < 0) winIdx = chosen.length - 1;

      const push = (callIdx: number[]) => {
        const meldIdx = callIdx.filter((i) => i !== winIdx);
        const open = callIdx.length > 0;
        const dedup = `${key}|${open ? "o" : "c"}`;
        if (seen.has(dedup)) return;
        seen.add(dedup);
        out.push({ structure: s, chosen, winIdx, callIdx, meldIdx, finalBlocks, key, open });
      };

      push([]);
      // Which slots a CALL may serve is a wider question than which one the
      // valuation melds: with two or more tiles still missing, claiming any of
      // them leaves another to win on — including the slot priced as the winner,
      // which is how a 役牌 pon stays plannable when the 役牌 happens to sort
      // last. At one tile short there is nothing left to win on afterwards, so
      // nothing is callable.
      if (chosen.length >= 2) {
        const callables = chosen.map((c, i) => (c.call !== null ? i : -1)).filter((i) => i >= 0);
        if (callables.length > 0) push(callables);
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// value: what the finished hand pays
// ---------------------------------------------------------------------------

/**
 * The completed hand run through the REAL scorer (`yaku.ts:scoreWin`), which is
 * the same code the round settles with — so 三色/一通/対々和/混一色 and the fu
 * ladder are exact rather than approximated, and a plan whose finished OPEN hand
 * scores nothing comes back `null`. That null IS the 後付け screen, and it is
 * strictly sharper than `hasYakuProspect`'s families: it judges the concrete
 * final 14 tiles, not the families they might belong to.
 *
 * Simplifications, all deliberate and all in the same direction (rank, not
 * price):
 *   - the win is priced as a RON. Tsumo splits differ by a few hundred points
 *     and by 門前清自摸和, which the closed bonus below already stands in for.
 *   - a menzen plan is scored WITH 立直 declared (+1 han) plus a flat +1 han for
 *     the 一発/裏ドラ/門前清自摸和 bundle. Riichi is nearly always available to a
 *     closed tenpai in this ruleset, and the bundle's expectation is close
 *     enough to a han that resolving it further would be false precision.
 *   - 赤 in tiles not yet held is not counted: a required 5筒 is materialised as
 *     the non-red copy, the same aka-blind convention `hand.ts` uses.
 *   - 供託 is excluded, 本場 included (as in `ronValue`).
 */
function valueOf(obs: Observation, c: Candidate, cfg: RuleConfig): number | null {
  const melds: Meld[] = [...obs.melds[0]];
  const hand: Tile[] = [...c.structure.blockTiles];
  const melded = new Set(c.meldIdx);
  const seat = obs.seat;

  for (let i = 0; i < c.chosen.length; i++) {
    const need = c.structure.needs[i];
    const opt = c.chosen[i];
    const called = idOf(opt.type);
    if (melded.has(i)) {
      melds.push({
        kind: opt.call === "pon" ? "pon" : "chi",
        who: seat,
        // Chi is kamicha-only; a pon can come from anyone, and the seat only
        // matters to 符 through nothing at all — but the field is required.
        fromWho: ((seat + 3) % 4) as Meld["fromWho"],
        tiles: [...need.held, called].sort((x, y) => x - y),
        calledTile: called,
      });
    } else {
      hand.push(...need.held, called);
    }
  }

  const winTile = idOf(c.chosen[c.winIdx].type);
  const menzen = melds.every((m) => m.kind === "ankan");
  let aka = 0;
  for (const t of hand) if (obs.akaIds.has(t)) aka++;
  for (const m of melds) for (const t of m.tiles) if (obs.akaIds.has(t)) aka++;

  const res = scoreWin({
    seat,
    hand,
    melds,
    winTile,
    tsumo: false,
    riichi: menzen,
    doubleRiichi: false,
    ippatsu: false,
    rinshan: false,
    chankan: false,
    haitei: false,
    houtei: false,
    tenhou: false,
    chiihou: false,
    seatWind: obs.seatWind,
    roundWind: obs.roundWind,
    doraTypes: obs.doraIndicators.map((t) => doraFromIndicatorType(tileType(t))),
    uraTypes: [],
    akaCount: aka,
    cfg,
  });
  if (!res) return null; // 役なし ⇒ not a future, a trap

  const dealer = obs.seatWind === 27;
  if (res.yakuman.length > 0) return ronPayment(dealer, res.base) + obs.honba * 300;
  const bonus = menzen ? 1 : 0; // 一発/裏/門前ツモ, lumped
  const { base } = basePoints(res.han + bonus, res.fu, cfg);
  return ronPayment(dealer, base) + obs.honba * 300;
}

// ---------------------------------------------------------------------------
// labels
// ---------------------------------------------------------------------------

const SUITS = "mps";
const HONORS = "東南西北白發中";

function typeName(ty: number): string {
  return ty < 27 ? `${rankOfType(ty)}${suitOfType(ty)}` : HONORS[ty - 27];
}

function blockName(b: FinalBlock): string {
  if (b.kind === "run") {
    const r = rankOfType(b.type);
    return `${r}${r + 1}${r + 2}${SUITS[Math.floor(b.type / 9)]}`;
  }
  const n = typeName(b.type);
  return b.kind === "triplet" ? n.repeat(3) : n.repeat(2);
}

function labelOf(c: Candidate, required: number[]): string {
  const shape = c.finalBlocks.map(blockName).join(" ");
  const wants = required.map(typeName).join("/");
  return `${c.open ? "副露" : "門前"} ${shape} ← ${wants}`;
}

// ---------------------------------------------------------------------------
// the entry point
// ---------------------------------------------------------------------------

/**
 * The ranked futures of this hand, best profit first.
 *
 * Empty beyond 3向聴 by design: a formless hand has no future worth locking on
 * to, every structure looks the same to the value model, and the base policy's
 * pure efficiency is the right tool until a shape exists.
 *
 * STANDARD FORM ONLY. 七対子 and 国士無双 are not enumerated, and they do not
 * need a special case: a hand whose `shanten` comes from the irregular forms has
 * no standard reading within `shanten + 1 + slack` tiles, so nothing is emitted
 * and the base policy keeps the hand. The one thing that must never happen —
 * locking a 七対子 hand onto a standard shape it is nowhere near — is prevented
 * by that same bound rather than by a form test.
 */
export function enumerateTargets(
  obs: Observation,
  avail: Availability,
  opts: PlannerOptions = {},
): TargetPlan[] {
  if (obs.shanten < 0 || obs.shanten > 3) return [];
  const cfg = opts.cfg ?? JANKI;
  const maxPlans = opts.maxPlans ?? 40;
  const draws = opts.draws ?? ownDraws(obs);
  const maxNeeds = obs.shanten + 1 + (opts.slack ?? 1);

  const structures = enumerateStructures(obs, opts.maxStructures ?? 60, maxNeeds);
  const cands = makeCandidates(obs, structures, avail);
  if (cands.length === 0) return [];

  const doraTypes = doraTypesOf(obs);

  // Cheap pass first: `pCompleteOf` is arithmetic, `valueOf` runs the scorer, so
  // only the most promising `maxPlans` futures are ever priced properly.
  const priced = cands.map((c) => {
    const required = c.chosen.map((o) => o.type);
    const callable = c.callIdx.map((i) => c.chosen[i].type);
    const p = pCompleteOf(required, callable, avail, draws);
    let dora = 0;
    for (const t of c.structure.blockTiles) {
      if (doraTypes.has(tileType(t)) || obs.akaIds.has(t)) dora++;
    }
    for (const ty of required) if (doraTypes.has(ty)) dora++;
    return { c, required, callable, p, pre: p * (1 + 0.2 * dora) };
  }).filter((x) => x.p > 0);

  priced.sort((a, b) => b.pre - a.pre);
  priced.length = Math.min(priced.length, maxPlans);

  const out: TargetPlan[] = [];
  for (const x of priced) {
    const value = valueOf(obs, x.c, cfg);
    if (value === null || value <= 0) continue;
    const keep = new Set<Tile>(x.c.structure.blockTiles);
    for (const n of x.c.structure.needs) for (const t of n.held) keep.add(t);
    out.push({
      key: x.c.key,
      keep,
      required: x.required,
      callable: x.callable,
      open: x.c.open,
      value,
      pComplete: x.p,
      profit: x.p * value,
      label: labelOf(x.c, x.required),
    });
  }
  out.sort((a, b) => b.profit - a.profit || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
  return out;
}

/**
 * The lock, with hysteresis. A plan is abandoned only when a challenger is
 * `margin` better than what the current target is worth NOW — re-priced with
 * this turn's availability, so a target whose tiles have died loses its lock by
 * decaying rather than by any special rule.
 *
 * `candidates` is expected sorted (profit desc) but the maximum is taken
 * explicitly: nothing about the lock should depend on a caller's sort.
 */
export function relock(
  current: TargetPlan | null,
  candidates: readonly TargetPlan[],
  margin = RELOCK_MARGIN,
): TargetPlan | null {
  if (candidates.length === 0) return null;
  let best = candidates[0];
  for (const c of candidates) if (c.profit > best.profit) best = c;
  if (!current) return best;
  // Same final shape, re-priced. Absent ⇒ the shape is no longer reachable.
  const refreshed = candidates.find((c) => c.key === current.key);
  if (!refreshed) return best;
  return best.profit > refreshed.profit * margin ? best : refreshed;
}
