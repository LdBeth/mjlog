// 計算の証拠 — the evidence vector a discard decision is made from.
//
// M9's thesis: the COMPUTED EVIDENCE (計算) stays exactly as it is, and only the
// hand-written CONSUMPTION of it is replaced by something learnable. This file
// is the seam between the two. It names, as plain numbers, every quantity the
// current discard score actually reads — nothing more, nothing less — so that
// `consumer.ts` can consume them without knowing where they came from, and so
// that a human can read a decision off a table of named values.
//
// THE HONESTY RULE. Every number here is produced by calling the policy's OWN
// methods through `EvidenceHooks`, never by a re-implementation. `riskOf`,
// `drawBonus`, `keepBonus`, `pressureOf` and `bufferScale` are all virtual on
// `HeuristicPolicy`, so an `AugmentedHeuristic` — whose overrides read an oracle
// or the 計算 counter — fills the same fields with its own better answers and
// the consumer downstream needs no idea which policy it is scoring for. A free
// function that recomputed `riskOf` from the danger map would silently pin every
// augmented seat back to the base policy's reading; the hooks exist to make that
// mistake impossible.
//
// THE SPLIT. Per-candidate fields vary tile by tile inside one decision;
// per-decision context fields are constant across the candidate set and are
// assembled once. That is not merely a saving — it is the shape the consumer
// needs: context enters as a MULTIPLIER on whole groups of candidate terms, and
// a multiplier that varied per candidate would not be one.
//
// Fields that no init-time curve consumes (`wideOpen`, `tsumogiri`, `open`,
// `closed`, `canRiichi`, `shanten`, `doraCount`) are here on purpose: they are
// the facts a later fit gets to reach for, and they cost one field read each.

import type { DangerLevel } from "mjrender/danger.ts";
import type { Tile } from "mjrender/model.ts";
import { tileType } from "mjrender/tiles.ts";
import { countsFromTiles, ukeireTypes } from "../kernel.ts";
import { isHonor } from "../tiles.ts";
import type { Observation } from "../observe.ts";
import type { Ctx } from "./heuristic.ts";

/** The danger ladder as an ordinal, cheapest first. Index = the level's rank. */
export const DANGER_ORDER: readonly DangerLevel[] = ["安全", "危険度低", "危険度中", "危険度高"];

/**
 * The policy methods evidence assembly needs, bound to one instance.
 *
 * Every one of these is `protected` (or private) on `HeuristicPolicy`, so the
 * class builds this object itself — see `evidenceHooks` there — and the
 * closures dispatch virtually. That is the whole mechanism by which an
 * augmented policy's overrides reach these fields.
 */
export interface EvidenceHooks {
  /** The 13-tile shape left behind by a discard. */
  handWithout(ctx: Ctx, tile: Tile): Tile[];
  /** Copies of a type this seat cannot see, minus the ones it already holds. */
  liveCopies(obs: Observation, type: number, counts: number[]): number;
  /** What letting this tile go costs defensively, before the fold multiplier. */
  riskOf(ctx: Ctx, tile: Tile): number;
  /** One-turn-lookahead bonus, already in score units. */
  drawBonus(ctx: Ctx, tile: Tile): number;
  /** Reason-to-hold-back malus, already in score units. */
  keepBonus(ctx: Ctx, tile: Tile): number;
  /** Threat volume at the table. */
  pressureOf(obs: Observation): number;
  /** The 持ち点8000未満 buffer scale on the push/fold gate. */
  bufferScale(obs: Observation): number;
  /** 順位効用's two multipliers — `{gain: 1, risk: 1}` when the layer is off. */
  standings(obs: Observation): { gain: number; risk: number };
}

/** Everything that varies from one discard candidate to the next. */
export interface CandidateEvidence {
  /** 向聴数 of the shape this discard leaves behind. −1 is a winning hand. */
  shantenAfter: number;
  /** Live copies of every accepted type. 0 unless this tile holds best shanten. */
  ukeireLive: number;
  /** Distinct accepted types — breadth. 0 unless this tile holds best shanten. */
  ukeireTypeCount: number;
  /** Dora (indicator dora + aka) still in the concealed hand afterwards. */
  doraKept: number;
  /** Concealed 役牌 pairs retained: a yaku seed and a safe holding. */
  yakuhaiPairs: number;
  /** Lone honors kept × min(junme, 12) — the term's own lateness scaling. */
  isolatedHonorLate: number;
  /** `riskOf`: the policy's own price for letting this tile go. */
  risk: number;
  /** The rule ladder's own reading, 0..3. 0 also means "no entry" — see below. */
  dangerLevel: number;
  /** 1 only when the assessor EXPLICITLY said 安全 (genbutsu — a proof). */
  explicitSafe: number;
  /** `drawBonus`, in score units. */
  drawBonus: number;
  /** `keepBonus`, in score units (a malus: subtracted downstream). */
  keepBonus: number;
  /** 1 when this discard holds the best shanten in the candidate set. */
  wideOpen: number;
  /** 1 when this is the drawn tile (ツモ切り). */
  tsumogiri: number;
}

/** Everything constant across one decision's candidate set. */
export interface ContextEvidence {
  /** `Ctx.eff` — the efficiency scale, already carrying the fold state. */
  eff: number;
  /** `Ctx.def` — the danger scale, already carrying fold × 順位効用 risk. */
  def: number;
  /** 1 while folding. Redundant with `eff`/`def` by construction; named anyway. */
  folding: number;
  /** `pressureOf`: how loud the table is. */
  pressure: number;
  /** 順位効用 gain scale (1 when the layer is off). */
  standingsGain: number;
  /** 順位効用 risk scale (1 when the layer is off). */
  standingsRisk: number;
  /** `bufferScale`: the 持ち点8000未満 damping on the push/fold gate. */
  bufferScale: number;
  junme: number;
  wallRemaining: number;
  /** The hand's shanten BEFORE the discard. */
  shanten: number;
  doraCount: number;
  /** Melds called (including ankan). */
  open: number;
  /** 1 when the hand is still 門前. */
  closed: number;
  /** 1 when riichi is on the table this turn. */
  canRiichi: number;
}

export interface EvidenceVector {
  context: ContextEvidence;
  candidate: CandidateEvidence;
}

/**
 * The per-decision half. Cheap, but not free (`pressureOf` walks the danger map,
 * 順位効用 runs a handful of `phi` calls), so it is assembled ONCE per decision
 * and handed to every candidate.
 */
export function assembleContext(h: EvidenceHooks, ctx: Ctx): ContextEvidence {
  const { obs } = ctx;
  const st = h.standings(obs);
  return {
    eff: ctx.eff,
    def: ctx.def,
    folding: ctx.folding ? 1 : 0,
    pressure: h.pressureOf(obs),
    standingsGain: st.gain,
    standingsRisk: st.risk,
    bufferScale: h.bufferScale(obs),
    junme: obs.junme,
    wallRemaining: obs.wallRemaining,
    shanten: obs.shanten,
    doraCount: obs.doraCount,
    open: ctx.open,
    closed: ctx.closed ? 1 : 0,
    canRiichi: ctx.canRiichi ? 1 : 0,
  };
}

/**
 * The per-candidate half, computed with the SAME code, in the same order, as
 * `HeuristicPolicy.scoreDiscard`'s hand-written arithmetic — the ukeire probe is
 * gated on `wideOpen` exactly as it was there, because it is the expensive part
 * (34 shanten probes) and only a tile holding the best shanten can win on it.
 */
export function assembleCandidate(
  h: EvidenceHooks,
  ctx: Ctx,
  tile: Tile,
  sh: number,
  wideOpen: boolean,
): CandidateEvidence {
  const { obs } = ctx;
  const rest = h.handWithout(ctx, tile);
  const counts = countsFromTiles(rest);

  let ukeireLive = 0;
  let ukeireTypeCount = 0;
  if (wideOpen) {
    const types = ukeireTypes(counts, ctx.open, ctx.closed, sh);
    for (const ty of types) ukeireLive += h.liveCopies(obs, ty, counts);
    ukeireTypeCount = types.length;
  }

  let doraKept = 0;
  for (const t of rest) {
    if (ctx.doraTypes.has(tileType(t))) doraKept++;
    if (obs.akaIds.has(t)) doraKept++;
  }

  let yakuhaiPairs = 0;
  let isolatedHonors = 0;
  for (let ty = 0; ty < 34; ty++) {
    if (counts[ty] >= 2 && ctx.valueHonors.has(ty)) yakuhaiPairs++;
    if (counts[ty] === 1 && isHonor(ty)) isolatedHonors++;
  }

  const level = obs.danger.get(tileType(tile))?.level;
  return {
    shantenAfter: sh,
    ukeireLive,
    ukeireTypeCount,
    doraKept,
    yakuhaiPairs,
    isolatedHonorLate: isolatedHonors * Math.min(obs.junme, 12),
    risk: h.riskOf(ctx, tile),
    dangerLevel: level === undefined ? 0 : DANGER_ORDER.indexOf(level),
    explicitSafe: level === "安全" ? 1 : 0,
    drawBonus: h.drawBonus(ctx, tile),
    keepBonus: h.keepBonus(ctx, tile),
    wideOpen: wideOpen ? 1 : 0,
    tsumogiri: obs.drawn !== null && tile === obs.drawn ? 1 : 0,
  };
}

/** Both halves at once. The convenient form; the decision loop uses the split. */
export function assembleEvidence(
  h: EvidenceHooks,
  ctx: Ctx,
  tile: Tile,
  sh: number,
  wideOpen: boolean,
): EvidenceVector {
  return {
    context: assembleContext(h, ctx),
    candidate: assembleCandidate(h, ctx, tile, sh, wideOpen),
  };
}
