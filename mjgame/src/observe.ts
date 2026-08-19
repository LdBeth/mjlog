// The imperfect-information projection a policy sees.
//
// Every per-seat array is indexed RELATIVE to the observing seat
// (0 = self, 1 = shimocha, 2 = toimen, 3 = kamicha). That is what lets one
// trained policy play all four seats without learning a seat embedding.

import type { Meld, Tile } from "mjrender/model.ts";
import type { RiverEntry } from "mjrender/state.ts";
import type { DangerAssessment } from "mjrender/danger.ts";
import { assessDanger } from "mjrender/danger.ts";
import { tileType } from "mjrender/tiles.ts";
import { countsFromTiles, shanten, ukeireTypes } from "./kernel.ts";
import type { Furiten } from "./table.ts";
import { Table } from "./table.ts";
import type { WinOracle } from "./legal.ts";
import { ANY_WIN } from "./legal.ts";
import { analyze } from "./hand.ts";
import type { ActionPreview } from "./penalty/preview.ts";
import { makePreview } from "./penalty/preview.ts";
import type { DojoConfig } from "./rules.ts";
import type { Action, Seat } from "./types.ts";
import { relSeat, SEATS } from "./types.ts";

export interface Ukeire {
  type: number;
  live: number; // copies not visible to this seat
}

/**
 * What letting a particular tile go would leave behind. Precomputed here rather
 * than in each policy because the katagari half needs the win oracle, which is
 * the referee's to hand out — and because every policy wants the shanten anyway.
 */
export interface DiscardInfo {
  shanten: number;
  /** 片和了り: the resulting tenpai has waits that score and waits that do not. */
  katagari: boolean;
  /**
   * The resulting tenpai has NO wait that scores. Like `katagari` this needs the
   * win oracle; unlike it, this is the discard that walks an OPEN hand straight
   * into the 後付け penalty (a closed hand cures the same shape with riichi).
   */
  yakuless: boolean;
}

export interface Observation {
  seat: Seat;
  kyoku: number;
  honba: number;
  kyotaku: number;
  junme: number;
  wallRemaining: number;

  hand: Tile[];
  drawn: Tile | null;
  /**
   * The tile just discarded, when this is a CLAIM decision (null/absent on a
   * turn decision). Optional so the many hand-built Observations in tests need
   * no edits; a policy that wants it on every claim should read it here rather
   * than digging it out of `legal`.
   */
  claimTile?: Tile | null;
  /** Relative index 0..3. */
  melds: Meld[][];
  rivers: RiverEntry[][];
  scores: number[];
  riichi: boolean[];
  riichiJunme: number[];

  doraIndicators: Tile[];
  seatWind: number;
  roundWind: number;
  akaIds: ReadonlySet<Tile>;

  shanten: number;
  waits: number[];
  /** Waits that would actually score on a ron (出和了可能形). */
  ronnable: number[];
  /** 片和了り right now: some waits carry a yaku and some do not. */
  katagari: boolean;
  ukeire: Ukeire[];
  doraCount: number;
  furiten: Furiten;
  /** Keyed by the tile of each legal discard. Empty when it is not our turn. */
  discardInfo: Map<Tile, DiscardInfo>;
  /**
   * 不聴時のドラ切りをポンされた: every tedashi for the rest of the round is a
   * violation. Unlike the post-riichi lock this is NOT enforced by `legal.ts`
   * — the dojo ledgers it — so a policy that cannot see it cannot avoid it.
   *
   * The lock is armed by the ドラ切り後の手出し rule, so it only ever goes true
   * for a driver that wired the dojo hooks (`makeDojoHooks` in `main.ts`);
   * a bare `runMatch` without them leaves this permanently false.
   */
  tsumogiriLock: boolean;

  /** Per candidate discard *type*, computed lazily-ish on construction. */
  danger: Map<number, DangerAssessment>;

  /**
   * The dojo referee, asked hypothetically: "would this action be ledgered?".
   * See `penalty/preview.ts` — it runs the real `DojoRule.check` predicates
   * against a hypothetical table, so a policy can DECLINE a 禁じ手 instead of
   * merely pricing it.
   *
   * Present only when the driver supplied a `DojoConfig` (both match drivers
   * do; hand-built Observations in tests do not), and valid only for the
   * decision this Observation was built for — it reads the live Table.
   */
  preview?: ActionPreview;

  violations: number[]; // relative counts
  legal: Action[];
}

/**
 * Per legal discard: the shanten it leaves, and whether that shape is 片和了り
 * or yakuless outright.
 *
 * The oracle can only judge the hand the table is actually holding, so each
 * candidate is evaluated by lifting the tile out and putting it straight back.
 * `analyze` is only worth running on shapes that reach tenpai, which is what
 * keeps this affordable — most turns it runs zero times.
 */
function discardInfoFor(
  t: Table,
  seat: Seat,
  legal: Action[],
  oracle: WinOracle,
): Map<Tile, DiscardInfo> {
  const out = new Map<Tile, DiscardInfo>();
  const hand = t.hands[seat];
  const open = t.melds[seat].length;
  const closed = t.isMenzen(seat);

  for (const a of legal) {
    if (a.t !== "discard" || out.has(a.tile)) continue;
    const i = hand.lastIndexOf(a.tile);
    if (i < 0) continue;

    const [lifted] = hand.splice(i, 1);
    try {
      const sh = shanten(countsFromTiles(hand), open, closed);
      const info = sh <= 0 ? analyze(t, seat, null, oracle) : null;
      out.set(a.tile, {
        shanten: sh,
        katagari: info?.katagari ?? false,
        yakuless: info !== null && info.tenpai && info.ronnable.length === 0,
      });
    } finally {
      hand.splice(i, 0, lifted);
    }
  }
  return out;
}

export function observe(
  t: Table,
  seat: Seat,
  legal: Action[],
  drawn: Tile | null,
  oracle: WinOracle = ANY_WIN,
  claimTile: Tile | null = null,
  dojo?: DojoConfig,
): Observation {
  const rel = <T>(pick: (s: Seat) => T): T[] => SEATS.map((s) => pick(((seat + s) % 4) as Seat));

  const hand = [...t.hands[seat]];
  const open = t.melds[seat].length;
  const closed = t.isMenzen(seat);
  const visible = t.visibleCounts(seat);

  // Resting-hand analysis is done on the 3n+1 shape: drop the drawn tile if we
  // are mid-turn, so shanten/ukeire mean "after discarding nothing yet".
  const resting = drawn === null ? hand : hand.filter((_, i) => i !== hand.lastIndexOf(drawn));
  const counts = countsFromTiles(resting);
  const sh = shanten(counts, open, closed);
  const waitTypes = sh <= 0 ? ukeireTypes(counts, open, closed, sh) : [];
  const ukeire: Ukeire[] = ukeireTypes(counts, open, closed, sh).map((type) => ({
    type,
    live: Math.max(0, 4 - visible[type]),
  }));

  const threats = t.threats(seat);
  const furo = t.furoThreats(seat);
  const ownCounts = countsFromTiles(hand);
  const danger = new Map<number, DangerAssessment>();
  if (threats.length > 0 || furo.length > 0) {
    for (const ty of new Set(hand.map(tileType))) {
      const d = assessDanger(ty, threats, furo, visible, ownCounts);
      if (d) danger.set(ty, d);
    }
  }

  const here = analyze(t, seat, drawn, oracle);
  const vio = SEATS.map((s) => t.ledger.filter((v) => v.seat === ((seat + s) % 4)).length);

  return {
    seat,
    kyoku: t.kyoku,
    honba: t.round.honba,
    kyotaku: t.round.kyotaku,
    junme: t.junme,
    wallRemaining: t.wall.remaining,
    hand,
    drawn,
    claimTile,
    melds: rel((s) => t.melds[s]),
    rivers: rel((s) => t.board.rivers[s]),
    scores: rel((s) => t.scores[s]),
    riichi: rel((s) => t.riichi[s]),
    riichiJunme: rel((s) => t.board.riichiJunme[s]),
    doraIndicators: [...t.indicators],
    seatWind: t.seatWindType(seat),
    roundWind: t.roundWindType,
    akaIds: t.cfg.akaIds,
    shanten: sh,
    waits: waitTypes,
    ronnable: here.ronnable,
    katagari: here.katagari,
    ukeire,
    doraCount: t.countDora(seat),
    furiten: { ...t.furiten[seat] },
    discardInfo: discardInfoFor(t, seat, legal, oracle),
    tsumogiriLock: t.tsumogiriLock[seat],
    danger,
    violations: vio,
    legal,
    // Only when the driver said which rules are in force: a preview judged by a
    // different DojoConfig than the round is played under would steer the policy
    // away from moves nobody is charging for (and towards ones somebody is).
    preview: dojo?.enabled ? makePreview(t, seat, dojo, oracle) : undefined,
  };
}

export { relSeat };
