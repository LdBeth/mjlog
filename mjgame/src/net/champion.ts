// The champion behind the arena wire: riichi.dev's `possible_actions` mapped
// into mjgame `Action`s, judged by the real "k" policy on the shadow table's
// Observation, and mapped back to the server's OWN entry objects.
//
// Legality is never derived locally — the reply is always one of the server's
// entries echoed verbatim, so a shadow gone wrong degrades to a bad-but-legal
// choice, never a chombo. Anything that smells like desync (an event the
// shadow refuses, a decoded blob that contradicts the board, a consumed tile
// we cannot locate) drops the kyoku to the tsumogiri fallback; the next
// `start_kyoku` rebuilds the shadow from scratch and heals it.
//
// Riichi is the one two-step exchange: the policy answers a single
// `{t:"discard", riichi:true}` action, the wire wants `reach` now and the
// dahai on the NEXT request. The chosen tile is remembered and answered
// without re-running the policy — a second decide would see pre-riichi state
// and could pick a different tile.

import type { Tile } from "mjrender/model.ts";
import { tileType } from "mjrender/tiles.ts";
import type { KTune, SeatPolicy } from "../harness.ts";
import { makePolicy } from "../harness.ts";
import { countsFromTiles, shanten } from "../kernel.ts";
import type { Observation } from "../observe.ts";
import type { Action, Seat } from "../types.ts";
import type { ArenaChooser, MjaiAction, MjaiEvent, RequestAction } from "./arena.ts";
import { TsumogiriChooser } from "./arena.ts";
import { idMatchesPai, idToPai, matchId, paiToType } from "./mjai.ts";
import { ShadowGame } from "./shadow.ts";

/** One offered choice: the mjgame action the policy sees, and the server
 *  entry that answers it. Several actions may share one entry (reach). */
export interface MappedAction {
  action: Action;
  entry: MjaiAction;
}

/** What the turn mapper needs to know about us; split out so the mapping is a
 *  pure function the tests can drive without a live shadow. */
export interface TurnCtx {
  /** Full concealed hand, drawn tile included. */
  hand: readonly Tile[];
  drawn: Tile | null;
  openMelds: number;
  menzen: boolean;
}

/**
 * Turn context: dahai/reach/hora/ankan/kakan entries → mjgame actions.
 *
 * The wire has ONE dahai entry per pai and the server decides the tsumogiri
 * flag itself, so the engine's tedashi-of-the-drawn-type variant is
 * inexpressible — the drawn tile's pai maps to the tsumogiri variant only
 * (which is also the one a 打牌固定 sanction would demand). Riichi variants
 * are offered for every tenpai-keeping discard, mirroring `tenpaiAfter`
 * (legal.ts:101): the server's own post-reach dahai offer is exactly that
 * set, so the remembered tile is always answerable.
 */
export function mapTurnActions(entries: readonly MjaiAction[], ctx: TurnCtx): MappedAction[] {
  const out: MappedAction[] = [];
  const reach = entries.find((a) => a.type === "reach");
  const tenpaiAfter = (tile: Tile): boolean => {
    const rest = ctx.hand.filter((x) => x !== tile);
    return shanten(countsFromTiles(rest), ctx.openMelds, ctx.menzen) <= 0;
  };
  for (const entry of entries) {
    switch (entry.type) {
      case "dahai": {
        const pai = String(entry.pai);
        const tile = ctx.drawn !== null && idMatchesPai(ctx.drawn, pai)
          ? ctx.drawn
          : matchId(ctx.hand, pai);
        if (tile === null) throw new Error(`打牌 ${pai} が手中にない`);
        const tsumogiri = tile === ctx.drawn;
        out.push({ action: { t: "discard", tile, riichi: false, tsumogiri }, entry });
        if (reach && tenpaiAfter(tile)) {
          out.push({ action: { t: "discard", tile, riichi: true, tsumogiri }, entry: reach });
        }
        break;
      }
      case "hora":
        out.push({ action: { t: "tsumo" }, entry });
        break;
      case "ankan": {
        const consumed = entry.consumed;
        if (!Array.isArray(consumed) || consumed.length !== 4) {
          throw new Error(`暗槓の晒し牌が読めない: ${JSON.stringify(entry)}`);
        }
        out.push({ action: { t: "ankan", type: paiToType(String(consumed[0])) }, entry });
        break;
      }
      case "kakan": {
        const tile = matchId(ctx.hand, String(entry.pai));
        if (tile === null) throw new Error(`加槓牌 ${entry.pai} が手中にない`);
        out.push({ action: { t: "kakan", tile }, entry });
        break;
      }
      case "reach": // consumed via the dahai pairing above
      case "none": // not expected on a draw turn; pass would map to nothing legal
        break;
      default:
        break; // an unknown offer is simply not offered to the policy
    }
  }
  return out;
}

/**
 * Claim context: none/hora/chi/pon/daiminkan entries → mjgame actions, pass
 * first as `claimActions` orders them (legal.ts:258). Consumed strings pick
 * DISTINCT concrete ids from our hand — two equal strings take two copies,
 * and a plain pai never takes the red one (matchId's rule).
 */
export function mapClaimActions(
  entries: readonly MjaiAction[],
  hand: readonly Tile[],
  claimTile: Tile,
): MappedAction[] {
  const out: MappedAction[] = [];
  const none = entries.find((a) => a.type === "none");
  if (none) out.push({ action: { t: "pass" }, entry: none });
  for (const entry of entries) {
    switch (entry.type) {
      case "hora":
        out.push({ action: { t: "ron" }, entry });
        break;
      case "chi":
      case "pon": {
        const pais = entry.consumed;
        if (!Array.isArray(pais) || pais.length !== 2) {
          throw new Error(`${entry.type} の晒し牌が読めない: ${JSON.stringify(entry)}`);
        }
        const ids: Tile[] = [];
        for (const pai of pais.map(String)) {
          const id = hand.find((x) => idMatchesPai(x, pai) && !ids.includes(x));
          if (id === undefined) throw new Error(`晒す牌 ${pai} が手中にない`);
          ids.push(id);
        }
        out.push({ action: { t: entry.type, tiles: [ids[0], ids[1]], called: claimTile }, entry });
        break;
      }
      case "daiminkan":
      case "kan":
        out.push({ action: { t: "daiminkan", called: claimTile }, entry });
        break;
      default:
        break;
    }
  }
  return out;
}

export interface ChampionOptions {
  ktune?: KTune;
  name?: string;
  /** Replay harness only: forget the reach two-step after every choose, so a
   *  reply that history then contradicts cannot poison later requests. */
  replay?: boolean;
  /** Test seam: stand-in for the policy's decide. Production never sets it. */
  decide?: (obs: Observation) => Action;
  log?: (line: string) => void;
}

export class ChampionChooser implements ArenaChooser {
  /** Times the kyoku was surrendered to the fallback (diagnostics). */
  fallbacks = 0;

  #shadow = new ShadowGame();
  #seat: SeatPolicy | null;
  #decide: (obs: Observation) => Action;
  #fallback = new TsumogiriChooser();
  #fallbackActive = false;
  #pendingReachPai: string | null = null;
  #replay: boolean;
  #log: (line: string) => void;

  constructor(opts: ChampionOptions = {}) {
    // The seed is inert for the champion (pure decide, epsilon 0) — any value.
    if (opts.decide) {
      this.#seat = null;
      this.#decide = opts.decide;
    } else {
      const seat = makePolicy({
        kind: "k",
        name: opts.name ?? "champion",
        seed: 1,
        ktune: opts.ktune,
      });
      this.#seat = seat;
      this.#decide = (obs) => seat.policy.decide(obs);
    }
    this.#replay = opts.replay === true;
    this.#log = opts.log ?? console.error;
  }

  close(): void {
    this.#seat?.close();
  }

  onEvent(e: MjaiEvent): void {
    // The safety net tracks the game in parallel so it is live the instant
    // the shadow is not.
    this.#fallback.onEvent(e);
    if (e.type === "start_kyoku") {
      this.#fallbackActive = false;
      this.#pendingReachPai = null;
      this.#seat?.reset(1);
    }
    try {
      this.#shadow.apply(e);
    } catch (err) {
      // Once desynced, later events may keep throwing — only the first one
      // per kyoku is news.
      if (!this.#fallbackActive) this.#enterFallback(`event ${e.type}: ${err}`);
    }
  }

  choose(req: RequestAction): MjaiAction {
    try {
      if (!this.#fallbackActive) return this.#chooseChampion(req);
    } catch (err) {
      this.#enterFallback(`request ${req.request_id}: ${err}`);
    } finally {
      if (this.#replay) this.#pendingReachPai = null;
    }
    return this.#fallback.choose(req);
  }

  #enterFallback(reason: string): void {
    this.#fallbackActive = true;
    this.#pendingReachPai = null;
    this.fallbacks++;
    this.#log(`champion: 影が同期を失った、今局はフォールバック — ${reason}`);
  }

  #chooseChampion(req: RequestAction): MjaiAction {
    const entries = req.possible_actions;
    let decoded: Record<string, unknown> | null = null;
    if (typeof req.observation === "string") {
      decoded = JSON.parse(atob(req.observation)) as Record<string, unknown>;
      this.#shadow.syncFromDecoded(decoded);
    }
    const t = this.#shadow.table;
    if (!t || this.#shadow.me < 0) throw new Error("局の外で request_action が来た");
    const me = this.#shadow.me as Seat;

    // Step 2 of a declared riichi: the tile was chosen last request; answering
    // anything else (or re-deciding) would split the declaration from its
    // discard. Absence of the remembered pai is a desync signal.
    if (this.#pendingReachPai !== null) {
      const pai = this.#pendingReachPai;
      this.#pendingReachPai = null;
      const entry = entries.find((a) => a.type === "dahai" && a.pai === pai);
      if (!entry) throw new Error(`宣言牌 ${pai} が打牌候補にない`);
      return entry;
    }

    const isTurn = entries.some((a) => a.type === "dahai" || a.type === "reach");
    let mapped: MappedAction[];
    let drawn: Tile | null = null;
    let claimTile: Tile | null = null;
    if (isTurn) {
      drawn = this.#shadow.drawnTile();
      mapped = mapTurnActions(entries, {
        hand: t.hands[me],
        drawn,
        openMelds: t.melds[me].length,
        menzen: t.isMenzen(me),
      });
    } else {
      claimTile = this.#claimTile(entries, decoded);
      mapped = mapClaimActions(entries, t.hands[me], claimTile);
    }
    if (mapped.length === 0) throw new Error("写像後の合法手が空");

    const obs = this.#shadow.observe(mapped.map((m) => m.action), drawn, claimTile);
    const action = this.#decide(obs);
    const hit = mapped.find((m) => m.action === action) ??
      mapped.find((m) => JSON.stringify(m.action) === JSON.stringify(action));
    if (!hit) throw new Error(`policy が写像外の手を返した: ${JSON.stringify(action)}`);

    if (hit.action.t === "discard" && hit.action.riichi) {
      this.#pendingReachPai = idToPai(hit.action.tile);
    }
    return hit.entry;
  }

  /** The id the claim is about: the last river tile of the discarder. The
   *  discarder itself comes from the shadow's own event stream (riichi.dev
   *  claim entries omit `target`), then a call entry's `target` if one ever
   *  appears, then the decoded blob's `last_discard` — last because its
   *  meaning drifted between RiichiEnv versions (discarder seat in the
   *  deployed arena, tile id at HEAD; a tile 0–3 would masquerade as a seat).
   *  For a 槍槓 window (no river entry) any id of the kakan's type serves —
   *  every consumer distinguishes claim tiles by type. */
  #claimTile(entries: readonly MjaiAction[], decoded: Record<string, unknown> | null): Tile {
    const t = this.#shadow.table!;
    const withTarget = entries.find((a) => a.target !== undefined);
    const from = this.#shadow.lastDiscarder() ?? withTarget?.target ?? decoded?.last_discard;
    if (typeof from !== "number" || from < 0 || from > 3) {
      throw new Error("claim の対象席が特定できない");
    }
    const claimed = entries.find((a) => a.type !== "none" && typeof a.pai === "string");
    const river = t.board.rivers[from as Seat];
    const tail = river[river.length - 1]?.tile;
    const pai = claimed?.pai;
    if (tail !== undefined && !(typeof pai === "string" && !idMatchesPai(tail, pai))) {
      return tail;
    }
    // Not that river: a 槍槓 window on someone's fresh kakan. `from` is the
    // last DISCARDER and therefore the wrong seat here — but the claimed pai
    // plus the shouminkan kind identify the meld uniquely, so search them all.
    if (typeof pai === "string") {
      for (const s of [0, 1, 2, 3] as const) {
        const meld = t.melds[s].find((m) =>
          m.kind === "shouminkan" && tileType(m.tiles[0]) === paiToType(pai)
        );
        if (meld) return meld.tiles[0];
      }
      throw new Error(`claim 対象 ${pai} が河にも副露にもない`);
    }
    throw new Error("claim 対象牌が特定できない");
  }
}
