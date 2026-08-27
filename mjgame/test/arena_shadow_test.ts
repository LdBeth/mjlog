// Loopback parity for the riichi.dev bridge: the engine plays a real hanchan,
// its PublicEvents are translated into the MJAI wire view one seat would
// receive (tiles as strings, other seats' draws masked to "?"), a ShadowGame
// consumes that stream — and at every decision point of the observed seat the
// shadow's Observation must equal the engine's REAL one, field by field.
//
// "Same chosen action" would NOT be a sufficient assertion: the champion
// degrades silently when `danger`/`discardInfo`/`preview` are missing or wrong
// (an absent danger reading costs 0, an absent preview passes everything), so
// only observation equality proves the shadow feeds the policy what self-play
// fed it. The preview referee is compared behaviorally — its verdict on every
// legal call/discard/kan — because it is four closures over the live Table.
//
// The capture runs WITHOUT the dojo hooks: riichi.dev judges no 禁じ手, so the
// production shadow's ledger is empty and `tsumogiriLock` never arms — the
// capture must be the same regime or `violations` could never match. `dojo`
// (the preview) stays ON both sides: that is the champion's graded regime.
//
// Both sides play under ARENA_CFG (aka = {16,52,88}), so ids agree end to end
// — the synthesized "decoded observation" (riichi.dev's base64 blob, which
// shares mjgame's 136-id scheme) carries the engine's absolute ids and
// `syncFromDecoded` must reconcile the shadow onto them exactly.

import { assert, assertEquals } from "@std/assert";
import type { Round, Tile } from "mjrender/model.ts";
import { loadKtune, makePolicy } from "../src/harness.ts";
import { runMatchSync } from "../src/match.ts";
import type { Observation } from "../src/observe.ts";
import { DOJO_HEADLESS } from "../src/rules.ts";
import { scorer } from "../src/score.ts";
import type { Table } from "../src/table.ts";
import type { Action, PublicEvent, Seat } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { ARENA_CFG, idToPai } from "../src/net/mjai.ts";
import { ShadowGame } from "../src/net/shadow.ts";

const CHAMPION = loadKtune(new URL("../weights/champion.json", import.meta.url).pathname);

// ---------------------------------------------------------------------------
// capture: one engine hanchan, with the observed seat's decisions recorded
// ---------------------------------------------------------------------------

interface ReqRecord {
  legal: Action[];
  drawn: Tile | null;
  claimTile: Tile | null;
  decoded: unknown;
  snap: string;
}
type Item = { ev: PublicEvent } | { req: ReqRecord };

function capture(seed: number, observed: Seat): { stream: Item[]; rounds: Round[] } {
  const stream: Item[] = [];
  const ref: { t: Table | null } = { t: null };
  const built = SEATS.map((s) =>
    makePolicy({
      kind: s === observed ? "k" : "h",
      name: `P${s}`,
      seed: seed * 4 + s,
      ktune: s === observed ? CHAMPION : undefined,
    })
  );
  const policies = SEATS.map((s) => {
    const p = built[s].policy;
    if (s !== observed) return p;
    return {
      name: p.name,
      decide(obs: Observation): Action {
        stream.push({
          req: {
            legal: obs.legal,
            drawn: obs.drawn,
            claimTile: obs.claimTile ?? null,
            decoded: decodedOf(ref.t!, observed, obs.drawn),
            snap: snapObservation(obs),
          },
        });
        return p.decide(obs);
      },
    };
  });
  const result = runMatchSync(policies, {
    seed,
    cfg: ARENA_CFG,
    dojo: DOJO_HEADLESS,
    scorer,
    tableRef: ref,
    sink: (e) => stream.push({ ev: e }),
  });
  for (const b of built) b.close();
  return { stream, rounds: result.rounds };
}

/** The arena's decoded-observation shape, filled with the engine's absolute
 *  ids — what the server's base64 blob provides at each request_action. */
function decodedOf(t: Table, me: Seat, drawn: Tile | null): unknown {
  return {
    player_id: me,
    hands: SEATS.map((s) => (s === me ? [...t.hands[s]] : [])),
    melds: SEATS.map((s) => t.melds[s].map((m) => [...m.tiles])),
    discards: SEATS.map((s) => t.board.rivers[s].map((r) => r.tile)),
    dora_indicators: [...t.board.indicators],
    scores: [...t.scores],
    riichi_declared: [...t.riichi],
    drawn_tile: drawn,
  };
}

// ---------------------------------------------------------------------------
// the canonical snapshot (both sides run through the same function)
// ---------------------------------------------------------------------------

function snapObservation(obs: Observation): string {
  const body = {
    seat: obs.seat,
    kyoku: obs.kyoku,
    honba: obs.honba,
    kyotaku: obs.kyotaku,
    junme: obs.junme,
    wallRemaining: obs.wallRemaining,
    hand: [...obs.hand],
    drawn: obs.drawn,
    claimTile: obs.claimTile ?? null,
    melds: obs.melds.map((ms) =>
      ms.map((m) => ({
        kind: m.kind,
        who: m.who,
        fromWho: m.fromWho,
        tiles: [...m.tiles],
        // An ankan's calledTile is the engine's hand-order-first copy —
        // unknowable from the wire and read by nothing downstream; normalize.
        calledTile: m.kind === "ankan" ? Math.min(...m.tiles) : m.calledTile,
      }))
    ),
    rivers: obs.rivers.map((rv) => rv.map((r) => ({ ...r }))),
    scores: [...obs.scores],
    riichi: [...obs.riichi],
    riichiJunme: [...obs.riichiJunme],
    doraIndicators: [...obs.doraIndicators],
    seatWind: obs.seatWind,
    roundWind: obs.roundWind,
    akaIds: [...obs.akaIds].sort((a, b) => a - b),
    shanten: obs.shanten,
    waits: [...obs.waits],
    ronnable: [...obs.ronnable],
    katagari: obs.katagari,
    ukeire: obs.ukeire.map((u) => ({ ...u })),
    doraCount: obs.doraCount,
    furiten: { ...obs.furiten },
    discardInfo: [...obs.discardInfo.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([k, v]) => [k, { ...v }]),
    tsumogiriLock: obs.tsumogiriLock,
    danger: [...obs.danger.entries()].sort((a, b) => a[0] - b[0]),
    violations: [...obs.violations],
    legal: obs.legal,
    preview: previewVerdicts(obs),
  };
  return JSON.stringify(
    body,
    (_k, v) => (v instanceof Set ? [...v].sort((a, b) => a - b) : v),
    1,
  );
}

/** The referee's opinion of every legal action — the behavioral face of
 *  `obs.preview`, which is otherwise closures over the live Table. */
function previewVerdicts(obs: Observation): unknown {
  const pv = obs.preview;
  if (!pv) return null;
  const out: Array<[string, string[]]> = [];
  let sawKan = false;
  for (const a of obs.legal) {
    if (a.t === "discard") {
      out.push([JSON.stringify(a), pv.discard(a, obs.drawn).map((v) => v.label)]);
    } else if (a.t === "chi" || a.t === "pon" || a.t === "daiminkan") {
      out.push([JSON.stringify(a), pv.call(a).map((v) => v.label)]);
    } else if (a.t === "ankan" || a.t === "kakan") {
      sawKan = true;
      out.push([JSON.stringify(a), pv.kan(a, obs.drawn).map((v) => v.label)]);
    }
  }
  if (sawKan) out.push(["skipKan", pv.skipKan(obs.drawn).map((v) => v.label)]);
  return out;
}

// ---------------------------------------------------------------------------
// replay: engine record → MJAI wire view → ShadowGame → assert parity
// ---------------------------------------------------------------------------

interface Coverage {
  riichi: boolean;
  kan: boolean;
  hora: boolean;
  honba: boolean;
  kyotaku: boolean;
  ownCall: boolean;
  claims: number;
  decisions: number;
}

function replayAndAssert(
  stream: Item[],
  rounds: Round[],
  observed: Seat,
  cov: Coverage,
  tag: string,
): void {
  const shadow = new ShadowGame();
  shadow.apply({ type: "start_game", id: observed });
  let roundIdx = -1;

  // The id-ledger invariant: the shadow board is a partial 136-permutation at
  // every step — no id twice, placeholders included. (A called river entry
  // legitimately shares its id with the meld that took it; excluded.) The
  // observation snapshots alone would catch most corruptions, but late and far
  // from their cause; this names the guilty event.
  const checkDup = (why: string) => {
    const t = shadow.table;
    if (!t) return;
    const seen = new Map<number, string>();
    const put = (id: number, w: string) => {
      if (seen.has(id)) throw new Error(`DUP ${id}: ${seen.get(id)} + ${w} after ${why}`);
      seen.set(id, w);
    };
    for (const s of SEATS) {
      t.hands[s].forEach((id) => put(id, `hand${s}`));
      t.board.rivers[s].forEach((r) => {
        if (r.calledBy === undefined) put(r.tile, `river${s}`);
      });
      t.melds[s].forEach((m, i) => m.tiles.forEach((id) => put(id, `meld${s}.${i}`)));
    }
    t.board.indicators.forEach((id) => put(id, "ind"));
  };

  for (const item of stream) {
    if ("req" in item) {
      cov.decisions++;
      if (item.req.claimTile !== null) cov.claims++;
      checkDup("pre-sync");
      shadow.syncFromDecoded(item.req.decoded);
      checkDup("post-sync");
      const obs = shadow.observe(item.req.legal, item.req.drawn, item.req.claimTile);
      assertEquals(
        snapObservation(obs),
        item.req.snap,
        `${tag} 決定#${cov.decisions}: 影の観測が本物と食い違う`,
      );
      continue;
    }
    const e = item.ev;
    switch (e.e) {
      case "deal": {
        roundIdx++;
        if (e.honba > 0) cov.honba = true;
        if (e.kyotaku > 0) cov.kyotaku = true;
        const startHands = rounds[roundIdx].startHands;
        shadow.apply({
          type: "start_kyoku",
          bakaze: e.kyoku < 4 ? "E" : e.kyoku < 8 ? "S" : "W",
          kyoku: (e.kyoku % 4) + 1,
          honba: e.honba,
          kyotaku: e.kyotaku,
          oya: e.dealer,
          scores: [...e.scores],
          dora_marker: idToPai(e.indicator),
          tehais: SEATS.map((s) => startHands[s].map((id) => s === observed ? idToPai(id) : "?")),
        });
        break;
      }
      case "draw":
        shadow.apply({
          type: "tsumo",
          actor: e.who,
          pai: e.who === observed && e.tile !== null ? idToPai(e.tile) : "?",
        });
        break;
      case "discard":
        shadow.apply({
          type: "dahai",
          actor: e.who,
          pai: idToPai(e.tile),
          tsumogiri: e.tsumogiri,
        });
        break;
      case "riichi":
        if (e.step === 1) shadow.apply({ type: "reach", actor: e.who });
        else {
          cov.riichi = true;
          shadow.apply({ type: "reach_accepted", actor: e.who });
        }
        break;
      case "call": {
        const m = e.meld;
        if (m.who === observed) cov.ownCall = true;
        const consumed = (() => {
          const ts = [...m.tiles];
          if (m.kind !== "ankan") ts.splice(ts.indexOf(m.calledTile), 1);
          return ts.map((id) => idToPai(id));
        })();
        if (m.kind === "chi" || m.kind === "pon" || m.kind === "daiminkan") {
          cov.kan ||= m.kind === "daiminkan";
          shadow.apply({
            type: m.kind,
            actor: m.who,
            target: m.fromWho,
            pai: idToPai(m.calledTile),
            consumed,
          });
        } else if (m.kind === "ankan") {
          cov.kan = true;
          shadow.apply({ type: "ankan", actor: m.who, consumed: m.tiles.map((id) => idToPai(id)) });
        } else if (m.kind === "shouminkan") {
          cov.kan = true;
          shadow.apply({ type: "kakan", actor: m.who, pai: idToPai(m.calledTile), consumed });
        }
        break;
      }
      case "dora":
        shadow.apply({ type: "dora", dora_marker: idToPai(e.indicator) });
        break;
      case "result":
        if (e.outcome.kind === "agari") {
          cov.hora = true;
          for (const w of e.outcome.wins) {
            shadow.apply({ type: "hora", actor: w.who, target: w.fromWho });
          }
        } else {
          shadow.apply({ type: "ryukyoku", reason: e.outcome.draw });
        }
        shadow.apply({ type: "end_kyoku" });
        break;
      case "violation":
        throw new Error("capture ran without dojo hooks; no violation can occur");
    }
    checkDup(JSON.stringify(e));
  }
}

// ---------------------------------------------------------------------------

// Seeds chosen so the union of streams exercises riichi, kans, wins, and
// honba/kyotaku carryover; the assertions below keep that from rotting.
const CASES: ReadonlyArray<{ seed: number; observed: Seat }> = [
  { seed: 101, observed: 0 },
  { seed: 505, observed: 2 },
  { seed: 909, observed: 0 },
];

Deno.test("arena shadow: wire-fed observations equal the engine's, field by field", () => {
  const cov: Coverage = {
    riichi: false,
    kan: false,
    hora: false,
    honba: false,
    kyotaku: false,
    ownCall: false,
    claims: 0,
    decisions: 0,
  };
  for (const { seed, observed } of CASES) {
    const { stream, rounds } = capture(seed, observed);
    replayAndAssert(stream, rounds, observed, cov, `種${seed}/席${observed}`);
  }
  // Coverage: the parity claim is only as strong as the flows the streams
  // actually contained. A seed reshuffle that loses one of these must fail.
  assert(cov.riichi, "立直を含む種がない");
  assert(cov.kan, "槓を含む種がない");
  assert(cov.hora, "和了を含む種がない");
  assert(cov.honba, "本場の持ち越しを含む種がない");
  assert(cov.kyotaku, "供託の持ち越しを含む種がない");
  assert(cov.ownCall, "観測席自身の副露を含む種がない");
  assert(cov.claims > 0, "鳴き判断が一度も観測されなかった");
  assert(cov.decisions > 100, `観測された決定が少なすぎる: ${cov.decisions}`);
});
