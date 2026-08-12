// Export a played match as a Tenhou mjlog XML.
//
// This is nearly free: the game master already accumulates `Round.events` in
// mjrender's own replay format, so a match we played *is* a mjrender `Game`.
// Writing it out means the whole existing toolchain — the Japanese commentary
// transcript, the snapshot/anchor MCP tools, the eval generator — works on your
// own games with no changes:
//
//   cd mjrender && deno task render ../mjgame/games/seed42.xml
//
// Two things Tenhou XML cannot carry, both written to a `.mjgame.json` sidecar
// with the same basename: the second red 5-pin (aka is identified by tile id, and
// only id 52 reads as red to a standard viewer), and the dojo violation ledger.

import type { Game, Meld, Round, Tile } from "mjrender/model.ts";
import { tileType } from "mjrender/tiles.ts";
import type { MatchResult } from "./match.ts";
import type { RuleConfig } from "./rules.ts";
import type { RoundOutcome, Seat, Violation } from "./types.ts";
import { SEATS } from "./types.ts";

/** Inverse of mjrender's `decodeMeld`. Bit layout documented in that file. */
export function encodeMeld(m: Meld): number {
  const rel = (m.fromWho - m.who + 4) % 4;

  switch (m.kind) {
    case "chi": {
      const sorted = [...m.tiles].sort((a, b) => a - b);
      const base = tileType(sorted[0]);
      const called = sorted.indexOf(m.calledTile);
      // The run start is re-encoded in Tenhou's 7-per-suit packing.
      const t = (Math.floor(base / 9) * 7 + (base % 9)) * 3 + called;
      return (
        rel |
        0x4 |
        ((sorted[0] % 4) << 3) |
        ((sorted[1] % 4) << 5) |
        ((sorted[2] % 4) << 7) |
        (t << 10)
      );
    }
    case "pon": {
      const sorted = [...m.tiles].sort((a, b) => a - b);
      const ty = tileType(sorted[0]);
      const present = new Set(sorted.map((x) => x % 4));
      const unused = [0, 1, 2, 3].find((o) => !present.has(o)) ?? 3;
      const offsets = [0, 1, 2, 3].filter((o) => o !== unused);
      const called = offsets.indexOf(m.calledTile % 4);
      const t = ty * 3 + Math.max(0, called);
      return rel | 0x8 | (unused << 5) | (t << 9);
    }
    case "shouminkan": {
      const ty = tileType(m.calledTile);
      const unused = m.calledTile % 4;
      const t = ty * 3; // the "called" sub-index is unused on decode
      return rel | 0x10 | (unused << 5) | (t << 9);
    }
    case "ankan":
      // rel must be 0 — that is the only thing distinguishing an ankan from a
      // daiminkan on decode. The tile byte identifies the type; decode rebuilds
      // all four copies from it, so round-tripping `calledTile` needs it here.
      return 0 | (m.calledTile << 8);
    case "daiminkan":
      return rel | (m.calledTile << 8);
    case "nuki":
      return rel | 0x20 | (m.calledTile << 8);
  }
}

const DRAW_TAG = ["T", "U", "V", "W"];
const DISCARD_TAG = ["D", "E", "F", "G"];

function goType(cfg: RuleConfig): number {
  // Matches mjrender/src/parse.ts's decode: bit 0x08 = hanchan (not tonpuu),
  // 0x02 = no aka, 0x04 = no kuitan, 0x10 = sanma.
  let n = 0;
  if (cfg.hanchan) n |= 0x08;
  if (cfg.akaIds.size === 0) n |= 0x02;
  if (!cfg.kuitan) n |= 0x04;
  return n;
}

function roundXml(round: Round, outcome: RoundOutcome | undefined, dealerSeat: Seat): string {
  const parts: string[] = [];
  const seed = [
    round.kyoku,
    round.honba,
    round.kyotaku,
    round.dice[0],
    round.dice[1],
    round.firstDora,
  ].join(",");
  const hai = round.startHands
    .map((h, i) => `hai${i}="${h.join(",")}"`)
    .join(" ");
  parts.push(
    `<INIT seed="${seed}" ten="${round.startScores.join(",")}" oya="${dealerSeat}" ${hai}/>`,
  );

  for (const e of round.events) {
    switch (e.t) {
      case "draw":
        parts.push(`<${DRAW_TAG[e.who]}${e.tile}/>`);
        break;
      case "discard":
        parts.push(`<${DISCARD_TAG[e.who]}${e.tile}/>`);
        break;
      case "call":
        parts.push(`<N who="${e.meld.who}" m="${encodeMeld(e.meld)}"/>`);
        break;
      case "reach":
        parts.push(
          `<REACH who="${e.who}" step="${e.step}"` +
            (e.scores ? ` ten="${e.scores.join(",")}"` : "") + "/>",
        );
        break;
      case "dora":
        parts.push(`<DORA hai="${e.indicator}"/>`);
        break;
    }
  }

  if (outcome?.kind === "agari") {
    for (const w of outcome.wins) {
      const sc = SEATS.flatMap((s) => [round.startScores[s], outcome.deltas[s] / 100]);
      const yaku = w.yaku.flatMap((y) => [y.id, y.han]).join(",");
      parts.push(
        `<AGARI ba="${round.honba},${round.kyotaku}" hai="${w.hand.join(",")}" ` +
          `machi="${w.winTile}" ten="${w.fu},${w.points},${w.limit}" ` +
          (w.yakuman.length ? `yakuman="${w.yakuman.join(",")}" ` : `yaku="${yaku}" `) +
          `doraHai="${w.doraIndicators.join(",")}" ` +
          (w.uraIndicators.length ? `doraHaiUra="${w.uraIndicators.join(",")}" ` : "") +
          `who="${w.who}" fromWho="${w.fromWho}" sc="${sc.join(",")}"/>`,
      );
    }
  } else if (outcome?.kind === "ryuukyoku") {
    const sc = SEATS.flatMap((s) => [round.startScores[s], outcome.deltas[s] / 100]);
    const type = outcome.draw === "exhaustive"
      ? ""
      : ` type="${
        { sanchahou: "ron3", suukaikan: "kan4", "suucha-riichi": "reach4", nagashi: "nm" }[
          outcome.draw
        ]
      }"`;
    const hands = outcome.tenpaiHands
      .map((h) => ` hai${h.who}="${h.hand.join(",")}"`)
      .join("");
    parts.push(
      `<RYUUKYOKU${type} ba="${round.honba},${round.kyotaku}" sc="${sc.join(",")}"${hands}/>`,
    );
  }
  return parts.join("");
}

export function toTenhouXml(m: MatchResult, cfg: RuleConfig): string {
  const g: Game = m.game;
  const names = g.players.map((p) => encodeURIComponent(p.name));
  const parts: string[] = [`<mjloggm ver="2.3">`];
  parts.push(`<SHUFFLE seed="mjgame-sfc32,${m.seed}"/>`);
  parts.push(`<GO type="${goType(cfg)}" lobby="0"/>`);
  parts.push(
    `<UN n0="${names[0]}" n1="${names[1]}" n2="${names[2]}" n3="${names[3]}" ` +
      `dan="0,0,0,0" rate="1500,1500,1500,1500" sx="C,C,C,C"/>`,
  );
  parts.push(`<TAIKYOKU oya="0"/>`);
  for (let i = 0; i < m.rounds.length; i++) {
    parts.push(roundXml(m.rounds[i], m.outcomes[i], m.rounds[i].dealer as Seat));
  }
  parts.push(`</mjloggm>`);
  return parts.join("");
}

export interface Sidecar {
  seed: number;
  ruleset: "janki";
  akaIds: number[];
  scores: number[];
  violations: Violation[];
}

export function toSidecar(m: MatchResult, cfg: RuleConfig): Sidecar {
  return {
    seed: m.seed,
    ruleset: "janki",
    akaIds: [...cfg.akaIds],
    scores: m.scores,
    violations: m.ledger,
  };
}

/** Write `<base>.xml` and `<base>.mjgame.json` side by side. */
export async function writeMatch(base: string, m: MatchResult, cfg: RuleConfig): Promise<void> {
  await Deno.writeTextFile(`${base}.xml`, toTenhouXml(m, cfg));
  await Deno.writeTextFile(
    `${base}.mjgame.json`,
    JSON.stringify(toSidecar(m, cfg), null, 2) + "\n",
  );
}

export type { Tile };
