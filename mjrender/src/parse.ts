// Parse Tenhou mjlog XML into a faithful Game model (no analysis here).
//
// The log is a flat, ordered sequence of elements under <mjloggm>. Draw and
// discard tiles are encoded in the *element name* itself (e.g. <T81/> = seat 0
// draws tile 81; <D122/> = seat 0 discards tile 122).

import { XMLParser } from "fast-xml-parser";
import { decodeMeld } from "./meld.ts";
import { parseYaku, parseYakuman } from "./yaku.ts";
import type {
  AgariResult,
  Game,
  PlayerInfo,
  Round,
  RuleSet,
  RyuukyokuResult,
  Tile,
} from "./model.ts";

interface Node {
  tag: string;
  attrs: Record<string, string>;
}

const ATTR = "@_";

function toNodes(children: unknown[]): Node[] {
  return (children as Array<Record<string, unknown>>).flatMap((child) => {
    const tag = Object.keys(child).find((k) => k !== ":@");
    if (!tag) return [];
    const rawAttrs = (child[":@"] ?? {}) as Record<string, string>;
    const attrs: Record<string, string> = Object.fromEntries(
      Object.entries(rawAttrs).map(
        ([k, v]) => [k.startsWith(ATTR) ? k.slice(ATTR.length) : k, String(v)],
      ),
    );
    return [{ tag, attrs }];
  });
}

function intList(s: string | undefined): number[] {
  if (!s) return [];
  return s.split(/[\s,]+/).filter((x) => x.length).map(Number);
}

// GO `type` bitfield (gameplay-relevant bits; verified against sample type=169
// = hanchan / yonma / aka-ari / kuitan-ari):
//   0x02 set ⇒ no aka   0x04 set ⇒ no kuitan   0x08 ⇒ hanchan   0x10 ⇒ sanma
function decodeGoType(type: number): RuleSet {
  return {
    raw: type,
    aka: (type & 0x02) === 0,
    kuitan: (type & 0x04) === 0,
    hanchan: (type & 0x08) !== 0,
    sanma: (type & 0x10) !== 0,
  };
}

const DRAW_SEAT: Record<string, number> = { T: 0, U: 1, V: 2, W: 3 };
const DISCARD_SEAT: Record<string, number> = { D: 0, E: 1, F: 2, G: 3 };

export function parseGame(xml: string): Game {
  const parser = new XMLParser({
    ignoreAttributes: false,
    preserveOrder: true,
    attributeNamePrefix: ATTR,
    parseTagValue: false,
  });
  const doc = parser.parse(xml) as Array<Record<string, unknown>>;
  const rootNode = doc.find((n) => "mjloggm" in n);
  if (!rootNode) throw new Error("no <mjloggm> root element found");

  const version = String((rootNode[":@"] as Record<string, string>)?.[ATTR + "ver"] ?? "");
  const children = toNodes(rootNode["mjloggm"] as unknown[]);

  let rules: RuleSet = decodeGoType(0);
  let players: PlayerInfo[] = [];
  const rounds: Round[] = [];
  let owari: number[] | undefined;

  let cur: Round | undefined; // current round being built
  const lastDraw = [-1, -1, -1, -1]; // last drawn tile per seat (for tsumogiri)
  const reachDeclared = [false, false, false, false]; // step-1 seen, discard pending
  let rinshanPending = false;

  const finishOwari = (attrs: Record<string, string>) => {
    if (attrs.owari) owari = intList(attrs.owari);
  };

  for (const node of children) {
    const { tag, attrs } = node;

    // draw / discard tags carry the tile in the element name
    const rest = tag.slice(1);
    const drawSeat = DRAW_SEAT[tag[0]];
    if (drawSeat !== undefined && /^\d+$/.test(rest)) {
      const tile = Number(rest);
      lastDraw[drawSeat] = tile;
      cur?.events.push({ t: "draw", who: drawSeat, tile, rinshan: rinshanPending });
      rinshanPending = false;
      continue;
    }
    const discardSeat = DISCARD_SEAT[tag[0]];
    if (discardSeat !== undefined && /^\d+$/.test(rest)) {
      const tile = Number(rest);
      const tsumogiri = tile === lastDraw[discardSeat];
      const riichi = reachDeclared[discardSeat];
      reachDeclared[discardSeat] = false;
      cur?.events.push({ t: "discard", who: discardSeat, tile, tsumogiri, riichi });
      continue;
    }

    switch (tag) {
      case "GO":
        rules = decodeGoType(Number(attrs.type ?? 0));
        break;
      case "UN": {
        if (players.length === 0) {
          players = [0, 1, 2, 3].map((seat) => ({ seat, name: "" }));
        }
        const dan = attrs.dan?.split(",");
        const rate = attrs.rate?.split(",");
        const sx = attrs.sx?.split(",");
        for (let i = 0; i < 4; i++) {
          const n = attrs[`n${i}`];
          if (n === undefined) continue; // partial UN = reconnect
          players[i].name = safeDecode(n);
          if (dan?.[i]) players[i].dan = dan[i];
          if (rate?.[i]) players[i].rate = Number(rate[i]);
          if (sx?.[i]) players[i].sex = sx[i];
        }
        break;
      }
      case "INIT": {
        const seed = intList(attrs.seed);
        cur = {
          kyoku: seed[0] ?? 0,
          honba: seed[1] ?? 0,
          kyotaku: seed[2] ?? 0,
          dealer: Number(attrs.oya ?? 0),
          dice: [seed[3] ?? 0, seed[4] ?? 0],
          startScores: intList(attrs.ten),
          startHands: [0, 1, 2, 3].map((i) => intList(attrs[`hai${i}`])),
          firstDora: seed[5] ?? 0,
          events: [],
          results: [],
        };
        rounds.push(cur);
        lastDraw.fill(-1);
        reachDeclared.fill(false);
        rinshanPending = false;
        break;
      }
      case "N": {
        const who = Number(attrs.who);
        const meld = decodeMeld(who, Number(attrs.m));
        cur?.events.push({ t: "call", meld });
        if (meld.kind === "ankan" || meld.kind === "daiminkan" || meld.kind === "shouminkan") {
          rinshanPending = true;
        }
        break;
      }
      case "REACH": {
        const who = Number(attrs.who);
        const step = Number(attrs.step) === 2 ? 2 : 1;
        if (step === 1) reachDeclared[who] = true;
        cur?.events.push({
          t: "reach",
          who,
          step,
          scores: step === 2 ? intList(attrs.ten) : undefined,
        });
        break;
      }
      case "DORA":
        cur?.events.push({ t: "dora", indicator: Number(attrs.hai) });
        break;
      case "AGARI":
        cur?.results.push(parseAgari(attrs));
        finishOwari(attrs);
        break;
      case "RYUUKYOKU":
        cur?.results.push(parseRyuukyoku(attrs));
        finishOwari(attrs);
        break;
    }
  }

  return { version, rules, players, rounds, owari };
}

function parseAgari(attrs: Record<string, string>): AgariResult {
  const ten = intList(attrs.ten); // [fu, points, limit]
  const who = Number(attrs.who);
  const melds = (attrs.m ? attrs.m.split(",") : []).map((code) => decodeMeld(who, Number(code)));
  return {
    kind: "agari",
    who,
    fromWho: Number(attrs.fromWho),
    machi: Number(attrs.machi),
    hand: intList(attrs.hai) as Tile[],
    melds,
    fu: ten[0] ?? 0,
    points: ten[1] ?? 0,
    limit: ten[2] ?? 0,
    yaku: parseYaku(attrs.yaku),
    yakuman: parseYakuman(attrs.yakuman),
    doraHai: intList(attrs.doraHai) as Tile[],
    uraDoraHai: intList(attrs.doraHaiUra) as Tile[],
    sc: intList(attrs.sc),
  };
}

function parseRyuukyoku(attrs: Record<string, string>): RyuukyokuResult {
  const tenpaiHands = [0, 1, 2, 3].flatMap((who) => {
    const h = attrs[`hai${who}`];
    return h === undefined ? [] : [{ who, hand: intList(h) as Tile[] }];
  });
  return { kind: "ryuukyoku", type: attrs.type, sc: intList(attrs.sc), tenpaiHands };
}

function safeDecode(s: string): string {
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}
