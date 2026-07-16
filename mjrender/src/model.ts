// Domain model for a parsed Tenhou mjlog (4-player).
//
// A tile is an integer 0..135. Its "type" (0..33) is `id >> 2`:
//   0-8   man 1-9      9-17  pin 1-9      18-26 sou 1-9
//   27-30 winds E,S,W,N (東南西北)         31-33 dragons 白發中
// The four physical copies of a type are ids `type*4 .. type*4+3`.

export type Tile = number;

/** Ruleset flags decoded from the GO `type` bitfield. */
export interface RuleSet {
  raw: number;
  aka: boolean; // red fives (aka dora) in play
  kuitan: boolean; // open tanyao allowed
  sanma: boolean; // 3-player
  hanchan: boolean; // true = East+South, false = East-only (tonpuu)
}

export interface PlayerInfo {
  seat: number; // 0..3
  name: string;
  dan?: string;
  rate?: number;
  sex?: string;
}

export type MeldKind =
  | "chi"
  | "pon"
  | "daiminkan" // open kan (called from a discard)
  | "shouminkan" // added kan (upgrade a pon)
  | "ankan" // concealed kan
  | "nuki"; // kita (sanma only)

export interface Meld {
  kind: MeldKind;
  who: number; // caller seat
  fromWho: number; // seat the called tile came from (== who for ankan/nuki)
  tiles: Tile[]; // all tiles forming the meld (sorted ascending)
  calledTile: Tile; // the tile that was taken / added
}

/** An ordered play event within a round. */
export type GameEvent =
  | { t: "draw"; who: number; tile: Tile; rinshan: boolean }
  | { t: "discard"; who: number; tile: Tile; tsumogiri: boolean; riichi: boolean }
  | { t: "call"; meld: Meld }
  | { t: "reach"; who: number; step: 1 | 2; scores?: number[] }
  | { t: "dora"; indicator: Tile };

export interface AgariResult {
  kind: "agari";
  who: number; // winner seat
  fromWho: number; // discarder seat (== who ⇒ tsumo)
  machi: Tile; // winning tile
  hand: Tile[]; // winner's final concealed tiles (from AGARI `hai`)
  melds: Meld[]; // winner's called melds at win time
  fu: number;
  points: number; // base points before ba/honba adjustments
  limit: number; // 0 normal, 1 mangan, 2 haneman, 3 baiman, 4 yakuman, 5 kazoe
  yaku: Array<{ id: number; han: number }>;
  yakuman: number[]; // yakuman ids (if any)
  doraHai: Tile[]; // dora indicators
  uraDoraHai: Tile[]; // ura-dora indicators
  sc: number[]; // score deltas: 4 pairs (before, delta), units of 100
}

export interface RyuukyokuResult {
  kind: "ryuukyoku";
  type?: string; // e.g. "yao9", "kaze4", "nm", "ron3" (undefined = exhaustive draw)
  sc: number[]; // tenpai/noten payments: 4 pairs (before, delta)
  tenpaiHands: Array<{ who: number; hand: Tile[] }>; // revealed tenpai hands
}

export type RoundResult = AgariResult | RyuukyokuResult;

export interface Round {
  kyoku: number; // 0=East1, 1=East2, ..., 4=South1, ...
  honba: number;
  kyotaku: number; // deposited riichi sticks (kyotaku) at round start
  dealer: number; // oya seat 0..3
  dice: [number, number];
  startScores: number[]; // 4 seats, units of 100
  startHands: Tile[][]; // 4 seats, 13 tiles each (in absolute seat order)
  firstDora: Tile; // initial dora indicator
  events: GameEvent[];
  results: RoundResult[]; // usually 1; multiple = double/triple ron
}

export interface Game {
  version: string;
  rules: RuleSet;
  players: PlayerInfo[]; // index = seat
  rounds: Round[];
  owari?: number[]; // final scores + placement points (pairs), from the last AGARI/RYUUKYOKU
}

export interface RenderOptions {
  hands: "key" | "all"; // reconstructed-hand verbosity
  /** "inline" embeds a full board snapshot above every anchor (for tool-less
   *  consumption); default "none" keeps the transcript lean — the LLM recalls
   *  positions via get_snapshot instead. */
  snapshots?: "none" | "inline";
}

/** What a commentary anchor is asking the LLM to judge. */
export type BeatKind =
  | "配牌評価"
  | "リーチ判断"
  | "押し引き"
  | "副露判断"
  | "局総括"
  | "流局評価"
  | "終局総括";

/**
 * A commentary beat: one 〔解説ポイント#N〕 anchor, addressable from outside.
 * `eventIndex` is a replayTo position within the round: -1 = right after the
 * deal, `events.length - 1` = end of the round.
 */
export interface Beat {
  id: number; // 1-based, sequential over the whole game
  kind: BeatKind;
  round: number; // index into game.rounds
  junme: number;
  seat?: number; // acting seat, when the beat is about one player
  eventIndex: number;
  topic: string; // anchor topic text as rendered
}
