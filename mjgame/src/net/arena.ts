// riichi.dev (RiichiLab) MJAI arena client — the transport half of the bridge.
//
// The arena owns the game: it streams MJAI events over a WebSocket and the bot
// answers ONLY `request_action` messages, whose `possible_actions` list is the
// authoritative legal set. The one unforgivable failure is answering with an
// action outside that list (chombo, recorded on the bot's profile) — so a
// chooser here never invents an action: it SELECTS one of the server's own
// `possible_actions` objects and echoes it back with `request_id`/`actor`
// attached. A chooser that comes back empty-handed degrades to `none`/first
// option, never to silence (timeouts are safe — the server tsumogiri-passes —
// but a dropped reply wastes bank for nothing).
//
// Endpoints (docs: riichi.dev/docs/protocol, /docs/validation):
//   wss://game.riichi.dev/ws/validate  — one East-only game vs 3 tsumogiri
//   wss://game.riichi.dev/ws/ranked    — full hanchan, matchmade
// Auth is `Authorization: Bearer <token>`, which plain WebSocket cannot set —
// hence WebSocketStream (Deno `--unstable-net`).

import { die } from "../cli/die.ts";
import { loadKtune } from "../harness.ts";

/** One entry of `request_action.possible_actions`, plus the reply fields. */
export interface MjaiAction {
  type: string;
  pai?: string;
  target?: number;
  consumed?: string[];
  actor?: number;
  tsumogiri?: boolean;
  request_id?: number;
  [k: string]: unknown;
}

export interface MjaiEvent {
  type: string;
  [k: string]: unknown;
}

export interface RequestAction extends MjaiEvent {
  type: "request_action";
  request_id: number;
  possible_actions: MjaiAction[];
  time?: { grace_ms: number; bank_ms: number; deadline_ms: number };
  observation?: string; // base64 RiichiEnv blob — unused; we track raw events
}

/**
 * A seat brain. `onEvent` sees every server event in arrival order (including
 * `request_action` itself and `action_ack`); `choose` must return ONE MEMBER of
 * `req.possible_actions` (object identity not required, but every field the
 * entry carries must be reproduced). Transport adds request_id/actor/tsumogiri.
 */
export interface ArenaChooser {
  onEvent(e: MjaiEvent): void;
  choose(req: RequestAction): MjaiAction;
  /** Release whatever the brain holds (native nets, policies). */
  close?(): void;
}

/**
 * The validation-grade fallback brain: win when the server itself says a win
 * is on the table, otherwise discard the tile just drawn, otherwise pass.
 * No calls, no riichi, no state beyond the last self-draw — nothing to get
 * out of sync. Also the safety net the champion bridge falls back to when its
 * shadow state cannot match an offered action.
 */
export class TsumogiriChooser implements ArenaChooser {
  private me = -1;
  private drawn: string | null = null;

  onEvent(e: MjaiEvent): void {
    if (e.type === "start_game") this.me = Number(e.id ?? -1);
    if (e.type === "tsumo" && e.actor === this.me) this.drawn = String(e.pai);
    if (e.type === "dahai" && e.actor === this.me) this.drawn = null;
    if (e.type === "start_kyoku") this.drawn = null;
  }

  choose(req: RequestAction): MjaiAction {
    const acts = req.possible_actions;
    const hora = acts.find((a) => a.type === "hora");
    if (hora) return hora;
    if (this.drawn !== null) {
      const cut = acts.find((a) => a.type === "dahai" && a.pai === this.drawn);
      if (cut) return cut;
    }
    return acts.find((a) => a.type === "none") ??
      acts.find((a) => a.type === "dahai") ?? acts[0];
  }
}

/**
 * The server hard-drops the socket after `end_game` (documented behaviour),
 * which surfaces here as a WebSocketError mid-read. Once a game has completed
 * that is a normal goodbye, not a failure; before one, it is a real transport
 * error and propagates.
 */
async function* iterateUntilServerDrop(
  readable: ReadableStream<string | Uint8Array>,
  gameDone: () => boolean,
  log: (line: string) => void,
): AsyncGenerator<string | Uint8Array> {
  try {
    for await (const raw of readable) yield raw;
  } catch (err) {
    if (!gameDone()) throw err;
    log(`arena: 対局後のサーバ切断 (${err instanceof Error ? err.name : err})`);
  }
}

export interface ArenaRunResult {
  games: number;
  /** Final `end_game.scores` of the last game, if one completed. */
  scores: number[] | null;
  /** action_ack statuses other than "accepted", for the postmortem. */
  rejections: string[];
}

/**
 * Connect, play until the server closes the stream (validation: one game;
 * ranked: the server disconnects after `end_game`). Returns a summary; throws
 * on transport errors before any game completed.
 */
export async function runArena(
  url: string,
  token: string,
  chooser: ArenaChooser,
  log: (line: string) => void = console.error,
  /** Raw wire tap: every inbound message and outbound reply, verbatim JSON. */
  trace?: (dir: "<" | ">", json: string) => void,
): Promise<ArenaRunResult> {
  const wss = new WebSocketStream(url, {
    headers: { Authorization: `Bearer ${token}` },
  });
  const { readable, writable } = await wss.opened;
  const writer = writable.getWriter();
  const out: ArenaRunResult = { games: 0, scores: null, rejections: [] };
  let seat = -1;

  try {
    for await (const raw of iterateUntilServerDrop(readable, () => out.games > 0, log)) {
      const text = typeof raw === "string" ? raw : new TextDecoder().decode(raw);
      trace?.("<", text);
      let e: MjaiEvent;
      try {
        e = JSON.parse(text);
      } catch {
        log(`arena: 解析不能なメッセージを無視: ${text.slice(0, 120)}`);
        continue;
      }

      chooser.onEvent(e);
      switch (e.type) {
        case "start_game":
          seat = Number(e.id ?? -1);
          log(`arena: start_game 席=${seat}`);
          break;
        case "start_kyoku":
          log(`arena: start_kyoku ${e.bakaze}${e.kyoku}局 本場${e.honba}`);
          break;
        case "action_ack": {
          const status = String(e.status);
          if (status !== "accepted") {
            out.rejections.push(`${e.request_id}:${status}`);
            log(
              `arena: action_ack ${status} (request_id=${e.request_id})` +
                (e.reason ? ` 理由: ${e.reason}` : ""),
            );
          }
          break;
        }
        case "hora":
          log(`arena: hora actor=${e.actor}` + (e.target !== undefined ? ` ←${e.target}` : ""));
          break;
        case "ryukyoku":
          log(`arena: ryukyoku ${e.reason ?? ""}`);
          break;
        case "end_kyoku":
          break;
        case "end_game":
          // riichi.dev sends a BARE end_game (no scores, unlike the docs);
          // final standings live in the accumulated hora/ryukyoku deltas.
          out.games++;
          out.scores = Array.isArray(e.scores) ? e.scores.map(Number) : null;
          log(`arena: end_game${out.scores ? ` scores=${out.scores.join("/")}` : ""}`);
          break;
        case "error":
          log(`arena: サーバ error: ${JSON.stringify(e)}`);
          break;
        case "request_action": {
          const req = e as RequestAction;
          let act: MjaiAction;
          try {
            act = chooser.choose(req);
          } catch (err) {
            // A broken brain must not become a chombo: fall back in-protocol.
            log(`arena: chooser 例外、フォールバック: ${err}`);
            act = req.possible_actions.find((a) => a.type === "none") ??
              req.possible_actions[0];
          }
          const reply: MjaiAction = { ...act, request_id: req.request_id };
          if (reply.type !== "none" && reply.actor === undefined) reply.actor = seat;
          const body = JSON.stringify(reply);
          trace?.(">", body);
          await writer.write(body);
          break;
        }
      }
    }
  } finally {
    try {
      writer.releaseLock();
      wss.close();
    } catch { /* already closed by server */ }
  }
  return out;
}

// ---------------------------------------------------------------------------
// CLI: deno task arena --token-file=PATH [--ranked] [--games=N]
//        [--brain=champion|tsumogiri] [--ktune=PATH] [--log=PATH]
// ---------------------------------------------------------------------------

function argOf(args: string[], name: string): string | undefined {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit?.slice(name.length + 3);
}

async function main() {
  const args = Deno.args;
  const tokenFile = argOf(args, "token-file") ?? die("--token-file=PATH が必要");
  const ranked = args.includes("--ranked");
  const games = Number(argOf(args, "games") ?? "1");
  const logPath = argOf(args, "log");
  const brain = argOf(args, "brain") ?? "champion";
  if (brain !== "champion" && brain !== "tsumogiri") {
    die(`--brain は champion か tsumogiri: ${brain}`);
  }
  // Same convention as main.ts's 助言 seat: the CURRENT champion by default,
  // resolved relative to the module so the task runs from any cwd.
  const ktunePath = argOf(args, "ktune") ??
    new URL("../../weights/champion.json", import.meta.url).pathname;
  const token = (await Deno.readTextFile(tokenFile)).trim();
  const url = ranked ? "wss://game.riichi.dev/ws/ranked" : "wss://game.riichi.dev/ws/validate";

  const logFile = logPath ? await Deno.open(logPath, { create: true, append: true }) : null;
  const enc = new TextEncoder();
  const trace = logFile
    ? (dir: "<" | ">", json: string) => {
      logFile.writeSync(enc.encode(`${dir} ${json}\n`));
    }
    : undefined;

  const { ChampionChooser } = await import("./champion.ts");
  const ktune = brain === "champion" ? loadKtune(ktunePath) : undefined;
  // `games` counts COMPLETED games: a refused or dropped connection must not
  // consume the budget, or one bad minute burns the whole batch in seconds.
  let blanks = 0;
  for (let g = 0; g < games;) {
    // A fresh brain per game: the shadow is per-game state, and champion
    // construction is two plain objects (no native memory).
    const chooser: ArenaChooser = brain === "champion"
      ? new ChampionChooser({ ktune })
      : new TsumogiriChooser();
    let played = false;
    try {
      const r = await runArena(url, token, chooser, console.error, trace);
      if (r.rejections.length) {
        console.error(`arena: 却下された応答 ${r.rejections.length} 件: ${r.rejections.join(" ")}`);
      }
      if (chooser instanceof ChampionChooser && chooser.fallbacks > 0) {
        console.error(`arena: フォールバック発動 ${chooser.fallbacks} 局`);
      }
      played = r.games > 0;
      console.log(
        played
          ? `対局完了 (${g + 1}/${games}) 最終得点: ${r.scores?.join("/") ?? "?"}`
          : `対局は完了しなかった (${g + 1}/${games})`,
      );
    } catch (e) {
      // A transport failure mid-game used to kill the batch (main().catch →
      // die). Survive it: the server holds the abandoned game for a while
      // ("This bot is already connected to a game"), so back off before
      // asking for the next one.
      console.error(`arena: 対局が中断 (${String(e)})`);
    } finally {
      chooser.close?.();
    }
    if (played) {
      g++;
      blanks = 0;
      continue;
    }
    if (++blanks >= 6) {
      console.error(`arena: 対局が始まらない状態が続く — ${g}/${games} で中断`);
      break;
    }
    const wait = Math.min(30_000 * 2 ** (blanks - 1), 300_000);
    console.error(`arena: ${wait / 1000}秒待って再接続 (${blanks}回目)`);
    await new Promise((res) => setTimeout(res, wait));
  }
  logFile?.close();
}

// NOT `await main()`: main dynamically imports champion.ts, which statically
// imports this module — a top-level await here would block arena.ts's own
// evaluation, which that dynamic import waits on. Deadlock (Deno reports it
// as "Top-level await promise never resolved").
if (import.meta.main) main().catch((e) => die(String(e)));
