// CLI entry: `play` (interactive TUI), `selfplay` (headless stats), `bench`.

import { RandomPolicy } from "./ai/random.ts";
import { runMatch, runMatchSync } from "./match.ts";
import { dojoHooks } from "./dojo.ts";
import type { MatchResult } from "./match.ts";
import { DOJO_DEFAULT, DOJO_HEADLESS, JANKI } from "./rules.ts";
import type { Policy, SyncPolicy } from "./policy.ts";
// Real yaku + fu scoring.
import { scorer } from "./score.ts";
import { SEATS } from "./types.ts";
import type { Seat } from "./types.ts";
import { App, PacedPolicy } from "./tui/app.ts";
import type { GlyphMode } from "./tui/glyph.ts";
import * as term from "./tui/term.ts";

// ---------------------------------------------------------------------------
// argument parsing (no std dependency — this project has zero third-party deps)
// ---------------------------------------------------------------------------

interface Args {
  cmd: string;
  seed: number;
  games: number;
  glyphs: GlyphMode;
  speed: number;
  /** Free allowance granted afresh every turn. */
  timerTurn: number;
  /** One pool for the whole match, spent only after the turn allowance. */
  timerBank: number;
  noIntro: boolean;
  help: boolean;
}

function parseArgs(argv: string[]): Args {
  const a: Args = {
    cmd: "",
    seed: Math.floor(Math.random() * 0x7fffffff),
    games: 100,
    glyphs: "kanji",
    speed: 350,
    timerTurn: 3_000,
    timerBank: 10_000,
    noIntro: false,
    help: false,
  };
  for (const arg of argv) {
    if (arg === "--help" || arg === "-h") a.help = true;
    else if (arg.startsWith("--seed=")) a.seed = Number(arg.slice(7));
    else if (arg.startsWith("--games=")) a.games = Number(arg.slice(8));
    else if (arg.startsWith("--speed=")) a.speed = Number(arg.slice(8));
    else if (arg === "--no-intro") a.noIntro = true;
    else if (arg.startsWith("--timer=")) {
      // "10+3" — a 10s match bank, plus 3s granted fresh each turn.
      const m = arg.slice(8).match(/^(\d+)(?:\+(\d+))?$/);
      if (!m) die(`--timer は 10+3 の形式: ${arg.slice(8)}`);
      a.timerBank = Number(m[1]) * 1000;
      a.timerTurn = Number(m[2] ?? 0) * 1000;
    } else if (arg.startsWith("--glyphs=")) {
      const v = arg.slice(9);
      if (v !== "ascii" && v !== "kanji") die(`--glyphs は ascii か kanji: ${v}`);
      a.glyphs = v;
    } else if (arg.startsWith("-")) die(`不明なオプション: ${arg}`);
    else if (!a.cmd) a.cmd = arg;
    else die(`余分な引数: ${arg}`);
  }
  return a;
}

function die(msg: string): never {
  console.error(msg);
  Deno.exit(2);
}

const USAGE = `mjgame — 雀鬼流ルールの4人麻雀 (人間1 + CPU3)

  deno run --allow-read --allow-write src/main.ts <command> [options]

コマンド:
  play       半荘を1回プレイする (端末UI)
  selfplay   CPU同士の対局を回して着順統計を出す
  bench      同上、半荘/秒だけを出す

オプション:
  --seed=N            乱数シード (再現用)
  --games=N           selfplay / bench の対局数 (既定 100)
  --glyphs=ascii      CJKフォントが弱い端末向けの2桁ASCII表記 (既定 kanji)
  --speed=MS          play でのCPU1手あたりの待ち時間 (既定 350)
  --timer=10+3        持ち時間: 半荘通しの持ち時間10秒 + 毎打3秒 (既定)。
                      毎打の3秒を超えると持ち時間を消費し、使い切ると
                      マイナス表示になる。打牌は強制されず、遅さの代償は
                      雀鬼流の長考ペナルティのみ
  --no-intro          開始演出と配牌アニメを飛ばす
  --help, -h          このヘルプ
`;

// ---------------------------------------------------------------------------
// play
// ---------------------------------------------------------------------------

async function cmdPlay(a: Args): Promise<void> {
  // Without a terminal there is no keyboard, so the first decision would block
  // forever behind a hidden alt-screen. Refuse with a hint instead of hanging.
  if (!term.isTty()) {
    die(
      "play は端末上でのみ動作します (標準入出力が tty ではありません)。\n" +
        "CPU同士の対局は `selfplay` を使ってください。",
    );
  }
  const names = ["あなた", "CPU東", "CPU南", "CPU西"];
  const app = new App({
    glyphs: a.glyphs,
    aka: JANKI.akaIds,
    names,
    thinkLimitMs: DOJO_DEFAULT.thinkLimitMs,
    timerTurnMs: a.timerTurn,
    timerBankMs: a.timerBank,
    cpuDelayMs: Math.max(0, a.speed),
    cfg: JANKI,
    noIntro: a.noIntro,
  });

  const cpu = (s: Seat) =>
    new PacedPolicy(new RandomPolicy(names[s], a.seed * 4 + s), () => app.paceDelay());
  const policies: Policy[] = [app.human, cpu(1), cpu(2), cpu(3)];

  try {
    app.start();
    await app.intro();
    const result = await runMatch(policies, {
      seed: a.seed,
      cfg: JANKI,
      dojo: DOJO_DEFAULT,
      scorer,
      players: SEATS.map((seat) => ({ seat, name: names[seat] })),
      // The human seat sees events through `notify`; CPU seats ignore them.
      sink: (e) => app.onEvent(e),
    });
    await app.showFinal(result);
  } finally {
    // Belt and braces: `term.leave()` is also wired to signals and `unload`.
    app.stop();
    term.leave();
  }
}

// ---------------------------------------------------------------------------
// selfplay / bench
// ---------------------------------------------------------------------------

function headless(games: number, seed: number): { results: MatchResult[]; ms: number } {
  const results: MatchResult[] = [];
  const t0 = performance.now();
  for (let g = 0; g < games; g++) {
    const s = seed + g;
    const policies: SyncPolicy[] = SEATS.map((seat) => new RandomPolicy(`R${seat}`, s * 4 + seat));
    // Without the hooks the ledger is always empty and the stats line would
    // report "違反 0件" no matter what actually happened.
    const hooks = dojoHooks({ dojo: DOJO_HEADLESS, oracle: scorer });
    results.push(runMatchSync(policies, {
      seed: s,
      cfg: JANKI,
      dojo: DOJO_HEADLESS,
      scorer,
      ...hooks,
    }));
  }
  return { results, ms: performance.now() - t0 };
}

function cmdSelfplay(a: Args): void {
  const { results, ms } = headless(a.games, a.seed);
  const place = SEATS.map(() => [0, 0, 0, 0]);
  const total = [0, 0, 0, 0];
  let rounds = 0;
  let violations = 0;
  for (const r of results) {
    rounds += r.rounds.length;
    violations += r.ledger.length;
    const order = r.scores
      .map((s, seat) => ({ seat, s }))
      .sort((x, y) => y.s - x.s || x.seat - y.seat);
    order.forEach((o, i) => place[o.seat][i]++);
    for (const s of SEATS) total[s] += r.scores[s];
  }
  console.log(`対局数 ${a.games}  (seed ${a.seed}..${a.seed + a.games - 1})`);
  console.log(`局数 ${rounds}  平均 ${(rounds / a.games).toFixed(2)} 局/半荘`);
  console.log(`違反 ${violations}件`);
  console.log("");
  console.log("席   1位   2位   3位   4位   平均順位   平均点");
  for (const s of SEATS) {
    const p = place[s];
    const avgRank = p.reduce((acc, n, i) => acc + n * (i + 1), 0) / a.games;
    const avgPts = total[s] / a.games;
    const cells = p.map((n) => String(n).padStart(5)).join(" ");
    console.log(
      `P${s} ${cells}   ${avgRank.toFixed(3).padStart(8)}   ${avgPts.toFixed(0).padStart(7)}`,
    );
  }
  console.log("");
  console.log(`所要 ${(ms / 1000).toFixed(2)}s  (${(a.games / (ms / 1000)).toFixed(1)} 半荘/秒)`);
}

function cmdBench(a: Args): void {
  const { results, ms } = headless(a.games, a.seed);
  const rounds = results.reduce((n, r) => n + r.rounds.length, 0);
  const secs = ms / 1000;
  console.log(`${a.games} 半荘 / ${rounds} 局 を ${secs.toFixed(3)}s`);
  console.log(`${(a.games / secs).toFixed(1)} 半荘/秒   ${(rounds / secs).toFixed(0)} 局/秒`);
}

// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const a = parseArgs(Deno.args);
  if (a.help || a.cmd === "" || a.cmd === "help") {
    console.log(USAGE);
    return;
  }
  switch (a.cmd) {
    case "play":
      await cmdPlay(a);
      break;
    case "selfplay":
      cmdSelfplay(a);
      break;
    case "bench":
      cmdBench(a);
      break;
    default:
      die(`不明なコマンド: ${a.cmd}\n\n${USAGE}`);
  }
}

if (import.meta.main) {
  try {
    await main();
  } finally {
    term.leave();
  }
}
