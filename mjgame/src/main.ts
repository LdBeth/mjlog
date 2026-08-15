// CLI entry: `play` (interactive TUI), `selfplay` (headless stats), `bench`.

import { RandomPolicy } from "./ai/random.ts";
import { HeuristicPolicy } from "./ai/heuristic.ts";
import { NeuralPolicy } from "./rl/policy.ts";
import { RecordingPolicy, TrajectoryWriter, writeMatchEnd } from "./rl/record.ts";
import { encodeOracle } from "./rl/features.ts";
import { runMatch, runMatchSync } from "./match.ts";
import { dojoHooks } from "./dojo.ts";
import type { MatchResult } from "./match.ts";
import { DOJO_DEFAULT, DOJO_HEADLESS, JANKI } from "./rules.ts";
import type { DojoConfig } from "./rules.ts";
import type { Policy, SyncPolicy } from "./policy.ts";
import { sfc32 } from "./rng.ts";
// Real yaku + fu scoring.
import { scorer } from "./score.ts";
import type { Table } from "./table.ts";
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
  /**
   * CPU kinds, "h" heuristic, "r" random or "n" neural. selfplay/bench read one
   * char per absolute seat (seat 0 first). play deals the first three chars to
   * the CPU seats in seat order, skipping wherever the human landed — so
   * "nhhh" always yields exactly one neural CPU regardless of the human's seat.
   */
  seats: string;
  /** Manifest for the "n" seats. */
  weights: string;
  /** Softmax temperature for the "n" seats; 0 keeps them deterministic. */
  temp: number;
  /** When set, self-play writes a trajectory JSONL here. */
  record: string;
  help: boolean;
}

/** Where `--weights` points by default: what the trainer writes. */
const DEFAULT_WEIGHTS = "weights/manifest.json";

/** Seat letter → policy. Seeded per seat so a match seed reproduces exactly. */
function makePolicy(
  kind: string,
  name: string,
  seed: number,
  weights = DEFAULT_WEIGHTS,
  temp = 0,
): SyncPolicy {
  if (kind === "r") return new RandomPolicy(name, seed);
  if (kind === "n") {
    // Eager load: a seat that cannot think is better refused at startup than
    // discovered mid-hanchan.
    try {
      return new NeuralPolicy(name, seed, weights, { temperature: temp });
    } catch (e) {
      die(
        `${e instanceof Error ? e.message : String(e)}\n` +
          `n席 (学習済みポリシー) には重みが要ります。先に trainer を回すか、\n` +
          `\`python train/randinit.py\` で初期重みを作ってから --weights=PATH を指定してください。`,
      );
    }
  }
  return new HeuristicPolicy(name, seed);
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
    seats: "hhhh",
    weights: DEFAULT_WEIGHTS,
    temp: 0,
    record: "",
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
    } else if (arg.startsWith("--weights=")) a.weights = arg.slice(10);
    else if (arg.startsWith("--temp=")) {
      const v = Number(arg.slice(7));
      if (!Number.isFinite(v) || v < 0) die(`--temp は 0 以上の実数: ${arg.slice(7)}`);
      a.temp = v;
    } else if (arg.startsWith("--record=")) a.record = arg.slice(9);
    else if (arg.startsWith("--seats=")) {
      const v = arg.slice(8);
      if (!/^[hrn]{1,4}$/.test(v)) die(`--seats は h, r, n を4文字まで: ${v}`);
      // Short forms repeat the last letter: "hr" ⇒ "hrrr".
      a.seats = v.padEnd(4, v[v.length - 1]);
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
  --seats=hrrn        CPUの種類: h=手作り評価関数, r=ランダム,
                      n=学習済みニューラルポリシー (既定 hhhh)。
                      短く書くと最後の文字を繰り返す ("hr" ⇒ "hrrr")。
                      selfplay/bench では席番号ごと。play では人間の席を
                      飛ばして先頭3文字を席順に割り当てるので、"nhhh" なら
                      必ずAI(学習済み)が1人入る。n のCPUは AI東 のように表示
  --weights=PATH      n席が読む manifest.json (既定 weights/manifest.json)。
                      読めなければ起動時にエラー — trainer か train/randinit.py で作る
  --temp=T            n席の方策温度。0=決定的(既定)、1=PPO自己対戦のサンプリング。
                      正の値なら合法手のソフトマックスから席ごとの乱数で1手引く
  --record=PATH       selfplay の全判断を軌跡JSONL (trajectory) に書き出す。
                      1行1判断 ("d") + 局結果 ("r") + 半荘結果 ("m")。学習器の入力。
                      "d" 行には非対称critic用のオラクル情報 (他家3人の手牌・
                      残り山・裏ドラ = "o"、他家の向聴数 = "sh") も必ず入る。
                      1判断あたり向聴計算が3回増える分だけ遅くなる (推論側は不使用)
  --no-intro          開始演出と配牌アニメを飛ばす
  --help, -h          このヘルプ
`;

// ---------------------------------------------------------------------------
// dojo wiring (shared by every driver)
// ---------------------------------------------------------------------------

/**
 * The penalty registry's entry points. EVERY driver needs these: rules only
 * ever run from `onAction`/`onRoundEnd`, so a driver that omits them plays with
 * a permanently empty 違反台帳 — and with `Table.tsumogiriLock` never armed,
 * because that flag is set by a rule (ドラ切り後の手出し), not by the engine.
 *
 * `dojo` is deliberately a parameter: it must be the same config the round is
 * run with, or the ledger would judge by rules the round was not played under.
 */
export function makeDojoHooks(dojo: DojoConfig) {
  return dojoHooks({ dojo, oracle: scorer });
}

// ---------------------------------------------------------------------------
// play
// ---------------------------------------------------------------------------

/**
 * play's reading of `--seats`: the first three kinds are dealt to the three
 * non-human seats in seat order, so "nhhh" is one neural CPU no matter where
 * the human landed. (selfplay/bench read the same string per absolute seat.)
 */
export function cpuKindAt(cpu: string, humanSeat: Seat, seat: Seat): string {
  let i = 0;
  for (let s = 0; s < seat; s++) if (s !== humanSeat) i++;
  return cpu[i] ?? "h";
}

async function cmdPlay(a: Args): Promise<void> {
  // Without a terminal there is no keyboard, so the first decision would block
  // forever behind a hidden alt-screen. Refuse with a hint instead of hanging.
  if (!term.isTty()) {
    die(
      "play は端末上でのみ動作します (標準入出力が tty ではありません)。\n" +
        "CPU同士の対局は `selfplay` を使ってください。",
    );
  }
  // Which seat the player sits in comes from the match seed, so it is fixed for
  // a given `--seed` but is not always East-1's dealer. Forked with a tag so it
  // does not consume — or mirror — the match's own stream.
  const humanSeat = sfc32(a.seed).fork(0x5ea7).int(4) as Seat;
  const WINDS = ["東", "南", "西", "北"];
  // Neural CPUs announce themselves: facing the learned policy should be a
  // choice the player can see, not a surprise discovered by its style.
  const names = SEATS.map((s) =>
    s === humanSeat
      ? "あなた"
      : `${cpuKindAt(a.seats, humanSeat, s) === "n" ? "AI" : "CPU"}${WINDS[s]}`
  );
  const app = new App({
    glyphs: a.glyphs,
    aka: JANKI.akaIds,
    names,
    thinkLimitMs: DOJO_DEFAULT.thinkLimitMs,
    timerTurnMs: a.timerTurn,
    timerBankMs: a.timerBank,
    cpuDelayMs: Math.max(0, a.speed),
    cfg: JANKI,
    humanSeat,
    noIntro: a.noIntro,
  });

  const cpu = (s: Seat) =>
    new PacedPolicy(
      makePolicy(cpuKindAt(a.seats, humanSeat, s), names[s], a.seed * 4 + s, a.weights),
      () => app.paceDelay(),
    );
  const policies: Policy[] = SEATS.map((s) => s === humanSeat ? app.human : cpu(s));

  try {
    app.start();
    await app.intro();
    const result = await runMatch(policies, {
      seed: a.seed,
      cfg: JANKI,
      dojo: DOJO_DEFAULT,
      scorer,
      players: SEATS.map((seat) => ({ seat, name: names[seat] })),
      // Every public event, including each `violation` the hooks below file,
      // reaches the UI here — that is what fills the 違反台帳 panel.
      sink: (e) => app.onEvent(e),
      ...makeDojoHooks(DOJO_DEFAULT),
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

interface HeadlessOptions {
  /** Manifest path handed to any "n" seat. */
  weights?: string;
  /** Softmax temperature for any "n" seat; omitted or 0 = greedy. */
  temp?: number;
  /** Trajectory JSONL to record into; one writer for the whole run. */
  record?: string;
}

function headless(
  games: number,
  seed: number,
  seats = "hhhh",
  opts: HeadlessOptions = {},
): { results: MatchResult[]; ms: number; traj: { d: number; r: number; m: number } | null } {
  const results: MatchResult[] = [];
  // One file, one handle, every seat and every match of the run: the trainer
  // reads a single stream and the "r"/"m" lines terminate each match in it.
  const writer = opts.record ? new TrajectoryWriter(opts.record) : null;
  // The recorder's window onto hidden state. `runMatchSync` points it at the
  // round in play, so `ref.t` is non-null for the whole life of a decision;
  // outside a round nobody calls the tap.
  const ref: { t: Table | null } = { t: null };
  const t0 = performance.now();
  try {
    for (let g = 0; g < games; g++) {
      const s = seed + g;
      const policies: SyncPolicy[] = SEATS.map((seat) => {
        const p = makePolicy(
          seats[seat],
          `${seats[seat].toUpperCase()}${seat}`,
          s * 4 + seat,
          opts.weights,
          opts.temp,
        );
        // Record ONLY neural seats: ppo.py recomputes behavior logp from
        // --init, so a heuristic seat's "d" lines would be treated as samples
        // from the neural policy and silently poison every importance ratio.
        // Heuristic seats still play (and appear in "r"/"m" lines); they just
        // never emit decisions.
        return writer && seats[seat] === "n"
          ? new RecordingPolicy(p, writer, (sq) => encodeOracle(ref.t!, sq as Seat))
          : p;
      });
      // Without the hooks the ledger is always empty and the stats line would
      // report "違反 0件" no matter what actually happened.
      const r = runMatchSync(policies, {
        seed: s,
        cfg: JANKI,
        dojo: DOJO_HEADLESS,
        scorer,
        tableRef: writer ? ref : undefined,
        ...makeDojoHooks(DOJO_HEADLESS),
      });
      results.push(r);
      // Round and match lines close the match out: a policy never sees a
      // result, so only the driver can write them.
      if (writer) writeMatchEnd(writer, r, JANKI);
    }
    return { results, ms: performance.now() - t0, traj: writer?.stats() ?? null };
  } finally {
    writer?.close();
  }
}

function cmdSelfplay(a: Args): void {
  const { results, ms, traj } = headless(a.games, a.seed, a.seats, {
    weights: a.weights,
    temp: a.temp,
    record: a.record || undefined,
  });
  const place = SEATS.map(() => [0, 0, 0, 0]);
  const total = [0, 0, 0, 0];
  const wins = [0, 0, 0, 0];
  const deals = [0, 0, 0, 0]; // 放銃
  const tenpai = [0, 0, 0, 0]; // 流局時聴牌
  const vio = [0, 0, 0, 0];
  const riichis = [0, 0, 0, 0]; // 立直を掛けた局数
  const furo = [0, 0, 0, 0]; // 副露した局数 (暗槓は門前なので除く)
  const winPts = [0, 0, 0, 0]; // 和了局の実収支合計 (本場・供託込み)
  const winN = [0, 0, 0, 0];
  const dealPts = [0, 0, 0, 0]; // 放銃局の実支出合計
  const dealN = [0, 0, 0, 0];
  let rounds = 0;
  let draws = 0;
  for (const r of results) {
    rounds += r.rounds.length;
    for (const v of r.ledger) vio[v.seat]++;
    for (const s of SEATS) {
      riichis[s] += (r.riichis ?? [0, 0, 0, 0])[s];
      furo[s] += (r.furoRounds ?? [0, 0, 0, 0])[s];
    }
    for (const o of r.outcomes) {
      if (o.kind === "agari") {
        for (const w of o.wins) {
          wins[w.who]++;
          if (w.fromWho !== w.who) deals[w.fromWho]++;
        }
        for (const w of o.wins) {
          winN[w.who]++;
          winPts[w.who] += o.deltas[w.who];
        }
        // 放銃打点は「振り込んだ局」単位。ダブロンは1局1回だけ数え、
        // その局の支出 (両家ぶん) をまとめて負担額とする。
        const head = o.wins[0];
        if (head && head.fromWho !== head.who) {
          dealN[head.fromWho]++;
          dealPts[head.fromWho] += -o.deltas[head.fromWho];
        }
      } else {
        draws++;
        o.tenpai.forEach((t, s) => {
          if (t) tenpai[s]++;
        });
      }
    }
    const order = r.scores
      .map((s, seat) => ({ seat, s }))
      .sort((x, y) => y.s - x.s || x.seat - y.seat);
    order.forEach((o, i) => place[o.seat][i]++);
    for (const s of SEATS) total[s] += r.scores[s];
  }
  const pct = (n: number, d: number) => d === 0 ? "  -  " : `${(100 * n / d).toFixed(1)}%`;

  console.log(`対局数 ${a.games}  (seed ${a.seed}..${a.seed + a.games - 1})  席 ${a.seats}`);
  console.log(`局数 ${rounds}  平均 ${(rounds / a.games).toFixed(2)} 局/半荘  流局 ${draws}`);
  console.log(`違反 ${vio.reduce((x, y) => x + y, 0)}件`);
  console.log("");
  console.log("席     1位   2位   3位   4位   平均順位   平均点    和了率  放銃率  聴牌率  違反");
  for (const s of SEATS) {
    const p = place[s];
    const avgRank = p.reduce((acc, n, i) => acc + n * (i + 1), 0) / a.games;
    const avgPts = total[s] / a.games;
    const cells = p.map((n) => String(n).padStart(5)).join(" ");
    console.log(
      `${a.seats[s]}P${s} ${cells}   ${avgRank.toFixed(3).padStart(8)}   ` +
        `${avgPts.toFixed(0).padStart(7)}   ${pct(wins[s], rounds).padStart(6)}  ` +
        `${pct(deals[s], rounds).padStart(6)}  ${pct(tenpai[s], draws).padStart(6)}  ` +
        `${String(vio[s]).padStart(4)}`,
    );
  }
  // Second table, appended rather than folded into the first: the rows above are
  // grepped by the training scripts and must stay byte-identical.
  const mean = (sum: number, n: number) => n === 0 ? "-" : String(Math.round(sum / n));
  console.log("");
  console.log("席     リーチ率  副露率  平均和了打点  平均放銃打点");
  for (const s of SEATS) {
    console.log(
      `${a.seats[s]}P${s}${pct(riichis[s], rounds).padStart(12)}` +
        `${pct(furo[s], rounds).padStart(8)}` +
        `${mean(winPts[s], winN[s]).padStart(14)}` +
        `${mean(dealPts[s], dealN[s]).padStart(14)}`,
    );
  }
  console.log("");
  // Placement here is by raw score. The dojo's own ranking puts every violator
  // below every clean player, so read the 違反 column alongside 平均順位.
  console.log(`所要 ${(ms / 1000).toFixed(2)}s  (${(a.games / (ms / 1000)).toFixed(1)} 半荘/秒)`);
  if (traj) {
    console.log(
      `軌跡 ${a.record}: 判断 ${traj.d}行  局 ${traj.r}行  半荘 ${traj.m}行`,
    );
  }
}

function cmdBench(a: Args): void {
  const { results, ms } = headless(a.games, a.seed, a.seats, { weights: a.weights });
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
