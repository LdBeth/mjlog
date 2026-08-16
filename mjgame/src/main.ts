// CLI entry: `play` (interactive TUI), `selfplay` (headless stats), `bench`.

import { RandomPolicy } from "./ai/random.ts";
import { HeuristicPolicy } from "./ai/heuristic.ts";
import { AugmentedHeuristic, noisyReads, oracleReads, parseChannels } from "./ai/augmented.ts";
import type { OracleChannel } from "./ai/augmented.ts";
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
import { finalStandings, scorer } from "./score.ts";
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
   * CPU kinds, "h" heuristic, "r" random, "n" neural or "o" oracle-augmented.
   * selfplay/bench read one char per absolute seat (seat 0 first). play deals
   * the first three chars to the CPU seats in seat order, skipping wherever the
   * human landed — so "nhhh" always yields exactly one neural CPU regardless of
   * the human's seat. "o" is headless-only: it reads the live Table.
   */
  seats: string;
  /** Hidden-information channels an "o" seat is allowed to see. */
  oracle: Set<OracleChannel>;
  /** `--oracle` as typed, for the report line. */
  oracleSpec: string;
  /** Per-decision probability that an "o" seat loses each information group. */
  noise: number;
  /** Manifest for the "n" seats. */
  weights: string;
  /** Softmax temperature for the "n" seats; 0 keeps them deterministic. */
  temp: number;
  /** When set, self-play writes a trajectory JSONL here. */
  record: string;
  /**
   * Record EVERY seat, not just the "n" ones. Behavior-cloning data only: a
   * dataset with non-neural decisions violates ppo.py's on-policy assumption
   * (behavior logp is recomputed from --init) and would poison every
   * importance ratio — bc.py is the sole intended consumer.
   */
  recordAll: boolean;
  help: boolean;
}

/** Where `--weights` points by default: what the trainer writes. */
const DEFAULT_WEIGHTS = "weights/manifest.json";

/** The ablation's standing default: the three defensive channels. */
const DEFAULT_ORACLE = "C1,C2,C3";

/** What an "o" seat needs: the live-Table tap and the channels it may read. */
export interface OracleWiring {
  get: () => Table | null;
  channels: Set<OracleChannel>;
  /**
   * Oracle fading: per decision, per information group, the probability that the
   * truth is withheld. 0 (the default) leaves the provider untouched.
   */
  noise?: number;
}

/** Seat letter → policy. Seeded per seat so a match seed reproduces exactly. */
function makePolicy(
  kind: string,
  name: string,
  seed: number,
  weights = DEFAULT_WEIGHTS,
  temp = 0,
  oracle?: OracleWiring,
): SyncPolicy {
  if (kind === "r") return new RandomPolicy(name, seed);
  if (kind === "o") {
    // The oracle reads the round in play through `MatchOptions.tableRef`. There
    // is no such tap in `play` — and a CPU that could see the human's hand is
    // not something to offer by accident.
    if (!oracle) {
      die(
        "o席 (オラクル増補) は selfplay / bench / paired 専用です。\n" +
          "play では隠蔽情報を読む席は作れません。",
      );
    }
    const truth = oracleReads(oracle.get, scorer, oracle.channels);
    const eps = oracle.noise ?? 0;
    return new AugmentedHeuristic(name, seed, eps > 0 ? noisyReads(truth, eps) : truth);
  }
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
    oracle: parseChannels(DEFAULT_ORACLE)!,
    oracleSpec: DEFAULT_ORACLE,
    noise: 0,
    weights: DEFAULT_WEIGHTS,
    temp: 0,
    record: "",
    recordAll: false,
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
    } else if (arg === "--record-all") a.recordAll = true;
    else if (arg.startsWith("--record=")) a.record = arg.slice(9);
    else if (arg.startsWith("--oracle=")) {
      const v = arg.slice(9);
      const ch = parseChannels(v);
      if (!ch) die(`--oracle は C1..C6 / C7O / C7P のカンマ区切り (または none): ${v}`);
      a.oracle = ch;
      a.oracleSpec = v;
    } else if (arg.startsWith("--noise=")) {
      const v = Number(arg.slice(8));
      if (!Number.isFinite(v) || v < 0 || v > 1) die(`--noise は 0..1 の実数: ${arg.slice(8)}`);
      a.noise = v;
    } else if (arg.startsWith("--seats=")) {
      const v = arg.slice(8);
      if (!/^[hrno]{1,4}$/.test(v)) die(`--seats は h, r, n, o を4文字まで: ${v}`);
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
  paired     同一牌山で --seats と hhhh を2回ずつ回し、席0の対応差を出す
  bench      同上、半荘/秒だけを出す

オプション:
  --seed=N            乱数シード (再現用)
  --games=N           selfplay / paired / bench の対局数 (既定 100)。
                      paired では1シードにつき2半荘なので実対局数は2倍
  --glyphs=ascii      CJKフォントが弱い端末向けの2桁ASCII表記 (既定 kanji)
  --speed=MS          play でのCPU1手あたりの待ち時間 (既定 350)
  --timer=10+3        持ち時間: 半荘通しの持ち時間10秒 + 毎打3秒 (既定)。
                      毎打の3秒を超えると持ち時間を消費し、使い切ると
                      マイナス表示になる。打牌は強制されず、遅さの代償は
                      雀鬼流の長考ペナルティのみ
  --seats=hrrn        CPUの種類: h=手作り評価関数, r=ランダム,
                      n=学習済みニューラルポリシー, o=オラクル増補 (既定 hhhh)。
                      短く書くと最後の文字を繰り返す ("hr" ⇒ "hrrr")。
                      selfplay/bench では席番号ごと。play では人間の席を
                      飛ばして先頭3文字を席順に割り当てるので、"nhhh" なら
                      必ずAI(学習済み)が1人入る。n のCPUは AI東 のように表示。
                      o は隠蔽情報 (他家の手牌・山) を直接読むので headless 専用
  --oracle=C1,C2,C3   o席が読んでよい情報チャネル (既定 C1,C2,C3)。
                      C1=放銃真値 C2=聴牌真値 C3=打点真値 C4=次のツモ
                      C5=次の槓ドラ C6=リーチ者の次のツモ。none で全部切る
                      (= h と完全に同一の打牌になる対照群)
                      C7O/C7P=最大利益ロックオン立案 (完成形を列挙し
                      P(完成)×打点 が最大の一つに狙いを固定する)。
                      C7O は山の残り牌構成と他家の手牌を真値で読み、
                      C7P は誰でも数えられる未見枚数だけを使う対照群
  --noise=E           オラクルの劣化度 0..1 (既定 0)。1判断ごとに、情報の
                      グループ (放銃/聴牌/打点/ツモ/ドラ/リーチ者ツモ/残り枚数)
                      それぞれを独立に確率Eで落とす。落ちたグループは「無い」
                      扱いになり、その項だけ手作り評価関数の推測に戻る。
                      E を振ると「どこまで読みが粗くなると優位が消えるか」が測れる。
                      E=1 では C7O が C7P と同じ挙動まで落ちる (立案機構は残る)
  --weights=PATH      n席が読む manifest.json (既定 weights/manifest.json)。
                      読めなければ起動時にエラー — trainer か train/randinit.py で作る
  --temp=T            n席の方策温度。0=決定的(既定)、1=PPO自己対戦のサンプリング。
                      正の値なら合法手のソフトマックスから席ごとの乱数で1手引く
  --record=PATH       selfplay の全判断を軌跡JSONL (trajectory) に書き出す。
                      1行1判断 ("d") + 局結果 ("r") + 半荘結果 ("m")。学習器の入力。
                      "d" 行には非対称critic用のオラクル情報 (他家3人の手牌・
                      残り山・裏ドラ = "o"、他家の向聴数 = "sh") も必ず入る。
                      1判断あたり向聴計算が3回増える分だけ遅くなる (推論側は不使用)
  --record-all        n席以外の判断も記録する (BC教師データ用)。ppo.py には
                      渡せない — 挙動方策が --init と一致する前提が壊れる
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
  /** Record every seat (BC teacher data), not just the "n" ones. */
  recordAll?: boolean;
  /** Hidden-information channels any "o" seat may read. */
  oracle?: Set<OracleChannel>;
  /** Per-decision, per-group dropout applied to those channels (0 = off). */
  noise?: number;
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
  // ...and the oracle seats' window onto the same thing. One tap serves both.
  const ref: { t: Table | null } = { t: null };
  const oracleSeats = seats.includes("o");
  const wiring: OracleWiring = {
    get: () => ref.t,
    channels: opts.oracle ?? new Set(),
    noise: opts.noise ?? 0,
  };
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
          wiring,
        );
        // Record ONLY neural seats: ppo.py recomputes behavior logp from
        // --init, so a heuristic seat's "d" lines would be treated as samples
        // from the neural policy and silently poison every importance ratio.
        // Heuristic seats still play (and appear in "r"/"m" lines); they just
        // never emit decisions. `recordAll` overrides for BC teacher datasets,
        // whose consumer (bc.py) never computes a ratio.
        return writer && (opts.recordAll || seats[seat] === "n")
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
        tableRef: writer || oracleSeats ? ref : undefined,
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

/**
 * The oracle arm as one string for a report header: the channel set, plus the
 * fading level whenever it is on. Printed so a saved log says which point of the
 * ε sweep produced it — the numbers below it are meaningless without that.
 */
function oracleLabel(a: Args): string {
  const ch = [...a.oracle].sort().join(",") || "none";
  return a.noise > 0 ? `${ch} ノイズ ${a.noise}` : ch;
}

function cmdSelfplay(a: Args): void {
  const { results, ms, traj } = headless(a.games, a.seed, a.seats, {
    weights: a.weights,
    temp: a.temp,
    record: a.record || undefined,
    recordAll: a.recordAll,
    oracle: a.oracle,
    noise: a.noise,
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

  // The oracle arm is named only when a seat actually reads it, so an all-h run
  // — the line the training scripts grep — stays byte-identical.
  const arm = a.seats.includes("o") ? `  オラクル ${oracleLabel(a)}` : "";
  console.log(
    `対局数 ${a.games}  (seed ${a.seed}..${a.seed + a.games - 1})  席 ${a.seats}${arm}`,
  );
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

// ---------------------------------------------------------------------------
// paired — the measuring instrument
// ---------------------------------------------------------------------------

/** Seat 0's placement by raw score. Ties go to the lower seat, as in selfplay. */
function rankOfSeat0(scores: number[]): number {
  let rank = 1;
  for (let s = 1; s < 4; s++) if (scores[s] > scores[0]) rank++;
  return rank;
}

/**
 * Seat 0's placement by the DOJO's own ranking: every seat carrying a ledger
 * entry finishes below every clean seat, whatever the scores say
 * (`score.ts::finalStandings`). This is the number the 雀鬼会 would read off the
 * table, and the one a compliance filter is actually optimising — a seat can
 * gain 8000 points and still lose the placement by filing one violation, which
 * the raw 順位差 line cannot show.
 *
 * 起家 is read off the first round rather than assumed: `match.ts::roundInit`
 * deals East-1 to seat 0 today, but the tie-break should not depend on that.
 */
function dojoRankOfSeat0(r: MatchResult): number {
  const east = (r.rounds[0]?.dealer ?? 0) as Seat;
  return finalStandings(r.scores, east, r.ledger, JANKI)
    .find((s) => s.seat === 0)!.place;
}

interface Diff {
  mean: number;
  sd: number;
  ci: number;
}

/** Sample mean, sample SD (n-1) and the 95% half-width 1.96·sd/√n. */
function summarize(xs: number[]): Diff {
  const n = xs.length;
  if (n === 0) return { mean: 0, sd: 0, ci: 0 };
  const mean = xs.reduce((a, b) => a + b, 0) / n;
  if (n < 2) return { mean, sd: 0, ci: 0 };
  const varSum = xs.reduce((a, x) => a + (x - mean) * (x - mean), 0);
  const sd = Math.sqrt(varSum / (n - 1));
  return { mean, sd, ci: 1.96 * sd / Math.sqrt(n) };
}

export interface PairedStats {
  games: number;
  seed: number;
  seats: string;
  /** Seat 0's mean rank / mean final score in the test arm and the control. */
  rankA: number;
  rankB: number;
  scoreA: number;
  scoreB: number;
  /** The same means under the dojo's ranking (violators below clean players). */
  rankDojoA: number;
  rankDojoB: number;
  /** Per-seed differences A−B: negative rank diff = the test arm placed better. */
  dRank: Diff;
  /** Same difference, ranked the way the dojo ranks. */
  dRankDojo: Diff;
  dScore: Diff;
  /** Sign counts on the rank diff. */
  better: number;
  tie: number;
  worse: number;
  /** Total ledger entries across all seats, per arm. */
  vioA: number;
  vioB: number;
  /**
   * Seat 0's ledger entries by rule label, per arm. Seat 0 is the only seat
   * whose policy differs between the arms, so this is the breakdown that says
   * WHICH rules an augmented seat trades away — the all-seat totals above
   * cannot (seats 1–3 drift too, because the games diverge).
   */
  vioA0: Map<string, number>;
  vioB0: Map<string, number>;
  ms: number;
}

/**
 * The paired ablation harness: every seed is played TWICE, once with `seats`
 * and once all-heuristic, with identical match seeds and identical per-seat
 * policy seeds. Identical seeds mean identical walls AND identical policy dice,
 * so the only thing that differs between the two arms is what seat 0 knows —
 * which is what makes a per-seed difference meaningful and the variance small
 * enough to see a rank effect in hundreds rather than tens of thousands of
 * games.
 *
 * Exported so a test (and any later sweep script) can measure without a CLI.
 */
export function pairedRun(
  games: number,
  seed: number,
  seats: string,
  opts: HeadlessOptions = {},
): PairedStats {
  const dRank: number[] = [];
  const dRankDojo: number[] = [];
  const dScore: number[] = [];
  let rankA = 0, rankB = 0, scoreA = 0, scoreB = 0, vioA = 0, vioB = 0;
  let rankDojoA = 0, rankDojoB = 0;
  let better = 0, tie = 0, worse = 0;
  const vioA0 = new Map<string, number>();
  const vioB0 = new Map<string, number>();
  const bump = (m: Map<string, number>, k: string) => m.set(k, (m.get(k) ?? 0) + 1);
  const t0 = performance.now();

  for (let g = 0; g < games; g++) {
    const s = seed + g;
    const a = headless(1, s, seats, opts).results[0];
    const b = headless(1, s, "hhhh", { weights: opts.weights }).results[0];
    const ra = rankOfSeat0(a.scores);
    const rb = rankOfSeat0(b.scores);
    const da = dojoRankOfSeat0(a);
    const db = dojoRankOfSeat0(b);
    rankA += ra;
    rankB += rb;
    rankDojoA += da;
    rankDojoB += db;
    dRankDojo.push(da - db);
    scoreA += a.scores[0];
    scoreB += b.scores[0];
    vioA += a.ledger.length;
    vioB += b.ledger.length;
    for (const v of a.ledger) if (v.seat === 0) bump(vioA0, v.label);
    for (const v of b.ledger) if (v.seat === 0) bump(vioB0, v.label);
    dRank.push(ra - rb);
    dScore.push(a.scores[0] - b.scores[0]);
    if (ra < rb) better++;
    else if (ra === rb) tie++;
    else worse++;
  }

  return {
    games,
    seed,
    seats,
    rankA: rankA / games,
    rankB: rankB / games,
    scoreA: scoreA / games,
    scoreB: scoreB / games,
    rankDojoA: rankDojoA / games,
    rankDojoB: rankDojoB / games,
    dRank: summarize(dRank),
    dRankDojo: summarize(dRankDojo),
    dScore: summarize(dScore),
    better,
    tie,
    worse,
    vioA,
    vioB,
    vioA0,
    vioB0,
    ms: performance.now() - t0,
  };
}

function cmdPaired(a: Args): void {
  if (a.games < 1) die("--games は1以上");
  const st = pairedRun(a.games, a.seed, a.seats, {
    weights: a.weights,
    temp: a.temp,
    oracle: a.oracle,
    noise: a.noise,
  });
  const sign = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(3);
  const signPt = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(0);

  console.log(
    `対局数 ${st.games}  (seed ${st.seed}..${st.seed + st.games - 1})  ` +
      `A席 ${st.seats} / B席 hhhh  オラクル ${oracleLabel(a)}`,
  );
  console.log("");
  console.log("腕     席0平均順位   道場順位   席0平均点   違反(全席)");
  console.log(
    `A ${st.seats}${st.rankA.toFixed(3).padStart(10)}${st.rankDojoA.toFixed(3).padStart(11)}` +
      `${st.scoreA.toFixed(0).padStart(12)}${String(st.vioA).padStart(13)}`,
  );
  console.log(
    `B hhhh${st.rankB.toFixed(3).padStart(10)}${st.rankDojoB.toFixed(3).padStart(11)}` +
      `${st.scoreB.toFixed(0).padStart(12)}${String(st.vioB).padStart(13)}`,
  );
  console.log("");
  // The headline. Negative = the augmented seat placed better on the same walls.
  console.log("対応差 (A−B, 同一牌山)   平均      SD      95%CI");
  console.log(
    `  順位差${sign(st.dRank.mean).padStart(18)}${st.dRank.sd.toFixed(3).padStart(9)}` +
      `   [${sign(st.dRank.mean - st.dRank.ci)}, ${sign(st.dRank.mean + st.dRank.ci)}]`,
  );
  // 素点順位ではなく、違反者を全員の下に落とす道場の順位で見た差。
  console.log(
    `  順位差(道場)${sign(st.dRankDojo.mean).padStart(12)}` +
      `${st.dRankDojo.sd.toFixed(3).padStart(9)}` +
      `   [${sign(st.dRankDojo.mean - st.dRankDojo.ci)}, ` +
      `${sign(st.dRankDojo.mean + st.dRankDojo.ci)}]`,
  );
  console.log(
    `  点差  ${signPt(st.dScore.mean).padStart(18)}${st.dScore.sd.toFixed(0).padStart(9)}` +
      `   [${signPt(st.dScore.mean - st.dScore.ci)}, ${signPt(st.dScore.mean + st.dScore.ci)}]`,
  );
  console.log("");
  console.log(`勝敗 A優位 ${st.better} / 同着 ${st.tie} / B優位 ${st.worse}`);
  // Which rules seat 0 (the only differing policy) actually trades away.
  const labels = [...new Set([...st.vioA0.keys(), ...st.vioB0.keys()])];
  if (labels.length > 0) {
    const width = Math.max(...labels.map((l) => l.length), 4);
    console.log("");
    console.log(`違反内訳 (席0のみ)${" ".repeat(2 * width - 14)}A     B    A−B`);
    labels.sort((x, y) => (st.vioA0.get(y) ?? 0) - (st.vioA0.get(x) ?? 0));
    let ta = 0, tb = 0;
    for (const l of labels) {
      const na = st.vioA0.get(l) ?? 0;
      const nb = st.vioB0.get(l) ?? 0;
      ta += na;
      tb += nb;
      const pad = " ".repeat(2 * (width - l.length));
      const d = na - nb;
      console.log(
        `  ${l}${pad}${String(na).padStart(6)}${String(nb).padStart(6)}` +
          `${((d >= 0 ? "+" : "") + d).padStart(7)}`,
      );
    }
    console.log(
      `  計${" ".repeat(2 * (width - 1))}${String(ta).padStart(6)}${String(tb).padStart(6)}` +
        `${((ta - tb >= 0 ? "+" : "") + (ta - tb)).padStart(7)}`,
    );
  }
  console.log(
    `所要 ${(st.ms / 1000).toFixed(2)}s  (${(2 * st.games / (st.ms / 1000)).toFixed(1)} 半荘/秒)`,
  );
}

function cmdBench(a: Args): void {
  const { results, ms } = headless(a.games, a.seed, a.seats, {
    weights: a.weights,
    oracle: a.oracle,
    noise: a.noise,
  });
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
    case "paired":
      cmdPaired(a);
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
