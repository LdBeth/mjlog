// CLI entry: `play` (interactive TUI), `selfplay` / `paired` / `bench`
// (headless). Nothing here computes anything: flags are parsed in `cli/args.ts`,
// matches are run by `harness.ts` and `paired.ts`, and what is left is the four
// commands' reports.

import { CalibrationWriter } from "./ai/calibration.ts";
import { mergeComputed } from "./ai/computed.ts";
import { HandCalibrationWriter } from "./ai/handcalib.ts";
import { mergeHand } from "./ai/handvalue.ts";
import { parseArgs } from "./cli/args.ts";
import type { Args } from "./cli/args.ts";
import { die } from "./cli/die.ts";
import { USAGE } from "./cli/usage.ts";
import { makeDojoHooks } from "./dojo.ts";
import { writeExport } from "./export.ts";
import { headless, headlessParallel, makePolicy } from "./harness.ts";
import type { HeadlessOptions, RunReport, SeatPolicy } from "./harness.ts";
import { pairedJson, pairedRun } from "./paired.ts";
import { runMatch } from "./match.ts";
import type { MatchResult } from "./match.ts";
import type { Policy } from "./policy.ts";
import { sfc32 } from "./rng.ts";
import { DOJO_DEFAULT, JANKI } from "./rules.ts";
// Real yaku + fu scoring.
import { scorer } from "./score.ts";
import { SEATS } from "./types.ts";
import type { Seat } from "./types.ts";
import { App, PacedPolicy } from "./tui/app.ts";
import * as term from "./tui/term.ts";

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

/**
 * `--export`: one Tenhou-XML/sidecar pair per match, and the line that says
 * where they went. A single match keeps the basename as typed; a run of many
 * numbers them `-0001`, `-0002`, … so the pairs sort in the order they were
 * played (the seed of match `i` is `--seed` + i, and the sidecar records it).
 */
function exportMatches(a: Args, results: MatchResult[]): void {
  if (!a.exportPath || results.length === 0) return;
  const width = Math.max(4, String(results.length).length);
  const many = results.length > 1;
  const written = results.map((r, i) =>
    writeExport(r, JANKI, a.exportPath, many ? `-${String(i + 1).padStart(width, "0")}` : "")
  );
  console.log(
    `牌譜 ${written[0].xml}${many ? ` … ${written[written.length - 1].xml}` : ""}` +
      ` (${written.length}半荘, 併せて .mjgame.json)`,
  );
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
    timerTurnMs: a.timerTurn,
    timerBankMs: a.timerBank,
    cpuDelayMs: Math.max(0, a.speed),
    cfg: JANKI,
    humanSeat,
    noIntro: a.noIntro,
  });

  const cpus: SeatPolicy[] = [];
  const policies: Policy[] = SEATS.map((s) => {
    if (s === humanSeat) return app.human;
    const cpu = makePolicy({
      kind: cpuKindAt(a.seats, humanSeat, s),
      name: names[s],
      seed: a.seed * 4 + s,
      weights: a.weights,
      plan: a.plan,
    });
    cpus.push(cpu);
    return new PacedPolicy(cpu.policy, () => app.paceDelay());
  });

  // Written after the alt-screen is gone, so the path (or the failure to write
  // it) is still on screen when the process exits.
  let result: MatchResult | undefined;
  try {
    app.start();
    await app.intro();
    result = await runMatch(policies, {
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
    for (const c of cpus) c.close();
  }
  exportMatches(a, result ? [result] : []);
}

// ---------------------------------------------------------------------------
// selfplay / bench
// ---------------------------------------------------------------------------

/**
 * The oracle arm as one string for a report header: the channel set, plus the
 * fading level whenever it is on. Printed so a saved log says which point of the
 * ε sweep produced it — the numbers below it are meaningless without that.
 */
function oracleLabel(a: Args): string {
  const ch = [...a.oracle].sort().join(",") || "none";
  return a.noise > 0 ? `${ch} ノイズ ${a.noise}` : ch;
}

/**
 * The calibration recorder for a run, or nothing. `parseArgs` has already
 * refused every layout that could not fill it, so reaching here with a path
 * means seat 0 is a "k" seat under a headless driver.
 *
 * The header states the 感性 vector the predictions were made with (the same
 * merge the seat's constructor performs), because every number in the file is a
 * function of it — a report against the wrong weights is worse than no report.
 */
function makeCalibrationWriter(a: Args): CalibrationWriter | undefined {
  if (!a.calibrate) return undefined;
  return new CalibrationWriter(a.calibrate, {
    seats: a.seats,
    seed: a.seed,
    games: a.games,
    w: mergeComputed({ planner: a.plan, ...a.ktune?.computed }),
  });
}

/** The one line a recording run prints about what it wrote. */
function calibrationReport(a: Args, cal?: CalibrationWriter): void {
  if (!cal) return;
  const st = cal.stats();
  console.log(`較正 ${a.calibrate}: 判断 ${st.rows}行  (半荘 ${st.games})`);
}

/**
 * The hand-value recorder for a run, or nothing — the mirror of the above.
 *
 * The header states the weight vector the predictions were made with, and it is
 * the SAME merge `makePolicy` performs for the seat (`DEFAULT_HAND` when the
 * `--ktune` file carries no `hand` block, which is exactly the bootstrap case:
 * record under the defaults, then fit them).
 */
function makeHandCalibWriter(a: Args): HandCalibrationWriter | undefined {
  if (!a.handcalib) return undefined;
  return new HandCalibrationWriter(a.handcalib, {
    seats: a.seats,
    seed: a.seed,
    games: a.games,
    w: mergeHand(a.ktune?.hand),
  });
}

/** What a hand-recording run wrote, dropped samples included (see the writer). */
function handCalibReport(a: Args, cal?: HandCalibrationWriter): void {
  if (!cal) return;
  const st = cal.stats();
  console.log(
    `手牌価値 ${a.handcalib}: 自摸番 ${st.rows}行  (半荘 ${st.games})` +
      (st.dropped > 0 ? `  未決着のゆえ破棄 ${st.dropped}行` : ""),
  );
}

/** Everything one seat's rows of the two tables below are counted from. */
interface SeatTally {
  /** Finishes in each place, 1位 first. */
  place: number[];
  /** Final scores, summed over the run. */
  total: number;
  wins: number;
  /** 放銃 — rounds this seat paid a ron out on. */
  deals: number;
  /** 流局時聴牌 */
  tenpai: number;
  /** Ledger entries. */
  vio: number;
  /** Rounds this seat declared riichi in. */
  riichis: number;
  /** Rounds this seat called in (暗槓 is closed, so it does not count). */
  furo: number;
  /** 和了局の実収支合計 (本場・供託込み) and the number of them. */
  winPts: number;
  winN: number;
  /** 放銃局の実支出合計 and the number of them. */
  dealPts: number;
  dealN: number;
}

async function cmdSelfplay(a: Args): Promise<void> {
  const calibrate = makeCalibrationWriter(a);
  const handCalib = makeHandCalibWriter(a);
  try {
    const opts: HeadlessOptions = {
      calibrate,
      handCalib,
      weights: a.weights,
      temp: a.temp,
      record: a.record || undefined,
      recordAll: a.recordAll,
      oracle: a.oracle,
      noise: a.noise,
      curriculum: a.curriculum,
      plan: a.plan,
      ktune: a.ktune,
      standings: a.standings,
      consumer: a.consumer,
    };
    // `--jobs=1` is the sequential loop itself, not a one-worker special case:
    // nothing below is reachable without the flag. Sharding a run shorter than
    // the job count would only spawn idle workers, so N is clamped there too —
    // `headlessParallel` does the clamping, this decides whether to shard at all.
    const run = a.jobs > 1 && a.games > 1
      ? await headlessParallel(a.games, a.seed, a.seats, a.jobs, opts)
      : headless(a.games, a.seed, a.seats, opts);
    reportSelfplay(a, run);
  } finally {
    calibrate?.close();
    handCalib?.close();
  }
  calibrationReport(a, calibrate);
  handCalibReport(a, handCalib);
}

/**
 * The two tables and the three lines around them. Everything it reads comes out
 * of `results` in GAME ORDER, so a sharded run prints exactly what the
 * sequential one does — the training scripts grep these rows.
 */
function reportSelfplay(a: Args, { results, ms, traj }: RunReport): void {
  const seats: SeatTally[] = SEATS.map(() => ({
    place: [0, 0, 0, 0],
    total: 0,
    wins: 0,
    deals: 0,
    tenpai: 0,
    vio: 0,
    riichis: 0,
    furo: 0,
    winPts: 0,
    winN: 0,
    dealPts: 0,
    dealN: 0,
  }));
  let rounds = 0;
  let draws = 0;
  for (const r of results) {
    rounds += r.rounds.length;
    for (const v of r.ledger) seats[v.seat].vio++;
    for (const s of SEATS) {
      seats[s].riichis += (r.riichis ?? [0, 0, 0, 0])[s];
      seats[s].furo += (r.furoRounds ?? [0, 0, 0, 0])[s];
    }
    for (const o of r.outcomes) {
      if (o.kind === "agari") {
        for (const w of o.wins) {
          seats[w.who].wins++;
          if (w.fromWho !== w.who) seats[w.fromWho].deals++;
        }
        for (const w of o.wins) {
          seats[w.who].winN++;
          seats[w.who].winPts += o.deltas[w.who];
        }
        // 放銃打点は「振り込んだ局」単位。ダブロンは1局1回だけ数え、
        // その局の支出 (両家ぶん) をまとめて負担額とする。
        const head = o.wins[0];
        if (head && head.fromWho !== head.who) {
          seats[head.fromWho].dealN++;
          seats[head.fromWho].dealPts += -o.deltas[head.fromWho];
        }
      } else {
        draws++;
        o.tenpai.forEach((t, s) => {
          if (t) seats[s].tenpai++;
        });
      }
    }
    const order = r.scores
      .map((s, seat) => ({ seat, s }))
      .sort((x, y) => y.s - x.s || x.seat - y.seat);
    order.forEach((o, i) => seats[o.seat].place[i]++);
    for (const s of SEATS) seats[s].total += r.scores[s];
  }
  const pct = (n: number, d: number) => d === 0 ? "  -  " : `${(100 * n / d).toFixed(1)}%`;

  // The oracle arm is named only when a seat actually reads it, so an all-h run
  // — the line the training scripts grep — stays byte-identical.
  const arm = (a.seats.includes("o") ? `  オラクル ${oracleLabel(a)}` : "") +
    (a.seats.includes("k") ? `  計算${a.plan ? " 立案あり" : ""}` : "") +
    (a.standings ? "  順位効用" : "");
  console.log(
    `対局数 ${a.games}  (seed ${a.seed}..${a.seed + a.games - 1})  席 ${a.seats}${arm}`,
  );
  console.log(`局数 ${rounds}  平均 ${(rounds / a.games).toFixed(2)} 局/半荘  流局 ${draws}`);
  console.log(`違反 ${seats.reduce((x, t) => x + t.vio, 0)}件`);
  console.log("");
  console.log("席     1位   2位   3位   4位   平均順位   平均点    和了率  放銃率  聴牌率  違反");
  for (const s of SEATS) {
    const t = seats[s];
    const avgRank = t.place.reduce((acc, n, i) => acc + n * (i + 1), 0) / a.games;
    const avgPts = t.total / a.games;
    const cells = t.place.map((n) => String(n).padStart(5)).join(" ");
    console.log(
      `${a.seats[s]}P${s} ${cells}   ${avgRank.toFixed(3).padStart(8)}   ` +
        `${avgPts.toFixed(0).padStart(7)}   ${pct(t.wins, rounds).padStart(6)}  ` +
        `${pct(t.deals, rounds).padStart(6)}  ${pct(t.tenpai, draws).padStart(6)}  ` +
        `${String(t.vio).padStart(4)}`,
    );
  }
  // Second table, appended rather than folded into the first: the rows above are
  // grepped by the training scripts and must stay byte-identical.
  const mean = (sum: number, n: number) => n === 0 ? "-" : String(Math.round(sum / n));
  console.log("");
  console.log("席     リーチ率  副露率  平均和了打点  平均放銃打点");
  for (const s of SEATS) {
    const t = seats[s];
    console.log(
      `${a.seats[s]}P${s}${pct(t.riichis, rounds).padStart(12)}` +
        `${pct(t.furo, rounds).padStart(8)}` +
        `${mean(t.winPts, t.winN).padStart(14)}` +
        `${mean(t.dealPts, t.dealN).padStart(14)}`,
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
  exportMatches(a, results);
}

// ---------------------------------------------------------------------------
// paired
// ---------------------------------------------------------------------------

function cmdPaired(a: Args): void {
  const calibrate = makeCalibrationWriter(a);
  const handCalib = makeHandCalibWriter(a);
  try {
    cmdPairedInner(a, calibrate, handCalib);
  } finally {
    calibrate?.close();
    handCalib?.close();
  }
  calibrationReport(a, calibrate);
  handCalibReport(a, handCalib);
}

function cmdPairedInner(
  a: Args,
  calibrate?: CalibrationWriter,
  handCalib?: HandCalibrationWriter,
): void {
  if (a.games < 1) die("--games は1以上");
  const st = pairedRun(a.games, a.seed, a.seats, {
    calibrate,
    handCalib,
    weights: a.weights,
    temp: a.temp,
    oracle: a.oracle,
    noise: a.noise,
    curriculum: a.curriculum,
    plan: a.plan,
    ktune: a.ktune,
    ktuneB: a.ktuneB,
    standings: a.standings,
    consumer: a.consumer,
    consumerB: a.consumerB,
  });
  if (a.json) {
    console.log(pairedJson(st));
    return;
  }
  const sign = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(3);
  const signPt = (x: number) => (x >= 0 ? "+" : "") + x.toFixed(0);

  console.log(
    `対局数 ${st.games}  (seed ${st.seed}..${st.seed + st.games - 1})  ` +
      `A席 ${st.seats} / B席 ${a.consumerBPath || a.ktuneBPath ? st.seats : "hhhh"}  オラクル ${
        oracleLabel(a)
      }` +
      (a.seats.includes("k") ? `  計算${a.plan ? " 立案あり" : ""}` : "") +
      (a.curriculum !== undefined ? `  カリキュラム E=${a.curriculum}` : "") +
      (a.standings ? "  順位効用" : "") +
      (a.ktunePath ? `  感性${a.ktuneBPath ? "A" : ""} ${a.ktunePath}` : "") +
      (a.ktuneBPath ? `  感性B ${a.ktuneBPath}` : "") +
      (a.consumerPath ? `  消費${a.consumerBPath ? "A" : ""} ${a.consumerPath}` : "") +
      (a.consumerBPath ? `  消費B ${a.consumerBPath}` : ""),
  );
  console.log("");
  console.log("腕     席0平均順位   道場順位   席0平均点   違反(全席)");
  console.log(
    `A ${st.seats}${st.rankA.toFixed(3).padStart(10)}${st.rankDojoA.toFixed(3).padStart(11)}` +
      `${st.scoreA.toFixed(0).padStart(12)}${String(st.vioA).padStart(13)}`,
  );
  // With --consumer-b / --ktune-b the control arm is the same seat kind as A,
  // not hhhh.
  console.log(
    `B ${(a.consumerBPath || a.ktuneBPath ? st.seats : "hhhh").padEnd(4)}${
      st.rankB.toFixed(3).padStart(10)
    }${st.rankDojoB.toFixed(3).padStart(11)}` +
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

// ---------------------------------------------------------------------------
// bench
// ---------------------------------------------------------------------------

function cmdBench(a: Args): void {
  const { results, ms } = headless(a.games, a.seed, a.seats, {
    weights: a.weights,
    oracle: a.oracle,
    noise: a.noise,
    curriculum: a.curriculum,
    plan: a.plan,
    ktune: a.ktune,
    standings: a.standings,
    consumer: a.consumer,
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
      await cmdSelfplay(a);
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
