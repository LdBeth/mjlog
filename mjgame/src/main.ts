// CLI entry: `play` (interactive TUI), `selfplay` / `paired` / `bench`
// (headless). Nothing here computes anything: flags are parsed in `cli/args.ts`,
// matches are run by `harness.ts` and `paired.ts`, and what is left is the four
// commands' reports.

import { CalibrationWriter } from "./ai/calibration.ts";
import { mergeComputed } from "./ai/computed.ts";
import { FoldCalibrationWriter } from "./ai/foldcalib.ts";
import { HandCalibrationWriter } from "./ai/handcalib.ts";
import { EvCalibrationWriter } from "./ai/evcalib.ts";
import { mergeEv } from "./ai/evparams.ts";
import { mergeHand } from "./ai/handvalue.ts";
import { parseArgs } from "./cli/args.ts";
import type { Args } from "./cli/args.ts";
import { die } from "./cli/die.ts";
import { USAGE } from "./cli/usage.ts";
import { makeDojoHooks } from "./dojo.ts";
import { writeExport } from "./export.ts";
import {
  closeArm,
  headless,
  headlessParallel,
  loadKtune,
  makePolicy,
  openArm,
  playGame,
} from "./harness.ts";
import type { HeuristicPolicy } from "./ai/heuristic.ts";
import type { HeadlessOptions, RunReport, SeatPolicy } from "./harness.ts";
import { kindString } from "./spec.ts";
import type { SeatKind } from "./spec.ts";
import { pairedJson, pairedRun } from "./paired.ts";
import { runMatch } from "./match.ts";
import type { MatchResult } from "./match.ts";
import type { Policy } from "./policy.ts";
import { sfc32 } from "./rng.ts";
import { DOJO_DEFAULT, JANKI } from "./rules.ts";
// Real yaku + fu scoring.
import { finalStandings, scorer } from "./score.ts";
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
export function cpuKindAt(cpu: string, humanSeat: Seat, seat: Seat): SeatKind {
  let i = 0;
  for (let s = 0; s < seat; s++) if (s !== humanSeat) i++;
  // `argError` has already vetted the letters, so the cast records a fact.
  return (cpu[i] ?? "h") as SeatKind;
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
  // The 助言 seat: a 計算 CPU consulted on the human's own Observation, under
  // the CURRENT CHAMPION (weights/champion.json — by convention always the
  // best validated vector; promotion updates the file, never this path) unless
  // `--ktune` names another. Forked seed so it never shares a stream with a
  // seat that actually plays.
  const advisor = makePolicy({
    kind: "k",
    name: "助言",
    seed: a.seed * 4 + 4,
    plan: a.plan,
    ktune: a.ktune ?? loadKtune(new URL("../weights/champion.json", import.meta.url).pathname),
  });
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
    advisor: advisor.policy as HeuristicPolicy,
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
    advisor.close();
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

/**
 * M13's fold recorder for a run, or nothing.
 *
 * `head` is the one thing the header cannot read off a flag: it says WHICH rule
 * produced the `verdict` column, so a reader knows whether `x[0] < 0` must
 * reproduce it (the incumbent gate) or not (a fitted head). It is exactly the
 * presence of a `fold` block in the `--ktune` file — the same condition
 * `makePolicy` builds the head on.
 */
function makeFoldCalibWriter(a: Args): FoldCalibrationWriter | undefined {
  if (!a.foldcalib) return undefined;
  return new FoldCalibrationWriter(a.foldcalib, {
    seats: a.seats,
    seed: a.seed,
    games: a.games,
    eps: a.foldEps,
    head: a.ktune?.fold ? "mlp" : "gate",
  });
}

/**
 * M15b's EV核 recorder for a run, or nothing.
 *
 * The header states the parameter vector the stored predictions were made
 * under, and it is `DEFAULT_EV` by construction: `argError` refuses the flag
 * beside an `ev` block, so the recording seat carries none and the writer's own
 * core is the only one in the process. `mergeEv({})` spells that out rather
 * than leaving it implicit — a file has to be self-describing.
 */
function makeEvCalibWriter(a: Args): EvCalibrationWriter | undefined {
  if (!a.evcalib) return undefined;
  return new EvCalibrationWriter(a.evcalib, {
    seats: a.seats,
    seed: a.seed,
    games: a.games,
    ev: mergeEv({}),
  });
}

/** What an EV-recording run wrote, dropped samples and truncations included. */
function evCalibReport(a: Args, cal?: EvCalibrationWriter): void {
  if (!cal) return;
  const st = cal.stats();
  console.log(
    `EV核 ${a.evcalib}: 自摸番 ${st.rows}行  (半荘 ${st.games})` +
      (st.truncated > 0 ? `  節点上限に当たった評価 ${st.truncated}件` : "") +
      (st.dropped > 0 ? `  未決着のゆえ破棄 ${st.dropped}行` : ""),
  );
}

/** What a fold-recording run wrote — rows, flips, and the D7 caveat's size. */
function foldCalibReport(a: Args, cal?: FoldCalibrationWriter): void {
  if (!cal) return;
  const st = cal.stats();
  const pct = st.rows > 0 ? (100 * st.flips / st.rows).toFixed(1) : "0.0";
  console.log(
    `押し引き ${a.foldcalib}: 判断 ${st.rows}行  (半荘 ${st.games}, 局 ${st.rounds})` +
      `  反転 ${st.flips}回 (${pct}%)  複数反転局 ${st.multiFlipRounds}` +
      (st.dropped > 0 ? `  未決着のゆえ破棄 ${st.dropped}行` : ""),
  );
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
  const foldCalib = makeFoldCalibWriter(a);
  const evCalib = makeEvCalibWriter(a);
  try {
    const opts: HeadlessOptions = {
      calibrate,
      handCalib,
      foldCalib,
      evCalib,
      foldEps: a.foldEps,
      weights: a.weights,
      temp: a.temp,
      record: a.record || undefined,
      recordAll: a.recordAll,
      oracle: a.oracle,
      noise: a.noise,
      curriculum: a.curriculum,
      plan: a.plan,
      ktune: a.ktune,
      ktuneOpp: a.ktuneOpp,
      standings: a.standings,
      consumer: a.consumer,
    };
    // `--jobs=1` is the sequential loop itself, not a one-worker special case:
    // nothing below is reachable without the flag. Sharding a run shorter than
    // the job count would only spawn idle workers, so N is clamped there too —
    // `headlessParallel` does the clamping, this decides whether to shard at all.
    // `--table` hands the driver the four explicit specs; the legacy flags go
    // through `resolveTable` inside. `a.seats` already prints the right kinds
    // either way (parseArgs derives it from the table).
    const run = a.jobs > 1 && a.games > 1
      ? await headlessParallel(a.games, a.seed, a.table ?? a.seats, a.jobs, opts)
      : headless(a.games, a.seed, a.table ?? a.seats, opts);
    reportSelfplay(a, run);
  } finally {
    calibrate?.close();
    handCalib?.close();
    foldCalib?.close();
    evCalib?.close();
  }
  calibrationReport(a, calibrate);
  handCalibReport(a, handCalib);
  foldCalibReport(a, foldCalib);
  evCalibReport(a, evCalib);
}

/**
 * The two tables and the three lines around them. Everything it reads comes out
 * of `results` in GAME ORDER, so a sharded run prints exactly what the
 * sequential one does. No machine consumer in-repo parses these rows; the
 * format is kept stable across runs so two runs can be eyeballed side by side.
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
  // 道場順位の合計。違反持ちは全員クリーンな席の下、という道場の順位付け
  // (`score.ts::finalStandings`) をそのまま席ごとに積む。
  const dojoSum = [0, 0, 0, 0];
  let rounds = 0;
  let draws = 0;
  for (const r of results) {
    rounds += r.rounds.length;
    // 起家は第1局から読む (`paired.ts::dojoRankOfSeat0` と同じ約束)。
    const east = (r.rounds[0]?.dealer ?? 0) as Seat;
    for (const st of finalStandings(r.scores, east, r.ledger, JANKI)) {
      dojoSum[st.seat] += st.place;
    }
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
    `対局数 ${a.games}  (seed ${a.seed}..${a.seed + a.games - 1})  席 ${a.seats}${arm}` +
      (a.tablePath ? `  卓 ${a.tablePath}` : ""),
  );
  console.log(`局数 ${rounds}  平均 ${(rounds / a.games).toFixed(2)} 局/半荘  流局 ${draws}`);
  console.log(`違反 ${seats.reduce((x, t) => x + t.vio, 0)}件`);
  console.log("");
  console.log(
    "席     1位   2位   3位   4位   平均順位   道場順位   平均点    和了率  放銃率  聴牌率  違反",
  );
  for (const s of SEATS) {
    const t = seats[s];
    const avgRank = t.place.reduce((acc, n, i) => acc + n * (i + 1), 0) / a.games;
    const avgDojo = dojoSum[s] / a.games;
    const avgPts = t.total / a.games;
    const cells = t.place.map((n) => String(n).padStart(5)).join(" ");
    console.log(
      `${a.seats[s]}P${s} ${cells}   ${avgRank.toFixed(3).padStart(8)}   ` +
        `${avgDojo.toFixed(3).padStart(8)}   ` +
        `${avgPts.toFixed(0).padStart(7)}   ${pct(t.wins, rounds).padStart(6)}  ` +
        `${pct(t.deals, rounds).padStart(6)}  ${pct(t.tenpai, draws).padStart(6)}  ` +
        `${String(t.vio).padStart(4)}`,
    );
  }
  // Second table, appended rather than folded into the first: no machine
  // consumer in-repo reads either table, but the first one stays narrow enough
  // to scan, so the slower-moving averages live down here.
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
  // 平均順位 is placement by raw score; 道場順位 is `score.ts::finalStandings`'
  // own ranking — every violator below every clean seat, 起家 tie-break — and is
  // the primary number to read, the same settlement paired's 順位差(道場) reports.
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
  const foldCalib = makeFoldCalibWriter(a);
  const evCalib = makeEvCalibWriter(a);
  try {
    cmdPairedInner(a, calibrate, handCalib, foldCalib, evCalib);
  } finally {
    calibrate?.close();
    handCalib?.close();
    foldCalib?.close();
    evCalib?.close();
  }
  calibrationReport(a, calibrate);
  handCalibReport(a, handCalib);
  foldCalibReport(a, foldCalib);
  evCalibReport(a, evCalib);
}

function cmdPairedInner(
  a: Args,
  calibrate?: CalibrationWriter,
  handCalib?: HandCalibrationWriter,
  foldCalib?: FoldCalibrationWriter,
  evCalib?: EvCalibrationWriter,
): void {
  if (a.games < 1) die("--games は1以上");
  const st = pairedRun(a.games, a.seed, a.table ?? a.seats, {
    calibrate,
    handCalib,
    foldCalib,
    evCalib,
    foldEps: a.foldEps,
    weights: a.weights,
    temp: a.temp,
    oracle: a.oracle,
    noise: a.noise,
    curriculum: a.curriculum,
    plan: a.plan,
    ktune: a.ktune,
    ktuneB: a.ktuneB,
    ktuneOpp: a.ktuneOpp,
    tableB: a.tableB,
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

  const bSeats = a.tableB
    ? kindString(a.tableB)
    : a.consumerBPath || a.ktuneBPath
    ? st.seats
    : "hhhh";
  console.log(
    `対局数 ${st.games}  (seed ${st.seed}..${st.seed + st.games - 1})  ` +
      `A席 ${st.seats} / B席 ${bSeats}  オラクル ${oracleLabel(a)}` +
      (a.seats.includes("k") ? `  計算${a.plan ? " 立案あり" : ""}` : "") +
      (a.curriculum !== undefined ? `  カリキュラム E=${a.curriculum}` : "") +
      (a.standings ? "  順位効用" : "") +
      (a.ktunePath ? `  感性${a.ktuneBPath ? "A" : ""} ${a.ktunePath}` : "") +
      (a.ktuneBPath ? `  感性B ${a.ktuneBPath}` : "") +
      (a.consumerPath ? `  消費${a.consumerBPath ? "A" : ""} ${a.consumerPath}` : "") +
      (a.consumerBPath ? `  消費B ${a.consumerBPath}` : "") +
      (a.tablePath ? `  卓 ${a.tablePath}` : "") +
      (a.tableBPath ? `  卓B ${a.tableBPath}` : ""),
  );
  console.log("");
  console.log("腕     席0平均順位   道場順位   席0平均点   違反(全席)");
  console.log(
    `A ${st.seats}${st.rankA.toFixed(3).padStart(10)}${st.rankDojoA.toFixed(3).padStart(11)}` +
      `${st.scoreA.toFixed(0).padStart(12)}${String(st.vioA).padStart(13)}`,
  );
  // With --consumer-b / --ktune-b / --table-b the control arm is not hhhh;
  // `bSeats` (computed with the header above) names what it actually is.
  console.log(
    `B ${bSeats.padEnd(4)}${st.rankB.toFixed(3).padStart(10)}${
      st.rankDojoB.toFixed(3).padStart(11)
    }` +
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
  // `openArm`/`playGame` rather than `headless`, for one reason: the throughput
  // this command exists to report is per DECISION, and only the seat knows how
  // many it was asked for. The counters live on the policy (see
  // `HeuristicPolicy.decisions` / `evStats`), never in `MatchResult` — a
  // `RunReport` that carried them would put wall-clock-shaped numbers inside
  // the object every byte-identity claim in the suite is stated against.
  const arm = openArm(a.table ?? a.seats, {
    weights: a.weights,
    oracle: a.oracle,
    noise: a.noise,
    curriculum: a.curriculum,
    plan: a.plan,
    ktune: a.ktune,
    ktuneOpp: a.ktuneOpp,
    standings: a.standings,
    consumer: a.consumer,
  });
  let rounds = 0;
  let secs: number;
  // Read BEFORE `closeArm`: closing frees the seat's native EV context.
  let seat0: Pick<HeuristicPolicy, "decisions" | "evStats"> | null = null;
  try {
    const t0 = performance.now();
    for (let g = 0; g < a.games; g++) rounds += playGame(arm, a.seed + g).rounds.length;
    secs = (performance.now() - t0) / 1000;
    const p = arm.built[0].policy as Partial<HeuristicPolicy>;
    if (typeof p.decisions === "number" && p.evStats) {
      seat0 = { decisions: p.decisions, evStats: p.evStats };
    }
  } finally {
    closeArm(arm);
  }
  console.log(`${a.games} 半荘 / ${rounds} 局 を ${secs.toFixed(3)}s`);
  console.log(`${(a.games / secs).toFixed(1)} 半荘/秒   ${(rounds / secs).toFixed(0)} 局/秒`);
  if (seat0) {
    console.log(`${(seat0.decisions / secs).toFixed(0)} 決定/秒 (席0, ${seat0.decisions} 決定)`);
    const ev = seat0.evStats;
    if (ev.calls > 0) {
      console.log(
        `EV核: ${ev.calls} 回  平均 ${(ev.nodes / ev.calls).toFixed(0)} nodes  ` +
          `最大 ${ev.maxNodes}  打ち切り ${ev.truncated} 回`,
      );
    }
  }
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
