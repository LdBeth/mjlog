// CLI entry: `play` (interactive TUI), `selfplay` (headless stats), `bench`.

import { RandomPolicy } from "./ai/random.ts";
import { HeuristicPolicy } from "./ai/heuristic.ts";
import type { HeuristicWeights } from "./ai/heuristic.ts";
import {
  AugmentedHeuristic,
  calibrationReads,
  curriculumReads,
  noisyReads,
  oracleReads,
  parseChannels,
} from "./ai/augmented.ts";
import type { AugmentedWeights, OracleChannel } from "./ai/augmented.ts";
import { CalibrationWriter } from "./ai/calibration.ts";
import type { CalibRecord } from "./ai/calibration.ts";
import { computedReads, mergeComputed } from "./ai/computed.ts";
import type { ComputedTraceRef, ComputedWeights } from "./ai/computed.ts";
import { parseConsumerParams } from "./ai/consumer.ts";
import type { ConsumerParams } from "./ai/consumer.ts";
import { DEFAULT_STANDINGS_WEIGHTS } from "./ai/standings.ts";
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
   * CPU kinds, "h" heuristic, "r" random, "n" neural, "o" oracle-augmented or
   * "k" 計算 (the combinatorial reader).
   * selfplay/bench read one char per absolute seat (seat 0 first). play deals
   * the first three chars to the CPU seats in seat order, skipping wherever the
   * human landed — so "nhhh" always yields exactly one neural CPU regardless of
   * the human's seat. "o" is headless-only: it reads the live Table. "k" is not
   * — it reads nothing the player at the table cannot count for themselves.
   */
  seats: string;
  /** Engage the C7 planner on the "k" seats (`--plan`). */
  plan: boolean;
  /**
   * Engage 順位効用 on seat 0 of the test arm (`--standings`). Seat 0 only, and
   * in `paired` the A arm only: the layer is measured the way every other
   * candidate is, against a baseline that never received it.
   */
  standings: boolean;
  /** Hidden-information channels an "o" seat is allowed to see. */
  oracle: Set<OracleChannel>;
  /** `--oracle` as typed, for the report line. */
  oracleSpec: string;
  /** Per-decision probability that an "o" seat loses each information group. */
  noise: number;
  /**
   * `--curriculum=E`: seat 0 of the test arm reads the ORACLE, with each
   * information group dropped per decision at rate E and answered by the 計算
   * reader instead of by nothing.
   * `undefined` = off, and off is the only state in which nothing changes.
   * E=1 is bit-identical to the plain "k" seat; E=0 to a pure oracle seat.
   */
  curriculum?: number;
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
  /** `--ktune` as typed (report line), and the vector it loaded. */
  ktunePath: string;
  ktune?: KTune;
  /**
   * `--ktune-b`: the CONTROL arm's 感性 vector (M10d). The scalar-space mirror
   * of `--consumer-b` — with it the B arm stops being plain hhhh and becomes the
   * same seat layout as A, carrying a DIFFERENT `--ktune` file. `paired` only.
   */
  ktuneBPath: string;
  ktuneB?: KTune;
  /**
   * `--consumer` as typed (report line), and the curve set it loaded. M9: the
   * learned consumer of the evidence vector, on seat 0 of the test arm only —
   * the same discipline `--standings` follows, and for the same reason.
   */
  consumerPath: string;
  consumer?: ConsumerParams;
  /**
   * `--consumer-b`: the CONTROL arm's curve set, which turns `paired` from an
   * absolute measurement into a candidate-vs-incumbent one. `paired` only.
   */
  consumerBPath: string;
  consumerB?: ConsumerParams;
  /**
   * `--calibrate=PATH`: seat 0 (a "k" seat) writes one calibration record per
   * decision to PATH — the 計算 model's predictions paired with the engine's
   * truth. The seat plays EXACTLY as it would without the flag; see
   * `calibrationReads`. `paired` records the A arm only.
   */
  calibrate: string;
  /** `paired --json`: one line of machine-readable stats instead of tables. */
  json: boolean;
  help: boolean;
}

/** Where `--weights` points by default: what the trainer writes. */
const DEFAULT_WEIGHTS = "weights/manifest.json";

/** The ablation's standing default: the three defensive channels. */
const DEFAULT_ORACLE = "C1,C2,C3";

/**
 * What `--calibrate` asks the oracle for, fixed and independent of `--oracle`:
 * the ron mask (C1), tenpai (C2) and the payment (C3) — the three truths the
 * 計算 model makes a claim about. Tying it to the user's channel set would let a
 * `--oracle=none` run write records with no truth in them.
 */
const CALIBRATION_CHANNELS = parseChannels("C1,C2,C3")!;

/**
 * A tuned 感性 vector: the three weight objects a "k" seat is built from, each
 * section a PARTIAL merged over its own defaults by the constructor that
 * receives it.
 *
 * Deliberately un-validated. `scripts/tune.ts` writes these files and
 * `ComputedWeights` grows fields as the reader learns to count more; a key
 * whitelist here would have to be edited in lockstep with that, and would
 * reject a forward-compatible file by silently dropping the very term under
 * test. Unknown keys are simply spread onto the defaults and ignored by
 * whatever does not read them.
 */
export interface KTune {
  heuristic?: Partial<HeuristicWeights>;
  augment?: Partial<AugmentedWeights>;
  computed?: Partial<ComputedWeights>;
}

/**
 * Read a `--ktune` file. Unreadable or malformed is fatal, never silent.
 * `flag` names the option in the diagnostics, so `--ktune-b` reports itself.
 */
export function loadKtune(path: string, flag = "--ktune"): KTune {
  let text: string;
  try {
    text = Deno.readTextFileSync(path);
  } catch (e) {
    return die(`${flag} のファイルが読めません: ${path}\n${e instanceof Error ? e.message : e}`);
  }
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (e) {
    return die(`${flag} のJSONが壊れています: ${path}\n${e instanceof Error ? e.message : e}`);
  }
  if (typeof json !== "object" || json === null || Array.isArray(json)) {
    die(`${flag} はオブジェクト {heuristic, augment, computed} である必要があります: ${path}`);
  }
  // Sections only — the contents pass through verbatim.
  const k = json as KTune;
  return { heuristic: k.heuristic, augment: k.augment, computed: k.computed };
}

/**
 * Read a `--consumer` file. Unreadable, malformed or incomplete is fatal: a
 * silently-defaulted consumer would measure the hand-written score and call it
 * the fitted one.
 */
export function loadConsumer(path: string): ConsumerParams {
  let text: string;
  try {
    text = Deno.readTextFileSync(path);
  } catch (e) {
    return die(`--consumer のファイルが読めません: ${path}\n${e instanceof Error ? e.message : e}`);
  }
  try {
    return parseConsumerParams(JSON.parse(text));
  } catch (e) {
    return die(`--consumer のJSONが不正です: ${path}\n${e instanceof Error ? e.message : e}`);
  }
}

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
  plan = false,
  ktune?: KTune,
  standings = false,
  consumer?: ConsumerParams,
  curriculum?: number,
  calibrate?: (rec: CalibRecord) => void,
): SyncPolicy {
  // 順位効用 (`--standings`). Only the heuristic family has a push/fold gate to
  // scale, so only "h" and "k" can carry it; the caller has already decided WHICH
  // seat is allowed to (see `headless`), because a layer applied to every seat at
  // once would move both sides of a paired measurement.
  const rank = standings ? { standings: DEFAULT_STANDINGS_WEIGHTS } : undefined;
  if (kind === "r") return new RandomPolicy(name, seed);
  if (kind === "k") {
    // 計算. The same consumption terms as an "o" seat, fed by exact counting
    // over the Observation instead of by the engine's hidden state — so unlike
    // "o" this seat needs no tap on the Table and is legal in `play` too.
    // `--oracle` / `--noise` configure `oracleReads` and do not reach it.
    //
    // `--ktune` reaches THIS seat and only this seat: the outcome tuner grades a
    // candidate 感性 vector against an untouched baseline, so anything the
    // vector could also move on the other side of the comparison would make the
    // measurement circular. A file that names `planner` outranks `--plan`,
    // because a vector is a complete description of a seat.
    //
    // M10a: with `calibrate` on, the provider also fills a trace out-param on
    // every call. The trace is written to a file and never read by the policy,
    // so the seat below is the same seat either way — that is what the paired
    // self-diff test in `test/calibration_test.ts` pins.
    const traceRef: ComputedTraceRef = { t: null };
    const computed = computedReads(
      { planner: plan, ...ktune?.computed },
      calibrate ? traceRef : undefined,
    );
    // M9c curriculum. Still the 計算 seat — the reader is what changes: each
    // information group is answered by the oracle with probability 1−E and by
    // the counting reader with probability E, per decision. E=1 returns
    // `computed` itself, so the seat this trains for and the seat it is finally
    // graded as are the same object. Headless only, like every oracle path: no
    // `oracle` wiring (i.e. `play`) means the curriculum is silently impossible,
    // and `parseArgs` refuses the flag there rather than letting it be ignored.
    if (calibrate && !oracle) {
      die("--calibrate は selfplay / paired 専用です (真値を読む Table の tap が要ります)");
    }
    const reads = calibrate
      ? calibrationReads(
        computed,
        traceRef,
        // The recorder's own channel set, deliberately NOT `--oracle`: the three
        // truths the model makes claims about (tenpai, the ron mask, the
        // payment), whatever an "o" seat elsewhere in the run was allowed.
        oracleReads(oracle!.get, scorer, CALIBRATION_CHANNELS),
        calibrate,
      )
      : curriculum !== undefined && oracle
      ? curriculumReads(
        oracleReads(oracle.get, scorer, oracle.channels),
        computed,
        curriculum,
      )
      : computed;
    return new AugmentedHeuristic(
      name,
      seed,
      reads,
      {
        // 順位効用 is merged AFTER the tuned vector so the two compose: a
        // `--ktune` file tunes the terms 順位効用 then scales.
        weights: rank ? { ...ktune?.heuristic, ...rank } : ktune?.heuristic,
        augment: ktune?.augment,
        // M9 composes AFTER `--ktune`, and orthogonally to it: the curves
        // replace the CONSUMPTION of the evidence, while the tuned vector still
        // decides what the evidence says (`riskOf`'s ladder, the fold scales,
        // the 計算 reader's own constants).
        consumer,
      },
    );
  }
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
  return new HeuristicPolicy(name, seed, { weights: rank, consumer });
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
    plan: false,
    standings: false,
    oracle: parseChannels(DEFAULT_ORACLE)!,
    oracleSpec: DEFAULT_ORACLE,
    noise: 0,
    weights: DEFAULT_WEIGHTS,
    temp: 0,
    record: "",
    recordAll: false,
    ktunePath: "",
    ktuneBPath: "",
    consumerPath: "",
    consumerBPath: "",
    calibrate: "",
    json: false,
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
    } else if (arg.startsWith("--curriculum=")) {
      const v = Number(arg.slice(13));
      if (!Number.isFinite(v) || v < 0 || v > 1) {
        die(`--curriculum は 0..1 の実数: ${arg.slice(13)}`);
      }
      a.curriculum = v;
    } else if (arg.startsWith("--ktune-b=")) {
      a.ktuneBPath = arg.slice(10);
      a.ktuneB = loadKtune(a.ktuneBPath, "--ktune-b");
    } else if (arg.startsWith("--ktune=")) {
      a.ktunePath = arg.slice(8);
      a.ktune = loadKtune(a.ktunePath);
    } else if (arg.startsWith("--consumer=")) {
      a.consumerPath = arg.slice(11);
      a.consumer = loadConsumer(a.consumerPath);
    } else if (arg.startsWith("--consumer-b=")) {
      a.consumerBPath = arg.slice(13);
      a.consumerB = loadConsumer(a.consumerBPath);
    } else if (arg.startsWith("--calibrate=")) {
      a.calibrate = arg.slice(12);
      if (!a.calibrate) die("--calibrate には書き出し先のパスが要ります");
    } else if (arg === "--json") a.json = true;
    else if (arg === "--plan") a.plan = true;
    else if (arg === "--standings") a.standings = true;
    else if (arg.startsWith("--seats=")) {
      const v = arg.slice(8);
      if (!/^[hrnok]{1,4}$/.test(v)) die(`--seats は h, r, n, o, k を4文字まで: ${v}`);
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
  const err = argError(a);
  if (err) die(err);
  return a;
}

/** The subset of `Args` the cross-flag rules below actually read. */
export type ArgCheck =
  & Pick<Args, "cmd" | "seats" | "calibrate">
  & Partial<Pick<Args, "curriculum" | "consumerBPath" | "ktuneBPath">>;

/**
 * Cross-flag legality, as a MESSAGE rather than an exit. Every rule here refuses
 * a layout rather than ignoring it — a flag that quietly does nothing is worse
 * than one that is missing — and returning the message instead of calling
 * `die` is what lets a test read the rules without spawning a process.
 */
export function argError(a: ArgCheck): string | null {
  // The curriculum reads the live Table (that is what makes it a curriculum),
  // so it belongs to the headless drivers alone.
  if (a.curriculum !== undefined && a.cmd === "play") {
    return "--curriculum は selfplay / bench / paired 専用です (play では隠蔽情報を読めません)";
  }
  // --consumer-b / --ktune-b name the CONTROL arm, and only `paired` has one.
  if (a.consumerBPath && a.cmd !== "paired") return "--consumer-b は paired 専用です";
  if (a.ktuneBPath && a.cmd !== "paired") return "--ktune-b は paired 専用です";
  // The recorder needs BOTH readers: the 計算 seat whose predictions are being
  // graded (seat 0, and only a "k" seat computes them) and the oracle's tap on
  // the live Table (headless drivers only). Every one of these is refused rather
  // than ignored — a calibration run that quietly recorded nothing would be
  // discovered a thousand hanchan later.
  if (a.calibrate) {
    if (a.cmd !== "selfplay" && a.cmd !== "paired") {
      return "--calibrate は selfplay / paired 専用です (play では隠蔽情報の真値を読めません)";
    }
    if (a.seats[0] !== "k") {
      return `--calibrate は席0が k席 (計算) のときだけ使えます: --seats=${a.seats}\n` +
        "計算の読みを真値と突き合わせる機能なので、記録する読み手が要ります。";
    }
    if (a.curriculum !== undefined) {
      // The curriculum REPLACES the computed answer with truth part of the time;
      // calibration measures the computed answer against truth. Recording under
      // a curriculum would grade a seat nobody ships.
      return "--calibrate と --curriculum は併用できません (較正するのは素の計算の読みです)";
    }
  }
  return null;
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
                      n=学習済みニューラルポリシー, o=オラクル増補,
                      k=計算 (公開情報だけの組合せ読み) (既定 hhhh)。
                      短く書くと最後の文字を繰り返す ("hr" ⇒ "hrrr")。
                      selfplay/bench では席番号ごと。play では人間の席を
                      飛ばして先頭3文字を席順に割り当てるので、"nhhh" なら
                      必ずAI(学習済み)が1人入る。n のCPUは AI東 のように表示。
                      o は隠蔽情報 (他家の手牌・山) を直接読むので headless 専用。
                      k は隠蔽情報を一切見ない (スジ・カベ・現物・見えている枚数・
                      副露・リーチ・巡目だけを数える) ので play でも使える
  --plan              k席で最大利益ロックオン立案 (C7) を有効にする (既定 無効)。
                      o席の立案は --oracle=C7O/C7P 側で指定する
  --standings         A腕席0で順位効用レイヤを有効にする (順位分布モデルで押し引きを
                      尺度化)。持ち点・局数・供託・本場という公開情報だけから
                      最終着順分布を閉形式で解き、和了の値打ちと放銃の代償を
                      平場 (全員25000) 比の2つの倍率にして押し引きに掛ける。
                      雀鬼流の補正として「順位−1の仮想プレイヤー」を常に自分の
                      8000点上に置くので、独走トップでも打つのを止めない。
                      h席・k席に効き、対照の B腕 (hhhh) には決して渡らない
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
  --curriculum=E      A腕席0 (k席) の読みを「オラクル→計算」のカリキュラムにする。
                      1判断ごとに情報グループを確率Eで落とし、落ちた分は
                      「無い」ではなく計算 (公開情報だけの読み) の答えで埋める。
                      E=0 は純オラクル席と、E=1 は素の k席とビット単位で同一。
                      --oracle= で読ませるチャネルを選ぶ。selfplay/bench/paired 専用。
                      学習用: 消費曲線を鍛えるとき、読みの精度だけを連続に劣化させる
  --ktune=PATH        k席の感性ベクトル {heuristic, augment, computed} のJSON。
                      selfplay / bench / paired のみ。paired では A腕の k席
                      だけに効き、対照の B腕 (hhhh) には決して渡らない。
                      scripts/tune.ts が書き出す形式
  --ktune-b=PATH      paired の対照 (B腕) にも感性ベクトルを積む。B腕は hhhh ではなく
                      A腕と同じ席種・同じ読み・同じ曲線のまま、--ktune の file だけ
                      こちらになる。つまり測るのは「候補 − 現行」であって
                      「候補 − 素の hhhh」ではない。--consumer-b のスカラ版で、
                      小さな摂動は大半の局をビット単位で不変に保つので、同じシード数で
                      桁違いに細かい差が読める (探索器の分散削減)。--consumer-b とも
                      併用でき、その場合 B腕は自前の感性ベクトルと自前の曲線を持つ。
                      paired 専用
  --consumer=PATH     M9: 打牌評価の「消費」を単調曲線に差し替える (証拠ベクトルの
                      名前つき素性 → 4節点の区分線形写像17本)。selfplay / bench /
                      paired の席0だけに効き、対照の B腕 (hhhh) には決して渡らない。
                      計算 (証拠の作り方) は一切変えない — 変わるのは消費だけ。
                      scripts/consumer_init.ts が書き出す初期値は現行の手書き評価と
                      ビット単位で同一なので、--consumer=weights/consumer-init.json
                      を渡した paired は必ず全局同着になる (これが健全性検査)。
                      --ktune と併用可 (危険度の梯子や降り倍率は感性側が決める)
  --consumer-b=PATH   paired の対照 (B腕) にも曲線を積む。B腕は hhhh ではなく
                      A腕と同じ席種・同じ --ktune・同じ読みのまま、曲線だけ
                      この file になる。つまり測るのは「候補 − 現行」であって
                      「候補 − 素の hhhh」ではない。小さな摂動は大半の局を
                      ビット単位で不変に保つので、同じシード数で桁違いに
                      細かい差が読める (探索器の分散削減)。paired 専用
  --calibrate=PATH    M10a: 席0 (k席) の1判断ごとに「計算の予測」と「真値」を
                      対にした較正記録を JSONL で書き出す。selfplay / paired 専用で、
                      paired では A腕だけ。打牌は一切変わらない (記録は out-param で、
                      席が読む Reads は素の計算のまま — だから記録あり/なしで
                      全局ビット単位で同一になる)。
                      1行1判断: 他家3人ぶんの聴牌確率・待ちの形の素の枚数 (パラメータ
                      非依存の整数) ・副露の内容読み・打点の材料と、真値の聴牌/ロン牌
                      集合/打点。パラメータを変えた再評価は再対局なしで閉じた式で
                      できる。読むのは scripts/calibrate_report.ts。
                      1半荘あたり約220KB (判断190行) — 出力先は作業用ディレクトリに
  --json              paired の結果を1行のJSONで出す (表の代わり)。
                      scripts/tune.ts が読む機械可読出力
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
      makePolicy(
        cpuKindAt(a.seats, humanSeat, s),
        names[s],
        a.seed * 4 + s,
        a.weights,
        undefined,
        undefined,
        a.plan,
      ),
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
  /**
   * M9c: SEAT 0's reader becomes an oracle→計算 curriculum at this dropout rate.
   * Seat 0 and seat 0 alone, for the same reason `standings` and `consumer` are:
   * a reader handed to every seat would move both sides of a paired comparison.
   * `undefined` leaves every seat exactly as it was.
   */
  curriculum?: number;
  /** Engage the C7 planner on any "k" seat. */
  plan?: boolean;
  /**
   * Engage 順位効用 on SEAT 0, and on seat 0 alone. Deliberately narrower than
   * every other option here: the layer applies to any heuristic-family seat, so
   * a run-wide switch would move all four of them at once and there would be
   * nothing left to measure the one against.
   */
  standings?: boolean;
  /**
   * Tuned 感性 vector for any "k" seat. Every other seat kind ignores it — see
   * `makePolicy`, and `pairedRun`, which deliberately withholds it from the
   * control arm.
   */
  ktune?: KTune;
  /**
   * `pairedRun` ONLY (`headless` ignores it): the CONTROL arm's 感性 vector.
   *
   * The scalar-space half of the variance-reduction instrument (M10d), and the
   * exact mirror of `consumerB` below — same seats, same policy dice, same
   * curves, same oracle/curriculum wiring, only the `--ktune` file differs. What
   * `paired` then measures is candidate MINUS INCUMBENT, which is the comparison
   * a search actually asks about; against plain `hhhh` a twelve-scalar nudge is
   * swamped by the ~1.4 per-seed SD of two different players' rank difference.
   * The two may compose: with both set, arm B gets its own vector AND its own
   * curves.
   */
  ktuneB?: KTune;
  /**
   * M9's learned consumer, for SEAT 0 and seat 0 alone — the same narrowing as
   * `standings`, and the same reason: a curve set applied to every seat would
   * move both sides of a paired measurement at once.
   */
  consumer?: ConsumerParams;
  /**
   * `pairedRun` ONLY (`headless` ignores it): the CONTROL arm's curve set.
   *
   * Setting it changes what `paired` measures, and it is the whole point of the
   * option. Against the default control (`hhhh`, no curves) a small perturbation
   * of a curve set is swamped: the two arms are different players, most games
   * diverge on the first turn, and the per-seed SD of the rank difference is
   * ~1.4 — so a 1000-seed run carries a ±0.08 CI, which cannot rank neighbours
   * that differ by 0.02. Against an INCUMBENT the two arms are the same player
   * up to a nudge, most games stay bit-identical to the end, and the difference
   * is measured where it actually occurs. Same seeds, same policy dice, same
   * `--ktune`, same oracle/curriculum wiring — only the curve file differs.
   */
  consumerB?: ConsumerParams;
  /**
   * M10a: SEAT 0's calibration recorder, one file for the whole run.
   *
   * A WRITER rather than a path, because `pairedRun` calls `headless` once per
   * game and a path would truncate the file at every seed. It is also why the
   * control arm never receives it — `pairedRun` strips it explicitly, the same
   * discipline `record` follows.
   */
  calibrate?: CalibrationWriter;
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
  // The curriculum's oracle half needs the same tap, on a "k" seat that would
  // otherwise never ask for it.
  const curriculumOn = opts.curriculum !== undefined && seats[0] === "k";
  // …and so does the calibration recorder, for the truth half of its records.
  const calibrateOn = opts.calibrate !== undefined && seats[0] === "k";
  const wiring: OracleWiring = {
    get: () => ref.t,
    channels: opts.oracle ?? new Set(),
    noise: opts.noise ?? 0,
  };
  const t0 = performance.now();
  try {
    for (let g = 0; g < games; g++) {
      const s = seed + g;
      if (calibrateOn) opts.calibrate!.beginGame(s);
      const policies: SyncPolicy[] = SEATS.map((seat) => {
        const p = makePolicy(
          seats[seat],
          `${seats[seat].toUpperCase()}${seat}`,
          s * 4 + seat,
          opts.weights,
          opts.temp,
          wiring,
          opts.plan,
          opts.ktune,
          (opts.standings ?? false) && seat === 0,
          seat === 0 ? opts.consumer : undefined,
          seat === 0 ? opts.curriculum : undefined,
          calibrateOn && seat === 0 ? opts.calibrate!.record : undefined,
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
        tableRef: writer || oracleSeats || curriculumOn || calibrateOn ? ref : undefined,
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

function cmdSelfplay(a: Args): void {
  const calibrate = makeCalibrationWriter(a);
  try {
    cmdSelfplayInner(a, calibrate);
  } finally {
    calibrate?.close();
  }
  calibrationReport(a, calibrate);
}

function cmdSelfplayInner(a: Args, calibrate?: CalibrationWriter): void {
  const { results, ms, traj } = headless(a.games, a.seed, a.seats, {
    calibrate,
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
  const arm = (a.seats.includes("o") ? `  オラクル ${oracleLabel(a)}` : "") +
    (a.seats.includes("k") ? `  計算${a.plan ? " 立案あり" : ""}` : "") +
    (a.standings ? "  順位効用" : "");
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
    // The control arm is handed ONE option, and it is not a tuning knob: the
    // manifest path, so an "n" seat in `seats` has a comparable baseline to
    // load. Nothing that could move seat 0's play — `oracle`, `noise`, `plan`,
    // `ktune`, `standings` — may cross this line, or the difference below would be measured
    // against a baseline the candidate had already edited. (hhhh has no "k" or
    // "o" seat to read them anyway; the point is that it stays that way.)
    //
    // …UNLESS `consumerB` (M9c-b) or `ktuneB` (M10d) names an INCUMBENT. Then
    // the control arm is the same seat, the same everything, carrying a
    // DIFFERENT curve set / a DIFFERENT 感性 vector — and the difference
    // measured is candidate minus incumbent rather than candidate minus
    // baseline. Whichever of the two is absent is inherited from arm A, so the
    // two arms still differ in exactly one file. See `consumerB` / `ktuneB`.
    const paired = opts.consumerB !== undefined || opts.ktuneB !== undefined;
    const b = paired
      ? headless(1, s, seats, {
        ...opts,
        record: undefined,
        // The control arm is not what is being calibrated, and its decisions
        // would interleave into the same file under the same seed.
        calibrate: undefined,
        consumer: opts.consumerB ?? opts.consumer,
        ktune: opts.ktuneB ?? opts.ktune,
      }).results[0]
      : headless(1, s, "hhhh", { weights: opts.weights }).results[0];
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

/**
 * `paired --json`: the same measurement as the tables below, in one line a
 * search loop can parse. Only the fields a tuner grades on — the Maps become
 * plain objects, and everything the human tables derive (勝敗 counts, the
 * per-arm score means) is left out on purpose: what is not printed cannot be
 * optimised against by accident.
 */
export function pairedJson(st: PairedStats): string {
  const obj = {
    games: st.games,
    seed: st.seed,
    seats: st.seats,
    rankA: st.rankA,
    rankB: st.rankB,
    dRank: st.dRank,
    dRankDojo: st.dRankDojo,
    dScore: st.dScore,
    vioA0: Object.fromEntries(st.vioA0),
    vioB0: Object.fromEntries(st.vioB0),
    ms: st.ms,
  };
  return JSON.stringify(obj);
}

function cmdPaired(a: Args): void {
  const calibrate = makeCalibrationWriter(a);
  try {
    cmdPairedInner(a, calibrate);
  } finally {
    calibrate?.close();
  }
  calibrationReport(a, calibrate);
}

function cmdPairedInner(a: Args, calibrate?: CalibrationWriter): void {
  if (a.games < 1) die("--games は1以上");
  const st = pairedRun(a.games, a.seed, a.seats, {
    calibrate,
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
