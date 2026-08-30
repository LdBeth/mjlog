// Command line → `Args`, and the cross-flag legality rules that go with it.
//
// This project has zero third-party dependencies, so the parser is a `for` loop
// over `Deno.args`. Every rule here refuses a layout rather than ignoring it: a
// flag that quietly does nothing is worse than one that is missing.

import { parseChannels } from "../ai/augmented.ts";
import type { OracleChannel } from "../ai/augmented.ts";
import type { ConsumerParams } from "../ai/consumer.ts";
import { DEFAULT_WEIGHTS, loadConsumer, loadKtune } from "../harness.ts";
import type { KTune } from "../harness.ts";
import { kindString, loadTable } from "../spec.ts";
import type { TableSpec } from "../spec.ts";
import type { GlyphMode } from "../tui/glyph.ts";
import { die } from "./die.ts";

export interface Args {
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
   * CPU kinds, "h" the frozen baseline (a 2026-08-25 snapshot of the default
   * "k" — see `ai/frozen.ts`), "r" random, "n" neural, "o" oracle-augmented or
   * "k" 計算 (the combinatorial reader).
   * selfplay/bench read one char per absolute seat (seat 0 first). play deals
   * the first three chars to the CPU seats in seat order, skipping wherever the
   * human landed — so "nhhh" always yields exactly one neural CPU regardless of
   * the human's seat. "o" is headless-only: it reads the live Table. "k" is not
   * — it reads nothing the player at the table cannot count for themselves.
   */
  seats: string;
  /** Whether `--seats` was typed (as opposed to the "hhhh" default). */
  seatsGiven: boolean;
  /** Whether `--weights` / `--temp` were typed (they have defaults). */
  weightsGiven: boolean;
  tempGiven: boolean;
  /**
   * `--table`: the modular seat description — a JSON file of four explicit
   * `SeatSpec`s, each seat with its OWN complete vector, components and
   * weights (`spec.ts::loadTable`). The flat per-run flags (`--ktune`,
   * `--ktune-opp`, `--plan`, `--standings`, `--consumer`, `--curriculum`,
   * `--weights`, `--temp`, `--seats`) are the legacy spelling of the same
   * thing, so each of them CONFLICTS with `--table` rather than composing:
   * a table is a complete description, and a flag that quietly lost to it
   * would be worse than one refused. `a.seats` is derived from the table for
   * the reports and the seat-0 checks below.
   */
  tablePath: string;
  table?: TableSpec;
  /**
   * `--table-b`: `paired`'s control arm as an explicit table. Seats 1–3 must
   * equal `--table`'s (the environment is shared; `pairedRun` enforces it) —
   * the two arms may differ in seat 0 only.
   */
  tableBPath: string;
  tableB?: TableSpec;
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
   * `--ktune-opp`: the OPPONENTS' (席1-3) 感性 vector, distinct from seat 0's
   * `--ktune`. Without it every seat shares `--ktune` exactly as before; with it
   * a second, differently tuned population can be built — which is what asking
   * whether fitted parameters TRANSFER requires. The opponents are the
   * environment, so in `paired` it reaches both arms alike.
   */
  ktuneOppPath: string;
  ktuneOpp?: KTune;
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
  /**
   * `--handcalib=PATH`: seat 0 (a "k" or "h" seat) writes one hand-value record
   * per TURN decision to PATH — `handOutlook`'s two predictions for the resting
   * shape it chose, labelled after the fact with how the round ended for it.
   * The mirror image of `--calibrate`: that one grades what we read about the
   * other three, this one grades what we believe about ourselves. The seat plays
   * EXACTLY as it would without the flag (the sink is an out-param, see
   * `src/ai/handcalib.ts`). `paired` records the A arm only.
   */
  handcalib: string;
  /**
   * `--foldcalib=PATH`: seat 0 (a "k" seat) writes one record per PUSH/FOLD
   * decision to PATH — the 37 features the M13 head reads, what the seat's own
   * rule decided, what was actually played, and the 局's own settlement as the
   * reward. Unlike `--calibrate`/`--handcalib` this lane can PERTURB the seat:
   * see `--fold-eps`. With ε 0 (the default) it is an observer like the other
   * two and the run is bit-identical to one without the flag.
   * `paired` records the A arm only.
   */
  foldcalib: string;
  /**
   * `--fold-eps=X`, 0 < X < 1: flip the fold verdict with probability X, on a
   * random stream of the seat's own. This is the exploration that makes the
   * lane a CONTEXTUAL BANDIT rather than a log of one policy's opinions — with
   * no flips there is no counterfactual and nothing to fit. Refused without
   * `--foldcalib`: a run that played worse and recorded nothing would be a
   * waste nobody asked for.
   */
  foldEps: number;
  /**
   * `--export=PATH`: write the played match(es) as Tenhou mjlog XML plus a
   * `.mjgame.json` sidecar, so ../mjrender can render and comment our own games.
   * PATH is a basename unless it already ends in `.xml`; a selfplay run of more
   * than one game numbers the pairs `PATH-0001.xml`, `PATH-0002.xml`, …
   */
  exportPath: string;
  /** `paired --json`: one line of machine-readable stats instead of tables. */
  json: boolean;
  /**
   * `--jobs=N`: play a selfplay run in N Deno Workers instead of one loop.
   * 1 (the default) IS the sequential path — no worker is spawned. Higher values
   * shard the games round-robin and reassemble them in game order, so every
   * table, the trajectory file and the exported 牌譜 are byte-identical to the
   * `--jobs=1` run of the same seed. N above `--games` is clamped to it.
   */
  jobs: number;
  help: boolean;
}

/** The ablation's standing default: the three defensive channels. */
const DEFAULT_ORACLE = "C1,C2,C3";

export function parseArgs(argv: string[]): Args {
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
    seatsGiven: false,
    weightsGiven: false,
    tempGiven: false,
    tablePath: "",
    tableBPath: "",
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
    ktuneOppPath: "",
    consumerPath: "",
    consumerBPath: "",
    calibrate: "",
    handcalib: "",
    foldEps: 0,
    foldcalib: "",
    exportPath: "",
    json: false,
    jobs: 1,
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
    } else if (arg.startsWith("--weights=")) {
      a.weights = arg.slice(10);
      a.weightsGiven = true;
    } else if (arg.startsWith("--temp=")) {
      const v = Number(arg.slice(7));
      if (!Number.isFinite(v) || v < 0) die(`--temp は 0 以上の実数: ${arg.slice(7)}`);
      a.temp = v;
      a.tempGiven = true;
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
    } else if (arg.startsWith("--table-b=")) {
      a.tableBPath = arg.slice(10);
      a.tableB = loadTable(a.tableBPath, "--table-b");
    } else if (arg.startsWith("--table=")) {
      a.tablePath = arg.slice(8);
      a.table = loadTable(a.tablePath);
    } else if (arg.startsWith("--ktune-b=")) {
      a.ktuneBPath = arg.slice(10);
      a.ktuneB = loadKtune(a.ktuneBPath, "--ktune-b");
    } else if (arg.startsWith("--ktune-opp=")) {
      a.ktuneOppPath = arg.slice(12);
      a.ktuneOpp = loadKtune(a.ktuneOppPath, "--ktune-opp");
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
    } else if (arg.startsWith("--handcalib=")) {
      a.handcalib = arg.slice(12);
      if (!a.handcalib) die("--handcalib には書き出し先のパスが要ります");
    } else if (arg.startsWith("--foldcalib=")) {
      a.foldcalib = arg.slice(12);
      if (!a.foldcalib) die("--foldcalib には書き出し先のパスが要ります");
    } else if (arg.startsWith("--fold-eps=")) {
      const v = Number(arg.slice(11));
      if (!Number.isFinite(v) || !(v > 0) || !(v < 1)) {
        die(`--fold-eps は 0 < X < 1 の実数: ${arg.slice(11)}`);
      }
      a.foldEps = v;
    } else if (arg.startsWith("--export=")) {
      a.exportPath = arg.slice(9);
      if (!a.exportPath) die("--export には書き出し先のパス (拡張子なしでも可) が要ります");
    } else if (arg.startsWith("--jobs=")) {
      const v = Number(arg.slice(7));
      if (!Number.isInteger(v) || v < 1) die(`--jobs は1以上の整数: ${arg.slice(7)}`);
      a.jobs = v;
    } else if (arg === "--json") a.json = true;
    else if (arg === "--plan") a.plan = true;
    else if (arg === "--standings") a.standings = true;
    else if (arg.startsWith("--seats=")) {
      const v = arg.slice(8);
      if (!/^[hrnok]{1,4}$/.test(v)) die(`--seats は h, r, n, o, k を4文字まで: ${v}`);
      // Short forms repeat the last letter: "hr" ⇒ "hrrr".
      a.seats = v.padEnd(4, v[v.length - 1]);
      a.seatsGiven = true;
    } else if (arg.startsWith("--glyphs=")) {
      const v = arg.slice(9);
      if (v !== "ascii" && v !== "kanji") die(`--glyphs は ascii か kanji: ${v}`);
      a.glyphs = v;
    } else if (arg.startsWith("-")) die(`不明なオプション: ${arg}`);
    else if (!a.cmd) a.cmd = arg;
    else die(`余分な引数: ${arg}`);
  }
  // A table decides the kinds; the derived string is what the reports print
  // and what the seat-0 rules below check. `argError` still refuses the
  // combination of `--table` with an EXPLICIT `--seats` (see `seatsGiven`).
  if (a.table) a.seats = kindString(a.table);
  const err = argError(a);
  if (err) die(err);
  return a;
}

/** The subset of `Args` the cross-flag rules below actually read. */
export type ArgCheck =
  & Pick<Args, "cmd" | "seats" | "calibrate">
  & Partial<Pick<Args, "handcalib" | "foldcalib" | "foldEps">>
  & Partial<
    Pick<
      Args,
      // The LOADED seat-0 vectors, not just their paths: `parseArgs` reads the
      // files before it calls `argError`, and M14's D6 rule is about a SECTION
      // inside one of them (`dealin`), which a path cannot answer.
      | "ktune"
      | "table"
      | "curriculum"
      | "consumerBPath"
      | "ktuneBPath"
      | "ktuneOppPath"
      | "consumerPath"
      | "ktunePath"
      | "tablePath"
      | "tableBPath"
      | "seatsGiven"
      | "weightsGiven"
      | "tempGiven"
      | "plan"
      | "standings"
      | "temp"
      | "record"
      | "recordAll"
      | "exportPath"
      | "json"
      | "jobs"
    >
  >;

/**
 * Cross-flag legality, as a MESSAGE rather than an exit. Every rule here refuses
 * a layout rather than ignoring it — a flag that quietly does nothing is worse
 * than one that is missing — and returning the message instead of calling
 * `die` is what lets a test read the rules without spawning a process.
 */
/** The vector seat 0 will actually carry — from `--ktune`, or from a `--table`
 *  file's `seats[0].ktune`. Both spellings reach the same seat, so a rule about
 *  a section of that vector has to read both. */
function ktuneOf(a: ArgCheck): KTune | undefined {
  return a.table ? a.table[0].ktune : a.ktune;
}

export function argError(a: ArgCheck): string | null {
  // `--table` is the modular spelling of the whole per-seat surface, so every
  // flag it subsumes conflicts instead of composing: a table is a COMPLETE
  // description of the four seats, and a flag that quietly lost to it would be
  // the silent no-op this function exists to refuse. Run-level wiring
  // (`--oracle`, `--noise`, `--record`, `--calibrate`, …) still composes.
  if (a.tablePath) {
    if (a.cmd === "play") return "--table は selfplay / bench / paired 専用です";
    if (a.seatsGiven) return "--table と --seats は併用できません (席種は table が決めます)";
    if (a.ktunePath) return "--table と --ktune は併用できません (席0の ktune は table に書きます)";
    if (a.ktuneOppPath) {
      return "--table と --ktune-opp は併用できません (席1-3の ktune は table に書きます)";
    }
    if (a.consumerPath) {
      return "--table と --consumer は併用できません (席ごとに table で指定します)";
    }
    if (a.standings) return "--table と --standings は併用できません (席ごとに table で指定します)";
    if (a.plan) return "--table と --plan は併用できません (席ごとに table で指定します)";
    if (a.curriculum !== undefined) {
      return "--table と --curriculum は併用できません (席ごとに table で指定します)";
    }
    if (a.weightsGiven) {
      return "--table と --weights は併用できません (n席の weights は table に書きます)";
    }
    if (a.tempGiven) return "--table と --temp は併用できません (n席の temp は table に書きます)";
    // A TableSpec arm ignores the flat seat options, so `--ktune-b` against a
    // table would build arm B identical to arm A and measure zero — silently.
    if (a.ktuneBPath) {
      return "--table と --ktune-b は併用できません (対照腕は --table-b で書きます)";
    }
    if (a.consumerBPath) {
      return "--table と --consumer-b は併用できません (対照腕は --table-b で書きます)";
    }
    // No implicit hhhh control for a table, either: the fallback carries no
    // environment guard, which is the exact leak that mis-crowned M11. A
    // table-based paired run states BOTH arms and gets the guard for free.
    if (a.cmd === "paired" && !a.tableBPath) {
      return "paired の --table には --table-b が要ります (対照腕も明示し、環境一致を検査します)";
    }
  }
  if (a.tableBPath) {
    if (!a.tablePath) {
      return "--table-b には --table が要ります (対照腕だけを table にはできません)";
    }
    if (a.ktuneBPath) return "--table-b と --ktune-b は併用できません";
    if (a.consumerBPath) return "--table-b と --consumer-b は併用できません";
  }
  // 2026-08-25 epoch: the "h" seat is a frozen copy of the default 計算 seat
  // and configurable by NOTHING, so each of these needs a "k" seat where its
  // routing points, or it would be accepted and reach nobody. `play` is
  // exempt from the ktune rule alone: there the vector also feeds the 助言
  // seat, which exists regardless of the CPU letters.
  if (a.cmd !== "play") {
    if (a.ktunePath && !a.seats.includes("k")) {
      return "--ktune には k席が要ります (h席は凍結済みでベクトルを受け取りません)";
    }
    if (a.ktuneOppPath && !a.seats.slice(1).includes("k")) {
      return "--ktune-opp には席1-3に k席が要ります (h席は凍結済みでベクトルを受け取りません)";
    }
    // The default control arm is frozen "hhhh", which no vector reaches — so
    // an opponents' vector under `paired` needs an incumbent control (same
    // seats both arms) for "the environment is identical" to stay true.
    if (a.ktuneOppPath && a.cmd === "paired" && !a.ktuneBPath && !a.consumerBPath) {
      return "--ktune-opp は paired では --ktune-b / --consumer-b (現行対照) と併用します " +
        "(既定の対照腕 hhhh は凍結席なので環境が一致しません)";
    }
    if (a.standings && a.seats[0] !== "k") {
      return `--standings は席0が k席のときだけ使えます: --seats=${a.seats} (h席は凍結済み)`;
    }
    if (a.consumerPath && a.seats[0] !== "k") {
      return `--consumer は席0が k席のときだけ使えます: --seats=${a.seats} (h席は凍結済み)`;
    }
  }
  if (a.ktuneBPath && a.seats[0] !== "k") {
    return `--ktune-b は席0が k席のときだけ使えます: --seats=${a.seats} (対照腕の席0に渡すものです)`;
  }
  if (a.consumerBPath && a.seats[0] !== "k") {
    return `--consumer-b は席0が k席のときだけ使えます: --seats=${a.seats} (対照腕の席0に渡すものです)`;
  }
  // The curriculum reads the live Table (that is what makes it a curriculum),
  // so it belongs to the headless drivers alone.
  if (a.curriculum !== undefined && a.cmd === "play") {
    return "--curriculum は selfplay / bench / paired 専用です (play では隠蔽情報を読めません)";
  }
  // --consumer-b / --ktune-b name the CONTROL arm, and only `paired` has one.
  if (a.consumerBPath && a.cmd !== "paired") return "--consumer-b は paired 専用です";
  if (a.ktuneBPath && a.cmd !== "paired") return "--ktune-b は paired 専用です";
  // `--json` is one line of paired's own measurement; nothing else emits it.
  if (a.json && a.cmd !== "paired") return "--json は paired 専用です";
  // The exporter writes ONE match per file. `paired` plays two arms over each
  // seed and `bench` measures speed; both would have to invent a naming scheme
  // for something nobody asked to keep, so the flag is refused there instead.
  if (a.exportPath && a.cmd !== "play" && a.cmd !== "selfplay") {
    return "--export は play / selfplay 専用です";
  }
  // `--jobs` shards a run of independent games and reassembles them in game
  // order. `play` has one game and a human in it; `bench` measures how fast ONE
  // thread plays; `paired` interleaves two arms over each seed and its whole
  // point is that the two share a wall. Only `selfplay` has anything to shard.
  // A `--jobs=1` elsewhere is not refused: 1 job is not a request that would be
  // ignored, it is exactly the loop those commands already run.
  if ((a.jobs ?? 1) > 1 && a.cmd !== "selfplay") return "--jobs は selfplay 専用です";
  // Everything below reaches a seat only through a headless driver's options,
  // so under `play` each one would be accepted and then quietly do nothing.
  if (a.cmd === "play") {
    if (a.standings) return "--standings は selfplay / bench / paired 専用です";
    if (a.consumerPath) return "--consumer は selfplay / bench / paired 専用です";
    if (a.temp) return "--temp は selfplay / bench / paired 専用です (play の n席は常に決定的)";
    if (a.record || a.recordAll) return "--record / --record-all は selfplay 専用です";
    // `cmdPlay` builds its seats by hand, never through `openArm`, so the
    // opponents' vector would be accepted and then reach nobody.
    if (a.ktuneOppPath) return "--ktune-opp は selfplay / bench / paired 専用です";
  }
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
    if ((a.jobs ?? 1) > 1) {
      // Every other per-game output of a sharded run has a seam to buffer it at
      // (a trajectory line is text, a MatchResult is a value). A calibration
      // record has none: it is written from INSIDE a decision, through a live
      // writer the seat holds, and the file's own header names the run. Sharding
      // it would either interleave the rows of four workers or need a second,
      // parallel notion of what a calibration file is. Refused rather than
      // shipped with an ordering nobody can reproduce.
      return "--calibrate と --jobs は併用できません (較正記録は1スレッドで書きます)";
    }
    if (a.curriculum !== undefined) {
      // The curriculum REPLACES the computed answer with truth part of the time;
      // calibration measures the computed answer against truth. Recording under
      // a curriculum would grade a seat nobody ships.
      return "--calibrate と --curriculum は併用できません (較正するのは素の計算の読みです)";
    }
    if (ktuneOf(a)?.dealin) {
      // M14 D6, and the third member of the same family as the two rules above:
      // a lane recorded under the head being fitted is a lane censored by that
      // head's own reads — `handvalue.ts`'s lesson, measured (+0.11 WORSE).
      return "--calibrate と dealin ブロックは併用できません " +
        "(較正レーンは素の計算の読みの上で録ります — 学習ヘッドは自分の出力で学習できません)";
    }
  }
  // The hand recorder needs far less than the deal-in one — no oracle tap, since
  // the label is the round's own result — but it still needs a seat that HAS a
  // hand-value model and a driver that plays whole rounds to label against.
  if (a.handcalib) {
    if (a.cmd !== "selfplay" && a.cmd !== "paired") {
      return "--handcalib は selfplay / paired 専用です (局の結末で札を貼るので、通しで打つ駆動が要ります)";
    }
    if (a.seats[0] !== "k" && a.seats[0] !== "h") {
      return `--handcalib は席0が k席 (計算) か h席 (凍結基準) のときだけ使えます: --seats=${a.seats}\n` +
        "手牌価値の読みを持たない席 (n / r / o) には記録するものがありません。";
    }
    if ((a.jobs ?? 1) > 1) {
      // Same seam problem as `--calibrate`, one layer worse: a hand record is
      // buffered until its round ends, so a sharded run would have four
      // independent buffers flushing into one file in wall-clock order. Refused
      // rather than shipped with an ordering nobody can reproduce.
      return "--handcalib と --jobs は併用できません (手牌価値の記録は1スレッドで書きます)";
    }
  }
  // M13's lane. Narrower than the hand lane on one axis (the head routes to "k"
  // seats only, so an "h" seat 0 has nothing to record) and wider on another:
  // `--fold-eps` deliberately PERTURBS the seat, so it is refused on its own.
  if (a.foldcalib) {
    if (a.cmd !== "selfplay" && a.cmd !== "paired") {
      return "--foldcalib は selfplay / paired 専用です (局の結末を報酬にするので、通しで打つ駆動が要ります)";
    }
    if (a.seats[0] !== "k") {
      return `--foldcalib は席0が k席 (計算) のときだけ使えます: --seats=${a.seats}\n` +
        "押し引きヘッドが載るのは k席だけなので、他の席で録っても当てはめる先がありません。";
    }
    if ((a.jobs ?? 1) > 1) {
      // The same seam problem as `--handcalib`: a fold record is buffered until
      // its round ends, so four workers would flush four independent buffers
      // into one file in wall-clock order. Refused rather than shipped with an
      // ordering nobody can reproduce.
      return "--foldcalib と --jobs は併用できません (押し引きの記録は1スレッドで書きます)";
    }
  } else if (a.foldEps) {
    return "--fold-eps は --foldcalib と一緒に使います (記録しないのに手を曲げても仕方がありません)";
  }
  return null;
}
