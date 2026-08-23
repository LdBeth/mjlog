// The application: layout, event handling, the render loop, and `HumanPolicy`.
//
// Rendering is event-driven — a new PublicEvent, a keypress, a 100 ms tick while
// a decision timer runs, or SIGWINCH. There is no fixed-rate loop, so an idle
// table costs nothing.
//
// The one hard safety rule: a decision is only ever resolved with an object that
// is *identically* an element of `obs.legal`. Every key handler goes through
// `submit()`, which looks the action up in `legal` and refuses anything else, so
// an illegal move is impossible by construction rather than by review.

import type { Meld, Tile } from "mjrender/model.ts";
import { roundName, tileType } from "mjrender/tiles.ts";
import type { Observation } from "../observe.ts";
import type { Policy } from "../policy.ts";
import type { SyncPolicy } from "../policy.ts";
import type { MatchResult } from "../match.ts";
import type { RuleConfig } from "../rules.ts";
import { finalStandings } from "../score.ts";
import type { Action, PublicEvent, RoundOutcome, Seat, Violation } from "../types.ts";
import { SEATS } from "../types.ts";
import { lineWidth, Screen, sp } from "./screen.ts";
import type { Line } from "./screen.ts";
import type { GlyphMode, GlyphOpts } from "./glyph.ts";
import { REL_LABEL, tileText, WINDS } from "./glyph.ts";
import * as W from "./widgets.ts";
import type { Ctx, Overlay, Phase, TimerState } from "./widgets.ts";
import { decodeKeys, isCtrlC, type KeyEvent, readKeys } from "./input.ts";
import * as term from "./term.ts";
import { padEnd, SGR, type Size, size, width } from "./term.ts";

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

const DIM = SGR.gray;

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

interface Layout {
  cols: number;
  rows: number;
  x0: number;
  inner: number;
  riverRows: number;
  panelRows: number;
  headerY: number;
  toimenY: number;
  sideY: number;
  sepY: number;
  ownY: number;
  ownRiverY: number;
  handY: number;
  caretY: number;
  metricsY: number;
  dangerY: number;
  ledgerHdrY: number;
  ledgerY: number;
  logHdrY: number;
  logY: number;
  logRows: number;
  actionY: number;
  bottomY: number;
}

function layout(cols: number, rows: number): Layout {
  const riverRows = rows >= 30 ? 4 : rows >= 26 ? 3 : 2;
  const panelRows = riverRows + 2; // head + rivers + melds
  const headerY = 0;
  const toimenY = headerY + 1;
  const sideY = toimenY + panelRows;
  const sepY = sideY + panelRows;
  const ownY = sepY + 1;
  const ownRiverY = ownY + 1;
  const handY = ownRiverY + 1;
  const caretY = handY + 1;
  const metricsY = caretY + 1;
  const dangerY = metricsY + 1;
  const ledgerHdrY = dangerY + 1;
  const ledgerY = ledgerHdrY + 1;
  const logHdrY = ledgerY + 1;
  const logY = logHdrY + 1;
  const bottomY = rows - 1;
  const actionY = rows - 2;
  return {
    cols,
    rows,
    x0: 1,
    inner: cols - 2,
    riverRows,
    panelRows,
    headerY,
    toimenY,
    sideY,
    sepY,
    ownY,
    ownRiverY,
    handY,
    caretY,
    metricsY,
    dangerY,
    ledgerHdrY,
    ledgerY,
    logHdrY,
    logY,
    logRows: Math.max(0, actionY - logY),
    actionY,
    bottomY,
  };
}

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

/**
 * The seat the person at the keyboard plays. It owns no UI state; it hands the
 * observation to the App and returns the promise the key handler resolves.
 */
export class HumanPolicy implements Policy {
  readonly name: string;

  #app: App;

  constructor(name: string, app: App) {
    this.name = name;
    this.#app = app;
  }

  decide(obs: Observation): Promise<Action> {
    return this.#app.awaitDecision(obs);
  }

  notify(e: PublicEvent): void {
    this.#app.onEvent(e);
  }
}

/**
 * Wraps a synchronous CPU policy in a real delay. Without it the whole CPU
 * sequence between two human decisions resolves in one microtask burst and the
 * board simply teleports; the delay also gives the render loop a chance to run.
 */
export class PacedPolicy implements Policy {
  readonly name: string;
  #inner: SyncPolicy;
  #delay: () => number | Promise<number>;

  constructor(inner: SyncPolicy, delay: () => number | Promise<number>) {
    this.name = inner.name;
    this.#inner = inner;
    this.#delay = delay;
  }

  async decide(obs: Observation): Promise<Action> {
    const a = this.#inner.decide(obs);
    const ms = await this.#delay();
    if (ms > 0) await sleep(ms);
    return a;
  }

  reset(seed: number): void {
    this.#inner.reset?.(seed);
  }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export interface AppOptions {
  glyphs: GlyphMode;
  aka: ReadonlySet<Tile>;
  names: string[];
  /** Free allowance granted afresh at the start of every decision, ms. */
  timerTurnMs: number;
  /** One pool for the WHOLE match, drawn on only once the turn allowance is
   *  spent. Never refills. */
  timerBankMs: number;
  /** Per-CPU-decision delay, ms. 0 makes the game instant (for debugging). */
  cpuDelayMs: number;
  cfg: RuleConfig;
  /** Absolute seat the human plays. Defaults to 0. */
  humanSeat?: Seat;
  /** Skip the opening sequence. */
  noIntro?: boolean;
  /** Where frames go. Defaults to stdout; tests pass a sink to stay quiet. */
  write?: (s: string) => void;
}

const ROW_ALIASES: Record<string, string> = { "\\": "¥" };

export class App {
  readonly human: HumanPolicy;
  private scr: Screen;
  private opts: AppOptions;
  private glyph: GlyphOpts;

  private obs: Observation | null = null;
  private resolver: ((a: Action) => void) | null = null;
  private phase: Phase = "idle";
  private slots: Tile[] = [];
  private drawnIndex = -1;
  private cursor = 0;
  private selectable: boolean[] = [];
  private riichiArmed = false;
  private overlayState: Overlay | null = null;
  private overlayResolve: (() => void) | null = null;
  private message = "";
  private log: string[] = [];
  private ledger: Violation[] = [];
  private startedAt = 0;
  /** Bank left for the rest of the match; never refills.  */
  private bankLeftMs = 0;
  /** Set when a deal arrives, consumed by the next decision (deal animation). */
  private pendingDeal = false;
  private introRunning = false;
  private introSkipped = false;
  private ticker: ReturnType<typeof setInterval> | null = null;
  private dirty = false;
  /** A pending render that must rebuild the whole board, not just the clock. */
  private dirtyFull = false;
  /** True once a full board frame is on screen, so a header-only repaint of it
   *  is meaningful. Cleared by anything that repaints something else. */
  private headerOnlyOk = false;
  private scheduled = false;
  private stopped = false;
  private unResize: (() => void) | null = null;
  /** Terminal size, refreshed on SIGWINCH — `Deno.consoleSize()` is a syscall
   *  and the timer ticks ten times a second. */
  private termSize: Size;

  /** Live corrections for fields an Observation froze at decision time. */
  private liveRiichi = [false, false, false, false];
  private liveScores: number[] | null = null;
  private liveWall: number | null = null;
  /** Armed while a 局結果 overlay is up; resolved by the keypress that closes it. */
  private resultGate: Promise<void> | null = null;

  constructor(opts: AppOptions) {
    this.opts = opts;
    this.glyph = { mode: opts.glyphs, aka: opts.aka };
    this.termSize = size();
    this.scr = new Screen(this.termSize.cols, this.termSize.rows, { write: opts.write });
    this.human = new HumanPolicy(opts.names[opts.humanSeat ?? 0] ?? "あなた", this);
    this.bankLeftMs = opts.timerBankMs;
  }

  /** Block until the player dismisses the round-result overlay, if one is up. */
  private async gate(): Promise<void> {
    if (this.resultGate) {
      await this.resultGate;
      this.resultGate = null;
    }
  }

  /** Delay a CPU decision should wait. It also parks on the 局結果 overlay, so
   *  the next hand cannot start behind the player's back. */
  async paceDelay(): Promise<number> {
    await this.gate();
    return this.opts.cpuDelayMs;
  }

  // --- lifecycle -----------------------------------------------------------

  start(): void {
    term.enter();
    this.unResize = term.onResize(() => {
      // `resize` already repaints in full, and no-ops when the size is the same.
      this.termSize = size();
      this.scr.resize(this.termSize.cols, this.termSize.rows);
      this.requestRender();
    });
    void this.inputLoop();
    this.requestRender();
  }

  /**
   * Opening sequence: title, seating, then the deal. Purely atmosphere — it
   * runs before the first hand and any key skips it. Without it the game jumps
   * from a bare shell straight into a live decision, which reads as a glitch.
   */
  async intro(): Promise<void> {
    if (this.opts.noIntro) return;
    this.introRunning = true;
    this.introSkipped = false;
    try {
      await this.introSequence();
    } finally {
      // However we got here — finished, skipped, or torn down — hand a clean
      // screen to the game.
      this.introRunning = false;
      this.scr.fullRepaint();
      this.requestRender();
    }
  }

  private async introSequence(): Promise<void> {
    /** Sleep in small steps so a keypress lands promptly. False ⇒ bail out. */
    const pause = async (ms: number): Promise<boolean> => {
      for (let waited = 0; waited < ms; waited += 20) {
        if (this.introSkipped || this.stopped) return false;
        await sleep(20);
      }
      return !this.introSkipped && !this.stopped;
    };

    const cfg = this.opts.cfg;
    const title = "雀  鬼  流";
    const card = [
      title,
      "",
      `${cfg.hanchan ? "東南戦" : "東風戦"}  ${cfg.startScore}点持ち / ${cfg.returnScore}点返し`,
      "赤五筒 二枚  一発・裏ドラ有り",
      "禁じ手は止めない — 評価点で記録する",
    ];

    const frame = (lines: Array<{ text: string; sgr?: string }>) => {
      const { cols, rows } = this.termSize;
      this.headerOnlyOk = false;
      this.scr.clear();
      const top = Math.max(0, Math.floor((rows - lines.length) / 2) - 2);
      lines.forEach((l, i) => {
        const x = Math.max(0, Math.floor((cols - width(l.text)) / 2));
        this.scr.draw(x, top + i, l.text, l.sgr ?? "");
      });
      this.scr.flush();
    };

    // 1. The title card builds up a line at a time.
    for (let n = 1; n <= card.length; n++) {
      frame(
        card.slice(0, n).map((text, i) => ({
          text,
          sgr: i === 0 ? SGR.bold : i === n - 1 ? SGR.brightWhite : SGR.gray,
        })),
      );
      if (!await pause(n === 1 ? 400 : 170)) return;
    }
    if (!await pause(340)) return;

    // 2. Then the seating, one player at a time.
    const seats = SEATS.map((s) => ({
      text: `${WINDS[s]}家   ${this.opts.names[s] ?? `P${s}`}`,
      sgr: s === (this.opts.humanSeat ?? 0) ? SGR.brightGreen : SGR.gray,
    }));
    for (let n = 1; n <= seats.length; n++) {
      frame([{ text: title, sgr: SGR.bold }, { text: "" }, ...seats.slice(0, n)]);
      if (!await pause(150)) return;
    }
    if (!await pause(380)) return;

    // 3. Hand off to the deal.
    frame([
      { text: title, sgr: SGR.bold },
      { text: "" },
      ...seats,
      { text: "" },
      { text: "配 牌", sgr: SGR.bold },
    ]);
    await pause(460);
  }

  /**
   * Reveal the dealt hand in Tenhou's real order — three blocks of four, then
   * the last tile — so the deal reads as a deal rather than a jump cut. Runs
   * BEFORE the decision clock starts.
   */
  private async animateDeal(): Promise<void> {
    if (this.opts.noIntro || !this.obs) return;
    const full = this.slots;
    const stops = [0, 4, 8, 12, full.length];
    for (const n of stops) {
      if (this.stopped) break;
      this.slots = full.slice(0, n);
      this.selectable = this.slots.map(() => false);
      this.cursor = Math.max(0, this.slots.length - 1);
      this.render();
      await sleep(n === full.length ? 220 : 150);
    }
    this.slots = full;
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.stopTicker();
    this.unResize?.();
    term.leave();
  }

  private quit(code = 0): never {
    this.stop();
    Deno.exit(code);
  }

  // --- decisions -----------------------------------------------------------

  /** Present `obs` and resolve once the key handler picks a legal action. */
  async awaitDecision(obs: Observation): Promise<Action> {
    await this.gate();
    if (this.overlayState?.kind === "text") this.closeOverlay();

    this.obs = obs;
    this.message = "";
    this.riichiArmed = false;
    this.liveRiichi = [false, false, false, false];
    this.liveWall = null;

    const isClaim = obs.legal.some((a) => a.t === "pass");
    this.phase = isClaim ? "claim" : "turn";
    this.buildSlots(obs);
    if (this.pendingDeal) {
      this.pendingDeal = false;
      // Deliberately before startedAt: the countdown must not run while the
      // player is watching an animation they cannot act during.
      await this.animateDeal();
      this.refreshSelectable();
    }

    // A forced move is not a decision. Post-riichi the hand is locked to the
    // drawn tile, so unless a kan or a win is also on offer there is nothing to
    // choose — play it without prompting and without starting the clock. Being
    // made to press a key every turn of a riichi is not a game mechanic.
    //
    // Only *that* shape auto-plays. Other singleton `legal` lists (a kuikae-
    // forced lone tedashi, say) are still real choices from the player's side of
    // the table — they get the normal prompt, so the move is at least seen.
    const only = obs.legal.length === 1 ? obs.legal[0] : null;
    if (only && only.t === "discard" && only.tsumogiri) {
      this.phase = "idle";
      // The clock is never started, so the move costs nothing from the bank.
      this.requestRender();
      const ms = await this.paceDelay();
      if (ms > 0) await sleep(ms);
      return only;
    }

    this.startedAt = Date.now();
    this.startTicker();

    if (isClaim) {
      this.overlayState = { kind: "call" };
    } else {
      this.overlayState = null;
    }
    this.requestRender();

    return await new Promise<Action>((resolve) => {
      this.resolver = resolve;
    });
  }

  /** Bank left for the rest of the match, in ms. Exposed for tests. */
  bankRemainingMs(): number {
    return this.timerState().bankLeftMs;
  }

  /** The countdown as the header renders it. Exposed for tests. */
  timerSnapshot(): TimerState {
    return this.timerState();
  }

  /** The text overlay's body, one plain string per line. Exposed for tests. */
  overlayLines(): string[] | null {
    const o = this.overlayState;
    if (!o || o.kind !== "text") return null;
    return o.body.map((l) => l.map((s) => s.text).join(""));
  }

  /** Remaining turn allowance / match bank for the decision in flight. */
  private timerState(): TimerState {
    const base = this.opts.timerTurnMs;
    const bank = this.opts.timerBankMs;
    if (this.phase === "idle") {
      return {
        turnMs: base,
        bankMs: bank,
        turnLeftMs: base,
        bankLeftMs: this.bankLeftMs,
        overMs: 0,
      };
    }
    const spent = Date.now() - this.startedAt;
    const overrun = Math.max(0, spent - base);
    const bankLeftMs = Math.max(0, this.bankLeftMs - overrun);
    return {
      turnMs: base,
      bankMs: bank,
      turnLeftMs: Math.max(0, base - spent),
      bankLeftMs,
      // Past base+bank the clock keeps running into the red. Nothing is forced
      // and nothing is charged: the countdown is pressure to look at, not a
      // penalty and not a stolen turn.
      overMs: Math.max(0, overrun - this.bankLeftMs),
    };
  }

  private buildSlots(obs: Observation): void {
    const rest = [...obs.hand];
    if (obs.drawn !== null) {
      const i = rest.lastIndexOf(obs.drawn);
      if (i >= 0) rest.splice(i, 1);
    }
    rest.sort((a, b) => a - b);
    this.slots = obs.drawn !== null ? [...rest, obs.drawn] : rest;
    this.drawnIndex = obs.drawn !== null ? rest.length : -1;
    this.refreshSelectable();
    // Default to the drawn tile: ツモ切り is the most common single keystroke.
    const start = this.drawnIndex >= 0 ? this.drawnIndex : this.slots.length - 1;
    this.cursor = this.selectable[start] ? start : this.nextSelectable(start, 1, true);
  }

  private refreshSelectable(): void {
    this.selectable = this.slots.map((id) => this.findDiscard(id) !== null);
  }

  /**
   * The legal discard for a slot. `legal` enumerates one representative id per
   * tile type (plus every aka copy), so a slot matches either exactly or by
   * type with the same aka-ness.
   */
  private findDiscard(tile: Tile): Action | null {
    const obs = this.obs;
    if (!obs) return null;
    const want = this.riichiArmed;
    const aka = this.opts.aka.has(tile);
    const exact = obs.legal.find(
      (a) => a.t === "discard" && a.tile === tile && a.riichi === want,
    );
    if (exact) return exact;
    const byType = obs.legal.find(
      (a) =>
        a.t === "discard" && a.riichi === want &&
        tileType(a.tile) === tileType(tile) && this.opts.aka.has(a.tile) === aka,
    );
    return byType ?? null;
  }

  private nextSelectable(from: number, dir: number, includeSelf = false): number {
    const n = this.slots.length;
    if (n === 0) return 0;
    for (let k = includeSelf ? 0 : 1; k <= n; k++) {
      const i = (from + dir * k + n * 2) % n;
      if (this.selectable[i]) return i;
    }
    return from;
  }

  /**
   * Resolve the pending decision. The `includes` guard is the hard safety rule
   * from the file header: every key handler picks its action out of `obs.legal`,
   * and anything that did not come from there is refused rather than played.
   */
  private submit(action: Action | null): void {
    if (!action || !this.resolver || !this.obs) return;
    if (!this.obs.legal.includes(action)) return;
    const elapsed = Date.now() - this.startedAt;
    // Whatever ran past this turn's allowance comes out of the match bank.
    this.bankLeftMs = Math.max(0, this.bankLeftMs - Math.max(0, elapsed - this.opts.timerTurnMs));

    const resolve = this.resolver;
    this.resolver = null;
    this.phase = "idle";
    this.riichiArmed = false;
    this.overlayState = null;
    this.stopTicker();

    this.requestRender();
    resolve(action);
  }

  // --- events --------------------------------------------------------------

  onEvent(e: PublicEvent): void {
    switch (e.e) {
      case "deal":
        this.pendingDeal = true;
        this.liveScores = [...e.scores];
        this.liveRiichi = [false, false, false, false];
        this.liveWall = null;
        this.pushLog(`── ${roundName(e.kyoku)}${e.honba}本場 ──`);
        break;
      case "draw":
        if (!e.rinshan && this.liveWall !== null) this.liveWall--;
        break;
      case "discard":
        this.lastDiscard = { tile: e.tile, from: e.who };
        this.pushLog(
          `P${e.who} 打 ${tileText(e.tile, this.glyph)}` +
            (e.riichi ? " リーチ" : e.tsumogiri ? " (ツモ切)" : ""),
        );
        break;
      case "call":
        this.pushLog(`P${e.meld.who} ${W.MELD_LABEL[e.meld.kind]} ${this.meldText(e.meld)}`);
        break;
      case "riichi":
        if (e.step === 1) this.liveRiichi[e.who] = true;
        break;
      case "dora":
        this.pushLog(`新ドラ表示 ${tileText(e.indicator, this.glyph)}`);
        break;
      case "violation":
        this.ledger.push(e.v);
        this.pushLog(`違反 P${e.v.seat} ${e.v.label} -${e.v.points}`);
        break;
      case "result":
        this.onResult(e);
        break;
    }
    this.requestRender();
  }

  private lastDiscard: { tile: Tile; from: Seat } | null = null;

  private onResult(ev: Extract<PublicEvent, { e: "result" }>): void {
    const outcome = ev.outcome;
    const body: Line[] = [];
    if (outcome.kind === "agari") {
      for (const w of outcome.wins) {
        const how = w.fromWho === w.who ? "ツモ" : `ロン (P${w.fromWho})`;
        body.push([
          sp(`P${w.who} ${how}  `, SGR.brightGreen),
          sp(`${w.han}翻${w.fu}符 ${w.points}点  `),
          sp("和了牌 "),
          sp(tileText(w.winTile, this.glyph)),
        ]);
        this.pushLog(`和了 P${w.who} ${how} ${w.points}点`);
      }
    } else {
      body.push([sp(`流局 (${outcome.draw})`, SGR.yellow)]);
      body.push([
        sp("聴牌 "),
        sp(outcome.tenpai.map((t, s) => `P${s}:${t ? "聴牌" : "不聴"}`).join("  ")),
      ]);
      this.pushLog(`流局 ${outcome.draw}`);
    }
    if (ev.hands) body.push(...this.revealLines(ev.hands, ev.melds, outcome));
    body.push([]);
    body.push([sp("点棒移動 " + outcome.deltas.map((d, s) => `P${s}${fmtDelta(d)}`).join("  "))]);
    if (this.liveScores) {
      for (let s = 0; s < 4; s++) this.liveScores[s] += outcome.deltas[s];
    }
    this.overlayState = {
      kind: "text",
      title: "局結果",
      body,
      footer: "任意のキーで次局へ",
    };
    // A hard gate, not a timed pause: the match may not roll on until the
    // player has actually seen the result and said so.
    this.resultGate = new Promise((r) => {
      this.overlayResolve = r;
    });
  }

  private meldText(m: Meld): string {
    return m.tiles.map((t) => tileText(t, this.glyph)).join("");
  }

  /**
   * 手牌公開: every seat's hand at the end of the round, not just the winner's
   * or the tenpai ones. The human seat and the winner(s) are picked out by
   * colour; riichi is called out in words, since colour is already spoken for.
   */
  private revealLines(
    hands: readonly Tile[][],
    melds: readonly Meld[][] | undefined,
    outcome: RoundOutcome,
  ): Line[] {
    const winners = new Set<Seat>(
      outcome.kind === "agari" ? outcome.wins.map((w) => w.who) : [],
    );
    const me = this.opts.humanSeat ?? 0;
    const out: Line[] = [[], [sp("手牌公開", DIM)]];
    for (const s of SEATS) {
      const line: Line = [
        sp(
          `P${s}`,
          winners.has(s) ? SGR.brightGreen : s === me ? SGR.brightCyan : DIM,
        ),
        sp(this.liveRiichi[s] ? " リーチ " : " ", this.liveRiichi[s] ? SGR.yellow : ""),
      ];
      const m = melds?.[s] ?? [];
      for (const meld of m) line.push(sp(`[${this.meldText(meld)}]`, DIM));
      if (m.length > 0) line.push(sp(" "));
      line.push(sp((hands[s] ?? []).map((t) => tileText(t, this.glyph)).join("")));
      out.push(line);
    }
    return out;
  }

  private pushLog(s: string): void {
    this.log.push(s);
    if (this.log.length > 200) this.log.splice(0, this.log.length - 200);
  }

  // --- input ---------------------------------------------------------------

  private async inputLoop(): Promise<void> {
    try {
      for await (const k of readKeys()) {
        if (this.stopped) return;
        this.handleKey(k);
      }
    } catch {
      // stdin closed — nothing more to feed. Play continues, but a 局結果
      // overlay can no longer be dismissed, so the match parks at the gate.
    }
  }

  /** Exposed for tests and for replaying a scripted key sequence. */
  feed(chunk: string): void {
    for (const k of decodeKeys(chunk)) this.handleKey(k);
  }

  handleKey(k: KeyEvent): void {
    if (isCtrlC(k)) this.quit(130);

    // Any key cuts the opening short — it is atmosphere, never a gate.
    if (this.introRunning) {
      this.introSkipped = true;
      return;
    }

    if (this.overlayState) {
      this.overlayKey(k);
      return;
    }
    if (this.phase === "turn") this.turnKey(k);
    else this.idleKey(k);
    this.requestRender();
  }

  private idleKey(k: KeyEvent): void {
    switch (k.name) {
      case "?":
        this.overlayState = { kind: "help" };
        break;
      case "d":
        this.overlayState = { kind: "danger" };
        break;
      case "s":
        this.dumpSnapshot();
        break;
      case "q":
        this.overlayState = { kind: "quit" };
        break;
    }
  }

  private turnKey(k: KeyEvent): void {
    const obs = this.obs;
    if (!obs) return;
    switch (k.name) {
      case "left":
      case "h":
        this.cursor = this.nextSelectable(this.cursor, -1);
        return;
      case "right":
      case "l":
        this.cursor = this.nextSelectable(this.cursor, 1);
        return;
      case "enter":
      case "space":
        this.submit(this.findDiscard(this.slots[this.cursor]));
        return;
      case "t": {
        const tg = obs.legal.find(
          (a) => a.t === "discard" && a.tsumogiri && a.riichi === this.riichiArmed,
        );
        if (tg) this.submit(tg);
        else this.message = "ツモ切りできません";
        return;
      }
      case "r": {
        const can = obs.legal.some((a) => a.t === "discard" && a.riichi);
        if (!can) {
          this.message = "リーチできません";
          return;
        }
        this.riichiArmed = true;
        this.refreshSelectable();
        this.cursor = this.selectable[this.cursor]
          ? this.cursor
          : this.nextSelectable(this.cursor, 1, true);
        this.message = "リーチ: 切る牌を選択";
        return;
      }
      case "escape":
        if (this.riichiArmed) {
          this.riichiArmed = false;
          this.refreshSelectable();
          this.message = "";
        }
        return;
      case "k": {
        const kans = obs.legal.filter((a) => a.t === "ankan" || a.t === "kakan");
        if (kans.length === 0) this.message = "カンできません";
        else this.overlayState = { kind: "pick", options: kans, title: "カン" };
        return;
      }
      case "a": {
        const win = obs.legal.find((a) => a.t === "tsumo" || a.t === "ron");
        if (win) this.submit(win);
        else this.message = "和了できません";
        return;
      }
      default: {
        const i = W.ROW_KEYS.indexOf(ROW_ALIASES[k.name] ?? k.name);
        if (i >= 0) {
          this.rowKey(i);
          return;
        }
        this.idleKey(k);
      }
    }
  }

  /**
   * The number row, one keystroke per discard. JIS layout: 1-9, 0, -, ^, ¥
   * are thirteen keys in a line, exactly a hand's width, read left to right
   * like the tiles on screen; the drawn tile stays on `t`. The key DISCARDS
   * rather than moving the cursor: cursor-then-Enter is what made the TUI
   * slow to play, and Enter/←→ remain for anyone who wants to look first.
   * A slot the referee would not let go (riichi armed, not a legal discard)
   * says so instead of doing nothing.
   */
  private rowKey(i: number): void {
    if (i >= this.slots.length || i === this.drawnIndex) {
      this.message = "その位置に牌はありません (ツモ切りは t)";
      return;
    }
    if (!this.selectable[i]) {
      this.cursor = i;
      this.message = this.riichiArmed ? "その牌ではリーチできません" : "その牌は切れません";
      return;
    }
    this.cursor = i;
    this.submit(this.findDiscard(this.slots[i]));
  }

  private overlayKey(k: KeyEvent): void {
    const o = this.overlayState!;
    switch (o.kind) {
      case "help":
      case "danger":
        this.closeOverlay();
        break;
      case "text":
        this.closeOverlay();
        this.overlayResolve?.();
        this.overlayResolve = null;
        break;
      case "quit":
        if (k.name === "y") this.quit(0);
        else if (k.name === "escape" || k.name === "n") this.closeOverlay();
        break;
      case "pick":
        // Esc backs out one level: to the call prompt if that is what opened
        // the menu, otherwise to the board.
        if (k.name === "escape") this.closeOverlay();
        else if (/^[1-9]$/.test(k.name)) {
          const a = o.options[Number(k.name) - 1];
          if (a) this.submit(a);
        }
        break;
      case "call":
        this.callKey(k);
        break;
    }
    this.requestRender();
  }

  private callKey(k: KeyEvent): void {
    const obs = this.obs;
    if (!obs) return;
    const pick = (kind: Action["t"], title: string) => {
      const opts = obs.legal.filter((a) => a.t === kind);
      if (opts.length === 0) {
        this.message = "選べません";
      } else if (opts.length === 1) {
        this.submit(opts[0]);
      } else {
        this.overlayState = { kind: "pick", options: opts, title };
      }
    };
    switch (k.name) {
      case "p":
        pick("pon", "ポンの形を選択");
        break;
      case "c":
        pick("chi", "チーの形を選択");
        break;
      case "n":
        pick("daiminkan", "大明槓");
        break;
      case "a":
      case "y":
      case "enter": {
        const ron = obs.legal.find((a) => a.t === "ron");
        if (ron) this.submit(ron);
        else this.submit(obs.legal.find((a) => a.t === "pass") ?? null);
        break;
      }
      case "escape":
        this.submit(obs.legal.find((a) => a.t === "pass") ?? null);
        break;
      case "d":
        this.overlayState = { kind: "danger" };
        break;
      case "?":
        this.overlayState = { kind: "help" };
        break;
      case "q":
        this.overlayState = { kind: "quit" };
        break;
    }
  }

  private closeOverlay(): void {
    // Returning from an informational overlay during a claim must restore the
    // call prompt, or the player would be left with no way to answer it.
    this.overlayState = this.phase === "claim" && this.resolver ? { kind: "call" } : null;
  }

  /** `runMatch` does not expose its Table, so this dumps what the seat can see. */
  private dumpSnapshot(): void {
    const o = this.obs;
    if (!o) return;
    this.pushLog(
      `── 記録 ${roundName(o.kyoku)}${o.honba}本場 ${o.junme}巡目 残り${o.wallRemaining}`,
    );
    this.pushLog(`手牌 ${this.slots.map((x) => tileText(x, this.glyph)).join("")}`);
    for (let r = 0; r < 4; r++) {
      const abs = (o.seat + r) % 4;
      const river = o.rivers[r].map((e) => tileText(e.tile, this.glyph)).join("");
      this.pushLog(`P${abs} ${REL_LABEL[r]} ${o.scores[r]} 河 ${river}`);
    }
  }

  // --- rendering -----------------------------------------------------------

  private startTicker(): void {
    if (this.ticker !== null) return;
    // The clock is the only thing that moved, so only the clock is repainted.
    this.ticker = setInterval(() => this.schedule(false), 100);
  }

  private stopTicker(): void {
    if (this.ticker === null) return;
    clearInterval(this.ticker);
    this.ticker = null;
  }

  requestRender(): void {
    this.schedule(true);
  }

  /**
   * Coalesce render requests into one frame. `full` rebuilds every widget; a
   * request that is not full only redraws the header clock — but if ANY full
   * request lands in the same batch, the whole batch renders full.
   */
  private schedule(full: boolean): void {
    this.dirty = true;
    if (full) this.dirtyFull = true;
    if (this.scheduled) return;
    this.scheduled = true;
    setTimeout(() => {
      this.scheduled = false;
      if (!this.dirty || this.stopped) return;
      const wasFull = this.dirtyFull;
      this.dirtyFull = false;
      if (wasFull || !this.headerOnlyOk) this.render();
      else this.renderHeader();
    }, 0);
  }

  /**
   * The tick fast path: overdraw the header row on the frame already on screen
   * and flush. Only valid straight after a full board frame at the same size —
   * `headerOnlyOk` tracks exactly that — and safe under an overlay, since the
   * overlay box never reaches row 0.
   */
  private renderHeader(): void {
    this.dirty = false;
    const L = layout(this.termSize.cols, this.termSize.rows);
    const c = this.ctx(L);
    this.scr.retain();
    this.scr.drawLine(L.x0 + 1, L.headerY, W.headerBar(c, L.inner - 2), L.inner - 2);
    this.scr.flush();
  }

  private ctx(L: Layout): Ctx {
    const o = this.obs;
    let obs = o;
    if (o) {
      // Patch the frozen snapshot with what we have seen since it was taken.
      const riichi = o.riichi.map((v, r) => v || this.liveRiichi[(o.seat + r) % 4]);
      const scores = this.liveScores
        ? o.scores.map((_, r) => this.liveScores![(o.seat + r) % 4])
        : o.scores;
      const wallRemaining = this.liveWall ?? o.wallRemaining;
      obs = { ...o, riichi, scores, wallRemaining };
    }
    return {
      obs,
      glyph: this.glyph,
      names: this.opts.names,
      slots: this.slots,
      drawnIndex: this.drawnIndex,
      cursor: this.cursor,
      selectable: this.selectable,
      riichiArmed: this.riichiArmed,
      phase: this.phase,
      timer: this.timerState(),
      ledger: this.ledger,
      log: this.log,
      message: this.message,
      riverRows: L.riverRows,
      claim: this.lastDiscard,
    };
  }

  render(): void {
    this.dirty = false;
    this.headerOnlyOk = false;
    const { cols, rows } = this.termSize;
    if (cols !== this.scr.cols || rows !== this.scr.rows) this.scr.resize(cols, rows);

    if (cols < term.MIN_SIZE.cols || rows < term.MIN_SIZE.rows) {
      this.scr.clear();
      const msg = "端末が小さすぎます";
      const req = `必要サイズ: ${term.MIN_SIZE.cols}x${term.MIN_SIZE.rows} (現在 ${cols}x${rows})`;
      this.scr.draw(0, 0, term.truncate(msg, cols), "1;91");
      this.scr.draw(0, 1, term.truncate(req, cols));
      this.scr.flush();
      return;
    }

    const L = layout(cols, rows);
    const c = this.ctx(L);
    this.scr.clear();
    this.frame(L);

    if (!c.obs) {
      this.scr.draw(L.x0 + 2, L.ownY, "配牌を待っています…", DIM);
      this.scr.drawLine(L.x0, L.actionY, W.actionBar(c, L.inner), L.inner);
      this.scr.flush();
      return;
    }

    // header, embedded in the top border rule
    this.scr.drawLine(L.x0 + 1, L.headerY, W.headerBar(c, L.inner - 2), L.inner - 2);

    // 対面 (relative 2), centred
    const panelW = Math.min(40, Math.floor(L.inner / 2));
    const toimen = W.seatPanel(c, 2, panelW);
    const tx = L.x0 + Math.max(0, Math.floor((L.inner - panelW) / 2));
    toimen.forEach((l, i) => this.scr.drawLine(tx, L.toimenY + i, l, panelW));

    // 上家 (3) left, 下家 (1) right, centre block between them
    const sideW = Math.min(34, Math.floor((L.inner - 16) / 2));
    W.seatPanel(c, 3, sideW).forEach((l, i) => this.scr.drawLine(L.x0, L.sideY + i, l, sideW));
    const rx = L.x0 + L.inner - sideW;
    W.seatPanel(c, 1, sideW).forEach((l, i) => this.scr.drawLine(rx, L.sideY + i, l, sideW));

    const box = W.centerBlock(c);
    const boxW = Math.max(...box.map(lineWidth), 0);
    const bx = L.x0 + Math.floor((L.inner - boxW) / 2);
    const by = L.sideY + Math.max(0, Math.floor((L.panelRows - box.length) / 2));
    box.forEach((l, i) => this.scr.drawLine(bx, by + i, l, boxW));

    // own seat
    const own = W.ownPanel(c, L.inner);
    this.scr.drawLine(L.x0, L.ownY, own[0], L.inner);
    this.scr.drawLine(L.x0, L.ownRiverY, own[1], L.inner);
    this.scr.drawLine(L.x0, L.handY, own[2], L.inner);
    this.scr.drawLine(L.x0, L.caretY, own[3], L.inner);
    this.scr.drawLine(L.x0, L.metricsY, W.metricsLine(c, L.inner), L.inner);
    this.scr.drawLine(L.x0, L.dangerY, W.dangerRow(c, L.inner), L.inner);

    // ledger + log
    W.ledgerPanel(c, L.inner).forEach((l, i) => this.scr.drawLine(L.x0, L.ledgerY + i, l, L.inner));
    const tail = this.log.slice(Math.max(0, this.log.length - L.logRows));
    tail.forEach((s, i) => this.scr.draw(L.x0 + 1, L.logY + i, term.truncate(s, L.inner - 2), DIM));

    this.scr.drawLine(L.x0, L.actionY, W.actionBar(c, L.inner), L.inner);

    if (this.overlayState) this.drawOverlay(L, c, this.overlayState);
    this.scr.flush();
    // A complete board is on screen: the next tick may repaint the clock alone.
    this.headerOnlyOk = true;
  }

  private frame(L: Layout): void {
    const bar = (y: number, l: string, m: string, r: string, label = "") => {
      this.scr.draw(0, y, l + m.repeat(Math.max(0, L.cols - 2)) + r, DIM);
      if (label) this.scr.draw(2, y, ` ${label} `, DIM);
    };
    bar(L.headerY, "┌", "─", "┐");
    bar(L.sepY, "├", "─", "┤");
    bar(L.ledgerHdrY, "├", "─", "┤", "違反台帳");
    bar(L.logHdrY, "├", "─", "┤", "記録");
    bar(L.bottomY, "└", "─", "┘");
    for (let y = 1; y < L.bottomY; y++) {
      if (y === L.sepY || y === L.ledgerHdrY || y === L.logHdrY) continue;
      this.scr.draw(0, y, "│", DIM);
      this.scr.draw(L.cols - 1, y, "│", DIM);
    }
  }

  private drawOverlay(L: Layout, c: Ctx, o: Overlay): void {
    const v = W.overlay(c, o);
    const contentW = Math.max(
      width(v.title) + 4,
      width(v.footer) + 4,
      ...v.body.map(lineWidth),
    );
    const boxW = Math.min(L.inner - 4, contentW + 4);
    const boxH = Math.min(L.rows - 4, v.body.length + 4);
    const x = L.x0 + Math.floor((L.inner - boxW) / 2);
    const y = Math.max(1, Math.floor((L.rows - boxH) / 2));

    for (let i = 0; i < boxH; i++) this.scr.fill(x, y + i, boxW, " ");
    this.scr.draw(x, y, "┌" + "─".repeat(boxW - 2) + "┐", SGR.brightCyan);
    this.scr.draw(x + 2, y, ` ${v.title} `, "1;96");
    for (let i = 1; i < boxH - 1; i++) {
      this.scr.draw(x, y + i, "│", SGR.brightCyan);
      this.scr.draw(x + boxW - 1, y + i, "│", SGR.brightCyan);
    }
    this.scr.draw(x, y + boxH - 1, "└" + "─".repeat(boxW - 2) + "┘", SGR.brightCyan);
    if (v.footer) this.scr.draw(x + 2, y + boxH - 1, ` ${v.footer} `, SGR.brightCyan);

    const rows = Math.min(v.body.length, boxH - 3);
    for (let i = 0; i < rows; i++) {
      this.scr.drawLine(x + 2, y + 1 + i, v.body[i], boxW - 4);
    }
  }

  // --- end of match --------------------------------------------------------

  /** Show the final standings and block until a key is pressed. */
  async showFinal(result: MatchResult): Promise<void> {
    // The last hand's 局結果 is still up; overwriting it would swallow the
    // result of the round that decided the match.
    await this.gate();
    this.phase = "idle";
    this.stopTicker();
    this.overlayState = {
      kind: "text",
      title: "最終結果",
      body: standingsBody(result, this.opts.cfg, this.opts.names),
      footer: "任意のキーで終了",
    };
    this.requestRender();
    return new Promise<void>((resolve) => {
      this.overlayResolve = resolve;
    });
  }
}

function fmtDelta(d: number): string {
  return d === 0 ? "±0" : d > 0 ? `+${d}` : String(d);
}

/**
 * 最終結果. The ranking is NOT recomputed here: it is read straight off
 * `score.ts::finalStandings`, the engine's own settlement, so the screen and
 * the match result cannot disagree. That matters because the dojo's headline
 * rule is not a score sort — a seat carrying any ledger entry finishes below
 * every clean seat, and equal scores break by 起家 proximity. 起家 is read from
 * the first round's dealer rather than assumed to be seat 0.
 */
function standingsBody(result: MatchResult, cfg: RuleConfig, names: string[]): Line[] {
  const east = (result.rounds[0]?.dealer ?? 0) as Seat;
  const order = finalStandings(result.scores, east, result.ledger, cfg);
  const body: Line[] = [
    [
      sp(padEnd("順位", 6), DIM),
      sp(padEnd("席", 14), DIM),
      sp(padEnd("点数", 8), DIM),
      sp(padEnd("収支", 8), DIM),
      sp("違反", DIM),
    ],
  ];
  for (const st of order) {
    body.push([
      sp(padEnd(`${st.place}位`, 6), st.place === 1 ? SGR.brightYellow : ""),
      sp(padEnd(`P${st.seat} ${names[st.seat] ?? ""}`, 14)),
      sp(padEnd(String(st.score), 8)),
      sp(padEnd(fmtDelta(st.points), 8), st.points >= 0 ? SGR.brightGreen : SGR.brightRed),
      sp(st.clean ? "—" : `${st.violations}件`, st.clean ? DIM : "1;91"),
    ]);
  }
  body.push([]);
  const vio = result.ledger.length;
  body.push([sp(`違反 ${vio}件`, vio ? SGR.yellow : DIM), sp(`   ${result.rounds.length}局`, DIM)]);
  if (order.some((s) => !s.clean)) {
    body.push([sp("※ 反則のある席は清廉な席より下に置かれる", SGR.yellow)]);
  }
  return body;
}
