// Terminal primitives: alt screen, raw mode, size, signals, and display width.
//
// `leave()` is the single most important function in the TUI: a crash that
// leaves the terminal in raw mode with a hidden cursor and the alt screen still
// active is unrecoverable for a casual user. It is idempotent and is wired to
// normal exit, signals, and `unload`.

const encoder = new TextEncoder();

/** Raw write to stdout. Synchronous on purpose — frame output must not interleave. */
export function out(s: string): void {
  if (s.length === 0) return;
  Deno.stdout.writeSync(encoder.encode(s));
}

// ---------------------------------------------------------------------------
// East-Asian display width
// ---------------------------------------------------------------------------

/**
 * Code point ranges that occupy two terminal columns (EAW Wide / Fullwidth).
 * Only Wide and Fullwidth are counted as 2; **Ambiguous is deliberately 1**,
 * which matches the default configuration of Terminal.app, iTerm2 and every
 * xterm derivative. That is why the tile tables in `glyph.ts` use ㈠ (U+3220,
 * Wide) rather than ① (U+2460, Ambiguous) — see the note there.
 */
const WIDE: ReadonlyArray<readonly [number, number]> = [
  [0x1100, 0x115f],
  [0x2e80, 0x303e], // CJK radicals … ideographic space; 0x303F is narrow
  [0x3041, 0x33ff], // kana, CJK symbols, ㈠..㈨ (0x3220..0x3228), ㌀..
  [0x3400, 0x4dbf],
  [0x4e00, 0x9fff], // 一..九, 東南西北白發中
  [0xa000, 0xa4cf],
  [0xa960, 0xa97f],
  [0xac00, 0xd7a3],
  [0xf900, 0xfaff],
  [0xfe10, 0xfe19],
  [0xfe30, 0xfe6f],
  [0xff00, 0xff60], // fullwidth forms: １..９ (0xff11..0xff19)
  [0xffe0, 0xffe6],
  [0x16fe0, 0x16fe4],
  [0x17000, 0x18aff],
  [0x1b000, 0x1b2ff],
  [0x1f200, 0x1f251],
  [0x1f300, 0x1f64f],
  [0x1f680, 0x1f6ff],
  [0x1f900, 0x1f9ff],
  [0x20000, 0x2fffd],
  [0x30000, 0x3fffd],
];

/** Combining and other zero-advance code points. */
const ZERO: ReadonlyArray<readonly [number, number]> = [
  [0x0300, 0x036f],
  [0x0483, 0x0489],
  [0x0591, 0x05bd],
  [0x0610, 0x061a],
  [0x064b, 0x065f],
  [0x0670, 0x0670],
  [0x1ab0, 0x1aff],
  [0x1dc0, 0x1dff],
  [0x200b, 0x200f],
  [0x20d0, 0x20f0],
  [0xfe00, 0xfe0f], // variation selectors
  [0xfe20, 0xfe2f],
  [0xe0100, 0xe01ef],
];

function inRanges(cp: number, ranges: ReadonlyArray<readonly [number, number]>): boolean {
  let lo = 0;
  let hi = ranges.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const [a, b] = ranges[mid];
    if (cp < a) hi = mid - 1;
    else if (cp > b) lo = mid + 1;
    else return true;
  }
  return false;
}

/** Columns occupied by one code point: 0 (combining/control), 1, or 2. */
export function charWidth(cp: number): 0 | 1 | 2 {
  if (cp === 0) return 0;
  if (cp < 0x20 || (cp >= 0x7f && cp < 0xa0)) return 0; // controls
  if (inRanges(cp, ZERO)) return 0;
  if (inRanges(cp, WIDE)) return 2;
  return 1;
}

/** Columns occupied by a string. East-Asian aware; the screen buffer's ruler. */
export function width(s: string): number {
  let n = 0;
  for (const ch of s) n += charWidth(ch.codePointAt(0)!);
  return n;
}

/** Truncate `s` to at most `max` columns (never splitting a wide glyph). */
export function truncate(s: string, max: number): string {
  if (max <= 0) return "";
  let n = 0;
  let outStr = "";
  for (const ch of s) {
    const w = charWidth(ch.codePointAt(0)!);
    if (n + w > max) break;
    outStr += ch;
    n += w;
  }
  return outStr;
}

/** Pad `s` on the right to exactly `n` columns (truncating if too wide). */
export function padEnd(s: string, n: number): string {
  const t = truncate(s, n);
  return t + " ".repeat(Math.max(0, n - width(t)));
}

// ---------------------------------------------------------------------------
// Screen mode
// ---------------------------------------------------------------------------

export const ALT_ON = "\x1b[?1049h";
export const ALT_OFF = "\x1b[?1049l";
export const CURSOR_HIDE = "\x1b[?25l";
export const CURSOR_SHOW = "\x1b[?25h";

export interface Size {
  cols: number;
  rows: number;
}

export const MIN_SIZE: Size = { cols: 80, rows: 24 };

export function size(): Size {
  try {
    const { columns, rows } = Deno.consoleSize();
    return { cols: columns, rows };
  } catch {
    return { cols: 100, rows: 32 }; // not a tty (piped output, tests)
  }
}

export function isTty(): boolean {
  try {
    return Deno.stdin.isTerminal() && Deno.stdout.isTerminal();
  } catch {
    return false;
  }
}

let active = false;
let guardsInstalled = false;

export function enter(): void {
  if (active) return;
  active = true;
  installExitGuards();
  if (isTty()) {
    try {
      Deno.stdin.setRaw(true);
    } catch {
      // A non-raw terminal still renders; input just arrives line-buffered.
    }
  }
  out(ALT_ON + CURSOR_HIDE + "\x1b[2J\x1b[H");
}

/** Restore the terminal. Safe to call any number of times, from any context. */
export function leave(): void {
  if (!active) return;
  active = false;
  try {
    if (isTty()) Deno.stdin.setRaw(false);
  } catch {
    // ignore — we are probably already unwinding
  }
  out("\x1b[0m" + CURSOR_SHOW + ALT_OFF);
}

/**
 * Wire `leave()` to every way this process can end. Registered once; the signal
 * handlers are intentionally never removed, because the window where they are
 * absent is exactly the window where a crash strands the terminal.
 */
export function installExitGuards(): void {
  if (guardsInstalled) return;
  guardsInstalled = true;
  globalThis.addEventListener("unload", () => leave());
  globalThis.addEventListener("error", () => leave());
  globalThis.addEventListener("unhandledrejection", () => leave());
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"] as const) {
    try {
      Deno.addSignalListener(sig, () => {
        leave();
        Deno.exit(sig === "SIGINT" ? 130 : 143);
      });
    } catch {
      // SIGHUP is not available everywhere; the others are.
    }
  }
}

/** Subscribe to terminal resizes. Returns an unsubscribe function. */
export function onResize(cb: () => void): () => void {
  try {
    Deno.addSignalListener("SIGWINCH", cb);
    return () => {
      try {
        Deno.removeSignalListener("SIGWINCH", cb);
      } catch { /* already gone */ }
    };
  } catch {
    return () => {};
  }
}

// ---------------------------------------------------------------------------
// SGR helpers (parameter strings, without the CSI prefix — see screen.ts)
// ---------------------------------------------------------------------------

export const SGR = {
  bold: "1",
  dim: "2",
  reverse: "7",
  green: "32",
  yellow: "33",
  cyan: "36",
  gray: "90",
  brightRed: "91",
  brightGreen: "92",
  brightYellow: "93",
  brightCyan: "96",
  brightWhite: "97",
  onGray: "100",
} as const;
