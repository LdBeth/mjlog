// Keyboard decoding: raw bytes → `{ name, ctrl, alt }` events.
//
// `decodeKeys` is a pure function over a decoded chunk so the escape-sequence
// table is testable without a tty. Partial sequences at a chunk boundary are
// rare enough (a terminal writes an escape sequence in one write) that we
// decode greedily and treat a truncated tail as a literal Escape.

export interface KeyEvent {
  /** "up" | "down" | "left" | "right" | "enter" | "escape" | "space" |
   *  "backspace" | "tab" | "home" | "end" | "delete" | "pageup" | "pagedown" |
   *  or the character itself for printable keys. */
  name: string;
  ctrl: boolean;
  alt: boolean;
  raw: string;
}

const key = (name: string, raw: string, ctrl = false, alt = false): KeyEvent => ({
  name,
  ctrl,
  alt,
  raw,
});

const CSI_FINAL: Record<string, string> = {
  A: "up",
  B: "down",
  C: "right",
  D: "left",
  H: "home",
  F: "end",
};

const CSI_TILDE: Record<string, string> = {
  "1": "home",
  "2": "insert",
  "3": "delete",
  "4": "end",
  "5": "pageup",
  "6": "pagedown",
};

/** Decode one chunk of terminal input into zero or more key events. */
export function decodeKeys(chunk: string): KeyEvent[] {
  const evs: KeyEvent[] = [];
  let i = 0;
  while (i < chunk.length) {
    const c = chunk[i];

    if (c === "\x1b") {
      const next = chunk[i + 1];

      // CSI: ESC [ params final
      if (next === "[") {
        let j = i + 2;
        let params = "";
        while (j < chunk.length && !/[A-Za-z~]/.test(chunk[j])) params += chunk[j++];
        if (j < chunk.length) {
          const final = chunk[j];
          const raw = chunk.slice(i, j + 1);
          // Modifier form is `1;5A` (5 = ctrl, 3 = alt).
          const mod = Number(params.split(";")[1] ?? "1") - 1;
          const ctrl = (mod & 4) !== 0;
          const alt = (mod & 2) !== 0;
          if (final === "~") {
            const name = CSI_TILDE[params.split(";")[0]] ?? "unknown";
            evs.push(key(name, raw, ctrl, alt));
          } else {
            evs.push(key(CSI_FINAL[final] ?? "unknown", raw, ctrl, alt));
          }
          i = j + 1;
          continue;
        }
        evs.push(key("escape", chunk.slice(i)));
        break;
      }

      // SS3: ESC O A  (application cursor keys)
      if (next === "O" && i + 2 < chunk.length) {
        const final = chunk[i + 2];
        evs.push(key(CSI_FINAL[final] ?? "unknown", chunk.slice(i, i + 3)));
        i += 3;
        continue;
      }

      // ESC <char> = Alt-<char>
      if (next !== undefined) {
        const inner = decodeKeys(next);
        const ev = inner[0] ?? key(next, next);
        evs.push({ ...ev, alt: true, raw: "\x1b" + ev.raw });
        i += 1 + ev.raw.length;
        continue;
      }

      evs.push(key("escape", c));
      i += 1;
      continue;
    }

    if (c === "\r" || c === "\n") {
      evs.push(key("enter", c));
      i++;
      continue;
    }
    if (c === "\t") {
      evs.push(key("tab", c));
      i++;
      continue;
    }
    if (c === "\x7f" || c === "\b") {
      evs.push(key("backspace", c));
      i++;
      continue;
    }
    if (c === " ") {
      evs.push(key("space", c));
      i++;
      continue;
    }

    const code = c.charCodeAt(0);
    if (code < 0x20) {
      // Ctrl-A..Ctrl-Z map to 0x01..0x1a.
      evs.push(key(String.fromCharCode(code + 96), c, true));
      i++;
      continue;
    }

    // Printable: take the whole code point (surrogate pairs stay intact).
    const cp = chunk.codePointAt(i)!;
    const ch = String.fromCodePoint(cp);
    evs.push(key(ch, ch));
    i += ch.length;
  }
  return evs;
}

export function isCtrlC(e: KeyEvent): boolean {
  return e.ctrl && e.name === "c";
}

/** Stream key events from stdin. Ends when stdin closes. */
export async function* readKeys(): AsyncGenerator<KeyEvent> {
  const dec = new TextDecoder();
  for await (const chunk of Deno.stdin.readable) {
    for (const e of decodeKeys(dec.decode(chunk, { stream: true }))) yield e;
  }
}
