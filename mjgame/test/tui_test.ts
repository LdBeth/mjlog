// TUI unit tests. The interactive loop cannot run from a non-tty, so the parts
// that *can* silently break the board — glyph widths, frame diffing, and escape
// decoding — are tested directly.

import { assert, assertEquals } from "@std/assert";
import { charWidth, padEnd, truncate, width } from "../src/tui/term.ts";
import { Screen, sp } from "../src/tui/screen.ts";
import { decodeKeys, isCtrlC } from "../src/tui/input.ts";
import { MARK, tileText, typeText } from "../src/tui/glyph.ts";
import { AKA_5P } from "../src/tiles.ts";

const AKA = new Set(AKA_5P);

Deno.test("width: every kanji-mode tile glyph is exactly 2 columns", () => {
  for (let type = 0; type < 34; type++) {
    const g = typeText(type, "kanji");
    assertEquals(width(g), 2, `type ${type} (${g}) is ${width(g)} columns`);
  }
});

Deno.test("width: pin glyphs are the Wide ㈠..㈨, not the Ambiguous ①..⑨", () => {
  for (let r = 1; r <= 9; r++) {
    const g = typeText(8 + r, "kanji"); // pin types 9..17
    assertEquals(
      g.codePointAt(0),
      0x3220 + r - 1,
      `pin ${r} should be U+${
        (0x3220 + r - 1)
          .toString(16)
      }`,
    );
    assertEquals(width(g), 2);
  }
  // The glyph we deliberately avoid really is ambiguous-width in our table.
  assertEquals(width("①"), 1);
});

Deno.test("width: ascii mode is also exactly 2 columns per tile", () => {
  for (let type = 0; type < 34; type++) {
    const g = typeText(type, "ascii");
    assertEquals(width(g), 2, `type ${type} (${g}) is ${width(g)} columns`);
  }
  // Red fives read `0p` and stay 2 columns.
  assertEquals(tileText(52, { mode: "ascii", aka: AKA }), "0p");
  assertEquals(width(tileText(52, { mode: "ascii", aka: AKA })), 2);
  // Aka in kanji mode is colour-only — never a 赤 prefix, which would be 4 cols.
  assertEquals(width(tileText(52, { mode: "kanji", aka: AKA })), 2);
});

Deno.test("width: river markers and misc glyphs", () => {
  for (const m of Object.values(MARK)) assertEquals(width(m.ch), 1, `marker ${m.ch}`);
  assertEquals(width("abc"), 3);
  assertEquals(width(""), 0);
  assertEquals(width("東1局"), 5); // 2 + 1 + 2
  assertEquals(charWidth("　".codePointAt(0)!), 2); // ideographic space
  assertEquals(charWidth(0x0301), 0); // combining acute
  assertEquals(charWidth(0x2500), 1); // box drawing: ambiguous → 1
});

Deno.test("truncate / padEnd never split a wide glyph", () => {
  assertEquals(truncate("一二三", 3), "一"); // 2 fits, 4 does not
  assertEquals(width(padEnd("一二", 7)), 7);
  assertEquals(padEnd("abc", 5), "abc  ");
});

// ---------------------------------------------------------------------------

function testScreen(cols = 20, rows = 4) {
  return new Screen(cols, rows, { write: () => {} });
}

Deno.test("Screen: drawing the same frame twice emits nothing the second time", () => {
  const s = testScreen();
  s.clear();
  s.draw(0, 0, "一二三", "33");
  s.draw(0, 1, "hello");
  const first = s.flush();
  assert(first.length > 0, "first frame must paint");

  s.clear();
  s.draw(0, 0, "一二三", "33");
  s.draw(0, 1, "hello");
  assertEquals(s.flush(), "", "identical frame must be a no-op");
});

Deno.test("Screen: only the changed cell is emitted", () => {
  const s = testScreen();
  s.clear();
  s.draw(0, 0, "abcd");
  s.flush();
  s.clear();
  s.draw(0, 0, "abXd");
  const diff = s.flush();
  assert(diff.includes("X"), diff);
  assert(!diff.includes("a"), `should not repaint unchanged cells: ${JSON.stringify(diff)}`);
  // one cursor move + one SGR + the char + the trailing reset
  assert(diff.length < 20, `diff too large: ${JSON.stringify(diff)}`);
});

Deno.test("Screen: a wide glyph occupies two columns and blanks its partner", () => {
  const s = testScreen();
  s.clear();
  s.draw(0, 0, "㈤東");
  assertEquals(s.rowText(0).trimEnd(), "㈤東");
  s.flush();
  // A flush swaps buffers, so a partial frame reopens on what is on screen.
  s.retain();
  // Overwrite the second column of ㈤: the first column must be blanked too.
  s.draw(1, 0, "x");
  assertEquals(s.rowText(0).trimEnd(), " x東");
});

Deno.test("Screen: full repaint after resize", () => {
  const s = testScreen();
  s.clear();
  s.draw(0, 0, "hi");
  s.flush();
  s.resize(30, 6);
  s.clear();
  s.draw(0, 0, "hi");
  assert(s.flush().length > 0, "a resize must force a repaint");
});

// ---------------------------------------------------------------------------

Deno.test("input: arrow keys", () => {
  assertEquals(decodeKeys("\x1b[A").map((k) => k.name), ["up"]);
  assertEquals(decodeKeys("\x1b[B").map((k) => k.name), ["down"]);
  assertEquals(decodeKeys("\x1b[C").map((k) => k.name), ["right"]);
  assertEquals(decodeKeys("\x1b[D").map((k) => k.name), ["left"]);
  // application cursor mode
  assertEquals(decodeKeys("\x1bOD").map((k) => k.name), ["left"]);
  // a burst of keys in one chunk
  assertEquals(
    decodeKeys("\x1b[C\x1b[Cr").map((k) => k.name),
    ["right", "right", "r"],
  );
});

Deno.test("input: Ctrl-C and other control keys", () => {
  const [c] = decodeKeys("\x03");
  assertEquals(c.name, "c");
  assertEquals(c.ctrl, true);
  assert(isCtrlC(c));

  assertEquals(decodeKeys("\r").map((k) => k.name), ["enter"]);
  assertEquals(decodeKeys("\n").map((k) => k.name), ["enter"]);
  assertEquals(decodeKeys(" ").map((k) => k.name), ["space"]);
  assertEquals(decodeKeys("\x7f").map((k) => k.name), ["backspace"]);
  assertEquals(decodeKeys("\x1b").map((k) => k.name), ["escape"]);
  assertEquals(decodeKeys("\t").map((k) => k.name), ["tab"]);
});

Deno.test("input: plain and alt characters", () => {
  assertEquals(decodeKeys("q").map((k) => k.name), ["q"]);
  assertEquals(decodeKeys("1234567890").map((k) => k.name).join(""), "1234567890");
  const [alt] = decodeKeys("\x1bx");
  assertEquals(alt.name, "x");
  assertEquals(alt.alt, true);
  // modified arrows report ctrl
  const [ctrlRight] = decodeKeys("\x1b[1;5C");
  assertEquals(ctrlRight.name, "right");
  assertEquals(ctrlRight.ctrl, true);
});

// ---------------------------------------------------------------------------

Deno.test("Screen: styled spans keep their own SGR and column budget", () => {
  const s = testScreen(12, 1);
  s.clear();
  s.drawLine(0, 0, [sp("一", "33"), sp("㈤", "36"), sp("１", "32")]);
  assertEquals(s.rowText(0).trimEnd(), "一㈤１");
  // clipping honours display width, not string length
  s.clear();
  s.drawLine(0, 0, [sp("一二三四五六七")], 5);
  assertEquals(width(s.rowText(0).trimEnd()), 4);
});
