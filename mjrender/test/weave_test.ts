// weaveCommentary: the harness side of the commentary loop. The LLM only
// supplies {anchor → comment}; every fact line must come through verbatim.

import {
  listAnchors,
  listStarSites,
  loadGame,
  renderGame,
  roundLabel,
  weaveCommentary,
} from "../src/core.ts";
import type { Game } from "../src/model.ts";

const SAMPLE = new URL("../../1.xml", import.meta.url).pathname;

let cached: Game | undefined;
async function game(): Promise<Game> {
  return cached ??= await loadGame(SAMPLE);
}

const ANCHOR = /^〔解説ポイント#\d+:/;
const COMMENT = /^◆解説（/;

Deno.test("weave: comments replace their anchors, map and list forms agree", async () => {
  const g = await game();
  const first = listAnchors(g)[0];
  const text = "一行目のコメント。\n二行目。";
  const viaList = weaveCommentary(g, [{ anchor: first.id, text }]);
  const viaMap = weaveCommentary(g, { [String(first.id)]: text });
  if (viaList.text !== viaMap.text) throw new Error("list and map input forms disagree");

  const want = `◆解説（${first.kind}）: 一行目のコメント。\n  二行目。`;
  if (!viaList.text.includes(want)) throw new Error(`missing woven comment block:\n${want}`);
  if (viaList.text.includes(`〔解説ポイント#${first.id}:`)) {
    throw new Error("filled anchor placeholder still present");
  }
  if (viaList.filled.length !== 1 || viaList.filled[0] !== first.id) {
    throw new Error(`filled ids wrong: ${viaList.filled}`);
  }
  if (viaList.missing.length !== listAnchors(g).length - 1) {
    throw new Error(`missing count wrong: ${viaList.missing.length}`);
  }
});

Deno.test("weave: every fact line survives verbatim, in order", async () => {
  const g = await game();
  const comments = Object.fromEntries(listAnchors(g).map((b) => [b.id, `解説${b.id}。`]));
  const woven = weaveCommentary(g, comments);
  if (woven.missing.length !== 0) throw new Error("all anchors were supplied");

  // Compare from the first ===… separator (skips the mode-specific preamble).
  const body = (t: string) => {
    const lines = t.split("\n");
    return lines.slice(lines.findIndex((l) => l.startsWith("====")));
  };
  const facts = body(renderGame(g)).filter((l) => !ANCHOR.test(l));
  const wovenFacts = body(woven.text).filter((l) => !COMMENT.test(l));
  if (facts.length !== wovenFacts.length) {
    throw new Error(`fact line count drifted: ${facts.length} → ${wovenFacts.length}`);
  }
  for (let i = 0; i < facts.length; i++) {
    if (facts[i] !== wovenFacts[i]) {
      throw new Error(`fact line changed at ${i}:\n  was:  ${facts[i]}\n  now:  ${wovenFacts[i]}`);
    }
  }
});

Deno.test("weave: missing anchors keep placeholders by default, strip removes them", async () => {
  const g = await game();
  const first = listAnchors(g)[0];
  // The reader preamble mentions 〔解説ポイント…〕, so count only real anchor LINES.
  const anchorLines = (t: string) => t.split("\n").filter((l) => ANCHOR.test(l)).length;
  const kept = weaveCommentary(g, [{ anchor: first.id, text: "x" }]);
  if (anchorLines(kept.text) !== listAnchors(g).length - 1) {
    throw new Error("keep should retain every unfilled placeholder");
  }
  const stripped = weaveCommentary(g, [{ anchor: first.id, text: "x" }], { missing: "strip" });
  if (anchorLines(stripped.text) !== 0) throw new Error("strip left placeholders behind");
});

Deno.test("weave: final document carries the reader preamble, not the fill instructions", async () => {
  const g = await game();
  const woven = weaveCommentary(g, [{ anchor: 1, text: "x" }]);
  if (woven.text.includes("解説者への指示")) throw new Error("fill instructions leaked");
  if (!woven.text.includes("織り込んだ観戦記")) throw new Error("reader preamble missing");
  if (!woven.text.includes("・牌表記:")) throw new Error("notation legend missing");
});

Deno.test("weave: ★ notes land after their line's ┗ block, addressed by kyoku+junme+seat", async () => {
  const g = await game();
  const site = listStarSites(g)[0];
  const woven = weaveCommentary(g, {
    notes: [{
      kyoku: String(site.round),
      junme: site.junme,
      seat: site.seat,
      text: "一言テスト。",
    }],
  });
  if (woven.notesApplied !== 1) throw new Error("note not counted");
  const lines = woven.text.split("\n");
  const at = lines.findIndex((l) => l === "  ◆一言: 一言テスト。");
  if (at < 0) throw new Error("note line missing");
  // The note must directly follow a ★ line or its ┗ continuation block.
  let j = at - 1;
  while (j >= 0 && lines[j].startsWith("  ┗")) j--;
  if (!lines[j].includes("★")) {
    throw new Error(`note not attached to a ★ line:\n${lines.slice(j, at + 1).join("\n")}`);
  }
});

Deno.test("weave: a ★ note for a position without a ★ line is rejected with the round's sites", async () => {
  const g = await game();
  const label = roundLabel(g, 0);
  try {
    weaveCommentary(g, { notes: [{ kyoku: "0", junme: 99, seat: 0, text: "x" }] });
    throw new Error("expected an error");
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (!msg.includes(label) || !msg.includes("★")) {
      throw new Error(`unhelpful error: ${msg}`);
    }
  }
});

Deno.test("weave: bad input is rejected", async () => {
  const g = await game();
  const n = listAnchors(g).length;
  const cases: Array<[string, () => void]> = [
    ["unknown id", () => weaveCommentary(g, [{ anchor: n + 1, text: "x" }])],
    ["zero id", () => weaveCommentary(g, [{ anchor: 0, text: "x" }])],
    ["empty text", () => weaveCommentary(g, [{ anchor: 1, text: "  " }])],
    ["duplicate", () => weaveCommentary(g, [{ anchor: 1, text: "a" }, { anchor: 1, text: "b" }])],
    ["empty input", () => weaveCommentary(g, {})],
  ];
  for (const [name, fn] of cases) {
    let threw = false;
    try {
      fn();
    } catch {
      threw = true;
    }
    if (!threw) throw new Error(`${name}: expected an error`);
  }
});
