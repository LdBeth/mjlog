// renderOutline: crude whole-game outline — header/result lines sliced
// verbatim from the full render, plus a per-kyoku anchor index; no per-turn
// detail.

import { listAnchors, loadGame, renderGame, renderOutline } from "../src/core.ts";

const SAMPLE = new URL("../../1.xml", import.meta.url).pathname;

Deno.test("outline: verbatim slices, full anchor coverage, no per-turn lines", async () => {
  const game = await loadGame(SAMPLE);
  const outline = renderOutline(game);
  const lines = outline.split("\n");
  const full = renderGame(game);
  const fullLines = new Set(full.split("\n"));

  // From the first kyoku header on, every line is either a verbatim slice of
  // the full render, an anchor-index line, a terminator, or blank.
  const anchorIndex = /^〔解説ポイント#\d+: .+〕（\d+巡(・P\d)?）$/;
  const body = lines.slice(lines.findIndex((l) => l.startsWith("【")));
  if (body.length === 0) throw new Error("outline has no kyoku header");
  for (const line of body) {
    if (!line.trim() || line === "―".repeat(20) || anchorIndex.test(line)) continue;
    if (!fullLines.has(line)) throw new Error(`outline line not in full render: "${line}"`);
  }

  // every anchor of the game is listed exactly once
  for (const b of listAnchors(game)) {
    const hits = lines.filter((l) => l.includes(`〔解説ポイント#${b.id}:`)).length;
    if (hits !== 1) throw new Error(`anchor #${b.id} appears ${hits} times in outline`);
  }

  // no per-turn detail leaks into the outline body (the format legend up top
  // legitimately MENTIONS these markers while explaining the notation)
  const bodyText = body.join("\n");
  for (const bad of ["◆配牌", "◇結果時点", "◇見逃し", "┗", " ▽"]) {
    if (bodyText.includes(bad)) throw new Error(`outline should not contain "${bad}"`);
  }

  // the outline must instruct the drill-down workflow
  for (const tool of ["mj_render_kyoku", "mj_get_snapshot", "mj_add_comment"]) {
    if (!outline.includes(tool)) throw new Error(`outline note missing ${tool}`);
  }

  // and it is actually crude
  const ratio = lines.length / full.split("\n").length;
  if (ratio > 0.25) throw new Error(`outline is not crude enough: ${ratio} of full size`);
});
