// Query core: kyoku selectors, anchor snapshots, per-kyoku rendering, and
// inline-snapshot equivalence with get_snapshot.

import {
  getSnapshot,
  listAnchors,
  loadGame,
  renderGame,
  renderKyoku,
  resolveKyoku,
} from "../src/core.ts";
import type { Game } from "../src/model.ts";

function eq<T>(a: T, b: T, msg: string): void {
  const as = JSON.stringify(a), bs = JSON.stringify(b);
  if (as !== bs) throw new Error(`${msg}: got ${as}, want ${bs}`);
}

const SAMPLE = new URL("../../1.xml", import.meta.url).pathname;
let cached: Game | undefined;
async function game(): Promise<Game> {
  return cached ??= await loadGame(SAMPLE);
}

Deno.test("query: kyoku selector forms resolve consistently", async () => {
  const g = await game();
  eq(resolveKyoku(g, "0"), [0], "round index");
  eq(resolveKyoku(g, "E1"), resolveKyoku(g, "東1"), "letter == kanji wind");
  const s3 = resolveKyoku(g, "S3");
  for (const i of s3) eq(g.rounds[i].kyoku, 4 + 2, "S3 = kyoku 6");
  let threw = false;
  try {
    resolveKyoku(g, "X9");
  } catch {
    threw = true;
  }
  eq(threw, true, "invalid selector throws");
});

Deno.test("query: riichi anchor snapshot is coherent (stick debited AND on the table)", async () => {
  const g = await game();
  const beat = listAnchors(g).find((b) => b.kind === "リーチ判断");
  if (!beat) throw new Error("no riichi beat in sample");
  const snap = getSnapshot(g, { anchor: beat.id });
  if (!snap.includes(`〔#${beat.id} リーチ判断｜`)) throw new Error("missing beat note in header");
  if (!snap.includes("リーチ(")) throw new Error("missing riichi state on the declarer's line");
  if (!snap.includes("*")) throw new Error("missing riichi-tile mark in the river");
  // the declaration stick: on the table (供託 ≥ 1) and debited from the declarer
  const kyotaku = Number(snap.match(/供託(\d+)/)?.[1]);
  if (!(kyotaku >= 1)) throw new Error(`stick not on the table: ${snap.split("\n")[0]}`);
  const startTotal = g.rounds[beat.round].startScores.reduce((a, b) => a + b, 0);
  const liveTotal = [...snap.matchAll(/ (\d+)点/g)].reduce((a, m) => a + Number(m[1]) / 100, 0);
  eq(liveTotal, startTotal - (kyotaku - g.rounds[beat.round].kyotaku) * 10, "scores account for placed sticks");
});

Deno.test("query: junme addressing replays to end of the go-around", async () => {
  const g = await game();
  const snap = getSnapshot(g, { kyoku: "0", junme: 3 });
  if (!snap.includes("3巡目")) throw new Error("wrong junme in header");
  if (!snap.includes("残り山")) throw new Error("missing wall count");
});

Deno.test("query: renderKyoku is self-contained and keeps game-global anchor ids", async () => {
  const g = await game();
  const beats = listAnchors(g);
  const target = beats.find((b) => b.round === 1); // a beat in the SECOND round
  if (!target) throw new Error("no beat in round 1");
  const text = renderKyoku(g, "1");
  if (!text.includes("■この牌譜の読み方")) throw new Error("missing preamble");
  if (!text.includes(`〔解説ポイント#${target.id}:`)) {
    throw new Error(`round-1 slice lost game-global anchor #${target.id}`);
  }
  if (text.includes("〔解説ポイント#1:")) {
    throw new Error("round-0 anchors leaked into the round-1 slice");
  }
});

Deno.test("query: inline snapshot equals get_snapshot for the same beat", async () => {
  const g = await game();
  const text = renderGame(g, { snapshots: "inline" });
  const lines = text.split("\n");
  const beat = listAnchors(g).find((b) => b.kind === "リーチ判断")!;
  const anchorIdx = lines.findIndex((l) => l.startsWith(`〔解説ポイント#${beat.id}:`));
  // the inline block sits directly above its anchor: └ on anchorIdx-1, ┌ further up
  let start = anchorIdx - 1;
  while (start >= 0 && !lines[start].startsWith("┌盤面")) start--;
  const inline = lines.slice(start, anchorIdx).join("\n");
  const direct = getSnapshot(g, { anchor: beat.id });
  // headers carry different notes (#N vs #N kind｜topic); compare the body
  eq(
    inline.split("\n").slice(1),
    direct.split("\n").slice(1),
    "inline and direct snapshot bodies match",
  );
});
