// Beat enumeration: IDs must match the transcript's anchors exactly, and each
// beat's (round, eventIndex) must replay to the state its anchor talks about.

import { enumerateBeats } from "../src/beats.ts";
import { renderGameAnnotated } from "../src/render.ts";
import { parseGame } from "../src/parse.ts";
import { replayTo } from "../src/state.ts";
import { loadXml } from "../src/load.ts";

function eq<T>(a: T, b: T, msg: string): void {
  const as = JSON.stringify(a), bs = JSON.stringify(b);
  if (as !== bs) throw new Error(`${msg}: got ${as}, want ${bs}`);
}

const SAMPLE = new URL("../../1.xml", import.meta.url).pathname;

Deno.test("beats: enumeration matches the transcript's anchors one-to-one", async () => {
  const game = parseGame(await loadXml(SAMPLE));
  const { text, beats } = renderGameAnnotated(game, { hands: "key" });

  // (^…#\d requires a real id — the format preamble mentions the literal "#N")
  const anchorLines = text.split("\n").filter((l) => /^〔解説ポイント#\d+:/.test(l));
  eq(anchorLines.length, beats.length, "one beat per anchor line");
  beats.forEach((b, i) => {
    const want = `〔解説ポイント#${b.id}: ${b.kind}｜${b.topic}〕`;
    if (anchorLines[i] !== want) {
      throw new Error(`anchor ${i} mismatch:\n  text: ${anchorLines[i]}\n  beat: ${want}`);
    }
  });
  eq(beats.map((b) => b.id), beats.map((_, i) => i + 1), "ids are 1..N in order");

  // stable across renders and equal to the dedicated enumerator
  eq(enumerateBeats(game), beats, "enumerateBeats == annotated render beats");

  // every round contributes its 配牌評価 beat; game ends with 終局総括
  const deals = beats.filter((b) => b.kind === "配牌評価");
  eq(deals.length, game.rounds.length, "one deal beat per round");
  eq(beats.at(-1)?.kind, "終局総括", "last beat is the game summary");
});

Deno.test("beats: リーチ判断 replays to a post-declaration state", async () => {
  const game = parseGame(await loadXml(SAMPLE));
  const beats = enumerateBeats(game).filter((b) => b.kind === "リーチ判断");
  if (beats.length === 0) throw new Error("sample has no riichi beats");
  for (const b of beats) {
    const st = replayTo(game, game.rounds[b.round], { eventIndex: b.eventIndex });
    if (b.seat === undefined) throw new Error("riichi beat must carry a seat");
    eq(st.riichiActive[b.seat], true, `beat #${b.id}: declarer's riichi is active`);
    eq(st.hands[b.seat].length % 3, 1, `beat #${b.id}: hand is resting (3n+1) after discard`);
    eq(st.junme, b.junme, `beat #${b.id}: junme agrees with replay`);
  }
});
