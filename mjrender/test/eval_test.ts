// Eval generator: items are internally consistent with the game model and the
// transcript's own facts.

import { generateEval } from "../src/eval.ts";
import { loadGame, renderGame } from "../src/core.ts";

function eq<T>(a: T, b: T, msg: string): void {
  const as = JSON.stringify(a), bs = JSON.stringify(b);
  if (as !== bs) throw new Error(`${msg}: got ${as}, want ${bs}`);
}

const SAMPLE = new URL("../../1.xml", import.meta.url).pathname;

Deno.test("eval: generated Q/A cover the game and agree with the transcript", async () => {
  const game = await loadGame(SAMPLE);
  const items = generateEval(game);
  const byCat = new Map<string, number>();
  for (const it of items) byCat.set(it.category, (byCat.get(it.category) ?? 0) + 1);

  eq(byCat.get("start_scores"), game.rounds.length, "one score question per round");
  eq(byCat.get("first_dora"), game.rounds.length, "one dora question per round");
  if (!byCat.has("riichi_wait")) throw new Error("no riichi questions in sample");
  eq(byCat.get("final_placements"), 1, "one final-placement question");

  // spot-check: every riichi_wait answer appears verbatim as a 待ち in the transcript
  const text = renderGame(game);
  for (const it of items) {
    if (it.category !== "riichi_wait") continue;
    if (!text.includes(`待ち: ${it.answer}`)) {
      throw new Error(`riichi wait ${it.answer} (${it.kyoku}) not in transcript`);
    }
  }

  // winners match the model
  const winners = game.rounds.flatMap((r) =>
    r.results.filter((x) => x.kind === "agari").map((x) => (x as { who: number }).who)
  );
  const answered = items.filter((i) => i.category === "winner_points").map((i) =>
    Number(i.answer.match(/^P(\d)/)![1])
  );
  eq(answered, winners, "winner answers match round results");
});
