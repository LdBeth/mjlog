// Replay a REAL riichi.dev wire capture through the champion chooser.
//
// The log (a tracked fixture; `< ` inbound / `> ` outbound) is
// the validation game the tsumogiri bot played, so history keeps contradicting
// what the champion would have done — which is exactly the point: the shadow
// must follow the SERVER's story without desyncing, and every champion reply
// must still be one of the server's own possible_actions (echoed verbatim; the
// transport only adds request_id/actor). `replay: true` forgets the reach
// two-step after each request, since a declaration history never honors would
// otherwise poison the next answer.

import { assert, assertEquals } from "@std/assert";
import type { MjaiEvent, RequestAction } from "../src/net/arena.ts";
import { TsumogiriChooser } from "../src/net/arena.ts";
import { ChampionChooser } from "../src/net/champion.ts";

const LOG = new URL("./fixtures/arena-validate-0827.jsonl", import.meta.url).pathname;

function inbound(): MjaiEvent[] {
  let text: string;
  try {
    text = Deno.readTextFileSync(LOG);
  } catch {
    throw new Error(
      `捕獲ログ ${LOG} がない — deno task arena --brain=tsumogiri --log=… で採り直す`,
    );
  }
  return text.split("\n").filter((l) => l.startsWith("< ")).map((l) => JSON.parse(l.slice(2)));
}

Deno.test("arena replay: the champion answers every request legally, no desync", () => {
  const events = inbound();
  const champion = new ChampionChooser({ replay: true, log: () => {} });
  const naive = new TsumogiriChooser();
  let requests = 0;
  let agreements = 0;
  try {
    for (const e of events) {
      champion.onEvent(e);
      naive.onEvent(e);
      if (e.type !== "request_action") continue;
      const req = e as RequestAction;
      requests++;
      const reply = champion.choose(req);
      // Verbatim: the reply IS one of the server's entries, every field intact.
      assert(
        req.possible_actions.includes(reply),
        `request ${req.request_id}: 応答が候補の実体でない: ${JSON.stringify(reply)}`,
      );
      if (reply === naive.choose(req)) agreements++;
    }
  } finally {
    champion.close();
  }
  assert(requests > 100, `捕獲ログが痩せている: request ${requests} 件`);
  assertEquals(champion.fallbacks, 0, "影が同期を失った");
  console.log(
    `    replay: ${requests} requests, 素朴応答との一致 ${agreements} ` +
      `(${(agreements * 100 / requests).toFixed(1)}%)`,
  );
});
