// Golden transcript test: the full render of the bundled sample must match the
// checked-in golden byte-for-byte. Catches any unintended formatting drift; when
// output changes ON PURPOSE, regenerate with:
//   deno run --allow-read --allow-write test/golden_update.ts

import { render } from "../src/core.ts";

const SAMPLE = new URL("../../1.xml", import.meta.url).pathname;
const GOLDEN = new URL("./golden/1.txt", import.meta.url);

Deno.test("golden: sample transcript is byte-identical", async () => {
  const got = await render(SAMPLE, { hands: "key" });
  const want = await Deno.readTextFile(GOLDEN);
  if (got !== want) {
    const g = got.split("\n"), w = want.split("\n");
    let i = 0;
    while (i < g.length && i < w.length && g[i] === w[i]) i++;
    throw new Error(
      `transcript differs from golden at line ${i + 1}:\n  got:  ${g[i]}\n  want: ${w[i]}\n` +
        `(if intentional, regenerate: deno run --allow-read --allow-write test/golden_update.ts)`,
    );
  }
});
