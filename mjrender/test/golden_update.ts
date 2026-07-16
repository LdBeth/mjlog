// Regenerate the golden transcript after an INTENTIONAL output change.
//   deno run --allow-read --allow-write test/golden_update.ts

import { render } from "../src/core.ts";

const SAMPLE = new URL("../../1.xml", import.meta.url).pathname;
const GOLDEN = new URL("./golden/1.txt", import.meta.url);

await Deno.mkdir(new URL("./golden/", import.meta.url), { recursive: true });
await Deno.writeTextFile(GOLDEN, await render(SAMPLE, { hands: "key" }));
console.log(`wrote ${GOLDEN.pathname}`);
