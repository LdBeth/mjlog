# CLAUDE.md

## Project Overview

A macOS tool to verify the integrity of Tenhou online mahjong game logs against their cryptographic seed. It confirms that the tile draws in a replay match what the PRNG would generate from the recorded seed, proving the game was not manipulated.

**References:**
- https://tenhou.net/stat/rand/
- https://81100118.github.io/2021/01/01/天凤牌山生成算法及其验证/

## Build

```sh
sh build.sh
```

This compiles with `clang++` using C++17, Objective-C ARC, and `-O3 -flto`. The output binary is `./out`.

## Usage

```sh
./out [-v] [-h] [-s <seat>] <mjlog.xml>
```

- `-v`: verbose (prints PRNG state, raw tile indices)
- `-h`: compute SHA-512 hashes of seed for all 24 seat permutations
- `-s <seat>`: compute SHA-512 hash for a specific seat string

Mjlog files from Tenhou are gzipped XML. The tool decompresses them automatically.

## Architecture

The codebase is three Objective-C++ source files:

- **`mt19937ar.cc`** / **`mt19937ar.h`**: Mersenne Twister PRNG (`_MTRND` class). Generates the raw random numbers used to shuffle tiles.

- **`mjlog.cc`** / **`mjlog.h`**: XML log parser and data model.
  - `MjLog`: read-only model holding parsed game data (seed string, dice rolls per round, drawn tiles per round, dead wall tiles).
  - `MjLogCtrl`: mutable subclass built by `MjLogParser` as it parses.
  - `MjLogParser` (`NSXMLParserDelegate`): walks the XML elements (`SHUFFLE`, `INIT`, `AGARI`/`RYUUKYOKU`, `DORA`, tile-draw elements `T`/`U`/`V`/`W`) to reconstruct the draw order and dead wall contents.

- **`tenhou.cc`**: verification logic and `main`.
  - `setup_seed`: decodes the base64 seed from the log and initializes the MT PRNG.
  - `checkMlogRounds`: for each round, generates 137 random values, hashes them via SHA-512 (two 512-bit blocks → one shuffle array), performs a Fisher-Yates shuffle on 136 tiles, then compares dices and draw order against what the log recorded.

## mjrender (`mjrender/`)

A **separate Deno/TypeScript tool** (not part of the C++ build): a
deterministic game-state oracle over a mjlog. It never calls an LLM; an LLM
consumes it three ways: (1) an **LLM-ready Japanese commentary transcript**
(play-by-play with reconstructed hands, calls, riichi, wins/scores, computed
metrics — shanten/ukeire/waits/dora/danger evidence — discard comparisons,
end-of-hand ground truth, and `〔解説ポイント#N: 種別｜…〕` anchors; 副露判断
anchors fire at an early 2nd meld (≤6巡) or the 3rd meld with a deterministic
`┗ 役読み:` yaku outlook; a 中間総括 anchor with a `== 南入 ==` standings block
fires at each wind boundary),
(2) **snapshot recall**: `mj_get_snapshot` reproduces the full board (rivers with
tsumogiri/riichi marks, melds, live scores/placements, hands) at any anchor
`#N` or kyoku+junme, and (3) **incremental commentary weaving**: the LLM saves
anchor comments in batches of one or more per call (plus optional ★-line
notes addressed by kyoku+junme+seat) into a server-side draft, and
`weave`/`mj_weave_commentary`
splices the accumulated draft into a re-rendered transcript written to a
file — the model never copies fact lines and the woven document never passes
through its context.

Run: `cd mjrender && deno task render ../1.mjlog` (file args also accept
tenhou.net replay/log URLs); CLI subcommands `outline`/`kyoku`/`anchors`/
`snapshot`/`facts`/`weave` (the CLI weave takes a one-shot comments JSON);
the stdio MCP server runs from the bundle
(`deno bundle -o mcp.mjs src/mcp.ts && deno run --allow-read --allow-write
--allow-env=HOME --allow-net=tenhou.net mcp.mjs`) — **stateful and paced**:
`mj_open_log` is the only tool taking a path, parsing the log into the session
(reply carries the focus line + notation legend, once per process); the rest
operate on the opened log and error until one is opened. A **focus cursor**
hard-gates per-turn detail to the current kyoku (renders/snapshots/facts/draft
writes; past rounds stay readable, comments there replace-only; ★ notes take
no kyoku arg — they address the focus kyoku, which stays notable after
mj_next_kyoku until the new focus is rendered; empty note text deletes); the
gate is a pacing device, not a spoiler
shield: mj_render_game's outline (headers, condensed results, anchor index —
no per-turn lines) stays ungated for orientation. `mj_next_kyoku` advances the
focus only when every focus-kyoku anchor is commented (wind boundaries demand
the 中間総括 standings review first), replies with a ★-note hint when notes
are sparse and instructs the LLM to END ITS TURN — one kyoku per chat turn.
Server state is in-memory only (no autosave/disk); after an MCP restart the LLM
reopens the log with mj_open_log and bulk-restores its whole draft (comments +
★ notes + focus, kyoku-addressed) from its own context via `mj_restore_state`
(wholesale replace, atomic, idempotent — comments beyond focus rejected, past
gaps warned). mj_get_final_standings was removed in 0.5.0 (outline's ◆終局
covers it; `finalStandings` stays in core for CLI/eval); 中間総括 insertion
renumbers anchors vs 0.4.x; `mj_restore_state` added in 0.6.0.
`deno task eval` emits ground-truth Q/A JSONL; `deno task bundle` builds the
`mjrender.mcpb` Claude Desktop extension (regenerating `mcp.mjs`, the bundled
compile input, from `src/mcp.ts` first).
Tests: `deno task test` (golden transcript test — regenerate deliberately with
`test/golden_update.ts` after output changes). See `mjrender/README.md`. Uses
Deno, not Node/npm.

**MCP SDK import extensions** (`src/mcp.ts`): the
`@modelcontextprotocol/sdk/server/*` imports are deliberately extensionless —
only that form makes `deno check` resolve the SDK's real types (with `.js`
the SDK silently typechecks as `any`). The trade-off: the extensionless form
does not resolve at runtime (the SDK's npm export map only has `.js`
entries), so `deno run src/mcp.ts` fails at import — there is no `mcp` task;
always run the server via the bundle (`deno bundle -o mcp.mjs src/mcp.ts`,
which resolves either form; the MCP test also bundles fresh before
spawning).

## Maude Specification (`mahjong.maude`)

A separate formal model (Maude, not C++) of mahjong hand classification —
independent of the verifier. Defines tile/hand sorts and rewrite rules for
winning decompositions, multi-tile waits (tenpai), chiitoitsu, and kokushi musou.
Requires the `maude` binary (separate from the clang++ build).

```sh
maude mahjong.maude        # then run queries interactively
```

Hands use shorthand `< 1 1 1 2 3 > M` (→ five M tiles) and honors written
directly (`haku haku chun`). Example queries are in the file footer:
- `search < … > M =>* W:Win .`        — find a winning decomposition
- `search < … > M =>* Wait(T:Tiles) .` — find listening (tenpai) tiles

## Patterns & Conventions

- **NSString → std::cout**: Use `[nsstring UTF8String]` to convert NSString to C string for C++ output.
- **Property redeclaration in subclass**: When redeclaring parent's property as `readwrite` in subclass, add explicit `@synthesize propertyName;` to `@implementation` to avoid warnings.
- **XML attribute URL encoding**: Player names and other data from XML attributes are percent-encoded; decode with `stringByRemovingPercentEncoding`.
- **Use `auto`**: Use type inference to reduce clutters.
- **Build verification**: Run `sh build.sh` after changes — expect zero warnings with current `-O3 -flto` configuration.

## Dead Wall Layout (yama indices 0–13)

- `yama[0..3]`: Rinshan (kan draw) tiles — order within `ord[] = {1,0,3,2}`
- `yama[4,6,8,10,12]`: Ura-dora indicators (parsed from `AGARI doraHaiUra`)
- `yama[5,7,9,11,13]`: Dora indicators (first from `INIT seed[5]`, rest from `DORA` elements)

`AGARI doraHaiUra` uses space- or comma-separated integers (unlike INIT's comma-only `seed`).
Multiple `AGARI` elements in one round (double/triple ron) — only the first should trigger `endRound`.
`allRounds[nKyoku]` contains 52 initial deal tiles (in deal order) followed by subsequent live wall draws. Rinshan tiles are excluded (they go to `deadWalls`).
`DORA` element sets a `kong` flag so the *next* T/U/V/W draw is recorded as rinshan, not a normal draw.

## TODO: Known Limitations

- [ ] 3-player (sanma) mahjong support (would use 108 tiles instead of 136).
- [ ] Log format version hardcoded to `2.3` — make configurable.
