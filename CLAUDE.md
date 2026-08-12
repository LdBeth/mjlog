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
notes addressed by kyoku+junme+seat) into an on-disk draft, and
`weave`/`mj_weave_commentary`
splices the accumulated draft into a re-rendered transcript written to a
file — the model never copies fact lines and the woven document never passes
through its context.

Run: `cd mjrender && deno task render ../1.mjlog` (file args also accept
tenhou.net replay/log URLs); CLI subcommands `outline`/`kyoku`/`anchors`/
`snapshot`/`facts`/`weave` (the CLI weave takes a one-shot comments JSON);
the stdio MCP server (v0.7.0) runs from source (`deno task mcp`) or from the
bundle (`deno bundle -o mcp.mjs src/mcp.ts && deno run --allow-read
--allow-write --allow-env=HOME --allow-net=tenhou.net mcp.mjs`, the
distribution path) — **stateless** (2026-07-28 MCP spec: no server-held
session): all 12 tools take `log` (local .mjlog/.xml path or tenhou.net URL)
as their first param — that value IS the handle; there is no open/close, no
cursor, nothing to restore. A parsed-game cache keyed by path+mtime is a
transparent optimization, not protocol state. Nothing is gated: any round is
renderable/commentable/replaceable at any time; ★ notes take a kyoku selector
per note (empty text deletes). Pacing (one kyoku per reply: mj_outline once to
orient → per reply mj_render_kyoku → mj_get_snapshot at riichi/tenpai →
mj_add_comment/mj_add_note → mj_draft_status) is ADVISORY, carried by tool
descriptions, the outline's ■アウトライン表示 note, and mj_draft_status hints
(incl. a ★-coverage HINT). Drafts persist on DISK at
`$HOME/.mjrender/drafts/<key>.json`, key = sha256 of the log's decoded XML
(so gzipped .mjlog / plain .xml / renamed copy / tenhou.net URL of one game
share a draft); every mj_add_comment/mj_add_note writes immediately via
tmp+rename, so restarts need no recovery step. Draft-touching handlers are
FIFO-serialized in-process; two server processes on one log are last-write-wins
at whole-file granularity (accepted for a single-user desktop tool). The draft
file's shape (`{version, log, savedAt, anchors:[{anchor,text}],
notes:[{kyoku,junme,seat,text}]}`) is valid CLI weave input verbatim
(`deno task render weave ~/.mjrender/drafts/<key>.json --out x.txt ../1.xml`).
Version notes: 0.8.0 extends the danger assessor with furo (open-hand) threat
reading — `P#副露N` evidence with 染め手模様/トイトイ模様/役牌副露/ドラ(副露内)
reads, activated at 2 open melds or 1 yakuhai triplet, damped one level while
tenpai is only suspected — and wait-shape enumeration: a `当たり形:` note lists
which of リャンメン/カンチャン/ペンチャン/シャンポン/タンキ can still hit the
tile after suji/kabe/visible-count kills (なし ⇒ provably safe, level 安全);
new 押し引き anchors fire vs open hands, so anchor ids SHIFT vs 0.7.0 drafts
(no tool-signature changes). 0.7.0 is the stateless redesign — mj_open_log/mj_next_kyoku/
mj_restore_state removed, mj_render_game → mj_outline (now also reports draft
coverage and always emits the notation legend), mj_add_note gained a per-note
kyoku, mj_clear_draft added, drafts moved to disk; NO anchor renumbering vs
0.6.0. mj_get_final_standings was removed in 0.5.0 (outline's ◆終局 covers it;
`finalStandings` stays in core for CLI/eval); 中間総括 insertion renumbers
anchors vs 0.4.x.
`deno task eval` emits ground-truth Q/A JSONL; `deno task bundle` builds the
`mjrender.mcpb` Claude Desktop extension (regenerating `mcp.mjs`, the bundled
compile input, from `src/mcp.ts` first).
Tests: `deno task test` (golden transcript test — regenerate deliberately with
`test/golden_update.ts` after output changes). See `mjrender/README.md`. Uses
Deno, not Node/npm.

**MCP SDK v2** (`src/mcp.ts`): `@modelcontextprotocol/server` 2.0.0 + zod 4;
entry is `serveStdio(buildServer)`, which serves the new protocol era and
still answers legacy 2025-era `initialize` clients (Claude Desktop) from the
same factory. Its export map carries both types and runtime entries, so plain
imports typecheck under `deno check` AND run — `deno run src/mcp.ts` /
`deno task mcp` work directly. The bundle (`deno bundle -o mcp.mjs
src/mcp.ts`) remains the `deno compile` input for the `.mcpb`.

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
