# mjrender

Render a Tenhou **mjlog** into an **LLM-ready Japanese commentary transcript**, and serve
**recallable board snapshots** for any position in the game.

mjrender is a deterministic **game-state oracle** — it never calls an LLM. It has two consumption
modes:

1. **Transcript** (`render` / `mj_render_game`): a lean, faithful play-by-play (reconstructed hands,
   calls, riichi, wins, scores) annotated with objective metrics (shanten / ukeire / waits / dora /
   danger) and explicit **commentary anchors** `〔解説ポイント#N: 種別｜…〕` — deal, riichi,
   push/fold, open-hand judgement (副露判断 at an early 2nd or the 3rd meld, with a deterministic
   `┗ 役読み:` yaku outlook), wind-boundary standings reviews (中間総括 at 南入/西入), and
   end-of-hand/game summaries.
2. **Snapshot recall** (`snapshot` / `mj_get_snapshot`): the consuming LLM calls _back into_ mjrender —
   via MCP or the CLI — to see the full board (all four rivers with tedashi/tsumogiri marks, melds,
   live scores + placements, riichi states, dora, remaining wall, every concealed hand + metrics) at
   any anchor `#N` or any explicit kyoku + junme. The transcript stays lean; state is read, not
   reconstructed from 40 lines of deltas.

## Requirements

[Deno](https://deno.land) 2.x (this repo uses `/usr/local/bin/deno`). No Node, no `npm install`, no
build step — Deno runs the TypeScript directly and fetches npm dependencies (`fast-xml-parser`; for
the MCP server `@modelcontextprotocol/sdk`

- `zod`) via `npm:` specifiers on first run.

## CLI usage

```sh
cd mjrender
deno task render ../1.mjlog                    # full transcript (gzipped or plain XML)
deno run --allow-read src/cli.ts outline ../1.mjlog         # crude outline: headers/results/anchor index
deno run --allow-read src/cli.ts kyoku S3 ../1.mjlog        # one round, self-contained
deno run --allow-read src/cli.ts anchors ../1.mjlog         # list commentary anchors
deno run --allow-read src/cli.ts snapshot --anchor 12 ../1.mjlog
deno run --allow-read src/cli.ts snapshot --kyoku E1.2 --junme 8 ../1.mjlog
deno run --allow-read src/cli.ts facts start S3 ../1.mjlog    # also: result <sel> / riichi [sel] / standings
deno run --allow-read --allow-write src/cli.ts weave comments.json --out final.txt ../1.mjlog
deno task render "https://tenhou.net/0/?log=<id>&tw=1"        # fetch straight from tenhou.net
```

`weave` splices LLM-written commentary into a re-rendered transcript deterministically — the model
never copies fact lines (long verbatim reproduction is where tile facts get corrupted); it only
produces the comments. `comments.json` is either a bare anchor map / list, or the full form with
optional ★-line notes addressed by game position:

```json
{
 "anchors": { "1": "配牌についての解説…", "2": "…" },
 "notes": [{ "kyoku": "E1", "junme": 5, "seat": 3, "text": "このドラ切りは早い。" }]
}
```

With `--out <file>` it writes the woven document and prints a one-line summary; without it the
document goes to stdout. `--missing keep|strip` controls unfilled anchor placeholders (default
keep, so partial drafts are valid and can be re-woven with a fuller comment set later). The woven
document swaps the commentator instructions for a reader-facing legend.

Every `<file>` argument also accepts a tenhou.net URL — a replay-viewer link (`/0/?log=<id>&tw=N`)
is rewritten to the raw log endpoint (`/0/log/find.cgi`) automatically (needs
`--allow-net=tenhou.net`, included in the tasks).

Options:

- `--hands key|all` — reconstructed-hand verbosity. `key` (default) shows a player's full hand only
  at flagged beats (advance / riichi / dangerous push / win / draw); `all` shows every player's hand
  after every discard.
- `--snapshots inline` — embed the full board snapshot above every anchor, for consumers without
  tool access (token-heavy; default is lean).
- Kyoku selectors: `S3` / `東1` (wind + number), `E1.2` (= 東1局2本場, when a kyoku repeats), or a
  0-based round index like `6`.

Other tasks: `deno task check` (typecheck), `deno task test`, `deno task bundle` (build the
`mjrender.mcpb` Claude Desktop extension: `deno compile` the MCP server + pack with
`@anthropic-ai/mcpb`).

## MCP server

The server runs from the bundle, not `src/mcp.ts` directly — the source keeps
extensionless `@modelcontextprotocol/sdk` imports so `deno check` resolves the
SDK's real types, and those only resolve at runtime after bundling:

```sh
deno bundle -o mcp.mjs src/mcp.ts   # or `deno task bundle` (also builds the .mcpb)
deno run --allow-read --allow-write --allow-env=HOME --allow-net=tenhou.net mcp.mjs
```

Register with Claude Code:

```sh
claude mcp add mjrender -- deno run --allow-read --allow-write --allow-env=HOME --allow-net=tenhou.net /path/to/mjrender/mcp.mjs
```

(`--allow-write`/`--allow-env=HOME` are only needed for `mj_weave_commentary`, which writes the
woven document to a file so it never round-trips through the model's context. A relative `out`
lands next to the log file — or under `$HOME` for URL sources — since the calling agent may not
share a filesystem with the server at all.)

For the Claude Desktop app, build the self-contained extension instead: `deno task bundle`, then
install `mjrender.mcpb` via Settings → Extensions (no Deno needed on the target machine).

The server is **stateful and paced**: `mj_open_log` is the only tool that takes a path (a local
file or a tenhou.net URL) — it parses the log once into the session, and every other tool operates
on the opened log (erroring until one is opened). The session holds a **focus cursor** (the
current kyoku, starting at the first round): per-turn detail — kyoku renders, snapshots, per-kyoku
facts, draft writes — is hard-gated to the focus round and earlier, and `mj_next_kyoku` advances
the cursor only once every anchor of the focus kyoku is commented. The gate is a *pacing device,
not a spoiler shield*: the `mj_render_game` outline (results included) stays ungated for
orientation; what the gate bounds is how much detail the consuming LLM can chew per chat turn.
The commentary draft also lives server-side. Tools (thin wrappers over `src/core.ts`):

| tool                         | arguments                        | returns / gating                                      |
| ---------------------------- | -------------------------------- | ----------------------------------------------------- |
| `mj_open_log`                | `path`, `fresh?`                 | parses the log into the session; starts/keeps a draft + focus; appends the notation legend (once per process) |
| `mj_render_game`             |                                  | **ungated** game outline: headers, results, anchor index — read once to orient |
| `mj_render_kyoku`            | `kyoku`, `hands?`, `snapshots?`  | one round ≤ focus, full per-turn detail; snapshots **inline by default**, no legend header; appends the owari section on the last round |
| `mj_list_anchors`            |                                  | `#id kind kyoku junme seat topic` per line, unlocked rounds only |
| `mj_get_snapshot`            | `anchor` \| (`kyoku`, `junme`)   | board snapshot block (round ≤ focus)                  |
| `mj_add_comment`             | `comments[{anchor,text}]`        | saves anchor comments (batch ≤10, atomic); focus round = new fills, past rounds = replace-only, future locked |
| `mj_add_note`                | `notes[{kyoku,junme,seat,text}]` | saves ★-line one-liners (batch ≤10, atomic); any round ≤ focus — past notes stay correctable (re-save replaces, empty `text` deletes); future locked |
| `mj_next_kyoku`              |                                  | advances the focus once all focus-kyoku anchors are filled (errors listing what's missing; wind boundaries demand the 中間総括 first); replies with a ★-note hint and an instruction to END THE TURN |
| `mj_draft_status`            |                                  | checklist: ✓/・ per unlocked anchor, plus saved ★ notes |
| `mj_weave_commentary`        | `out`, `missing?`, `hands?`      | ungated; writes the woven draft to `out`, returns summary only (loud `warning: partial weave` when anchors are unfilled) |
| `mj_get_kyoku_start`         | `kyoku`                          | JSON: dealer/honba/kyotaku/dora, scores + placements (round ≤ focus) |
| `mj_get_kyoku_result`        | `kyoku`                          | JSON: winner/tile/points/yaku, or draw + tenpai seats (round ≤ focus) |
| `mj_get_riichi_declarations` | `kyoku?`                         | JSON: seat/junme/waits/live count/anchor id (filtered to unlocked rounds) |

Intended flow — **one kyoku per chat turn**: the agent opens the log (players, focus line, legend)
and orients once with the ungated `mj_render_game` outline; then each turn it renders the focus
kyoku with `mj_render_kyoku` (board snapshots already embedded above each anchor — except the
配牌評価, whose deal block is the board), pulls extra positions with `mj_get_snapshot` if needed,
saves that kyoku's anchor comments with `mj_add_comment` plus ★ one-liners with `mj_add_note`
(best written now — `mj_next_kyoku` nudges when they're sparse; past notes stay editable), and
calls `mj_next_kyoku`, whose reply says to end the turn. At each wind boundary (南入/西入) the
advance is held until the 中間総括 anchor — a standings/score-condition review — is written; the
gate reply carries the current standings. After the last round (終局総括 included),
`mj_weave_commentary` splices the accumulated draft into a re-rendered transcript. The agent never
reproduces fact lines, and the finished document is written to a file rather than passed back
through the model.

Upgrading from 0.4.x: the wind-boundary 中間総括 anchor is inserted into the id sequence, so saved
comment JSONs from 0.4.x shift by +1 past each boundary; `mj_get_final_standings` was removed (the
ungated outline's ◆終局 block carries the same data — `finalStandings` remains in core/CLI/eval).

## Eval harness (ground truth only)

```sh
deno task eval ../1.mjlog > qa.jsonl
```

Emits JSONL `{question, answer, kyoku, category}` — per-round scores/dora, winners and winning
tiles, ryuukyoku tenpai lists, riichi waits + live counts, final placements — all computed by the
replay engine. mjrender never calls an LLM: feed the transcript + questions to a target model
yourself and score its answers against these to settle formatting questions empirically.

## The commentary-anchor convention

The transcript is plain Japanese text with three interleaved layers:

1. **Fact lines** — e.g. `P1 ツモ ④ → 打 白  〔向聴1 受入5種14枚 ドラ0〕`. The `〔…〕` tag carries
   computed metrics for the acting player after the play.
2. **Reconstructed hands** — `┗ P1手: …` lines under a flagged beat, showing the exact concealed
   hand (+ melds) at that decision point. `★` marks the beat.
3. **Commentary anchors** — `〔解説ポイント#N: 種別｜…〕` lines. **Each is a slot the consuming LLM
   writes a comment for — by id, not by rewriting the transcript** (`weave` does the merging).
   `#N` is a stable position id: `mj_get_snapshot` (MCP) / `snapshot --anchor N` (CLI) reproduce
   the exact board state the slot is about. 種別 says what the slot wants: 配牌評価 / リーチ判断 /
   押し引き / 副露判断 / 局総括 / 流局評価 / 中間総括 / 終局総括. ★ lines can additionally take an
   optional one-liner note, addressed by kyoku + junme + seat.

Anchors are placed after: the deal (配牌), every riichi declaration, any push of a flagged dangerous
tile, every win/draw (和了/流局), each wind boundary (`== 南入 ==` block with standings — the
中間総括 slot), plus a final 終局 summary.

Metrics vocabulary: `向聴N` (shanten), `受入 X種Y枚` (ukeire kinds/tiles), `聴牌 待ち…` (tenpai
waits), `ドラN` (dora in hand), `危険度低/中/高` (a rough genbutsu/suji danger heuristic — the LLM
supplies real push/fold judgement).

## Module map

```
src/
  cli.ts      subcommands (render/outline/kyoku/anchors/snapshot/facts/weave) → core → stdout
  mcp.ts      stdio MCP server (mj_render_game/…/mj_get_snapshot/mj_weave_commentary/…)
  core.ts     query API: loadGame, renderGame, renderKyoku, listAnchors, getSnapshot, weaveCommentary
  load.ts     read file or tenhou.net URL + transparent gzip (DecompressionStream)
  parse.ts    mjlog XML → faithful Game model (fast-xml-parser, order-preserving)
  model.ts    domain types (incl. Beat = one addressable commentary anchor)
  state.ts    BoardState replay engine: rivers/melds/scores/wall/riichi + replayTo
  beats.ts    beat enumeration (delegates to the annotated render — ids can't drift)
  snapshot.ts render one BoardState as a self-sufficient board block
  tiles.ts    tile id↔type, Japanese notation, red-fives, dora successor
  meld.ts     decode the packed <N m="…"> meld bitfield
  yaku.ts     yaku / yakuman id → name tables
  shanten.ts  shanten (standard/chiitoi/kokushi) + ukeire engine
  danger.ts   discard danger: summary level + per-threat evidence (suji/kabe/counts)
  scoring.ts  placements (起家 tie-break) + オーラス overtake-needs search
  eval.ts     ground-truth Q/A generator (JSONL) for transcript evals
  render.ts   replay via BoardState, emit the anchored transcript + beat list
```

## Scope

4-player (yonma), log format `ver 2.3`. Sanma (3-player) is detected and rejected. Danger scoring is
an explicitly-labelled heuristic, not a solver. mjrender never calls an LLM; its only network access
is fetching a log from tenhou.net when given a URL.
