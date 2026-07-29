# mjrender

Render a Tenhou **mjlog** into an **LLM-ready Japanese commentary transcript**, and serve
**recallable board snapshots** for any position in the game.

mjrender is a deterministic **game-state oracle** — it never calls an LLM. It has two consumption
modes:

1. **Transcript** (`render` / `mj_render_kyoku`): a lean, faithful play-by-play (reconstructed
   hands, calls, riichi, wins, scores) annotated with objective metrics (shanten / ukeire / waits /
   dora / danger) and explicit **commentary anchors** `〔解説ポイント#N: 種別｜…〕` — deal, riichi,
   push/fold, open-hand judgement (副露判断 at an early 2nd or the 3rd meld, with a deterministic
   `┗ 役読み:` yaku outlook), wind-boundary standings reviews (中間総括 at 南入/西入), and
   end-of-hand/game summaries.
2. **Snapshot recall** (`snapshot` / `mj_get_snapshot`): the consuming LLM calls _back into_
   mjrender — via MCP or the CLI — to see the full board (all four rivers with tedashi/tsumogiri
   marks, melds, live scores + placements, riichi states, dora, remaining wall, every concealed
   hand + metrics) at any anchor `#N` or any explicit kyoku + junme. The transcript stays lean;
   state is read, not reconstructed from 40 lines of deltas.

## Requirements

[Deno](https://deno.land) 2.x (this repo uses `/usr/local/bin/deno`). No Node, no `npm install`, no
build step — Deno runs the TypeScript directly and fetches npm dependencies (`fast-xml-parser`; for
the MCP server `@modelcontextprotocol/server` 2.x — the MCP SDK v2 — and `zod` 4) via `npm:`
specifiers on first run.

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
document goes to stdout. `--missing keep|strip` controls unfilled anchor placeholders (default keep,
so partial drafts are valid and can be re-woven with a fuller comment set later). The woven document
swaps the commentator instructions for a reader-facing legend.

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

Other tasks: `deno task check` (typecheck), `deno task test`, `deno task mcp` (run the stdio MCP
server from source), `deno task bundle` (build the `mjrender.mcpb` Claude Desktop extension: bundle
`src/mcp.ts` → `mcp.mjs`, `deno compile` it + pack with `@anthropic-ai/mcpb`).

## MCP server

The stdio server (v0.7.0) is **stateless**, per the 2026-07-28 MCP spec's guidance: it holds no
session at all. **Every tool takes `log`** — a local `.mjlog`/`.xml` path or a tenhou.net URL — as
its first parameter; that value _is_ the handle the model passes back on each call. There is no
open/close, no cursor, nothing to restore. (A parsed-game cache keyed by path+mtime keeps the
per-call re-parse cheap; it is a transparent cache, not protocol state — a fresh process recomputes
identically.)

Run it directly, or from the bundle:

```sh
deno task mcp                       # run src/mcp.ts directly (SDK v2 resolves at runtime)
deno bundle -o mcp.mjs src/mcp.ts   # canonical distribution build (`deno task bundle` also packs the .mcpb)
deno run --allow-read --allow-write --allow-env=HOME --allow-net=tenhou.net mcp.mjs
```

Register with Claude Code:

```sh
claude mcp add mjrender -- deno run --allow-read --allow-write --allow-env=HOME --allow-net=tenhou.net /path/to/mjrender/mcp.mjs
```

(`--allow-write`/`--allow-env=HOME` cover the on-disk draft and `mj_weave_commentary`, which writes
the woven document to a file so it never round-trips through the model's context. A relative `out`
lands next to the log file — or under `$HOME` for URL sources — since the calling agent may not
share a filesystem with the server at all.)

For the Claude Desktop app, build the self-contained extension instead: `deno task bundle`, then
install `mjrender.mcpb` via Settings → Extensions (no Deno needed on the target machine). The bundle
(`mcp.mjs`) is the `deno compile` input for that extension.

Entry point is `serveStdio(buildServer)` on MCP SDK v2, which serves the new protocol era and still
answers legacy 2025-era `initialize` clients (Claude Desktop) from the same server factory.

### Drafts on disk

The commentary draft being built against a log lives at `$HOME/.mjrender/drafts/<key>.json`, where
`key` is the **sha256 of the log's decoded XML text**. The same game therefore reaches the same
draft whether it arrives as a gzipped `.mjlog`, a plain `.xml`, a renamed copy, or a tenhou.net URL.
Every `mj_add_comment` / `mj_add_note` persists immediately, whole-file via tmp+rename (atomic on
one filesystem), so a crash never loses saved work and **a server restart needs no recovery step** —
the next call with the same `log` sees the whole draft again.

Concurrency: draft-touching handlers are FIFO-serialized within a process, so a pipelining client
can never interleave two read-modify-write cycles. Two _separate_ server processes on the same log
are last-write-wins at whole-file granularity — an accepted trade-off for a single-user desktop
tool.

The file's shape is deliberately a valid `weave` input, so a raw draft can be fed to the CLI with no
transformation:

```json
{
  "version": 1,
  "log": "../1.mjlog",
  "savedAt": "…",
  "anchors": [{ "anchor": 1, "text": "配牌についての解説…" }],
  "notes": [{ "kyoku": "0", "junme": 5, "seat": 3, "text": "このドラ切りは早い。" }]
}
```

```sh
deno task render weave ~/.mjrender/drafts/<key>.json --out final.txt ../1.xml
```

`mj_clear_draft` deletes the file when you want to start over.

### Tools

Thin wrappers over `src/core.ts`; `log` is implicit in every row.

| tool                         | arguments (besides `log`)        | returns                                                                                                                                                              |
| ---------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mj_outline`                 |                                  | game outline: notation legend (always), players, per-kyoku headers + condensed results + anchor index — no per-turn lines; also reports the on-disk draft's coverage |
| `mj_render_kyoku`            | `kyoku`, `hands?`, `snapshots?`  | one round in full per-turn detail; snapshots **inline by default**, no legend header; appends the owari section on the last round                                    |
| `mj_list_anchors`            |                                  | the whole checklist: ✓/・, `#id kind kyoku junme seat topic` per line, plus the coverage line                                                                        |
| `mj_get_snapshot`            | `anchor` \| (`kyoku`, `junme`)   | board snapshot block for that position                                                                                                                               |
| `mj_add_comment`             | `comments[{anchor,text}]`        | saves anchor comments to the disk draft (batch ≤10, atomic); any anchor any time, re-save replaces                                                                   |
| `mj_add_note`                | `notes[{kyoku,junme,seat,text}]` | saves ★-line one-liners (batch ≤10, atomic); re-save replaces, empty `text` deletes                                                                                  |
| `mj_draft_status`            |                                  | checklist + saved ★ notes + the draft file's path + an advisory ★-coverage `HINT`                                                                                    |
| `mj_clear_draft`             |                                  | deletes the on-disk draft for the log (irreversible)                                                                                                                 |
| `mj_weave_commentary`        | `out`, `missing?`, `hands?`      | writes the woven document to `out`, returns a summary only (loud `warning: partial weave` when anchors are unfilled)                                                 |
| `mj_get_kyoku_start`         | `kyoku`                          | JSON: dealer/honba/kyotaku/dora, scores + placements                                                                                                                 |
| `mj_get_kyoku_result`        | `kyoku`                          | JSON: winner/tile/points/yaku, or draw + tenpai seats                                                                                                                |
| `mj_get_riichi_declarations` | `kyoku?`                         | JSON: seat/junme/waits/live count/anchor id (whole game when `kyoku` is omitted)                                                                                     |

### Pacing is advisory

Nothing is gated or locked: any round is readable and commentable at any time, and comments are
replace-anywhere (revising earlier rounds at 中間総括 / 終局総括 time is encouraged). The **one
kyoku per reply** rhythm is a _recommendation_, carried by the tool descriptions, the outline's
■アウトライン表示 note, and `mj_draft_status` hints — batch-reading rounds is what degrades
commentary, not a rule violation.

Recommended loop:

1. `mj_outline` **once** to orient (legend, players, results — not treated as spoilers here) and to
   see how far the draft already got.
2. Per reply, **one kyoku**: `mj_render_kyoku` (snapshots already embedded above each anchor, except
   the 配牌評価 whose deal block is the board) → `mj_get_snapshot` at riichi/tenpai moments →
   `mj_add_comment` for that round's anchors, plus optional `mj_add_note` ★ one-liners →
   `mj_draft_status` for what remains. Then end the turn.
3. Return to `mj_outline` at each wind boundary (中間総括) and at 終局 for the recap.
4. Once the checklist is full, `mj_weave_commentary` splices the draft into a re-rendered transcript
   and writes it to a file. The agent never reproduces fact lines, and the finished document never
   passes through the model's context.

`deno task test` covers the stateless flow end to end, including draft persistence across separate
server processes.

Upgrading to **0.7.0 (stateless redesign)**: removed `mj_open_log` / `mj_next_kyoku` /
`mj_restore_state`; `mj_render_game` → `mj_outline`; `mj_add_note` now takes `kyoku` per note;
drafts moved to disk (restart-proof); **no anchor renumbering** vs 0.6.0. Upgrading from 0.4.x: the
wind-boundary 中間総括 anchor is inserted into the id sequence, so saved comment JSONs from 0.4.x
shift by +1 past each boundary; `mj_get_final_standings` was removed (the outline's ◆終局 block
carries the same data — `finalStandings` remains in core/CLI/eval).

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
   writes a comment for — by id, not by rewriting the transcript** (`weave` does the merging). `#N`
   is a stable position id: `mj_get_snapshot` (MCP) / `snapshot --anchor N` (CLI) reproduce the
   exact board state the slot is about. 種別 says what the slot wants: 配牌評価 / リーチ判断 /
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
  mcp.ts      stateless stdio MCP server (mj_outline/…/mj_get_snapshot/mj_weave_commentary/…)
  draft.ts    on-disk commentary drafts ($HOME/.mjrender/drafts/<sha256-of-xml>.json, atomic writes)
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
