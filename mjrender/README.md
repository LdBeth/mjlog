# mjrender

Render a Tenhou **mjlog** into an **LLM-ready Japanese commentary transcript**, and serve
**recallable board snapshots** for any position in the game.

mjrender is a deterministic **game-state oracle** — it never calls an LLM. It has two consumption
modes:

1. **Transcript** (`render` / `mj_render_game`): a lean, faithful play-by-play (reconstructed hands,
   calls, riichi, wins, scores) annotated with objective metrics (shanten / ukeire / waits / dora /
   danger) and explicit **commentary anchors** `〔解説ポイント#N: 種別｜…〕`.
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
deno run --allow-read src/cli.ts kyoku S3 ../1.mjlog        # one round, self-contained
deno run --allow-read src/cli.ts anchors ../1.mjlog         # list commentary anchors
deno run --allow-read src/cli.ts snapshot --anchor 12 ../1.mjlog
deno run --allow-read src/cli.ts snapshot --kyoku E1.2 --junme 8 ../1.mjlog
deno run --allow-read src/cli.ts facts start S3 ../1.mjlog    # also: result <sel> / riichi [sel] / standings
deno task render "https://tenhou.net/0/?log=<id>&tw=1"        # fetch straight from tenhou.net
```

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

```sh
deno task mcp        # stdio MCP server
```

Register with Claude Code:

```sh
claude mcp add mjrender -- deno run --allow-read --allow-net=tenhou.net /path/to/mjrender/src/mcp.ts
```

For the Claude Desktop app, build the self-contained extension instead: `deno task bundle`, then
install `mjrender.mcpb` via Settings → Extensions (no Deno needed on the target machine).

Tools (thin wrappers over `src/core.ts`); `path` is a local file or a tenhou.net URL:

| tool                         | arguments                              | returns                                               |
| ---------------------------- | -------------------------------------- | ----------------------------------------------------- |
| `mj_render_game`             | `path`, `hands?`, `snapshots?`         | full lean transcript                                  |
| `mj_render_kyoku`            | `path`, `kyoku`, …                     | one round, self-contained                             |
| `mj_list_anchors`            | `path`                                 | `#id kind kyoku junme seat topic` per line            |
| `mj_get_snapshot`            | `path`, `anchor` \| (`kyoku`, `junme`) | board snapshot block                                  |
| `mj_get_kyoku_start`         | `path`, `kyoku`                        | JSON: dealer/honba/kyotaku/dora, scores + placements  |
| `mj_get_kyoku_result`        | `path`, `kyoku`                        | JSON: winner/tile/points/yaku, or draw + tenpai seats |
| `mj_get_riichi_declarations` | `path`, `kyoku?`                       | JSON: seat/junme/waits/live count/anchor id           |
| `mj_get_final_standings`     | `path`                                 | JSON: place/seat/name/score/±pt                       |

Intended flow: the agent renders the transcript once, then while writing commentary at each
`〔解説ポイント#N〕` recalls that anchor's exact board state with `mj_get_snapshot` instead of
re-deriving it from the fact lines.

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
3. **Commentary anchors** — `〔解説ポイント#N: 種別｜…〕` lines. **Each is a slot for the consuming
   LLM to replace with commentary prose.** `#N` is a stable position id: `mj_get_snapshot` (MCP) /
   `snapshot --anchor N` (CLI) reproduce the exact board state the slot is about, and downstream
   tooling can merge commentary back by id. 種別 says what the slot wants: 配牌評価 / リーチ判断 /
   押し引き / 局総括 / 流局評価 / 終局総括.

Anchors are placed after: the deal (配牌), every riichi declaration, any push of a flagged dangerous
tile, and every win/draw (和了/流局), plus a final 終局 summary.

Metrics vocabulary: `向聴N` (shanten), `受入 X種Y枚` (ukeire kinds/tiles), `聴牌 待ち…` (tenpai
waits), `ドラN` (dora in hand), `危険度低/中/高` (a rough genbutsu/suji danger heuristic — the LLM
supplies real push/fold judgement).

## Module map

```
src/
  cli.ts      subcommands (render/kyoku/anchors/snapshot) → core → stdout
  mcp.ts      stdio MCP server (mj_render_game/mj_render_kyoku/mj_list_anchors/mj_get_snapshot/…)
  core.ts     query API: loadGame, renderGame, renderKyoku, listAnchors, getSnapshot
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
