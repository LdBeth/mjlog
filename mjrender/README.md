# mjrender

Render a Tenhou **mjlog** into an **LLM-ready Japanese commentary transcript** —
a faithful play-by-play (reconstructed hands, calls, riichi, wins, scores)
annotated with objective metrics (shanten / ukeire / waits / dora / danger) and
explicit **commentary-insertion anchors**. The tool makes no API calls; an LLM
reading the transcript writes the actual TV-style commentary at the marked slots.

This is the CLI phase. The core (`src/core.ts` → `render(path)`) is designed to
be wrapped unchanged by a future MCP server.

## Requirements

[Deno](https://deno.land) 2.x (this repo uses `/usr/local/bin/deno`). No Node,
no `npm install`, no build step — Deno runs the TypeScript directly and fetches
the one npm dependency (`fast-xml-parser`) via an `npm:` specifier on first run.

## Usage

```sh
cd mjrender
deno task render ../1.mjlog                 # gzipped or plain Xml, both work
deno run --allow-read src/cli.ts ../1.xml   # equivalent
deno run --allow-read src/cli.ts --hands=all ../1.xml   # show every hand every turn
```

Options:

- `--hands key|all` — reconstructed-hand verbosity. `key` (default) shows a
  player's full hand only at flagged beats (advance / riichi / dangerous push /
  win / draw); `all` shows every player's hand after every discard.

Other tasks: `deno task check` (typecheck), `deno task test`.

## The commentary-anchor convention

The transcript is plain Japanese text with three interleaved layers:

1. **Fact lines** — e.g. `P1 ツモ ④ → 打 白  〔シャンテン1 受入5種14枚 ドラ0〕`.
   The `〔…〕` tag carries computed metrics for the acting player after the play.
2. **Reconstructed hands** — `┗ P1手: …` lines under a flagged beat, showing the
   exact concealed hand (+ melds) at that decision point. `★` marks the beat.
3. **Commentary anchors** — `〔解説ポイント: …〕` lines. **Each of these is a slot
   for the consuming LLM to replace with commentary prose**, using the facts and
   the reconstructed hand shown directly above it.

Anchors are placed after: the deal (配牌), every riichi declaration, any push of
a flagged dangerous tile, and every win/draw (和了/流局), plus a final 終局 summary.

Metrics vocabulary: `シャンテン` (shanten), `受入 X種Y枚` (ukeire kinds/tiles),
`テンパイ 待ち…` (tenpai waits), `ドラN` (dora in hand), `危険度低/中/高` (a rough
genbutsu/suji danger heuristic — the LLM supplies real push/fold judgement).

## Module map

```
src/
  cli.ts     arg parsing → core.render → stdout
  core.ts    render(path, opts): load → parse → renderGame   (MCP wraps this)
  load.ts    read file + transparent gzip (DecompressionStream)
  parse.ts   mjlog XML → faithful Game model (fast-xml-parser, order-preserving)
  model.ts   domain types
  tiles.ts   tile id↔type, Japanese notation, red-fives, dora successor
  meld.ts    decode the packed <N m="…"> meld bitfield
  yaku.ts    yaku / yakuman id → name tables
  shanten.ts shanten (standard/chiitoi/kokushi) + ukeire engine
  danger.ts  riichi discard danger heuristic (genbutsu / suji)
  render.ts  replay events, reconstruct hands, emit the anchored transcript
```

## Scope

4-player (yonma), log format `ver 2.3`. Sanma (3-player) is detected and
rejected. Danger scoring is an explicitly-labelled heuristic, not a solver.
