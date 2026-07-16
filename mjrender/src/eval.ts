// Ground-truth Q/A generator for transcript evals.
//
//   deno task eval <file.mjlog|xml>     → JSONL {question, answer, kyoku, category}
//
// A thin consumer of the structured fact queries in core.ts (the same ones the
// MCP fact tools serve) — the questions and the tools can never disagree.
// No LLM is involved: feed the transcript + questions to a target model
// separately and score its answers against these.

import type { Game } from "./model.ts";
import { finalStandings, kyokuResults, kyokuStart, loadGame, riichiDeclarations, roundLabel } from "./core.ts";

export interface EvalItem {
  question: string;
  answer: string;
  kyoku: string; // round label, or "全体" for game-level facts
  category: string;
}

export function generateEval(game: Game): EvalItem[] {
  const items: EvalItem[] = [];

  for (let r = 0; r < game.rounds.length; r++) {
    const label = roundLabel(game, r);
    const start = kyokuStart(game, String(r));
    items.push({
      question: `${label}の開始時点数は？（P0/P1/P2/P3の順）`,
      answer: start.seats.map((s) => s.score).join("/"),
      kyoku: label,
      category: "start_scores",
    });
    items.push({
      question: `${label}の開始時のドラ表示牌は？`,
      answer: start.doraIndicator,
      kyoku: label,
      category: "first_dora",
    });
    for (const res of kyokuResults(game, String(r))) {
      if (res.type === "agari") {
        items.push({
          question: `${label}の和了者と決め方（ツモ/ロン）と素点は？`,
          answer: res.tsumo
            ? `P${res.who}ツモ${res.points}点`
            : `P${res.who}がP${res.fromWho}からロン${res.points}点`,
          kyoku: label,
          category: "winner_points",
        });
        items.push({
          question: `${label}の和了牌は？`,
          answer: res.winningTile,
          kyoku: label,
          category: "winning_tile",
        });
      } else if (res.reason === "荒牌平局") {
        items.push({
          question: `${label}の流局時に聴牌していたのは？（全員ノーテンなら「なし」）`,
          answer: res.tenpaiSeats.map((s) => `P${s}`).join(",") || "なし",
          kyoku: label,
          category: "ryuukyoku_tenpai",
        });
      }
    }
  }

  for (const r of riichiDeclarations(game)) {
    items.push({
      question: `${r.kyoku}でP${r.seat}がリーチ宣言（${r.junme}巡目）した時点の待ち牌は？`,
      answer: r.waits || "?",
      kyoku: r.kyoku,
      category: "riichi_wait",
    });
    items.push({
      question: `${r.kyoku}のP${r.seat}のリーチ（${r.junme}巡目）宣言時、待ち牌は場に何枚残っていた？`,
      answer: `${r.waitCount}枚`,
      kyoku: r.kyoku,
      category: "riichi_wait_count",
    });
  }

  const standings = finalStandings(game);
  if (standings) {
    items.push({
      question: "最終順位は？（1位から順にP番号）",
      answer: standings.map((s) => `P${s.seat}`).join(","),
      kyoku: "全体",
      category: "final_placements",
    });
  }

  return items;
}

if (import.meta.main) {
  const file = Deno.args[0];
  if (!file) {
    console.error("usage: deno run --allow-read src/eval.ts <file.mjlog|xml>");
    Deno.exit(2);
  }
  try {
    const game = await loadGame(file);
    for (const item of generateEval(game)) console.log(JSON.stringify(item));
  } catch (err) {
    console.error("error:", err instanceof Error ? err.message : String(err));
    Deno.exit(1);
  }
}
