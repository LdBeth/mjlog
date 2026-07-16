// Self-sufficient board snapshot: everything a commentator needs to reason
// about ONE position without replaying the transcript — all four rivers (with
// tedashi/tsumogiri/riichi/called-away marks), melds, live scores + placement,
// riichi states, dora, wall count, and each seat's concealed hand + metrics.
//
// Consumed by the get_snapshot MCP tool / CLI subcommand, and optionally
// inlined above transcript anchors (--snapshots=inline).

import type { Game, Round } from "./model.ts";
import { placements } from "./scoring.ts";
import type { BoardState, RestInfo } from "./state.ts";
import {
  doraFromIndicatorType,
  renderHand,
  renderMeld,
  tileGlyph,
  tileType,
  typeGlyph,
} from "./tiles.ts";

const WIND = ["東", "南", "西", "北"];

function roundName(kyoku: number): string {
  return `${WIND[Math.floor(kyoku / 4) % 4]}${(kyoku % 4) + 1}局`;
}

function riverLine(st: BoardState, seat: number, aka: boolean): string {
  if (st.rivers[seat].length === 0) return "河:（なし）";
  const parts = st.rivers[seat].map((r) => {
    let s = tileGlyph(r.tile, aka);
    if (r.tsumogiri) s += "▽";
    if (r.riichiDeclare) s += "*";
    if (r.calledBy !== undefined) s += `(→P${r.calledBy})`;
    return s;
  });
  return `河: ${parts.join(" ")}`;
}

function metricText(info: RestInfo, doraN: number): string {
  if (info.shanten <= 0) {
    const waits = info.types.map((t) => typeGlyph(t)).join("");
    return `〔聴牌 待ち${waits || "?"} ドラ${doraN}〕`;
  }
  return `〔向聴${info.shanten} 受入${info.kinds}種${info.count}枚 ドラ${doraN}〕`;
}

/**
 * Render the board block for a replayed position. `note` (when given) is shown
 * in the header — e.g. the beat topic the snapshot was requested for.
 */
export function renderSnapshot(
  game: Game,
  round: Round,
  st: BoardState,
  note?: string,
): string {
  const aka = game.rules.aka;
  const out: string[] = [];

  const doraTypes = st.indicators.map((id) => doraFromIndicatorType(tileType(id)));
  const doraTxt = `ドラ:${doraTypes.map((t) => typeGlyph(t)).join("")}` +
    `(表示:${st.indicators.map((id) => tileGlyph(id, aka)).join("")})`;
  // riichi sticks placed this round sit on the table alongside the carried-over deposit
  const kyotaku = round.kyotaku + st.sticksPlaced;
  out.push(
    `┌盤面 ${roundName(round.kyoku)}${round.honba}本場 ${st.junme}巡目  残り山${st.wallRemaining}` +
      `  ${doraTxt}  供託${kyotaku}` + (note ? `  〔${note}〕` : ""),
  );

  const rank = placements(st.scores, game.rounds[0].dealer);
  for (let seat = 0; seat < 4; seat++) {
    const wind = WIND[(seat - round.dealer + 4) % 4];
    const name = game.players[seat].name;
    const riichi = st.riichiActive[seat] ? `・リーチ(${st.riichiJunme[seat]}巡)` : "";
    const melds = st.melds[seat].length
      ? `  副露${st.melds[seat].map((m) => renderMeld(m, aka)).join("")}`
      : "";
    out.push(
      `│${wind}家P${seat}(${name}) ${st.scores[seat] * 100}点 ${rank[seat]}位${riichi}: ` +
        `${riverLine(st, seat, aka)}${melds}`,
    );
  }

  for (let seat = 0; seat < 4; seat++) {
    const hand = renderHand(st.hands[seat], [], aka);
    // Metrics describe a resting (3n+1) hand; a snapshot taken mid-turn (after a
    // draw, before the discard) shows the 14-tile hand as a pending choice.
    const metric = st.hands[seat].length % 3 === 1
      ? `  ${metricText(st.restInfo(seat), st.countDora(seat))}`
      : "  （ツモ後・打牌選択前）";
    out.push(`│手牌 P${seat}: ${hand}${metric}`);
  }

  out.push("└");
  return out.join("\n");
}
