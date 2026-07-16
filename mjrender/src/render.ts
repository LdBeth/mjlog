// Replay a parsed Game and emit an LLM-ready Japanese commentary transcript:
// facts + reconstructed hands at key beats + 〔解説ポイント〕 commentary anchors.
//
// All game-state tracking lives in BoardState (state.ts); this module is the
// formatter that steps the state event-by-event and renders what it sees.

import type {
  AgariResult,
  Beat,
  BeatKind,
  Game,
  Meld,
  RenderOptions,
  Round,
  RyuukyokuResult,
  Tile,
} from "./model.ts";
import { assessDanger } from "./danger.ts";
import { overtakeNeeds, placements } from "./scoring.ts";
import { renderSnapshot } from "./snapshot.ts";
import { BoardState, type RestInfo } from "./state.ts";
import { countsFromTiles, shanten } from "./shanten.ts";
import { limitName, yakuName } from "./yaku.ts";
import {
  doraFromIndicatorType,
  isAka,
  renderHand,
  tileGlyph,
  tileType,
  typeGlyph,
} from "./tiles.ts";

const WIND = ["東", "南", "西", "北"];
const DISCARD_MARK = "★"; // salience flag on a commentary-worthy beat
const TSUMOGIRI_MARK = "▽"; // marks a tsumogiri (drawn tile discarded unchanged)

function roundName(kyoku: number): string {
  return `${WIND[Math.floor(kyoku / 4) % 4]}${(kyoku % 4) + 1}局`;
}

function P(seat: number): string {
  return `P${seat}`;
}

// Tenhou rank (段位) is a 0-based index into this table, NOT the dan number.
const DAN_NAMES = [
  "新人",
  "９級",
  "８級",
  "７級",
  "６級",
  "５級",
  "４級",
  "３級",
  "２級",
  "１級",
  "初段",
  "二段",
  "三段",
  "四段",
  "五段",
  "六段",
  "七段",
  "八段",
  "九段",
  "十段",
  "天鳳位",
];

function danName(dan: string): string {
  const i = Number(dan);
  return Number.isInteger(i) && i >= 0 && i < DAN_NAMES.length ? DAN_NAMES[i] : `段位${dan}`;
}

/** Self-describing preamble so any LLM receiving only the transcript knows how to read it. */
function formatInstruction(): string {
  return [
    "■この牌譜の読み方（解説者への指示）",
    "これは天鳳の対局記録です。あなたは麻雀の実況解説者として、文中の",
    "〔解説ポイント#N: 種別｜…〕 の各箇所を、その直前の局面・手牌をふまえた解説コメントに",
    "置き換えてください。それ以外の行は事実情報なので改変しないでください。",
    "また ★ の付いた行（注目の局面）には、任意でその行の直後に短い一言解説を",
    "添えてもよい。不要と判断すれば事実行のまま残してよい（★行への追記は任意）。",
    "・〔解説ポイント#N: 種別｜…〕のNは局面ID。この牌譜を出力したツール（mjrender）の",
    "  snapshot機能（MCPツール get_snapshot / CLI snapshot コマンド）にIDを渡すと、その時点の",
    "  全員の河・点数・手牌つき盤面を呼び出せる（利用できる場合のみ）。",
    "・牌表記: 一〜九=萬子 / ①〜⑨=筒子 / １〜９=索子 / 東南西北白發中=字牌 / 赤=赤ドラ",
    "・場風=局名で決まる風（東n局→東、南n局→南）、配牌の(東家/南家/西家/北家)=各家の自風。役牌判断に用いる",
    "・「N巡」=巡目。親（東家）が山からツモるたび1巡進む卓全体共有のカウンタで、各巡の最初の打牌にのみ表示",
    "・打牌表記: 「A → B」=Aをツモって手出しでB打、牌の後の「▽」=ツモ切り（引いた牌をそのまま捨て）、「（リーチ後）」=リーチ後の強制ツモ切り",
    "・〔向聴N 受入X種Y枚 ドラZ〕=打牌後の手牌評価、〔聴牌 待ち…〕=聴牌と待ち牌",
    "・「嶺上ツモ」=カン後の嶺上牌ツモ、「＋新ドラ」=カンによる新ドラ表示（表示位置は実際のめくり順）",
    "・★=注目の局面（任意で一言解説を添えてよい／不要ならそのまま） / ┗…手:=その時点の手牌",
    "・危険度低/中/高=リーチに対する放銃危険度の簡易目安（役牌＝場風/自風/三元は高めに評価）、「← 押し」=脅威に対する押し",
    "・点況=局開始時の順位と1つ上の順位との点差（▲）、「残り最短N局」=連荘なしと仮定した残り局数",
    "・「逆転条件」=オーラス・延長戦でトップに立つ最低の和了（本場・供託込み。ロン=他家から/直撃=トップから/ツモ）",
    "",
  ].join("\n");
}

/** Transcript text plus the machine-readable beat list behind its anchors. */
export interface RenderResult {
  text: string;
  beats: Beat[];
  /** Text split by section, so one kyoku can be served self-contained:
   *  header = format preamble + game/player block; rounds[i] = round i's lines. */
  sections: { header: string; rounds: string[]; owari?: string };
}

export function renderGame(game: Game, opts: RenderOptions): string {
  return renderGameAnnotated(game, opts).text;
}

/**
 * Render the transcript AND enumerate its commentary beats in the same pass.
 * Anchor IDs are assigned here and nowhere else — the query side (list_anchors /
 * get_snapshot) reads this enumeration, so IDs can never drift from the text.
 */
export function renderGameAnnotated(game: Game, opts: RenderOptions): RenderResult {
  const out: string[] = [];
  const beats: Beat[] = [];
  const g = game;

  out.push(formatInstruction());
  out.push("=".repeat(48));
  const rule = g.rules;
  const rw = [
    rule.hanchan ? "東南戦" : "東風戦",
    rule.sanma ? "三人打ち" : "四人打ち",
    rule.aka ? "赤有り" : "赤無し",
    rule.kuitan ? "喰断有り" : "喰断無し",
  ].join(" / ");
  out.push(`Tenhou 牌譜  ver${g.version}  [${rw}]`);
  for (const p of g.players) {
    out.push(
      `  ${P(p.seat)}: ${p.name}` +
        (p.dan ? `  (${danName(p.dan)}${p.rate ? ` R${p.rate}` : ""})` : ""),
    );
  }
  out.push("=".repeat(48));
  out.push("");
  const headerEnd = out.length;

  let lastJunme = 0;
  const roundBounds: Array<[number, number]> = [];
  for (let r = 0; r < g.rounds.length; r++) {
    const start = out.length;
    lastJunme = renderRound(g, g.rounds[r], r, opts, out, beats);
    out.push("");
    roundBounds.push([start, out.length]);
  }

  const owariStart = out.length;
  if (g.owari) {
    renderOwari(g, out, beats, {
      round: g.rounds.length - 1,
      junme: lastJunme,
      eventIndex: (g.rounds.at(-1)?.events.length ?? 0) - 1,
    });
  }
  return {
    text: out.join("\n"),
    beats,
    sections: {
      header: out.slice(0, headerEnd).join("\n"),
      rounds: roundBounds.map(([s, e]) => out.slice(s, e).join("\n")),
      owari: g.owari ? out.slice(owariStart).join("\n") : undefined,
    },
  };
}

/** Renders one round; returns its final 巡目 (for the game-end beat's position). */
function renderRound(
  g: Game,
  round: Round,
  roundIndex: number,
  opts: RenderOptions,
  out: string[],
  beats: Beat[],
): number {
  const aka = g.rules.aka;
  const st = new BoardState(g, round);

  // Each player's seat wind (自風) is their offset from the dealer (東家=親, then
  // 南西北 counterclockwise); it drives yakuhai value, so surface it at 配牌.
  const seatWind = (seat: number) => WIND[(seat - round.dealer + 4) % 4];

  // `shownJunme` tracks the last 巡目 printed so the `N巡` marker appears once per
  // go-around (at the first discard after the bump), not on every line.
  let shownJunme = 0;
  // Event index of a REACH step-2 already absorbed by renderDiscard (so the
  // riichi anchor's state — and its inline snapshot — includes the placed
  // stick, exactly like replayTo does); the main loop must skip it.
  let absorbedReach = -1;

  const metricTag = (info: RestInfo, seat: number): string => {
    const d = st.countDora(seat);
    if (info.shanten <= 0) {
      const waits = info.types.map((t) => typeGlyph(t)).join("");
      return `〔聴牌 待ち${waits || "?"} ドラ${d}〕`;
    }
    return `〔向聴${info.shanten} 受入${info.kinds}種${info.count}枚 ドラ${d}〕`;
  };

  const handLine = (seat: number, note = ""): string =>
    `  ┗ ${P(seat)}手: ${renderHand(st.hands[seat], st.melds[seat], aka)}${note ? "  " + note : ""}`;

  // Record a beat and emit its anchor line. The beat's board position is
  // (roundIndex, eventIndex): replayTo that position reproduces the state the
  // commentary slot is about.
  const pushAnchor = (kind: BeatKind, topic: string, eventIndex: number, seat?: number): void => {
    const beat: Beat = {
      id: beats.length + 1,
      kind,
      round: roundIndex,
      junme: st.junme,
      seat,
      eventIndex,
      topic,
    };
    beats.push(beat);
    // st IS the replayTo(round, eventIndex) state at every anchor site, so the
    // inline block matches what get_snapshot would serve for this beat.
    if (opts.snapshots === "inline") out.push(renderSnapshot(g, round, st, `#${beat.id}`));
    out.push(anchorLine(beat.id, kind, topic));
  };

  // --- header ---
  const indGlyph = tileGlyph(round.firstDora, aka);
  const doraGlyph = typeGlyph(doraFromIndicatorType(tileType(round.firstDora)));
  // Minimum hands left assumes no renchan; past the scheduled last hand it's 延長戦.
  const lastIdx = g.rules.hanchan ? 7 : 3;
  const remainTxt = round.kyoku > lastIdx ? "延長戦" : `残り最短${lastIdx - round.kyoku + 1}局`;
  out.push(
    `【${roundName(round.kyoku)} ${round.honba}本場】親: ${P(round.dealer)}(${
      g.players[round.dealer].name
    })` +
      `  供託${round.kyotaku}  ドラ表示:${indGlyph}(→ドラ${doraGlyph})  ${remainTxt}`,
  );

  // Placement-sorted scores with the gap (▲) to the seat one place above —
  // the frame every push/fold and オーラス judgement needs.
  const initialEast = g.rounds[0].dealer;
  const rank = placements(round.startScores, initialEast);
  const byRank = [0, 1, 2, 3].sort((a, b) => rank[a] - rank[b]);
  out.push(
    `  点況: ` + byRank.map((s, i) => {
      const gap = i === 0
        ? ""
        : `(▲${(round.startScores[byRank[i - 1]] - round.startScores[s]) * 100})`;
      return `${rank[s]}位${P(s)} ${round.startScores[s] * 100}${gap}`;
    }).join(" / "),
  );
  if (round.kyoku >= lastIdx) {
    for (const s of byRank.slice(1)) {
      const needs = overtakeNeeds({
        scores: round.startScores,
        seat: s,
        dealer: round.dealer,
        honba: round.honba,
        kyotaku: round.kyotaku,
        initialEast,
      });
      if (!needs) continue;
      out.push(
        needs.impossible
          ? `  ${P(s)}(${rank[s]}位) 逆転条件: なし（役満直撃でも届かず）`
          : `  ${P(s)}(${rank[s]}位) 逆転条件: ${
            [needs.ron, needs.direct, needs.tsumo].filter(Boolean).join(" / ")
          }`,
      );
    }
  }

  // --- 配牌 (all four starting hands) ---
  out.push("◆配牌");
  for (let seat = 0; seat < 4; seat++) {
    const info = st.restInfo(seat);
    st.restShanten[seat] = info.shanten;
    const swMark = `${seatWind(seat)}家${seat === round.dealer ? "・親" : ""}`;
    out.push(
      `  ${P(seat)}(${swMark}): ${renderHand(st.hands[seat], [], aka)}  ${metricTag(info, seat)}`,
    );
  }
  pushAnchor("配牌評価", "各家の配牌評価（手役の見込み・スピード・押し引きの構え）", -1);
  out.push("――");

  // --- event replay ---
  const ev = round.events;
  for (let i = 0; i < ev.length; i++) {
    const e = ev[i];

    if (e.t === "draw") {
      const before = st.restShanten[e.who];
      st.draw(e.who, e.tile, e.rinshan);
      const open = st.melds[e.who].length;
      const after = shanten(countsFromTiles(st.hands[e.who]), open, open === 0);
      const advanced = after < before;

      // peek: draw usually immediately followed by that player's discard
      const nxt = ev[i + 1];
      if (nxt && nxt.t === "discard" && nxt.who === e.who) {
        i++;
        renderDiscard(
          i,
          e.who,
          nxt.tile,
          nxt.tsumogiri,
          nxt.riichi,
          e.tile,
          e.rinshan,
          advanced,
          before,
          after,
        );
      } else {
        const rin = e.rinshan ? "(嶺上)" : "";
        out.push(`${P(e.who)} ツモ ${tileGlyph(e.tile, aka)}${rin}`);
      }
      continue;
    }

    if (e.t === "discard") {
      // a discard not preceded by this player's draw (e.g. right after a call)
      renderDiscard(
        i,
        e.who,
        e.tile,
        e.tsumogiri,
        e.riichi,
        -1,
        false,
        false,
        st.restShanten[e.who],
        st.restShanten[e.who],
      );
      continue;
    }

    if (e.t === "call") {
      const m = e.meld;
      const beforeCall = st.restShanten[m.who];
      st.applyMeld(m); // updates restShanten[m.who] to the post-call value
      const afterCall = st.restShanten[m.who];
      const fromTxt = m.kind === "ankan" || m.kind === "nuki" ? "" : ` (${P(m.fromWho)}から)`;
      const label = meldVerb(m);
      const meldHead = `${P(m.who)} ${label}${renderMeldTiles(m)}${fromTxt}`;
      const isKan = m.kind === "ankan" || m.kind === "daiminkan" || m.kind === "shouminkan";

      if (isKan) {
        // Integrate the kan turn into one line: kan → (dora) → rinshan draw →
        // (dora) → discard, keeping each new-dora reveal at its true stream
        // position (ankan reveals before the draw; minkan after it, per the log).
        const doraSeg = (ind: Tile) =>
          `＋新ドラ${tileGlyph(ind, aka)}(→ドラ${typeGlyph(doraFromIndicatorType(tileType(ind)))})`;
        const lead = [meldHead];
        let j = i + 1;
        let drawn = -1, rinshan = true, after = afterCall;
        let handled = false;
        while (j < ev.length) {
          const n = ev[j];
          if (n.t === "dora") {
            st.revealDora(n.indicator);
            lead.push(doraSeg(n.indicator));
            j++;
            continue;
          }
          if (n.t === "draw" && n.who === m.who && drawn < 0) {
            drawn = n.tile;
            rinshan = n.rinshan;
            st.draw(m.who, n.tile, n.rinshan);
            after = st.restInfo(m.who).shanten;
            lead.push(`嶺上ツモ ${tileGlyph(n.tile, aka)}`);
            j++;
            continue;
          }
          if (n.t === "discard" && n.who === m.who) {
            i = j;
            renderDiscard(
              j,
              m.who,
              n.tile,
              n.tsumogiri,
              n.riichi,
              drawn,
              rinshan,
              after < afterCall,
              afterCall,
              after,
              lead.join("  "),
            );
            handled = true;
            break;
          }
          break; // rinshan-kaihou win or unexpected — let the outer loop handle it
        }
        if (!handled) {
          out.push(lead.join("  ")); // no discard (e.g. 嶺上開花 tsumo); win renders next
          i = j - 1;
        }
        continue;
      }

      // non-kan call (chi / pon): show the shanten advance from fixing the set
      const advanced = afterCall < beforeCall;
      const afterTxt = afterCall <= 0 ? "聴牌" : `${afterCall}`;
      const delta = `向聴${beforeCall}→${afterTxt}`;
      const shText = advanced ? delta : afterCall <= 0 ? "聴牌" : `向聴${afterCall}`;
      out.push(
        `${meldHead}  〔${shText} ドラ${st.countDora(m.who)}〕${advanced ? " " + DISCARD_MARK : ""}`,
      );
      if (advanced) out.push(handLine(m.who, `(${label}で${delta}前進)`));
      continue;
    }

    if (e.t === "reach") {
      // step 1 is the declaration; the flagged discard follows and is handled there.
      // step 2 places the stick — BoardState keeps the live scores current.
      if (i !== absorbedReach) st.reach(e.who, e.step, e.scores);
      continue;
    }

    if (e.t === "dora") {
      st.revealDora(e.indicator);
      out.push(
        `＊新ドラ表示: ${tileGlyph(e.indicator, aka)}(→ドラ${
          typeGlyph(doraFromIndicatorType(tileType(e.indicator)))
        })`,
      );
      continue;
    }
  }

  // --- results ---
  for (const res of round.results) {
    if (res.kind === "agari") {
      // consistency guard: reconstructed concealed hand (+ ron tile) vs log `hai`
      const rec = [...st.hands[res.who]];
      if (res.who !== res.fromWho) rec.push(res.machi);
      const a = rec.map(tileType).sort((x, y) => x - y).join(",");
      const b = res.hand.map(tileType).sort((x, y) => x - y).join(",");
      if (a !== b) {
        warnInconsistent(`round ${round.kyoku} agari hand mismatch: rec=[${a}] log=[${b}]`);
      }
      renderAgari(g, round, res, out);
      pushAnchor("局総括", "決着の評価と局全体の総括（勝負の分かれ目）", ev.length - 1, res.who);
    } else {
      renderRyuukyoku(g, res, st.hands, st.melds, out);
      pushAnchor("流局評価", "流局時の聴牌・ノーテンと点棒状況の評価", ev.length - 1);
    }
  }
  out.push("―".repeat(20));
  return st.junme;

  // ===== nested helpers that close over round state =====

  function renderDiscard(
    eventIndex: number,
    who: number,
    tile: Tile,
    tsumogiri: boolean,
    riichi: boolean,
    drawn: Tile,
    rinshan: boolean,
    advanced: boolean,
    before: number,
    after: number,
    lead?: string, // when set (kan turn), replaces the "P# X →" prefix
  ): void {
    // Print the 巡 marker only when it just advanced (first discard of the go-around).
    const jmMark = st.junme !== shownJunme ? `${st.junme}巡 ` : "";
    shownJunme = st.junme;
    // Danger is assessed BEFORE the discard mutates state (the tile is judged
    // against the rivers as the player saw them when choosing it).
    const danger = st.riichiActive[who]
      ? null
      : assessDanger(tileType(tile), st.threats(who), st.publicVisible);

    // apply discard to state
    if (!st.discard(who, tile, tsumogiri, riichi)) {
      warnInconsistent(`${P(who)} discarded ${tile} not in hand (round ${round.kyoku})`);
    }

    // hands/melds/publicVisible are now stable through the rest of this function,
    // so resolve the resting-hand analysis ONCE and reuse it below (restShanten,
    // metricTag, and the riichi-wait readout) instead of 2–3 times.
    const info = st.restInfo(who);
    st.restShanten[who] = info.shanten;
    const rin = rinshan ? "(嶺上)" : "";
    const highDanger = !!danger && danger.level === "危険度高";
    const showAll = opts.hands === "all";
    // Letting a dora / red five leave the hand is a value decision worth flagging.
    // (A red five outranks a plain dora tile in the note; a tile can be both.)
    const doraKind = aka && isAka(tile) ? "赤ドラ" : st.doraTypeSet.has(tileType(tile)) ? "ドラ" : "";
    const isDoraDiscard = doraKind !== "";

    // Tsumogiri (and not a riichi declaration): the hand is unchanged, so collapse
    // the redundant "ツモ X → 打 X" into just "X ▽" and drop the metric tag. No
    // advance is realized (the drawn tile is thrown), so only high danger stars it.
    // (Skipped inside an integrated kan turn, which keeps the explicit "→ X ▽".)
    if (tsumogiri && !riichi && lead === undefined) {
      // After riichi the discard is forced (hand locked), not a choice — mark it,
      // and flag when the player is forced to pass a dora / red five.
      const forced = st.riichiActive[who];
      let state = "";
      if (forced) {
        state = `（リーチ後${doraKind ? "・" + doraKind : ""}）`;
      }
      const note = !forced && danger && (danger.level === "危険度高" || danger.level === "危険度中")
        ? `  ${danger.level}(${danger.seats.map(P).join(",")}リーチ)`
        : "";
      // dora note only when the forced-riichi `state` isn't already spelling it out
      const doraNote = !forced && isDoraDiscard ? `  ${doraKind}切り` : "";
      const star = highDanger || isDoraDiscard ? " " + DISCARD_MARK : "";
      out.push(
        `${P(who)} ${jmMark}${
          tileGlyph(tile, aka)
        }${rin} ${TSUMOGIRI_MARK}${state}${note}${doraNote}${star}`,
      );
      if (showAll) { for (let s = 0; s < 4; s++) out.push(handLine(s)); }
      return;
    }

    // A dangerous discard is only a "push" worth a push/fold anchor when the
    // discarder actually has a hand to defend (tenpai or 1-shanten). Far-from-
    // tenpai players discarding into a riichi are folding/forced — just tag it.
    const isDanger = !!danger && danger.seats.length > 0 &&
      (danger.level === "危険度高" || danger.level === "危険度中");
    const isPush = isDanger && st.restShanten[who] <= 1;
    const dangerSeats = isDanger ? danger!.seats.map(P).join(",") : "";

    // build the fact line (a chosen discard from hand)
    const star = advanced || riichi || highDanger || isPush || isDoraDiscard
      ? " " + DISCARD_MARK
      : "";
    const flagTxt = riichi ? `(${st.junme}巡目リーチ宣言・横向き)` : "";
    const tgMark = tsumogiri && !riichi ? ` ${TSUMOGIRI_MARK}` : "";
    const prefix = lead !== undefined
      ? `${lead}  → `
      : `${P(who)} ${jmMark}${drawn >= 0 ? `${tileGlyph(drawn, aka)}${rin} → ` : ""}`;
    const inlineDanger = isDanger && !isPush ? `  ${danger!.level}(${dangerSeats}リーチ)` : "";
    // a riichi declaration discarding a dora already reads as notable; skip the
    // redundant tag there, but flag any other dora/aka discard inline.
    const inlineDora = isDoraDiscard && !riichi ? `  ${doraKind}切り` : "";
    out.push(
      `${prefix}${tileGlyph(tile, aka)}${flagTxt}${tgMark}  ${
        metricTag(info, who)
      }${inlineDanger}${inlineDora}${star}`,
    );

    // reconstructed-hand displays at key beats
    if (riichi) {
      const nxt = ev[eventIndex + 1];
      if (nxt && nxt.t === "reach" && nxt.step === 2) {
        st.reach(nxt.who, 2, nxt.scores);
        absorbedReach = eventIndex + 1;
      }
      const waits = info.types.map((t) => typeGlyph(t)).join("") || "?";
      out.push(handLine(who, `待ち: ${waits}`));
      pushAnchor(
        "リーチ判断",
        `${P(who)}のリーチ判断と待ちの良し悪し（打点・待ち枚数・巡目）`,
        eventIndex,
        who,
      );
    } else if (isPush) {
      const stTxt = st.restShanten[who] <= 0 ? "聴牌" : `向聴${st.restShanten[who]}`;
      const adv = advanced ? `・${tileGlyph(drawn, aka)}ツモで${before}→${after}前進` : "";
      out.push(handLine(who, `${danger!.level}(${dangerSeats}リーチ) 自分${stTxt}${adv} ← 押し`));
      pushAnchor(
        "押し引き",
        `${P(who)}の押し引き（自分の手牌価値 vs リーチの脅威）`,
        eventIndex,
        who,
      );
    } else if (advanced) {
      out.push(handLine(who, `(${tileGlyph(drawn, aka)}ツモで向聴${before}→${after})`));
    } else if (showAll) {
      for (let s = 0; s < 4; s++) out.push(handLine(s));
    }
  }
}

function meldVerb(m: Meld): string {
  switch (m.kind) {
    case "chi":
      return "チー";
    case "pon":
      return "ポン";
    case "daiminkan":
      return "大明槓";
    case "shouminkan":
      return "加槓";
    case "ankan":
      return "暗槓";
    case "nuki":
      return "抜き";
  }
}

function renderMeldTiles(m: Meld): string {
  return m.tiles.map((t) => tileGlyph(t)).join("");
}

function anchorLine(id: number, kind: BeatKind, topic: string): string {
  return `〔解説ポイント#${id}: ${kind}｜${topic}〕`;
}

/** Dev-time self-check: replay inconsistencies go to stderr, never to the transcript. */
function warnInconsistent(msg: string): void {
  console.error(`[warn] ${msg}`);
}

function renderAgari(g: Game, round: Round, res: AgariResult, out: string[]): void {
  const aka = g.rules.aka;
  const tsumo = res.who === res.fromWho;
  const how = tsumo ? "ツモ" : `ロン(${P(res.fromWho)}から)`;
  out.push(`◆和了 ${P(res.who)}(${g.players[res.who].name}) ${how} ${tileGlyph(res.machi, aka)}`);
  out.push(
    `  和了手: ${renderHand(res.hand, res.melds, aka)}  （和了牌 ${tileGlyph(res.machi, aka)}）`,
  );

  const yakuParts: string[] = [];
  let doraN = 0, uraN = 0, akaN = 0;
  for (const y of res.yaku) {
    if (y.id === 52) doraN += y.han;
    else if (y.id === 53) uraN += y.han;
    else if (y.id === 54) akaN += y.han;
    else yakuParts.push(`${yakuName(y.id)}(${y.han})`);
  }
  for (const ym of res.yakuman) yakuParts.push(`${yakuName(ym)}(役満)`);
  const extra: string[] = [];
  if (doraN) extra.push(`ドラ${doraN}`);
  if (uraN) extra.push(`裏${uraN}`);
  if (akaN) extra.push(`赤${akaN}`);
  const lim = limitName(res.limit);
  out.push(
    `  役: ${[...yakuParts, ...extra].join(" ")}  ${res.fu ? res.fu + "符" : ""}${res.points}点${
      lim ? " " + lim : ""
    }`,
  );
  if (res.doraHai.length) {
    out.push(
      `  ドラ表示: ${res.doraHai.map((t) => tileGlyph(t, aka)).join("")}` +
        (res.uraDoraHai.length
          ? `  裏ドラ表示: ${res.uraDoraHai.map((t) => tileGlyph(t, aka)).join("")}`
          : ""),
    );
  }
  out.push(`  点棒: ${scoreDeltaLine(res.sc)}`);
}

function renderRyuukyoku(
  g: Game,
  res: RyuukyokuResult,
  hands: Tile[][],
  melds: Meld[][],
  out: string[],
): void {
  const aka = g.rules.aka;
  const typeLabel: Record<string, string> = {
    yao9: "九種九牌",
    kaze4: "四風連打",
    reach4: "四家立直",
    ron3: "三家和",
    kan4: "四槓散了",
    nm: "流し満貫",
  };
  const label = res.type ? (typeLabel[res.type] ?? res.type) : "荒牌平局";
  out.push(`◆流局（${label}）`);
  for (const th of res.tenpaiHands) {
    out.push(`  ${P(th.who)} 聴牌: ${renderHand(th.hand, melds[th.who], aka)}`);
  }
  if (res.sc.length) out.push(`  点棒: ${scoreDeltaLine(res.sc)}`);
}

function scoreDeltaLine(sc: number[]): string {
  const parts: string[] = [];
  for (let s = 0; s < 4 && s * 2 + 1 < sc.length; s++) {
    const before = sc[s * 2] * 100;
    const delta = sc[s * 2 + 1] * 100;
    const sign = delta > 0 ? "+" : "";
    parts.push(`${P(s)} ${sign}${delta} → ${before + delta}`);
  }
  return parts.join(" / ");
}

function renderOwari(
  g: Game,
  out: string[],
  beats: Beat[],
  pos: { round: number; junme: number; eventIndex: number },
): void {
  const o = g.owari!;
  out.push("=".repeat(48));
  out.push("◆終局");
  const rows = [];
  for (let s = 0; s < 4 && s * 2 + 1 < o.length; s++) {
    rows.push({ seat: s, score: o[s * 2] * 100, pt: o[s * 2 + 1] });
  }
  rows.sort((a, b) => b.score - a.score);
  rows.forEach((r, idx) => {
    out.push(
      `  ${idx + 1}位 ${P(r.seat)}(${g.players[r.seat].name})  ${r.score}点  (${
        r.pt >= 0 ? "+" : ""
      }${r.pt})`,
    );
  });
  const topic = "対局全体の総括（着順・打ち回しの評価）";
  const beat: Beat = { id: beats.length + 1, kind: "終局総括", topic, ...pos };
  beats.push(beat);
  out.push(anchorLine(beat.id, beat.kind, topic));
  out.push("=".repeat(48));
}
