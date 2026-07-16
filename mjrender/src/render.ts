// Replay a parsed Game and emit an LLM-ready Japanese commentary transcript:
// facts + reconstructed hands at key beats + 〔解説ポイント〕 commentary anchors.

import type {
  AgariResult,
  Game,
  Meld,
  RenderOptions,
  Round,
  RyuukyokuResult,
  Tile,
} from "./model.ts";
import { assessDanger, type RiichiThreat } from "./danger.ts";
import { concealedTilesUsed } from "./meld.ts";
import { countsFromTiles, shanten, ukeireTypes } from "./shanten.ts";
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
  "新人", "９級", "８級", "７級", "６級", "５級", "４級", "３級", "２級", "１級",
  "初段", "二段", "三段", "四段", "五段", "六段", "七段", "八段", "九段", "十段", "天鳳位",
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
    "〔解説ポイント: …〕 の各箇所を、その直前の局面・手牌をふまえた解説コメントに",
    "置き換えてください。それ以外の行は事実情報なので改変しないでください。",
    "また ★ の付いた行（注目の局面）には、任意でその行の直後に短い一言解説を",
    "添えてもよい。不要と判断すれば事実行のまま残してよい（★行への追記は任意）。",
    "・牌表記: 一〜九=萬子 / ①〜⑨=筒子 / １〜９=索子 / 東南西北白發中=字牌 / 赤=赤ドラ",
    "・場風=局名で決まる風（東n局→東、南n局→南）、配牌の(東家/南家/西家/北家)=各家の自風。役牌判断に用いる",
    "・「N巡」=巡目。親（東家）が山からツモるたび1巡進む卓全体共有のカウンタで、各巡の最初の打牌にのみ表示",
    "・打牌表記: 「A → B」=Aをツモって手出しでB打、牌の後の「▽」=ツモ切り（引いた牌をそのまま捨て）、「（リーチ後）」=リーチ後の強制ツモ切り",
    "・〔向聴N 受入X種Y枚 ドラZ〕=打牌後の手牌評価、〔聴牌 待ち…〕=聴牌と待ち牌",
    "・「嶺上ツモ」=カン後の嶺上牌ツモ、「＋新ドラ」=カンによる新ドラ表示（表示位置は実際のめくり順）",
    "・★=注目の局面（任意で一言解説を添えてよい／不要ならそのまま） / ┗…手:=その時点の手牌",
    "・危険度低/中/高=リーチに対する放銃危険度の簡易目安（役牌＝場風/自風/三元は高めに評価）、「← 押し」=脅威に対する押し",
    "",
  ].join("\n");
}

export function renderGame(game: Game, opts: RenderOptions): string {
  const out: string[] = [];
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
    out.push(`  ${P(p.seat)}: ${p.name}` + (p.dan ? `  (${danName(p.dan)}${p.rate ? ` R${p.rate}` : ""})` : ""));
  }
  out.push("=".repeat(48));
  out.push("");

  for (const round of g.rounds) {
    renderRound(g, round, opts, out);
    out.push("");
  }

  if (g.owari) renderOwari(g, out);
  return out.join("\n");
}

function renderRound(g: Game, round: Round, opts: RenderOptions, out: string[]): void {
  const aka = g.rules.aka;

  // Each player's seat wind (自風) is their offset from the dealer (東家=親, then
  // 南西北 counterclockwise); it drives yakuhai value, so surface it at 配牌.
  const seatWind = (seat: number) => WIND[(seat - round.dealer + 4) % 4];
  // Same winds as tile *types* (東=27..北=30) plus the three dragons (31..33):
  // the yakuhai for a given seat, used to raise discard danger against its riichi.
  const roundWindType = 27 + (Math.floor(round.kyoku / 4) % 4);
  const seatWindType = (seat: number) => 27 + ((seat - round.dealer + 4) % 4);
  const valueHonors = (seat: number) =>
    new Set<number>([roundWindType, seatWindType(seat), 31, 32, 33]);

  // --- per-round mutable state ---
  const hands: Tile[][] = round.startHands.map((h) => [...h]);
  const melds: Meld[][] = [[], [], [], []];
  const discardTypes: Array<Set<number>> = [new Set(), new Set(), new Set(), new Set()];
  const publicVisible = new Array<number>(34).fill(0);
  const indicators: Tile[] = [round.firstDora];
  const riichiActive = [false, false, false, false];
  const safe: Array<Set<number>> = [new Set(), new Set(), new Set(), new Set()];
  const lastDraw = [-1, -1, -1, -1];
  const restShanten = [0, 0, 0, 0];
  // 巡目 (turn number): a single table-wide counter. One 巡 = one full go-around,
  // so it advances only when the dealer (親/東家) draws from the wall (rinshan
  // excluded). Every player shares the same current 巡目. `shownJunme` tracks the
  // last value printed so the `N巡` marker appears once per go-around (at the
  // first discard after the bump), not on every line.
  let junme = 0;
  let shownJunme = 0;

  const bumpVisible = (id: Tile) => publicVisible[tileType(id)]++;
  indicators.forEach(bumpVisible);

  const doraTypeList = () => indicators.map((id) => doraFromIndicatorType(tileType(id)));

  const countDora = (seat: number): number => {
    const dts = doraTypeList();
    const all = [...hands[seat], ...melds[seat].flatMap((m) => m.tiles)];
    let n = 0;
    for (const id of all) {
      for (const dt of dts) if (tileType(id) === dt) n++;
      if (aka && isAka(id)) n++;
    }
    return n;
  };

  // resting (3n+1) hand analysis for a seat
  const restInfo = (seat: number) => {
    const counts = countsFromTiles(hands[seat]);
    const open = melds[seat].length;
    const closed = open === 0;
    const s = shanten(counts, open, closed);
    const types = ukeireTypes(counts, open, closed);
    let total = 0;
    for (const t of types) total += Math.max(0, 4 - publicVisible[t] - counts[t]);
    return { shanten: s, kinds: types.length, count: total, types };
  };

  const metricTag = (seat: number): string => {
    const info = restInfo(seat);
    const d = countDora(seat);
    if (info.shanten <= 0) {
      const waits = info.types.map((t) => typeGlyph(t)).join("");
      return `〔聴牌 待ち${waits || "?"} ドラ${d}〕`;
    }
    return `〔向聴${info.shanten} 受入${info.kinds}種${info.count}枚 ドラ${d}〕`;
  };

  const handLine = (seat: number, note = ""): string =>
    `  ┗ ${P(seat)}手: ${renderHand(hands[seat], melds[seat], aka)}${note ? "  " + note : ""}`;

  // --- header ---
  const indGlyph = tileGlyph(round.firstDora, aka);
  const doraGlyph = typeGlyph(doraFromIndicatorType(tileType(round.firstDora)));
  out.push(
    `【${roundName(round.kyoku)} ${round.honba}本場】親: ${P(round.dealer)}(${g.players[round.dealer].name})` +
      `  供託${round.kyotaku}  ドラ表示:${indGlyph}(→ドラ${doraGlyph})  点棒:${round.startScores.map((s) => s * 100).join("/")}`,
  );

  // --- 配牌 (all four starting hands) ---
  out.push("◆配牌");
  for (let seat = 0; seat < 4; seat++) {
    restShanten[seat] = shanten(countsFromTiles(hands[seat]), 0, true);
    const swMark = `${seatWind(seat)}家${seat === round.dealer ? "・親" : ""}`;
    out.push(`  ${P(seat)}(${swMark}): ${renderHand(hands[seat], [], aka)}  ${metricTag(seat)}`);
  }
  out.push(anchor("各家の配牌評価（手役の見込み・スピード・押し引きの構え）"));
  out.push("――");

  // --- event replay ---
  const ev = round.events;
  for (let i = 0; i < ev.length; i++) {
    const e = ev[i];

    if (e.t === "draw") {
      if (!e.rinshan && e.who === round.dealer) junme++; // dealer's wall draw = new 巡
      hands[e.who].push(e.tile);
      lastDraw[e.who] = e.tile;
      const before = restShanten[e.who];
      const open = melds[e.who].length;
      const after = shanten(countsFromTiles(hands[e.who]), open, open === 0);
      const advanced = after < before;

      // peek: draw usually immediately followed by that player's discard
      const nxt = ev[i + 1];
      if (nxt && nxt.t === "discard" && nxt.who === e.who) {
        i++;
        renderDiscard(e.who, nxt.tile, nxt.tsumogiri, nxt.riichi, e.tile, e.rinshan, advanced, before, after);
      } else {
        const rin = e.rinshan ? "(嶺上)" : "";
        out.push(`${P(e.who)} ツモ ${tileGlyph(e.tile, aka)}${rin}`);
      }
      continue;
    }

    if (e.t === "discard") {
      // a discard not preceded by this player's draw (e.g. right after a call)
      renderDiscard(e.who, e.tile, e.tsumogiri, e.riichi, -1, false, false, restShanten[e.who], restShanten[e.who]);
      continue;
    }

    if (e.t === "call") {
      const m = e.meld;
      const beforeCall = restShanten[m.who];
      applyMeld(m); // updates restShanten[m.who] to the post-call value
      const afterCall = restShanten[m.who];
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
            indicators.push(n.indicator);
            bumpVisible(n.indicator);
            lead.push(doraSeg(n.indicator));
            j++;
            continue;
          }
          if (n.t === "draw" && n.who === m.who && drawn < 0) {
            drawn = n.tile;
            rinshan = n.rinshan;
            hands[m.who].push(n.tile);
            lastDraw[m.who] = n.tile;
            after = restInfo(m.who).shanten;
            lead.push(`嶺上ツモ ${tileGlyph(n.tile, aka)}`);
            j++;
            continue;
          }
          if (n.t === "discard" && n.who === m.who) {
            i = j;
            renderDiscard(
              m.who, n.tile, n.tsumogiri, n.riichi, drawn, rinshan, after < afterCall, afterCall, after,
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
      out.push(`${meldHead}  〔${shText} ドラ${countDora(m.who)}〕${advanced ? " " + DISCARD_MARK : ""}`);
      if (advanced) out.push(handLine(m.who, `(${label}で${delta}前進)`));
      continue;
    }

    if (e.t === "reach") {
      // step 1 is the declaration; the flagged discard follows and is handled there.
      // step 2 carries post-stick scores (informational).
      continue;
    }

    if (e.t === "dora") {
      indicators.push(e.indicator);
      bumpVisible(e.indicator);
      out.push(
        `＊新ドラ表示: ${tileGlyph(e.indicator, aka)}(→ドラ${typeGlyph(doraFromIndicatorType(tileType(e.indicator)))})`,
      );
      continue;
    }
  }

  // --- results ---
  for (const res of round.results) {
    if (res.kind === "agari") {
      // consistency guard: reconstructed concealed hand (+ ron tile) vs log `hai`
      const rec = [...hands[res.who]];
      if (res.who !== res.fromWho) rec.push(res.machi);
      const a = rec.map(tileType).sort((x, y) => x - y).join(",");
      const b = res.hand.map(tileType).sort((x, y) => x - y).join(",");
      if (a !== b) warnInconsistent(`round ${round.kyoku} agari hand mismatch: rec=[${a}] log=[${b}]`);
      renderAgari(g, round, res, out);
    } else {
      renderRyuukyoku(g, res, hands, melds, out);
    }
  }
  out.push("―".repeat(20));

  // ===== nested helpers that close over round state =====

  function renderDiscard(
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
    const jmMark = junme !== shownJunme ? `${junme}巡 ` : "";
    shownJunme = junme;
    const threats: RiichiThreat[] = [];
    for (let s = 0; s < 4; s++) {
      if (s !== who && riichiActive[s]) {
        threats.push({ seat: s, safeTypes: safe[s], valueHonors: valueHonors(s) });
      }
    }
    const danger = riichiActive[who] ? null : assessDanger(tileType(tile), threats, publicVisible);

    // apply discard to state
    const di = hands[who].indexOf(tile);
    if (di < 0) warnInconsistent(`${P(who)} discarded ${tile} not in hand (round ${round.kyoku})`);
    else hands[who].splice(di, 1);
    discardTypes[who].add(tileType(tile));
    bumpVisible(tile);
    if (riichi) {
      riichiActive[who] = true;
      safe[who] = new Set(discardTypes[who]);
    }
    for (let s = 0; s < 4; s++) if (riichiActive[s]) safe[s].add(tileType(tile));

    restShanten[who] = shanten(countsFromTiles(hands[who]), melds[who].length, melds[who].length === 0);
    const rin = rinshan ? "(嶺上)" : "";
    const highDanger = !!danger && danger.level === "危険度高";
    const showAll = opts.hands === "all";

    // Tsumogiri (and not a riichi declaration): the hand is unchanged, so collapse
    // the redundant "ツモ X → 打 X" into just "X ▽" and drop the metric tag. No
    // advance is realized (the drawn tile is thrown), so only high danger stars it.
    // (Skipped inside an integrated kan turn, which keeps the explicit "→ X ▽".)
    if (tsumogiri && !riichi && lead === undefined) {
      // After riichi the discard is forced (hand locked), not a choice — mark it,
      // and flag when the player is forced to pass a dora / red five.
      const forced = riichiActive[who];
      let state = "";
      if (forced) {
        const kind = isAka(tile) ? "・赤ドラ" : doraTypeList().includes(tileType(tile)) ? "・ドラ" : "";
        state = `（リーチ後${kind}）`;
      }
      const note = !forced && danger && (danger.level === "危険度高" || danger.level === "危険度中")
        ? `  ${danger.level}(${danger.seats.map(P).join(",")}リーチ)`
        : "";
      out.push(`${P(who)} ${jmMark}${tileGlyph(tile, aka)}${rin} ${TSUMOGIRI_MARK}${state}${note}${highDanger ? " " + DISCARD_MARK : ""}`);
      if (showAll) for (let s = 0; s < 4; s++) out.push(handLine(s));
      return;
    }

    // A dangerous discard is only a "push" worth a push/fold anchor when the
    // discarder actually has a hand to defend (tenpai or 1-shanten). Far-from-
    // tenpai players discarding into a riichi are folding/forced — just tag it.
    const isDanger = !!danger && danger.seats.length > 0 &&
      (danger.level === "危険度高" || danger.level === "危険度中");
    const isPush = isDanger && restShanten[who] <= 1;
    const dangerSeats = isDanger ? danger!.seats.map(P).join(",") : "";

    // build the fact line (a chosen discard from hand)
    const star = advanced || riichi || highDanger || isPush ? " " + DISCARD_MARK : "";
    const flagTxt = riichi ? `(${junme}巡目リーチ宣言・横向き)` : "";
    const tgMark = tsumogiri && !riichi ? ` ${TSUMOGIRI_MARK}` : "";
    const prefix = lead !== undefined
      ? `${lead}  → `
      : `${P(who)} ${jmMark}${drawn >= 0 ? `${tileGlyph(drawn, aka)}${rin} → ` : ""}`;
    const inlineDanger = isDanger && !isPush ? `  ${danger!.level}(${dangerSeats}リーチ)` : "";
    out.push(`${prefix}${tileGlyph(tile, aka)}${flagTxt}${tgMark}  ${metricTag(who)}${inlineDanger}${star}`);

    // reconstructed-hand displays at key beats
    if (riichi) {
      const info = restInfo(who);
      const waits = info.types.map((t) => typeGlyph(t)).join("") || "?";
      out.push(handLine(who, `待ち: ${waits}`));
      out.push(anchor(`${P(who)}のリーチ判断と待ちの良し悪し（打点・待ち枚数・巡目）`));
    } else if (isPush) {
      const st = restShanten[who] <= 0 ? "聴牌" : `向聴${restShanten[who]}`;
      const adv = advanced ? `・${tileGlyph(drawn, aka)}ツモで${before}→${after}前進` : "";
      out.push(handLine(who, `${danger!.level}(${dangerSeats}リーチ) 自分${st}${adv} ← 押し`));
      out.push(anchor(`${P(who)}の押し引き（自分の手牌価値 vs リーチの脅威）`));
    } else if (advanced) {
      out.push(handLine(who, `(${tileGlyph(drawn, aka)}ツモで向聴${before}→${after})`));
    } else if (showAll) {
      for (let s = 0; s < 4; s++) out.push(handLine(s));
    }
  }

  function applyMeld(m: Meld): void {
    if (m.kind === "shouminkan") {
      // upgrade the existing pon of this type to a kan
      const tt = tileType(m.calledTile);
      const idx = melds[m.who].findIndex((x) => x.kind === "pon" && tileType(x.tiles[0]) === tt);
      if (idx >= 0) melds[m.who].splice(idx, 1);
      const i = hands[m.who].indexOf(m.calledTile);
      if (i >= 0) hands[m.who].splice(i, 1);
      melds[m.who].push(m);
    } else {
      for (const t of concealedTilesUsed(m)) {
        const i = hands[m.who].indexOf(t);
        if (i >= 0) hands[m.who].splice(i, 1);
      }
      melds[m.who].push(m);
    }
    for (const t of m.tiles) publicVisible[tileType(t)]++;
    restShanten[m.who] = shanten(
      countsFromTiles(hands[m.who]),
      melds[m.who].length,
      false,
    );
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

function anchor(topic: string): string {
  return `〔解説ポイント: ${topic}〕`;
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
  out.push(`  和了手: ${renderHand(res.hand, res.melds, aka)}  （和了牌 ${tileGlyph(res.machi, aka)}）`);

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
    `  役: ${[...yakuParts, ...extra].join(" ")}  ${res.fu ? res.fu + "符" : ""}${res.points}点${lim ? " " + lim : ""}`,
  );
  if (res.doraHai.length) {
    out.push(`  ドラ表示: ${res.doraHai.map((t) => tileGlyph(t, aka)).join("")}` +
      (res.uraDoraHai.length ? `  裏ドラ表示: ${res.uraDoraHai.map((t) => tileGlyph(t, aka)).join("")}` : ""));
  }
  out.push(`  点棒: ${scoreDeltaLine(res.sc)}`);
  out.push(anchor("決着の評価と局全体の総括（勝負の分かれ目）"));
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
  out.push(anchor("流局時の聴牌・ノーテンと点棒状況の評価"));
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

function renderOwari(g: Game, out: string[]): void {
  const o = g.owari!;
  out.push("=".repeat(48));
  out.push("◆終局");
  const rows = [];
  for (let s = 0; s < 4 && s * 2 + 1 < o.length; s++) {
    rows.push({ seat: s, score: o[s * 2] * 100, pt: o[s * 2 + 1] });
  }
  rows.sort((a, b) => b.score - a.score);
  rows.forEach((r, idx) => {
    out.push(`  ${idx + 1}位 ${P(r.seat)}(${g.players[r.seat].name})  ${r.score}点  (${r.pt >= 0 ? "+" : ""}${r.pt})`);
  });
  out.push(anchor("対局全体の総括（着順・打ち回しの評価）"));
  out.push("=".repeat(48));
}
