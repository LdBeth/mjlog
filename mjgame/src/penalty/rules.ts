// The 雀鬼流 禁じ手, as predicates.
//
// Tier A rules are exact readings of a published prohibition. Tier B rules
// approximate a judgment call — each states its approximation in the violation's
// `detail` so a disputed call can be argued with rather than merely trusted.
//
// Nothing here blocks a move. Every check runs after the action has committed.

import type { Meld, Tile } from "mjrender/model.ts";
import { doraFromIndicatorType, rankOfType, suitOfType, tileType } from "mjrender/tiles.ts";
import { countsFromTiles, shanten } from "../kernel.ts";
import type { TenpaiInfo } from "../hand.ts";
import { analyze, isOkurikan, wouldChangeWait } from "../hand.ts";
import { PENALTY } from "../rules.ts";
import type { Table } from "../table.ts";
import {
  DRAGON_TYPES,
  GREEN_TYPES,
  isHonor,
  isYaochu,
  TERMINAL_TYPES,
  WIND_TYPES,
} from "../tiles.ts";
import type { Seat, Violation } from "../types.ts";
import { SEATS } from "../types.ts";
import type { DojoRule, RuleCtx } from "./mod.ts";
import { makeRunner } from "./mod.ts";

// --- shared helpers ---------------------------------------------------------

/** Copies of a type visible to everyone EXCLUDING the observer's own hand. */
function tableVisible(t: Table, type: number): number {
  let n = 0;
  for (const s of SEATS) {
    for (const r of t.board.rivers[s]) {
      if (r.calledBy === undefined && tileType(r.tile) === type) n++;
    }
    for (const m of t.melds[s]) {
      for (const x of m.tiles) if (tileType(x) === type) n++;
    }
  }
  for (const i of t.indicators) if (tileType(i) === type) n++;
  return n;
}

function isDoraTile(t: Table, tile: Tile): boolean {
  const ty = tileType(tile);
  return t.indicators.some((i) => doraFromIndicatorType(tileType(i)) === ty);
}

function shantenAfterDiscard(t: Table, seat: Seat): number {
  return shanten(
    countsFromTiles(t.hands[seat]),
    t.melds[seat].length,
    t.isMenzen(seat),
  );
}

/** Last kyoku of the match — where several dojo exceptions kick in. */
function isOrasu(t: Table): boolean {
  return t.kyoku >= (t.cfg.hanchan ? 7 : 3);
}

function tripletTypes(melds: readonly Meld[]): number[] {
  return melds
    .filter((m) => m.kind !== "chi")
    .map((m) => tileType(m.tiles[0]));
}

/**
 * Ask a question of the winner's PRE-WIN hand — the shape it was waiting with
 * one tile ago, which is what the 後付け family is actually about.
 *
 * On a ron the table already holds exactly that: the winning tile is in the
 * discarder's river, never in the hand. On a tsumo the drawn tile IS in the
 * hand, and handing `analyze` a `drawn` argument is not enough — `analyze`
 * derives `ronnable` from `oracle.hasYaku`, which reads `t.hands` directly
 * (score.ts::buildContext) and would score a 15-tile hand. So the tile comes
 * out of the table for the duration of the question and goes back in the same
 * tick, before anything else can observe the table.
 *
 * Returns null when the pre-win hand cannot be recovered (a tsumo whose drawn
 * tile the hook was not given) — silence beats charging off a mis-sized hand.
 */
function onPreWinHand<T>(
  ctx: RuleCtx,
  ask: (info: TenpaiInfo, hand: Tile[]) => T | null,
): T | null {
  const { t, seat, action, drawn, oracle } = ctx;
  const hand = t.hands[seat];
  let cut = -1;
  if (action.t === "tsumo") {
    cut = drawn === null ? -1 : hand.lastIndexOf(drawn);
    if (cut < 0) return null;
    hand.splice(cut, 1);
  }
  try {
    return ask(analyze(t, seat, null, oracle), [...hand]);
  } finally {
    if (cut >= 0) hand.splice(cut, 0, drawn!);
  }
}

// --- Tier A -----------------------------------------------------------------

const firstHonor: DojoRule = {
  id: "first-honor",
  label: "第一打字牌切り",
  tier: "A",
  points: PENALTY.light,
  hooks: ["post-discard"],
  check(ctx) {
    const { t, seat, action } = ctx;
    if (action.t !== "discard") return null;
    if (t.board.rivers[seat].length !== 1) return null;
    if (!isHonor(tileType(action.tile))) return null;
    // 例外: it is a double-riichi final form.
    if (action.riichi && t.doubleRiichi[seat]) return null;
    return [{ detail: `第一打に字牌 ${tileType(action.tile)} を切った` }];
  },
};

const notenDora: DojoRule = {
  id: "noten-dora",
  label: "不聴時ドラ切り",
  tier: "A",
  points: PENALTY.medium,
  hooks: ["post-discard"],
  check(ctx) {
    const { t, seat, action, cfg } = ctx;
    if (action.t !== "discard") return null;
    const sh = shantenAfterDiscard(t, seat);
    if (sh <= 2) return null; // 2向聴以内 — allowed

    const ty = tileType(action.tile);
    // 赤5筒 may be cut before tenpai; only indicator dora is restricted here.
    if (cfg.akaIds.has(action.tile) && !isDoraTile(t, action.tile)) return null;
    if (!isDoraTile(t, action.tile)) return null;
    // 例外: an honor dora already twice in the rivers is spent.
    if (isHonor(ty) && tableVisible(t, ty) >= 2) return null;
    // 例外: オーラス conditions can justify it.
    if (isOrasu(t)) return null;
    return [{ detail: `不聴のままドラを切った (向聴${sh})` }];
  },
};

const doraCalledLock: DojoRule = {
  id: "dora-pon-lock",
  label: "ドラ切り後の手出し",
  tier: "A",
  points: PENALTY.medium,
  hooks: ["on-call", "post-discard"],
  check(ctx) {
    const { t, seat, action } = ctx;
    // Arming half: a pre-tenpai dora discard that gets called locks the
    // discarder into tsumogiri for the rest of the round. This is the one rule
    // that carries state, so it sets the flag on the table itself.
    if (action.t === "pon" || action.t === "chi" || action.t === "daiminkan") {
      const called = action.called;
      const river = t.board.rivers;
      for (const s of SEATS) {
        const last = river[s][river[s].length - 1];
        if (last?.tile === called && isDoraTile(t, called)) {
          if (shanten(countsFromTiles(t.hands[s]), t.melds[s].length, t.isMenzen(s)) > 0) {
            t.tsumogiriLock[s] = true;
          }
        }
      }
      return null;
    }
    if (action.t !== "discard") return null;
    if (!t.tsumogiriLock[seat] || action.tsumogiri) return null;
    return [{ detail: "ドラ切りをポンされた後に手出しした" }];
  },
};

const minkan: DojoRule = {
  id: "minkan",
  label: "明槓",
  tier: "A",
  points: PENALTY.medium,
  hooks: ["on-kan", "on-call"],
  check(ctx) {
    const { action } = ctx;
    if (action.t === "daiminkan") return [{ detail: "大明槓は禁止" }];
    if (action.t === "kakan") return [{ detail: "加槓は禁止" }];
    return null;
  },
};

const ankanForm: DojoRule = {
  id: "ankan-form",
  label: "暗槓条件違反",
  tier: "A",
  points: PENALTY.medium,
  hooks: ["on-kan"],
  check(ctx) {
    const { t, seat, action, drawn, oracle } = ctx;
    if (action.t !== "ankan") return null;
    const out: Array<Partial<Violation>> = [];

    if (!t.isMenzen(seat)) out.push({ detail: "門前でない状態での暗槓" });
    const before = analyze(t, seat, drawn, oracle);
    if (!before.tenpai && !t.riichi[seat]) out.push({ detail: "聴牌していない状態での暗槓" });
    if (isOkurikan(action.type, drawn)) out.push({ detail: "送りカン" });
    // This hook fires after `round.ts` has emitted the meld, so the kan is
    // normally already among `t.melds[seat]` — but the same rule is reachable
    // from a pre-commit caller, so ask the table rather than assume.
    const laid = t.melds[seat].some(
      (m) => m.kind === "ankan" && tileType(m.tiles[0]) === action.type,
    );
    if (wouldChangeWait(t, seat, action.type, laid)) {
      out.push({ detail: "テンパイが変わるカン" });
    }
    return out.length ? out : null;
  },
};

const riichiKanSkip: DojoRule = {
  id: "riichi-kan-skip",
  label: "立直後カン見送り",
  tier: "A",
  points: PENALTY.medium,
  hooks: ["post-discard"],
  check(ctx) {
    const { t, seat, action, drawn } = ctx;
    if (action.t !== "discard" || !t.riichi[seat] || drawn === null) return null;
    // Not the declaring discard. `t.riichi` is set before this hook runs, so a
    // declaration already looks like a riichi turn here — but that discard was
    // chosen while riichi did not yet exist, so the rule cannot apply, whether
    // the declaration cut the drawn tile or a held one.
    if (action.riichi) return null;
    // Only a tsumogiri can have passed up a kan: a tedashi leaves `drawn` in
    // the hand, so the reconstruction below would count it twice and invent a
    // fourth copy that was never there.
    if (!action.tsumogiri) return null;
    // The kan would have to be the fourth copy of a type we already hold three
    // of, and it must not change the wait (otherwise it was never available).
    const ty = tileType(drawn);
    const counts = countsFromTiles([...t.hands[seat], drawn]);
    if (counts[ty] < 4) return null;
    // Nothing has been melded — this is the discard that passed the kan up.
    if (wouldChangeWait(t, seat, ty)) return null;
    return [{ detail: "立直中にカンできる牌をツモってカンしなかった" }];
  },
};

const hadakaTanki: DojoRule = {
  id: "hadaka-tanki",
  label: "裸単騎",
  tier: "A",
  points: PENALTY.medium,
  hooks: ["post-discard"],
  check(ctx) {
    const { t, seat, action, oracle } = ctx;
    if (action.t !== "discard") return null;
    if (t.melds[seat].filter((m) => m.kind !== "ankan").length < 4) return null;
    const info = analyze(t, seat, null, oracle);
    if (!info.tenpai) return null;
    // 例外: a confirmed yakuman (字一色 / 清老頭 etc.) makes it 勇気, not 無謀.
    const all = [...t.hands[seat], ...t.melds[seat].flatMap((m) => m.tiles)].map(tileType);
    if (all.every(isHonor) || all.every((ty) => TERMINAL_TYPES.includes(ty))) return null;
    return [{ detail: "4副露の裸単騎" }];
  },
};

const jigokuTanki: DojoRule = {
  id: "jigoku-tanki",
  label: "地獄単騎立直",
  tier: "A",
  points: PENALTY.heavy,
  hooks: ["on-riichi"],
  check(ctx) {
    const { t, seat, oracle } = ctx;
    const info = analyze(t, seat, null, oracle);
    if (info.waits.length !== 1) return null;
    const w = info.waits[0];
    if (!isHonor(w)) return null;
    const gone = tableVisible(t, w);
    if (gone < 2) return null;
    return [{ detail: `${gone}枚切れの字牌単騎での立直 (出和了偏重)` }];
  },
};

const suuankouRiichi: DojoRule = {
  id: "suuankou-riichi",
  label: "一手変わり四暗刻での立直",
  tier: "A",
  points: PENALTY.medium,
  hooks: ["on-riichi"],
  check(ctx) {
    const { t, seat, oracle } = ctx;
    if (!t.isMenzen(seat)) return null;
    const counts = countsFromTiles(t.hands[seat]);
    const ankou = counts.filter((n) => n >= 3).length;
    if (ankou < 3) return null;
    const info = analyze(t, seat, null, oracle);
    // 例外: 目に見えて空テン — the yakuman tile is already all gone.
    const live = info.waits.some((w) => 4 - tableVisible(t, w) > 0);
    if (!live) return null;
    return [{ detail: `暗刻${ankou}組、四暗刻への手替わりを残して立直した` }];
  },
};

const bigThreatRiichi: DojoRule = {
  id: "yakuman-threat-riichi",
  label: "役満模様への立直",
  tier: "A",
  points: PENALTY.heavy,
  hooks: ["on-riichi"],
  check(ctx) {
    const { t, seat } = ctx;
    for (const s of SEATS) {
      if (s === seat) continue;
      const trips = tripletTypes(t.melds[s]);
      const dragons = trips.filter((ty) => DRAGON_TYPES.includes(ty)).length;
      const winds = trips.filter((ty) => WIND_TYPES.includes(ty)).length;
      if (dragons >= 2) return [{ detail: `P${s} が三元牌を2組鳴いている` }];
      if (winds >= 2) return [{ detail: `P${s} が風牌を2組鳴いている` }];
    }
    return null;
  },
};

/**
 * 片和了り, judged at the WIN.
 *
 * A split wait — some winning tiles carry a yaku, some do not — is not itself a
 * foul. Sitting in one is a shape, and a shape can still be pushed off or
 * folded; what the dojo objects to is CASHING it. So the rule asks its question
 * of the hand as it stood one tile ago, at the moment the seat declared.
 *
 * 例外:
 *   (a) 門前ツモ — 門前清自摸和 is a yaku on every winning tile, so the split
 *       never decided anything. (立直 needs no clause: it too puts a yaku on
 *       every wait, so `analyze` reports `katagari: false` for a riichi hand and
 *       the rule never reaches its test.)
 *   (b) 純カラ — the yakuless side has no live copy left, so there was nothing
 *       one-sided to exploit.
 * A RON on the scoring side charges whether the hand was 門前 or not: the seat
 * sat on a tile it could not use and took the one it could. An OPEN 自摸 on such
 * a wait charges for the same reason.
 */
const katagariAgari: DojoRule = {
  id: "katagari",
  label: "片和了り",
  tier: "A",
  points: PENALTY.medium,
  hooks: ["on-win"],
  check(ctx) {
    const { t, seat, action, drawn } = ctx;
    if (action.t !== "ron" && action.t !== "tsumo") return null;
    if (action.t === "tsumo" && t.isMenzen(seat)) return null; // 例外(a)
    return onPreWinHand(ctx, (info, hand) => {
      if (!info.katagari) return null;
      // 例外(b): the yakuless side is 純カラ (no live copies left).
      const counts = countsFromTiles(hand);
      const dead = info.waits
        .filter((w) => !info.ronnable.includes(w))
        .every((w) => 4 - tableVisible(t, w) - counts[w] <= 0);
      if (dead) return null;
      const waits = info.waits.join("/");
      const yaku = info.ronnable.join("/");
      return [{
        detail: action.t === "tsumo"
          ? `片和了りの待ち ${waits} (役があるのは ${yaku}) のまま ${tileType(drawn!)} でツモ和了`
          : `片和了りの待ち ${waits} のうち役のある ${yaku} でロン和了`,
      }];
    });
  },
};

/**
 * 後付け, judged at the WIN.
 *
 * The dojo test this implements is "the hand plus any one of its winning tiles
 * must carry a yaku". A tenpai where SOME waits score is 片和了り and belongs to
 * the rule above; a tenpai where NO wait scores at all is 後付け — the hand is
 * playing for a yaku it does not have.
 *
 * Nothing is charged while such a hand merely SITS there. An open 形式聴牌 is a
 * legitimate shape (it collects 聴牌料 at an exhaustive draw), and a seat that
 * folds it or never completes it has taken nothing. The foul is the win.
 *
 * Where a win with no scoring wait can come from, honestly stated:
 *   - The ledger's oracle IS the real scorer (main.ts wiring). `legal.ts` never
 *     offers a ron with no scoring wait, so what reaches this hook is a TSUMO
 *     whose only yaku is one the wait never carried — 海底摸月, 嶺上開花 — taken
 *     on an open hand. That is 後付け in its purest form, and it is charged.
 *   - The round ran under a permissive oracle (self-play harnesses, fixtures)
 *     while the ledger judges with the real scorer: the yakuless win goes
 *     through and is charged here.
 *   - The ledger's own oracle is the placeholder `ANY_WIN`: every wait is
 *     ronnable, `ronnable.length === 0` is never true, and the rule cannot fire.
 *     That is the intended degradation — with no scorer wired in there is no
 *     evidence, and the ledger does not guess. (The predecessor state-time rule
 *     went silent under `ANY_WIN` for the same reason; only the moment of the
 *     question moved.)
 *
 * 門前 is exempt on both counts: 立直 certifies every wait, and 門前ツモ is itself
 * the yaku the hand was missing. No dedupe clause is needed — unlike the
 * state-time version this replaces, a seat wins at most once per round.
 *
 * The 純カラ exception the state-time version carried is gone with it: a hand
 * that just won on a wait has proved the wait was live, and on a ron the tile is
 * already in the river, where `tableVisible` would count it against itself.
 */
const atozuke: DojoRule = {
  id: "atozuke",
  label: "後付け",
  tier: "A",
  points: PENALTY.medium,
  hooks: ["on-win"],
  check(ctx) {
    const { t, seat, action } = ctx;
    if (action.t !== "ron" && action.t !== "tsumo") return null;
    if (t.isMenzen(seat)) return null;
    return onPreWinHand(ctx, (info) => {
      if (!info.tenpai) return null;
      // Some wait scored ⇒ 片和了り, which the rule above charges for.
      if (info.ronnable.length > 0) return null;
      return [{
        detail: `どの待ちにも役の無い副露手 (待ち ${info.waits.join("/")}) のまま和了した`,
      }];
    });
  },
};

const hikkakeRiichi: DojoRule = {
  id: "hikkake",
  label: "即引っかけ立直",
  tier: "A",
  points: PENALTY.heavy,
  hooks: ["on-riichi"],
  check(ctx) {
    const { t, seat, action, oracle } = ctx;
    if (action.t !== "discard") return null;
    const d = tileType(action.tile);
    if (isHonor(d)) return null;
    const info = analyze(t, seat, null, oracle);
    // The declaring tile suji-traps our own wait: exactly what the rule bans.
    const trapped = info.waits.filter(
      (w) => suitOfType(w) === suitOfType(d) && Math.abs(rankOfType(w) - rankOfType(d)) === 3,
    );
    if (trapped.length === 0) return null;
    if (isOrasu(t)) return null; // オーラス条件は例外
    if (t.doubleRiichi[seat]) return null; // ダブリー最終形は例外
    return [{
      detail: `宣言牌のスジ ${trapped.join("/")} が自分の待ち (1巡置いてから立直すべき)`,
    }];
  },
};

/**
 * Yakuman watches, keyed purely on opponents' OPEN melds. Note the asymmetric
 * threshold the sources are explicit about: 大三元 fires from 2 melds, every
 * other watch from 3.
 */
const YAKUMAN_WATCH: Array<{
  name: string;
  trigger: (m: readonly Meld[]) => boolean;
  related: (m: readonly Meld[]) => number[];
}> = [
  {
    name: "大三元",
    trigger: (m) => tripletTypes(m).filter((ty) => DRAGON_TYPES.includes(ty)).length >= 2,
    related: (m) => DRAGON_TYPES.filter((ty) => !tripletTypes(m).includes(ty)),
  },
  {
    name: "字一色",
    trigger: (m) => m.length >= 3 && m.every((x) => x.tiles.every((t) => isHonor(tileType(t)))),
    related: () => [...WIND_TYPES, ...DRAGON_TYPES],
  },
  {
    name: "四喜和",
    trigger: (m) =>
      m.length >= 3 && tripletTypes(m).filter((ty) => WIND_TYPES.includes(ty)).length >= 2,
    related: (m) => WIND_TYPES.filter((ty) => !tripletTypes(m).includes(ty)),
  },
  {
    name: "清老頭",
    trigger: (m) =>
      m.length >= 3 &&
      m.every((x) => x.kind !== "chi" && TERMINAL_TYPES.includes(tileType(x.tiles[0]))),
    related: () => [...TERMINAL_TYPES],
  },
  {
    name: "緑一色",
    trigger: (m) =>
      m.length >= 3 && m.every((x) => x.tiles.every((t) => GREEN_TYPES.includes(tileType(t)))),
    related: () => [...GREEN_TYPES],
  },
];

const yakumanRelated: DojoRule = {
  id: "yakuman-related",
  label: "役満関連牌切り",
  tier: "A",
  points: PENALTY.heavy,
  hooks: ["post-discard"],
  check(ctx) {
    const { t, seat, action, oracle } = ctx;
    if (action.t !== "discard") return null;
    const ty = tileType(action.tile);

    // 例外: you may cut it if you are yakuman-tenpai yourself. Approximated as
    // "tenpai with an all-yaochu or single-suit hand" — the exact test needs the
    // scorer, which the ledger deliberately does not depend on. The question is
    // about OUR hand alone, so it does not vary with the watched seat: ask it at
    // most once, and only if some watch actually triggered.
    let exempt: boolean | null = null;
    const selfYakumanTenpai = (): boolean => {
      if (exempt === null) {
        const info = analyze(t, seat, null, oracle);
        const mine = [...t.hands[seat], ...t.melds[seat].flatMap((m) => m.tiles)].map(tileType);
        exempt = info.tenpai &&
          (mine.every(isYaochu) || new Set(mine.map(suitOfType)).size === 1);
      }
      return exempt;
    };

    const out: Array<Partial<Violation>> = [];
    for (const s of SEATS) {
      if (s === seat) continue;
      const melds = t.melds[s];
      for (const w of YAKUMAN_WATCH) {
        if (!w.trigger(melds) || !w.related(melds).includes(ty)) continue;
        if (selfYakumanTenpai()) continue;
        out.push({ detail: `P${s} の${w.name}模様に対して関連牌を切った` });
      }
    }
    return out.length ? out : null;
  },
};

const underEightThousand: DojoRule = {
  id: "under-8000",
  label: "持ち点8000点未満",
  tier: "A",
  points: PENALTY.medium,
  hooks: ["on-round-end"],
  check(ctx) {
    const { t, seat } = ctx;
    if (t.scores[seat] >= 8000) return null;
    return [{ detail: `局終了時の持ち点 ${t.scores[seat]}点` }];
  },
};

// --- Tier B -----------------------------------------------------------------

// 長考 (`chouko`) and 腰 (`koshi`) used to live here. Removed 2026-08-23: both
// priced wall-clock hesitation, which only a human at the TUI could ever
// produce, and a keyboard interface cannot meet a physical table's 3秒/1.2秒
// norms. See CLAUDE.md before considering a re-add.

const ukiCrush: DojoRule = {
  id: "uki-crush",
  label: "一人浮きにする和了り",
  tier: "B",
  points: PENALTY.medium,
  hooks: ["on-win"],
  check(ctx) {
    const { t, seat } = ctx;
    if (t.kyoku < 5) return null; // 南2局以降
    const line = t.cfg.returnScore;
    const before = t.scores.filter((s) => s > line).length;
    if (before < 2) return null;
    if (t.scores[seat] >= line) return null; // only binds the low-scoring player
    return [{
      detail: `${before}人浮きを崩す和了り (南2局以降、持ち点${t.scores[seat]}点)。` +
        `浮きの基準を${line}点とした解釈`,
      confidence: 0.6,
    }];
  },
};

const misehai: DojoRule = {
  id: "misehai",
  label: "見せ牌",
  tier: "B",
  points: PENALTY.light,
  hooks: ["on-call"],
  check(ctx) {
    const { t, seat } = ctx;
    if (t.exposed[seat].size === 0) return null;
    // Exposing a tile forfeits 出和了 on its whole suit (honors: that tile only).
    for (const id of t.exposed[seat]) {
      const ty = tileType(id);
      if (isHonor(ty)) t.ronBlocked[seat].add(ty);
      else {
        const suit = suitOfType(ty);
        for (let x = 0; x < 27; x++) if (suitOfType(x) === suit) t.ronBlocked[seat].add(x);
      }
    }
    return [{
      detail: "牌を見せたため、その色は出和了不可",
      confidence: 1,
    }];
  },
};

// ---------------------------------------------------------------------------

export const RULES: DojoRule[] = [
  firstHonor,
  notenDora,
  doraCalledLock,
  minkan,
  ankanForm,
  riichiKanSkip,
  hadakaTanki,
  jigokuTanki,
  suuankouRiichi,
  bigThreatRiichi,
  katagariAgari,
  atozuke,
  hikkakeRiichi,
  yakumanRelated,
  underEightThousand,
  ukiCrush,
  misehai,
];

export const runHook = makeRunner(RULES);
export type { RuleCtx };
