// The 雀鬼流 禁じ手, as predicates.
//
// Tier A rules are exact readings of a published prohibition. Tier B rules
// approximate a judgment call — each states its approximation in the violation's
// `detail` so a disputed call can be argued with rather than merely trusted.
//
// Nothing here blocks a move. Every check runs after the action has committed.

import type { Meld, Tile } from "mjrender/model.ts";
import { countsFromTiles, shanten } from "mjrender/shanten.ts";
import { doraFromIndicatorType, rankOfType, suitOfType, tileType } from "mjrender/tiles.ts";
import {
  analyze,
  concealedYakuhaiPairs,
  confirmedYaku,
  isOkurikan,
  wouldChangeWait,
} from "../hand.ts";
import { PENALTY } from "../rules.ts";
import type { Table } from "../table.ts";
import {
  DRAGON_TYPES,
  GREEN_TYPES,
  isHonor,
  isYaochu,
  sujiTypes,
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
    if (shantenAfterDiscard(t, seat) <= 0) return null; // tenpai — allowed

    const ty = tileType(action.tile);
    // 赤5筒 may be cut before tenpai; only indicator dora is restricted here.
    if (cfg.akaIds.has(action.tile) && !isDoraTile(t, action.tile)) return null;
    if (!isDoraTile(t, action.tile)) return null;
    // 例外: an honor dora already twice in the rivers is spent.
    if (isHonor(ty) && tableVisible(t, ty) >= 2) return null;
    // 例外: オーラス conditions can justify it.
    if (isOrasu(t)) return null;
    return [{ detail: `不聴のままドラを切った (向聴${shantenAfterDiscard(t, seat)})` }];
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
      const called = action.t === "daiminkan" ? action.called : action.called;
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
    if (wouldChangeWait(t, seat, action.type, drawn)) {
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
    // The kan would have to be the fourth copy of a type we already hold three
    // of, and it must not change the wait (otherwise it was never available).
    const ty = tileType(drawn);
    const counts = countsFromTiles([...t.hands[seat], drawn]);
    if (counts[ty] < 4) return null;
    if (wouldChangeWait(t, seat, ty, drawn)) return null;
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

const katagariAgari: DojoRule = {
  id: "katagari",
  label: "片和了り",
  tier: "A",
  points: PENALTY.medium,
  hooks: ["on-riichi", "post-discard"],
  check(ctx) {
    const { t, seat, action, oracle } = ctx;
    if (action.t !== "discard") return null;
    const info = analyze(t, seat, null, oracle);
    if (!info.katagari) return null;
    // 例外: allowed when the yakuless side is 純カラ (no live copies left).
    const dead = info.waits
      .filter((w) => !info.ronnable.includes(w))
      .every((w) => 4 - tableVisible(t, w) - countsFromTiles(t.hands[seat])[w] <= 0);
    if (dead) return null;
    return [{
      detail: `待ち ${info.waits.join("/")} のうち役があるのは ${info.ronnable.join("/")}`,
    }];
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

    const out: Array<Partial<Violation>> = [];
    for (const s of SEATS) {
      if (s === seat) continue;
      const melds = t.melds[s];
      for (const w of YAKUMAN_WATCH) {
        if (!w.trigger(melds) || !w.related(melds).includes(ty)) continue;
        // 例外: you may cut it if you are yakuman-tenpai yourself. Approximated
        // as "tenpai with an all-yaochu or single-suit hand" — the exact test
        // needs the scorer, which the ledger deliberately does not depend on.
        const info = analyze(t, seat, null, oracle);
        const mine = [...t.hands[seat], ...t.melds[seat].flatMap((m) => m.tiles)].map(tileType);
        if (info.tenpai && (mine.every(isYaochu) || new Set(mine.map(suitOfType)).size === 1)) {
          continue;
        }
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

const atozuke: DojoRule = {
  id: "atozuke",
  label: "後付け",
  tier: "B",
  points: PENALTY.medium,
  hooks: ["on-call"],
  check(ctx) {
    const { t, seat, action, cfg } = ctx;
    if (action.t !== "pon" && action.t !== "chi") return null;
    const confirmed = confirmedYaku(
      t.hands[seat],
      t.melds[seat],
      t.valueHonors(seat),
      cfg.kuitan,
    );
    if (confirmed.length > 0) return null;
    const backs = concealedYakuhaiPairs(t.hands[seat], t.valueHonors(seat));
    if (backs.length === 0) return null;
    return [{
      detail: `確定役なしで鳴き、役牌 ${backs.join("/")} の暗刻待ち (バック)。` +
        `構造判定のため、稀に確定役を見落とすことがある`,
      confidence: 0.8,
    }];
  },
};

const koshi: DojoRule = {
  id: "koshi",
  label: "腰",
  tier: "B",
  points: PENALTY.light,
  hooks: ["on-call", "post-discard"],
  check(ctx) {
    const { t, seat, timing } = ctx;
    const ms = timing?.callPromptMs;
    if (ms === undefined || ms < ctx.dojo.koshiMs) return null;
    // The information leak the rule punishes is real here: the caller visibly
    // considered the tile. Block ron on it and its suji for the round.
    const tile = ctx.action.t === "pon" || ctx.action.t === "chi"
      ? tileType(ctx.action.called)
      : null;
    if (tile === null) return null;
    for (const ty of [tile, ...sujiTypes(tile)]) t.ronBlocked[seat].add(ty);
    return [{
      detail: `鳴きの判断に ${ms}ms 掛かった。当該牌とそのスジは出和了不可`,
      confidence: 0.7,
    }];
  },
};

const chouko: DojoRule = {
  id: "chouko",
  label: "長考",
  tier: "B",
  points: PENALTY.light,
  hooks: ["post-discard", "on-call", "on-kan", "on-riichi"],
  check(ctx) {
    const ms = ctx.timing?.elapsedMs;
    if (ms === undefined || ms <= ctx.dojo.thinkLimitMs) return null;
    const hard = ms > ctx.dojo.thinkHardMs;
    return [{
      label: hard ? "大長考" : "長考",
      points: hard ? PENALTY.medium : PENALTY.light,
      detail: `打牌に ${(ms / 1000).toFixed(1)}秒 掛かった (規範 ${
        ctx.dojo.thinkLimitMs / 1000
      }秒)`,
      confidence: 1,
    }];
  },
};

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
  hikkakeRiichi,
  yakumanRelated,
  underEightThousand,
  atozuke,
  koshi,
  chouko,
  ukiCrush,
  misehai,
];

export const runHook = makeRunner(RULES);
export type { RuleCtx };
