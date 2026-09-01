// The shipped champion, pinned.
//
// `weights/champion.json` is BY CONVENTION the current champion — the ONE
// vector the project ships as its best seat, and the one the TUI 助言 advisor
// reads. It is a baseline in the measurement sense: every later milestone is
// graded against it with `--ktune-b`, so a silent drift in how it plays — a
// default nudged, a merge order changed, a kernel rebuilt with contraction on
// — would move every number measured afterwards without anyone noticing.
// These fingerprints make the drift loud. A DELIBERATE change to the champion
// (a promotion under controlled paired evidence) regenerates them, with the
// reason written here; see `computed_test.ts` for the discipline.
//
// Built through `makePolicy`, not by hand: the pin covers the loader, the
// merge of the four sections and the seat wiring, which is what a CLI run uses.

import { assert, assertEquals } from "@std/assert";
import { loadKtune, makePolicy } from "../src/harness.ts";
import { playHanchan } from "./helpers.ts";

const CHAMPION = new URL("../weights/champion.json", import.meta.url).pathname;

function fingerprint(seed: number, ktune = loadKtune(CHAMPION)): string {
  const r = playHanchan(seed, (s) => {
    const p = makePolicy({
      kind: s === 0 ? "k" : "h",
      name: s === 0 ? "K0" : `H${s}`,
      seed: seed * 4 + s,
      ktune: s === 0 ? ktune : undefined,
    });
    return p.policy;
  });
  const body = JSON.stringify({
    scores: r.scores,
    outcomes: r.outcomes,
    ledger: r.ledger.map((v) => `${v.seat}:${v.label}`),
    riichis: r.riichis,
    furo: r.furoRounds,
  });
  let h = 0x811c9dc5;
  for (let i = 0; i < body.length; i++) h = Math.imul(h ^ body.charCodeAt(i), 0x01000193) >>> 0;
  return `${r.scores.join("/")}#${h.toString(16).padStart(8, "0")}`;
}

// RE-CAPTURED 2026-08-28 (owner-directed, shared-code bug fix): the danger
// assessor was being fed the discarder's own tiles TWICE — `observe.ts`
// passed the own-inclusive `Table.visibleCounts` as the public count AND
// `ownCounts` — so a held pair rated 安全 through the killed-shapes cap. Fixed
// to mjrender's contract (public counts + hand). Found from the ranked arena
// wire log (a 12,000 単騎 ron on an East held as a pair). The champion's
// weights are untouched; every seat's world changed, so every pin file was
// re-captured the same day. The retired 08-27 pins ended
// 101:#3af23773 404:#8828a996 505:#06404d89 606:#7b409038 707:#6edf323d.
//
// REGENERATED 2026-08-27 (owner rulings): two things moved at once. The dojo
// rules correction (under-8000 judged at 終局 with the buffer 南入以降; call
// gate 対々和/バック tightening — shared code, every pin file re-captured that
// day), and the 最終形リーチ doctrine head added to champion.json:
// {bias 0.1, holdShape −1, tenpaiHeld 0.5} — immediate riichi for >2 live
// acceptance with more hand than riichi(+平和)のみ (M11 value model prices
// the cheapness) and for the sanctioned 単騎 (ドラ/七対子/四暗刻/国士);
// everything else holds ~2 unimproved turns. Graded vs the pre-head
// incumbent on the frozen-h field (300 paired games): 道場順位差 +0.037
// ±0.054 (CI spans zero), riichi rate 24.0%→21.5%, violations flat — the
// doctrine ships by the owner's ruling, at a measured cost of ≈nil, and beat
// every cell of the earlier improvable-based 4-cell sweep.
//
// REGENERATED 2026-08-25 (promotion): the post-epoch sweep re-grade removed
// the M11 hand block. Under the pre-registered rule (道場順位差 negative, 95%
// CI clear of zero on both lanes) NO (pushScale, evWeight) cell qualified:
// both the 08-23 fit and a fresh refit against the frozen field measured
// positive (worse) in all 31 completed cells, monotone in pushScale — a
// structural transfer defect (a single scalar cannot match the push table's
// shanten-shape), not a prediction defect. The champion is now the M10
// computed calibration alone: champion.json ≡ computed-calibrated.json as of
// this date. The retired 08-23 champion pins ended
// 101:#95fc878c 404:#3ceab9a1 505:#ef69d46c 606:#3fdda809 707:#285f37c5.
// Re-captured 2026-08-30 — PROMOTION by the owner's word ("go ahead and
// promote 0.25"): the M14 learned deal-in reads (`dealin` block, weights/
// dealin-0829.json: P(ron | opp, tile) head 54→32→32→1 + tenpai head
// 22→16→1) and `augment.floor` 0.25 join the champion. Evidence: 50 ranked
// riichi.dev games (535 局) vs the pre-M14 arena baseline — 放銃率 10.8% vs
// 13.9%, 平均放銃打点 4,924 vs 7,871, 満貫以上の放銃 21% vs 59% of feeds,
// 放銃失点/局 534 vs 1,097; cost 和了率 22.8% vs 28.7%; 平均順位 2.56 vs 2.67
// (neutral at n=50). Home paired grades at floor 0.5/0.25/0 were all inside
// noise; 0.25 is the arena-tested floor. Frozen "h" and the league pins are
// UNTOUCHED — seat 0 no longer equals seats 1-3. The retired 08-29 pins ended
// 101:#4ca96e4f 404:#a66c8f04 505:#3941f846 606:#f077b710 707:#eb7da2d4.
//
// Re-captured 2026-08-29 — PROMOTION by the owner's word after a stable ~1600
// rating on riichi.dev ranked play: champion.json is now the arena vector
// minus its arena-only buffer overrides (M10 computed calibration + 最終形
// riichi head + 色読み sense trio + liveYakuhai 200 + keepTriplet 1). The
// same day the frozen "h" seat was re-bound to this champion (see
// frozen_test), so seat 0 here plays the same seat as seats 1-3 — the scores
// below equal frozen_test's EPOCH_PIN scores on the shared seeds by
// construction (the hashes differ: this body carries `furo`).
const PINNED: Record<number, string> = {
  101: "41600/41400/29300/7700#0fbe0d47",
  404: "23800/46400/14800/35000#7c3e8a5c",
  505: "57800/33000/11600/17600#de6d9435",
  606: "17000/24800/32300/45900#432c9a52",
  707: "28000/18900/39000/34100#4a402e07",
};

Deno.test("champion: the computed calibration, and NO hand block", () => {
  const k = loadKtune(CHAMPION);
  assert(k.computed, "champion.json must carry `computed`");
  assert(k.riichi, "champion.json must carry the 最終形 doctrine `riichi` block (2026-08-27)");
  // The 2026-08-25 sweep verdict: an own-hand ev spent through the fold gate's
  // single scalar measured WORSE than the incumbent push table at every tested
  // authority, however well calibrated. A hand block reappearing here would
  // silently re-enable that — it takes new controlled paired evidence, not a
  // merge accident.
  assertEquals(k.hand, undefined, "hand ブロックは 2026-08-25 に除去済み (sweep 判定)");
  // M13's fold head is a CANDIDATE, not the champion: promoting it takes the
  // pre-registered paired grade (道場順位差 negative, 95% CI clear of zero) and
  // the owner's word. A block appearing here by merge accident would silently
  // re-crown an ungraded seat, exactly as a `hand` block would.
  assertEquals(k.fold, undefined, "fold ブロックは未昇格 (M13 は対戦評価と主の判断待ち)");
  assertEquals(k.ev, undefined, "ev ブロックは未昇格 (M15)");
  // M14 PROMOTED 2026-08-30: the learned deal-in read is the champion's price
  // of a tile. `augment.floor` (the rule ladder's floor under the estimate) is
  // the owner's separate decision — 0.25 is the arena-tested value; 0.5 and 0
  // were graded at home only. A change here is a promotion, not a merge.
  assert(k.dealin, "champion.json must carry the M14 `dealin` block (2026-08-30)");
  assertEquals(k.augment?.floor, 0.25, "augment.floor は主の裁定で 0.25 (2026-08-30)");
});

Deno.test("champion: whole-hanchan decision streams are pinned", () => {
  for (const [seed, want] of Object.entries(PINNED)) {
    assertEquals(fingerprint(Number(seed)), want, `種${seed}: 基準席の打牌が変わった`);
  }
});
