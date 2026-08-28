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
const PINNED: Record<number, string> = {
  101: "44800/35400/26100/13700#3af23773",
  404: "33900/33300/26100/26700#8828a996",
  505: "45500/25700/23300/25500#06404d89",
  606: "19100/39300/22100/39500#7b409038",
  707: "66100/4800/29300/19800#6edf323d",
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
});

Deno.test("champion: whole-hanchan decision streams are pinned", () => {
  for (const [seed, want] of Object.entries(PINNED)) {
    assertEquals(fingerprint(Number(seed)), want, `種${seed}: 基準席の打牌が変わった`);
  }
});
