// The shipped champion, pinned.
//
// `weights/champion.json` is the ONE vector the project ships as its best seat
// (計算 calibrated in M10 + the M11 own-hand block). It is a baseline in the
// measurement sense: every later milestone is graded against it with
// `--ktune-b`, so a silent drift in how it plays — a default nudged, a merge
// order changed, a kernel rebuilt with contraction on — would move every
// number measured afterwards without anyone noticing. These fingerprints make
// the drift loud. A DELIBERATE change to the champion regenerates them, with
// the reason written here; see `computed_test.ts` for the discipline.
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

// M11 baseline (2026-08-23): computed-calibrated (tune-m10d) + hand block
// fitted on runs/hand/lane-k.jsonl, pushScale 6000 / evWeight 0.2.
//
// REGENERATED 2026-08-25 (epoch): the champion's own play did not change —
// the OPPONENTS did. The "h" seat was re-bound that day to a frozen copy of
// the default 計算 seat (`ai/frozen.ts`), so the three H seats these hanchan
// are played against are different players, and every stream moved. The
// champion.json vector itself is bit-identical to the 08-23 pin.
const PINNED: Record<number, string> = {
  101: "15600/33600/51100/19700#95fc878c",
  404: "43500/47200/15700/13600#3ceab9a1",
  505: "33300/17500/41600/27600#ef69d46c",
  606: "25000/17600/36900/40500#3fdda809",
  707: "38200/7700/44200/29900#285f37c5",
};

Deno.test("champion: the shipped vector carries all four sections", () => {
  const k = loadKtune(CHAMPION);
  assert(k.computed && k.hand, "champion.json must carry `computed` and `hand`");
  assertEquals(k.hand.pushScale, 6000);
  assertEquals(k.hand.evWeight, 0.2);
});

Deno.test("champion: whole-hanchan decision streams are pinned", () => {
  for (const [seed, want] of Object.entries(PINNED)) {
    assertEquals(fingerprint(Number(seed)), want, `種${seed}: 基準席の打牌が変わった`);
  }
});

Deno.test("champion: the hand block is live — dropping it changes the stream", () => {
  const k = loadKtune(CHAMPION);
  let diverged = 0;
  for (const seed of Object.keys(PINNED)) {
    if (fingerprint(Number(seed), { ...k, hand: undefined }) !== PINNED[Number(seed)]) diverged++;
  }
  assert(diverged > 0, "hand ブロックを外しても同一 — 消費に届いていない");
});
