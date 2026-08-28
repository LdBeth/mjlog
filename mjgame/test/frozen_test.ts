// The frozen baseline seat — the 2026-08-25 epoch pin.
//
// THIS PIN NEVER REGENERATES. `ai/frozen.ts` is the epoch's contract: the "h"
// seat is a frozen copy of the default 計算 seat as it stood on 2026-08-25,
// and it plays the same hanchan forever. Drift here is a BUG — in frozen.ts
// (a "frozen" value that turned out to reference a live default) or in a
// shared code path (engine, kernel, reads) — never a legitimate behaviour
// change. When a shared-path change is intended anyway, it must argue with
// this comment first: the whole point of the epoch was a baseline that later
// work cannot silently move.
//
// (Verified at epoch time, not pinned: frozen-h ≡ default-k, bit for bit, on
// 2026-08-25. That equality is EXPECTED to break as "k" evolves — it is the
// difference between a snapshot and a reference.)
//
// RE-CAPTURED 2026-08-27, by the owner's explicit direction — not a
// regeneration of convenience: the DOJO'S OWN RULES were corrected (持ち点
// 8000点未満 judged at 終局, buffer 南入以降; the call gate's 対々和/バック
// clauses tightened) and the owner ruled the frozen seat plays under the
// corrected dojo too ("apply fixes to both h agent"). The seat's weights are
// untouched; the world it plays in changed. The frozen-h ≡ frozen-0825
// league equality was re-verified to hold across the re-capture.
//
// RE-CAPTURED AGAIN 2026-08-28, by the owner's explicit direction — a
// shared-path BUG, exactly the case this comment names: the danger assessor
// was fed the discarder's own tiles twice (`observe.ts` passed the
// own-inclusive `Table.visibleCounts` as the public count AND `ownCounts`),
// so a held pair rated 安全 through the killed-shapes cap. Fixed to mjrender's
// contract (public counts + hand); found from the ranked arena wire log (a
// 12,000 単騎 ron on an East held as a pair). The seat's weights are
// untouched. frozen-h ≡ frozen-0825 re-verified across this re-capture
// (identical on all three seeds).

import { assertEquals } from "@std/assert";
import { closeArm, headless, openArm } from "../src/harness.ts";
import type { TableSpec } from "../src/spec.ts";

function fingerprint(seed: number): string {
  const r = headless(1, seed, "hhhh", {}).results[0];
  const body = JSON.stringify({
    scores: r.scores,
    outcomes: r.outcomes,
    ledger: r.ledger.map((v) => `${v.seat}:${v.label}`),
    riichis: r.riichis,
  });
  let x = 0x811c9dc5;
  for (let i = 0; i < body.length; i++) x = Math.imul(x ^ body.charCodeAt(i), 0x01000193) >>> 0;
  return `${r.scores.join("/")}#${x.toString(16).padStart(8, "0")}`;
}

const EPOCH_PIN: Record<number, string> = {
  101: "61400/22200/18800/17600#b32bd33b",
  505: "32700/20100/21200/46000#03e1b70e",
  909: "65400/28600/10000/16000#8f05e7a0",
};

Deno.test("frozen h: the epoch pin — these streams never change", () => {
  for (const [seed, want] of Object.entries(EPOCH_PIN)) {
    assertEquals(fingerprint(Number(seed)), want, `種${seed}: 凍結席が動いた — これはバグです`);
  }
});

Deno.test("frozen h: configuration is inert even when smuggled past the loaders", () => {
  // `loadTable`/`argError` refuse configuration on an "h" seat loudly; an
  // in-memory TableSpec can still carry it, and `makePolicy` must IGNORE it —
  // the frozen seat has no seam a vector could enter through.
  const bare: TableSpec = [{ kind: "h" }, { kind: "h" }, { kind: "h" }, { kind: "h" }];
  const smuggled: TableSpec = [
    { kind: "h", ktune: { heuristic: { shanten: 1 }, hand: {}, riichi: { bias: -1000 } } },
    { kind: "h", plan: true, standings: true },
    { kind: "h", curriculum: 0.5 },
    { kind: "h" },
  ];
  const a = headless(2, 4242, bare);
  const b = headless(2, 4242, smuggled);
  assertEquals(b.results, a.results);
  // …and the built seats really are one construction, not four variants.
  const arm = openArm(smuggled);
  try {
    // deno-lint-ignore no-explicit-any
    const p = (s: number) => arm.built[s].policy as any;
    for (const s of [0, 1, 2]) {
      assertEquals(p(s).w, p(3).w);
      assertEquals(p(s).hand, null);
      assertEquals(p(s).riichiHead, null);
    }
  } finally {
    closeArm(arm);
  }
});
