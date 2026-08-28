// The 色読み oracle lane — ground truth for the field sense (2026-08-28).
//
// The oracle-guided training method, applied to the sense: play headless
// hanchan with the shipping configuration, and at every turn decision of the
// subject seat record the sense's RAW evidence (`fieldSenseDetail`) next to
// the TRUTH read straight off the live Table — each opponent's actual hand
// composition (suit shares, concealed pairs, shanten) — plus, joined after the
// match, the round's outcome (who won, with what suit concentration, for how
// much). `scripts/sense_fit.ts` turns the lane into reliability tables:
// P(dye-committed | evidence), E[cost], トイツ場 calibration — so the weights
// and the fact constants are chosen against measured truth instead of a blind
// grid.
//
// The recorder is a WRAPPER around the subject policy: it computes facts and
// reads the tap, then delegates the decision untouched. The seat plays exactly
// the game it would play without the lane — the sink is invisible, the same
// contract `--calibrate` keeps for M10.
//
//   deno task check && deno run --allow-read --allow-write --allow-ffi \
//     --allow-env=MJGAME_NATIVE scripts/sense_lane.ts \
//     --games=600 --seed=90000 --out=runs/sense/lane-MMDD.jsonl \
//     [--ktune=weights/champion.json]

import { tileType } from "mjrender/tiles.ts";
import { dojoHooks } from "../src/dojo.ts";
import { loadKtune, makePolicy } from "../src/harness.ts";
import { countsFromTiles, shanten } from "../src/kernel.ts";
import { runMatchSync } from "../src/match.ts";
import type { Observation } from "../src/observe.ts";
import type { SyncPolicy } from "../src/policy.ts";
import { DOJO_HEADLESS, JANKI } from "../src/rules.ts";
import { scorer } from "../src/score.ts";
import type { Table } from "../src/table.ts";
import type { Seat } from "../src/types.ts";
import { SEATS } from "../src/types.ts";
import { fieldSense, fieldSenseDetail } from "../src/ai/sense.ts";

interface OppTruth {
  /** (suit tiles + honors) / all tiles, hand and melds together, per suit. */
  share: [number, number, number];
  /** Concealed pair types. */
  pairs: number;
  /**
   * Pon (+ 明槓) melds — pairs that LEFT the concealed hand by pairing loudly.
   * A pairing-field truth label must count them: the first lane's top toitsuba
   * bin anti-calibrated precisely because pon consumed the pairs it proved.
   */
  pons: number;
  sh: number;
  open: number;
}

interface Row {
  g: number; // seed
  k: number; // kyoku
  h: number; // honba
  j: number; // junme
  ev: {
    /** Per opponent (relative 1..3): void score, meld boost, nNum, honor discards. */
    v: [number, number, number][];
    m: [number, number, number][];
    n: number[];
    z: number[];
    tPon: number;
    tDup: number;
    ownPairs: number;
    someba: [number, number, number];
    hot: number;
    toitsuba: number;
  };
  tr: OppTruth[];
  out?: {
    kind: "agari" | "ryuukyoku";
    /** Winner seat RELATIVE to the subject (0 = subject), first win only. */
    win?: number;
    /** Winner's whole-hand suit-share vector, same definition as `tr.share`. */
    winShare?: [number, number, number];
    /** The subject's point delta for the round. */
    delta0?: number;
  };
}

function shareOf(tiles: number[]): [number, number, number] {
  const bySuit = [0, 0, 0];
  let honors = 0;
  for (const t of tiles) {
    const ty = tileType(t);
    if (ty >= 27) honors++;
    else bySuit[ty < 9 ? 0 : ty < 18 ? 1 : 2]++;
  }
  const total = tiles.length || 1;
  return [
    (bySuit[0] + honors) / total,
    (bySuit[1] + honors) / total,
    (bySuit[2] + honors) / total,
  ];
}

function truthOf(t: Table, self: Seat): OppTruth[] {
  const out: OppTruth[] = [];
  for (let i = 0; i < 3; i++) {
    const o = ((self + i + 1) % 4) as Seat;
    const hand = t.hands[o];
    const meldTiles = t.melds[o].flatMap((m) => m.tiles);
    const counts = countsFromTiles(hand);
    let pairs = 0;
    for (let ty = 0; ty < 34; ty++) if (counts[ty] >= 2) pairs++;
    let pons = 0;
    for (const m of t.melds[o]) {
      if (m.kind === "pon" || m.kind === "daiminkan" || m.kind === "shouminkan") pons++;
    }
    out.push({
      share: shareOf([...hand, ...meldTiles]),
      pairs,
      pons,
      sh: shanten(counts, t.melds[o].length, t.isMenzen(o)),
      open: t.melds[o].length,
    });
  }
  return out;
}

const flags = new Map<string, string>();
for (const arg of Deno.args) {
  const m = /^--([a-z]+)=(.+)$/.exec(arg);
  if (!m) {
    console.error(`unknown argument: ${arg}`);
    Deno.exit(2);
  }
  flags.set(m[1], m[2]);
}
const games = Number(flags.get("games") ?? "600");
const seed0 = Number(flags.get("seed") ?? "90000");
const outPath = flags.get("out");
if (!outPath || !Number.isFinite(games) || !Number.isFinite(seed0)) {
  console.error("usage: sense_lane --games=N --seed=S --out=FILE [--ktune=PATH]");
  Deno.exit(2);
}
const ktune = loadKtune(flags.get("ktune") ?? "weights/champion.json");

await Deno.mkdir(outPath.replace(/\/[^/]+$/, ""), { recursive: true }).catch(() => {});
const f = await Deno.open(outPath, { write: true, create: true, truncate: true });
const enc = new TextEncoder();
let rows = 0;

const t0 = performance.now();
for (let g = 0; g < games; g++) {
  const seed = seed0 + g;
  const ref: { t: Table | null } = { t: null };
  const matchRows: Row[] = [];

  const policies = SEATS.map((s): SyncPolicy => {
    const p = makePolicy({
      kind: s === 0 ? "k" : "h",
      name: s === 0 ? "K0" : `H${s}`,
      seed: seed * 4 + s,
      ktune: s === 0 ? ktune : undefined,
    }).policy;
    if (s !== 0) return p;
    return {
      name: p.name,
      reset: (sd) => p.reset?.(sd),
      notify: (e) => p.notify?.(e),
      decide: (obs: Observation) => {
        const t = ref.t;
        if (t && obs.drawn !== null) {
          const d = fieldSenseDetail(obs);
          const fs = fieldSense(obs);
          matchRows.push({
            g: seed,
            k: t.kyoku,
            h: t.round.honba,
            j: obs.junme,
            ev: {
              v: d.opps.map((o) => o.voidScore),
              m: d.opps.map((o) => o.meldBoost),
              n: d.opps.map((o) => o.nNum),
              z: d.opps.map((o) => o.honors),
              tPon: d.tPon,
              tDup: d.tDup,
              ownPairs: d.ownPairs,
              someba: fs.someba,
              hot: fs.hot,
              toitsuba: fs.toitsuba,
            },
            tr: truthOf(t, 0),
          });
        }
        return p.decide(obs) as ReturnType<SyncPolicy["decide"]>;
      },
    };
  });

  const r = runMatchSync(policies, {
    seed,
    cfg: JANKI,
    dojo: DOJO_HEADLESS,
    scorer,
    tableRef: ref,
    ...dojoHooks({ dojo: DOJO_HEADLESS, oracle: scorer }),
  });

  // Join outcomes through `r.rounds` — index-aligned with `r.outcomes` and
  // carrying (kyoku, honba), so a round that ended before the subject's first
  // draw (an early ron) cannot skew the mapping the way an order-of-appearance
  // key list observed from decisions would. The pair is unique in a hanchan:
  // 連荘 bumps honba, a passing win advances kyoku.
  const roundIdx = new Map<string, number>();
  r.rounds.forEach((rd, i) => roundIdx.set(`${rd.kyoku}:${rd.honba}`, i));
  for (const row of matchRows) {
    const idx = roundIdx.get(`${row.k}:${row.h}`) ?? -1;
    const o = r.outcomes[idx];
    if (!o) continue;
    if (o.kind === "agari") {
      const w = o.wins[0];
      row.out = {
        kind: "agari",
        win: (w.who - 0 + 4) % 4,
        winShare: shareOf([...w.hand, ...w.melds.flatMap((m) => m.tiles)]),
        delta0: o.deltas[0],
      };
    } else {
      row.out = { kind: "ryuukyoku", delta0: o.deltas[0] };
    }
  }
  for (const row of matchRows) {
    await f.write(enc.encode(JSON.stringify(row) + "\n"));
  }
  rows += matchRows.length;
}
f.close();
const dt = (performance.now() - t0) / 1000;
console.log(
  `sense lane ${outPath}: ${rows} rows over ${games} 半荘 (${dt.toFixed(1)}s, ${
    (games / dt).toFixed(1)
  } 半荘/秒)`,
);
