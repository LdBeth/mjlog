// Reliability tables for the 色読み lane (2026-08-28) — the reading half of the
// oracle-guided method. `sense_lane.ts` recorded evidence next to truth; this
// script bins them and prints what the evidence is actually worth:
//
//   1. 染め場: P(the flagged opponent is dye-committed | heat bin), by junme
//      band, plus what a dyed round costs the subject on average — the numbers
//      that justify (or refuse) `someRisk`/`somePressure` and the HEAT_BAR.
//   2. トイツ場: mean opponent pairing and P(field pairs | toitsuba bin) — what
//      backs the chiitoi tax's `1 − toitsuba` scale.
//   3. The honor-retention corroborator, evaluated OFFLINE from the raw
//      components: does "few honors discarded" sharpen the dye read?
//
// Like every fit in this repo it RECOMMENDS; the sweep decides. No file is
// written — the output is tables to choose weights against, and any change to
// a fact constant goes back through the paired grade.
//
//   deno run --allow-read scripts/sense_fit.ts runs/sense/lane-MMDD.jsonl

interface Row {
  g: number;
  k: number;
  h: number;
  j: number;
  ev: {
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
  tr: { share: [number, number, number]; pairs: number; pons?: number; sh: number; open: number }[];
  out?: { kind: string; win?: number; winShare?: [number, number, number]; delta0?: number };
}

/** Dye-committed truth: one suit+honors covers ≥3/4 of the hand, and it is going somewhere. */
const DYE_SHARE = 0.75;
const DYE_SH = 2;

const path = Deno.args[0];
if (!path) {
  console.error("usage: sense_fit LANE.jsonl");
  Deno.exit(2);
}
const rows: Row[] = [];
for (const line of (await Deno.readTextFile(path)).split("\n")) {
  if (line.trim()) rows.push(JSON.parse(line));
}
console.log(`lane: ${rows.length} rows\n`);

const fmt = (x: number) => x.toFixed(3);
const pct = (a: number, b: number) => b === 0 ? "  —  " : `${(a * 100 / b).toFixed(1)}%`;

// Per-(opponent, suit) instances: evidence heat vs dye truth.
interface Inst {
  heat: number;
  dye: boolean;
  j: number;
  honors: number;
  nNum: number;
  /** Did THIS opponent win this round with a dyed hand, and what did it pay? */
  dyeWin: boolean;
  delta0: number;
}
const inst: Inst[] = [];
for (const r of rows) {
  for (let i = 0; i < 3; i++) {
    const t = r.tr[i];
    for (let s = 0; s < 3; s++) {
      const heat = Math.min(1, r.ev.v[i][s] + r.ev.m[i][s]);
      const dye = t.share[s] >= DYE_SHARE && t.sh <= DYE_SH;
      const dyeWin = r.out?.kind === "agari" && r.out.win === i + 1 &&
        (r.out.winShare?.[s] ?? 0) >= DYE_SHARE;
      inst.push({ heat, dye, j: r.j, honors: r.ev.z[i], nNum: r.ev.n[i], dyeWin, delta0: r.out?.delta0 ?? 0 });
    }
  }
}

console.log("== 染め場: P(dye-committed | heat), by junme band ==");
console.log("heat bin      巡1-6            巡7-12           巡13-      (n dye/total)");
const BINS: [number, number][] = [[0, 0.001], [0.001, 0.35], [0.35, 0.5], [0.5, 0.7], [0.7, 1.01]];
for (const [lo, hi] of BINS) {
  const cells = [[1, 6], [7, 12], [13, 99]].map(([jl, jh]) => {
    const xs = inst.filter((x) => x.heat >= lo && x.heat < hi && x.j >= jl && x.j <= jh);
    const dye = xs.filter((x) => x.dye).length;
    return `${pct(dye, xs.length)} (${dye}/${xs.length})`;
  });
  console.log(`[${lo.toFixed(2)},${hi > 1 ? "1.00" : hi.toFixed(2)})  ${cells.map((c) => c.padEnd(17)).join("")}`);
}

console.log("\n== 染め場: base rate and the honor corroborator (巡7+, heat ≥ 0.35) ==");
{
  const late = inst.filter((x) => x.j >= 7);
  const dye = late.filter((x) => x.dye).length;
  console.log(`base rate P(dye) 巡7+: ${pct(dye, late.length)} (${dye}/${late.length})`);
  const hot = late.filter((x) => x.heat >= 0.35);
  for (const [lbl, f] of [
    ["honors ≤ 1", (x: Inst) => x.honors <= 1],
    ["honors ≤ 3", (x: Inst) => x.honors <= 3],
    ["honors ≥ 4", (x: Inst) => x.honors >= 4],
  ] as const) {
    const xs = hot.filter(f);
    const d = xs.filter((x) => x.dye).length;
    console.log(`heat≥0.35 & ${lbl}: P(dye) ${pct(d, xs.length)} (${d}/${xs.length})`);
  }
}

console.log("\n== 染め場: what a flagged round costs (round-level, 巡7+) ==");
{
  // One sample per (round, opponent-suit) at its LAST recorded decision.
  const last = new Map<string, Inst & { key: string }>();
  let i = 0;
  for (const r of rows) {
    for (let oi = 0; oi < 3; oi++) {
      for (let s = 0; s < 3; s++) {
        const key = `${r.g}:${r.k}:${r.h}:${oi}:${s}`;
        const heat = Math.min(1, r.ev.v[oi][s] + r.ev.m[oi][s]);
        const dyeWin = r.out?.kind === "agari" && r.out.win === oi + 1 &&
          (r.out.winShare?.[s] ?? 0) >= DYE_SHARE;
        last.set(key, {
          key,
          heat,
          dye: r.tr[oi].share[s] >= DYE_SHARE && r.tr[oi].sh <= DYE_SH,
          j: r.j,
          honors: r.ev.z[oi],
          nNum: r.ev.n[oi],
          dyeWin,
          delta0: r.out?.delta0 ?? 0,
        });
        i++;
      }
    }
  }
  const finals = [...last.values()];
  for (const [lo, hi] of [[0, 0.35], [0.35, 0.7], [0.7, 1.01]] as const) {
    const xs = finals.filter((x) => x.heat >= lo && x.heat < hi);
    const wins = xs.filter((x) => x.dyeWin);
    const meanD = xs.length ? xs.reduce((a, x) => a + x.delta0, 0) / xs.length : 0;
    const meanW = wins.length ? wins.reduce((a, x) => a + x.delta0, 0) / wins.length : 0;
    console.log(
      `heat [${lo.toFixed(2)},${hi > 1 ? "1.00" : hi.toFixed(2)}): rounds ${xs.length}, ` +
        `P(dye win by flagged opp) ${pct(wins.length, xs.length)}, ` +
        `subject Δ mean ${meanD.toFixed(0)} (dye-win rounds: ${meanW.toFixed(0)})`,
    );
  }
}

console.log("\n== トイツ場: field pairing truth | toitsuba bin ==");
{
  // A pair that PONNED is still a pair the field delivered — count it, or the
  // pon-heavy top bin anti-calibrates against its own evidence.
  const pairsEver = (t: Row["tr"][number]) => t.pairs + (t.pons ?? 0);
  for (const [lo, hi] of [[0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 1.01]] as const) {
    const xs = rows.filter((r) => r.ev.toitsuba >= lo && r.ev.toitsuba < hi);
    if (xs.length === 0) continue;
    const meanPairs = xs.reduce(
      (a, r) => a + (pairsEver(r.tr[0]) + pairsEver(r.tr[1]) + pairsEver(r.tr[2])) / 3,
      0,
    ) / xs.length;
    const pairField = xs.filter((r) => r.tr.filter((t) => pairsEver(t) >= 3).length >= 2).length;
    console.log(
      `toitsuba [${lo.toFixed(1)},${hi > 1 ? "1.0" : hi.toFixed(1)}): n=${xs.length}, ` +
        `mean opp pairs-ever ${meanPairs.toFixed(2)}, P(≥2 opps pair-rich) ${pct(pairField, xs.length)}`,
    );
  }
}

console.log("\n== alignment (doctrine check): own pairing vs the field ==");
{
  const pairHands = rows.filter((r) => r.ev.ownPairs >= 4 && r.j >= 4 && r.j <= 10);
  const aligned = pairHands.filter((r) => r.ev.toitsuba >= 0.4);
  const lone = pairHands.filter((r) => r.ev.toitsuba < 0.2);
  const mean = (xs: Row[]) => xs.length ? xs.reduce((a, r) => a + (r.out?.delta0 ?? 0), 0) / xs.length : 0;
  console.log(
    `own ≥4 pairs, 巡4-10: n=${pairHands.length}; aligned (toitsuba≥0.4) n=${aligned.length} ` +
      `Δ ${mean(aligned).toFixed(0)}; lone (toitsuba<0.2) n=${lone.length} Δ ${mean(lone).toFixed(0)}`,
  );
}
