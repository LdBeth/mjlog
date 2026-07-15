// Decode Tenhou's packed meld code (the `m` attribute on <N who=".." m=".."/>).
//
// Bit layout (see https://81100118.github.io and community mjlog docs):
//   bits 0-1 : caller-relative source (0=self, 1=shimocha, 2=toimen, 3=kamicha)
//   bit  2   : chi
//   bit  3   : pon
//   bit  4   : added-kan (shouminkan)
//   bit  5   : nuki (kita, sanma)
//   (none of 2-5 set) ⇒ kan: daiminkan if source != self, else ankan.

import type { Meld, MeldKind, Tile } from "./model.ts";

export function decodeMeld(who: number, m: number): Meld {
  const rel = m & 0x3;
  const from = (who + rel) % 4;

  if (m & 0x4) {
    // --- Chi (always from kamicha; rel == 3) ---
    let t = (m & 0xfc00) >> 10;
    const called = t % 3; // which of the 3 tiles was taken
    t = Math.floor(t / 3);
    const base = Math.floor(t / 7) * 9 + (t % 7); // run start as tile type
    const b = base * 4;
    const tiles: Tile[] = [
      b + 0 + ((m & 0x0018) >> 3),
      b + 4 + ((m & 0x0060) >> 5),
      b + 8 + ((m & 0x0180) >> 7),
    ];
    return finalize("chi", who, from, tiles, tiles[called]);
  }

  if (m & 0x8) {
    // --- Pon or added-kan (shouminkan) ---
    const unused = (m & 0x0060) >> 5; // which of the 4 copies is not in the pon
    let t = (m & 0xfe00) >> 9;
    const called = t % 3;
    t = Math.floor(t / 3); // tile type
    const b = t * 4;
    if (m & 0x10) {
      // shouminkan: the previously-unused copy is added to the existing pon
      const tiles: Tile[] = [b, b + 1, b + 2, b + 3];
      return finalize("shouminkan", who, from, tiles, b + unused);
    }
    const offsets = [0, 1, 2, 3].filter((o) => o !== unused);
    const tiles: Tile[] = offsets.map((o) => b + o);
    return finalize("pon", who, from, tiles, b + offsets[called]);
  }

  if (m & 0x20) {
    // --- Nuki / kita (sanma) ---
    const hai = (m & 0xff00) >> 8;
    return finalize("nuki", who, who, [hai], hai);
  }

  // --- Kan (daiminkan if called from another player, else ankan) ---
  const hai = (m & 0xff00) >> 8;
  const b = (hai >> 2) * 4;
  const tiles: Tile[] = [b, b + 1, b + 2, b + 3];
  const kind: MeldKind = rel === 0 ? "ankan" : "daiminkan";
  return finalize(kind, who, rel === 0 ? who : from, tiles, hai);
}

function finalize(
  kind: MeldKind,
  who: number,
  fromWho: number,
  tiles: Tile[],
  calledTile: Tile,
): Meld {
  return { kind, who, fromWho, tiles: [...tiles].sort((a, b) => a - b), calledTile };
}

/**
 * Which concealed tiles this meld consumed from the caller's own hand
 * (i.e. all meld tiles except the one taken from an opponent's discard).
 * For ankan all four are concealed; for shouminkan only the added tile.
 */
export function concealedTilesUsed(m: Meld): Tile[] {
  switch (m.kind) {
    case "ankan":
      return [...m.tiles]; // all four come from the hand
    case "shouminkan":
    case "nuki":
      return [m.calledTile]; // only the added tile leaves the hand now
    default: {
      // chi, pon, daiminkan: everything except the tile taken from the discard
      const out = [...m.tiles];
      const i = out.indexOf(m.calledTile);
      if (i >= 0) out.splice(i, 1);
      return out;
    }
  }
}
