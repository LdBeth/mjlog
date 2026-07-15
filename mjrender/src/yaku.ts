// Tenhou yaku / yakuman name table (ids 0..54).
// Regular wins list `(id, han)` pairs in AGARI `yaku`; yakuman wins list ids in
// AGARI `yakuman`. ids 37..51 are yakuman; 52..54 are dora / ura / aka.

export const YAKU_NAMES: readonly string[] = [
  "門前清自摸和", // 0
  "立直", // 1
  "一発", // 2
  "槍槓", // 3
  "嶺上開花", // 4
  "海底摸月", // 5
  "河底撈魚", // 6
  "平和", // 7
  "断幺九", // 8
  "一盃口", // 9
  "自風 東", // 10
  "自風 南", // 11
  "自風 西", // 12
  "自風 北", // 13
  "場風 東", // 14
  "場風 南", // 15
  "場風 西", // 16
  "場風 北", // 17
  "役牌 白", // 18
  "役牌 發", // 19
  "役牌 中", // 20
  "両立直", // 21
  "七対子", // 22
  "混全帯幺九", // 23
  "一気通貫", // 24
  "三色同順", // 25
  "三色同刻", // 26
  "三槓子", // 27
  "対々和", // 28
  "三暗刻", // 29
  "小三元", // 30
  "混老頭", // 31
  "二盃口", // 32
  "純全帯幺九", // 33
  "混一色", // 34
  "清一色", // 35
  "人和", // 36
  "天和", // 37 (yakuman from here)
  "地和", // 38
  "大三元", // 39
  "四暗刻", // 40
  "四暗刻単騎", // 41
  "字一色", // 42
  "緑一色", // 43
  "清老頭", // 44
  "九蓮宝燈", // 45
  "純正九蓮宝燈", // 46
  "国士無双", // 47
  "国士無双13面", // 48
  "大四喜", // 49
  "小四喜", // 50
  "四槓子", // 51
  "ドラ", // 52
  "裏ドラ", // 53
  "赤ドラ", // 54
];

export function yakuName(id: number): string {
  return YAKU_NAMES[id] ?? `役${id}`;
}

export function isYakumanId(id: number): boolean {
  return id >= 37 && id <= 51;
}

/** Parse an AGARI `yaku="id,han,id,han,..."` attribute. */
export function parseYaku(attr: string | undefined): Array<{ id: number; han: number }> {
  if (!attr) return [];
  const nums = attr.split(/[\s,]+/).filter((s) => s.length).map(Number);
  const out: Array<{ id: number; han: number }> = [];
  for (let i = 0; i + 1 < nums.length; i += 2) {
    out.push({ id: nums[i], han: nums[i + 1] });
  }
  return out;
}

/** Parse an AGARI `yakuman="id,id,..."` attribute. */
export function parseYakuman(attr: string | undefined): number[] {
  if (!attr) return [];
  return attr.split(/[\s,]+/).filter((s) => s.length).map(Number);
}

const LIMIT_NAMES = ["", "満貫", "跳満", "倍満", "役満", "数え役満"];

export function limitName(limit: number): string {
  return LIMIT_NAMES[limit] ?? "";
}
