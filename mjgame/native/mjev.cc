// mjev.cc — M15: the 計算 seat's expected-value core.
//
// Three sections, in dependency order: the two *exact* sub-parts the DP stands
// on — a re-entrant shanten/ukeire core (`mjev_shanten` / `mjev_ukeire_mask`)
// and an integer scorer (`mjev_score`) — and then the dynamic program itself
// (`mjev_eval_discard` / `mjev_eval_rest`), which prices a discard, the riichi
// declaration after it, and giving up. The DP's own contract is documented at
// the head of its section; this preamble is about what makes it trustworthy.
//
// Unlike `mjkernel.cc`, this library has NO TypeScript twin (owner decision,
// 2026-08-30). What keeps it honest is that its two sub-parts DO have one:
//
//   * the scorer is a transliteration of `src/decompose.ts` + `src/fu.ts` +
//     `src/yaku.ts` + `src/score.ts`, all integer arithmetic, and
//     `test/ev_native_test.ts` compares it field by field against `scoreWin`
//     on hundreds of thousands of random complete hands at ZERO tolerance;
//   * shanten/ukeire are `mjkernel.cc`'s group-word algorithm, fuzzed against
//     `src/kernel.ts` the same way.
//
// RE-ENTRANCY (plan D2). `mjkernel.cc` keeps its two lazy tables in globals,
// which is why the harness has to warm worker 0 before forking work out. Here
// the memo lives INSIDE the context: an open-addressing hash from the base-5
// group index to the packed summary word, grown on demand. No globals hold
// state, no statics, no lock, no warm-up — each worker dlopens its own image
// and each seat owns its own `mjev_create` handle.
//
// FLOATING POINT. The DP is the only part of this file that does any, and it
// has to answer the same double twice — `--jobs=N` identity depends on the
// value memo being a pure function of the inputs — so `-ffp-contract=off` is on
// the build line and every accumulation below is in a fixed order. There is no
// reduction, no threading and no library math call anywhere in the search.

#include <cstdint>
#include <cstdlib>
#include <cstring>

#define MJ_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

// ---------------------------------------------------------------------------
// the wire layout — mirrors src/ai/evlayout.ts BY NAME
// ---------------------------------------------------------------------------
//
// One contract, two readers. Editing an offset here without editing there (or
// the reverse) is an ABI break; `mjev_abi()` is what the wrapper checks.

enum { EV_ABI_VERSION = 1 };

/** `ints` of `mjev_eval_discard` / `mjev_eval_rest`. */
enum {
  I_HAND = 0,
  I_NMELDS = 34,
  I_MELDS = 35,
  I_SEAT_WIND = 47,
  I_ROUND_WIND = 48,
  I_DEALER = 49,
  I_HONBA = 50,
  I_KYOTAKU = 51,
  I_OWN_RIICHI = 52,
  I_FURITEN_PERM = 53,
  I_FURITEN_TEMP = 54,
  I_JUNME = 55,
  I_T = 56,
  I_AKA_HELD = 57,
  I_AKA_UNSEEN = 58,
  I_CLOSED = 59,
  I_CAND = 60,
  I_UNSEEN = 94,
  I_DORA = 128,
  I_RIVER = 162,
  I_KANDORA_ON = 196,
  I_HAS_DRAW = 197,
  I_K = 198,
  I_HAS_POOL = 199,
  I_HAS_URA = 200,
  I_HAS_NEXTDORA = 201,
  I_MODE = 202,
  INTS_LEN = 208
};

/** `dbls` of `mjev_eval_discard` / `mjev_eval_rest`. */
enum {
  D_TENPAI = 0,
  D_EXPLOSS = 3,
  D_PIN = 6,
  D_COSTIN = 40,
  D_GAIN = 74,
  D_RISK = 75,
  KMAX = 4,
  D_DRAW = 76,
  D_POOL = 212,
  D_URA = 246,
  D_NEXTDORA = 280,
  DBLS_LEN = 320
};

/** `out` of `mjev_eval_discard`. */
enum {
  O_STRIDE = 4,
  O_TOTAL = 0,
  O_DAMA = 1,
  O_RIICHI = 2,
  O_FOLDLINE = 3,
  O_NODES = 136,
  O_TRUNC = 137,
  O_BEST_PUSH = 138,
  O_BEST_FOLD = 139,
  OUT_LEN = 140
};

/** `meta` of `mjev_eval_rest`. */
enum {
  R_VALUE = 0,
  R_NODES = 1,
  R_TRUNC = 2,
  R_PTENPAI = 3,
  R_PWIN = 4,
  R_EVALUE = 5,
  R_ECOST = 6,
  REST_META_LEN = 8
};

/** `mjev_score` input. */
enum {
  S_COUNTS = 0,
  S_NMELDS = 34,
  S_MELDS = 35,
  S_WINTYPE = 47,
  S_TSUMO = 48,
  S_RIICHI = 49,
  S_DOUBLE = 50,
  S_IPPATSU = 51,
  S_RINSHAN = 52,
  S_CHANKAN = 53,
  S_HAITEI = 54,
  S_HOUTEI = 55,
  S_TENHOU = 56,
  S_CHIIHOU = 57,
  S_SEAT_WIND = 58,
  S_ROUND_WIND = 59,
  S_DORA = 60,
  S_URA = 94,
  S_AKA = 128,
  S_KUITAN = 129,
  S_KAZOE = 130,
  S_KIRIAGE = 131,
  S_DWFU = 132,
  S_IPPATSU_CFG = 133,
  SCORE_IN_LEN = 136
};

/** `mjev_score` output. */
enum {
  SO_OK = 0,
  SO_HAN = 1,
  SO_FU = 2,
  SO_BASE = 3,
  SO_YAKUMAN = 4,
  SO_LIMIT = 5,
  SO_RON = 6,
  SO_TSUMO_TOTAL = 7,
  SCORE_OUT_LEN = 8
};

/** `mjev_create` parameter vector — `EV_PARAM_ORDER` in evlayout.ts. */
enum {
  P_MEAN_UKEIRE0 = 0,
  P_MEAN_UKEIRE1 = 1,
  P_MEAN_UKEIRE2 = 2,
  P_MEAN_UKEIRE3 = 3,
  P_RON_FACTOR = 4,
  P_OPP_HAZARD = 5,
  P_OPP_GROWTH = 6,
  P_VALUE_RIICHI = 7,
  P_VALUE_DAMATEN = 8,
  P_VALUE_OPEN = 9,
  P_VALUE_HONITSU = 10,
  P_VALUE_PER_DORA = 11,
  P_VALUE_YAKUHAI = 12,
  P_VALUE_DEALER = 13,
  P_VALUE_CAP = 14,
  P_DEALIN_RATE = 15,
  P_TSUMO_SHARE = 16,
  P_FOLD_HAZARD = 17,
  P_RIICHI_DEALIN_MULT = 18,
  P_IPPATSU_P = 19,
  P_STICK_AT_DRAW = 20,
  P_DEALER_RENCHAN = 21,
  P_CALL_MARGIN = 22,
  P_RIICHI_MARGIN = 23,
  P_POINTS_PER_SCORE = 24,
  P_EXACT_SHANTEN = 25,
  P_SAME_SHANTEN_RUNGS = 26,
  P_MAX_NODES = 27,
  P_DISCARD = 28,
  P_RIICHI = 29,
  P_CALLS = 30,
  P_KUITAN = 31,
  P_KAZOE_YAKUMAN = 32,
  P_KIRIAGE_MANGAN = 33,
  P_DOUBLE_WIND_FU = 34,
  P_NOTEN_PENALTY_TOTAL = 35,
  EV_PARAMS_LEN = 36
};

// ---------------------------------------------------------------------------
// tile predicates (src/tiles.ts)
// ---------------------------------------------------------------------------

inline bool isHonor(int t) { return t >= 27; }
inline bool isTerminal(int t) { return t < 27 && (t % 9 == 0 || t % 9 == 8); }
inline bool isYaochu(int t) { return isHonor(t) || isTerminal(t); }
inline bool isSimple(int t) { return !isYaochu(t); }
/** 緑一色: 2,3,4,6,8 索 + 發. */
inline bool isGreen(int t) {
  return t == 19 || t == 20 || t == 21 || t == 23 || t == 25 || t == 32;
}
/** 0 = 萬, 1 = 筒, 2 = 索; honors have no suit. */
inline int suitOf(int t) { return t / 9; }

const int YAOCHU[13] = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33};

// ===========================================================================
// SHANTEN — mjkernel.cc's group-word algorithm, with the memo in the context
// ===========================================================================

// --- reference: an exact transliteration of standardShanten() in shanten.ts ---

struct RefState {
  uint8_t c[34];
  int cap;
  int best;
};

void refDfs(RefState &st, int i, int sets, int partials, int head) {
  uint8_t *c = st.c;
  while (i < 34 && c[i] == 0) i++;

  const int used = sets < st.cap ? sets : st.cap;
  const int room = st.cap - used;
  const int v = 2 * used + (partials < room ? partials : room) + head;
  if (v > st.best) st.best = v;
  if (i >= 34 || (used + partials >= st.cap && head)) return;

  const int rank = i < 27 ? i % 9 : -1;

  if (c[i] >= 3) {
    c[i] -= 3;
    refDfs(st, i, sets + 1, partials, head);
    c[i] += 3;
  }
  if (rank >= 0 && rank <= 6 && c[i + 1] > 0 && c[i + 2] > 0) {
    c[i]--, c[i + 1]--, c[i + 2]--;
    refDfs(st, i, sets + 1, partials, head);
    c[i]++, c[i + 1]++, c[i + 2]++;
  }
  if (!head && c[i] >= 2) {
    c[i] -= 2;
    refDfs(st, i, sets, partials, 1);
    c[i] += 2;
  }
  if (c[i] >= 2) {
    c[i] -= 2;
    refDfs(st, i, sets, partials + 1, head);
    c[i] += 2;
  }
  if (rank >= 0 && rank <= 7 && c[i + 1] > 0) {
    c[i]--, c[i + 1]--;
    refDfs(st, i, sets, partials + 1, head);
    c[i]++, c[i + 1]++;
  }
  if (rank >= 0 && rank <= 6 && c[i + 2] > 0) {
    c[i]--, c[i + 2]--;
    refDfs(st, i, sets, partials + 1, head);
    c[i]++, c[i + 2]++;
  }
  c[i]--;
  refDfs(st, i, sets, partials, head);
  c[i]++;
}

int refStandard(const uint8_t *counts, int openMelds) {
  const int cap = 4 - openMelds;
  if (cap < 0) return 8;
  RefState st;
  std::memcpy(st.c, counts, 34);
  st.cap = cap;
  st.best = 0;
  refDfs(st, 0, 0, 0, 0);
  return 2 * cap - st.best;
}

// --- per-group summary words ---

// best[m*2 + h] = max reachable partials for that (melds, head), -1 = none.
typedef int8_t Summary[10];

// Base-5 index spaces: a suit is 9 counts of 0..4 (< 5^9), honors are 7
// (< 5^7). The honor keys are shifted past the suit ones so one memo serves both.
const uint32_t SUIT_STATES = 1953125; // 5^9

/**
 * The block-peeling DFS of shanten.ts restricted to ONE group, with the pruning
 * removed: the summary has to hold the whole Pareto frontier, not just the
 * branch that happens to be best globally.
 */
void groupDfs(uint8_t *c, int len, bool suited, int i, int m, int p, int h, Summary best) {
  while (i < len && c[i] == 0) i++;

  const int mm = m > 4 ? 4 : m;
  const int pp = p > 4 ? 4 : p;
  if (pp > best[mm * 2 + h]) best[mm * 2 + h] = static_cast<int8_t>(pp);

  // Everything below this node lands in the same saturated slot.
  if (i >= len || (mm == 4 && pp == 4 && h == 1)) return;

  if (c[i] >= 3) {
    c[i] -= 3;
    groupDfs(c, len, suited, i, m + 1, p, h, best);
    c[i] += 3;
  }
  if (suited && i + 2 < len && c[i + 1] > 0 && c[i + 2] > 0) {
    c[i]--, c[i + 1]--, c[i + 2]--;
    groupDfs(c, len, suited, i, m + 1, p, h, best);
    c[i]++, c[i + 1]++, c[i + 2]++;
  }
  if (!h && c[i] >= 2) {
    c[i] -= 2;
    groupDfs(c, len, suited, i, m, p, 1, best);
    c[i] += 2;
  }
  if (c[i] >= 2) {
    c[i] -= 2;
    groupDfs(c, len, suited, i, m, p + 1, h, best);
    c[i] += 2;
  }
  if (suited && i + 1 < len && c[i + 1] > 0) {
    c[i]--, c[i + 1]--;
    groupDfs(c, len, suited, i, m, p + 1, h, best);
    c[i]++, c[i + 1]++;
  }
  if (suited && i + 2 < len && c[i + 2] > 0) {
    c[i]--, c[i + 2]--;
    groupDfs(c, len, suited, i, m, p + 1, h, best);
    c[i]++, c[i + 2]++;
  }
  c[i]--;
  groupDfs(c, len, suited, i, m, p, h, best);
  c[i]++;
}

/** Ten nibbles: slot value 0 = unreachable, otherwise partials + 1. */
uint64_t computeWord(const uint8_t *src, int len, bool suited) {
  uint8_t c[9];
  std::memcpy(c, src, static_cast<size_t>(len));
  Summary best;
  for (int k = 0; k < 10; k++) best[k] = -1;
  groupDfs(c, len, suited, 0, 0, 0, 0, best);

  uint64_t w = 0;
  for (int k = 0; k < 10; k++) {
    const uint64_t v = best[k] < 0 ? 0u : static_cast<uint64_t>(best[k] + 1);
    w |= v << (4 * k);
  }
  return w;
}

/**
 * The per-context memo: open addressing, key = the group's base-5 index (with
 * the honor space shifted past the suit space), value = the packed word.
 *
 * A flat 2-million-entry table like `mjkernel.cc`'s would be 16 MB per SEAT
 * here, and real play touches a few thousand distinct groups — so this grows
 * from 16k slots by doubling and stays a rounding error in practice, while a
 * fuzz that walks the whole domain simply pays for what it walks.
 */
struct WordCache {
  uint32_t *keys; // stored key + 1; 0 = empty
  uint64_t *vals;
  uint32_t cap; // power of two
  uint32_t used;
};

inline uint32_t mixKey(uint32_t k) {
  k ^= k >> 16;
  k *= 0x7feb352du;
  k ^= k >> 15;
  k *= 0x846ca68bu;
  k ^= k >> 16;
  return k;
}

bool cacheInit(WordCache &wc, uint32_t cap) {
  wc.keys = static_cast<uint32_t *>(std::calloc(cap, sizeof(uint32_t)));
  wc.vals = static_cast<uint64_t *>(std::malloc(static_cast<size_t>(cap) * sizeof(uint64_t)));
  wc.cap = cap;
  wc.used = 0;
  if (!wc.keys || !wc.vals) {
    std::free(wc.keys);
    std::free(wc.vals);
    wc.keys = nullptr;
    wc.vals = nullptr;
    wc.cap = 0;
    return false;
  }
  return true;
}

void cacheFree(WordCache &wc) {
  std::free(wc.keys);
  std::free(wc.vals);
  wc.keys = nullptr;
  wc.vals = nullptr;
  wc.cap = 0;
  wc.used = 0;
}

void cacheGrow(WordCache &wc) {
  WordCache next;
  if (!cacheInit(next, wc.cap * 2)) return; // out of memory: keep the old table
  const uint32_t mask = next.cap - 1;
  for (uint32_t i = 0; i < wc.cap; i++) {
    if (wc.keys[i] == 0) continue;
    uint32_t j = mixKey(wc.keys[i]) & mask;
    while (next.keys[j] != 0) j = (j + 1) & mask;
    next.keys[j] = wc.keys[i];
    next.vals[j] = wc.vals[i];
    next.used++;
  }
  cacheFree(wc);
  wc = next;
}

uint64_t cachedWord(WordCache &wc, const uint8_t *c, int len, bool suited) {
  uint32_t idx = 0;
  for (int i = len - 1; i >= 0; i--) idx = idx * 5 + c[i];
  const uint32_t key = (suited ? idx : SUIT_STATES + idx) + 1;

  if (wc.cap == 0) return computeWord(c, len, suited);
  uint32_t mask = wc.cap - 1;
  uint32_t j = mixKey(key) & mask;
  // Bounded: `cacheGrow` keeps the old table when it cannot allocate, and an
  // unbounded probe over a full one never returns.
  for (uint32_t step = 0; wc.keys[j] != 0; step++) {
    if (wc.keys[j] == key) return wc.vals[j];
    if (step >= wc.cap) return computeWord(c, len, suited);
    j = (j + 1) & mask;
  }
  const uint64_t w = computeWord(c, len, suited);
  wc.keys[j] = key;
  wc.vals[j] = w;
  wc.used++;
  // Half full is where linear probing stops being cheap.
  if (wc.used * 2 >= wc.cap) cacheGrow(wc);
  return w;
}

/** out = merge(a, word). 10 x 10, head total capped at one. */
inline void mergeWord(const Summary a, uint64_t w, Summary out) {
  for (int k = 0; k < 10; k++) out[k] = -1;
  for (int ma = 0; ma <= 4; ma++) {
    for (int ha = 0; ha < 2; ha++) {
      const int pa = a[ma * 2 + ha];
      if (pa < 0) continue;
      for (int mb = 0; mb <= 4; mb++) {
        for (int hb = 0; hb < 2; hb++) {
          if (ha + hb > 1) continue;
          const int raw = static_cast<int>((w >> (4 * (mb * 2 + hb))) & 0xF);
          if (raw == 0) continue;
          int m = ma + mb;
          if (m > 4) m = 4;
          int p = pa + (raw - 1);
          if (p > 4) p = 4;
          int8_t *o = &out[m * 2 + ha + hb];
          if (p > *o) *o = static_cast<int8_t>(p);
        }
      }
    }
  }
}

/** The `best` of shanten.ts: max decomposition value under `cap` set slots. */
inline int evalSummary(const Summary s, int cap) {
  int best = 0;
  for (int m = 0; m <= 4; m++) {
    for (int h = 0; h < 2; h++) {
      const int p = s[m * 2 + h];
      if (p < 0) continue;
      const int used = m < cap ? m : cap;
      const int room = cap - used;
      const int v = 2 * used + (p < room ? p : room) + h;
      if (v > best) best = v;
    }
  }
  return best;
}

const Summary IDENTITY = {0, -1, -1, -1, -1, -1, -1, -1, -1, -1};

/** Whether the memo path applies. Fuzz reaches the `false` branch; play never does. */
inline bool tabulatable(const uint8_t *counts, int openMelds) {
  if (openMelds < 0 || openMelds > 4) return false;
  for (int g = 0; g < 4; g++) {
    const int len = g == 3 ? 7 : 9;
    int total = 0;
    for (int i = 0; i < len; i++) {
      const int v = counts[g * 9 + i];
      if (v > 4) return false;
      total += v;
    }
    if (total > 14) return false; // keeps a pathological cache fill impossible
  }
  return true;
}

inline void groupWords(WordCache &wc, const uint8_t *counts, uint64_t w[4]) {
  w[0] = cachedWord(wc, counts + 0, 9, true);
  w[1] = cachedWord(wc, counts + 9, 9, true);
  w[2] = cachedWord(wc, counts + 18, 9, true);
  w[3] = cachedWord(wc, counts + 27, 7, false);
}

inline int standardFromWords(const uint64_t w[4], int cap) {
  Summary a, b;
  mergeWord(IDENTITY, w[0], a);
  mergeWord(a, w[1], b);
  mergeWord(b, w[2], a);
  mergeWord(a, w[3], b);
  return 2 * cap - evalSummary(b, cap);
}

int standardShanten(WordCache &wc, const uint8_t *counts, int openMelds) {
  const int cap = 4 - openMelds;
  if (cap < 0) return 8;
  if (!tabulatable(counts, openMelds)) return refStandard(counts, openMelds);
  uint64_t w[4];
  groupWords(wc, counts, w);
  return standardFromWords(w, cap);
}

// --- the two closed-only forms ---

inline int chiitoiFrom(int pairs, int kinds) {
  const int shy = 7 - kinds;
  return 6 - pairs + (shy > 0 ? shy : 0);
}

inline int kokushiFrom(int kinds, int hasPair) { return 13 - kinds - hasPair; }

struct Closed {
  int pairs;
  int kinds;
  int yKinds;
  int yPair;
};

Closed closedStats(const uint8_t *counts) {
  Closed s = {0, 0, 0, 0};
  for (int t = 0; t < 34; t++) {
    if (counts[t] >= 1) s.kinds++;
    if (counts[t] >= 2) s.pairs++;
  }
  for (int k = 0; k < 13; k++) {
    const int t = YAOCHU[k];
    if (counts[t] >= 1) s.yKinds++;
    if (counts[t] >= 2) s.yPair = 1;
  }
  return s;
}

int shantenImpl(WordCache &wc, const uint8_t *counts, int openMelds, int closed) {
  int s = standardShanten(wc, counts, openMelds);
  if (closed && openMelds == 0) {
    const Closed st = closedStats(counts);
    const int ch = chiitoiFrom(st.pairs, st.kinds);
    const int ko = kokushiFrom(st.yKinds, st.yPair);
    if (ch < s) s = ch;
    if (ko < s) s = ko;
  }
  return s;
}

// ===========================================================================
// SCORER — src/decompose.ts + src/fu.ts + src/yaku.ts + src/score.ts
// ===========================================================================

enum BlockKind { BK_RUN = 0, BK_TRIPLET = 1, BK_KAN = 2, BK_PAIR = 3 };

enum WaitShape {
  W_NONE = -1,
  W_RYANMEN = 0,
  W_KANCHAN,
  W_PENCHAN,
  W_SHANPON,
  W_TANKI,
  W_KOKUSHI,
  W_KOKUSHI13,
  W_CHIITOI
};

enum Form { F_STANDARD = 0, F_CHIITOI = 1, F_KOKUSHI = 2 };

struct Block {
  int kind;
  int type;
  int concealed;
};

/**
 * One reading. `blocks` holds the 4 sets EXCLUDING the head for the standard
 * form (called melds first, in call order) and the 7 pairs for 七対子; `pair` is
 * the head, or — for 七対子 — the pair the winning tile completed, which is also
 * listed inside `blocks` (hence `allBlocks` below).
 */
struct Decomp {
  int form;
  int nBlocks;
  Block blocks[7];
  Block pair;
  int winBlock; // -1 ⇒ the winning tile completed `pair`
  int wait;
};

/** The parsed `mjev_score` input. */
struct WinCtx {
  uint8_t counts[34]; // concealed, INCLUDING the winning tile
  uint8_t all[34];    // + meld tiles (a kan contributes 4)
  int nMelds;
  Block called[4];
  int winType;
  int tsumo, riichi, doubleRiichi, ippatsu;
  int rinshan, chankan, haitei, houtei, tenhou, chiihou;
  int seatWind, roundWind;
  int dora[34], ura[34];
  int akaCount;
  int kuitan, kazoe, kiriage, dwFu, ippatsuCfg;
  bool menzen;
};

struct Res {
  int ok;
  int han;
  int fu;
  int base;
  int yakuman;
  int limit;
};

/** Rank readings by (役満数, 飜, 符) — the order the scoring rules prescribe. */
inline bool better(const Res &a, const Res &b) {
  if (a.yakuman != b.yakuman) return a.yakuman > b.yakuman;
  if (a.han != b.han) return a.han > b.han;
  return a.fu > b.fu;
}

inline bool isTripletish(const Block &b) { return b.kind == BK_TRIPLET || b.kind == BK_KAN; }

/** The blocks that partition the hand — 七対子 already lists its head. */
inline int allBlocks(const Decomp &d, Block *out) {
  int n = d.nBlocks;
  for (int i = 0; i < n; i++) out[i] = d.blocks[i];
  if (d.form != F_CHIITOI) out[n++] = d.pair;
  return n;
}

inline bool everyType(const uint8_t *c, bool (*pred)(int)) {
  for (int t = 0; t < 34; t++) {
    if (c[t] > 0 && !pred(t)) return false;
  }
  return true;
}

inline bool anyType(const uint8_t *c, bool (*pred)(int)) {
  for (int t = 0; t < 34; t++) {
    if (c[t] > 0 && pred(t)) return true;
  }
  return false;
}

/** 雀頭が役牌か: dragons, the seat wind, or the round wind. */
inline bool isYakuhaiPair(int type, const WinCtx &w) {
  return type >= 31 || type == w.seatWind || type == w.roundWind;
}

/**
 * Whether `blocks[i]` counts as a *concealed* triplet/kan. A shanpon ron is the
 * one exception: the winning tile came from another player, so the set it
 * completed is a 明刻 even inside a closed hand.
 */
inline bool isConcealedSet(const Decomp &d, int i, const WinCtx &w) {
  const Block &b = d.blocks[i];
  if (!isTripletish(b) || !b.concealed) return false;
  return w.tsumo || i != d.winBlock || d.wait != W_SHANPON;
}

inline int ankouCount(const Decomp &d, const WinCtx &w) {
  int n = 0;
  for (int i = 0; i < d.nBlocks; i++) {
    if (isConcealedSet(d, i, w)) n++;
  }
  return n;
}

inline bool isPinfu(const Decomp &d, const WinCtx &w) {
  if (d.form != F_STANDARD) return false;
  if (!w.menzen) return false;
  for (int i = 0; i < d.nBlocks; i++) {
    if (d.blocks[i].kind != BK_RUN) return false;
  }
  if (d.wait != W_RYANMEN) return false;
  return !isYakuhaiPair(d.pair.type, w);
}

/** Fu for one triplet/kan block, given whether it reads as concealed. */
inline int setFu(const Block &b, bool concealed) {
  const bool cheap = isSimple(b.type);
  if (b.kind == BK_KAN) return concealed ? (cheap ? 16 : 32) : (cheap ? 8 : 16);
  return concealed ? (cheap ? 4 : 8) : (cheap ? 2 : 4);
}

inline int ceil10(int n) { return (n + 9) / 10 * 10; }
inline int ceil100(int n) { return (n + 99) / 100 * 100; }

/** src/fu.ts `countFu`, value only. */
int countFu(const Decomp &d, const WinCtx &w) {
  if (d.form == F_CHIITOI) return 25;
  if (d.form == F_KOKUSHI) return 20; // a yakuman: this 副底 is never paid on

  const bool pinfu = isPinfu(d, w);
  int fu = 20;
  if (w.menzen && !w.tsumo) fu += 10;
  // 平和自摸 is the one hand that forgoes the 自摸符 — that is what pins it to 20.
  if (w.tsumo && !pinfu) fu += 2;

  for (int i = 0; i < d.nBlocks; i++) {
    const Block &b = d.blocks[i];
    if (!isTripletish(b)) continue;
    fu += setFu(b, isConcealedSet(d, i, w));
  }

  const int p = d.pair.type;
  if (p == w.seatWind && p == w.roundWind) fu += w.dwFu;
  else if (isYakuhaiPair(p, w)) fu += 2;

  if (d.wait == W_KANCHAN || d.wait == W_PENCHAN || d.wait == W_TANKI) fu += 2;

  // 喰い平和形: an open hand with no fu-bearing part would score 20, which no
  // ruleset pays — it is settled as a flat 30.
  if (!w.menzen && fu == 20) return 30;
  return ceil10(fu);
}

// --- shape helpers (src/yaku.ts) ---

/** Number of identical-run pairs: 1 ⇒ 一盃口, 2 ⇒ 二盃口. */
int peikoPairs(const Decomp &d) {
  int runs[34] = {0};
  for (int i = 0; i < d.nBlocks; i++) {
    if (d.blocks[i].kind == BK_RUN) runs[d.blocks[i].type]++;
  }
  int n = 0;
  for (int t = 0; t < 34; t++) n += runs[t] / 2;
  return n;
}

bool hasIttsu(const Decomp &d) {
  bool runs[34] = {false};
  for (int i = 0; i < d.nBlocks; i++) {
    if (d.blocks[i].kind == BK_RUN) runs[d.blocks[i].type] = true;
  }
  for (int base = 0; base <= 18; base += 9) {
    if (runs[base] && runs[base + 3] && runs[base + 6]) return true;
  }
  return false;
}

bool hasSanshokuDoujun(const Decomp &d) {
  bool runs[34] = {false};
  for (int i = 0; i < d.nBlocks; i++) {
    if (d.blocks[i].kind == BK_RUN) runs[d.blocks[i].type] = true;
  }
  for (int r = 0; r < 9; r++) {
    if (runs[r] && runs[r + 9] && runs[r + 18]) return true;
  }
  return false;
}

bool hasSanshokuDoukou(const Decomp &d) {
  bool trips[34] = {false};
  for (int i = 0; i < d.nBlocks; i++) {
    if (isTripletish(d.blocks[i])) trips[d.blocks[i].type] = true;
  }
  for (int t = 0; t < 9; t++) {
    if (trips[t] && trips[t + 9] && trips[t + 18]) return true;
  }
  return false;
}

/** 45 九蓮宝燈 / 46 純正九蓮宝燈 / 0 for neither. `counts` is the concealed hand. */
int chuurenId(const WinCtx &w) {
  if (w.nMelds > 0) return 0;
  // The TypeScript reads the suit off `hand[0]`; with every tile required to
  // sit in one numeric suit below, the lowest held type answers the same.
  int first = -1;
  for (int t = 0; t < 34; t++) {
    if (w.counts[t] > 0) {
      first = t;
      break;
    }
  }
  if (first < 0 || isHonor(first)) return 0;
  const int base = suitOf(first) * 9;
  for (int t = 0; t < 34; t++) {
    if (w.counts[t] > 0 && (t < base || t >= base + 9)) return 0;
  }
  const int need[9] = {3, 1, 1, 1, 1, 1, 1, 1, 3};
  for (int r = 0; r < 9; r++) {
    if (w.counts[base + r] < need[r]) return 0;
  }
  // 純正 = the 13 tiles before the win were exactly 1112345678999.
  for (int r = 0; r < 9; r++) {
    const int c = w.counts[base + r] - (base + r == w.winType ? 1 : 0);
    if (c != need[r]) return 45;
  }
  return 46;
}

/** src/yaku.ts `detectYakuman`, count only (every id is worth one 役満). */
int detectYakumanCount(const Decomp &d, const WinCtx &w) {
  int n = 0;
  if (w.tenhou) n++;  // 37
  if (w.chiihou) n++; // 38

  if (d.form == F_KOKUSHI) return n + 1; // 47 / 48 — and nothing else applies

  if (everyType(w.all, isHonor)) n++;    // 42 字一色
  if (everyType(w.all, isGreen)) n++;    // 43 緑一色
  if (everyType(w.all, isTerminal)) n++; // 44 清老頭

  if (d.form == F_STANDARD) {
    int dragons = 0, winds = 0, kans = 0;
    for (int i = 0; i < d.nBlocks; i++) {
      const Block &b = d.blocks[i];
      if (b.kind == BK_KAN) kans++;
      if (!isTripletish(b)) continue;
      if (b.type >= 31) dragons++;
      if (b.type >= 27 && b.type <= 30) winds++;
    }
    if (dragons == 3) n++; // 39 大三元
    if (winds == 4) n++;   // 49 大四喜
    else if (winds == 3 && d.pair.type >= 27 && d.pair.type <= 30) n++; // 50 小四喜
    if (kans == 4) n++;                                                // 51 四槓子
    if (ankouCount(d, w) == 4) n++;                                    // 40 / 41
  }

  if (chuurenId(w)) n++; // 45 / 46
  return n;
}

/** src/yaku.ts `detectYaku`, han total only. */
int detectYakuHan(const Decomp &d, const WinCtx &w) {
  int han = 0;
  const bool menzen = w.menzen;
  // 喰い下がり: one han less once the hand is open.
  const int kd = menzen ? 0 : 1;

  // --- situational ---
  if (menzen && w.tsumo) han += 1; // 0 門前清自摸和
  if (w.doubleRiichi) han += 2;    // 21 両立直
  else if (w.riichi) han += 1;     // 1 立直
  if (w.ippatsu && w.ippatsuCfg && (w.riichi || w.doubleRiichi)) han += 1; // 2
  if (w.chankan) han += 1;                                                 // 3
  if (w.rinshan) han += 1;                                                 // 4
  if (w.haitei && w.tsumo) han += 1;                                       // 5
  if (w.houtei && !w.tsumo) han += 1;                                      // 6

  Block bl[8];
  const int nb = allBlocks(d, bl);
  bool hasRun = false;
  for (int i = 0; i < nb; i++) {
    if (bl[i].kind == BK_RUN) hasRun = true;
  }
  const bool hasHonor = anyType(w.all, isHonor);

  if (d.form == F_CHIITOI) han += 2; // 22

  if (d.form == F_STANDARD) {
    if (isPinfu(d, w)) han += 1; // 7

    // 役牌: a wind can score twice when it is both the seat and the round wind.
    int kans = 0, dragons = 0;
    bool allTrips = true;
    for (int i = 0; i < d.nBlocks; i++) {
      const Block &b = d.blocks[i];
      if (b.kind == BK_KAN) kans++;
      if (!isTripletish(b)) {
        allTrips = false;
        continue;
      }
      if (b.type == w.seatWind) han += 1;  // 10..13
      if (b.type == w.roundWind) han += 1; // 14..17
      if (b.type >= 31) {
        han += 1; // 18..20
        dragons++;
      }
    }

    if (menzen) {
      const int peiko = peikoPairs(d);
      if (peiko >= 2) han += 3;      // 32 二盃口
      else if (peiko == 1) han += 1; // 9 一盃口
    }
    if (hasIttsu(d)) han += 2 - kd;           // 24
    if (hasSanshokuDoujun(d)) han += 2 - kd;  // 25
    if (hasSanshokuDoukou(d)) han += 2;       // 26
    if (kans == 3) han += 2;                  // 27 三槓子
    if (allTrips) han += 2;                   // 28 対々和
    if (ankouCount(d, w) == 3) han += 2;      // 29 三暗刻
    if (dragons == 2 && d.pair.type >= 31) han += 2; // 30 小三元
  }

  // --- 幺九系 ---
  if (everyType(w.all, isYaochu) && hasHonor && !hasRun) {
    han += 2; // 31 混老頭 (清老頭 is a yakuman and never reaches here)
  } else if (d.form == F_STANDARD && hasRun) {
    bool everyTerm = true, everyYao = true;
    for (int i = 0; i < nb; i++) {
      const Block &b = bl[i];
      const int lo = b.type;
      const int hi = b.kind == BK_RUN ? b.type + 2 : b.type;
      bool term = false, yao = false;
      for (int t = lo; t <= hi; t++) {
        if (isTerminal(t)) term = true;
        if (isYaochu(t)) yao = true;
      }
      if (!term) everyTerm = false;
      if (!yao) everyYao = false;
    }
    if (everyTerm) han += 3 - kd;     // 33 純全帯幺九
    else if (everyYao) han += 2 - kd; // 23 混全帯幺九
  }

  if (everyType(w.all, isSimple) && (menzen || w.kuitan)) han += 1; // 8 断幺九

  int suits = 0;
  for (int s = 0; s < 3; s++) {
    for (int r = 0; r < 9; r++) {
      if (w.all[s * 9 + r] > 0) {
        suits++;
        break;
      }
    }
  }
  if (suits == 1) {
    if (hasHonor) han += 3 - kd; // 34 混一色
    else han += 6 - kd;          // 35 清一色
  }

  return han;
}

/** src/yaku.ts `basePoints`. */
void basePoints(int han, int fu, const WinCtx &w, int *base, int *limit) {
  if (han >= 13) {
    if (w.kazoe) {
      *base = 8000;
      *limit = 5;
    } else {
      *base = 6000;
      *limit = 4;
    }
    return;
  }
  if (han >= 11) {
    *base = 6000;
    *limit = 4;
    return;
  }
  if (han >= 8) {
    *base = 4000;
    *limit = 3;
    return;
  }
  if (han >= 6) {
    *base = 3000;
    *limit = 2;
    return;
  }
  if (han >= 5) {
    *base = 2000;
    *limit = 1;
    return;
  }
  const int b = fu * (1 << (2 + han));
  if (b >= 2000) {
    *base = 2000;
    *limit = 1;
    return;
  }
  // 切り上げ満貫 promotes exactly the 1920 cell (4飜30符 / 3飜60符).
  if (w.kiriage && b >= 1920) {
    *base = 2000;
    *limit = 1;
    return;
  }
  *base = b;
  *limit = 0;
}

/** src/yaku.ts `evaluate`. `ok = 0` ⇒ 役なし: dora alone never opens a win. */
Res evaluate(const Decomp &d, const WinCtx &w) {
  Res r = {0, 0, 0, 0, 0, 0};
  r.fu = countFu(d, w);

  const int ym = detectYakumanCount(d, w);
  if (ym > 0) {
    // A yakuman suppresses every normal yaku and all dora.
    r.ok = 1;
    r.yakuman = ym;
    r.han = 13 * ym;
    r.base = 8000 * ym;
    r.limit = 5;
    return r;
  }

  const int yakuHan = detectYakuHan(d, w);
  if (yakuHan == 0) return r; // ok stays 0

  int dora = 0, ura = 0;
  for (int t = 0; t < 34; t++) {
    dora += w.all[t] * w.dora[t];
    ura += w.all[t] * w.ura[t];
  }
  r.ok = 1;
  r.han = yakuHan + dora + ura + w.akaCount;
  basePoints(r.han, r.fu, w, &r.base, &r.limit);
  return r;
}

// --- decomposeWin, streamed straight into `evaluate` ---
//
// The TypeScript materialises every reading and dedups them by block code; here
// each reading is scored the moment it is formed and only the running maximum
// survives. Dedup is then unnecessary rather than skipped: duplicate readings
// are duplicates *of the same blocks*, so they evaluate to the same `Res` and
// cannot move a maximum.

struct Enumerator {
  const WinCtx *w;
  uint8_t c[34];
  Block acc[4];
  int pairType;
  Res best;

  void consider(const Decomp &d) {
    const Res r = evaluate(d, *w);
    if (!r.ok) return;
    if (!best.ok || better(r, best)) best = r;
  }

  /** Which wait shape does `winType` complete `b` with? W_NONE if it does not. */
  int waitFor(const Block &b) const {
    const int winType = w->winType;
    if (b.kind == BK_TRIPLET) return b.type == winType ? W_SHANPON : W_NONE;
    if (b.kind != BK_RUN) return W_NONE;
    const int lo = b.type;
    if (winType == lo + 1) return W_KANCHAN;
    if (winType == lo) return lo % 9 == 6 ? W_PENCHAN : W_RYANMEN; // 7 of 789
    if (winType == lo + 2) return lo % 9 == 0 ? W_PENCHAN : W_RYANMEN; // 3 of 123
    return W_NONE;
  }

  void emit(int nAcc) {
    Decomp d;
    d.form = F_STANDARD;
    d.nBlocks = 4;
    for (int i = 0; i < w->nMelds; i++) d.blocks[i] = w->called[i];
    // Canonical order (type, run before triplet) — cosmetic here, but it is
    // what the TypeScript's dedup key is built on.
    Block s[4];
    for (int i = 0; i < nAcc; i++) s[i] = acc[i];
    for (int i = 1; i < nAcc; i++) {
      Block key = s[i];
      int j = i - 1;
      while (j >= 0 &&
             (s[j].type > key.type ||
              (s[j].type == key.type && s[j].kind != key.kind && key.kind == BK_RUN))) {
        s[j + 1] = s[j];
        j--;
      }
      s[j + 1] = key;
    }
    for (int i = 0; i < nAcc; i++) d.blocks[w->nMelds + i] = s[i];
    d.pair.kind = BK_PAIR;
    d.pair.type = pairType;
    d.pair.concealed = 1;

    if (pairType == w->winType) {
      d.winBlock = -1;
      d.wait = W_TANKI;
      consider(d);
    }
    // A called meld is never the winning block: it was complete before the win.
    for (int i = w->nMelds; i < 4; i++) {
      const int wt = waitFor(d.blocks[i]);
      if (wt == W_NONE) continue;
      d.winBlock = i;
      d.wait = wt;
      consider(d);
    }
  }

  /** src/decompose.ts `peelSets`: peel `need` sets off the lowest non-empty type. */
  void peel(int i, int nAcc, int need_) {
    while (i < 34 && c[i] == 0) i++;
    if (i == 34) {
      if (need_ == 0) emit(nAcc);
      return;
    }
    if (need_ == 0) return; // leftover tiles ⇒ not a partition

    if (c[i] >= 3) {
      c[i] -= 3;
      acc[nAcc].kind = BK_TRIPLET;
      acc[nAcc].type = i;
      acc[nAcc].concealed = 1;
      peel(i, nAcc + 1, need_ - 1);
      c[i] += 3;
    }
    const int rank = i < 27 ? i % 9 : -1; // honors form no runs
    if (rank >= 0 && rank <= 6 && c[i + 1] > 0 && c[i + 2] > 0) {
      c[i]--, c[i + 1]--, c[i + 2]--;
      acc[nAcc].kind = BK_RUN;
      acc[nAcc].type = i;
      acc[nAcc].concealed = 1;
      peel(i, nAcc + 1, need_ - 1);
      c[i]++, c[i + 1]++, c[i + 2]++;
    }
  }
};

bool isChiitoiShape(const uint8_t *c) {
  int pairs = 0;
  for (int t = 0; t < 34; t++) {
    if (c[t] == 0) continue;
    if (c[t] != 2) return false;
    pairs++;
  }
  return pairs == 7;
}

/** The doubled yaochu type of a complete kokushi, or -1. */
int kokushiPairType(const uint8_t *c) {
  int doubled = -1;
  for (int k = 0; k < 13; k++) {
    const int t = YAOCHU[k];
    if (c[t] == 0) return -1;
    if (c[t] == 2) {
      if (doubled >= 0) return -1;
      doubled = t;
    } else if (c[t] != 1) {
      return -1;
    }
  }
  return doubled;
}

/** src/yaku.ts `scoreWin`: the best reading, or `ok = 0`. */
Res scoreWin(const WinCtx &w) {
  Res none = {0, 0, 0, 0, 0, 0};
  const int need = 4 - w.nMelds;
  if (need < 0) return none;
  if (w.winType < 0 || w.winType >= 34) return none;
  if (w.counts[w.winType] == 0) return none; // the winning tile must be in the hand

  int total = 0;
  for (int t = 0; t < 34; t++) total += w.counts[t];
  if (total != need * 3 + 2) return none;

  Enumerator e;
  e.w = &w;
  e.best = none;
  std::memcpy(e.c, w.counts, 34);

  // --- standard form: a head plus 4 sets (melds fill the first `nMelds`) ---
  for (int p = 0; p < 34; p++) {
    if (e.c[p] < 2) continue;
    e.c[p] -= 2;
    e.pairType = p;
    e.peel(0, 0, need);
    e.c[p] += 2;
  }

  // --- irregular forms: closed hands only ---
  if (w.nMelds == 0) {
    if (isChiitoiShape(w.counts)) {
      Decomp d;
      d.form = F_CHIITOI;
      d.nBlocks = 0;
      for (int t = 0; t < 34; t++) {
        if (w.counts[t] == 2) {
          d.blocks[d.nBlocks].kind = BK_PAIR;
          d.blocks[d.nBlocks].type = t;
          d.blocks[d.nBlocks].concealed = 1;
          d.nBlocks++;
        }
      }
      d.pair.kind = BK_PAIR;
      d.pair.type = w.winType;
      d.pair.concealed = 1;
      d.winBlock = -1;
      d.wait = W_CHIITOI;
      e.consider(d);
    }
    const int doubled = kokushiPairType(w.counts);
    if (doubled >= 0) {
      Decomp d;
      d.form = F_KOKUSHI;
      d.nBlocks = 0;
      d.pair.kind = BK_PAIR;
      d.pair.type = doubled;
      d.pair.concealed = 1;
      d.winBlock = -1;
      // Drawing the 14th of 13 distinct yaochu ⇒ the hand was a 13-sided wait.
      d.wait = w.counts[w.winType] == 2 ? W_KOKUSHI13 : W_KOKUSHI;
      e.consider(d);
    }
  }

  return e.best;
}
// ===========================================================================
// UKEIRE — the mask, factored out so the DP and the ABI share one body
// ===========================================================================

/**
 * Bit t set ⇔ adding one tile of type t drops shanten below `base`.
 * `mjev_ukeire_mask` is a thin wrapper; the DP calls this directly.
 */
uint64_t ukeireMaskImpl(WordCache &wc, const uint8_t *counts, int openMelds, int closed,
                        int base) {
  uint8_t c[34];
  std::memcpy(c, counts, 34);

  const int cap = 4 - openMelds;
  const bool cl = closed && openMelds == 0;
  uint64_t mask = 0;

  // Slow, exact path: whatever the memo cannot index, the reference DFS can.
  if (cap < 0 || !tabulatable(c, openMelds)) {
    for (int t = 0; t < 34; t++) {
      if (c[t] >= 4) continue;
      c[t]++;
      const int s = shantenImpl(wc, c, openMelds, closed);
      c[t]--;
      if (s < base) mask |= 1ull << t;
    }
    return mask;
  }

  uint64_t w[4];
  groupWords(wc, c, w);

  // For each group, the merge of the OTHER three — the part a probe cannot move.
  Summary rest[4];
  for (int g = 0; g < 4; g++) {
    Summary a, b;
    std::memcpy(a, IDENTITY, sizeof(Summary));
    bool flip = false;
    for (int k = 0; k < 4; k++) {
      if (k == g) continue;
      if (flip) mergeWord(b, w[k], a);
      else mergeWord(a, w[k], b);
      flip = !flip;
    }
    std::memcpy(rest[g], flip ? b : a, sizeof(Summary));
  }

  const Closed st = cl ? closedStats(c) : Closed{0, 0, 0, 0};

  for (int t = 0; t < 34; t++) {
    if (c[t] >= 4) continue;
    const int g = t < 27 ? t / 9 : 3;
    const int len = g == 3 ? 7 : 9;
    const uint8_t *src = c + (g == 3 ? 27 : g * 9);

    uint8_t tmp[9];
    std::memcpy(tmp, src, static_cast<size_t>(len));
    tmp[t - (g == 3 ? 27 : g * 9)]++;

    Summary merged;
    mergeWord(rest[g], cachedWord(wc, tmp, len, g != 3), merged);
    int s = 2 * cap - evalSummary(merged, cap);

    if (cl) {
      const int kinds = st.kinds + (c[t] == 0 ? 1 : 0);
      const int pairs = st.pairs + (c[t] == 1 ? 1 : 0);
      const int ch = chiitoiFrom(pairs, kinds);
      if (ch < s) s = ch;
      bool yaochu = false;
      for (int k = 0; k < 13; k++) {
        if (YAOCHU[k] == t) {
          yaochu = true;
          break;
        }
      }
      const int yk = st.yKinds + (yaochu && c[t] == 0 ? 1 : 0);
      const int yp = (st.yPair || (yaochu && c[t] >= 1)) ? 1 : 0;
      const int ko = kokushiFrom(yk, yp);
      if (ko < s) s = ko;
    }

    if (s < base) mask |= 1ull << t;
  }
  return mask;
}

// ===========================================================================
// THE DP (unit B) — an expected-value search over our own remaining draws
// ===========================================================================
//
// WHAT THIS COMPUTES. A state is a 13-tile RESTING shape, the number of own
// draws already taken, how many red 5p it holds, and whether we have declared
// riichi. Its value is the expected points, from here to the end of the hand,
// of playing it out: our wins (scored exactly by the integer scorer above),
// the opponents' wins and our deal-ins (population rates over the M14 reads,
// never a per-seat sequential inference — the 雀鬼流 constitution), and the
// 流局 settlement. Everything is POINTS; nothing here is a score unit.
//
// TURN SHAPE. One turn is: we draw (we may win, or improve, or throw the tile
// straight back), then the other three seats act (the hazard). That is the
// same order `handvalue.ts:201-243` walks, and the tail below IS that chain.
//
// PROBABILITY (plan D1, "pool by net absorption"). Our future draws are a
// uniform sequence WITHOUT replacement from the unseen pool, so the copies of
// type k still available in a state are `pool_k = base_k − max(0, hand_k −
// root_k)` — what the root could not see, less what this line of play has
// absorbed since — and the denominator at turn j is `N_j = N_root − j`. Both
// halves matter: the shrinking denominator is what makes a wait's chance
// exactly hypergeometric on a no-shape-change path (P(miss T times) =
// C(N−w,T)/C(N,T), which `test/ev_native_test.ts` asserts to 1e-9), and the
// per-type absorption is what stops a hand from drawing the same tile twice.
//
// ⚑ The tiles we THREW AWAY are not removed from the pool (we collapse every
// uninteresting draw into one branch, so we do not know which tile it was).
// The excess probability mass that creates — Σ_k pool_k/N_j > 1 once we have
// discarded anything — is charged to the collapsed tsumogiri branch, which is
// exactly where those tiles went. Win and improvement probabilities therefore
// stay exact; only the "nothing happened" branch absorbs the approximation.
//
// DETERMINISM. Pure function of (params, ints, dbls): no globals, no threads,
// no clock, no randomness, every loop in a fixed order (candidates ascending,
// draw types ascending, discards ascending), and a node budget that truncates
// to the closed-form tail at exactly the same place every time. The caches
// live in the context and are cleared per evaluation, so two contexts fed the
// same bytes answer the same bits — `--jobs` identity depends on it.

enum { T_MAX = 20, AKA_MAX = 3, EXTRA_MAX = 12, MAX_WAITS = 13 };

/**
 * What the closed-form tail has to be multiplied by to sit on the same scale as
 * the exact search it hands over to.
 *
 * `handvalue.ts`'s scalars were never calibrated as POINTS. The incumbent
 * consumes its chain through `evWeight` (0.1) and `pushScale`, so only the
 * ORDERING of its output ever mattered and `valueRiichi = 7000` is a number on
 * a scale of its own. The DP cannot use it that way: at a 3向聴 root every
 * candidate is priced by the tail, at a 2向聴 root the shanten-keeping discards
 * are priced by the exact search and the rest by the tail, and the two have to
 * be comparable or the seat simply prefers whichever side of `exactShanten` it
 * is standing on. (It did: it never advanced past 3向聴, 聴牌率 20%.)
 *
 * The constant is MEASURED, not chosen: `test/ev_native_test.ts` prices 200
 * random 1向聴/2向聴 rests both ways with every hazard silent and asserts the
 * mean of exact/tail lands within ±20% of 1. Re-run it after any change to
 * either model — a drift here is a seam, and a seam is a seat that stops
 * advancing.
 */
// Isotonic (non-increasing) by construction. The free per-level fit came back
// 0.40/0.40/0.40/0.46/0.71/1.25 — RISING past 3向聴, which lets a 5向聴 hand
// outscore a 4向聴 one however monotone the raw chain is (measured: 巡6 keep-sh5
// 1448 against advance-to-sh4 1365). The measured values are kept where they do
// not break the ordering and clamped to the running minimum where they do; the
// levels above 3向聴 were the ones fitted against a reference that was itself
// mostly tail, so they are the ones with least claim to survive the constraint.
/**
 * The per-rung calibration of the tail's WIN TERM against the exact search.
 *
 * ⚑ RE-MEASURED, NOT MOVED (2026-08-31). The 2026-08-30 numbers were fitted
 * while `cal` scaled the whole tail; with the hazard half now unscaled (see
 * `tailValue`) the same constants were re-measured on 40 random rests a rung,
 * `maxNodes` 2,000,000, and they are still the right ones — what changed is
 * that they now hold on the table the seat plays at:
 *
 *   | Σ聴牌率 | 1向聴 | 2向聴 | 3向聴 | 4向聴 | 5向聴 | 6向聴 |
 *   | 0       | 0.911 | 0.767 | 0.781 | 0.931 | 1.363 | 1.638 |
 *   | 0.15    | 2.132 | 1.552 | 1.106 | 0.975 | 0.981 | 0.988 |
 *   | 0.45    | 1.327 | 1.059 | 0.993 | 1.000 | 1.000 | 1.000 |
 *
 * against −3.64 / 2.91 / 10.34 / 1.88 for the same cells before the split.
 * Where the tail is the model that actually governs — a field whose worst
 * candidate is 4向聴+, i.e. junk at 巡1-3 — the live arms are 0.97–1.11.
 *
 * A LIVE RE-FIT WAS TRIED AND DOES NOT CONVERGE, and the reason is worth
 * keeping: from 2向聴 up the exact reference truncates into the very tail it is
 * being compared against (22/40 at 3向聴, 30/40 at 4向聴 at `maxNodes`
 * 300,000), so raising `cal[s]` raises both sides and the ratio barely moves —
 * four damped iterations walked `cal[3]` 0.395 → 0.526 with the ratio still at
 * 1.14. At 1–2向聴 the live ratio is ill-conditioned in the other direction:
 * the exact and tail values straddle zero there (fold is `max`-ed in and does
 * not scale with `cal` at all), and the mean ratio swung −0.70, 4.57, 0.82,
 * 1.82 over the same four iterations. The silent arm IS well conditioned and
 * would ask for 0.374 / 0.303 at rungs 1–2; a compromise between the two arms
 * lands back within a few percent of the numbers below. So they stay, the
 * arms are printed by `test/ev_native_test.ts`, and the silent-table error
 * (the tail reads ~1.3× high there) is the one being accepted.
 */
const double TAIL_CAL[7] = {1, 0.4102, 0.3951, 0.3951, 0.3951, 0.3951, 0.3951};

/** The calibration for a hand `s` away, clamped to the table above. */
inline double tailCal(int s) {
  const int i = s < 0 ? 0 : (s > 6 ? 6 : s);
  return TAIL_CAL[i];
}

inline double clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

/** The dora type an indicator of type `t` names (mjrender `doraFromIndicatorType`). */
inline int doraFromIndicator(int t) {
  if (t < 27) {
    const int base = (t / 9) * 9;
    return base + ((t - base + 1) % 9);
  }
  if (t <= 30) return t == 30 ? 27 : t + 1;
  return t == 33 ? 31 : t + 1;
}

// ---------------------------------------------------------------------------
// growable arrays and open-addressing tables (per context, cleared per eval)
// ---------------------------------------------------------------------------

template <typename T>
struct Arr {
  T *v;
  uint32_t n, cap;
};

template <typename T>
bool arrReserve(Arr<T> &a, uint32_t need) {
  if (need <= a.cap) return true;
  uint32_t cap = a.cap ? a.cap : 64;
  while (cap < need) cap *= 2;
  T *nv = static_cast<T *>(std::realloc(a.v, static_cast<size_t>(cap) * sizeof(T)));
  if (!nv) return false;
  a.v = nv;
  a.cap = cap;
  return true;
}

template <typename T>
int32_t arrPush(Arr<T> &a, const T &x) {
  if (!arrReserve(a, a.n + 1)) return -1;
  a.v[a.n] = x;
  return static_cast<int32_t>(a.n++);
}

template <typename T>
void arrFree(Arr<T> &a) {
  std::free(a.v);
  a.v = nullptr;
  a.n = 0;
  a.cap = 0;
}

/** uint64 key → uint32 value. Keys are stored +1, so 0 is the empty slot. */
struct Table {
  uint64_t *keys;
  uint32_t *vals;
  uint32_t cap, used;
};

inline uint64_t mix64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ull;
  x ^= x >> 33;
  return x;
}

bool tableInit(Table &t, uint32_t cap) {
  t.keys = static_cast<uint64_t *>(std::calloc(cap, sizeof(uint64_t)));
  t.vals = static_cast<uint32_t *>(std::malloc(static_cast<size_t>(cap) * sizeof(uint32_t)));
  t.cap = cap;
  t.used = 0;
  if (!t.keys || !t.vals) {
    std::free(t.keys);
    std::free(t.vals);
    t.keys = nullptr;
    t.vals = nullptr;
    t.cap = 0;
    return false;
  }
  return true;
}

void tableFree(Table &t) {
  std::free(t.keys);
  std::free(t.vals);
  t.keys = nullptr;
  t.vals = nullptr;
  t.cap = 0;
  t.used = 0;
}

void tableClear(Table &t) {
  if (t.keys) std::memset(t.keys, 0, static_cast<size_t>(t.cap) * sizeof(uint64_t));
  t.used = 0;
}

bool tableGrow(Table &t) {
  Table next;
  if (!tableInit(next, t.cap ? t.cap * 2 : 1024)) return false;
  const uint32_t mask = next.cap - 1;
  for (uint32_t i = 0; i < t.cap; i++) {
    if (t.keys[i] == 0) continue;
    uint32_t j = static_cast<uint32_t>(mix64(t.keys[i])) & mask;
    while (next.keys[j] != 0) j = (j + 1) & mask;
    next.keys[j] = t.keys[i];
    next.vals[j] = t.vals[i];
    next.used++;
  }
  tableFree(t);
  t = next;
  return true;
}

/** The value stored under `key`, or -1. */
int64_t tableGet(const Table &t, uint64_t key) {
  if (!t.cap) return -1;
  const uint64_t stored = key + 1;
  const uint32_t mask = t.cap - 1;
  uint32_t j = static_cast<uint32_t>(mix64(stored)) & mask;
  while (t.keys[j] != 0) {
    if (t.keys[j] == stored) return static_cast<int64_t>(t.vals[j]);
    j = (j + 1) & mask;
  }
  return -1;
}

void tablePut(Table &t, uint64_t key, uint32_t val) {
  if (!t.cap) return;
  if ((t.used + 1) * 2 >= t.cap && !tableGrow(t)) return;
  const uint64_t stored = key + 1;
  const uint32_t mask = t.cap - 1;
  uint32_t j = static_cast<uint32_t>(mix64(stored)) & mask;
  while (t.keys[j] != 0) {
    if (t.keys[j] == stored) {
      t.vals[j] = val;
      return;
    }
    j = (j + 1) & mask;
  }
  t.keys[j] = stored;
  t.vals[j] = val;
  t.used++;
}

// ---------------------------------------------------------------------------
// the search's own records
// ---------------------------------------------------------------------------

/**
 * The widest fan `ensureEdges` can keep after an accepting draw: one slot per
 * tile type, so "keep everything" needs no special case in the ranking. The
 * arrays are one longer for the reserved dora slot.
 */
const int KEEP_MAX = 34;

/**
 * 待ち替え lines kept, ranked by expected win value. TWO since 2026-08-31:
 * with one, a 聴牌 hand could only ever weigh a single wait swap against
 * standing pat, and the oracle-mode diff put 5.7% mean / 18.9% worst of the
 * pruning loss on exactly these 1向聴/聴牌 rests.
 */
const int UPGRADE_KEEP = 2;

/**
 * One 13-tile resting shape, interned once per `mjev_eval_*` call and SHARED by
 * every root candidate of it.
 *
 * WHAT MAY LIVE HERE AND WHAT MAY NOT (the 529/203 leak class, 2026-08-30). A
 * discard root prices `nCand` candidates, and each of them holds a DIFFERENT
 * 13-tile root, so the draw pool `base − max(0, hand − root)` differs per
 * candidate. Anything computed off that pool is therefore candidate-scoped and
 * may not be cached here unguarded — that is exactly what made the same tile
 * come back at a different price depending on which candidates preceded it.
 *
 * So the fields split in two, and the split is the whole safety argument:
 *
 *   GEOMETRY, a pure function of `c` (plus the melds/rules, which are fixed for
 *   the root): `shanten`, `accept`, `statExtra`, `honitsu`, and the per-draw
 *   `Geo` below. Computed once, read by every candidate.
 *
 *   POOL-DEPENDENT: `mass` and the successor edge list (whose selection AND
 *   ranking run off the mass). Stamped with the candidate generation
 *   `EvCtx::gen` and recomputed the first time each candidate asks.
 *
 * Everything else the search caches — the value memo, the wait prices — already
 * carries the thrown type in its key and is cleared per candidate anyway.
 */
struct Shape {
  uint8_t c[34];
  uint64_t accept; // draws that lower shanten (34 bits) — GEOMETRY
  double mass;     // Σ pool over `accept` — POOL-DEPENDENT, stamped `massGen`
  double statExtra; // valuePerDora·dora(aka excluded) + valueYakuhai·(triplets + ½ pairs)
  int32_t edgeFrom, edgeTo; // POOL-DEPENDENT, stamped `edgeGen`
  int32_t geo;              // index into `EvCtx::geos`, or −1 — GEOMETRY
  uint32_t massGen, edgeGen;
  int8_t shanten;
  uint8_t haveAccept; // `accept` filled (the MASK, not the mass)
  uint8_t haveStat;   // `statExtra`/`honitsu` filled
  uint8_t honitsu;
};

/**
 * The per-draw half of a shape's geometry: for each drawn type `k`, what the
 * best post-draw 向聴 is and which discards achieve it — `discardAnalysis`'s
 * whole answer, which is a pure function of the 14 counts and nothing else.
 *
 * It is the hot loop of the search (one `groupWords` plus a merge per held
 * type, per draw type, per expanded shape), and before candidates shared the
 * arena every one of them paid it again for the same tiles. `best[k] == NOT_YET`
 * marks a draw nobody has asked about.
 */
const int8_t GEO_NOT_YET = -128;

struct Geo {
  uint64_t mask[34];
  int8_t best[34];
};

/**
 * How many shapes may carry a `Geo`. It is 312 bytes a shape, so this is a
 * memory ceiling and not a correctness one: past it `ensureEdges` simply
 * recomputes `discardAnalysis` the way it always did.
 */
const uint32_t GEO_CAP = 1u << 15;

/**
 * One considered continuation: after drawing `k` we throw `d` and land in
 * `succ`. `d == k` is the tsumogiri edge (only ever emitted on the 待ち替え
 * branch, where standing pat competes with the upgrade).
 */
struct Edge {
  int32_t succ;
  int16_t k;
  int16_t d;
};

/**
 * The cashable value of every wait of one shape, for one (riichi, aka) pair —
 * points already scaled by 順位効用 `gain`, 本場/供託/立直棒 included. Cached
 * because it does not depend on the turn: only the probability of collecting
 * it does.
 */
struct WinVals {
  double tsumo[MAX_WAITS];
  double ron[MAX_WAITS];
  uint8_t waitType[MAX_WAITS];
  uint8_t cashT[MAX_WAITS];
  uint8_t cashR[MAX_WAITS];
  uint8_t n;
};

/**
 * One seat's evaluation context: the parameter vector, the shanten memo, and
 * the DP's four arenas plus their indexes. Everything after `cache` is scratch
 * that `clearEval` empties at the start of every evaluation — kept alive across
 * calls only so a self-play run's millions of decisions allocate once.
 */
struct EvCtx {
  double p[EV_PARAMS_LEN];
  WordCache cache;
  Arr<Shape> shapes;
  Arr<Geo> geos;
  Arr<Edge> edges;
  Arr<double> vals;
  Arr<WinVals> wins;
  Table shapeTab; // counts hash  → shape id
  Table memo;     // (shape, turn, aka, riichi) → value index
  Table winTab;   // (shape, aka, riichi, 一発) → wait-price index
  /**
   * The CANDIDATE generation. Bumped by `clearCand` between the root candidates
   * of one `mjev_eval_discard` (and by `clearEval` between calls); every
   * pool-dependent field of a `Shape` carries the generation it was computed
   * under, so a stale one is a recompute rather than a wrong price. This is
   * what lets the shape arena be shared without re-opening the 529/203 leak.
   */
  uint32_t gen;
};

/** Cache slots to start from; play touches a few thousand distinct groups. */
const uint32_t CACHE_START = 1u << 14;

/** Everything one `mjev_eval_*` call needs; rebuilt per call, caches reused. */
struct Eval {
  EvCtx *X;
  const double *p;

  uint8_t root[34]; // the ROOT hand (14 tiles for a discard root, 13 for a rest)
  uint8_t meldC[34]; // meld tiles, a kan counting 4
  int nMelds, nKans;
  Block melds[4];
  int closed, seatWind, roundWind, dealer, honba, kyotaku;
  int ownRiichi, furiten, kanDoraOn;
  int T;
  int akaUnseen;
  int dora[34];
  int river[34]; // our OWN discards, as a bag — the 振聴 proof
  int thrown;    // the candidate type this search is standing behind (34 = none)
  double base[34]; // the root draw pool (unseen, or the supplied composition)
  double Nroot;
  uint64_t forced; // types a supplied `drawDist` row insists on

  int hasDraw, K, hasUra, hasNextDora;
  const double *drawDist;
  const double *uraDist;
  const double *nextDoraDist;
  double indP[34]; // P(a hidden indicator is type t) from the pool

  double tenpaiP[3];
  double sumTenpai, meanLoss, gain, risk;

  int64_t nodes, maxNodes;
  int64_t escapes; // times the budget escape fired — a value built over one is not memoised
  uint32_t shapeCap, shapeCapTotal;
  int trunc;
  int riichiOn;    // always 1: declaring is a legal action whoever decides to take it
  int riichiReport; // `ev.riichi` — whether the ROOT hands a riichi decision back
  int exactShanten, sameRungs;
  // ORACLE MODE (a NEGATIVE `maxNodes`): every discard at every state is a
  // successor — no top-three by acceptance mass, no 待ち替え gate, no reserved
  // dora slot, and shanten-RAISING discards enumerated too. It exists so the
  // recursion can be diffed against a brute-force evaluator written outside
  // this file (`test/ev_native_test.ts`): with the pruning off, the only thing
  // left to disagree about is the model. Never used in play — the state space
  // is exponential in the hand's width.
  int noPrune;
};

// ---------------------------------------------------------------------------
// the pool, the shapes
// ---------------------------------------------------------------------------

/** Plan D1: what the root could not see, less what this line has absorbed. */
inline void poolOf(const Eval &E, const uint8_t *c, double *pool) {
  for (int t = 0; t < 34; t++) {
    const double taken = c[t] > E.root[t] ? static_cast<double>(c[t] - E.root[t]) : 0.0;
    const double v = E.base[t] - taken;
    pool[t] = v > 0 ? v : 0.0;
  }
}

/** Draws still worth branching on: anything the pool holds, plus anything a
 *  supplied `drawDist` row puts mass on (a read may name a tile the counting
 *  posterior thinks is gone — the read wins, and the branch has to exist). */
inline uint64_t liveDraws(const Eval &E, const double *pool, const uint8_t *c) {
  uint64_t m = 0;
  for (int t = 0; t < 34; t++) {
    if (c[t] >= 4) continue;
    if (pool[t] > 0) m |= 1ull << t;
  }
  return m | E.forced;
}

/**
 * Intern a 13-tile shape: its shanten, its acceptance, its ukeire mass and the
 * static half of the tail's hand value, computed once and shared by every turn,
 * every aka split and every root candidate of this evaluation.
 */
int32_t internShape(Eval &E, const uint8_t *c, int knownShanten = -1) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (int t = 0; t < 34; t++) {
    h ^= static_cast<uint64_t>(c[t]) + 1u;
    h *= 0x100000001b3ull;
  }
  // A 64-bit collision would silently merge two hands, so probe a short
  // deterministic chain of derived keys and compare the counts on every hit.
  for (uint64_t attempt = 0; attempt < 8; attempt++) {
    const uint64_t key = h + attempt;
    const int64_t got = tableGet(E.X->shapeTab, key);
    if (got >= 0) {
      const Shape &s = E.X->shapes.v[got];
      if (std::memcmp(s.c, c, 34) == 0) return static_cast<int32_t>(got);
      continue;
    }
    if (E.X->shapes.n >= E.shapeCap) {
      // Out of budget: the caller's draw collapses into the tsumogiri branch
      // and the answer is a truncated one, said so out loud.
      E.trunc = 1;
      return -1;
    }
    Shape s;
    std::memcpy(s.c, c, 34);
    // `discardAnalysis` already knows what every successor's shanten is; paying
    // for it twice is the difference between one probe and a hundred per node.
    s.shanten = static_cast<int8_t>(
        knownShanten >= 0 ? knownShanten : shantenImpl(E.X->cache, c, E.nMelds, E.closed));
    // Generation 0 is never a live candidate generation (`gen` starts at 1), so
    // a fresh shape's pool-dependent halves read as stale to everybody.
    s.edgeGen = 0;
    s.massGen = 0;
    s.geo = -1;
    s.edgeFrom = 0;
    s.edgeTo = 0;
    // Acceptance and the static hand value are LAZY. A shape reached as one of
    // several shanten-keeping discards is usually only ever recursed into, and
    // the ukeire probe that fills `accept` costs an order of magnitude more
    // than the shanten that decides whether it is worth keeping at all.
    s.haveAccept = 0;
    s.haveStat = 0;
    s.accept = 0;
    s.mass = 0;
    s.statExtra = 0;
    s.honitsu = 0;

    const int32_t id = arrPush(E.X->shapes, s);
    if (id < 0) return -1;
    tablePut(E.X->shapeTab, key, static_cast<uint32_t>(id));
    return id;
  }
  return -1;
}

/**
 * The acceptance MASK — which draws lower this shape's 向聴. Counts in, counts
 * out: nothing about the pool enters it, so it is filled once and read by every
 * candidate. The ukeire probe behind it costs an order of magnitude more than
 * the shanten that decides whether the shape is worth keeping at all, which is
 * why it stays lazy.
 */
void ensureMask(Eval &E, int32_t sid) {
  if (E.X->shapes.v[sid].haveAccept) return;
  E.X->shapes.v[sid].haveAccept = 1;
  const int sh = E.X->shapes.v[sid].shanten;
  if (sh >= 8) return;
  E.X->shapes.v[sid].accept =
      ukeireMaskImpl(E.X->cache, E.X->shapes.v[sid].c, E.nMelds, E.closed, sh);
}

/**
 * The live ukeire MASS behind that mask — Σ pool over the accepted types, and
 * therefore a property of the CANDIDATE and not of the shape. Stamped with the
 * candidate generation: the first ask of each candidate recomputes it, every
 * later one is a compare. Thirty-four adds; the mask above is what was
 * expensive.
 */
void ensureMass(Eval &E, int32_t sid) {
  ensureMask(E, sid);
  if (E.X->shapes.v[sid].massGen == E.X->gen) return;
  E.X->shapes.v[sid].massGen = E.X->gen;
  const uint64_t acc = E.X->shapes.v[sid].accept;
  double pool[34];
  poolOf(E, E.X->shapes.v[sid].c, pool);
  double mass = 0;
  for (int t = 0; t < 34; t++) {
    if ((acc >> t) & 1) mass += pool[t];
  }
  E.X->shapes.v[sid].mass = mass;
}

/**
 * This shape's per-draw geometry slot, allocating it on first ask. Returns null
 * once `GEO_CAP` is reached, and `ensureEdges` then simply recomputes — the
 * cache is a speed device with no say in any answer, which is the only reason
 * it is allowed to survive a candidate boundary.
 */
Geo *geoOf(Eval &E, int32_t sid) {
  int32_t g = E.X->shapes.v[sid].geo;
  if (g < 0) {
    if (E.X->geos.n >= GEO_CAP) return nullptr;
    Geo fresh;
    for (int t = 0; t < 34; t++) {
      fresh.best[t] = GEO_NOT_YET;
      fresh.mask[t] = 0;
    }
    g = arrPush(E.X->geos, fresh);
    if (g < 0) return nullptr;
    E.X->shapes.v[sid].geo = g;
  }
  return &E.X->geos.v[g];
}

/** Fill the static half of `handvalue.ts#handValue` — only the tail wants it. */
void ensureStat(Eval &E, int32_t sid) {
  if (E.X->shapes.v[sid].haveStat) return;
  E.X->shapes.v[sid].haveStat = 1;
  const uint8_t *c = E.X->shapes.v[sid].c;
  uint8_t all[34];
  int doraNoAka = 0, yTrip = 0, yPair = 0;
  for (int t = 0; t < 34; t++) {
    all[t] = static_cast<uint8_t>(c[t] + E.meldC[t]);
    doraNoAka += all[t] * E.dora[t];
    const bool yakuhai = t >= 31 || t == E.seatWind || t == E.roundWind;
    if (yakuhai && all[t] >= 3) yTrip++;
    if (yakuhai && c[t] == 2) yPair++;
  }
  E.X->shapes.v[sid].statExtra =
      E.p[P_VALUE_PER_DORA] * doraNoAka + E.p[P_VALUE_YAKUHAI] * (yTrip + 0.5 * yPair);
  // 染め手模様: at most two strays outside one numeric suit (honors welcome).
  int bestStray = 99;
  for (int suit = 0; suit < 3; suit++) {
    int stray = 0;
    for (int t = 0; t < 27; t++) {
      if (t / 9 != suit) stray += all[t];
    }
    if (stray < bestStray) bestStray = stray;
  }
  E.X->shapes.v[sid].honitsu = bestStray <= 2 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// hazards (plan §1.4) — population rates over the M14 reads, never a read of
// one seat's river. Both grow with the turn exactly as handvalue.ts:238 does.
// ---------------------------------------------------------------------------

/** P(a tenpai opponent ends the hand before our next draw). `handvalue.ts:238`. */
inline double qEndAt(const Eval &E, int j) {
  const double q = E.p[P_OPP_HAZARD] * (E.sumTenpai + E.p[P_OPP_GROWTH] * j);
  return q < 0 ? 0 : (q > 0.95 ? 0.95 : q);
}

/**
 * P(the tile we are about to throw deals in), for a FUTURE discard we cannot
 * name yet: a base rate per tenpai opponent. The ROOT discard never uses this
 * — there `pIn`/`costIn` are exact, packed by TypeScript with the doctrine
 * already priced in. ⚑ Under our own riichi the discard is forced, which the
 * `riichiDealinMult` multiplier stands in for.
 */
inline double qInAt(const Eval &E, int j, int r) {
  // `handvalue.ts:238`: the drift is added ONCE to the SUM, not once per
  // opponent. Reading it per-opponent tripled it, and on a quiet table that
  // alone retired most of the hand's mass over twenty turns.
  double q = E.p[P_DEALIN_RATE] * (E.sumTenpai + E.p[P_OPP_GROWTH] * j);
  if (r) q *= E.p[P_RIICHI_DEALIN_MULT];
  if (q < 0) q = 0;
  return q > 0.5 ? 0.5 : q;
}

/** What one un-won turn costs in expectation, once the hazards have fired. */
/** Plan §1.4: our share of an opponent's tsumo — half of it when we deal. */
inline double tsumoShareOf(const Eval &E) {
  return E.dealer ? 0.5 : E.p[P_TSUMO_SHARE];
}

inline double hazardCost(const Eval &E, int j, int r) {
  return (qInAt(E, j, r) * E.meanLoss + qEndAt(E, j) * tsumoShareOf(E) * E.meanLoss) * E.risk;
}

inline double surviveAt(const Eval &E, int j, int r) {
  const double s = 1.0 - qInAt(E, j, r) - qEndAt(E, j);
  return s > 0 ? s : 0.0;
}

/**
 * 流局: the exact 3000 split over independent Bernoulli opponent tenpai. A
 * tenpai seat collects `total/k`, a noten seat pays `total/(4−k)`, and k = 0 or
 * 4 settles nothing — the rule, enumerated over all eight opponent outcomes in
 * a fixed order.
 */
double notenValue(const Eval &E, bool weTenpai) {
  const double total = E.p[P_NOTEN_PENALTY_TOTAL];
  double acc = 0;
  for (int m = 0; m < 8; m++) {
    double pr = 1;
    int cnt = 0;
    for (int i = 0; i < 3; i++) {
      const double t = clamp01(E.tenpaiP[i]);
      if (m & (1 << i)) {
        pr *= t;
        cnt++;
      } else {
        pr *= 1.0 - t;
      }
    }
    if (pr <= 0) continue;
    const int k = cnt + (weTenpai ? 1 : 0);
    if (k == 0 || k == 4) continue;
    if (weTenpai) acc += pr * (total / k) * E.gain;
    else acc -= pr * (total / (4 - k)) * E.risk;
  }
  return acc;
}

/** The hand ends at 流局 with `shanten`; our riichi stick is gone by then. */
double ryuukyokuValue(const Eval &E, int shanten, int r) {
  double v = notenValue(E, shanten == 0);
  // The 1000 was already paid at the declaration, so only the part the ruleset
  // does NOT treat as lost comes back.
  if (r) v += 1000.0 * (1.0 - E.p[P_STICK_AT_DRAW]) * E.gain;
  if (shanten == 0 && E.dealer) v += E.p[P_DEALER_RENCHAN] * E.gain;
  return v;
}

/**
 * Giving up (plan §1.5): pay the residual deal-in rate of a defended hand for
 * every turn that is left, then the noten penalty. ⚑ Held 現物 beyond the root
 * are not counted, and the opponents' tsumo is not charged to the fold line —
 * folding does not stop a tsumo, so this reads the fold slightly high; the
 * ROOT deal-in cost, which is the number the seat actually acts on, is exact.
 */
double foldValue(const Eval &E, int j) {
  // THE SAME SWEEP THE PUSH LINE WALKS. A flat "residual rate × turns left"
  // plus an undiscounted 流局 settlement is not on the push line's mass scale,
  // and the fold verdict is a comparison BETWEEN the two: mass that the table
  // has already retired cannot go on to pay a noten penalty, and giving up
  // stops us dealing in but not anyone drawing their own winning tile.
  double alive = 1.0, v = 0.0;
  for (int t = j; t < E.T; t++) {
    const double qe = qEndAt(E, t);
    v -= alive * (E.p[P_FOLD_HAZARD] * E.meanLoss + qe * tsumoShareOf(E) * E.meanLoss) * E.risk;
    alive *= 1.0 - qe;
  }
  return v + alive * notenValue(E, false);
}

// ---------------------------------------------------------------------------
// what a win pays
// ---------------------------------------------------------------------------

/** src/score.ts, in points: what the table hands us for one agari. */
inline double paymentOf(int base, bool dealerWins, bool tsumo) {
  if (tsumo) {
    return dealerWins ? static_cast<double>(ceil100(base * 2) * 3)
                      : static_cast<double>(ceil100(base * 2) + ceil100(base) * 2);
  }
  return static_cast<double>(ceil100(base * (dealerWins ? 6 : 4)));
}

/**
 * The distribution of the extra han a hidden indicator adds, by exact counting
 * over the indicator's own posterior: P(indicator = u) × (copies of the type it
 * names that this hand holds). Used for 裏ドラ under riichi and for the kan-dora
 * still to be flipped, with `dist` either the pool posterior or the supplied
 * `uraDist`/`nextDoraDist` (plan D7 — a one-hot from an oracle gives the true
 * han, a soft vector from a learned module gives its belief).
 *
 * The plan calls for a P(0)/P(1)/P(≥2) lumping; the exact bucketing is the same
 * arithmetic and one array longer, so the ⚑ is spent here rather than kept.
 */
void extraHanDist(const uint8_t *all14, const double *dist, double *out, int &n) {
  for (int i = 0; i < EXTRA_MAX; i++) out[i] = 0;
  for (int u = 0; u < 34; u++) {
    const double p = dist[u];
    if (p <= 0) continue;
    int h = all14[doraFromIndicator(u)];
    if (h >= EXTRA_MAX) h = EXTRA_MAX - 1;
    out[h] += p;
  }
  n = 1;
  for (int i = EXTRA_MAX - 1; i > 0; i--) {
    if (out[i] > 0) {
      n = i + 1;
      break;
    }
  }
}

/** `a` ⊛ `b`, truncated at EXTRA_MAX — two independent han bonuses. */
void convolve(double *a, int &an, const double *b, int bn) {
  double out[EXTRA_MAX];
  for (int i = 0; i < EXTRA_MAX; i++) out[i] = 0;
  for (int i = 0; i < an; i++) {
    if (a[i] <= 0) continue;
    for (int k = 0; k < bn; k++) {
      if (b[k] <= 0) continue;
      int h = i + k;
      if (h >= EXTRA_MAX) h = EXTRA_MAX - 1;
      out[h] += a[i] * b[k];
    }
  }
  int n = 1;
  for (int i = EXTRA_MAX - 1; i > 0; i--) {
    if (out[i] > 0) {
      n = i + 1;
      break;
    }
  }
  for (int i = 0; i < EXTRA_MAX; i++) a[i] = out[i];
  an = n;
}

// ---------------------------------------------------------------------------
// terminals: what a tenpai shape's waits are worth
// ---------------------------------------------------------------------------

/** The packed scorer input for "this hand, plus `winType`, won like this". */
void buildWinCtx(const Eval &E, const uint8_t *c14, int winType, int tsumo, int riichi,
                 WinCtx &w) {
  std::memcpy(w.counts, c14, 34);
  w.nMelds = E.nMelds;
  w.menzen = true;
  for (int m = 0; m < E.nMelds; m++) {
    w.called[m] = E.melds[m];
    if (!(E.melds[m].kind == BK_KAN && E.melds[m].concealed)) w.menzen = false;
  }
  for (int t = 0; t < 34; t++) w.all[t] = static_cast<uint8_t>(c14[t] + E.meldC[t]);
  w.winType = winType;
  w.tsumo = tsumo;
  w.riichi = riichi;
  w.doubleRiichi = 0;
  w.ippatsu = 0; // added as a han bonus by the mixture, never by the flag
  w.rinshan = 0;
  w.chankan = 0;
  w.haitei = 0;
  w.houtei = 0;
  w.tenhou = 0;
  w.chiihou = 0;
  w.seatWind = E.seatWind;
  w.roundWind = E.roundWind;
  for (int t = 0; t < 34; t++) {
    w.dora[t] = E.dora[t];
    w.ura[t] = 0; // ura rides in as extra han, so one reading serves every draw
  }
  w.akaCount = 0; // likewise: a constant addition cannot reorder two readings
  w.kuitan = E.p[P_KUITAN] != 0;
  w.kazoe = E.p[P_KAZOE_YAKUMAN] != 0;
  w.kiriage = E.p[P_KIRIAGE_MANGAN] != 0;
  w.dwFu = static_cast<int>(E.p[P_DOUBLE_WIND_FU]);
  w.ippatsuCfg = 1;
}

/**
 * Points collected for one agari, mixed over the han a hidden indicator (裏 /
 * 新ドラ) and 一発 may still add. Dora, aka, ura and 一発 are all CONSTANT
 * additions to every reading of the same hand, so the decomposition runs once
 * and only `basePoints` is re-evaluated per outcome — which is also why the
 * mixture is exact rather than a re-scoring approximation.
 */
double winPayout(const Eval &E, const WinCtx &wc, const Res &b, bool tsumo, int akaCount,
                 int riichi, const double *ex, int exN) {
  if (!b.ok) return 0.0;
  double acc = 0;
  for (int h = 0; h < exN; h++) {
    if (ex[h] <= 0) continue;
    int base = 0, lim = 0;
    if (b.yakuman > 0) base = 8000 * b.yakuman; // a yakuman suppresses dora entirely
    else basePoints(b.han + akaCount + h, b.fu, wc, &base, &lim);
    acc += ex[h] * paymentOf(base, E.dealer != 0, tsumo);
  }
  acc += 300.0 * E.honba + 1000.0 * E.kyotaku;
  // Our own stick comes back on a win — but only if WE put it out this hand;
  // an already-declared riichi's 1000 is in `kyotaku` already.
  if (riichi && !E.ownRiichi) acc += 1000.0;
  return acc;
}

/**
 * Every wait of a tenpai shape priced for one (riichi, aka, 一発) reading, in
 * points already scaled by 順位効用 `gain`. Cached per evaluation: the price
 * does not move with the turn, only the chance of collecting it does.
 */
int32_t winValsFor(Eval &E, int32_t sid, int r, int aka, int ip) {
  const uint64_t key = ((static_cast<uint64_t>(sid) << 6 | static_cast<uint64_t>(E.thrown)) << 4) |
      (static_cast<uint64_t>(aka) << 2) | (static_cast<uint64_t>(r) << 1) |
      static_cast<uint64_t>(ip);
  const int64_t got = tableGet(E.X->winTab, key);
  if (got >= 0) return static_cast<int32_t>(got);

  ensureMask(E, sid); // the mask only — no pool enters a wait price here
  uint8_t c[34];
  std::memcpy(c, E.X->shapes.v[sid].c, 34);
  const uint64_t waits = E.X->shapes.v[sid].accept;

  double pool[34];
  poolOf(E, c, pool);
  // ⚑ Red 5p are not depleted from the pool; the split below is the same
  // proportion at every turn.
  const double pa = (E.akaUnseen > 0 && pool[13] > 0)
      ? clamp01(static_cast<double>(E.akaUnseen) / pool[13])
      : 0.0;

  // 振聴 is a property of the SHAPE, not of one wait: a single wait sitting in
  // our own river kills the ron on every wait this hand has. (The plan's
  // per-wait phrasing is subsumed by the rule.)
  bool furitenShape = false;
  for (int w = 0; w < 34; w++) {
    if (!((waits >> w) & 1)) continue;
    // The tile this candidate is THROWING counts: a discard into one's own
    // wait is 振聴, and pricing that line with a live ron made the engine
    // recommend it (`123456789m 11234p`, cut 2p: 5646 and the argmax, against
    // 2912 with the same tile already in the river).
    if (E.river[w] > 0 || w == E.thrown) {
      furitenShape = true;
      break;
    }
  }

  WinVals v;
  v.n = 0;
  for (int w = 0; w < 34 && v.n < MAX_WAITS; w++) {
    if (!((waits >> w) & 1)) continue;
    if (c[w] >= 4) continue;
    uint8_t c14[34];
    std::memcpy(c14, c, 34);
    c14[w]++;

    // The extra han a hidden indicator adds, exactly (plan D7).
    double ex[EXTRA_MAX];
    int exN = 1;
    for (int i = 0; i < EXTRA_MAX; i++) ex[i] = 0;
    ex[0] = 1.0;
    uint8_t all14[34];
    for (int t = 0; t < 34; t++) all14[t] = static_cast<uint8_t>(c14[t] + E.meldC[t]);
    if (r) {
      double u[EXTRA_MAX];
      int un = 0;
      extraHanDist(all14, E.hasUra ? E.uraDist : E.indP, u, un);
      convolve(ex, exN, u, un);
    }
    if (E.kanDoraOn) {
      double u[EXTRA_MAX];
      int un = 0;
      extraHanDist(all14, E.hasNextDora ? E.nextDoraDist : E.indP, u, un);
      convolve(ex, exN, u, un);
    }
    if (ip && r) {
      const double q = clamp01(E.p[P_IPPATSU_P]);
      const double one[2] = {1.0 - q, q};
      convolve(ex, exN, one, 2);
    }

    WinCtx wt, wr;
    buildWinCtx(E, c14, w, 1, r, wt);
    buildWinCtx(E, c14, w, 0, r, wr);
    const Res bt = scoreWin(wt);
    const Res br = scoreWin(wr);

    // A 5p we WIN on may itself be red; that split lives here rather than in
    // the state, because the tile is never held.
    const int akaHi = (w == 13 && pa > 0) ? 1 : 0;
    double vt = winPayout(E, wt, bt, true, aka, r, ex, exN);
    double vr = winPayout(E, wr, br, false, aka, r, ex, exN);
    if (akaHi) {
      vt = (1 - pa) * vt + pa * winPayout(E, wt, bt, true, aka + 1, r, ex, exN);
      vr = (1 - pa) * vr + pa * winPayout(E, wr, br, false, aka + 1, r, ex, exN);
    }

    v.waitType[v.n] = static_cast<uint8_t>(w);
    v.cashT[v.n] = bt.ok ? 1 : 0;
    // ⚑ A temporary/riichi furiten at the root is treated as lasting the whole
    // horizon; it really only lasts one go-around.
    v.cashR[v.n] = (br.ok && !E.furiten && !furitenShape) ? 1 : 0;
    v.tsumo[v.n] = vt * E.gain;
    v.ron[v.n] = vr * E.gain;
    v.n++;
  }

  const int32_t idx = arrPush(E.X->wins, v);
  if (idx < 0) return -1;
  tablePut(E.X->winTab, key, static_cast<uint32_t>(idx));
  return idx;
}

// ---------------------------------------------------------------------------
// successors
// ---------------------------------------------------------------------------

/**
 * min over discards of shanten(hand14 − d), and every discard achieving it.
 *
 * The hot loop of the whole search, so it is written the way
 * `ukeireMaskImpl` is: merge the three groups a discard cannot touch ONCE, then
 * pay one cached word lookup and one merge per candidate — and carry the
 * 七対子/国士 counts incrementally, since removing one copy of one type moves
 * each of them by at most one.
 */
void discardAnalysis(Eval &E, uint8_t *c14, int &bestOut, uint64_t &maskOut) {
  int best = 99;
  uint64_t mask = 0;
  const int cap = 4 - E.nMelds;
  const bool cl = E.closed && E.nMelds == 0;

  if (cap < 0 || !tabulatable(c14, E.nMelds)) {
    for (int d = 0; d < 34; d++) {
      if (c14[d] == 0) continue;
      c14[d]--;
      const int s = shantenImpl(E.X->cache, c14, E.nMelds, E.closed);
      c14[d]++;
      if (s < best) {
        best = s;
        mask = 1ull << d;
      } else if (s == best) {
        mask |= 1ull << d;
      }
    }
    bestOut = best;
    maskOut = mask;
    return;
  }

  uint64_t w[4];
  groupWords(E.X->cache, c14, w);
  Summary rest[4];
  for (int g = 0; g < 4; g++) {
    Summary a, b;
    std::memcpy(a, IDENTITY, sizeof(Summary));
    bool flip = false;
    for (int k = 0; k < 4; k++) {
      if (k == g) continue;
      if (flip) mergeWord(b, w[k], a);
      else mergeWord(a, w[k], b);
      flip = !flip;
    }
    std::memcpy(rest[g], flip ? b : a, sizeof(Summary));
  }

  int kinds = 0, pairs = 0, yKinds = 0, yPairs = 0;
  bool yaochu[34] = {false};
  if (cl) {
    for (int t = 0; t < 34; t++) {
      if (c14[t] >= 1) kinds++;
      if (c14[t] >= 2) pairs++;
    }
    for (int k = 0; k < 13; k++) {
      const int t = YAOCHU[k];
      yaochu[t] = true;
      if (c14[t] >= 1) yKinds++;
      if (c14[t] >= 2) yPairs++;
    }
  }

  for (int d = 0; d < 34; d++) {
    if (c14[d] == 0) continue;
    const int g = d < 27 ? d / 9 : 3;
    const int off = g == 3 ? 27 : g * 9;
    const int len = g == 3 ? 7 : 9;
    uint8_t tmp[9];
    std::memcpy(tmp, c14 + off, static_cast<size_t>(len));
    tmp[d - off]--;
    Summary merged;
    mergeWord(rest[g], cachedWord(E.X->cache, tmp, len, g != 3), merged);
    int s = 2 * cap - evalSummary(merged, cap);
    if (cl) {
      const int ch = chiitoiFrom(pairs - (c14[d] == 2 ? 1 : 0), kinds - (c14[d] == 1 ? 1 : 0));
      if (ch < s) s = ch;
      const int yk = yKinds - ((yaochu[d] && c14[d] == 1) ? 1 : 0);
      const int yp = yPairs - ((yaochu[d] && c14[d] == 2) ? 1 : 0);
      const int ko = kokushiFrom(yk, yp > 0 ? 1 : 0);
      if (ko < s) s = ko;
    }
    if (s < best) {
      best = s;
      mask = 1ull << d;
    } else if (s == best) {
      mask |= 1ull << d;
    }
  }
  bestOut = best;
  maskOut = mask;
}

/**
 * Build the continuations of one shape, once. Draw types ascending, discards
 * ascending — the fixed exploration order the node budget truncates against.
 *
 *   * shanten drops ⇒ every shanten-keeping discard is a candidate;
 *   * shanten holds at a rung ≤ `sameShantenRungs` ⇒ 待ち替え: standing pat
 *     (the self edge, emitted in discard order like any other) against the
 *     discards that strictly widen the acceptance mass;
 *   * anything else has no edge at all and falls into the collapsed tsumogiri
 *     branch, which is why an idle draw costs the search nothing.
 */
bool ensureEdges(Eval &E, int32_t sid) {
  // PER CANDIDATE, not per shape: the selection AND the ranking below run off
  // `mass`, which is measured against this candidate's pool. The geometry the
  // rebuild stands on (`Geo`, `accept`, `shanten`) is shared; the edge list is
  // not, and the generation stamp is what says so.
  if (E.X->shapes.v[sid].edgeGen == E.X->gen) return true;

  uint8_t c14[34];
  std::memcpy(c14, E.X->shapes.v[sid].c, 34);
  const int s = E.X->shapes.v[sid].shanten;
  // Only 待ち替え needs the shape's own mass, and only there is `baseMass`ever
  // read — but a shared arena can hold another candidate's number, so it is
  // zeroed rather than left to be stale-but-unused.
  double baseMass = 0;
  if (s <= E.sameRungs) {
    ensureMass(E, sid);
    baseMass = E.X->shapes.v[sid].mass;
  }

  double pool[34];
  poolOf(E, E.X->shapes.v[sid].c, pool);
  const uint64_t live = liveDraws(E, pool, E.X->shapes.v[sid].c);

  const int32_t from = static_cast<int32_t>(E.X->edges.n);
  bool ok = true;
  for (int k = 0; k < 34; k++) {
    if (!((live >> k) & 1)) continue;
    if (c14[k] >= 4) continue;
    c14[k]++;
    if (E.noPrune) {
      // EVERY discard, including the ones that make the hand worse. This is the
      // branch a brute-force oracle enumerates, so it is the branch the engine
      // has to be able to enumerate too when it is being checked against one.
      for (int d = 0; d < 34 && ok; d++) {
        if (c14[d] == 0) continue;
        c14[d]--;
        const int32_t su = internShape(E, c14);
        c14[d]++;
        if (su < 0) {
          ok = false;
          break;
        }
        Edge e;
        e.succ = su;
        e.k = static_cast<int16_t>(k);
        e.d = static_cast<int16_t>(d);
        if (arrPush(E.X->edges, e) < 0) ok = false;
      }
      c14[k]--;
      if (!ok) break;
      continue;
    }
    // THE HOT LOOP, ONCE PER (shape, draw) AND NOT ONCE PER CANDIDATE. What
    // `discardAnalysis` answers is a pure function of the 14 counts, so it is
    // cached on the shape and read by every candidate that reaches it — while
    // everything the answer is USED for below (the mass, the ranking, the kept
    // set) stays candidate-scoped.
    int best = 0;
    uint64_t mask = 0;
    Geo *geo = geoOf(E, sid);
    if (geo && geo->best[k] != GEO_NOT_YET) {
      best = geo->best[k];
      mask = geo->mask[k];
    } else {
      discardAnalysis(E, c14, best, mask);
      if (geo) {
        geo->best[k] = static_cast<int8_t>(best);
        geo->mask[k] = mask;
      }
    }
    const bool accept = best < s;
    // 待ち替え only where it is a real decision — 聴牌 and 1向聴 — and only for a
    // strict widening. `sameShantenRungs` above 1 is CLAMPED to 1: enumerating
    // an upgrade at 2向聴 multiplies the shape space by the acceptance count for
    // a difference the tail cannot even see.
    const int upRungs = E.sameRungs < 1 ? E.sameRungs : 1;
    const bool upgrade = best == s && s <= upRungs;
    if (accept || upgrade) {
      // HOW WIDE THE FAN IS, PER RUNG (owner 2026-08-31: "we can sacrifice
      // speed for the best result we could get"). A shanten-keeping discard
      // outside the kept set is a line the search never prices, and the
      // oracle-mode diff measured what that costs: 0.1% at 2向聴 with two draws
      // left, but 5.7% mean / 18.9% worst on 1向聴 rests. Where the loss is
      // that big the fan is opened all the way; where it is small the old
      // bound stays, because the branching factor is still what decides
      // whether the search finishes at all, and an unfinished search is priced
      // by the closed-form tail — the one thing that must not happen to SOME
      // candidates and not others.
      //
      //   s ≤ 1 : EVERY shanten-keeping discard (a 1向聴 node accepting into
      //           聴牌 is choosing its wait, and the wait is the hand);
      //   s = 2 : the best six by acceptance mass, plus the reserved dora slot;
      //   s ≥ 3 : the best three, as before — the tail cannot see the
      //           difference and these are the widest, most expensive nodes.
      //
      // Rank by what the discard is FOR. Acceptance mass is the right proxy
      // while the hand is still being built, but at 聴牌 a 待ち替え is about
      // what the wait PAYS, and at any rung the widest shape is not always the
      // one holding the dora — so the value ranking is used where it belongs
      // and one slot is reserved for the best kept dora count.
      //
      // 待ち替え keeps the best TWO by expected win value (was one). The
      // second slot is what lets a 聴牌 hand compare two real wait swaps
      // instead of one against standing pat.
      const int keep = upgrade ? UPGRADE_KEEP : (s <= 1 ? KEEP_MAX : (s == 2 ? 6 : 3));
      // With the fan fully open the ranking has nothing to decide — every
      // shanten-keeping discard is emitted, in the same ascending-type order —
      // so the acceptance probe behind `mass` is not paid at all. That is most
      // of what the widening at s ≤ 1 would otherwise have cost.
      int nMask = 0;
      for (int d = 0; d < 34; d++) nMask += (mask >> d) & 1;
      const bool keepAll = !upgrade && keep >= nMask;
      int bd[KEEP_MAX + 1];
      int32_t bs[KEEP_MAX + 1];
      double bm[KEEP_MAX + 1];
      for (int i = 0; i <= KEEP_MAX; i++) {
        bd[i] = 0;
        bs[i] = 0;
        bm[i] = 0;
      }
      int nb = 0;
      int doraBest = -1, doraAt = -1;
      int32_t doraSucc = -1;
      for (int d = 0; d < 34; d++) {
        if (!((mask >> d) & 1)) continue;
        c14[d]--;
        const int32_t su = internShape(E, c14, best);
        c14[d]++;
        if (su < 0) {
          ok = false;
          break;
        }
        const bool standPat = upgrade && d == k;
        double m = 0;
        if (!standPat && !keepAll) {
          ensureMass(E, su);
          m = E.X->shapes.v[su].mass;
          // A 待ち替え has to BUY something. At 聴牌 two live copies is a real
          // wait swap; at 1向聴 almost every idle draw offers some nominally
          // wider shape, so the price of admission there is much higher.
          //
          // THE GATE IS ON MASS, THE RANKING IS ON VALUE. Ranking by expected
          // win value and then testing that value against `baseMass + 2` puts
          // points on one side of the comparison and a tile count on the other,
          // so every candidate passes and the tenpai chain explodes (measured:
          // 1653 states to 55005, 0.7 ms to 372 ms).
          const double need = s == 0 ? 2.0 : 8.0;
          if (upgrade && !(m >= baseMass + need)) continue;
          if (upgrade) {
            // 待ち替え is ORDERED by what the wait it buys is worth.
            const int32_t wi = winValsFor(E, su, 0, 0, 0);
            double ev = 0;
            if (wi >= 0) {
              double sp[34];
              poolOf(E, E.X->shapes.v[su].c, sp);
              const WinVals &wv = E.X->wins.v[wi];
              for (int q = 0; q < wv.n; q++) {
                if (wv.cashT[q]) ev += sp[wv.waitType[q]] * wv.tsumo[q];
              }
            }
            m = ev;
          } else {
            // One reserved slot: the shanten-keeping discard that keeps the
            // most dora, when the mass ranking would have dropped it.
            int dora = 0;
            for (int q = 0; q < 34; q++) {
              dora += (E.X->shapes.v[su].c[q] + E.meldC[q]) * E.dora[q];
            }
            if (dora > doraBest) {
              doraBest = dora;
              doraAt = d;
              doraSucc = su;
            }
          }
          // Standing pat is always on the table; a 待ち替え has to buy at least
          // one more live copy to be worth a branch.
          // ...and never into a wait the discard itself just killed.
          if (upgrade && ((E.X->shapes.v[su].accept >> d) & 1)) continue;
        }
        if (standPat) {
          // Emitted unconditionally, outside the ranking.
          Edge e;
          e.succ = su;
          e.k = static_cast<int16_t>(k);
          e.d = static_cast<int16_t>(d);
          if (arrPush(E.X->edges, e) < 0) {
            ok = false;
            break;
          }
          continue;
        }
        // Insertion sort by (mass desc, discard type asc) — a fixed order, so
        // the pruned set is a pure function of the shape. With `keepAll` every
        // `m` is zero and the sort degenerates into "append in type order",
        // which is exactly what the emission loop below wants anyway.
        int at = nb < keep ? nb : keep;
        for (int i = 0; i < (nb < keep ? nb : keep); i++) {
          if (m > bm[i]) {
            at = i;
            break;
          }
        }
        if (at >= keep) continue;
        for (int i = (nb < keep ? nb : keep - 1); i > at; i--) {
          bm[i] = bm[i - 1];
          bd[i] = bd[i - 1];
          bs[i] = bs[i - 1];
        }
        bm[at] = m;
        bd[at] = d;
        bs[at] = su;
        if (nb < keep) nb++;
      }
      if (!ok) break;
      // The reserved dora slot is meaningless under `keepAll` (nothing was
      // dropped) and `doraSucc` is not even filled there.
      if (!upgrade && !keepAll && doraSucc >= 0 && nb > 0) {
        bool have = false;
        for (int i = 0; i < nb; i++) {
          if (bd[i] == doraAt) have = true;
        }
        if (!have && nb <= KEEP_MAX) {
          bd[nb] = doraAt;
          bs[nb] = doraSucc;
          bm[nb] = 0;
          nb++;
        }
      }
      // Ascending discard type, so the exploration order is the documented one.
      for (int pass = 0; pass < 34 && ok; pass++) {
        for (int i = 0; i < nb; i++) {
          if (bd[i] != pass) continue;
          Edge e;
          e.succ = bs[i];
          e.k = static_cast<int16_t>(k);
          e.d = static_cast<int16_t>(bd[i]);
          if (arrPush(E.X->edges, e) < 0) ok = false;
        }
      }
    }
    c14[k]--;
    if (!ok) break;
  }
  // A shape whose successors could not all be created is NOT a shape with no
  // successors. Freezing it as one used to turn the state into "tsumogiri until
  // 流局" — worth exactly the noten settlement, i.e. 0 on a quiet table — and,
  // because the shape arena is shared across root candidates, one expensive
  // candidate could exhaust it and leave every LATER candidate priced at 0. So
  // the partial work is rolled back, the shape stays un-analyzed, and the state
  // takes the closed-form tail like any other truncation.
  if (!ok) {
    E.X->edges.n = static_cast<uint32_t>(from);
    E.trunc = 1;
    return false;
  }
  E.X->shapes.v[sid].edgeGen = E.X->gen;
  E.X->shapes.v[sid].edgeFrom = from;
  E.X->shapes.v[sid].edgeTo = static_cast<int32_t>(E.X->edges.n);
  return true;
}

// ---------------------------------------------------------------------------
// the value functions
// ---------------------------------------------------------------------------

/**
 * What the value is MADE OF, carried alongside it under the same optimal
 * policy. Nothing in the search reads these — they exist so a hand can be
 * audited in the terms it is actually played in ("how often does this reach
 * tenpai, how often does it win, what does it collect, what does it pay")
 * rather than as one number nobody can check. `mjev_eval_rest` hands them back
 * in the meta slots; the sanity table in `test/ev_native_test.ts` prints them.
 */
struct Diag {
  double pWin;    // P(this hand is the one that wins)
  double pTenpai; // P(it is ever tenpai before the hand ends)
  double winSum;  // Σ P(win) × points collected — divide by pWin for E[value|win]
  double cost;    // expected points paid to the table on the way
};

inline void diagZero(Diag &d) {
  d.pWin = 0;
  d.pTenpai = 0;
  d.winSum = 0;
  d.cost = 0;
}

inline void diagAdd(Diag &acc, const Diag &d, double w) {
  acc.pWin += w * d.pWin;
  acc.pTenpai += w * d.pTenpai;
  acc.winSum += w * d.winSum;
  acc.cost += w * d.cost;
}

double memoValue(Eval &E, int32_t sid, int j, int aka, int r, Diag &dg);

/**
 * The closed-form tail (plan §1.8): `handvalue.ts#handPwin` × `handValue`,
 * ported chain for chain — the fitted mean acceptance for the rungs below this
 * one, the exact live acceptance at it, the same per-turn opponent sweep —
 * minus the hazard cost accumulated on the surviving mass, plus what the
 * leftover mass settles at 流局.
 *
 * ⚑ Only the opponents' own win reduces the mass, exactly as `handvalue.ts:238`
 * does; our deal-in rate is charged as a cost without killing the hand.
 */
double tailValue(Eval &E, int32_t sid, int j, int aka, int r, Diag &dg) {
  const int shanten = E.X->shapes.v[sid].shanten;
  diagZero(dg);
  int turns = E.T - j;
  if (turns <= 0) {
    dg.pTenpai = shanten == 0 ? 1.0 : 0.0;
    return ryuukyokuValue(E, shanten, r);
  }
  if (turns > T_MAX) turns = T_MAX;

  double U = E.Nroot - j;
  if (U < 1) U = 1;
  const int sh0 = shanten < 0 ? 0 : shanten;
  // ONE LEVEL PER ADVANCE. `handvalue.ts` folds every hand at 3向聴 or worse
  // into a single rung, because it only ever prices the seat's OWN resting
  // hand and 4向聴 vs 3向聴 is a distinction it never has to make. The DP DOES
  // have to make it: the tail is what ranks a discard that keeps shanten
  // against one that throws it away, and a chain that charges both the same
  // three advances rates the wider, worse hand higher — the seat then breaks
  // its hand every turn. So the chain is `shanten` advances long, and only the
  // fitted mean it reads at each level is clamped to the four rungs. For
  // shanten ≤ 3 this is `handPwin` step for step.
  const int L = sh0 > 8 ? 8 : sh0;
  ensureMass(E, sid);
  ensureStat(E, sid);
  const double liveU = E.X->shapes.v[sid].mass;

  double adv[9];
  for (int i = 0; i < L; i++) {
    const int d = L - i; // shanten at level i
    // The rung we stand on is COUNTED; the ones below it are the fitted means,
    // because the hand has not chosen those shapes yet. Capping the counted
    // rung at its own mean was tried and is wrong: at 3向聴 every candidate
    // exceeds the mean, so every discard scored alike and the hand never
    // consolidated (聴牌率 16%). The scale mismatch it was meant to fix is
    // `TAIL_CALIBRATION`'s job.
    // The rung we stand on is COUNTED — and it must be, because it is the ONLY
    // thing that tells one candidate discard from another at 3向聴+: capping it
    // at the fitted mean (which every shapeless hand exceeds) put every
    // shanten-keeping candidate within five points of every other, and the
    // argmax then ran on the tie-break. The scale mismatch the cap was fighting
    // is `TAIL_CAL`'s job, per level, measured.
    double u = i == 0 ? liveU : E.p[P_MEAN_UKEIRE0 + (3 - (d < 3 ? d : 3))];
    if (u < 0) u = 0;
    adv[i] = u / U;
    if (adv[i] > 1) adv[i] = 1;
  }
  double waits = L == 0 ? liveU : E.p[P_MEAN_UKEIRE3];
  if (waits < 0) waits = 0;

  // Whether a discard can be COLLECTED on. `handvalue.ts` reads "closed and not
  // furiten" as ronnable because the riichi declaration is always there to cure
  // a yaku-less wait — but under `ev.riichi:false` (plan D3) this engine is
  // forbidden to declare, so the cure is not available and the hand prices as a
  // damaten. Getting this wrong is what made every hand past `exactShanten`
  // look like a 7000-point riichi hand next to an exactly-priced dama one.
  const bool mayDeclare = E.ownRiichi || (E.closed && !E.furiten);
  const bool pricesAsRiichi = r != 0 || mayDeclare;
  // An OPEN hand collects on a discard like any other (`heuristic.ts:1753`);
  // a yaku-less one is vetoed upstream by `hasYakuProspect`, so being open is
  // enough. Requiring `closed` here priced every melded hand as tsumo-only.
  const bool ronnable = (r != 0 || !E.closed || pricesAsRiichi) && !E.furiten;
  const double rons = ronnable ? 3.0 * E.p[P_RON_FACTOR] : 0.0;
  const double h = clamp01((waits / U) * (1.0 + rons));

  // `handValue`: the static price of landing it.
  const double statBase = E.closed
      ? (pricesAsRiichi ? E.p[P_VALUE_RIICHI] : E.p[P_VALUE_DAMATEN])
      : (E.X->shapes.v[sid].honitsu ? E.p[P_VALUE_HONITSU] : E.p[P_VALUE_OPEN]);
  double value = statBase + E.X->shapes.v[sid].statExtra + E.p[P_VALUE_PER_DORA] * aka;
  if (E.dealer) value *= E.p[P_VALUE_DEALER];
  if (value > E.p[P_VALUE_CAP]) value = E.p[P_VALUE_CAP];
  value += 300.0 * E.honba + 1000.0 * E.kyotaku;

  double mass[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  mass[0] = 1;
  double pwin = 0, cost = 0;
  double arrived = L == 0 ? 1.0 : 0.0; // ever reached the tenpai level
  for (int t = 0; t < turns; t++) {
    const double won = mass[L] * h;
    pwin += won;
    mass[L] -= won;
    // Downward, so no level climbs twice in one draw.
    for (int i = L - 1; i >= 0; i--) {
      const double moved = mass[i] * adv[i];
      mass[i] -= moved;
      mass[i + 1] += moved;
      if (i + 1 == L) arrived += moved;
    }
    double alive = 0;
    for (int i = 0; i <= L; i++) alive += mass[i];
    cost += alive * hazardCost(E, j + t, r);
    // THE SAME SURVIVAL THE EXACT PATH USES. `handvalue.ts` decays its mass by
    // the opponents' win chance alone and charges our own deal-in as a cost
    // against mass that never dies — harmless when the chain is only ever
    // ordered against itself, ruinous here: over fifteen turns it bills a
    // deal-in rate whose total exceeds one, and once the win side was on the
    // scorer's scale that cost sank every push line to the fold line (聴牌率
    // 3.8%). A hand that deals in is over, so it leaves the sweep.
    const double survive = surviveAt(E, j + t, r);
    for (int i = 0; i <= L; i++) mass[i] *= survive;
  }
  pwin = clamp01(pwin);

  double left = 0;
  for (int i = 0; i < L; i++) left += mass[i];
  dg.pWin = pwin;
  dg.pTenpai = clamp01(arrived);
  const double cal = tailCal(sh0);
  dg.winSum = pwin * value * cal * E.gain;
  // THE HAZARD HALF IS NOT CALIBRATED, IT IS THE EXACT PATH'S OWN (2026-08-31).
  //
  // `cal` used to scale the whole tail — win term, hazard cost and 流局
  // settlement together — and that is defensible only while the reads are
  // silent, which is the one table the seat never plays on. `TAIL_CAL` is
  // measured against the exact DP with `tenpaiP` at zero, so on a live table it
  // was discounting the COST by the same 0.4 it discounts the upside, and the
  // two models drifted 2-10× apart (the ratio table in
  // `test/ev_native_test.ts`): at Σ聴牌率 0.15 a 2向聴 rest priced −1244
  // exactly against −601 for a 4向聴 one on the tail — the worse hand worth
  // more.
  //
  // What is APPROXIMATE here is `handvalue.ts`'s chain: how fast a shapeless
  // hand advances and how often it collects. The sweep around it — `alive ×
  // hazardCost(j+t)` on mass decayed by `surviveAt`, and the 流局 settlement —
  // is the SAME arithmetic `turnValue` runs turn by turn, on the same
  // population rates, and it needs no calibration because it is not a
  // different model. So `cal` now scales the win term ALONE and the hazard
  // half rides at full weight, exactly as the exact path charges it.
  //
  // The earlier note against this ("scaling only the winnings left the LEVEL
  // uncalibrated ... a tail-priced hand crossed the fold line at Σ tenpaiP ≈
  // 0.02") was written when `TAIL_CAL` was a single constant fitted on 1–2向聴
  // and read 2–3× high at 3向聴+, i.e. against an upside that was still wrong
  // after scaling. With the per-rung table re-measured under this split the
  // fold verdicts agree again (`同じ手の 押し引き は 尾部と厳密で同じ判定`).
  dg.cost = cost;
  return cal * pwin * value * E.gain - cost + mass[L] * ryuukyokuValue(E, 0, r) +
      left * ryuukyokuValue(E, 1, r);
}

/** Riichi is available here: menzen, tenpai, a live wait, and a turn to use it. */
inline bool riichiEligible(Eval &E, int32_t sid, int j) {
  if (!E.riichiOn || E.ownRiichi || !E.closed) return false;
  if (E.X->shapes.v[sid].shanten != 0 || j >= E.T) return false;
  ensureMass(E, sid);
  return E.X->shapes.v[sid].mass > 0;
}

/**
 * One turn from a resting shape: draw, maybe win, otherwise choose what to
 * throw, then let the table act.
 *
 * The probability bookkeeping is the whole point (see the header comment):
 * `pool_k/N_j` is used AS IS for every branch that matters — a win, an
 * acceptance, a 待ち替え — and the leftover mass is charged to the collapsed
 * "nothing happened" branch, so the exactness lives where the decision does.
 */
double turnValue(Eval &E, int32_t sid, int j, int aka, int r, int ippatsu, Diag &dg,
                 int declare = 0) {
  const int s = E.X->shapes.v[sid].shanten;
  // DECLARING. The draw that happens at this turn is still a DAMA draw — the
  // declaration rides on the discard after it — so the win here scores without
  // the riichi han, and 一発 belongs to the next draw, which is the first one
  // the declaration has been alive for.
  const int winR = declare ? 0 : r;
  const int winIp = declare ? 0 : ippatsu;
  diagZero(dg);
  if (j >= E.T) {
    dg.pTenpai = s == 0 ? 1.0 : 0.0;
    return ryuukyokuValue(E, s, r);
  }
  if (E.nodes >= E.maxNodes) {
    // A REAL escape: this state was going to be searched and was not.
    E.trunc = 1;
    E.escapes++;
    return tailValue(E, sid, j, aka, r, dg);
  }
  E.nodes++;
  if (s > E.exactShanten) return tailValue(E, sid, j, aka, r, dg);

  uint8_t c[34];
  std::memcpy(c, E.X->shapes.v[sid].c, 34);
  double pool[34];
  poolOf(E, c, pool);
  double Nj = E.Nroot - j;
  if (Nj < 1) Nj = 1;

  double pr[34];
  if (E.hasDraw && j < E.K) {
    for (int t = 0; t < 34; t++) pr[t] = E.drawDist[j * 34 + t];
  } else {
    for (int t = 0; t < 34; t++) pr[t] = pool[t] / Nj;
  }

  // --- wins ---
  double winTerm = 0, wmass = 0;
  uint64_t wonMask = 0;
  if (s == 0) {
    const int32_t wi = winValsFor(E, sid, winR, aka, winIp);
    if (wi >= 0) {
      const WinVals &wv = E.X->wins.v[wi];
      const double ronF = 3.0 * E.p[P_RON_FACTOR];
      for (int i = 0; i < wv.n; i++) {
        const int w = wv.waitType[i];
        const double q = pr[w];
        if (q <= 0) continue;
        if (wv.cashT[i]) {
          winTerm += q * wv.tsumo[i];
          wmass += q;
          wonMask |= 1ull << w;
        }
        if (wv.cashR[i] && ronF > 0) {
          winTerm += q * ronF * wv.ron[i];
          wmass += q * ronF;
        }
      }
    }
  }

  // --- everything else ---
  double imass = 0, cont = 0;
  Diag contD;
  diagZero(contD);
  if (r == 0 && !declare) {
    if (!ensureEdges(E, sid)) {
      E.escapes++;
      return tailValue(E, sid, j, aka, r, dg);
    }
    const int32_t from = E.X->shapes.v[sid].edgeFrom;
    const int32_t to = E.X->shapes.v[sid].edgeTo;
    int i = from;
    while (i < to) {
      const int k = E.X->edges.v[i].k;
      int end = i;
      while (end < to && E.X->edges.v[end].k == k) end++;
      const double q = pr[k];
      if (q > 0 && !((wonMask >> k) & 1)) {
        double best = 0;
        Diag bestD;
        diagZero(bestD);
        bool have = false;
        for (int e = i; e < end; e++) {
          const int d = E.X->edges.v[e].d;
          // Which red 5p survive: a drawn 5p may be red, and a discarded 5p is
          // the plain copy first.
          const int drewAka = (k == 13 && E.akaUnseen > 0 && pool[13] > 0) ? 1 : 0;
          const double pa = drewAka ? clamp01(static_cast<double>(E.akaUnseen) / pool[13]) : 0.0;
          double v = 0;
          Diag vd;
          diagZero(vd);
          for (int hi = 0; hi <= drewAka; hi++) {
            const int aka14 = aka + hi;
            int akaNext = aka14;
            if (d == 13) {
              const int copies = c[13] + (k == 13 ? 1 : 0) - 1; // 5p left after the cut
              akaNext = aka14 < copies ? aka14 : copies;
            }
            if (akaNext > 2) akaNext = 2;
            if (akaNext < 0) akaNext = 0;
            const double w = drewAka ? (hi ? pa : 1.0 - pa) : 1.0;
            if (w > 0) {
              Diag sub;
              v += w * memoValue(E, E.X->edges.v[e].succ, j + 1, akaNext, r, sub);
              diagAdd(vd, sub, w);
            }
          }
          if (!have || v > best) {
            best = v;
            bestD = vd;
            have = true;
          }
        }
        if (have) {
          cont += q * best;
          imass += q;
          diagAdd(contD, bestD, q);
        }
      }
      i = end;
    }
  }

  // --- normalise: the excess mass is what we threw away (see the header ⚑) ---
  double scale = 1.0;
  const double tot = wmass + imass;
  double restP = 1.0 - tot;
  if (tot > 1.0) {
    scale = 1.0 / tot;
    restP = 0.0;
  }
  winTerm *= scale;
  cont *= scale;
  contD.pWin *= scale;
  contD.pTenpai *= scale;
  contD.winSum *= scale;
  contD.cost *= scale;
  const double wonP = wmass * scale;
  if (restP > 0) {
    Diag sub;
    // After a declaration the hand is locked and the next draw is the 一発 one.
    const double nv = declare ? turnValue(E, sid, j + 1, aka, 1, 1, sub, 0)
                              : memoValue(E, sid, j + 1, aka, r, sub);
    cont += restP * nv;
    diagAdd(contD, sub, restP);
  }

  const double notWon = 1.0 - wonP;
  // The table acts AFTER this draw and this discard, so j + 1 own draws have
  // gone by when it does. (The tail keeps `handvalue.ts`'s indexing, which is
  // what the parity test compares against.)
  const double surv = surviveAt(E, j + 1, r);
  const double haz = notWon * hazardCost(E, j + 1, r);
  dg.pWin = wonP + surv * contD.pWin;
  dg.pTenpai = s == 0 ? 1.0 : wonP + surv * contD.pTenpai;
  dg.winSum = winTerm + surv * contD.winSum;
  dg.cost = haz + surv * contD.cost;
  return winTerm - haz + surv * cont;
}

/**
 * The value of a resting shape: push, declare, or give up — whichever pays. The
 * memo key is the whole state, so two root candidates that converge on the same
 * shape at the same turn share every subtree below it.
 */
double memoValue(Eval &E, int32_t sid, int j, int aka, int r, Diag &dg) {
  const uint64_t key = ((static_cast<uint64_t>(sid) << 6 | static_cast<uint64_t>(E.thrown)) << 8) |
      (static_cast<uint64_t>(j) << 3) | (static_cast<uint64_t>(aka) << 1) | static_cast<uint64_t>(r);
  const int64_t got = tableGet(E.X->memo, key);
  if (got >= 0) {
    const double *v = &E.X->vals.v[got];
    (void)0;
    dg.pWin = v[1];
    dg.pTenpai = v[2];
    dg.winSum = v[3];
    dg.cost = v[4];
    return v[0];
  }

  const int64_t escapes0 = E.escapes;
  double v = turnValue(E, sid, j, aka, r, 0, dg);
  if (r == 0) {
    if (riichiEligible(E, sid, j)) {
      Diag rd;
      const double rv = turnValue(E, sid, j, aka, 1, 0, rd, 1) - 1000.0;
      if (rv > v) {
        v = rv;
        dg = rd;
      }
    }
    const double f = foldValue(E, j);
    if (f > v) {
      v = f;
      // Giving up: no win, and the hand is tenpai only if it already was.
      const double wasTenpai = E.X->shapes.v[sid].shanten == 0 ? 1.0 : 0.0;
      dg.pWin = 0;
      dg.pTenpai = wasTenpai;
      dg.winSum = 0;
      dg.cost = -f;
    }
  }
  // ⚑ A value built over a budget escape IS memoised. Refusing to store it —
  // which is the clean answer to "one candidate's exhaustion must not leak into
  // the next" — turns the shared DAG back into a tree the moment the escape
  // starts firing, because every ancestor then recomputes its whole subtree:
  // measured at 27× the states and 400 ms on a plain 聴牌 root against 0.9 ms.
  // What is done instead is to make the escape itself candidate-INDEPENDENT: a
  // fixed equal slice (above), so every candidate meets the wall in the same
  // place rather than later candidates inheriting a wall earlier ones hit.
  (void)escapes0;
  const int32_t idx = arrPush(E.X->vals, v);
  if (idx >= 0) {
    arrPush(E.X->vals, dg.pWin);
    arrPush(E.X->vals, dg.pTenpai);
    arrPush(E.X->vals, dg.winSum);
    if (arrPush(E.X->vals, dg.cost) >= 0) {
      tablePut(E.X->memo, key, static_cast<uint32_t>(idx));
    }
  }
  return v;
}

} // namespace

// ---------------------------------------------------------------------------
// ABI
// ---------------------------------------------------------------------------

/** Bumped together with `EV_ABI` in src/ai/evlayout.ts. */
MJ_EXPORT int32_t mjev_abi(void) { return EV_ABI_VERSION; }

/**
 * One evaluation context: the parameter vector in `EV_PARAM_ORDER` plus the
 * shanten memo. `n` must be exactly `EV_PARAMS_LEN`; anything else is a stale
 * wrapper and gets 0 rather than a silently misread vector.
 */
MJ_EXPORT int64_t mjev_create(const double *params, int32_t n) {
  if (!params || n != EV_PARAMS_LEN) return 0;
  EvCtx *ctx = static_cast<EvCtx *>(std::malloc(sizeof(EvCtx)));
  if (!ctx) return 0;
  for (int i = 0; i < EV_PARAMS_LEN; i++) ctx->p[i] = params[i];
  ctx->shapes.v = nullptr;
  ctx->shapes.n = ctx->shapes.cap = 0;
  ctx->geos.v = nullptr;
  ctx->geos.n = ctx->geos.cap = 0;
  ctx->gen = 0;
  ctx->edges.v = nullptr;
  ctx->edges.n = ctx->edges.cap = 0;
  ctx->vals.v = nullptr;
  ctx->vals.n = ctx->vals.cap = 0;
  ctx->wins.v = nullptr;
  ctx->wins.n = ctx->wins.cap = 0;
  if (!cacheInit(ctx->cache, CACHE_START) || !tableInit(ctx->shapeTab, 1u << 12) ||
      !tableInit(ctx->memo, 1u << 13) || !tableInit(ctx->winTab, 1u << 10)) {
    cacheFree(ctx->cache);
    tableFree(ctx->shapeTab);
    tableFree(ctx->memo);
    tableFree(ctx->winTab);
    std::free(ctx);
    return 0;
  }
  return static_cast<int64_t>(reinterpret_cast<intptr_t>(ctx));
}

MJ_EXPORT void mjev_destroy(int64_t handle) {
  if (!handle) return;
  EvCtx *ctx = reinterpret_cast<EvCtx *>(static_cast<intptr_t>(handle));
  cacheFree(ctx->cache);
  arrFree(ctx->shapes);
  arrFree(ctx->geos);
  arrFree(ctx->edges);
  arrFree(ctx->vals);
  arrFree(ctx->wins);
  tableFree(ctx->shapeTab);
  tableFree(ctx->memo);
  tableFree(ctx->winTab);
  std::free(ctx);
}

/**
 * Minimum shanten across standard / chiitoitsu / kokushi — `mj_shanten`'s
 * semantics exactly, but off the context's own memo.
 */
MJ_EXPORT int32_t mjev_shanten(int64_t handle, const uint8_t *counts, int32_t openMelds,
                               int32_t closed) {
  if (!counts) return 8;
  if (!handle) {
    // No context ⇒ no memo; the answer is still exact.
    WordCache none = {nullptr, nullptr, 0, 0};
    return shantenImpl(none, counts, openMelds, closed);
  }
  EvCtx *ctx = reinterpret_cast<EvCtx *>(static_cast<intptr_t>(handle));
  return shantenImpl(ctx->cache, counts, openMelds, closed);
}

/**
 * Bit t set ⇔ adding one tile of type t drops shanten below `base`.
 * Mirrors `ukeireTypes()`: types already held four times are skipped outright.
 */
MJ_EXPORT uint64_t mjev_ukeire_mask(int64_t handle, const uint8_t *counts, int32_t openMelds,
                                    int32_t closed, int32_t base) {
  if (!counts) return 0;
  WordCache none = {nullptr, nullptr, 0, 0};
  WordCache &wc = handle
      ? reinterpret_cast<EvCtx *>(static_cast<intptr_t>(handle))->cache
      : none;
  return ukeireMaskImpl(wc, counts, openMelds, closed, base);
}

/**
 * The stateless scorer: a packed `WinContext` in, `[ok, han, fu, base,
 * yakumanCount, limit, ronPayment, tsumoTotal]` out. No tile ids cross — the
 * hand is types and the melds are (kind, type, concealed) triples.
 *
 * `ok = 0` means 役なし (or a shape that does not win at all), and then every
 * other field is 0. Returns -1 on a null pointer, 0 otherwise.
 */
MJ_EXPORT int32_t mjev_score(const int32_t *in, int32_t *out) {
  if (!in || !out) return -1;
  for (int i = 0; i < SCORE_OUT_LEN; i++) out[i] = 0;

  WinCtx w;
  for (int t = 0; t < 34; t++) w.counts[t] = static_cast<uint8_t>(in[S_COUNTS + t]);
  w.nMelds = in[S_NMELDS];
  if (w.nMelds < 0 || w.nMelds > 4) return 0;

  w.menzen = true;
  for (int m = 0; m < w.nMelds; m++) {
    Block b;
    b.kind = in[S_MELDS + m * 3 + 0];
    b.type = in[S_MELDS + m * 3 + 1];
    b.concealed = in[S_MELDS + m * 3 + 2];
    if (b.type < 0 || b.type >= 34) return 0;
    w.called[m] = b;
    // 門前: an ankan does not open the hand, any other meld does.
    if (!(b.kind == BK_KAN && b.concealed)) w.menzen = false;
  }

  // hand + melds, a kan contributing 4 — invariant across readings.
  for (int t = 0; t < 34; t++) w.all[t] = w.counts[t];
  for (int m = 0; m < w.nMelds; m++) {
    const Block &b = w.called[m];
    if (b.kind == BK_RUN) {
      if (b.type % 9 > 6 || b.type >= 27) return 0;
      w.all[b.type]++;
      w.all[b.type + 1]++;
      w.all[b.type + 2]++;
    } else if (b.kind == BK_TRIPLET) {
      w.all[b.type] = static_cast<uint8_t>(w.all[b.type] + 3);
    } else if (b.kind == BK_KAN) {
      w.all[b.type] = static_cast<uint8_t>(w.all[b.type] + 4);
    } else {
      return 0;
    }
  }

  w.winType = in[S_WINTYPE];
  w.tsumo = in[S_TSUMO];
  w.riichi = in[S_RIICHI];
  w.doubleRiichi = in[S_DOUBLE];
  w.ippatsu = in[S_IPPATSU];
  w.rinshan = in[S_RINSHAN];
  w.chankan = in[S_CHANKAN];
  w.haitei = in[S_HAITEI];
  w.houtei = in[S_HOUTEI];
  w.tenhou = in[S_TENHOU];
  w.chiihou = in[S_CHIIHOU];
  w.seatWind = in[S_SEAT_WIND];
  w.roundWind = in[S_ROUND_WIND];
  for (int t = 0; t < 34; t++) {
    w.dora[t] = in[S_DORA + t];
    w.ura[t] = in[S_URA + t];
  }
  w.akaCount = in[S_AKA];
  w.kuitan = in[S_KUITAN];
  w.kazoe = in[S_KAZOE];
  w.kiriage = in[S_KIRIAGE];
  w.dwFu = in[S_DWFU];
  w.ippatsuCfg = in[S_IPPATSU_CFG];

  const Res r = scoreWin(w);
  if (!r.ok) return 0;

  // src/score.ts: 親 is the seat holding 東 as its seat wind.
  const bool dealerWins = w.seatWind == 27;
  const int ron = ceil100(r.base * (dealerWins ? 6 : 4));
  int tsumoTotal;
  if (dealerWins) {
    tsumoTotal = ceil100(r.base * 2) * 3;
  } else {
    tsumoTotal = ceil100(r.base * 2) + ceil100(r.base) * 2;
  }

  out[SO_OK] = 1;
  out[SO_HAN] = r.han;
  out[SO_FU] = r.fu;
  out[SO_BASE] = r.base;
  out[SO_YAKUMAN] = r.yakuman;
  out[SO_LIMIT] = r.limit;
  out[SO_RON] = ron;
  out[SO_TSUMO_TOTAL] = tsumoTotal;
  return 0;
}

// ---------------------------------------------------------------------------
// input validation and the per-call working set
// ---------------------------------------------------------------------------

inline bool finiteD(double x) { return x == x && x > -1e308 && x < 1e308; }

/** A supplied posterior: finite, non-negative, and a probability mass. */
int checkDist(const double *d) {
  double sum = 0;
  for (int i = 0; i < 34; i++) {
    if (!finiteD(d[i]) || d[i] < 0) return 2;
    sum += d[i];
  }
  return (sum > 1.0 - 1e-9 && sum < 1.0 + 1e-9) ? 0 : 2;
}

/**
 * BETWEEN ROOT CANDIDATES. Everything measured against a candidate's own pool
 * goes; everything that is a property of the 13 tiles alone stays.
 *
 *   GOES — the value memo and the wait prices (both already carry the thrown
 *   type in their key, so they could not be read across a candidate anyway, and
 *   dropping them keeps the arenas from growing 34-fold), and the edge lists,
 *   whose SELECTION and RANKING are pool-dependent.
 *
 *   STAYS — the shape arena and its index, and the per-draw `Geo`: shanten,
 *   acceptance mask, static hand value, and `discardAnalysis`'s answer are pure
 *   functions of the counts. This is the sharing the 529/203 repair had to
 *   withdraw wholesale because a `Shape` then also held `mass`; the generation
 *   stamp is what makes the safe half of it available again.
 *
 * The generation bump is the whole mechanism: no pool-dependent field can be
 * read by the next candidate without being recomputed first.
 */
void clearCand(EvCtx *X) {
  X->edges.n = 0;
  X->vals.n = 0;
  X->wins.n = 0;
  tableClear(X->memo);
  tableClear(X->winTab);
  // Generation 0 means "never computed" on a fresh shape, so it is skipped on
  // the wrap — 2^32 candidate boundaries is not reachable inside one call, but
  // a counter that can alias a sentinel is a silent wrong price, not a slow one.
  if (++X->gen == 0) X->gen = 1;
}

/** BETWEEN CALLS: nothing at all survives, the geometry included. */
void clearEval(EvCtx *X) {
  X->shapes.n = 0;
  X->geos.n = 0;
  tableClear(X->shapeTab);
  clearCand(X);
}

/**
 * Read one evaluation's wire into `E`, refusing anything that could only be a
 * caller bug: 1 a bad mode or hand, 2 a malformed hidden-information posterior.
 * Nothing is clamped into silent plausibility except the horizon, which the
 * packer already clamps and which the tail's own chain caps at 20 anyway.
 */
int parseEval(EvCtx *X, const int32_t *ints, const double *dbls, int mode, Eval &E) {
  if (ints[I_MODE] != mode) return 1;
  E.X = X;
  E.p = X->p;

  E.nMelds = ints[I_NMELDS];
  E.nKans = 0;
  if (E.nMelds < 0 || E.nMelds > 4) return 1;
  for (int t = 0; t < 34; t++) E.meldC[t] = 0;
  for (int m = 0; m < E.nMelds; m++) {
    Block b;
    b.kind = ints[I_MELDS + m * 3 + 0];
    b.type = ints[I_MELDS + m * 3 + 1];
    b.concealed = ints[I_MELDS + m * 3 + 2] ? 1 : 0;
    if (b.type < 0 || b.type >= 34) return 1;
    if (b.kind == BK_RUN) {
      if (b.type >= 27 || b.type % 9 > 6) return 1;
      E.meldC[b.type]++;
      E.meldC[b.type + 1]++;
      E.meldC[b.type + 2]++;
    } else if (b.kind == BK_TRIPLET) {
      E.meldC[b.type] = static_cast<uint8_t>(E.meldC[b.type] + 3);
    } else if (b.kind == BK_KAN) {
      E.meldC[b.type] = static_cast<uint8_t>(E.meldC[b.type] + 4);
      E.nKans++;
    } else {
      return 1;
    }
    E.melds[m] = b;
  }

  int sum = 0;
  for (int t = 0; t < 34; t++) {
    const int v = ints[I_HAND + t];
    if (v < 0 || v > 4) return 1;
    E.root[t] = static_cast<uint8_t>(v);
    sum += v;
  }
  // No type can exist five times across the hand and the melds.
  for (int t = 0; t < 34; t++) {
    if (E.root[t] + E.meldC[t] > 4) return 1;
  }
  // 14 − 3·melds for a discard root, 13 − 3·melds for a rest one.
  //
  // ⚑ A 槓 physically holds four tiles where a set holds three, so the seat's
  // real concealed count with a kan is one LOWER than this. The count is not
  // corrected here on purpose: `scoreWin` — and the TypeScript `scoreWin` it is
  // bit-compared against — takes the concealed part of a kan hand as
  // `need·3 + 2`, so subtracting the kan here alone would make every kan hand
  // fail the scorer's own shape check and never win. The two conventions have
  // to move together, and the TypeScript scorer is the one that defines it.
  if (sum != (mode == 0 ? 14 : 13) - 3 * E.nMelds) return 1;

  E.seatWind = ints[I_SEAT_WIND];
  E.roundWind = ints[I_ROUND_WIND];
  if (E.seatWind < 27 || E.seatWind > 30 || E.roundWind < 27 || E.roundWind > 30) return 1;
  E.dealer = ints[I_DEALER] ? 1 : 0;
  E.honba = ints[I_HONBA];
  E.kyotaku = ints[I_KYOTAKU];
  if (E.honba < 0 || E.kyotaku < 0) return 1;
  E.ownRiichi = ints[I_OWN_RIICHI] ? 1 : 0;
  E.furiten = (ints[I_FURITEN_PERM] || ints[I_FURITEN_TEMP]) ? 1 : 0;
  E.kanDoraOn = ints[I_KANDORA_ON] ? 1 : 0;
  E.closed = ints[I_CLOSED] ? 1 : 0;

  E.T = ints[I_T];
  if (E.T < 0) E.T = 0;
  if (E.T > T_MAX) E.T = T_MAX;

  E.akaUnseen = ints[I_AKA_UNSEEN];
  if (E.akaUnseen < 0) E.akaUnseen = 0;
  if (E.akaUnseen > 2) E.akaUnseen = 2;

  double nUnseen = 0; // kept for the record; the pool override arrives in counts
  for (int t = 0; t < 34; t++) {
    const int u = ints[I_UNSEEN + t];
    const int d = ints[I_DORA + t];
    const int rv = ints[I_RIVER + t];
    if (u < 0 || d < 0 || rv < 0) return 1;
    E.dora[t] = d;
    E.river[t] = rv;
    E.base[t] = u;
    nUnseen += u;
  }

  E.hasDraw = ints[I_HAS_DRAW] ? 1 : 0;
  E.K = ints[I_K];
  E.hasUra = ints[I_HAS_URA] ? 1 : 0;
  E.hasNextDora = ints[I_HAS_NEXTDORA] ? 1 : 0;
  E.drawDist = dbls + D_DRAW;
  E.uraDist = dbls + D_URA;
  E.nextDoraDist = dbls + D_NEXTDORA;
  E.forced = 0;
  if (E.hasDraw) {
    if (E.K < 1 || E.K > KMAX) return 2;
    for (int k = 0; k < E.K; k++) {
      const int rc = checkDist(dbls + D_DRAW + k * 34);
      if (rc) return rc;
      for (int t = 0; t < 34; t++) {
        if (dbls[D_DRAW + k * 34 + t] > 0) E.forced |= 1ull << t;
      }
    }
  } else {
    E.K = 0;
  }
  if (ints[I_HAS_POOL]) {
    // COUNTS, not a distribution: the live-wall composition arrives in the same
    // units as `unseen`, so it is checked for finiteness and non-negativity and
    // nothing else — `N_root = Σ pool` then still counts real tiles and
    // `N_j = N_root − j` keeps its meaning.
    double sum = 0;
    for (int t = 0; t < 34; t++) {
      if (!finiteD(dbls[D_POOL + t]) || dbls[D_POOL + t] < 0) return 2;
      sum += dbls[D_POOL + t];
    }
    // A composition that adds up to less than one tile is 河底 arithmetic, not
    // a malformed input: fall back to the counting posterior rather than
    // refusing the whole evaluation.
    if (sum >= 1.0) {
      for (int t = 0; t < 34; t++) E.base[t] = dbls[D_POOL + t];
    }
  }
  (void)nUnseen;
  if (E.hasUra) {
    const int rc = checkDist(dbls + D_URA);
    if (rc) return rc;
  }
  if (E.hasNextDora) {
    const int rc = checkDist(dbls + D_NEXTDORA);
    if (rc) return rc;
  }

  E.Nroot = 0;
  for (int t = 0; t < 34; t++) E.Nroot += E.base[t];
  // An empty wall is not a hand the DP can price: every draw probability would
  // be a division by a guard.
  if (E.Nroot < 1) return 1;
  for (int t = 0; t < 34; t++) E.indP[t] = E.Nroot > 0 ? E.base[t] / E.Nroot : 0.0;

  double wsum = 0, wloss = 0;
  for (int i = 0; i < 3; i++) {
    const double t = dbls[D_TENPAI + i];
    const double l = dbls[D_EXPLOSS + i];
    if (!finiteD(t) || !finiteD(l)) return 2;
    E.tenpaiP[i] = clamp01(t);
    wsum += E.tenpaiP[i];
    wloss += E.tenpaiP[i] * l;
  }
  E.sumTenpai = wsum;
  E.meanLoss = wsum > 0 ? wloss / wsum : 0.0;
  E.gain = dbls[D_GAIN];
  E.risk = dbls[D_RISK];
  if (!finiteD(E.gain) || !finiteD(E.risk) || E.gain < 0 || E.risk < 0) return 2;
  if (mode == 0) {
    for (int t = 0; t < 34; t++) {
      if (!finiteD(dbls[D_PIN + t]) || !finiteD(dbls[D_COSTIN + t])) return 2;
      if (dbls[D_COSTIN + t] < 0) return 2;
      if (dbls[D_PIN + t] < 0 || dbls[D_PIN + t] > 1) return 2;
    }
  }

  E.nodes = 0;
  E.trunc = 0;
  E.thrown = 34; // "nothing thrown" — the rest root, and the discard root's default
  E.escapes = 0;
  double mn = E.p[P_MAX_NODES];
  // A NEGATIVE budget is the oracle switch (see `Eval::noPrune`): its magnitude
  // is still the node cap, so the two modes can be run at the same budget and
  // the only difference between them is the pruning. `mergeEv` refuses a
  // negative `maxNodes`, so this cannot arrive from a ktune — only from a test
  // that packs the parameter vector itself.
  E.noPrune = 0;
  if (mn < 0) {
    E.noPrune = 1;
    mn = -mn;
  }
  if (!(mn >= 1)) mn = 1;
  if (mn > 4e6) mn = 4e6;
  E.maxNodes = static_cast<int64_t>(mn);
  E.exactShanten = static_cast<int>(E.p[P_EXACT_SHANTEN]);
  E.sameRungs = static_cast<int>(E.p[P_SAME_SHANTEN_RUNGS]);
  // `ev.riichi` (plan D3) says who MAKES the riichi decision, not whether riichi
  // exists. With the sub-switch off the seat keeps its own riichi block and
  // still declares, so a model that forbids the declaration prices every closed
  // hand as a damaten — which is how the tail came to read a 4200-point win
  // where the exact path could only ever collect a ~1300-point 門前清自摸和,
  // a threefold seam that made the seat prefer whichever side of
  // `exactShanten` it happened to be standing on. So the search always models
  // the declaration; only the ROOT's reported decision is gated.
  E.riichiOn = 1;
  E.riichiReport = E.p[P_RIICHI] != 0;
  // Distinct shapes are only useful to states that reach them, so the shape
  // arena is bounded by the node budget too — with room to spare, because one
  // expanded state mints up to a few hundred successors and a shape arena that
  // binds BEFORE the node budget is a truncation nobody asked for. The absolute
  // ceiling is what keeps a pathological hand from allocating without limit.
  //
  // ⚑ RAISED to 2^20 on 2026-08-31, when the shape arena stopped being emptied
  // between root candidates: it is now the UNION over the whole field rather
  // than one candidate's working set, and 262144 began to bind on wide 3向聴
  // roots (`4m5m7m9m 1p2p2p3p 1s2s ESSF` at T = 17 truncated at 1.84M value
  // states with the node budget still untouched). A truncation there is not a
  // slow answer, it is the whole field dropping to the closed-form tail. The
  // arena is grown by doubling and only ever reaches what a hand actually
  // needs, so the ceiling costs nothing until a hand claims it.
  int64_t cap = E.maxNodes * 8 + 4096;
  if (cap > (1 << 20)) cap = 1 << 20;
  E.shapeCapTotal = static_cast<uint32_t>(cap);
  E.shapeCap = E.shapeCapTotal;
  return 0;
}

// ---------------------------------------------------------------------------
// the two entry points
// ---------------------------------------------------------------------------

/**
 * Price every candidate discard. Per type: `total` (throw it and then play
 * on — push, declare or give up, whichever is best), `dama` and `riichi` (so
 * the TypeScript riichi gates run FIRST and only then consult the price), and
 * `foldLine` (throw it and give up now). Plus `bestPush`/`bestFold` over the
 * whole candidate set, which is the fold verdict: `bestFold > bestPush`.
 *
 * `bestPush` is the best PURE push, deliberately not the max of `total` — a
 * seat whose every line is better folded has to be able to say so, and a
 * `total` that already absorbed the fold option could never.
 */
MJ_EXPORT int32_t mjev_eval_discard(int64_t handle, const int32_t *ints, const double *dbls,
                                    double *out) {
  if (!out) return 3;
  // −∞ everywhere, not 0: a refusal must not leave a number an argmax could
  // take seriously if a caller ignored the return code.
  for (int i = 0; i < OUT_LEN; i++) out[i] = -__builtin_inf();
  out[O_NODES] = 0.0;
  out[O_TRUNC] = -1.0;
  if (!handle || !ints || !dbls) return 3;
  EvCtx *X = reinterpret_cast<EvCtx *>(static_cast<intptr_t>(handle));

  Eval E;
  const int rc = parseEval(X, ints, dbls, 0, E);
  if (rc) return rc;
  // ONCE for the whole field: the shape geometry is shared by every candidate
  // and by both attempts of the level ladder (a shape's 向聴 and acceptance do
  // not depend on how deep the search that reached it was allowed to go).
  clearEval(X);

  const double NEG = -__builtin_inf();
  const double qe = qEndAt(E, 0);
  const double endCost = qe * tsumoShareOf(E) * E.meanLoss * E.risk;
  const double fold0 = foldValue(E, 0);
  const int r0 = E.ownRiichi;

  int cands[34];
  int nCand = 0;
  int shMin = 99, shMax = -1;
  for (int ty = 0; ty < 34; ty++) {
    if (!ints[I_CAND + ty] || E.root[ty] == 0) continue;
    cands[nCand++] = ty;
    E.root[ty]--;
    const int s = shantenImpl(E.X->cache, E.root, E.nMelds, E.closed);
    E.root[ty]++;
    if (s < shMin) shMin = s;
    if (s > shMax) shMax = s;
  }

  // ONE CANDIDATE, ONE SEARCH, ONE ARENA — and every candidate priced by the
  // SAME model.
  //
  // Both halves of that sentence are repairs of the same defect, found by
  // pricing `4m5m7m9m 1p2p2p3p 1s2s ESSF` (3向聴, quiet table, T = 17) two ways:
  // `mjev_eval_discard` answered 416 for 切1s where `mjev_eval_rest` on the very
  // hand that discard leaves answered 224 — impossible by construction, since
  // the rest evaluation is `max(push, riichi, fold)` over the same subtree the
  // discard root pushes into and the root additionally pays the thrown tile's
  // 振聴.
  //
  // It was two things, both of them CROSS-CANDIDATE leakage:
  //
  //  1. THE NODE BUDGET WAS SLICED. `maxNodes` is one root's cap in
  //     `mjev_eval_rest`, but the discard root handed each candidate
  //     `maxNodes / nCand` of it. That slice was invented when the value memo
  //     was shared across candidates — which it has not been since the thrown
  //     tile joined the memo key (a discard into one's own wait is 振聴, and
  //     the whole subtree's ron cashability depends on it). With the memo
  //     partitioned there is nothing left to share, so the slice bought no
  //     speed and cost the answer: the candidates with the widest shapes are
  //     the ones whose subtrees do not fit, they escape onto the closed-form
  //     tail, and the tail prices a 3向聴 shape at 1.6–2.2× what the exact path
  //     collects. The argmax then selected whichever candidate the budget
  //     starved — systematically the shapeless one.
  //  2. THE SHAPE ARENA WAS SHARED, and a `Shape` caches POOL-DEPENDENT data
  //     (`mass`, and the top-three pruning `ensureEdges` performs off it),
  //     while the pool is `base − max(0, hand − root)` and the ROOT differs per
  //     candidate. A shape first analysed under 切1s was reused verbatim under
  //     切4m with the wrong live counts behind it. Same tile, different price,
  //     depending on which candidates preceded it.
  //
  // So each candidate now starts from an EMPTY arena with the FULL budget, and
  // is therefore evaluated by exactly the search `mjev_eval_rest` would run on
  // the 13 tiles it leaves. That makes `O_TOTAL(ty) ≤ eval_rest(rest_ty)` a
  // theorem rather than a hope (the root only ever subtracts: the thrown tile's
  // 振聴, and the root deal-in cost).
  //
  // THE LEVEL IS UNIFORM, AND IT IS CHOSEN, NOT ASSUMED. Paying the full budget
  // per candidate would cost `nCand ×` a rest evaluation, which at the default
  // `exactShanten` 3 is seconds per decision on a 3向聴 root. Worse, it would
  // not even buy a coherent answer: the cheap candidates would finish exactly
  // and the expensive ones would still escape, which is the seam above with a
  // bigger budget. So `exactShanten` is a MAXIMUM: the whole field is priced at
  // the deepest level every candidate can afford. A level whose first candidate
  // escapes is abandoned and the next one down is tried, until one completes
  // with no escape anywhere — at worst the level below zero, where every state
  // is the closed-form tail and the answer costs one node per candidate. The
  // reported `O_TRUNC` is 1 whenever the level actually used is below the
  // configured one, because that IS the truncation the caller has to know
  // about. ⚑ At a degraded level a candidate that keeps a worse shanten is
  // priced by the tail while one that advances is priced exactly — the
  // exact/tail seam, which `TAIL_CAL` exists to close and which no budget
  // policy can remove.
  const int64_t budget = E.maxNodes;
  const int cfgLevel = E.exactShanten;
  double vTotal[34], vDama[34], vRiichi[34], vFold[34];
  for (int t = 0; t < 34; t++) {
    vTotal[t] = NEG;
    vDama[t] = NEG;
    vRiichi[t] = NEG;
    vFold[t] = NEG;
  }
  int64_t totalNodes = 0;
  // AT MOST TWO ATTEMPTS, AND NEITHER OF THEM SPLITS THE FIELD.
  //
  // Any level at or above the WORST candidate's 向聴 runs the identical search
  // (nothing in the tree ever has a worse 向聴 than the root it grew from), so
  // there is exactly one "everybody exact" level — `shMax` — and walking down
  // from `exactShanten` to it would just pay the same failing probe several
  // times over. Below the BEST candidate's 向聴 there is exactly one "nobody
  // exact" level, and it costs one node per candidate.
  //
  // The levels in between are the ones that must not be used: they price the
  // advancing discards exactly and the shape-wrecking ones by the closed-form
  // tail, so the argmax is settled by whichever of the two models is the more
  // generous. `TAIL_CAL` exists to make them agree and is measured against the
  // DEFAULT hazards; under a FITTED hazard vector the tail reads 1.6-2.2× the
  // exact price at 3向聴, and the seat then wrecks its hand on purpose
  // (measured on `4m5m7m9m 1p2p2p3p 1s2s ESSF` with `weights/ev-0830b.json`:
  // 切9m to 3向聴 priced 444 against 切2p to 2向聴 priced 203, so 切9m was the
  // argmax). One model for the whole field, whichever model it has to be.
  int attempts[2];
  bool isTail[2];
  int nAttempt = 0;
  if (E.noPrune) {
    // ORACLE MODE keeps the configured level and nothing else: a discard is
    // allowed to make the hand WORSE there, so "any level at or above `shMax`
    // is the same search" — the premise the ladder rests on — is not true, and
    // a brute-force reference has no ladder to compare against anyway.
    attempts[nAttempt] = cfgLevel;
    isTail[nAttempt] = false;
    nAttempt++;
  } else {
    if (cfgLevel >= shMax) {
      attempts[nAttempt] = shMax;
      isTail[nAttempt] = false;
      nAttempt++;
    }
    attempts[nAttempt] = cfgLevel < shMin - 1 ? cfgLevel : shMin - 1;
    isTail[nAttempt] = true;
    nAttempt++;
  }
  int level = attempts[0];
  int chosen = 0;
  bool failed = false;
  for (int a = 0; a < nAttempt; a++) {
    chosen = a;
    level = attempts[a];
    // The tail attempt is not budgeted: no state below it is ever expanded, so
    // it costs one node per candidate and it is the floor that makes this
    // terminate.
    const bool floorLevel = isTail[a];
    E.exactShanten = level;
    int64_t used = 0;
    bool okAll = true;
    for (int i = 0; i < nCand && okAll; i++) {
      const int ty = cands[i];
      clearCand(X);
      E.nodes = 0;
      E.escapes = 0;
      E.trunc = 0;
      E.shapeCap = E.shapeCapTotal;
      E.maxNodes = floorLevel ? 8 : (budget - used > 1 ? budget - used : 1);

      // A discarded 5p is the plain copy first; the red one stays as long as it
      // can. `I_AKA_HELD` counts the reds in the hand AND in our own melds
      // (score.ts prices both), so the clamp is against the 5p we hold anywhere.
      int aka0 = ints[I_AKA_HELD];
      if (aka0 < 0) aka0 = 0;
      if (aka0 > 2) aka0 = 2;
      const int held5p = E.root[13] + E.meldC[13] - (ty == 13 ? 1 : 0);
      if (aka0 > held5p) aka0 = held5p;

      // THE CANDIDATE IS IN ITS OWN RIVER from here on (`E.thrown`, in the memo
      // and wait-price keys), and the pool is measured against the hand we are
      // LEFT with — `E.root` is the 14-tile root, so without the decrement a
      // drawn copy of the discarded type read as "no absorption" and the copy in
      // the river stayed in the wall.
      E.thrown = ty;
      E.root[ty]--;
      const int32_t sid = internShape(E, E.root);
      double push = 0, rv = 0;
      bool canRiichi = false;
      if (sid >= 0) {
        Diag pd;
        push = turnValue(E, sid, 0, aka0, r0, 0, pd);
        if (E.riichiReport && !r0 && riichiEligible(E, sid, 0)) {
          canRiichi = true;
          Diag rd;
          rv = turnValue(E, sid, 0, aka0, 1, 0, rd, 1) - 1000.0;
        }
      }
      E.root[ty]++;
      used += E.nodes;
      totalNodes += E.nodes;
      if (sid < 0 || E.trunc || E.escapes > 0) {
        okAll = false;
        break;
      }

      // The root deal-in is EXACT and already in points with the doctrine priced
      // in (`dealinCostPts` = `riskOf`'s arithmetic), so the standings `risk`
      // multiplier is NOT applied again here — and neither is `pIn`. `costIn` is
      // already the EXPECTED cost (Σ p·v, with the doctrine floor): multiplying
      // it by the probability a second time made a dangerous tile cost a fifth
      // of what it costs, and — because a threatened hand's continuation is
      // NEGATIVE — let the extra deal-in probability BUY the seat out of a bad
      // future. `pIn` does one job only: the hand ends on a deal-in, so it
      // removes survival mass from what comes after.
      const double pin = dbls[D_PIN + ty];
      const double cin = dbls[D_COSTIN + ty];
      const double rootHaz = -cin - endCost;
      double surv = 1.0 - pin - qe;
      if (surv < 0) surv = 0;

      // PUSH ONLY. Taking `max(push, fold)` here looked harmless — the fold is a
      // real option at every state — but `fold` does not depend on the shape, so
      // the moment it dominates, EVERY candidate collapses onto the same number
      // and the argmax is decided by the root deal-in cost alone. The seat then
      // discards the safest tile every turn and never advances (聴牌率 12%,
      // 放銃率 8%: not defence, paralysis). The fold line is reported in its own
      // slot and `bestFold > bestPush` is what declares the fold.
      vDama[ty] = rootHaz + surv * push;
      vRiichi[ty] = canRiichi ? rootHaz + surv * rv : NEG;
      vFold[ty] = r0 ? NEG : rootHaz + surv * fold0;
      vTotal[ty] = vRiichi[ty] > vDama[ty] ? vRiichi[ty] : vDama[ty];
    }
    failed = !okAll;
    if (okAll || a + 1 >= nAttempt) break;
    // A level nobody could finish tells us nothing about anybody: throw the
    // partial field away rather than mixing two levels in one argmax.
    for (int t = 0; t < 34; t++) {
      vTotal[t] = NEG;
      vDama[t] = NEG;
      vRiichi[t] = NEG;
      vFold[t] = NEG;
    }
  }

  double bestPush = NEG, bestFold = NEG;
  for (int i = 0; i < nCand; i++) {
    const int ty = cands[i];
    const int at = ty * O_STRIDE;
    out[at + O_TOTAL] = vTotal[ty];
    out[at + O_DAMA] = vDama[ty];
    out[at + O_RIICHI] = vRiichi[ty];
    out[at + O_FOLDLINE] = vFold[ty];
    // `bestPush` is the best PURE push — `O_TOTAL` with the declaration folded
    // in, which is what the seat would actually play — against `bestFold`.
    if (vTotal[ty] > bestPush) bestPush = vTotal[ty];
    if (vFold[ty] > bestFold) bestFold = vFold[ty];
  }

  E.thrown = 34;
  E.maxNodes = budget;
  E.exactShanten = cfgLevel;
  E.shapeCap = E.shapeCapTotal;
  out[O_NODES] = static_cast<double>(totalNodes);
  // TRUNCATED means "not the model you configured". Standing on `shMax` when
  // `exactShanten` is above it is not a fallback at all — the search is
  // identical — and neither is the tail attempt when the configured level was
  // already below every candidate. Answering from the tail when the configured
  // level would have expanded something IS, whether the exact attempt ran out
  // of budget or was never offered because it would have split the field.
  out[O_TRUNC] = (failed || (isTail[chosen] && cfgLevel >= shMin)) ? 1.0 : 0.0;
  out[O_BEST_PUSH] = bestPush;
  out[O_BEST_FOLD] = bestFold;
  return 0;
}

/**
 * What holding this 13-tile hand is worth — the PASS side of a call and the
 * post-暗槓 root. No discard is made here, so the only entry hazard is the
 * table's own chance to end the hand before our next draw.
 */
MJ_EXPORT double mjev_eval_rest(int64_t handle, const int32_t *ints, const double *dbls,
                                double *meta) {
  if (meta) {
    for (int i = 0; i < REST_META_LEN; i++) meta[i] = 0.0;
    meta[R_TRUNC] = -1.0;
  }
  const double bad = __builtin_nan("");
  if (!handle || !ints || !dbls) return bad;
  EvCtx *X = reinterpret_cast<EvCtx *>(static_cast<intptr_t>(handle));

  Eval E;
  if (parseEval(X, ints, dbls, 1, E)) return bad;
  clearEval(X);

  const int32_t sid = internShape(E, E.root);
  if (sid < 0) return bad;

  int aka0 = ints[I_AKA_HELD];
  if (aka0 < 0) aka0 = 0;
  if (aka0 > 2) aka0 = 2;
  if (aka0 > E.root[13] + E.meldC[13]) aka0 = E.root[13] + E.meldC[13];

  const double qe = qEndAt(E, 0);
  // `tsumoShareOf`, not the raw scalar: a dealer pays half of every opponent's
  // tsumo, and the discard root has always charged it that way. Reading the
  // parameter directly here made the two entry points disagree on the entry
  // hazard of a dealer's hand — the one comparison (`chooseCall`'s PASS line
  // against the post-call push) that puts them side by side.
  const double endCost = qe * tsumoShareOf(E) * E.meanLoss * E.risk;
  Diag dg;
  const double v = -endCost + (1.0 - qe) * memoValue(E, sid, 0, aka0, E.ownRiichi, dg);

  if (meta) {
    meta[R_VALUE] = v;
    meta[R_NODES] = static_cast<double>(E.nodes);
    meta[R_TRUNC] = E.escapes > 0 ? 1.0 : 0.0;
    // What the value is made of, under the same policy that produced it.
    meta[R_PTENPAI] = dg.pTenpai;
    meta[R_PWIN] = dg.pWin;
    meta[R_EVALUE] = dg.pWin > 0 ? dg.winSum / dg.pWin : 0.0;
    meta[R_ECOST] = dg.cost;
  }
  return v;
}
