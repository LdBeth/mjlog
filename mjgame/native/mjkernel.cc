// mjkernel.cc — the shanten / ukeire compute kernel.
//
// This is a drop-in accelerator for `mjrender/src/shanten.ts`. It is a
// SEMANTIC MIRROR of that file, not merely "a shanten function": every edge
// case the TypeScript has (cap < 0 → 8, chiitoitsu/kokushi only when closed and
// openMelds == 0, the `counts[t] >= 4` skip in ukeire) is reproduced here, and
// the differential fuzz in `test/kernel_native_test.ts` is what keeps it so.
//
// Three entry points:
//
//   int32_t  mj_shanten(counts[34], openMelds, closed)
//   uint64_t mj_ukeire_mask(counts[34], openMelds, closed, base)
//   void     mj_shape_masses(unseen[34], flags[34], honitsuSuit, toitoi, w[17], out[306])
//
// `mj_ukeire_mask` exists because the TypeScript ukeire probe costs 34 shanten
// evaluations; folding them into one FFI call is most of the win.
//
// `mj_shape_masses` is the 計算 reader's wait model — a SEMANTIC MIRROR of
// `shapeRowTS` in `src/ai/computed.ts`, in the same sense as the above, except
// that it is FLOAT arithmetic and so the mirroring has to be finer: every
// expression is transliterated with its own associativity intact, and the build
// passes `-ffp-contract=off` so the compiler may not fuse a multiply and an add
// into an FMA the JavaScript would not have performed. See the section at the
// bottom of this file. `test/kernel_native_test.ts` fuzzes the pair to EXACT
// equality; a single ulp of drift is a failure, because the seat's decision
// fingerprints are pinned.
//
// ---------------------------------------------------------------------------
// Algorithm
// ---------------------------------------------------------------------------
//
// The TypeScript computes standard-form shanten with one backtracking DFS over
// all 34 tile types. That DFS peels *blocks* (triplet / sequence / pair-head /
// pair / ryanmen / kanchan) off the leftmost non-empty tile, and blocks NEVER
// span a suit boundary. So the decomposition space factors into four
// independent groups — man, pin, sou, honors — and the objective
//
//     v = 2*min(M, cap) + min(P, cap - min(M, cap)) + H
//
// is monotone non-decreasing in the totals M (melds), P (partials) and H (head)
// that the groups contribute. That makes a per-group summary sufficient: for
// each (melds m ∈ 0..4, head h ∈ 0..1) the group only needs its MAXIMUM
// reachable partial count p. Ten small numbers, packed as ten nibbles of a
// uint64.
//
// So: tabulate that summary per group *distribution* (a suit is 9 counts of
// 0..4 → a base-5 index below 5^9; honors are 7 → below 5^7), then a shanten
// evaluation is four table reads, three 10x10 merges and a 10-way max. The
// tables are filled LAZILY by the very same block-peeling DFS, restricted to
// one group — so the table cannot disagree with the search that defines it, and
// a process pays only for the distributions it actually meets (real play touches
// a few thousand). A packed word of 0 means "not computed yet"; a computed
// entry is never 0, because (m=0, h=0, p=0) is reachable from every
// distribution.
//
// Clamping m and p at 4 is exact rather than approximate: cap = 4 - openMelds
// is at most 4, and both appear under a `min` with something ≤ cap.
//
// `mj_ukeire_mask` sharpens this further. Adding one tile perturbs exactly ONE
// group, so the merge of the other three is computed once per group and reused
// across that group's tiles: 34 probes cost ~34 merges instead of ~102.
//
// A reference transliteration of the TypeScript DFS (`refStandard`) is kept as
// the fallback for inputs outside the table's domain — a count above 4, a group
// holding more than 14 tiles, or a negative openMelds. Those never occur in
// play; they occur in fuzzing, and answering them exactly is cheaper than
// arguing about them.
//
// NOT re-entrant and not thread-safe: the lazy tables are written without a
// lock. One caller at a time, which is exactly how Deno FFI uses it.

#include <cstdint>
#include <cstdlib>
#include <cstring>

#define MJ_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

// ---------------------------------------------------------------------------
// reference: an exact transliteration of standardShanten() in shanten.ts
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// per-group summary tables
// ---------------------------------------------------------------------------

// best[m*2 + h] = max reachable partials for that (melds, head), -1 = no such
// decomposition. m and p are clamped at 4 (see the header note).
typedef int8_t Summary[10];

const int SUIT_STATES = 1953125; // 5^9
const int HONOR_STATES = 78125;  // 5^7

uint64_t *g_suitTab = nullptr;
uint64_t *g_honorTab = nullptr;

/**
 * The block-peeling DFS of shanten.ts restricted to ONE group, with the
 * pruning removed: the table has to hold the whole Pareto frontier, not just
 * the branch that happens to be best globally.
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

inline uint64_t groupWord(const uint8_t *c, int len, bool suited) {
  int idx = 0;
  for (int i = len - 1; i >= 0; i--) idx = idx * 5 + c[i];
  uint64_t *tab = suited ? g_suitTab : g_honorTab;
  uint64_t w = tab[idx];
  if (w == 0) {
    w = computeWord(c, len, suited);
    tab[idx] = w;
  }
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

/** Whether the table path applies. Fuzz reaches the `false` branch; play never does. */
inline bool tabulatable(const uint8_t *counts, int openMelds) {
  if (openMelds < 0 || openMelds > 4) return false;
  if (!g_suitTab || !g_honorTab) return false;
  for (int g = 0; g < 4; g++) {
    const int len = g == 3 ? 7 : 9;
    int total = 0;
    for (int i = 0; i < len; i++) {
      const int v = counts[g * 9 + i];
      if (v > 4) return false;
      total += v;
    }
    if (total > 14) return false; // keeps a pathological table fill impossible
  }
  return true;
}

inline void groupWords(const uint8_t *counts, uint64_t w[4]) {
  w[0] = groupWord(counts + 0, 9, true);
  w[1] = groupWord(counts + 9, 9, true);
  w[2] = groupWord(counts + 18, 9, true);
  w[3] = groupWord(counts + 27, 7, false);
}

inline int standardFromWords(const uint64_t w[4], int cap) {
  Summary a, b;
  mergeWord(IDENTITY, w[0], a);
  mergeWord(a, w[1], b);
  mergeWord(b, w[2], a);
  mergeWord(a, w[3], b);
  return 2 * cap - evalSummary(b, cap);
}

int standardShanten(const uint8_t *counts, int openMelds) {
  const int cap = 4 - openMelds;
  if (cap < 0) return 8;
  if (!tabulatable(counts, openMelds)) return refStandard(counts, openMelds);
  uint64_t w[4];
  groupWords(counts, w);
  return standardFromWords(w, cap);
}

// ---------------------------------------------------------------------------
// the two closed-only forms
// ---------------------------------------------------------------------------

const int YAOCHU[13] = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33};

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

int shantenImpl(const uint8_t *counts, int openMelds, int closed) {
  int s = standardShanten(counts, openMelds);
  if (closed && openMelds == 0) {
    const Closed st = closedStats(counts);
    const int ch = chiitoiFrom(st.pairs, st.kinds);
    const int ko = kokushiFrom(st.yKinds, st.yPair);
    if (ch < s) s = ch;
    if (ko < s) s = ko;
  }
  return s;
}

} // namespace

// ---------------------------------------------------------------------------
// ABI
// ---------------------------------------------------------------------------

/** Bumped whenever the meaning of an entry point changes, or one is added. */
MJ_EXPORT int32_t mj_kernel_version(void) { return 2; }

MJ_EXPORT int32_t mj_shanten(const uint8_t *counts, int32_t openMelds, int32_t closed) {
  if (!counts) return 8;
  if (!g_suitTab) {
    g_suitTab = static_cast<uint64_t *>(std::calloc(SUIT_STATES, sizeof(uint64_t)));
    g_honorTab = static_cast<uint64_t *>(std::calloc(HONOR_STATES, sizeof(uint64_t)));
  }
  return shantenImpl(counts, openMelds, closed);
}

/**
 * Bit t set ⇔ adding one tile of type t drops shanten below `base`.
 * Mirrors ukeireTypes(): types already held four times are skipped outright.
 */
MJ_EXPORT uint64_t mj_ukeire_mask(const uint8_t *counts, int32_t openMelds, int32_t closed,
                                  int32_t base) {
  if (!counts) return 0;
  if (!g_suitTab) {
    g_suitTab = static_cast<uint64_t *>(std::calloc(SUIT_STATES, sizeof(uint64_t)));
    g_honorTab = static_cast<uint64_t *>(std::calloc(HONOR_STATES, sizeof(uint64_t)));
  }

  uint8_t c[34];
  std::memcpy(c, counts, 34);

  const int cap = 4 - openMelds;
  const bool cl = closed && openMelds == 0;
  uint64_t mask = 0;

  // Slow, exact path: whatever the table cannot index, the reference DFS can.
  if (cap < 0 || !tabulatable(c, openMelds)) {
    for (int t = 0; t < 34; t++) {
      if (c[t] >= 4) continue;
      c[t]++;
      const int s = shantenImpl(c, openMelds, closed);
      c[t]--;
      if (s < base) mask |= 1ull << t;
    }
    return mask;
  }

  uint64_t w[4];
  groupWords(c, w);

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
    mergeWord(rest[g], groupWord(tmp, len, g != 3), merged);
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

// ---------------------------------------------------------------------------
// the wait model: mj_shape_masses
// ---------------------------------------------------------------------------
//
// A transliteration of `shapeRowTS` (src/ai/computed.ts), which is itself the
// flat twin of `shapeBaseMasses` + `waitRowFrom` + `combineShapes` there. Read
// those for what the arithmetic MEANS — 現物 kills a cell outright, スジ kills
// one リャンメン holding and books the residue, the counts are ways-to-hold
// numerators over fixed denominators. What follows only has to reproduce it.
//
// Three rules keep it bit-identical to the JavaScript:
//
//   1. Associativity is copied, not simplified. `prK * kanchan / 16` is
//      `(prK * kanchan) / 16` while リャンメン is `prR * (mass / 32)`; float
//      multiplication is not associative and those are different numbers.
//   2. No contraction. `-ffp-contract=off` in native/build_kernel.sh (and in
//      the test's own compile line) — an FMA is a DIFFERENT result from a
//      multiply followed by an add, and JavaScript never fuses.
//   3. `Math.min(1, x)` is not `fmin(1.0, x)`: they disagree on NaN, which
//      `jsMin1` below settles the JavaScript way. Nothing in play produces a
//      NaN, and a rule that only matters off the beaten path is exactly the
//      kind that decays if it is not written down.
//
// The weight vector `w` is packed by `packShapeWeights`; slot 5 is already
// `doraBridge − 1`, so this file performs no arithmetic on the weights at all.

namespace {

/** `Math.min(1, x)` — including its NaN, which fmin(1.0, NaN) would swallow. */
inline double jsMin1(double x) { return (x != x) ? x : (x < 1.0 ? x : 1.0); }

const int SHAPE_BF = 8; // doubles per tile type in the base block

// flags[] bits
const int F_GENBUTSU = 1;
const int F_VALUE_HONOR = 2;
const int F_DORA = 4;

// base block slots, in the field order of `ShapeBase`
const int B_RYANMEN = 0;
const int B_RYANMEN_DORA = 1;
const int B_RYANMEN_HALF = 2;
const int B_RYANMEN_FULL = 3;
const int B_KANCHAN = 4;
const int B_PENCHAN = 5;
const int B_SHANPON = 6;
const int B_TANKI = 7;

/** `shapeBasesFlat`: the parameter-free counts, integer throughout. */
void shapeBases(const int32_t *unseen, const int32_t *flags, double *base) {
  for (int i = 0; i < 34 * SHAPE_BF; i++) base[i] = 0.0;
  for (int ty = 0; ty < 34; ty++) {
    if (flags[ty] & F_GENBUTSU) continue; // 現物 — furiten, every shape refuted
    double *b = base + ty * SHAPE_BF;
    const int uty = unseen[ty];
    b[B_SHANPON] = uty < 2 ? 0.0 : static_cast<double>(uty * (uty - 1)) / 2.0;
    b[B_TANKI] = static_cast<double>(uty);
    if (ty >= 27) continue; // 字牌: シャンポン/タンキ IS the whole mechanism
    const int r = ty % 9 + 1;
    const bool up = r <= 6; // the (ty+1, ty+2) bridge
    const bool dn = r >= 4; // the (ty-1, ty-2) bridge
    const bool upRef = up && (flags[ty + 3] & F_GENBUTSU) != 0;
    const bool dnRef = dn && (flags[ty - 3] & F_GENBUTSU) != 0;
    const bool full = (!up || upRef) && (!dn || dnRef); // 全スジ
    int ryan = 0, rdora = 0, rhalf = 0, rfull = 0;
    if (up) {
      const int m = unseen[ty + 1] * unseen[ty + 2];
      if (!upRef) {
        ryan += m;
        if ((flags[ty + 1] & F_DORA) || (flags[ty + 2] & F_DORA)) rdora += m;
      } else if (full) {
        rfull += m;
      } else {
        rhalf += m;
      }
    }
    if (dn) {
      const int m = unseen[ty - 1] * unseen[ty - 2];
      if (!dnRef) {
        ryan += m;
        if ((flags[ty - 1] & F_DORA) || (flags[ty - 2] & F_DORA)) rdora += m;
      } else if (full) {
        rfull += m;
      } else {
        rhalf += m;
      }
    }
    b[B_RYANMEN] = static_cast<double>(ryan);
    b[B_RYANMEN_DORA] = static_cast<double>(rdora);
    b[B_RYANMEN_HALF] = static_cast<double>(rhalf);
    b[B_RYANMEN_FULL] = static_cast<double>(rfull);
    if (r >= 2 && r <= 8) b[B_KANCHAN] = static_cast<double>(unseen[ty - 1] * unseen[ty + 1]);
    if (r == 3) b[B_PENCHAN] = static_cast<double>(unseen[ty - 1] * unseen[ty - 2]);
    if (r == 7) b[B_PENCHAN] = static_cast<double>(unseen[ty + 1] * unseen[ty + 2]);
  }
}

} // namespace

/**
 * One opponent's whole row: P(waiting on each type | tenpai), plus the counts.
 *
 *   unseen[34]   copies not visible to the observer, 0..4
 *   flags[34]    bit0 現物, bit1 役牌 for this opponent, bit2 ドラ type
 *   honitsuSuit  0 なし, 1 m, 2 p, 3 s  (a tile's own 字 suit is 0 too)
 *   toitoi       トイトイ模様, 0/1
 *   w[17]        packed weights; see packShapeWeights()
 *   out          34 likelihoods, then 34 * 8 counts
 */
MJ_EXPORT void mj_shape_masses(const int32_t *unseen, const int32_t *flags, int32_t honitsuSuit,
                               int32_t toitoi, const double *w, double *out) {
  if (!unseen || !flags || !w || !out) return;
  double *row = out;
  double *base = out + 34;
  shapeBases(unseen, flags, base);

  const double prR = w[0], prK = w[1], prP = w[2], prS = w[3], prT = w[4];
  const double dB = w[5], sH = w[6], sF = w[7];
  const double yakuhaiShanpon = w[8], doraPairW = w[9];
  const double hot = w[10], cold = w[11];
  const double run = toitoi ? w[13] : 1.0;  // toitoiRun
  const double pair = toitoi ? w[12] : 1.0; // toitoiPair

  double total = 0.0;
  for (int ty = 0; ty < 34; ty++) {
    const double *b = base + ty * SHAPE_BF;
    const double ryanmen =
        b[B_RYANMEN] + dB * b[B_RYANMEN_DORA] + sH * b[B_RYANMEN_HALF] + sF * b[B_RYANMEN_FULL];
    const double doraPair = (flags[ty] & F_DORA) ? doraPairW : 1.0;
    double sRyan = prR * (ryanmen / 32.0);
    double sKan = prK * b[B_KANCHAN] / 16.0;
    double sPen = prP * b[B_PENCHAN] / 16.0;
    double sSha = prS * (b[B_SHANPON] / 6.0) *
                  ((flags[ty] & F_VALUE_HONOR) ? yakuhaiShanpon : 1.0) * doraPair;
    double sTan = prT * (b[B_TANKI] / 4.0) * doraPair;
    // applyMeldRead, shape by shape: a killed shape stays killed.
    const int suit = ty < 9 ? 1 : ty < 18 ? 2 : ty < 27 ? 3 : 0;
    const double flush =
        honitsuSuit == 0 ? 1.0 : ((suit == honitsuSuit || suit == 0) ? hot : cold);
    if (sRyan != 0.0) sRyan *= flush * run;
    if (sKan != 0.0) sKan *= flush * run;
    if (sPen != 0.0) sPen *= flush * run;
    if (sSha != 0.0) sSha *= flush * pair;
    if (sTan != 0.0) sTan *= flush * pair;
    const double mass = sRyan + sKan + sPen + sSha + sTan;
    row[ty] = mass;
    total += mass;
  }

  if (w[16] == 0.0) { // !waitNormalize
    const double sc = w[14];
    for (int ty = 0; ty < 34; ty++) row[ty] = jsMin1(row[ty] * sc);
    return;
  }
  if (total <= 0.0) {
    for (int ty = 0; ty < 34; ty++) row[ty] = 0.0;
    return;
  }
  const double expMass = w[15];
  for (int ty = 0; ty < 34; ty++) {
    const double m = row[ty];
    row[ty] = m <= 0.0 ? 0.0 : jsMin1(expMass * (m / total));
  }
}
