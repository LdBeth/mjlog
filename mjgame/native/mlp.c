// mlp.c — the small learned heads of the 計算 seat, in C.
//
// This is a drop-in accelerator for the forward loop in `src/ai/mlp.ts`, which
// is itself the `src/rl/net.ts` loop. It is a SEMANTIC MIRROR of that loop, not
// "an MLP function": `test/mlp_native_test.ts` fuzzes the pair to EXACT
// equality — a single ulp of drift is a failure, because a head's output
// decides a fold and the seat's decision streams are pinned.
//
// Five entry points, ABI 1:
//
//   int32_t mjmlp_abi(void);
//   int64_t mjmlp_create(n_layers, dims[n_layers+1], acts[n_layers], blob);
//   void    mjmlp_forward(h, in, out);
//   void    mjmlp_forward_batch(h, n, in, out);
//   void    mjmlp_destroy(h);
//
// `blob` is the `packMlp` layout, which is `policy.f32`'s verbatim: per layer,
// in order, the row-major [out][in] weight matrix followed by the [out] bias,
// little-endian float32, no header and no padding. It is COPIED.
//
// ---------------------------------------------------------------------------
// The floating-point contract — the whole reason this file exists
// ---------------------------------------------------------------------------
//
// The reference is JavaScript, so the arithmetic is JavaScript's:
//
//   1. THE ACCUMULATOR IS A DOUBLE, seeded with the float32 bias. JS numbers
//      are IEEE doubles and `Float32Array` reads widen exactly, so
//      `let acc = b[o]; acc += w[i] * x[i];` is a double sum of exactly-widened
//      float32 products.
//   2. THE ORDER IS SEQUENTIAL AND ASCENDING — `o` outer ascending, `i` inner
//      ascending, one rounding per add. No blocking, no unrolled partial sums,
//      no vector reduction: those are a DIFFERENT double. (This is exactly why
//      `rlnet.c`'s Accelerate `sgemv` could not be reused — see native/README.md.)
//   3. RELU IS APPLIED TO THE DOUBLE, then the result is stored once as float32:
//      `y[o] = act ? (acc > 0 ? acc : 0) : acc` where `y` is a Float32Array, so
//      the store rounds to nearest-even — which is what C's `(float)` cast does.
//      Doubles therefore NEVER cross a layer boundary; layer k+1 reads the
//      float32 values layer k stored.
//
// Built with `-ffp-contract=off` (see native/build_mlp.sh, and keep the compile
// line in `test/mlp_native_test.ts` in step): without it clang may fuse
// `acc + w*x` into an FMA, which is a different double from the multiply-then-add
// JavaScript performs and would break rule 1 silently.
//
// What the differential fuzz actually catches, measured by mutation: replacing
// the double accumulator with a float one fails it immediately (rule 1 is the
// load-bearing rule). Reversing the inner loop does NOT — with a double
// accumulator over at most a few dozen float32 products, a reordering perturbs
// the sum ~1e-16 relatively and the store to float32 rounds that away again. So
// rule 2 is defence in depth, kept because the reorderings that WOULD show up
// are the ones a library does: float32 partial sums and vector reductions,
// which is exactly what Accelerate's sgemv would have given us.
//
// Nothing here knows any mahjong; it is dims, activations and floats.
//
// Not re-entrant: a context owns two scratch buffers. One caller at a time,
// which is exactly how Deno FFI uses it. `in` and `out` must not overlap.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MJMLP_EXPORT __attribute__((visibility("default")))

#define MJMLP_ABI 1

/* Sanity bounds. Heads are hundreds to a few thousand weights; anything past
 * these is a corrupt argument, not a network, and is refused rather than
 * multiplied out into an allocation. */
#define MJMLP_MAX_LAYERS 16
#define MJMLP_MAX_DIM 65536

typedef struct {
    int32_t n;          /* layer count                                     */
    int32_t *dims;      /* n+1: in0, out0(=in1), ..., outN-1               */
    uint8_t *acts;      /* n: 1 = relu, 0 = identity                       */
    float *blob;        /* copied weights, packMlp layout                  */
    float *scratch[2];  /* widest layer, alternated between hidden layers  */
} mjmlp;

MJMLP_EXPORT int32_t mjmlp_abi(void) { return MJMLP_ABI; }

MJMLP_EXPORT int64_t mjmlp_create(int32_t n_layers, const int32_t *dims,
                                  const uint8_t *acts, const float *blob) {
    if (n_layers < 1 || n_layers > MJMLP_MAX_LAYERS) return 0;
    if (dims == NULL || acts == NULL || blob == NULL) return 0;

    size_t n_floats = 0;
    int32_t widest = 0;
    for (int32_t i = 0; i <= n_layers; i++) {
        if (dims[i] < 1 || dims[i] > MJMLP_MAX_DIM) return 0;
        if (i > 0 && dims[i] > widest) widest = dims[i];
    }
    for (int32_t i = 0; i < n_layers; i++) {
        n_floats += (size_t)dims[i] * (size_t)dims[i + 1] + (size_t)dims[i + 1];
    }

    mjmlp *m = (mjmlp *)calloc(1, sizeof(mjmlp));
    if (m == NULL) return 0;
    m->n = n_layers;
    m->dims = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n_layers + 1));
    m->acts = (uint8_t *)malloc(sizeof(uint8_t) * (size_t)n_layers);
    m->blob = (float *)malloc(sizeof(float) * n_floats);
    m->scratch[0] = (float *)malloc(sizeof(float) * (size_t)widest);
    m->scratch[1] = (float *)malloc(sizeof(float) * (size_t)widest);
    if (m->dims == NULL || m->acts == NULL || m->blob == NULL ||
        m->scratch[0] == NULL || m->scratch[1] == NULL) {
        free(m->dims);
        free(m->acts);
        free(m->blob);
        free(m->scratch[0]);
        free(m->scratch[1]);
        free(m);
        return 0;
    }
    memcpy(m->dims, dims, sizeof(int32_t) * (size_t)(n_layers + 1));
    memcpy(m->acts, acts, sizeof(uint8_t) * (size_t)n_layers);
    memcpy(m->blob, blob, sizeof(float) * n_floats);
    return (int64_t)(intptr_t)m;
}

/* One row. The three rules of the contract, literally. */
static void forward_one(const mjmlp *m, const float *in, float *out) {
    const float *src = in;
    const float *w = m->blob;
    for (int32_t li = 0; li < m->n; li++) {
        const int32_t n_in = m->dims[li];
        const int32_t n_out = m->dims[li + 1];
        const float *b = w + (size_t)n_out * (size_t)n_in;
        /* The last layer writes the caller's buffer; hidden layers alternate
         * between the two scratches, so src and dst are never the same array. */
        float *dst = (li == m->n - 1) ? out : m->scratch[li & 1];
        for (int32_t o = 0; o < n_out; o++) {
            const float *row = w + (size_t)o * (size_t)n_in;
            double acc = (double)b[o];
            for (int32_t i = 0; i < n_in; i++) {
                acc += (double)row[i] * (double)src[i];
            }
            if (m->acts[li]) acc = acc > 0.0 ? acc : 0.0;
            dst[o] = (float)acc;
        }
        w = b + n_out;
        src = dst;
    }
}

MJMLP_EXPORT void mjmlp_forward(int64_t h, const float *in, float *out) {
    const mjmlp *m = (const mjmlp *)(intptr_t)h;
    if (m == NULL || in == NULL || out == NULL) return;
    forward_one(m, in, out);
}

/* `n` rows, contiguous: row r is in[r * dims[0] ...] and out[r * dims[n] ...].
 * Bit-identical to `n` calls of mjmlp_forward by construction — it IS `n`
 * calls; the batch exists to make it ONE FFI crossing. */
MJMLP_EXPORT void mjmlp_forward_batch(int64_t h, int32_t n, const float *in, float *out) {
    const mjmlp *m = (const mjmlp *)(intptr_t)h;
    if (m == NULL || in == NULL || out == NULL || n < 1) return;
    const size_t n_in = (size_t)m->dims[0];
    const size_t n_out = (size_t)m->dims[m->n];
    for (int32_t r = 0; r < n; r++) {
        forward_one(m, in + (size_t)r * n_in, out + (size_t)r * n_out);
    }
}

MJMLP_EXPORT void mjmlp_destroy(int64_t h) {
    mjmlp *m = (mjmlp *)(intptr_t)h;
    if (m == NULL) return;
    free(m->dims);
    free(m->acts);
    free(m->blob);
    free(m->scratch[0]);
    free(m->scratch[1]);
    free(m);
}
