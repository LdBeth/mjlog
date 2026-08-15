// rlnet — a generic dense-MLP forward pass over Accelerate's cblas_sgemv,
// plus (v4) a fixed-shape self-attention sequence encoder.
//
// There is NO mahjong in the MLP part of this file, and deliberately so: it
// takes a layer chain (dims + activations) and one flat float32 blob laid out
// exactly like mjgame's `policy.f32` (per layer: the row-major [out][in] weight
// matrix, then the [out] bias), copies what it needs into a malloc'd context,
// and evaluates it. The TypeScript side (`src/rl/net.ts`) is what knows the
// blob came from a policy network; here it is just numbers.
//
// The v4 attention encoder (`rlnet_attn_*`, bottom of the file) is the one
// shape-aware piece: it implements train/V4_SPEC.md's "Encoder forward"
// verbatim (D=64, H=4, head dim 16, scale 1/4) over packed int8 river tokens.
// It still knows no mahjong rules — only the 42-dim dense expansion the spec
// fixes. Its output z[64] is handed back to TypeScript, which concatenates it
// into the 1738-wide policy input and calls the generic MLP above; no
// concatenation happens in C.
//
// Build:
//   clang -O3 -dynamiclib -framework Accelerate -o native/librlnet.dylib native/rlnet.c
// or `deno task build-native` from mjgame/.
//
// THREAD SAFETY: a context owns two scratch buffers and is therefore NOT safe
// to call concurrently. One context per network instance (which is what
// `net.ts` creates — one per loaded `Net`), and one caller at a time.
//
// `ACCELERATE_NEW_LAPACK` is defined before the include so the plain build
// command above stays warning-free: the legacy CBLAS headers are deprecated as
// of macOS 13.3. It selects headers, not an ABI change for the 32-bit-int
// (non-ILP64) entry points this file uses.

#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int32_t in;
  int32_t out;
  uint8_t relu;
  const float *w; /* [out][in], row-major, into ctx->params */
  const float *b; /* [out] */
} rlnet_layer;

typedef struct {
  int32_t n_layers;
  int32_t width;   /* widest layer output — the scratch buffer size */
  rlnet_layer *layers;
  float *params;   /* one allocation holding the whole blob copy */
  float *scratch0;
  float *scratch1;
} rlnet_ctx;

/* Guard against absurd dimensions before any multiplication is done. */
#define RLNET_MAX_DIM 1000000

/*
 * Builds a context. `dims` is n_layers+1 long (in0, out0(=in1), ..., outN),
 * `acts` is n_layers long (1 = relu, 0 = identity), `blob` holds the
 * concatenated weights+bias of every layer in order. Everything is COPIED, so
 * the caller may free its buffers immediately. Returns an opaque handle, or 0
 * if the arguments are unusable or an allocation fails.
 */
int64_t rlnet_create(int32_t n_layers, const int32_t *dims, const uint8_t *acts,
                     const float *blob) {
  if (n_layers <= 0 || dims == NULL || acts == NULL || blob == NULL) return 0;

  for (int32_t i = 0; i <= n_layers; i++) {
    if (dims[i] <= 0 || dims[i] > RLNET_MAX_DIM) return 0;
  }

  size_t total = 0;
  int32_t width = 0;
  for (int32_t i = 0; i < n_layers; i++) {
    const size_t in = (size_t)dims[i], out = (size_t)dims[i + 1];
    total += out * in + out;
    if (dims[i + 1] > width) width = dims[i + 1];
  }

  rlnet_ctx *ctx = (rlnet_ctx *)calloc(1, sizeof(rlnet_ctx));
  if (ctx == NULL) return 0;
  ctx->n_layers = n_layers;
  ctx->width = width;
  ctx->layers = (rlnet_layer *)calloc((size_t)n_layers, sizeof(rlnet_layer));
  ctx->params = (float *)malloc(total * sizeof(float));
  ctx->scratch0 = (float *)malloc((size_t)width * sizeof(float));
  ctx->scratch1 = (float *)malloc((size_t)width * sizeof(float));
  if (ctx->layers == NULL || ctx->params == NULL || ctx->scratch0 == NULL ||
      ctx->scratch1 == NULL) {
    free(ctx->layers);
    free(ctx->params);
    free(ctx->scratch0);
    free(ctx->scratch1);
    free(ctx);
    return 0;
  }
  memcpy(ctx->params, blob, total * sizeof(float));

  size_t off = 0;
  for (int32_t i = 0; i < n_layers; i++) {
    const int32_t in = dims[i], out = dims[i + 1];
    ctx->layers[i].in = in;
    ctx->layers[i].out = out;
    ctx->layers[i].relu = acts[i] ? 1 : 0;
    ctx->layers[i].w = ctx->params + off;
    off += (size_t)out * (size_t)in;
    ctx->layers[i].b = ctx->params + off;
    off += (size_t)out;
  }
  return (int64_t)(intptr_t)ctx;
}

/*
 * One forward pass: `in` is dims[0] long, `out` is dims[n_layers] long. The two
 * buffers must not overlap, and neither may alias the context's scratch.
 *
 * Each layer is bias-then-gemv: the bias is memcpy'd into the destination and
 * cblas_sgemv accumulates onto it with beta=1, so no separate add pass runs.
 * Weights are row-major [out][in], which is exactly CblasRowMajor/CblasNoTrans
 * with lda=in — nothing is transposed on the way in.
 */
void rlnet_forward(int64_t handle, const float *in, float *out) {
  rlnet_ctx *ctx = (rlnet_ctx *)(intptr_t)handle;
  if (ctx == NULL || in == NULL || out == NULL) return;

  const float *x = in;
  for (int32_t i = 0; i < ctx->n_layers; i++) {
    const rlnet_layer *l = &ctx->layers[i];
    /* The final layer writes straight into the caller's buffer; the others
       ping-pong between the two scratch buffers so `x` is never the target. */
    float *y = (i + 1 == ctx->n_layers) ? out
                                        : (x == ctx->scratch0 ? ctx->scratch1 : ctx->scratch0);
    memcpy(y, l->b, (size_t)l->out * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, l->out, l->in, 1.0f, l->w, l->in, x, 1, 1.0f, y, 1);
    if (l->relu) {
      for (int32_t o = 0; o < l->out; o++) {
        if (y[o] < 0.0f) y[o] = 0.0f;
      }
    }
    x = y;
  }
}

/* Frees a context. A 0 handle is a no-op; a handle must not be used after. */
void rlnet_destroy(int64_t handle) {
  rlnet_ctx *ctx = (rlnet_ctx *)(intptr_t)handle;
  if (ctx == NULL) return;
  free(ctx->layers);
  free(ctx->params);
  free(ctx->scratch0);
  free(ctx->scratch1);
  free(ctx);
}

/* ------------------------------------------------------------------------
 * v4: attention river encoder (train/V4_SPEC.md).
 *
 * Everything about the shape is fixed by the spec, so nothing here is a
 * parameter: D=64, H=4, head dim 16, softmax scale 1/4, x is 42-dim, L<=96.
 * The context holds every scratch matrix it will ever need (~200 KB), so
 * `rlnet_attn_encode` performs no allocation at all.
 * ------------------------------------------------------------------------ */

#define ATTN_D 64                        /* d_model                          */
#define ATTN_H 4                         /* heads                            */
#define ATTN_DH (ATTN_D / ATTN_H)        /* head dim = 16                    */
#define ATTN_X 42                        /* dense token width                */
#define ATTN_SEQ_MAX 96                  /* SEQ_MAX                          */
#define ATTN_SCALE 0.25f                 /* 1/sqrt(16), spelled 1/4 in spec  */

/* Layout of attn.f32, in file order. NOTE: this total is derived from the
   spec's own tensor list; V4_SPEC.md's prose says 23,872, which contradicts
   its list by exactly 256 floats (4*64) — see native/README.md. */
#define ATTN_N_FLOATS                                                          \
  (ATTN_D * ATTN_X + ATTN_D          /* W_in, b_in            */               \
   + 4 * (ATTN_D * ATTN_D + ATTN_D)  /* Wq,bq Wk,bk Wv,bv Wo,bo */             \
   + ATTN_D                          /* u                     */               \
   + ATTN_D * ATTN_D + ATTN_D)       /* Wz, bz                */

typedef struct rlnet_attn rlnet_attn;

struct rlnet_attn {
  float *params; /* one allocation: the whole file, in file order */
  /* views into params */
  const float *w_in, *b_in;
  const float *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo;
  const float *u;
  const float *wz, *bz;
  /* scratch, one allocation (see rlnet_attn_create) */
  float *scratch;
  float *x; /* [L][42]  dense tokens        */
  float *h; /* [L][64]  relu(W_in x + b_in) */
  float *q; /* [L][64]                      */
  float *k; /* [L][64]                      */
  float *v; /* [L][64]                      */
  float *o; /* [L][64]  concat of heads     */
  float *m; /* [L][64]  Wo O + bo           */
  float *s; /* [L][L]   one head's scores   */
  float *a; /* [L]      pooling weights     */
  float *p; /* [64]     pooled vector       */
};

/* In-place softmax over `n` contiguous floats, max-subtracted. Called on
   attention rows and on the pooling logits. `n >= 1` is a precondition; the
   sum is then >= 1 (the max element contributes expf(0) = 1), so the divide
   cannot produce inf/NaN however extreme the logits are. */
static void attn_softmax(float *v, int n) {
  float max = v[0];
  for (int i = 1; i < n; i++) {
    if (v[i] > max) max = v[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    const float e = expf(v[i] - max);
    v[i] = e;
    sum += e;
  }
  const float inv = 1.0f / sum;
  for (int i = 0; i < n; i++) v[i] *= inv;
}

/* Copies `bias` [64] into each of the `rows` rows of `dst` [rows][64], so the
   following sgemm can accumulate onto it with beta=1 (same trick as the MLP). */
static void attn_fill_bias(float *dst, const float *bias, int rows) {
  for (int i = 0; i < rows; i++) {
    memcpy(dst + (size_t)i * ATTN_D, bias, ATTN_D * sizeof(float));
  }
}

/* dst[rows][64] = src[rows][in_dim] * W^T + bias, W being [64][in_dim]. */
static void attn_affine(float *dst, const float *src, int in_dim, const float *w,
                        const float *bias, int rows) {
  attn_fill_bias(dst, bias, rows);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, ATTN_D, in_dim, 1.0f, src,
              in_dim, w, in_dim, 1.0f, dst, ATTN_D);
}

/*
 * Loads `attn.f32` — exactly ATTN_N_FLOATS little-endian float32, no header,
 * no padding, in the spec's tensor order. A file of any other size is
 * REJECTED (returns NULL) rather than partially read: a size mismatch means
 * the writer and this reader disagree about the layout, and that must fail
 * loudly at load time instead of silently producing garbage z.
 *
 * Returns NULL on a bad path, wrong size, short read or failed allocation.
 * The caller owns the returned pointer until rlnet_attn_destroy.
 */
rlnet_attn *rlnet_attn_create(const char *attn_f32_path) {
  if (attn_f32_path == NULL) return NULL;

  FILE *f = fopen(attn_f32_path, "rb");
  if (f == NULL) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  const long size = ftell(f);
  if (size != (long)(ATTN_N_FLOATS * sizeof(float)) || fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return NULL;
  }

  rlnet_attn *ctx = (rlnet_attn *)calloc(1, sizeof(rlnet_attn));
  if (ctx == NULL) {
    fclose(f);
    return NULL;
  }
  ctx->params = (float *)malloc(ATTN_N_FLOATS * sizeof(float));
  /* x, h, q, k, v, o, m, s, a, p — one block, in that order. */
  const size_t scratch_n = (size_t)ATTN_SEQ_MAX * ATTN_X +
                           (size_t)ATTN_SEQ_MAX * ATTN_D * 6 +
                           (size_t)ATTN_SEQ_MAX * ATTN_SEQ_MAX + ATTN_SEQ_MAX + ATTN_D;
  ctx->scratch = (float *)malloc(scratch_n * sizeof(float));
  if (ctx->params == NULL || ctx->scratch == NULL) {
    free(ctx->params);
    free(ctx->scratch);
    free(ctx);
    fclose(f);
    return NULL;
  }
  const size_t got = fread(ctx->params, sizeof(float), ATTN_N_FLOATS, f);
  fclose(f);
  if (got != (size_t)ATTN_N_FLOATS) {
    free(ctx->params);
    free(ctx->scratch);
    free(ctx);
    return NULL;
  }

  float *p = ctx->params;
  ctx->w_in = p; p += ATTN_D * ATTN_X;  ctx->b_in = p; p += ATTN_D;
  ctx->wq   = p; p += ATTN_D * ATTN_D;  ctx->bq   = p; p += ATTN_D;
  ctx->wk   = p; p += ATTN_D * ATTN_D;  ctx->bk   = p; p += ATTN_D;
  ctx->wv   = p; p += ATTN_D * ATTN_D;  ctx->bv   = p; p += ATTN_D;
  ctx->wo   = p; p += ATTN_D * ATTN_D;  ctx->bo   = p; p += ATTN_D;
  ctx->u    = p; p += ATTN_D;
  ctx->wz   = p; p += ATTN_D * ATTN_D;  ctx->bz   = p;

  float *s = ctx->scratch;
  ctx->x = s; s += (size_t)ATTN_SEQ_MAX * ATTN_X;
  ctx->h = s; s += (size_t)ATTN_SEQ_MAX * ATTN_D;
  ctx->q = s; s += (size_t)ATTN_SEQ_MAX * ATTN_D;
  ctx->k = s; s += (size_t)ATTN_SEQ_MAX * ATTN_D;
  ctx->v = s; s += (size_t)ATTN_SEQ_MAX * ATTN_D;
  ctx->o = s; s += (size_t)ATTN_SEQ_MAX * ATTN_D;
  ctx->m = s; s += (size_t)ATTN_SEQ_MAX * ATTN_D;
  ctx->s = s; s += (size_t)ATTN_SEQ_MAX * ATTN_SEQ_MAX;
  ctx->a = s; s += ATTN_SEQ_MAX;
  ctx->p = s;
  return ctx;
}

/*
 * Encodes one river token stream into z[64].
 *
 * `tokens` is 4*ntok packed int8 — [type, seatRel, idx, flags] per token, the
 * "seq" d-line field decoded — and `out_z` receives 64 floats. ntok is clamped
 * into [0, 96]; a negative count is treated as 0 and anything past token 96 is
 * ignored (the spec truncates each river to 24 entries * 4 seats = 96, so a
 * longer stream is a caller bug, not a shape this file can represent).
 *
 * ntok == 0 is the spec's special case: p = 0, so z = bz, copied verbatim.
 *
 * Field values outside their documented ranges (type 0-33, seatRel 0-3, idx
 * 0-23) set NO one-hot bit rather than writing outside x — a corrupt stream
 * degrades the encoding instead of corrupting memory. flags bits above bit2
 * are ignored.
 *
 * NOT re-entrant: the scratch matrices live in the context.
 */
void rlnet_attn_encode(const rlnet_attn *ctx, const int8_t *tokens, int ntok, float *out_z) {
  if (ctx == NULL || out_z == NULL) return;
  if (ntok < 0 || tokens == NULL) ntok = 0;
  if (ntok > ATTN_SEQ_MAX) ntok = ATTN_SEQ_MAX;

  /* Step 6 with p = 0: z = bz. Also the ntok<0 / NULL-tokens fallback. */
  memcpy(out_z, ctx->bz, ATTN_D * sizeof(float));
  if (ntok == 0) return;

  const int L = ntok;

  /* Dense 42-dim expansion:
     [0..33] onehot34(type) [34..37] onehot4(seatRel) [38] idx/24
     [39] tsumogiri [40] riichi declaration [41] called away.
     idx/24 is computed in double and rounded once to float, which is what a
     TS `Float32Array[i] = idx / 24` store does. */
  memset(ctx->x, 0, (size_t)L * ATTN_X * sizeof(float));
  for (int i = 0; i < L; i++) {
    float *xi = ctx->x + (size_t)i * ATTN_X;
    const int type = tokens[(size_t)i * 4 + 0];
    const int seat = tokens[(size_t)i * 4 + 1];
    const int idx = tokens[(size_t)i * 4 + 2];
    const int flags = tokens[(size_t)i * 4 + 3];
    if (type >= 0 && type < 34) xi[type] = 1.0f;
    if (seat >= 0 && seat < 4) xi[34 + seat] = 1.0f;
    if (idx >= 0 && idx < 24) xi[38] = (float)((double)idx / 24.0);
    if (flags & 1) xi[39] = 1.0f;
    if (flags & 2) xi[40] = 1.0f;
    if (flags & 4) xi[41] = 1.0f;
  }

  /* 1. h = relu(W_in x + b_in) */
  attn_affine(ctx->h, ctx->x, ATTN_X, ctx->w_in, ctx->b_in, L);
  for (size_t i = 0; i < (size_t)L * ATTN_D; i++) {
    if (ctx->h[i] < 0.0f) ctx->h[i] = 0.0f;
  }

  /* 2. q, k, v */
  attn_affine(ctx->q, ctx->h, ATTN_D, ctx->wq, ctx->bq, L);
  attn_affine(ctx->k, ctx->h, ATTN_D, ctx->wk, ctx->bk, L);
  attn_affine(ctx->v, ctx->h, ATTN_D, ctx->wv, ctx->bv, L);

  /* 3. per head: O_h = softmax(Q_h K_h^T / 4) V_h. Heads are the contiguous
     16-column slices of the 64-wide q/k/v (the reshape(L,H,16) convention);
     each is a [L][16] submatrix with row stride 64, which sgemm takes as
     lda = 64 — no repacking. Full bidirectional attention, no mask. */
  for (int hd = 0; hd < ATTN_H; hd++) {
    const int off = hd * ATTN_DH;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, L, L, ATTN_DH, ATTN_SCALE,
                ctx->q + off, ATTN_D, ctx->k + off, ATTN_D, 0.0f, ctx->s, L);
    for (int i = 0; i < L; i++) attn_softmax(ctx->s + (size_t)i * L, L);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, L, ATTN_DH, L, 1.0f, ctx->s, L,
                ctx->v + off, ATTN_D, 0.0f, ctx->o + off, ATTN_D);
  }

  /* 4. m = Wo O + bo */
  attn_affine(ctx->m, ctx->o, ATTN_D, ctx->wo, ctx->bo, L);

  /* 5. learned-query pooling: a = softmax_i(u . m_i); p = sum_i a_i m_i.
     The second gemv is M^T a, i.e. the same [L][64] matrix transposed. */
  cblas_sgemv(CblasRowMajor, CblasNoTrans, L, ATTN_D, 1.0f, ctx->m, ATTN_D, ctx->u, 1, 0.0f,
              ctx->a, 1);
  attn_softmax(ctx->a, L);
  cblas_sgemv(CblasRowMajor, CblasTrans, L, ATTN_D, 1.0f, ctx->m, ATTN_D, ctx->a, 1, 0.0f,
              ctx->p, 1);

  /* 6. z = Wz p + bz (out_z already holds bz). */
  cblas_sgemv(CblasRowMajor, CblasNoTrans, ATTN_D, ATTN_D, 1.0f, ctx->wz, ATTN_D, ctx->p, 1,
              1.0f, out_z, 1);
}

/* Frees an encoder. NULL is a no-op; the pointer must not be used after. */
void rlnet_attn_destroy(rlnet_attn *ctx) {
  if (ctx == NULL) return;
  free(ctx->params);
  free(ctx->scratch);
  free(ctx);
}
