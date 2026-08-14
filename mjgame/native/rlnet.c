// rlnet — a generic dense-MLP forward pass over Accelerate's cblas_sgemv.
//
// There is NO mahjong in this file, and deliberately so: it takes a layer
// chain (dims + activations) and one flat float32 blob laid out exactly like
// mjgame's `policy.f32` (per layer: the row-major [out][in] weight matrix, then
// the [out] bias), copies what it needs into a malloc'd context, and evaluates
// it. The TypeScript side (`src/rl/net.ts`) is what knows the blob came from a
// policy network; here it is just numbers.
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
#include <stdint.h>
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
