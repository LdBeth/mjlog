/* attn_test — standalone self-check for the v4 attention encoder in rlnet.c.
 *
 * NOT part of the library and NOT wired into deno.json. It links rlnet.c and
 * calls the three v4 entry points exactly as an external consumer (the Deno
 * FFI glue) sees them: an opaque `rlnet_attn *`, packed int8 tokens, float
 * out_z[64]. Build/run (ASan on, which is the point of it):
 *
 *   clang -O1 -g -fsanitize=address -Wall -Wextra -framework Accelerate \
 *         -o native/attn_test native/rlnet.c native/attn_test.c
 *   ./native/attn_test
 *
 * The three small tests use weights chosen so the whole forward collapses to
 * arithmetic doable on paper (see the comment above each). The fourth is a
 * full-size random-weight run whose only judge is ASan plus a finiteness /
 * determinism check.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- the ABI under test, declared the way a foreign consumer declares it -- */
typedef struct rlnet_attn rlnet_attn;
rlnet_attn *rlnet_attn_create(const char *attn_f32_path);
void rlnet_attn_encode(const rlnet_attn *ctx, const int8_t *tokens, int ntok, float *out_z);
void rlnet_attn_destroy(rlnet_attn *ctx);

/* ---- spec shape ---------------------------------------------------------- */
#define D 64
#define XD 42
#define NF (D * XD + D + 4 * (D * D + D) + D + D * D + D) /* 23616 */

static int failures = 0;
static int checks = 0;

/* A view over one attn.f32 image, in the spec's tensor order. */
typedef struct {
  float *buf;
  float *w_in, *b_in, *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo, *u, *wz, *bz;
} weights;

static weights *w_alloc(void) {
  weights *w = (weights *)calloc(1, sizeof(weights));
  w->buf = (float *)calloc(NF, sizeof(float));
  float *p = w->buf;
  w->w_in = p; p += D * XD;  w->b_in = p; p += D;
  w->wq   = p; p += D * D;   w->bq   = p; p += D;
  w->wk   = p; p += D * D;   w->bk   = p; p += D;
  w->wv   = p; p += D * D;   w->bv   = p; p += D;
  w->wo   = p; p += D * D;   w->bo   = p; p += D;
  w->u    = p; p += D;
  w->wz   = p; p += D * D;   w->bz   = p; p += D;
  if (p != w->buf + NF) {
    fprintf(stderr, "FATAL: layout walk ended at %ld, expected %d\n",
            (long)(p - w->buf), NF);
    exit(2);
  }
  return w;
}

static void w_free(weights *w) {
  free(w->buf);
  free(w);
}

/* m[rows][cols], 1.0 on the main diagonal. For the 64x42 W_in this is the
   "identity-ish" case: h = relu(x) padded with 22 zeros. */
static void w_eye(float *m, int rows, int cols) {
  memset(m, 0, (size_t)rows * cols * sizeof(float));
  for (int i = 0; i < rows && i < cols; i++) m[i * cols + i] = 1.0f;
}

static const char *w_write(const weights *w, const char *name) {
  static char path[512];
  const char *dir = getenv("TMPDIR");
  snprintf(path, sizeof(path), "%s%s", dir && *dir ? dir : "/tmp/", name);
  FILE *f = fopen(path, "wb");
  if (f == NULL) {
    perror("fopen");
    exit(2);
  }
  if (fwrite(w->buf, sizeof(float), NF, f) != (size_t)NF) {
    perror("fwrite");
    exit(2);
  }
  fclose(f);
  return path;
}

static void expect(const char *what, float got, float want) {
  checks++;
  if (!(fabsf(got - want) <= 1e-6f)) {
    printf("  FAIL %-34s got %.9g want %.9g\n", what, (double)got, (double)want);
    failures++;
  }
}

/* Asserts z equals `want` on the listed indices and is 0 everywhere else. */
static void expect_sparse(const float *z, const int *idx, const float *val, int n) {
  for (int i = 0; i < D; i++) {
    float want = 0.0f;
    for (int j = 0; j < n; j++) {
      if (idx[j] == i) want = val[j];
    }
    char label[64];
    snprintf(label, sizeof(label), "z[%d]", i);
    expect(label, z[i], want);
  }
}

/* --------------------------------------------------------------------------
 * 1. Pass-through: every matrix is the identity, every bias 0, L = 1.
 *    h = relu(x) = x (x >= 0 always); with L=1 the attention row is
 *    softmax(single logit) = 1 whatever the logit, so O = v = x; m = x;
 *    pooling over one token is 1; z = x. The test therefore reads out the
 *    dense 42-dim expansion itself.
 *
 *    token [type=5, seatRel=2, idx=3, flags=0b101]
 *      -> onehot34: x[5]=1 | onehot4: x[34+2]=1 | x[38]=3/24=0.125
 *         x[39]=1 (bit0 tsumogiri) x[40]=0 (bit1 clear) x[41]=1 (bit2 called)
 * -------------------------------------------------------------------------- */
static void test_passthrough_single(void) {
  printf("test 1: identity weights, L=1 -> z == dense(x)\n");
  weights *w = w_alloc();
  w_eye(w->w_in, D, XD);
  w_eye(w->wq, D, D);
  w_eye(w->wk, D, D);
  w_eye(w->wv, D, D);
  w_eye(w->wo, D, D);
  w_eye(w->wz, D, D);
  w->u[0] = 1.0f; /* irrelevant at L=1 */
  const char *path = w_write(w, "rlnet_attn_t1.f32");

  rlnet_attn *ctx = rlnet_attn_create(path);
  if (ctx == NULL) {
    printf("  FAIL create returned NULL\n");
    failures++;
    w_free(w);
    return;
  }
  const int8_t tok[4] = {5, 2, 3, 5};
  float z[D];
  memset(z, 0x7f, sizeof(z));
  rlnet_attn_encode(ctx, tok, 1, z);

  const int idx[] = {5, 36, 38, 39, 41};
  const float val[] = {1.0f, 1.0f, 0.125f, 1.0f, 1.0f};
  expect_sparse(z, idx, val, 5);
  rlnet_attn_destroy(ctx);
  w_free(w);
}

/* --------------------------------------------------------------------------
 * 2. L = 0 -> z = bz, bit-exact. Also the defensive paths (NULL tokens,
 *    negative ntok) must land on the same answer, and a NULL ctx or NULL
 *    out_z must not write or crash.
 * -------------------------------------------------------------------------- */
static void test_empty_sequence(void) {
  printf("test 2: L=0 -> z == bz (bit-exact)\n");
  weights *w = w_alloc();
  for (int i = 0; i < NF; i++) w->buf[i] = (float)((i % 37) - 18) * 0.03125f;
  for (int i = 0; i < D; i++) w->bz[i] = (float)i * 0.5f - 3.0f;
  const char *path = w_write(w, "rlnet_attn_t2.f32");

  rlnet_attn *ctx = rlnet_attn_create(path);
  if (ctx == NULL) {
    printf("  FAIL create returned NULL\n");
    failures++;
    w_free(w);
    return;
  }
  float z[D];
  memset(z, 0x7f, sizeof(z));
  rlnet_attn_encode(ctx, NULL, 0, z);
  checks++;
  if (memcmp(z, w->bz, sizeof(z)) != 0) {
    printf("  FAIL ntok=0 output is not bit-identical to bz\n");
    failures++;
  }

  const int8_t tok[4] = {0, 0, 0, 0};
  memset(z, 0x7f, sizeof(z));
  rlnet_attn_encode(ctx, NULL, 5, z); /* NULL stream, positive count */
  checks++;
  if (memcmp(z, w->bz, sizeof(z)) != 0) {
    printf("  FAIL NULL tokens did not fall back to bz\n");
    failures++;
  }
  memset(z, 0x7f, sizeof(z));
  rlnet_attn_encode(ctx, tok, -3, z); /* negative count */
  checks++;
  if (memcmp(z, w->bz, sizeof(z)) != 0) {
    printf("  FAIL negative ntok did not fall back to bz\n");
    failures++;
  }
  rlnet_attn_encode(NULL, tok, 1, z); /* must be a silent no-op */
  rlnet_attn_encode(ctx, tok, 1, NULL);
  checks++;
  if (memcmp(z, w->bz, sizeof(z)) != 0) {
    printf("  FAIL NULL ctx/out_z was not a no-op\n");
    failures++;
  }
  rlnet_attn_destroy(ctx);
  rlnet_attn_destroy(NULL);
  w_free(w);
}

/* --------------------------------------------------------------------------
 * 3. Head split + softmax overflow + learned-query pooling, hand-derived.
 *
 *    W_in = eye, Wk = Wv = Wo = Wz = eye, all biases 0, so h = k = v = x.
 *    Wq is zero except row 0 = G*e_34, i.e. q_i[0] = G * x_i[34] and q_i[d]=0
 *    for d >= 1. Head 0 covers dims 0..15, so its logits are
 *        s_ij = 0.25 * q_i[0] * k_j[0] = 0.25 * G * x_i[34] * x_j[0].
 *    Heads 1..3 see q = 0 -> all logits 0 -> uniform 1/L rows.
 *
 *    tokens: t0 = [type 0, seat 0, idx 0, flags 0] -> x0[0]=1, x0[34]=1
 *            t1 = [type 1, seat 1, idx 12, flags 3]
 *                 -> x1[1]=1, x1[35]=1, x1[38]=12/24=0.5, x1[39]=1, x1[40]=1
 *
 *    G = 4000 so row 0's head-0 logits are (1000, 0): expf(1000) is +inf, so
 *    an implementation without max-subtraction yields inf/inf = NaN here.
 *    With it: A_0 = (1, 0) exactly (expf(-1000) underflows to 0).
 *      head0: O_0[0..15] = v_0[0..15] = e_0
 *             O_1[0..15] = mean(v)[0..15]  (q_1[0] = G*x1[34] = 0 -> uniform)
 *      heads1-3: both rows = mean(v)
 *    so m_0 = e_0 on dims 0..15 and mean elsewhere, m_1 = mean(x0,x1) whole.
 *
 *    u = P*e_1, P = 2000: u.m_0 = P*m_0[1] = 0, u.m_1 = P*0.5 = 1000. Pooling
 *    softmax(0, 1000) = (0, 1) -> p = m_1, and again NaN without max-sub.
 *
 *    z = m_1 = 0.5*(x0 + x1):
 *        z[0]=0.5 z[1]=0.5 z[34]=0.5 z[35]=0.5 z[38]=0.25 z[39]=0.5 z[40]=0.5
 * -------------------------------------------------------------------------- */
static void test_heads_and_pooling(void) {
  printf("test 3: 4-head split, overflow-safe softmax, learned-query pooling\n");
  weights *w = w_alloc();
  w_eye(w->w_in, D, XD);
  w_eye(w->wk, D, D);
  w_eye(w->wv, D, D);
  w_eye(w->wo, D, D);
  w_eye(w->wz, D, D);
  w->wq[0 * D + 34] = 4000.0f; /* q_i[0] = 4000 * x_i[34] */
  w->u[1] = 2000.0f;
  const char *path = w_write(w, "rlnet_attn_t3.f32");

  rlnet_attn *ctx = rlnet_attn_create(path);
  if (ctx == NULL) {
    printf("  FAIL create returned NULL\n");
    failures++;
    w_free(w);
    return;
  }
  const int8_t tok[8] = {0, 0, 0, 0, 1, 1, 12, 3};
  float z[D];
  memset(z, 0x7f, sizeof(z));
  rlnet_attn_encode(ctx, tok, 2, z);

  const int idx[] = {0, 1, 34, 35, 38, 39, 40};
  const float val[] = {0.5f, 0.5f, 0.5f, 0.5f, 0.25f, 0.5f, 0.5f};
  expect_sparse(z, idx, val, 7);
  rlnet_attn_destroy(ctx);
  w_free(w);
}

/* --------------------------------------------------------------------------
 * 4. Full-size random weights: L = 96 (SEQ_MAX), L = 1, an over-long count
 *    that must clamp, and a stream of out-of-range fields. Checked for
 *    finiteness and determinism; the real judge is AddressSanitizer.
 * -------------------------------------------------------------------------- */
static uint32_t rng_state = 12345u;
static float rnd(void) {
  rng_state = rng_state * 1664525u + 1013904223u;
  return ((float)(rng_state >> 8) / 8388608.0f - 1.0f) * 0.5f; /* ~U(-.5,.5) */
}

static void check_finite(const char *what, const float *z) {
  checks++;
  for (int i = 0; i < D; i++) {
    if (!isfinite(z[i])) {
      printf("  FAIL %s: z[%d] = %g is not finite\n", what, i, (double)z[i]);
      failures++;
      return;
    }
  }
}

static void test_full_size_random(void) {
  printf("test 4: full-size random weights, L=96 / clamping / junk fields\n");
  weights *w = w_alloc();
  for (int i = 0; i < NF; i++) w->buf[i] = rnd();
  const char *path = w_write(w, "rlnet_attn_t4.f32");

  rlnet_attn *ctx = rlnet_attn_create(path);
  if (ctx == NULL) {
    printf("  FAIL create returned NULL\n");
    failures++;
    w_free(w);
    return;
  }
  int8_t tok[96 * 4];
  for (int i = 0; i < 96; i++) {
    tok[i * 4 + 0] = (int8_t)(i % 34);
    tok[i * 4 + 1] = (int8_t)(i / 24);
    tok[i * 4 + 2] = (int8_t)(i % 24);
    tok[i * 4 + 3] = (int8_t)(i % 8);
  }
  float z[D], z2[D];
  rlnet_attn_encode(ctx, tok, 96, z);
  check_finite("L=96", z);
  rlnet_attn_encode(ctx, tok, 96, z2);
  checks++;
  if (memcmp(z, z2, sizeof(z)) != 0) {
    printf("  FAIL L=96 is not deterministic across calls\n");
    failures++;
  }
  for (int n = 1; n <= 96; n++) {
    float zn[D];
    rlnet_attn_encode(ctx, tok, n, zn);
    check_finite("sweep", zn);
  }
  /* Over-long count: must clamp to 96 and read no further than 96 tokens,
     which is what ASan checks (tok holds exactly 96). */
  rlnet_attn_encode(ctx, tok, 4096, z2);
  checks++;
  if (memcmp(z, z2, sizeof(z)) != 0) {
    printf("  FAIL ntok > SEQ_MAX did not clamp to the L=96 result\n");
    failures++;
  }
  /* Out-of-range fields must not index outside the 42-dim row. */
  int8_t junk[8 * 4];
  for (int i = 0; i < 8; i++) {
    junk[i * 4 + 0] = (int8_t)(i % 2 ? 120 : -120);
    junk[i * 4 + 1] = (int8_t)(i % 2 ? 99 : -99);
    junk[i * 4 + 2] = (int8_t)(i % 2 ? 127 : -128);
    junk[i * 4 + 3] = (int8_t)-1;
  }
  rlnet_attn_encode(ctx, junk, 8, z);
  check_finite("junk fields", z);
  rlnet_attn_destroy(ctx);
  w_free(w);
}

/* 5. Load-time rejection: bad path, short file, long file. */
static void test_bad_files(void) {
  printf("test 5: attn.f32 load rejection\n");
  checks++;
  if (rlnet_attn_create("/nonexistent/attn.f32") != NULL) {
    printf("  FAIL missing file did not return NULL\n");
    failures++;
  }
  checks++;
  if (rlnet_attn_create(NULL) != NULL) {
    printf("  FAIL NULL path did not return NULL\n");
    failures++;
  }
  weights *w = w_alloc();
  const char *path = w_write(w, "rlnet_attn_t5.f32");
  char cmd[600];
  FILE *f = fopen(path, "ab");
  fwrite(w->buf, sizeof(float), 256, f); /* the 23,872 misprint's size */
  fclose(f);
  checks++;
  if (rlnet_attn_create(path) != NULL) {
    printf("  FAIL oversized file (23872 floats) was accepted\n");
    failures++;
  }
  snprintf(cmd, sizeof(cmd), "%s.short", path);
  f = fopen(cmd, "wb");
  fwrite(w->buf, sizeof(float), NF - 1, f);
  fclose(f);
  checks++;
  if (rlnet_attn_create(cmd) != NULL) {
    printf("  FAIL short file was accepted\n");
    failures++;
  }
  w_free(w);
}

int main(void) {
  printf("attn.f32 float count = %d\n\n", NF);
  test_passthrough_single();
  test_empty_sequence();
  test_heads_and_pooling();
  test_full_size_random();
  test_bad_files();
  printf("\n%d checks, %d failures\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
