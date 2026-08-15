# native/ — optional Accelerate inference shim

`rlnet.c` is a C library with two independent halves, both over Apple's Accelerate BLAS:

- a **generic dense MLP** (`rlnet_create` / `rlnet_forward` / `rlnet_destroy`) over `cblas_sgemv` —
  dimensions, activations and one flat float32 blob, nothing else;
- the **v4 attention river encoder** (`rlnet_attn_create` / `rlnet_attn_encode` /
  `rlnet_attn_destroy`) over `cblas_sgemm`, implementing `train/V4_SPEC.md` § "Encoder forward".

Both exist only to make inference faster; **nothing depends on them**. With the dylib absent, FFI
denied, or the gate turned off, `src/rl/net.ts` runs its pure-TypeScript loop and produces the same
numbers (modulo float32 summation order, which BLAS reorders — differences are at the 1e-6 level;
measured against an independent float64 reference the encoder's `z` agrees to ~2e-6 absolute on
`|z| ~ 13`, i.e. float32 epsilon, far inside the spec's 1e-4 tolerance).

There is no mahjong in the MLP half. The encoder half knows exactly one mahjong-shaped thing — the
42-dim dense expansion of a river token that the spec fixes — and no rules; it returns `z[64]` and
the TypeScript side is what splices it into the 1738-wide policy input.

## Build

```sh
deno task build-native
# = clang -O3 -dynamiclib -framework Accelerate -o native/librlnet.dylib native/rlnet.c
```

Warning-free under `-Wall -Wextra`. `ACCELERATE_NEW_LAPACK` is defined inside the `.c` so the plain
command above needs no extra flags (the legacy CBLAS headers are deprecated as of macOS 13.3).

macOS only in practice — Accelerate is an OS framework. `net.ts` looks for `librlnet.so` / `.dll` on
other platforms, so a port only needs a BLAS.

## ABI — MLP (v3, unchanged)

```c
int64_t rlnet_create(int32_t n_layers,
                     const int32_t *dims,  /* n_layers+1: in0, out0(=in1), …, outN */
                     const uint8_t *acts,  /* n_layers: 1 = relu, 0 = identity     */
                     const float   *blob); /* see below; copied                    */
void    rlnet_forward(int64_t ctx, const float *in, float *out);
void    rlnet_destroy(int64_t ctx);
```

`blob` is the `policy.f32` layout verbatim: per layer, in order, the row-major `[out][in]` weight
matrix followed by the `[out]` bias, little-endian float32, no header and no padding. Row-major
`[out][in]` is what `CblasRowMajor`/`CblasNoTrans` with `lda = in` wants, so nothing is transposed
at load time. Everything is copied into the context, which owns two scratch buffers of the widest
layer's size; the bias is `memcpy`'d into the destination and `sgemv` accumulates onto it with
`beta = 1`.

`rlnet_create` returns 0 on bad arguments or a failed allocation.

These three are **untouched by v4** and stay the whole story for v3 manifests. The v4 policy forward
is also just this MLP, over a 1738-wide input that `net.ts` assembles: **`z` is concatenated in
TypeScript, never in C.**

## ABI — attention encoder (v4)

```c
typedef struct rlnet_attn rlnet_attn;              /* opaque */

rlnet_attn *rlnet_attn_create(const char *attn_f32_path);
void        rlnet_attn_encode(const rlnet_attn *ctx,
                              const int8_t *tokens,  /* 4 bytes/token, packed  */
                              int ntok,              /* [0, 96]                */
                              float *out_z);         /* 64 floats written      */
void        rlnet_attn_destroy(rlnet_attn *ctx);
```

Note the handle here is a real pointer, not the `int64_t` the MLP half uses — that is what
`V4_SPEC.md` § "Native C ABI" specifies (Deno FFI: `"pointer"` in and out, `null` return = failure).

`tokens` is the decoded `seq` d-line field: `[type, seatRel, idx, flags]` per token, `flags` bit0 =
tsumogiri, bit1 = riichi declaration, bit2 = called away. `rlnet_attn_encode` expands each token to
the spec's 42 dims itself. Shapes are fixed by the spec, not parameters: D = 64, 4 heads of 16,
softmax scale 1/4, `SEQ_MAX` 96.

Boundary behaviour, all of it defined here rather than left to the caller:

| input                              | behaviour                                              |
| ---------------------------------- | ------------------------------------------------------ |
| `ntok == 0`                        | `z = bz` verbatim (the spec's `p = 0` case), bit-exact  |
| `ntok < 0`, or `tokens == NULL`    | treated as `ntok == 0`                                  |
| `ntok > 96`                        | clamped to 96; no byte past token 96 is read            |
| `type`/`seatRel`/`idx` out of range | that one-hot is simply not set — never an OOB write     |
| `ctx == NULL` or `out_z == NULL`   | silent no-op, like `rlnet_forward`                      |

### `attn.f32`

Tensor order and shapes live in **`train/V4_SPEC.md` § "Weight file attn.f32"** — that file is the
contract; this one deliberately does not restate the layout. Same convention as `policy.f32`
(row-major `[out][in]` then bias, little-endian float32, no header/padding).

> **Spec defect, unresolved:** V4_SPEC.md's tensor list sums to **23,616** floats
> (`64*42+64 + 4*(64*64+64) + 64 + 64*64+64` = 2752 + 16640 + 64 + 4160), but the line under it
> states **23,872** — a 256-float (4×64) arithmetic slip in the prose. This loader implements the
> **tensor list** (23,616 floats = 94,464 bytes) and **rejects any other file size** with `NULL`
> rather than reading a prefix, so a writer that believes the 23,872 figure fails loudly at load
> instead of silently misaligning every tensor. If the spec is corrected the other way, `ATTN_N_FLOATS`
> in `rlnet.c` is the single place to change.

### Numerics

Softmax is hand-rolled, max-subtracted, `expf` + normalize — both in attention rows and in the
learned-query pooling. Because the max element always contributes `expf(0) = 1`, the denominator is
≥ 1 and no logit magnitude can produce `inf`/`NaN` (the test harness pushes logits to 1000, where a
naive `expf` overflows). `expf` vs TypeScript's `Math.exp` differ by well under the spec's 1e-4
forward tolerance; the softmax output is a normalized ratio, which damps the difference further.

Heads are the **contiguous** 16-column slices of the 64-wide q/k/v (the `reshape(L, H, 16)`
convention), read in place as submatrices with `lda = 64` — no repacking, and the same convention
the MLX trainer's `reshape` produces.

## Rules of use

- **Not re-entrant**, both halves: a context holds scratch state. One caller at a time; one context
  per network instance (`net.ts` creates one per loaded `Net`).
- `in` and `out` must not overlap, and `out` must be `dims[n_layers]` long. Likewise `out_z` must be
  64 floats and must not alias `tokens`.
- The MLP handle is a pointer in disguise, the encoder handle is one outright; never call
  `rlnet_forward` / `rlnet_attn_encode` after the matching destroy.
- An `rlnet_attn` allocates ~290 KB up front (94 KB weights + ~196 KB scratch) and **allocates
  nothing per call** — `x`, `h`, `q`, `k`, `v`, `o`, `m`, the `[96][96]` score matrix and the pooling
  vectors all live in the context.

## Test harness

`attn_test.c` is a standalone self-check for the encoder — its own `main`, deliberately **not** wired
into `deno.json` and not built by `deno task build-native`:

```sh
clang -O1 -g -fsanitize=address -Wall -Wextra -framework Accelerate \
      -o native/attn_test native/rlnet.c native/attn_test.c
./native/attn_test          # exit 0 = pass; prints "N checks, 0 failures"
```

It declares the ABI itself, the way the Deno FFI glue does, so it also fails if a signature drifts.
Five tests: (1) all-identity weights at L=1, where the encoder collapses to the dense expansion, so
`z` is readable by hand; (2) `ntok = 0` / `NULL` / negative against `bz`, `memcmp`-exact; (3) a
hand-derived 2-token case with one head made dominant by a query bias of 4000 (logits 1000 vs 0,
which is `NaN` without max-subtraction) plus a learned query that selects the second token, expected
`z = ½(x₀+x₁)`; (4) full-size random weights swept over L = 1…96, plus over-long counts and
out-of-range fields, judged by ASan, finiteness and determinism; (5) load rejection of a missing,
short, and 23,872-float file. Scratch weight files are written under `$TMPDIR`.

The harness was mutation-checked: dropping the max-subtraction, striding the heads wrong, dividing
`idx` by 23, and replacing the pooling softmax with a uniform mean are each caught.

## Using it from TypeScript

`src/rl/net.ts` handles all of this: `loadNet` calls `loadNative`, which `Deno.dlopen`s
`native/librlnet.dylib` **resolved relative to the module** (`src/rl/net.ts` → `../../native/`), so
the working directory is irrelevant.

Gate — the `MJGAME_NATIVE` environment variable:

| value | behaviour                                                     |
| ----- | ------------------------------------------------------------- |
| `0`   | force the TypeScript path even with the dylib built           |
| `1`   | require native; a missing dylib throws, naming the build line |
| unset | try native, fall back silently                                |

Needs `--allow-ffi` (and `--allow-env=MJGAME_NATIVE` to read the gate); both are on the `play` /
`selfplay` / `bench` / `test` tasks. Contexts live for the process; `closeNet(net)` releases one
early.
