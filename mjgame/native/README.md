# native/ — optional accelerators

Three unrelated dylibs live here, built separately and loaded separately:

| source        | dylib               | build                    | what it accelerates                 |
| ------------- | ------------------- | ------------------------ | ----------------------------------- |
| `rlnet.c`     | `librlnet.dylib`    | `deno task build-native` | neural inference (Accelerate)       |
| `mjkernel.cc` | `libmjkernel.dylib` | `deno task build-kernel` | shanten / ukeire / 待ち形 (C++17)   |
| `mlp.c`       | `libmjmlp.dylib`    | `deno task build-mlp`    | the 計算 seat's small learned heads |

They share the `MJGAME_NATIVE` gate and nothing else — no symbols, no headers, no build flags.
Either can be absent; the TypeScript path behind each is the reference implementation. The kernel
and the head shim are documented at the bottom of this file; everything up to there is about
`rlnet.c`.

## rlnet.c

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

| input                               | behaviour                                              |
| ----------------------------------- | ------------------------------------------------------ |
| `ntok == 0`                         | `z = bz` verbatim (the spec's `p = 0` case), bit-exact |
| `ntok < 0`, or `tokens == NULL`     | treated as `ntok == 0`                                 |
| `ntok > 96`                         | clamped to 96; no byte past token 96 is read           |
| `type`/`seatRel`/`idx` out of range | that one-hot is simply not set — never an OOB write    |
| `ctx == NULL` or `out_z == NULL`    | silent no-op, like `rlnet_forward`                     |

### `attn.f32`

Tensor order and shapes live in **`train/V4_SPEC.md` § "Weight file attn.f32"** — that file is the
contract; this one deliberately does not restate the layout. Same convention as `policy.f32`
(row-major `[out][in]` then bias, little-endian float32, no header/padding).

> **Size:** the tensor list sums to **23,616** floats (`64*42+64 + 4*(64*64+64) + 64 + 64*64+64` =
> 2752 + 16640 + 64 + 4160) = 94,464 bytes, which is what this loader implements — and it **rejects
> any other file size** with `NULL` rather than reading a prefix, so a mis-sized blob fails loudly
> at load instead of silently misaligning every tensor. (An early revision of V4_SPEC.md's prose
> said 23,872, a 256-float slip; the spec now states 23,616 and says the tensor list was always
> authoritative. `ATTN_N_FLOATS` in `rlnet.c` is the single place the number lives on this side.)

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
  nothing per call** — `x`, `h`, `q`, `k`, `v`, `o`, `m`, the `[96][96]` score matrix and the
  pooling vectors all live in the context.

## Test harness

`attn_test.c` is a standalone self-check for the encoder — its own `main`, deliberately **not**
wired into `deno.json` and not built by `deno task build-native`:

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

---

# mjkernel.cc — the shanten / ukeire / 待ち形 kernel

`mjkernel.cc` is a self-contained C++17 library with no dependencies at all — no BLAS, no framework,
no data file. It answers the three questions that dominate self-play wall time:

```c
int32_t  mj_kernel_version(void);                     /* ABI, currently 2 */
int32_t  mj_shanten(const uint8_t counts[34], int32_t open_melds, int32_t closed);
uint64_t mj_ukeire_mask(const uint8_t counts[34], int32_t open_melds, int32_t closed, int32_t base);
void     mj_shape_masses(const int32_t unseen[34], const int32_t flags[34],
                         int32_t honitsu_suit, int32_t toitoi,
                         const double w[17], double out[306]);
```

The version is checked on every `dlopen`, so a dylib built before an entry point existed fails the
check and the whole module degrades to TypeScript rather than half-loading — `dlopen` throws on a
missing symbol, and there is no such thing as a partly-present ABI here.

It is a **semantic mirror of `mjrender/src/shanten.ts`**, not an independent shanten engine: the
same `cap < 0 → 8`, the same "chiitoitsu and kokushi only when `closed && open_melds == 0`", the
same `counts[t] >= 4` skip in ukeire. Results are _equal_, not _close_ — this is integer arithmetic,
so unlike the float32 inference shim there is no tolerance to negotiate.
`test/kernel_native_test.ts` is what holds that line.

`mj_ukeire_mask` returns bit `t` set ⇔ adding one tile of type `t` puts shanten below `base`; it
exists because the TypeScript ukeire probe costs 34 shanten evaluations, and folding them into one
FFI crossing is most of the win.

`mj_shape_masses` is the 計算 reader's wait model, and the same mirror discipline applies to it —
with one extra constraint, because it is the only **float** entry point here. See its own section
below.

## Build

```sh
deno task build-kernel
# = sh native/build_kernel.sh
# = clang++ -std=c++17 -O3 -flto -ffp-contract=off -Wall -Wextra -fvisibility=hidden \
#           -dynamiclib -o native/libmjkernel.dylib native/mjkernel.cc
```

Warning-free under `-Wall -Wextra`. Portable C++ — the build script picks `.so` off Darwin, and
nothing in the source is Apple-specific.

`-ffp-contract=off` is load-bearing rather than hygiene: without it clang may fuse `a + b*c` into an
FMA, which is a _different double_ from the multiply-then-add JavaScript performs, and
`mj_shape_masses` has to agree with the TypeScript to the last bit. `test/kernel_native_test.ts`
compiles with the same flag when it has to build the dylib itself; keep the two in step.

## Algorithm

The TypeScript computes standard-form shanten with one backtracking DFS that peels blocks (triplet /
sequence / pair-head / pair / ryanmen / kanchan) off the leftmost non-empty tile. Blocks never span
a suit boundary, so the decomposition space **factors** into four independent groups — man, pin,
sou, honours — and the objective

```
v = 2*min(M, cap) + min(P, cap - min(M, cap)) + H
```

is monotone non-decreasing in the totals each group contributes. That makes a tiny per-group summary
sufficient: for each (melds `m` ∈ 0..4, head `h` ∈ 0..1), only the group's **maximum** reachable
partial count matters. Ten numbers, packed as ten nibbles of a `uint64`.

So a shanten evaluation is four table reads, three 10×10 merges and a 10-way max. The tables are
indexed by the group's count distribution (base-5: below 5^9 for a suit, 5^7 for honours) and filled
**lazily by the very same block-peeling DFS restricted to one group** — so the table cannot disagree
with the search that defines it, no generated data file has to be shipped or trusted, and a process
pays only for the distributions it actually meets. A packed word of 0 means "not computed yet",
which is unambiguous because `(m=0, h=0, p=0)` is reachable from every distribution.

Clamping `m` and `p` at 4 is exact rather than approximate: `cap = 4 - open_melds` never exceeds 4
and both totals appear under a `min` with something ≤ `cap`.

`mj_ukeire_mask` sharpens this once more. Adding one tile perturbs exactly ONE group, so the merge
of the other three is computed once per group and reused across that group's 7–9 tiles; chiitoitsu
and kokushi are updated incrementally from the untouched hand's pair/kind counts. 34 probes cost ~34
merges instead of ~102.

A transliteration of the TypeScript DFS (`refStandard`) stays in the file as the fallback for inputs
outside the table's domain — a count above 4, a group holding more than 14 tiles, a negative
`open_melds`. Those cannot arise in play; they arise in fuzzing, and answering them exactly is
cheaper than arguing about them. The fuzz deliberately visits that branch.

## `mj_shape_masses` — one opponent's whole wait row

The 計算 reader asks, per decision and per opponent, _how strongly does public counting support
"they are waiting on this type"_ — for all 34 types. The answer is a **row**, not 34 verdicts: with
`waitNormalize` on it is divided by its own total, so no per-type call could answer anyway. That
fixes the cut point at **one crossing per (opponent, decision)**, three per decision.

```
unseen[34]   copies of each type not visible to the observer, 0..4
flags[34]    bit0 現物, bit1 役牌 for this opponent, bit2 ドラ type
honitsu_suit 染め手模様: 0 なし, 1 m, 2 p, 3 s   (a tile's own 字 suit is 0 too)
toitoi       トイトイ模様, 0/1
w[17]        the packed ComputedWeights slice (packShapeWeights in computed.ts)
out[306]     34 wait likelihoods, then 34 × 8 parameter-free counts (ShapeBase)
```

The counts come back alongside the row because the calibration recorder wants exactly them, and
because having one buffer means the trace and the live seat cannot be reading different numbers.

It is a transliteration of `shapeRowTS` in `src/ai/computed.ts`, which is itself the flat twin of
`shapeBaseMasses` + `waitRowFrom` + `combineShapes` there. Being **float**, the mirroring is finer
than the shanten half's:

- associativity is copied, not simplified — `prK * kanchan / 16` is `(prK * kanchan) / 16` while
  リャンメン is `prR * (mass / 32)`, and those are different doubles;
- no contraction (`-ffp-contract=off`, above);
- `Math.min(1, x)` is not `fmin(1.0, x)` — they disagree on NaN, and `jsMin1` settles it the
  JavaScript way.

`w[5]` arrives as `doraBridge − 1` rather than `doraBridge`, so the C performs no arithmetic on the
weights at all. `src/kernel.ts` refuses a context outside the integer domain (a count that is not
0..4) and takes the TypeScript instead, the same way it refuses a count vector that is not 34 long.

The fuzz in `test/kernel_native_test.ts` compares all 306 slots under 26 weight vectors — the
shipped one, normalized and not, plus randomized ones that put every multiplier to work — and
demands **bit equality**, not closeness. It was mutation-checked: perturbing the `/ 32` divisor by
1e-7 fails it.

## Measured

Apple M4 Pro, machine under mixed load, so read these as ratios rather than absolutes:

| call                        | native  | TypeScript (cold) | TypeScript (warm memo) |
| --------------------------- | ------- | ----------------- | ---------------------- |
| `mj_shanten`                | 0.20 µs | 13.8 µs           | 0.90 µs                |
| `mj_ukeire_mask` (34 types) | 2.59 µs | 678 µs            | 24.4 µs                |

The TypeScript memo makes repeat shapes cheap, so real play sits between the two right-hand columns.

`mj_shape_masses` has no memo to compete with, and its TypeScript twin is a flat typed-array loop
rather than the object-allocating definition, so the margin is much narrower — the crossing costs
nearly as much as the arithmetic:

| one opponent's row                                 | µs/row |
| -------------------------------------------------- | ------ |
| `shapeBaseMasses` + `waitRowFrom` (the definition) | 7.41   |
| `shapeRowTS` (flat, the fallback)                  | 1.12   |
| `mj_shape_masses`                                  | 0.82   |

Most of the win is the flattening, which the no-FFI path gets too; the kernel takes another ~26% of
what is left. On a `kkkk` bench (120 半荘) that is 5.81 s → 4.06 s → 3.88 s.

## Rules of use

- **Not re-entrant and not thread-safe**: the lazy tables are written without a lock. One caller at
  a time, which is exactly how Deno FFI uses it.
- `counts` must be exactly 34 bytes, and `mj_shape_masses`'s `unseen`/`flags`/`w`/`out` exactly
  34/34/17/306 elements. `src/kernel.ts` refuses anything else and falls back to TypeScript rather
  than let a short array be read past its end.
- The tables are ~16 MB of `calloc`, allocated on the first call and never freed — they are
  zero-filled and paged in on touch, so the resident cost is a few hundred KB in practice.

## Using it from TypeScript

`src/kernel.ts` is the only importer, and every mjgame call site goes through it instead of
importing `mjrender/shanten.ts` directly. It resolves the dylib **relative to the module** (`src/` →
`../native/`), so the working directory is irrelevant, and it holds one reusable 34-byte buffer for
the whole process rather than allocating per call. `src/ai/computed.ts` keeps its own pair of
process-wide `Int32Array(34)` scratches for `mj_shape_masses` on the same argument: a row evaluation
is a leaf, so a fresh pair per opponent would be pure garbage.

Same gate as `rlnet.c`:

| value | behaviour                                                     |
| ----- | ------------------------------------------------------------- |
| `0`   | force the TypeScript path even with the dylib built           |
| `1`   | require native; a missing dylib throws, naming the build line |
| unset | try native, fall back silently                                |

The gate is resolved on first use rather than at import, so a test can set the variable and call
`closeKernel()` to re-arm it. **mjrender never learns any of this exists**: the dependency points
one way, mjgame → mjrender, and the acceleration lives entirely on the mjgame side of the fence.

---

# mlp.c — the small learned heads

`mlp.c` is a self-contained C11 library with no dependencies at all — no BLAS, no framework, no data
file. It runs the forward pass of the 計算 seat's learned decision rules (M13's fold head, M14's
deal-in reads): dense nets of a few hundred to a few thousand weights that ship INLINE in a
`--ktune` JSON and are built by `src/ai/mlp.ts`.

```c
int32_t mjmlp_abi(void);                                  /* ABI, currently 1 */
int64_t mjmlp_create(int32_t n_layers,
                     const int32_t *dims,   /* n_layers+1: in0, out0(=in1), … */
                     const uint8_t *acts,   /* n_layers: 1 = relu, 0 = identity */
                     const float   *blob);  /* packMlp layout; copied         */
void    mjmlp_forward(int64_t h, const float *in, float *out);
void    mjmlp_forward_batch(int64_t h, int32_t n, const float *in, float *out);
void    mjmlp_destroy(int64_t h);
```

`blob` is `policy.f32`'s layout verbatim — per layer, in order, the row-major `[out][in]` weight
matrix then the `[out]` bias, little-endian float32, no header and no padding — which is what
`packMlp` writes and what `train/common.py`'s `export_mlp_block` describes in JSON. `mjmlp_create`
returns 0 on bad arguments or a failed allocation, and the version is checked on every `dlopen`, so
a stale dylib degrades the module to TypeScript rather than half-loading.

`mjmlp_forward_batch` is `n` rows of `dims[0]` inputs to `n` rows of `dims[n_layers]` outputs,
contiguous. It IS `n` calls of `mjmlp_forward`; it exists so that M14's 34 rows per opponent cost
one FFI crossing instead of 34.

## Why not `rlnet_create`

`rlnet.c` already has a generic dense MLP, and reusing it would have been free. It is over
Accelerate's `cblas_sgemv`, which **reorders the summation** — that is why `test/rl_native_test.ts`
grades that path with a 1e-4 tolerance. A tolerance is fine for a policy net whose 78 logits are
argmaxed; it is not fine for a gate whose SIGN decides a fold, because one flipped decision rewrites
a whole hanchan and the seat's decision fingerprints are pinned. So this is a separate dylib of
plain loops, graded at zero tolerance, and the only thing it buys is the removal of the interpreter
overhead.

## The floating-point contract

The reference is JavaScript (`src/ai/mlp.ts`, itself `src/rl/net.ts`'s loop), so the arithmetic is
JavaScript's, and the C mirrors it expression for expression:

1. the accumulator is a **double** seeded with the float32 bias (`Float32Array` reads widen
   exactly);
2. the summation is **sequential and ascending**, `o` outer and `i` inner;
3. relu is applied to the **double**, and the result is then stored **once as float32** — so doubles
   never cross a layer boundary and layer k+1 reads exactly the float32 layer k wrote. C's `(float)`
   cast is round-to-nearest-even, which is what a `Float32Array` store does.

## Build

```sh
deno task build-mlp
# = sh native/build_mlp.sh
# = clang -std=c11 -O3 -ffp-contract=off -Wall -Wextra -fvisibility=hidden \
#         -dynamiclib -o native/libmjmlp.dylib native/mlp.c
```

Warning-free under `-Wall -Wextra`. Portable C11 — the build script picks `.so` off Darwin.
`-ffp-contract=off` is load-bearing, not hygiene: without it clang may fuse `acc + w*x` into an FMA,
a different double from the multiply-then-add JavaScript performs. `test/mlp_native_test.ts`
compiles with the same flags when it has to build the dylib itself; keep the two in step.

## Test harness

`test/mlp_native_test.ts` fuzzes 300 random nets (1–3 layers, widths 1–64, relu and identity mixed,
four weight scales) × 10 inputs and demands **bit equality** on the float32 patterns — which also
settles −0 and NaN — plus `batch(n) == n` single calls and the gate's behaviour (`0` ⇒ TypeScript;
`1` with the dylib moved aside ⇒ a throw naming `deno task build-mlp`). It was mutation-checked:
replacing the double accumulator with a float one fails it at once. Reversing the inner loop does
**not** fail it — over a few dozen float32 products a reordering perturbs the double sum by ~1e-16
and the store to float32 rounds it away — so rule 2 above is defence in depth against the
reorderings a _library_ would perform (float32 partial sums, vector reductions), which do show.

`test/mlp_test.ts` is the other half: it forces the TypeScript path and checks a hand-computed
forward pass plus `test/fixtures/mlp-parity.json`, the fixture `train/mlp_selftest.py` writes from
`common.mlp_forward_np` — the numpy/Python mirror of the same three rules, written as an explicit
loop precisely because `np.dot` would reorder the sum.

## Rules of use

- **Not re-entrant**: a context owns two scratch buffers of the widest layer. One caller at a time,
  which is how Deno FFI uses it; one context per head (`buildMlp` creates one per `Mlp`).
- `in` and `out` must not overlap, and `out` must be `dims[n_layers]` long.
- The handle is a pointer in disguise; never call `mjmlp_forward` after `mjmlp_destroy` (`closeMlp`
  on the TypeScript side).
- `mjmlp_create` refuses more than 16 layers or a dimension above 65536 — those are corrupt
  arguments, not networks.

## Using it from TypeScript

`src/ai/mlp.ts` handles all of it: the dylib is resolved **relative to the module** (`src/ai/` →
`../../native/`), so the working directory is irrelevant, and the gate is the usual one:

| value | behaviour                                                     |
| ----- | ------------------------------------------------------------- |
| `0`   | force the TypeScript path even with the dylib built           |
| `1`   | require native; a missing dylib throws, naming the build line |
| unset | try native, fall back silently                                |

The gate is resolved on the first `buildMlp` rather than at import, so a test can set the variable
and call `closeMlpLib()` to re-arm it. Under `MJGAME_NATIVE=1` a missing dylib throws at
construction, not on the first decision.
