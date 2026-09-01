# native/ — optional accelerators

Four unrelated dylibs live here, built separately and loaded separately:

| source        | dylib               | build                    | what it does                        |
| ------------- | ------------------- | ------------------------ | ----------------------------------- |
| `rlnet.c`     | `librlnet.dylib`    | `deno task build-native` | neural inference (Accelerate)       |
| `mjkernel.cc` | `libmjkernel.dylib` | `deno task build-kernel` | shanten / ukeire / 待ち形 (C++17)   |
| `mlp.c`       | `libmjmlp.dylib`    | `deno task build-mlp`    | the 計算 seat's small learned heads |
| `mjev.cc`     | `libmjev.dylib`     | `deno task build-ev`     | the 計算 seat's期待値コア (C++17)   |

They share the `MJGAME_NATIVE` gate and nothing else — no symbols, no headers, no build flags.

The first three are **accelerators**: each has a TypeScript twin beside it that is the reference
implementation, so an absent dylib is a slowdown and nothing else. `mjev.cc` is **not** — it has no
TypeScript twin at all, so a seat that asks for it and cannot have it is a refusal. See its section
at the bottom. Everything up to the `mjkernel.cc` heading is about `rlnet.c`.

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

---

# mjev.cc — the 期待値 core (M15)

`libmjev` is the 計算 seat's expected-value engine: an exact integer scorer, a re-entrant
shanten/ukeire core, and the dynamic program that prices a discard, the riichi declaration after it,
and giving up.

**It is not an accelerator.** The other three dylibs here wrap a TypeScript reference; this one
replaces nothing, because the DP exists only in C++ (owner decision, 2026-08-30). A ktune carrying
an `ev` block therefore REQUIRES the dylib and `--allow-ffi`, and REFUSES `MJGAME_NATIVE=0` —
`src/ai/ev.ts` throws rather than quietly playing a different game. With no `ev` block the module is
never reached and no FFI is touched.

What keeps a twin-less library honest is that its sub-parts do have twins:

- `mjev_score` is a transliteration of `src/decompose.ts` + `src/fu.ts` + `src/yaku.ts` +
  `src/score.ts`, all integer, compared field by field against `scoreWin` on >100k random complete
  hands at zero tolerance;
- `mjev_shanten` / `mjev_ukeire_mask` are `mjkernel.cc`'s group-word algorithm, fuzzed against
  `src/kernel.ts` on 200k hands plus the kokushi and chiitoi batteries.

The DP has no twin at all, so it is graded on hands whose answer can be written down (a tenpai
wait's hypergeometric closed form, a two-stage 1向聴→聴牌→和了 chain), on invariants (monotone in
waits, turns, `gain`, `costIn`, Σ tenpaiP), on determinism (same bits twice, and across two
contexts), and — the one that matters — on whether it advances a hand. All in
`test/ev_native_test.ts`.

## Re-entrancy

`mjkernel.cc` keeps its two lazy tables in globals, which is why the self-play harness warms worker
0 before handing work out. `mjev` does not: the shanten memo is an open-addressing hash **inside the
context** (key = the group's base-5 index, honors shifted past the suits; value = the packed summary
word), grown by doubling from 16k slots. No globals hold state, no statics, no lock, no warm-up —
each worker dlopens its own image and each seat owns its own `mjev_create` handle.

## ABI

```c
int32_t  mjev_abi(void);                                   /* 1 */
int64_t  mjev_create(const double *params, int32_t n);     /* n must be 36; 0 on bad args */
void     mjev_destroy(int64_t ctx);
int32_t  mjev_score(const int32_t *in, int32_t *out);      /* stateless */
int32_t  mjev_shanten(int64_t ctx, const uint8_t *counts, int32_t openMelds, int32_t closed);
uint64_t mjev_ukeire_mask(int64_t ctx, const uint8_t *counts, int32_t openMelds,
                          int32_t closed, int32_t base);
int32_t  mjev_eval_discard(int64_t, const int32_t *ints, const double *dbls, double *out);
double   mjev_eval_rest(int64_t, const int32_t *ints, const double *dbls, double *meta);
```

Every offset in `in`/`out`/`ints`/`dbls`/`params` is fixed in **`src/ai/evlayout.ts`** and mirrored
by name in `mjev.cc` (`enum S_*`, `SO_*`, `I_*`, `D_*`, `O_*`, `R_*`, `P_*`). One contract, two
readers: changing an offset on one side alone is an ABI break, and `mjev_abi()` is what the wrapper
checks. No tile ids ever cross — hands are 34-type counts and melds are `{kind, type, concealed}`
triples.

`mjev_score` answers `[ok, han, fu, base, 役満数, limit, ronPayment, tsumoTotal]`; `ok = 0` means
役なし (or a shape that does not win), and then every other field is 0.

`mjev_eval_discard` returns 0 on success, `1` for a bad mode or a hand that is not `14 − 3·melds`
(`13 − 3·melds` for a rest root), `2` for a malformed hidden-information posterior, `3` for a null
context or buffer; `mjev_eval_rest` answers NaN with `R_TRUNC = −1` in the same cases.
`src/ai/ev.ts` turns a non-zero code into a throw rather than letting a half-written `out` be
argmaxed. The rest meta also carries a diagnostic breakdown of the value it returns (`R_PTENPAI`,
`R_PWIN`, `R_EVALUE`, `R_ECOST`) — no decision reads it; it exists so a hand can be audited in the
terms it is played in.

## The DP

The state is `(13-tile resting shape, own draws taken, red 5p held, riichi)`; melds are fixed per
root. One turn is: we draw (win, improve, or throw it straight back), then the other three seats
act.

- **Probability (plan D1).** Copies of type _k_ still available are
  `pool_k = base_k − max(0, hand_k − root_k)` and the denominator at turn _j_ is `N_j = N_root − j`.
  The shrinking denominator is what makes a wait exactly hypergeometric on a no-shape-change path
  (`P(miss T times) = C(N−w,T)/C(N,T)`, asserted to 1e-9), and the per-type absorption is what stops
  a hand from drawing the same tile twice. ⚑ Tiles we DISCARDED are not removed from the pool —
  every uninteresting draw collapses into one branch, so the search does not know which tile it was
  — and the excess probability mass that creates is charged to exactly that collapsed branch. Win
  and improvement probabilities therefore stay exact.
- **Terminals.** Waits are priced by the scorer above, once per (shape, wait, riichi, aka, 一発) and
  cached: dora, aka, 裏ドラ and 一発 are constant additions to _every_ reading of one hand, so the
  decomposition runs once and only `basePoints` is re-evaluated per outcome — the ura/kan-dora
  mixtures are exact, not a re-scoring approximation. 流局 settles the 3000 exactly over independent
  Bernoulli opponent tenpai.
- **Hazards.** `handvalue.ts:238`'s form, for both our deal-in and the opponents' win: the drift is
  added ONCE to the sum of the reads, `rate·(Σ tenpaiP + oppGrowth·j)`, not once per opponent.
  Reading it per-opponent tripled it and on a quiet table retired most of the hand's mass over
  twenty turns.
- **`ev.riichi` gates the DECISION, not the action.** With the sub-switch off the seat keeps its own
  riichi block and still declares, so the search always models the declaration; only the root's
  reported `O_RIICHI` is withheld. A search forbidden to declare prices every closed hand as a
  damaten, and the tail then reads a 4200-point win where the exact path can only collect a
  ~1300-point 門前清自摸和.
- **`O_TOTAL` is the PUSH line.** Folding is a real option at every state, but its value does not
  depend on the shape — so the moment it dominates, every candidate collapses onto one number and
  the argmax is decided by the root deal-in cost alone. The seat then discards the safest tile every
  turn and never advances. The fold has its own slot, and `bestFold > bestPush` is the verdict.
- **Root deal-in.** `costIn` is already the EXPECTED cost in points with the doctrine priced in, so
  it is subtracted once and neither `pIn` nor the standings `risk` is applied to it again. `pIn`
  does one job: the hand ends on a deal-in, so it removes survival mass from what comes after.
- **Branching.** After an accepting draw the fan is opened by rung (widened 2026-08-31, owner: "we
  can sacrifice speed for the best result we could get"): at shanten ≤ 1 **every** shanten-keeping
  discard is enumerated, at 2向聴 the best six by acceptance mass plus one reserved slot for the
  discard that keeps the most dora, deeper the best three. 待ち替え runs at 聴牌 and 1向聴, keeps
  the best **two** by expected win value (was one), and must buy at least two more live copies at
  聴牌 / eight at 1向聴. Without any bound a plain tenpai root needed 145k states and was priced by
  the tail for most of them — and a truncated candidate beside an exact one is a seam.

  Measured cost of the widening on 40 real discard roots (`runs/ev/lane-800000.jsonl`,
  `exactShanten` 3, `maxNodes` 600000): 1向聴 roots 63.0k → 82.1k value states, 2向聴 180k → 219k,
  total wall time 11.9 s → 12.4 s. Most of what it would have cost is not paid at all: when the fan
  is fully open the ranking has nothing to decide, so the ukeire probe behind `mass` is skipped.

  Measured VALUE of it, against oracle mode on 20 real 1向聴 rests at T = 2: **zero**. The residual
  pruning loss is not the width of the fan, it is the acceptance-mass GATE on 待ち替え — dropping
  that gate from `baseMass + 8` to 0 takes the 1向聴 loss from 6.05% mean / 22.76% worst to 0.67% /
  6.66%, and costs 25× the states (every root class then truncates, and a truncated field is priced
  by the closed-form tail — a much larger error than the 6% being bought). At a gate of 4 the trade
  is 4.15% for 2×. The gate stays at 8. ⚑ The T = 2 horizon exaggerates that 6%: with two draws left
  most of a 1向聴 rest's value is the 流局 tenpai settlement, so a wider shape is worth a great
  deal; over the ten-plus draws a real 1向聴 hand has, nearly every shape reaches tenpai.
- **Budget, and the one model per field rule.** `maxNodes` value states, a fixed order (candidates
  ascending, draw types ascending, discards ascending), the closed-form tail beyond it. A
  candidate's price is a pure function of the 13 tiles it leaves and the tile it throws — which
  makes `O_TOTAL(ty) == mjev_eval_rest(rest_ty)` an identity once the rest is asked with the thrown
  tile in its own river and with the fold line read from `O_FOLDLINE` (`O_TOTAL` is the push line;
  see the test). It was not, until 2026-08-30: see **The 529/203 defect** below.

  **What is shared and what is not** (2026-08-31). The 529/203 repair withdrew ALL cross-candidate
  sharing, because a `Shape` held pool-dependent data. The split is now explicit and the safe half
  is back:

  | shared across candidates                                       | rebuilt per candidate                                                 |
  | -------------------------------------------------------------- | --------------------------------------------------------------------- |
  | the shape arena and its index (identity, 向聴)                 | `mass` (Σ pool over the acceptance)                                   |
  | the acceptance MASK (`ukeireMaskImpl` — counts in, counts out) | the successor edge list (selection AND ranking run off the mass)      |
  | `statExtra` / 染め手模様 (dora, melds, winds)                  | the value memo and the wait prices (already keyed by the thrown type) |
  | `Geo` — `discardAnalysis`'s (best, mask) per drawn type        |                                                                       |

  The mechanism is a generation counter, `EvCtx::gen`, bumped by `clearCand`: every pool-dependent
  field carries the generation it was computed under, so a stale one is a recompute and never a
  wrong price. `Geo` is the one that pays — `discardAnalysis` is the hot loop of the whole search
  (one `groupWords` plus a merge per held type, per draw type, per expanded shape) and its answer is
  a pure function of the 14 counts.

  Measured on 40 real discard roots at `exactShanten` 3 / `maxNodes` 600000: 12.4 s → **8.9 s**
  (1.39×) with **identical node counts to the last state** — which is the point: sharing geometry
  cannot move a value, only the time it takes to reach it. A probe that shared everything unsafely
  (no `clearCand` at all) ran 7.9 s, so 1.39× of a possible 1.57× is bought without giving up the
  identity above.

  ⚑ The shape arena is now the UNION over the whole field rather than one candidate's working set,
  so its ceiling was raised 262144 → 2^20. At the old ceiling `4m5m7m9m 1p2p2p3p 1s2s ESSF` at T =
  17 truncated with 1.84M value states and the node budget still untouched — and a truncation there
  is not a slow answer, it is the whole field dropping to the closed-form tail.

  The candidates cannot all be searched to `exactShanten` at any affordable budget, so
  `exactShanten` is a MAXIMUM and the whole field is priced at ONE level:

  - any level at or above the worst candidate's 向聴 runs the identical search (nothing in the tree
    is ever worse than the root it grew from), so there is exactly one "everybody exact" level —
    `shMax` — and it is tried first;
  - if any candidate escapes there, the level drops below the BEST candidate's 向聴, where nobody is
    exact, every state is the closed-form tail, and the answer costs one node per candidate;
  - the levels in between are never attempted — including `exactShanten` itself when it happens to
    be one of them. They price the advancing discards exactly and the shape-wrecking ones by the
    tail, so the argmax is decided by whichever of the two models is the more generous, and the two
    are 1.6–2.2× apart at 3向聴 under a fitted hazard vector (below).

  Total cost is therefore still bounded by `maxNodes` plus one node per candidate: a failed attempt
  is abandoned at the first escape, and the attempt below it expands nothing. `O_TRUNC` is 1 when
  the answer came off the tail attempt while the configured level would have expanded something —
  whether the exact attempt ran out of budget or was never offered because it would have split the
  field — and 0 when the level used is the configured model (`shMax` below `exactShanten` is the
  same search, not a fallback).

  ⚑ At the defaults (`exactShanten` 3, `maxNodes` 1200000 since 2026-08-31) a 3向聴 root usually
  still does NOT get the exact model — its field needs level 4, which costs 0.65–1.2M states — and
  is answered from the tail. What the repair bought is that the whole field is answered from the
  SAME model, so the argmax is a comparison. Why the level is not simply raised: an attempt that
  escapes is abandoned and the field falls to the tail ANYWAY, having spent the whole budget, so an
  unaffordable level is strictly worse than one that is never offered. On 60 real lane roots
  `exactShanten` 4 at `maxNodes` 1.2M reaches the exact model on 5 of 9 3向聴 roots and doubles the
  mean decision (278 → 506 ms, p95 920 → 2122 ms); at 2.5M it reaches 5 of 6 at p95 3.5 s.

### The 529/203 defect (2026-08-30)

`mjev_eval_discard` priced 切1s on `4m5m7m9m 1p2p2p3p 1s2s ESSF` (3向聴, quiet table, T = 17,
`weights/ev-0830b.json`) at **416** while `mjev_eval_rest` on the thirteen tiles that discard leaves
answered **224**. The rest evaluation is `max(push, riichi, fold)` over the very subtree the discard
root pushes into, so the discard root cannot be worth more. It was two independent cross-candidate
leaks:

1. **The node budget was sliced** `maxNodes / nCand`. That slice was invented when the value memo
   was shared across candidates — which it has not been since the thrown tile entered the memo key
   (a discard into one's own wait is 振聴, and every terminal below depends on it). With the memo
   partitioned the slice bought no sharing at all and starved exactly the candidates with the widest
   shapes onto the tail, which prices a 3向聴 shape at 1.6–2.2× what the exact path collects. The
   argmax then selected whichever candidate the budget had starved — systematically the shapeless
   one, which is the "seat builds slow hands, 聴牌率 21%" symptom from the first grade.
2. **The shape arena was shared**, but a `Shape` caches POOL-dependent data (`mass`, and the
   top-three pruning `ensureEdges` performs off it) and the pool is `base − max(0, hand − root)`
   with a root that differs per candidate. A shape first analysed under 切1s was reused verbatim
   under 切4m with the wrong live counts behind it: the same tile came back at a different price
   depending on which candidates preceded it.

Priced one at a time, every candidate already agreed with its own `eval_rest` to the last bit; it
was only the field that was wrong. `test/ev_native_test.ts` pins both halves — the reported hand,
and equality across every candidate of twenty random roots.

### The brute-force oracle, and oracle mode

The DP has no reference implementation (owner decision), so `test/ev_native_test.ts` carries one for
instances small enough to enumerate: a naive recursion written from the specification, sharing
nothing with this file except `mjev_score` (itself bit-compared against `scoreWin`) and
`kernel.ts`'s shanten. The instances are a 中 pon plus ten concealed tiles over a handful of types,
a wall of at most a dozen tiles and at most three draws — open, so riichi is never on offer and
every win has a yaku. It reproduces plan D1's pool (`base − max(0, hand − root)`, denominator
`N_root − j`) rather than true depletion, because that approximation is part of the model under
test.

For the comparison to be about the MODEL and not the pruning, the engine is asked for the same
answer with its pruning switched off: a **NEGATIVE `maxNodes`** is the oracle switch (magnitude is
still the node cap). Every discard at every state becomes a successor, shanten-RAISING ones included
— so the attempt ladder above is also disabled, since "any level at or above `shMax` is the same
search" stops being true the moment a hand is allowed to get worse. `mergeEv` refuses a negative
budget, so this cannot arrive from a ktune; only a test that packs the parameter vector itself can
reach it.

Result: `mjev_eval_rest` agrees with the oracle to **0 ulp** on 120 instances (silent and hazardous
tables), and `mjev_eval_discard`'s per-candidate `O_TOTAL` agrees on 120 candidates over 20 discard
roots, root cost formula and fold line included — unchanged by the 2026-08-31 widening and sharing,
as it must be: oracle mode disables every one of those bounds. What the pruning costs on a real
13-tile hand is measured the same way and reported: 0.00% at one draw left, 0.10% mean / 2.07% worst
at two on random rests, and on twenty REAL 1向聴 rests from `runs/ev/lane-800000.jsonl` 6.05% mean /
22.76% worst at two draws — the same before and after the widening, which is what located the
residual in the 待ち替え mass gate rather than in the width of the fan (see **Branching**). Three
draws cannot be brute-forced — un-pruned, one state mints up to 34 × 14 successors and the shape
arena runs out (36,782 value states and a truncated answer on every hand tried).

### The tail, and why it is calibrated per rung

Past `exactShanten` (and past the budget) the value is `handvalue.ts`'s chain × its static value,
minus the hazard cost accrued on the surviving mass. Four things there are NOT a straight port, each
measured rather than chosen, and each of them was a seat that could not play:

- **One level per advance.** `handvalue.ts` folds every hand at 3向聴 or worse into one rung —
  harmless when it only prices the seat's own hand, fatal when it RANKS discards, because a 4向聴
  hand is then charged the same three advances as a 3向聴 one and its wider acceptance wins. The
  chain is `shanten` advances long; only the fitted mean it reads at each level is clamped to the
  four rungs. For shanten ≤ 3 it is `handPwin` step for step.
- **The counted rung stays counted.** It is the only thing that tells one candidate discard from
  another at 3向聴+. Capping it at the fitted mean (which every shapeless hand exceeds) put every
  shanten-keeping candidate within five points of every other and left the argmax to the tie-break.
- **The same survival the exact path uses.** `handvalue.ts` decays its mass by the opponents' win
  chance alone and bills our own deal-in against mass that never dies. Over fifteen turns that
  charges a deal-in rate whose total exceeds one, and it sank every push line onto the fold line.
- **The hazard half is not calibrated — it is the exact path's own** (2026-08-31). `cal` used to
  scale the WHOLE tail. What the tail actually approximates is `handvalue.ts`'s advance chain: how
  fast a shapeless hand improves and how often it collects. The sweep around that chain — mass
  decayed by `surviveAt`, `alive × hazardCost(j+t)` charged per turn, the 流局 settlement — is the
  same arithmetic `turnValue` runs on the same population rates, so it is not a second model and
  needs no calibration. `cal` now scales the win term alone. That single change is what closed the
  live-reads gap below; the constants did not move.

- **`TAIL_CAL[s]`, per rung, measured.** `handvalue.ts`'s scalars were never calibrated as POINTS —
  the incumbent consumes the chain through `evWeight` (0.1), so only its ordering ever mattered. The
  DP needs the tail and the exact search to be comparable, and the discrepancy is not a fixed
  multiple: it GROWS with the number of rungs (the fitted means are a shaped hand's, and a shapeless
  one neither advances nor narrows the way they say), then turns and becomes PESSIMISTIC past four.
  One factor fitted on 1–2向聴 read 2–3× high at 3向聴+, and the seat kept a 4向聴 hand over a 3向聴
  one at every root it saw.

  | s          | 1      | 2      | 3      | 4      | 5      | 6      |
  | ---------- | ------ | ------ | ------ | ------ | ------ | ------ |
  | `TAIL_CAL` | 0.4102 | 0.3951 | 0.3951 | 0.3951 | 0.3951 | 0.3951 |

  Measured level by level — `exactShanten = s` prices a shape exactly, `s − 1` forces the same shape
  onto the tail — over random rests, iterated to a fixed point. It scales the WHOLE tail value, win
  term and hazard cost and 流局 settlement together: scaling only the winnings left the LEVEL
  uncalibrated the moment the table was live, and the fold verdict is a comparison of levels, so a
  tail-priced hand crossed the fold line at Σ tenpaiP ≈ 0.02 where the same hand priced exactly
  never folded at all. **Calibrated on the DEFAULTS**, because one constant cannot satisfy both
  hazard settings (the cost and settlement are a different fraction of the value on a live table
  than on a silent one); the silent arm is printed beside it and runs 0.63–1.24×. ⚑ From 2向聴 up
  the exact reference truncates into the tail it is being compared against (2/40 at 2向聴, 29/40 at
  4向聴), so those levels are calibrated against a partly-tail reference; the truncation count is
  printed with every ratio.

  **IT USED TO HOLD ONLY WITH THE READS SILENT, AND NOW IT DOES NOT HAVE TO** (2026-08-31). Every
  arm the constants were fitted on leaves `tenpaiP` at zero — the "defaults" there are the
  PARAMETERS, not the opponents — and the seat never plays on a table with no opponents. With `cal`
  scaling the whole tail it discounted the COST by the same factor it discounted the upside, and the
  two models drifted apart by 2–10×, sign flips included. Splitting the halves (above) fixed it
  without moving a constant:

  | Σ 聴牌率      | 1向聴 | 2向聴 | 3向聴 | 4向聴 | 5向聴 | 6向聴 |
  | ------------- | ----- | ----- | ----- | ----- | ----- | ----- |
  | 0 — before    | 0.94  | 0.98  | 0.94  | 1.10  |       |       |
  | 0.15 — before | −3.64 | −1.67 | 2.91  | 1.39  |       |       |
  | 0.45 — before | 10.34 | 3.31  | 1.88  | 1.33  |       |       |
  | 0 — after     | 0.911 | 0.767 | 0.781 | 0.931 | 1.363 | 1.638 |
  | 0.15 — after  | 2.132 | 1.552 | 1.106 | 0.975 | 0.981 | 0.988 |
  | 0.45 — after  | 1.327 | 1.059 | 0.993 | 1.000 | 1.000 | 1.000 |

  Where the tail is the model that actually GOVERNS — a field whose worst candidate is past
  `exactShanten`, i.e. 3向聴+ roots, junk hands at 巡1-3 — the live arms are now 0.97–1.11, and the
  push/fold verdict agrees with the exact path at every Σ聴牌率 the test sweeps. What is left is the
  mirror of the old error and it is the one being accepted: on a SILENT table the tail reads ~1.3×
  high, because the win term is nearly all of a silent hand's value.

  A LIVE RE-FIT OF THE CONSTANTS WAS TRIED AND DOES NOT CONVERGE. From 2向聴 up the exact reference
  truncates into the very tail it is being compared against (22/40 at 3向聴, 30/40 at 4向聴 at
  `maxNodes` 300000), so raising `cal[s]` raises both sides: four damped iterations walked `cal[3]`
  0.395 → 0.526 with the ratio still at 1.14. At 1–2向聴 the ratio is ill-conditioned in the other
  direction — the exact and tail values straddle zero there, because the fold option is `max`-ed in
  and does not scale with `cal` at all — and the mean swung −0.70, 4.57, 0.82, 1.82 over the same
  four iterations. The silent arm IS well conditioned and asks for 0.374 / 0.303 at rungs 1–2; a
  compromise between the arms lands within a few percent of the table above. So the constants stay
  and both arms are printed by `test/ev_native_test.ts`.

## Cost

`mjev_eval_discard` on discard roots built from real lane wires (`runs/ev/lane-800000.jsonl`; the 13
tiles the champion kept plus one drawn tile), bucketed by the root's own 向聴, at the defaults after
the 2026-08-31 widening + sharing — 60 roots, `exactShanten` 3, `maxNodes` 1,200,000:

| 根の向聴 | n  | 平均 ms | p95 ms | 平均 節点 | p95 節点  | 尾部に落ちた |
| -------- | -- | ------- | ------ | --------- | --------- | ------------ |
| 聴牌     | 8  | 5.1     | 9.5    | 1,032     | 1,686     | 0            |
| 1向聴    | 21 | 108.6   | 505.4  | 62,905    | 176,275   | 0            |
| 2向聴    | 21 | 676.6   | 1758.4 | 329,246   | 1,200,010 | 2            |
| 3向聴    | 9  | 15.9    | 1.0    | 12,091    | 13        | 8            |
| 4向聴    | 1  | 0.8     | 0.8    | 12        | 12        | 0            |
| 全体     | 60 | 277.8   | 920.2  |           |           |              |

**3.6 決定/秒.** A 3向聴 root is cheap only because its field needs level 4 and never gets it: the
ladder sees the configured level below `shMax`, skips the exact attempt entirely, and answers from
the tail at one node per candidate. That is the tail's whole remaining jurisdiction — about a
seventh of the decisions, all of them junk hands at 巡1-3.

Raising the level (same 60 roots, `maxNodes` 1.2M): `exactShanten` 4 → 2.0 決定/秒, mean 506 ms, p95
2122 ms, 5 of 9 3向聴 roots reaching the exact model and the other 4 spending the whole budget to
learn they cannot. `exactShanten` 5 is byte-identical to 4 on these roots. At `maxNodes` 2.5M, level
4 reaches 5 of 6 3向聴 roots at 1.7 決定/秒 and p95 3.5 s.

Where the time went, on 40 of the same roots at `maxNodes` 600000:

| build                           | total  | 1向聴 mean | 2向聴 mean |
| ------------------------------- | ------ | ---------- | ---------- |
| 2026-08-30                      | 11.9 s | 167.6 ms   | 721.0 ms   |
| + widened fan                   | 12.4 s | 214.4 ms   | 713.7 ms   |
| + shared geometry (`clearCand`) | 8.9 s  | 146.2 ms   | 528.8 ms   |
| unsafe full sharing (probe)     | 7.9 s  | 148.4 ms   | 448.8 ms   |

待ち替え at 1向聴 is what the `sameShantenRungs` switch buys and it is not cheap: almost every idle
draw offers some nominally wider shape, and each one is a new shape that lives for the rest of the
hand. The gate that holds it down is on ACCEPTANCE MASS (two more live copies at 聴牌, eight at
1向聴); ranking those candidates by expected win value while testing that value against the mass
threshold puts points on one side of a comparison and a tile count on the other, which is how a
plain 聴牌 root once cost 55005 states and 372 ms instead of 1653 and 0.7 ms.

One `mjev_eval_discard` still spends at most `maxNodes` value states plus one per candidate: the
budget is a FIELD budget, spent by the candidates in order, and a failed attempt is abandoned at the
first escape while the attempt below it expands nothing.

### Does it advance?

The only thing the seat does with any of this is choose a discard, so the measurement that matters
is whether the best shanten-ADVANCING candidate outscores the best shanten-keeping one. Over 100
random 3向聴 roots and 100 random 4向聴 roots with the hazards silent (`test/ev_native_test.ts`
asserts ≥ 90%):

| root  | exactShanten 2 | exactShanten 3 |
| ----- | -------------- | -------------- |
| 3向聴 | 100% (88/88)   | —              |
| 4向聴 | 100% (92/92)   | 100% (92/92)   |

(A 3向聴 root at `exactShanten` 3 is the case the level ladder answers exactly rather than by
comparison, so it is not a tail measurement and the test does not run it. The 4向聴 row rising from
93% to 100% is the 2026-08-31 tail split: with the hazard half no longer discounted, a hand that
advances is no longer competing against a shape whose costs were scaled away.)

### Where it still disagrees with the champion (2026-08-31)

A 20-game smoke (`selfplay --games=20 --seed=5 --seats=khhh`, champion + the fitted `ev` block,
`riichi`/`calls` off) after the widening, the sharing and the tail split:

|                                | 道場順位  | 和了率     | 放銃率     | 聴牌率     | 立直率 | 平均和了打点 |
| ------------------------------ | --------- | ---------- | ---------- | ---------- | ------ | ------------ |
| 2026-08-30                     | 3.000     | 12.2%      | 12.7%      | 32.7%      |        |              |
| this build                     | **2.550** | 13.5%      | **9.9%**   | **36.5%**  | 9.9%   | 6,393        |
| champion (h seats, same table) | 2.35–2.70 | 17.5–26.5% | 12.1–15.7% | 38.5–40.4% | 22–25% | 5,041–7,384  |

The seat now reaches 流局聴牌 as often as anyone and feeds least of anyone; what it does not do is
WIN. Replaying its decisions against the champion's on the same observations (three hanchan, quiet
tables only, 232 discards) says where:

| root   | agreement | EV kept a WORSE 向聴 |
| ------ | --------- | -------------------- |
| 聴牌   | 100%      | 0                    |
| 1向聴  | 91%       | 0                    |
| 2向聴  | 52%       | 10                   |
| 3向聴  | 59%       | 31                   |
| 4向聴+ | 87%       | 4                    |

45 of 232, all at 巡1-4, and the champion does it zero times. The traces are unambiguous — on
`3p 1p 8s 4m 4p 9m 3m 5s 8s N 2m 4p 7s 1s` (2向聴, 巡1) the DP prices 切1p to **3向聴** at 707
against 切1s keeping 2向聴 at 552 — the seat spends a rung to buy acceptance.

IT IS NOT THE TAIL AND IT IS NOT SPEED, both measured. Re-running the same replay with
`exactShanten` 4 / `maxNodes` 2,500,000, so that 3向聴 fields are priced by the exact search instead
of the closed form, moves it from 45 to 42 (3向聴: 31 → 24). Setting `dealinRate` back to the 0.05
default, so a future discard is no longer nearly free, moves it not at all (45 → 45). The exact
recursion itself prefers the wider, worse shape.

Which is a statement about the MODEL, and a coherent one: at 巡1-3 there are fifteen draws left, so
one more rung costs almost no win probability while the extra acceptance is collected every turn —
and the static value model does not discriminate enough (`valueDamaten` 4200 flat plus
`valuePerDora`) for the narrower, faster hand to win the comparison on 打点. The champion's
`−1000·shanten` forbids the trade a priori and is right for the wrong reason. The consequence runs
all the way through the smoke: wide early ⇒ tenpai late ⇒ 聴牌率 36.5% but 立直率 9.9% against the
field's 22-25% ⇒ 和了率 13.5%. The next lever is the win-VALUE half of the model (M15b's `eV`
audit), not the search.

## Build

```sh
deno task build-ev
# = sh native/build_ev.sh
# = clang++ -std=c++17 -O3 -flto -ffp-contract=off -Wall -Wextra -fvisibility=hidden \
#           -dynamiclib -o native/libmjev.dylib native/mjev.cc
```

Warning-free under `-Wall -Wextra`. `-ffp-contract=off` matters for the DP, whose value memo has to
answer the same double twice for `--jobs` identity — the scorer and the shanten core are integer
arithmetic end to end and would not care. Keep it in step with the compile line in
`test/ev_native_test.ts`, which rebuilds a missing or stale artifact itself.

## Using it from TypeScript

`src/ai/ev.ts` resolves the dylib **relative to the module** (`src/ai/` → `../../native/`), asks for
permissions with `querySync` (never `request`), and checks `mjev_abi()` against `EV_ABI`. The gate
has only two settings here:

| value    | behaviour                                                           |
| -------- | ------------------------------------------------------------------- |
| `0`      | REFUSED — there is no TypeScript path to fall back to               |
| anything | require native; missing dylib / ABI / `--allow-ffi` throws, by name |

`closeEvLib()` re-arms the gate for tests that move the artifact aside; `closeEv(core)` frees one
context and is idempotent.
