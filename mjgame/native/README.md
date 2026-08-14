# native/ — optional Accelerate inference shim

`rlnet.c` is a ~150-line C library that evaluates a dense MLP with `cblas_sgemv` from Apple's
Accelerate framework. It exists only to make `forward()` faster; **nothing depends on it**. With the
dylib absent, FFI denied, or the gate turned off, `src/rl/net.ts` runs its pure-TypeScript loop and
produces the same numbers (modulo float32 summation order, which BLAS reorders — differences are at
the 1e-6 level).

There is no mahjong in this file. It takes dimensions, activations and one flat blob of float32; the
TypeScript side is what knows those weights are a policy.

## Build

```sh
deno task build-native
# = clang -O3 -dynamiclib -framework Accelerate -o native/librlnet.dylib native/rlnet.c
```

Warning-free under `-Wall -Wextra`. `ACCELERATE_NEW_LAPACK` is defined inside the `.c` so the plain
command above needs no extra flags (the legacy CBLAS headers are deprecated as of macOS 13.3).

macOS only in practice — Accelerate is an OS framework. `net.ts` looks for `librlnet.so` / `.dll` on
other platforms, so a port only needs a BLAS.

## ABI

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

## Rules of use

- **Not re-entrant**: a context holds scratch state. One caller at a time; one context per network
  instance (`net.ts` creates one per loaded `Net`).
- `in` and `out` must not overlap, and `out` must be `dims[n_layers]` long.
- The handle is a pointer in disguise; never call `rlnet_forward` after `rlnet_destroy`.

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
