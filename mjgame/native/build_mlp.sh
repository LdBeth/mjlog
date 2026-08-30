#!/bin/sh
# Builds the small-head MLP shim next to the other two dylibs.
#
#   sh native/build_mlp.sh        (or: deno task build-mlp)
#
# Plain C11, no frameworks and no BLAS — deliberately: `librlnet.dylib` already
# has a generic MLP over Accelerate, but BLAS reorders summation and its parity
# is graded with a 1e-4 tolerance. The learned heads need BIT equality (their
# sign decides a fold), so this one is hand-written loops. The artifact lands
# beside this script because `src/ai/mlp.ts` resolves it module-relative.
set -e

dir=`dirname "$0"`

case `uname -s` in
Darwin) ext=dylib ;;
*)      ext=so ;;
esac

# -ffp-contract=off is load-bearing, not hygiene: the forward pass has to be
# BIT-identical to the TypeScript it replaces, and clang would otherwise be free
# to fuse `acc + w*x` into an FMA — a different double from the one JavaScript,
# which never fuses, would have produced. Keep it in step with the compile line
# in `test/mlp_native_test.ts`.
exec clang -std=c11 -O3 -ffp-contract=off -Wall -Wextra -fvisibility=hidden \
	-dynamiclib -o "$dir/libmjmlp.$ext" "$dir/mlp.c"
