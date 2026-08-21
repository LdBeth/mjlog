#!/bin/sh
# Builds the shanten/ukeire compute kernel next to `librlnet.dylib`.
#
#   sh native/build_kernel.sh        (or: deno task build-kernel)
#
# Pure C++17, no frameworks — unlike rlnet.c it needs no BLAS. The artifact
# lands beside this script because `src/kernel.ts` resolves it module-relative.
set -e

dir=`dirname "$0"`

case `uname -s` in
Darwin) ext=dylib ;;
*)      ext=so ;;
esac

# -ffp-contract=off is load-bearing, not hygiene: `mj_shape_masses` has to be
# BIT-identical to the TypeScript it replaces, and clang would otherwise be free
# to fuse `a + b*c` into an FMA — a different double from the one JavaScript,
# which never fuses, would have produced. Keep it in step with the compile line
# in `test/kernel_native_test.ts`.
exec clang++ -std=c++17 -O3 -flto -ffp-contract=off -Wall -Wextra -fvisibility=hidden \
	-dynamiclib -o "$dir/libmjkernel.$ext" "$dir/mjkernel.cc"
