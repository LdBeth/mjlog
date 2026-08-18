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

exec clang++ -std=c++17 -O3 -flto -Wall -Wextra -fvisibility=hidden \
	-dynamiclib -o "$dir/libmjkernel.$ext" "$dir/mjkernel.cc"
