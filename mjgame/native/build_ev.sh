#!/bin/sh
# Builds the expected-value core next to `libmjkernel.dylib`.
#
#   sh native/build_ev.sh            (or: deno task build-ev)
#
# Pure C++17, no frameworks. The artifact lands beside this script because
# `src/ai/ev.ts` resolves it module-relative — and unlike the kernel and the MLP
# shim, this one is REQUIRED whenever a ktune carries an `ev` block: there is no
# TypeScript twin to fall back to, so a missing dylib is a refusal, not a
# slowdown.
set -e

dir=`dirname "$0"`

case `uname -s` in
Darwin) ext=dylib ;;
*)      ext=so ;;
esac

# -ffp-contract=off buys nothing for unit A (the scorer and the shanten core are
# integer arithmetic end to end) and everything for unit B's DP, whose value
# memo has to answer the same double twice for `--jobs` identity. Keep it in
# step with the compile line in `test/ev_native_test.ts`.
exec clang++ -std=c++17 -O3 -flto -ffp-contract=off -Wall -Wextra -fvisibility=hidden \
	-dynamiclib -o "$dir/libmjev.$ext" "$dir/mjev.cc"
