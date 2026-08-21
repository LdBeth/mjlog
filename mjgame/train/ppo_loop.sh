#!/bin/sh
# PPO self-play loop: collect on-policy rollouts, update, repeat.
#
#   usage: sh train/ppo_loop.sh <iters> [games-per-iter=400] [jobs=4] [start-seed=50000]
#
# Run from the mjgame directory (the one holding src/, weights/, train/).
#
# Each iteration plays `games-per-iter` hanchan in ONE deno process using
# `--jobs` worker threads (selfplay's output is byte-identical to a sequential
# run with the same seed, so parallelism cannot perturb the data), all four
# seats driven by the CURRENT weights/ at --temp=1 (PPO's importance ratio is
# only valid if the rollouts were sampled from exactly the policy that
# train/ppo.py loads as --init), then runs one PPO update in place:
# --init weights --out weights.
#
# Seeds never overlap: iteration i plays the consecutive seeds
# start + i*10000 .. start + i*10000 + games - 1, so the 10000-wide iteration
# lanes stay disjoint as long as games-per-iter stays under 10000.
#
# POSIX sh only -- no bashisms, no GNU-only flags.

set -e

usage() {
    echo "usage: sh train/ppo_loop.sh <iters> [games-per-iter=400] [jobs=4] [start-seed=50000]" >&2
    exit 2
}

[ $# -ge 1 ] || usage

ITERS=$1
GAMES=${2:-400}
JOBS=${3:-4}
START=${4:-50000}

case "$ITERS$GAMES$JOBS$START" in
    *[!0-9]*) echo "ppo_loop: all arguments must be non-negative integers" >&2; usage ;;
esac
[ "$ITERS" -ge 1 ] || usage
[ "$JOBS" -ge 1 ] || usage
[ "$GAMES" -ge 1 ] || usage
[ "$GAMES" -lt 10000 ] || { echo "ppo_loop: games-per-iter must stay under 10000 (seed lane width)" >&2; exit 2; }

PY=./train/.venv/bin/python

[ -f src/main.ts ] || { echo "ppo_loop: run me from the mjgame directory" >&2; exit 2; }
[ -f weights/manifest.json ] || { echo "ppo_loop: no weights/manifest.json -- train BC first" >&2; exit 2; }
[ -x "$PY" ] || { echo "ppo_loop: no venv python at $PY" >&2; exit 2; }

mkdir -p runs/ppo

echo "=== ppo_loop: $ITERS iteration(s), $GAMES game(s)/iter over $JOBS job(s), start seed $START"

i=1
while [ "$i" -le "$ITERS" ]; do
    SEED=$((START + i * 10000))
    echo
    echo "=== iter $i/$ITERS: collecting $GAMES game(s) with weights/ (temp=1, jobs=$JOBS, seed $SEED)"
    rm -f "runs/ppo/iter$i.jsonl"

    # Same permissions as deno.json's `selfplay` task: without --allow-ffi
    # (and the env var that gates it) every rollout falls back to the pure-TS
    # forward and shanten -- the identical policy, just far slower.
    deno run --allow-read --allow-write --allow-ffi --allow-env=MJGAME_NATIVE \
        src/main.ts selfplay \
        --games="$GAMES" \
        --seed="$SEED" \
        --jobs="$JOBS" \
        --seats=nnnn \
        --weights=weights/manifest.json \
        --temp=1 \
        --record="runs/ppo/iter$i.jsonl" \
        >"runs/ppo/iter$i.log" 2>&1

    [ -s "runs/ppo/iter$i.jsonl" ] || {
        echo "ppo_loop: iteration $i produced no data -- see runs/ppo/iter$i.log" >&2
        exit 1
    }

    echo "=== iter $i/$ITERS: PPO update (weights -> weights)"
    "$PY" train/ppo.py \
        --data "runs/ppo/iter$i.jsonl" \
        --init weights \
        --out weights \
        --epochs 3

    echo "=== iter $i/$ITERS: done"
    i=$((i + 1))
done

echo
echo "=== ppo_loop: finished $ITERS iteration(s); weights/ holds the latest policy"
