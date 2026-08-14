#!/bin/sh
# PPO self-play loop: collect on-policy rollouts, update, repeat.
#
#   usage: sh train/ppo_loop.sh <iters> [games-per-iter=400] [shards=4] [start-seed=50000]
#
# Run from the mjgame directory (the one holding src/, weights/, train/).
#
# Each iteration plays `games-per-iter` hanchan split across `shards` parallel
# deno processes, all four seats driven by the CURRENT weights/ at --temp=1
# (PPO's importance ratio is only valid if the rollouts were sampled from
# exactly the policy that train/ppo.py loads as --init), then runs one PPO
# update in place: --init weights --out weights.
#
# Seeds never overlap: seed = start + i*10000 + j*1000, and one shard consumes
# at most `games-per-iter / shards` consecutive seeds, so the 1000-wide lanes
# stay disjoint as long as a shard plays fewer than 1000 games.
#
# POSIX sh only -- no bashisms, no GNU-only flags.

set -e

usage() {
    echo "usage: sh train/ppo_loop.sh <iters> [games-per-iter=400] [shards=4] [start-seed=50000]" >&2
    exit 2
}

[ $# -ge 1 ] || usage

ITERS=$1
GAMES=${2:-400}
SHARDS=${3:-4}
START=${4:-50000}

case "$ITERS$GAMES$SHARDS$START" in
    *[!0-9]*) echo "ppo_loop: all arguments must be non-negative integers" >&2; usage ;;
esac
[ "$ITERS" -ge 1 ] || usage
[ "$SHARDS" -ge 1 ] || usage

PER=$((GAMES / SHARDS))
[ "$PER" -ge 1 ] || { echo "ppo_loop: games-per-iter must be >= shards" >&2; exit 2; }
[ "$PER" -lt 1000 ] || { echo "ppo_loop: games per shard must stay under 1000 (seed lane width)" >&2; exit 2; }

PY=./train/.venv/bin/python

[ -f src/main.ts ] || { echo "ppo_loop: run me from the mjgame directory" >&2; exit 2; }
[ -f weights/manifest.json ] || { echo "ppo_loop: no weights/manifest.json -- train BC first" >&2; exit 2; }
[ -x "$PY" ] || { echo "ppo_loop: no venv python at $PY" >&2; exit 2; }

mkdir -p runs/ppo

echo "=== ppo_loop: $ITERS iteration(s), $GAMES game(s)/iter over $SHARDS shard(s) ($PER each), start seed $START"

i=1
while [ "$i" -le "$ITERS" ]; do
    echo
    echo "=== iter $i/$ITERS: collecting $GAMES game(s) with weights/ (temp=1)"
    rm -f runs/ppo/iter$i.s*.jsonl

    j=0
    while [ "$j" -lt "$SHARDS" ]; do
        SEED=$((START + i * 10000 + j * 1000))
        echo "--- iter $i shard $j: $PER game(s), seed $SEED -> runs/ppo/iter$i.s$j.jsonl"
        deno run --allow-read --allow-write src/main.ts selfplay \
            --games="$PER" \
            --seed="$SEED" \
            --seats=nnnn \
            --weights=weights/manifest.json \
            --temp=1 \
            --record="runs/ppo/iter$i.s$j.jsonl" \
            >"runs/ppo/iter$i.s$j.log" 2>&1 &
        j=$((j + 1))
    done
    wait

    j=0
    while [ "$j" -lt "$SHARDS" ]; do
        [ -s "runs/ppo/iter$i.s$j.jsonl" ] || {
            echo "ppo_loop: shard $j produced no data -- see runs/ppo/iter$i.s$j.log" >&2
            exit 1
        }
        j=$((j + 1))
    done

    echo "=== iter $i/$ITERS: PPO update (weights -> weights)"
    "$PY" train/ppo.py \
        --data "runs/ppo/iter$i.s*.jsonl" \
        --init weights \
        --out weights \
        --epochs 3

    echo "=== iter $i/$ITERS: done"
    i=$((i + 1))
done

echo
echo "=== ppo_loop: finished $ITERS iteration(s); weights/ holds the latest policy"
