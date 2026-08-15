#!/usr/bin/env bash
# Runs the large-scale BF vs HNSW benchmark (bench/benchmark_large_scale.cpp)
# for N in {100k, 250k, 500k, 1M} in parallel (build time at ef_construction=200
# dominates and each run is single-threaded, so running the four sizes
# concurrently cuts wall-clock roughly to the N=1M run instead of the sum),
# then merges the per-N CSVs into one bench_large_scale.csv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:-$REPO_ROOT/build-release/run_bench_large}"
OUT_DIR="$REPO_ROOT"
SIZES=(100000 250000 500000 1000000)

if [ ! -x "$BIN" ]; then
    echo "run_bench_large binary not found at $BIN (build it first)" >&2
    exit 1
fi

pids=()
for n in "${SIZES[@]}"; do
    "$BIN" "$n" "$OUT_DIR/bench_large_scale_${n}.csv" \
        > "$OUT_DIR/bench_large_scale_${n}.log" 2>&1 &
    pids+=($!)
    echo "started N=$n (pid $!)"
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

merged="$OUT_DIR/benchmark_large_scale.csv"
head -n1 "$OUT_DIR/bench_large_scale_${SIZES[0]}.csv" > "$merged"
for n in "${SIZES[@]}"; do
    tail -n +2 "$OUT_DIR/bench_large_scale_${n}.csv" >> "$merged"
done

echo "merged results written to $merged"
exit $status
