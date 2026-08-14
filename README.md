# hnsw-cpp

From-scratch C++17 implementation of HNSW (Hierarchical Navigable Small World)
approximate nearest-neighbor search, with a brute-force exact k-NN baseline
for correctness testing and recall/latency benchmarking.

## Status

- [x] Brute-force exact k-NN baseline
- [x] Flat graph construction (single layer)
- [x] Hierarchical layers
- [ ] Parameter sweeps (M, efConstruction, efSearch)
- [ ] Benchmark suite (recall@k vs. QPS)

## Build

```
mkdir build && cd build
cmake .. && make
./run_tests
./hnsw_cli
./run_bench
```

