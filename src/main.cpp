// CLI entry point
//
// Builds both a BruteForceIndex and an HNSWIndex on the same
// randomly generated dataset, runs a query against both, and reports
// construction time, query time, and recall for that query.
//
// Usage:
//   ./hnsw_cli --n 10000 --dim 128 --M 16 --ef_construction 64 --k 10 --ef_search 50
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>
 
#include "brute_force.hpp"
#include "hnsw.hpp"
#include "types.hpp"
 
namespace {
 
struct Args {
    int n = 10000;
    int dim = 128;
    int M = 16;
    int ef_construction = 64;
    int k = 10;
    int ef_search = 50;
    unsigned seed = 42;
};
 
int get_int_flag(int argc, char** argv, const std::string& name, int fallback) {
    for (int i = 1; i < argc - 1; i++) {
        if (name == argv[i]) {
            return std::stoi(argv[i + 1]);
        }
    }
    return fallback;
}
 
Args parse_args(int argc, char** argv) {
    Args args;
    args.n = get_int_flag(argc, argv, "--n", args.n);
    args.dim = get_int_flag(argc, argv, "--dim", args.dim);
    args.M = get_int_flag(argc, argv, "--M", args.M);
    args.ef_construction = get_int_flag(argc, argv, "--ef_construction", args.ef_construction);
    args.k = get_int_flag(argc, argv, "--k", args.k);
    args.ef_search = get_int_flag(argc, argv, "--ef_search", args.ef_search);
    args.seed = static_cast<unsigned>(get_int_flag(argc, argv, "--seed", static_cast<int>(args.seed)));
    return args;
}
 
std::vector<Vec> generate_random_vectors(int count, int dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
 
    std::vector<Vec> vectors(count, Vec(dim));
    for (auto& v : vectors) {
        for (auto& x : v) x = dist(rng);
    }
    return vectors;
}
 
double elapsed_ms(std::chrono::high_resolution_clock::time_point start,
                   std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
 
}  
 
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
 
    std::cout << "Config: n=" << args.n << " dim=" << args.dim
              << " M=" << args.M << " ef_construction=" << args.ef_construction
              << " k=" << args.k << " ef_search=" << args.ef_search
              << " seed=" << args.seed << "\n\n";
 
    std::cout << "Generating " << args.n << " random vectors...\n";
    auto vectors = generate_random_vectors(args.n, args.dim, args.seed);
 
    BruteForceIndex bf_index(args.dim);
    auto bf_build_start = std::chrono::high_resolution_clock::now();
    for (const auto& v : vectors) bf_index.add(v);
    auto bf_build_end = std::chrono::high_resolution_clock::now();
 
    HNSWIndex hnsw_index(args.dim, args.M, args.ef_construction);
    auto hnsw_build_start = std::chrono::high_resolution_clock::now();
    for (const auto& v : vectors) hnsw_index.add(v);
    auto hnsw_build_end = std::chrono::high_resolution_clock::now();
 
    std::cout << "Brute-force build time: " << elapsed_ms(bf_build_start, bf_build_end) << " ms\n";
    std::cout << "HNSW build time:        " << elapsed_ms(hnsw_build_start, hnsw_build_end) << " ms\n\n";
 

    const Vec& query = vectors[0];
 
    auto bf_query_start = std::chrono::high_resolution_clock::now();
    auto bf_results = bf_index.search(query, args.k);
    auto bf_query_end = std::chrono::high_resolution_clock::now();
 
    auto hnsw_query_start = std::chrono::high_resolution_clock::now();
    auto hnsw_results = hnsw_index.search(query, args.k, args.ef_search);
    auto hnsw_query_end = std::chrono::high_resolution_clock::now();
 
    double bf_query_ms = elapsed_ms(bf_query_start, bf_query_end);
    double hnsw_query_ms = elapsed_ms(hnsw_query_start, hnsw_query_end);
 
    std::cout << "Brute-force top-" << args.k << " (query time " << bf_query_ms << " ms):\n";
    for (const auto& nb : bf_results) {
        std::cout << "  id=" << nb.id << "  dist=" << nb.dist << "\n";
    }
 
    std::cout << "\nHNSW top-" << args.k << " (query time " << hnsw_query_ms << " ms):\n";
    for (const auto& nb : hnsw_results) {
        std::cout << "  id=" << nb.id << "  dist=" << nb.dist << "\n";
    }
 
    int overlap = 0;
    for (const auto& bf_nb : bf_results) {
        for (const auto& hnsw_nb : hnsw_results) {
            if (bf_nb.id == hnsw_nb.id) {
                ++overlap;
                break;
            }
        }
    }
    double recall = static_cast<double>(overlap) / args.k;
 
    std::cout << "\nRecall@" << args.k << " for this query: " << recall
              << " (" << overlap << "/" << args.k << ")\n";
    std::cout << "Speedup (HNSW vs brute-force, this query): "
              << (bf_query_ms / hnsw_query_ms) << "x\n";
 
    if (bf_results.empty() || bf_results[0].id != 0) {
        std::cerr << "\nWARNING: nearest neighbor of vector 0 was not itself -- "
                      "something may be wrong.\n";
    }
 
    return 0;
}
 