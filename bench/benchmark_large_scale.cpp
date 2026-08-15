// Large-scale BF vs HNSW benchmark.
//
// Sweeps dataset size N in {100k, 250k, 500k, 1M} at a fixed
// dim=64, k=10, M=16, ef_construction=200, and reports brute-force vs
// HNSW query latency, speedup, and recall@10 for ef_search in {200, 300}.
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "brute_force.hpp"
#include "hnsw.hpp"
#include "types.hpp"

namespace {

std::vector<Vec> generate_random_vectors(int count, int dim, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<Vec> vectors(count, Vec(dim));
  for (auto &v : vectors) {
    for (auto &x : v)
      x = dist(rng);
  }
  return vectors;
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point start,
                  std::chrono::high_resolution_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double compute_recall(const std::vector<Neighbor> &bf_results,
                      const std::vector<Neighbor> &hnsw_results) {
  int overlap = 0;
  for (const auto &bf_nb : bf_results) {
    for (const auto &hnsw_nb : hnsw_results) {
      if (bf_nb.id == hnsw_nb.id) {
        ++overlap;
        break;
      }
    }
  }
  return bf_results.empty() ? 0.0
                            : static_cast<double>(overlap) / bf_results.size();
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <N> [output_csv]\n";
    return 1;
  }
  const int n = std::atoi(argv[1]);
  const std::string out_path =
      argc >= 3 ? argv[2] : ("bench_large_scale_" + std::to_string(n) + ".csv");

  const int dim = 64;
  const int k = 10;
  const int M = 16;
  const int ef_construction = 200;
  const int num_queries = 500;
  const unsigned data_seed = 42;
  const unsigned query_seed = 999;
  const std::vector<int> ef_search_values = {200, 300};

  std::cout << "[N=" << n << "] generating data...\n" << std::flush;
  auto vectors = generate_random_vectors(n, dim, data_seed);
  auto queries = generate_random_vectors(num_queries, dim, query_seed);

  std::cout << "[N=" << n << "] building brute-force index...\n" << std::flush;
  BruteForceIndex bf_index(dim);
  auto bf_build_start = std::chrono::high_resolution_clock::now();
  for (const auto &v : vectors)
    bf_index.add(v);
  auto bf_build_end = std::chrono::high_resolution_clock::now();
  double bf_build_ms = elapsed_ms(bf_build_start, bf_build_end);
  std::cout << "[N=" << n << "] bf_build_ms=" << bf_build_ms << "\n" << std::flush;

  std::cout << "[N=" << n << "] building HNSW index (M=" << M
            << ", ef_construction=" << ef_construction << ")...\n" << std::flush;
  HNSWIndex hnsw_index(dim, M, ef_construction);
  auto hnsw_build_start = std::chrono::high_resolution_clock::now();
  for (const auto &v : vectors)
    hnsw_index.add(v);
  auto hnsw_build_end = std::chrono::high_resolution_clock::now();
  double hnsw_build_ms = elapsed_ms(hnsw_build_start, hnsw_build_end);
  std::cout << "[N=" << n << "] hnsw_build_ms=" << hnsw_build_ms << "\n" << std::flush;

  // Exact top-k ground truth, computed once and reused for every ef_search.
  std::cout << "[N=" << n << "] computing exact ground truth + BF query latency...\n"
            << std::flush;
  std::vector<std::vector<Neighbor>> ground_truth;
  ground_truth.reserve(num_queries);
  double bf_total_ms = 0.0;
  for (const auto &q : queries) {
    auto start = std::chrono::high_resolution_clock::now();
    auto results = bf_index.search(q, k);
    auto end = std::chrono::high_resolution_clock::now();
    bf_total_ms += elapsed_ms(start, end);
    ground_truth.push_back(std::move(results));
  }
  double bf_avg_query_ms = bf_total_ms / num_queries;
  std::cout << "[N=" << n << "] bf_avg_query_ms=" << bf_avg_query_ms << "\n" << std::flush;

  std::ofstream csv(out_path);
  csv << "n,dim,k,M,ef_construction,ef_search,bf_build_ms,hnsw_build_ms,"
         "bf_avg_query_ms,hnsw_avg_query_ms,speedup,recall_at_10\n";

  for (int ef_search : ef_search_values) {
    double hnsw_total_ms = 0.0, total_recall = 0.0;
    for (int i = 0; i < num_queries; ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      auto hnsw_results = hnsw_index.search(queries[i], k, ef_search);
      auto end = std::chrono::high_resolution_clock::now();
      hnsw_total_ms += elapsed_ms(start, end);
      total_recall += compute_recall(ground_truth[i], hnsw_results);
    }
    double hnsw_avg_query_ms = hnsw_total_ms / num_queries;
    double speedup = hnsw_avg_query_ms > 0 ? bf_avg_query_ms / hnsw_avg_query_ms : 0.0;
    double avg_recall = total_recall / num_queries;

    std::cout << "[N=" << n << "] ef_search=" << ef_search
              << "  bf_avg_query_ms=" << bf_avg_query_ms
              << "  hnsw_avg_query_ms=" << hnsw_avg_query_ms
              << "  speedup=" << speedup << "x"
              << "  recall@10=" << avg_recall << "\n" << std::flush;

    csv << n << "," << dim << "," << k << "," << M << "," << ef_construction
        << "," << ef_search << "," << bf_build_ms << "," << hnsw_build_ms
        << "," << bf_avg_query_ms << "," << hnsw_avg_query_ms << ","
        << speedup << "," << avg_recall << "\n";
  }

  std::cout << "[N=" << n << "] done, wrote " << out_path << "\n" << std::flush;
  return 0;
}
