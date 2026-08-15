// Benchmark suite
//
// Produces four CSV files:
//   1. bench_dataset_size.csv      -- brute-force vs HNSW query time as N
//   grows,
//                                     showing the crossover point where HNSW
//                                     starts winning.
//   2. bench_ef_search.csv         -- recall vs QPS as ef_search varies, at a
//                                     fixed large N. Classic recall/latency
//                                     curve.
//   3. bench_m_sweep.csv           -- build time and recall as M varies.
//   4. bench_ef_construction.csv   -- build time, query latency, and recall as
//                                     ef_construction varies at N=100k.
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
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

int main() {
  const int dim = 64;
  const int k = 10;
  const int num_queries = 500;
  const unsigned data_seed = 42;
  const unsigned query_seed = 999;

  // ================================================================
  // Benchmark 1: dataset size sweep
  // ================================================================
  {
      std::cout << "=== Benchmark 1: dataset size sweep ===\n";
      std::vector<int> sizes = {1000, 5000, 10000, 25000, 50000, 100000};
      const int M = 16;
      const int ef_construction = 64;
      const int ef_search = 50;

      std::ofstream csv("bench_dataset_size.csv");
      csv << "n,bf_build_ms,hnsw_build_ms,bf_avg_query_ms,hnsw_avg_query_ms,"
             "bf_qps,hnsw_qps,avg_recall\n";

      for (int n : sizes) {
          auto vectors = generate_random_vectors(n, dim, data_seed);
          auto queries = generate_random_vectors(num_queries, dim,
          query_seed);

          BruteForceIndex bf_index(dim);
          auto bf_build_start = std::chrono::high_resolution_clock::now();
          for (const auto& v : vectors) bf_index.add(v);
          auto bf_build_end = std::chrono::high_resolution_clock::now();
          double bf_build_ms = elapsed_ms(bf_build_start, bf_build_end);

          HNSWIndex hnsw_index(dim, M, ef_construction);
          auto hnsw_build_start = std::chrono::high_resolution_clock::now();
          for (const auto& v : vectors) hnsw_index.add(v);
          auto hnsw_build_end = std::chrono::high_resolution_clock::now();
          double hnsw_build_ms = elapsed_ms(hnsw_build_start,
          hnsw_build_end);

          double bf_total_ms = 0.0, hnsw_total_ms = 0.0, total_recall = 0.0;
          for (const auto& q : queries) {
              auto bf_start = std::chrono::high_resolution_clock::now();
              auto bf_results = bf_index.search(q, k);
              auto bf_end = std::chrono::high_resolution_clock::now();
              bf_total_ms += elapsed_ms(bf_start, bf_end);

              auto hnsw_start = std::chrono::high_resolution_clock::now();
              auto hnsw_results = hnsw_index.search(q, k, ef_search);
              auto hnsw_end = std::chrono::high_resolution_clock::now();
              hnsw_total_ms += elapsed_ms(hnsw_start, hnsw_end);

              total_recall += compute_recall(bf_results, hnsw_results);
          }

          double bf_avg_ms = bf_total_ms / num_queries;
          double hnsw_avg_ms = hnsw_total_ms / num_queries;
          double bf_qps = 1000.0 / bf_avg_ms;
          double hnsw_qps = 1000.0 / hnsw_avg_ms;
          double avg_recall = total_recall / num_queries;

          std::cout << "N=" << n
                    << "  bf_query=" << bf_avg_ms << "ms"
                    << "  hnsw_query=" << hnsw_avg_ms << "ms"
                    << "  recall=" << avg_recall << "\n";

          csv << n << "," << bf_build_ms << "," << hnsw_build_ms << ","
              << bf_avg_ms << "," << hnsw_avg_ms << ","
              << bf_qps << "," << hnsw_qps << "," << avg_recall << "\n";
      }
      std::cout << "Results written to bench_dataset_size.csv\n\n";
  }

  // // ================================================================
  // // Benchmark 2: ef_search sweep (recall vs QPS tradeoff curve)
  // // ================================================================
  {
    std::cout << "=== Benchmark 2: ef_search sweep ===\n";
    const int n = 100000;
    const int M = 16;
    const int ef_construction = 64;
    std::vector<int> ef_search_values = {50, 100, 150, 200, 300, 400};

    auto vectors = generate_random_vectors(n, dim, data_seed);
    auto queries = generate_random_vectors(num_queries, dim, query_seed);

    BruteForceIndex bf_index(dim);
    for (const auto &v : vectors)
      bf_index.add(v);

    HNSWIndex hnsw_index(dim, M, ef_construction);
    for (const auto &v : vectors)
      hnsw_index.add(v);

    std::vector<std::vector<Neighbor>> ground_truth;
    auto bf_start = std::chrono::high_resolution_clock::now();
    ground_truth.reserve(num_queries);
    for (const auto &q : queries) {
      ground_truth.push_back(bf_index.search(q, k));
    }
    auto bf_end = std::chrono::high_resolution_clock::now();
    double bf_total_ms =
        std::chrono::duration<double, std::milli>(bf_end - bf_start).count();

    double bf_avg_query_ms = bf_total_ms / queries.size();

    double bf_qps = 1000.0 / bf_avg_query_ms;

    std::cout << "Brute force: avg_query=" << bf_avg_query_ms
              << "ms  qps=" << bf_qps << "\n\n";

    std::ofstream csv("bench_ef_search_100k.csv");
    csv << "ef_search,avg_query_ms,qps,avg_recall\n";

    for (int ef_search : ef_search_values) {
      double total_ms = 0.0, total_recall = 0.0;
      for (int i = 0; i < num_queries; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto hnsw_results = hnsw_index.search(queries[i], k, ef_search);
        auto end = std::chrono::high_resolution_clock::now();
        total_ms += elapsed_ms(start, end);
        total_recall += compute_recall(ground_truth[i], hnsw_results);
      }
      double avg_ms = total_ms / num_queries;
      double qps = 1000.0 / avg_ms;
      double avg_recall = total_recall / num_queries;

      std::cout << "ef_search=" << ef_search << "  avg_query=" << avg_ms << "ms"
                << "  qps=" << qps << "  recall=" << avg_recall << "\n";

      csv << ef_search << "," << avg_ms << "," << qps << "," << avg_recall
          << "\n";
    }
    std::cout << "Results written to bench_ef_search_100k.csv\n\n";
  }

  // // ================================================================
  // // Benchmark 3: M sweep (graph density vs build time/recall)
  // // ================================================================
  {
      std::cout << "=== Benchmark 3: M sweep ===" << std::endl;
      const int n = 50000;
      const int ef_search = 50;
      std::vector<int> m_values = {4, 8, 12, 16, 24, 32};

      auto vectors = generate_random_vectors(n, dim, data_seed);
      auto queries = generate_random_vectors(num_queries, dim, query_seed);

      BruteForceIndex bf_index(dim);
      for (const auto& v : vectors) bf_index.add(v);

      std::vector<std::vector<Neighbor>> ground_truth;
      ground_truth.reserve(num_queries);
      for (const auto& q : queries) {
          ground_truth.push_back(bf_index.search(q, k));
      }

      std::ofstream csv("bench_m_sweep.csv");
      csv << "M,ef_construction,build_ms,avg_query_ms,qps,avg_recall\n";

      for (int M : m_values) {
          int ef_construction = M * 4;

          HNSWIndex hnsw_index(dim, M, ef_construction);
          auto build_start = std::chrono::high_resolution_clock::now();
          for (const auto& v : vectors) hnsw_index.add(v);
          auto build_end = std::chrono::high_resolution_clock::now();
          double build_ms = elapsed_ms(build_start, build_end);

          double total_ms = 0.0, total_recall = 0.0;
          for (int i = 0; i < num_queries; ++i) {
              auto start = std::chrono::high_resolution_clock::now();
              auto hnsw_results = hnsw_index.search(queries[i], k,
              ef_search); auto end =
              std::chrono::high_resolution_clock::now(); total_ms +=
              elapsed_ms(start, end); total_recall +=
              compute_recall(ground_truth[i], hnsw_results);
          }
          double avg_ms = total_ms / num_queries;
          double qps = 1000.0 / avg_ms;
          double avg_recall = total_recall / num_queries;

          std::cout << "M=" << M << " ef_construction=" << ef_construction
                    << "  build=" << build_ms << "ms"
                    << "  avg_query=" << avg_ms << "ms"
                    << "  recall=" << avg_recall << "\n";

          csv << M << "," << ef_construction << "," << build_ms << ","
              << avg_ms << "," << qps << "," << avg_recall << "\n";
      }
      std::cout << "Results written to bench_m_sweep.csv\n";
  }

  // ================================================================
  // Benchmark 4: ef_construction sweep at N=100k
  // ================================================================
  // Holds M and ef_search fixed so we can isolate how spending more
  // work during construction affects graph quality at larger scale.
  {
    std::cout << "\n=== Benchmark 4: ef_construction sweep (N=100k) ==="
              << std::endl;

    const int n = 100000;
    const int M = 16;
    const int ef_search = 50;
    std::vector<int> ef_construction_values = {64, 100, 150, 200, 300};

    auto vectors = generate_random_vectors(n, dim, data_seed);
    auto queries = generate_random_vectors(num_queries, dim, query_seed);

    // Compute exact top-k once and reuse it for every HNSW configuration.
    BruteForceIndex bf_index(dim);
    for (const auto &v : vectors) {
      bf_index.add(v);
    }

    std::vector<std::vector<Neighbor>> ground_truth;
    ground_truth.reserve(num_queries);
    for (const auto &q : queries) {
      ground_truth.push_back(bf_index.search(q, k));
    }

    std::ofstream csv("bench_ef_construction.csv");
    csv << "ef_construction,build_ms,avg_query_ms,qps,avg_recall\n";

    for (int ef_construction : ef_construction_values) {
      std::cout << "Building ef_construction=" << ef_construction << "..."
                << std::endl;

      HNSWIndex hnsw_index(dim, M, ef_construction);

      auto build_start = std::chrono::high_resolution_clock::now();
      for (const auto &v : vectors) {
        hnsw_index.add(v);
      }
      auto build_end = std::chrono::high_resolution_clock::now();
      double build_ms = elapsed_ms(build_start, build_end);

      double total_ms = 0.0;
      double total_recall = 0.0;

      for (int i = 0; i < num_queries; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto hnsw_results = hnsw_index.search(queries[i], k, ef_search);
        auto end = std::chrono::high_resolution_clock::now();

        total_ms += elapsed_ms(start, end);
        total_recall += compute_recall(ground_truth[i], hnsw_results);
      }

      double avg_ms = total_ms / num_queries;
      double qps = 1000.0 / avg_ms;
      double avg_recall = total_recall / num_queries;

      std::cout << "ef_construction=" << ef_construction
                << "  build=" << build_ms << "ms"
                << "  avg_query=" << avg_ms << "ms"
                << "  qps=" << qps << "  recall=" << avg_recall << "\n";

      csv << ef_construction << "," << build_ms << "," << avg_ms << "," << qps
          << "," << avg_recall << "\n";
    }

    std::cout << "Results written to bench_ef_construction.csv\n";
  }

  return 0;
}
