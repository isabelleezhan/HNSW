#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
 
#include "brute_force.hpp"
#include "hnsw.hpp"
#include "types.hpp"
 
namespace {
 
int tests_run = 0;
int tests_passed = 0;
 
void check(bool condition, const std::string& name) {
    ++tests_run;
    if (condition) {
        ++tests_passed;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        std::cout << "  [FAIL] " << name << "\n";
    }
}
 
bool contains_id(const std::vector<Neighbor>& results, VecId id) {
    for (const auto& n : results) {
        if (n.id == id) return true;
    }
    return false;
}
 
// --- Test 1: single node ---
void test_single_node() {
    HNSWIndex index(2, /*M=*/4, /*ef_construction=*/8);
    index.add({0.0f, 0.0f});
 
    auto results = index.search({5.0f, 5.0f}, /*k=*/1, /*ef_search=*/8);
    check(results.size() == 1 && results[0].id == 0,
          "hnsw: single-node index returns that node regardless of query");
}
 
// --- Test 2: exact match ---
void test_query_matches_indexed_point() {
    HNSWIndex index(2, /*M=*/4, /*ef_construction=*/8);
    index.add({0.0f, 0.0f});   // id 0
    index.add({10.0f, 10.0f}); // id 1
    index.add({1.0f, 1.0f});   // id 2
 
    auto results = index.search({0.0f, 0.0f}, /*k=*/1, /*ef_search=*/8);
    check(results.size() == 1 && results[0].id == 0,
          "hnsw: querying with an indexed point returns itself as nearest");
}
 
// --- Test 3: k larger than dataset ---
void test_k_larger_than_n() {
    HNSWIndex index(2, /*M=*/4, /*ef_construction=*/8);
    index.add({0.0f, 0.0f});
    index.add({1.0f, 1.0f});
 
    auto results = index.search({0.0f, 0.0f}, /*k=*/10, /*ef_search=*/10);
    check(results.size() == 2,
          "hnsw: k > N returns all available points without crashing");
}
 
// --- Test 4: cross-check against brute force, small dataset ---
void test_matches_brute_force_small() {
    const int dim = 8;
    const int n = 50;
    const int k = 5;
    const int ef_search = 20;  // > k, gives the search room to explore
 
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
 
    HNSWIndex hnsw_index(dim, /*M=*/8, /*ef_construction=*/32);
    BruteForceIndex bf_index(dim);
 
    for (int i = 0; i < n; ++i) {
        Vec v(dim);
        for (auto& x : v) x = dist(rng);
        hnsw_index.add(v);
        bf_index.add(v);
    }
 
    Vec query(dim);
    for (auto& x : query) x = dist(rng);
 
    auto bf_results = bf_index.search(query, k);
    auto hnsw_results = hnsw_index.search(query, k, ef_search);
 
    check(hnsw_results.size() == static_cast<size_t>(k),
          "hnsw: returns k results on a non-trivial dataset");
 
    int overlap = 0;
    for (const auto& n : bf_results) {
        if (contains_id(hnsw_results, n.id)) ++overlap;
    }
    std::cout << "    (overlap with brute-force top-" << k << ": "
              << overlap << "/" << k << ")\n";
 
    check(overlap >= k - 1,
          "hnsw: recall on small dataset is close to brute-force ground truth");
}
 
// --- Test 5: M is never exceeded ---
void test_m_never_exceeded() {
    const int dim = 8;
    const int n = 200;
    const int M = 6;
 
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
 
    HNSWIndex index(dim, M, /*ef_construction=*/24);
    for (int i = 0; i < n; ++i) {
        Vec v(dim);
        for (auto& x : v) x = dist(rng);
        index.add(v);
    }
 
    bool all_within_m = true;
    for (VecId id = 0; id < n; ++id) {
        if (index.neighbor_count(id) > static_cast<size_t>(M)) {
            all_within_m = false;
            std::cout << "    node " << id << " has "
                      << index.neighbor_count(id) << " neighbors (M=" << M << ")\n";
        }
    }
    check(all_within_m, "hnsw: no node exceeds M neighbors after trimming");
}
 
// --- Test 6: neighbor symmetry (undirected graph invariant) ---
void test_neighbor_symmetry() {
    const int dim = 8;
    const int n = 150;
    const int M = 5;
 
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
 
    HNSWIndex index(dim, M, /*ef_construction=*/20);
    for (int i = 0; i < n; ++i) {
        Vec v(dim);
        for (auto& x : v) x = dist(rng);
        index.add(v);
    }
 
    bool symmetric = true;
    for (VecId a = 0; a < n; ++a) {
        for (VecId b : index.neighbors_of(a)) {
            const auto& b_neighbors = index.neighbors_of(b);
            bool found = std::find(b_neighbors.begin(), b_neighbors.end(), a)
                         != b_neighbors.end();
            if (!found) {
                symmetric = false;
                std::cout << "    asymmetry: " << a << " -> " << b
                          << " but not " << b << " -> " << a << "\n";
            }
        }
    }
    check(symmetric, "hnsw: every edge is symmetric (undirected graph invariant)");
}
 
// --- Test 7: mismatched dimension throws ---
void test_mismatched_dimension_throws() {
    HNSWIndex index(4, /*M=*/4, /*ef_construction=*/8);
    index.add({0.0f, 0.0f, 0.0f, 0.0f});
 
    bool threw = false;
    try {
        index.add({1.0f, 2.0f});  // wrong dimension: 2 instead of 4
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "hnsw: add() throws on mismatched vector dimension");
}
 
// --- Test 8: duplicate vectors don't break anything ---
void test_duplicate_vectors() {
    HNSWIndex index(2, /*M=*/4, /*ef_construction=*/8);
    index.add({1.0f, 1.0f});  // id 0
    index.add({1.0f, 1.0f});  // id 1, identical to id 0
    index.add({5.0f, 5.0f});  // id 2
 
    auto results = index.search({1.0f, 1.0f}, 2, /*ef_search=*/8);
    check(results.size() == 2,
          "hnsw: duplicate vectors don't crash search");
    check((contains_id(results, 0) && contains_id(results, 1)),
          "hnsw: both duplicate points are found as nearest to themselves");
}
 
// --- Test 9: larger-scale recall check ---
void test_recall_larger_scale() {
    const int dim = 32;
    const int n = 2000;
    const int k = 10;
    const int ef_search = 50;  // >> k on purpose
 
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
 
    HNSWIndex hnsw_index(dim, /*M=*/16, /*ef_construction=*/64);
    BruteForceIndex bf_index(dim);
 
    for (int i = 0; i < n; ++i) {
        Vec v(dim);
        for (auto& x : v) x = dist(rng);
        hnsw_index.add(v);
        bf_index.add(v);
    }
 
    const int num_queries = 20;
    double total_recall = 0.0;
 
    for (int q = 0; q < num_queries; ++q) {
        Vec query(dim);
        for (auto& x : query) x = dist(rng);
 
        auto bf_results = bf_index.search(query, k);
        auto hnsw_results = hnsw_index.search(query, k, ef_search);
 
        int overlap = 0;
        for (const auto& n : bf_results) {
            if (contains_id(hnsw_results, n.id)) ++overlap;
        }
        total_recall += static_cast<double>(overlap) / k;
    }
 
    double avg_recall = total_recall / num_queries;
    std::cout << "    average recall@" << k << " (ef_search=" << ef_search
              << ") over " << num_queries << " queries: " << avg_recall << "\n";
 
    check(avg_recall >= 0.8,
          "hnsw: average recall at scale is strong with proper ef_search");
}
 
}  // namespace
 
int main() {
    std::cout << "Running HNSW tests...\n";
    test_single_node();
    test_query_matches_indexed_point();
    test_k_larger_than_n();
    test_matches_brute_force_small();
    test_m_never_exceeded();
    test_neighbor_symmetry();
    test_mismatched_dimension_throws();
    test_duplicate_vectors();
    test_recall_larger_scale();
 
    std::cout << "\n" << tests_passed << "/" << tests_run << " tests passed.\n";
    return (tests_passed == tests_run) ? 0 : 1;
}
