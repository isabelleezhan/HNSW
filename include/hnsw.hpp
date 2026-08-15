#ifndef HNSW_HPP
#define HNSW_HPP
#include "types.hpp"
#include <random>
#include <vector>

struct Node {
  VecId id;
  int top_layer; 
  // neighbors[layer] = neighbor ids at that layer
  std::vector<std::vector<VecId>> neighbors;   
};

// Exact k-NN via hsnw
// for computing recall of the HNSW index later
class HNSWIndex {
public:
  HNSWIndex(int dim, int M, int ef_construction);
  void add(const Vec &v);
  std::vector<Neighbor> greedy_search(const Vec &query, VecId entry,
                                      int ef, int lvl) const;
  std::vector<Neighbor> search(const Vec &query, int k, int ef_search) const;
  void trim_neighbors(VecId id, int lvl);

  // --- Debug/test accessors ---
  size_t neighbor_count(VecId id, int layer) const { return nodes_[id].neighbors[layer].size(); }
  const std::vector<VecId> &neighbors_of(VecId id, int layer) const {
    return nodes_[id].neighbors[layer];
  }
  int top_layer_of(VecId id) const { return nodes_[id].top_layer; }

private:
  int ef_construction_;     // candidate-list width used during insertion
  int dim_;                 // dim: vector dimensionality
  int M_;                   // max neighbors per layer 
  int M0_;                  // max neighbors at layer 0
  std::vector<Vec> data_;   // actual vector data, indexed by VecId
  std::vector<Node> nodes_; // graph structure, indexed by VecId
  VecId entry_point_ = -1;  // where search starts

  mutable std::mt19937 rng_;        // random number generator
  mutable std::uniform_real_distribution<float> level_dist_{0.0f, 1.0f};
  float mL_;                // normalization factor = 1 / ln(M)

  int random_level() const;
  std::vector<Neighbor> select_neighbors_heuristic(
    const Vec& query,
    const std::vector<Neighbor>& candidates,
    int max_neighbors
) const;
};

#endif