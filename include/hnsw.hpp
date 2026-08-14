#ifndef HNSW_HPP
#define HNSW_HPP
#include "types.hpp"
#include <vector>

struct Node {
    VecId id;
    std::vector<VecId> neighbors;
};

// Exact k-NN via hsnw
// for computing recall of the HNSW index later
class HNSWIndex {
public:
  HNSWIndex(int dim, int M, int ef_construction);
    void add(const Vec& v);
    std::vector<Neighbor> greedy_search(const Vec& query, VecId entry, int ef) const;
    std::vector<Neighbor> search(const Vec& query, int k) const {
        return greedy_search(query, entry_point_, k);
    }
    void trim_neighbors(VecId id);


private:
    int ef_construction_;        // candidate-list width used during insertion
    int dim_;                    // dim: vector dimensionality
    int M_;                      // max neighbors per layer (M0 = 2*M at layer 0)
    std::vector<Vec> data_;      // actual vector data, indexed by VecId
    std::vector<Node> nodes_;    // graph structure, indexed by VecId
    VecId entry_point_ = -1;          // where search starts 
};

#endif