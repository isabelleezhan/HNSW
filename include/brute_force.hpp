#pragma once
#include "types.hpp"
#include <vector>

// Exact k-NN via linear scan
// for computing recall of the HNSW index later
class BruteForceIndex {
public:
  explicit BruteForceIndex(int dim);
  void add(const Vec &v);
  std::vector<Neighbor> search(const Vec &query, int k) const;
  size_t size() const { return data_.size(); }

private:
  std::vector<Vec> data_;
  int dim_;
};
