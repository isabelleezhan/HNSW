#include "brute_force.hpp"
#include <algorithm>
#include <queue>
#include <stdexcept>

BruteForceIndex::BruteForceIndex(int dim) : dim_(dim) {}

void BruteForceIndex::add(const Vec &v) {
  if (static_cast<int>(v.size()) != dim_) {
    throw std::invalid_argument(
        "Vector dimension does not match index dimension.");
  }
  data_.push_back(v);
}

std::vector<Neighbor> BruteForceIndex::search(const Vec &query, int k) const {
  if (static_cast<int>(query.size()) != dim_) {
    throw std::invalid_argument(
        "Vector dimension does not match index dimension.");
  }
  // Max-heap of size k: keeps the k smallest distances seen so far.
  std::priority_queue<Neighbor> heap;
  for (VecId i = 0; i < static_cast<VecId>(data_.size()); ++i) {
    float d = squared_l2(query, data_[i]);
    if (static_cast<int>(heap.size()) < k) {
      heap.push({i, d});
    } else if (d < heap.top().dist) {
      heap.pop();
      heap.push({i, d});
    }
  }

  std::vector<Neighbor> result;
  result.reserve(heap.size());
  while (!heap.empty()) {
    result.push_back(heap.top());
    heap.pop();
  }
  // heap pops largest-first; reverse so result is sorted nearest-first
  std::reverse(result.begin(), result.end());
  return result;
}
