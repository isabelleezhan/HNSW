#ifndef TYPES_HPP
#define TYPES_HPP
#include <cstdint>
#include <vector>

using Vec = std::vector<float>;
using VecId = int32_t;

// Squared L2 distance
inline float squared_l2(const Vec &a, const Vec &b) {
  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

struct Neighbor {
  VecId id;
  float dist;
  // For max-heaps (keep the k smallest, pop the largest when exceed k)
  bool operator<(const Neighbor &other) const { return dist < other.dist; }
  bool operator>(const Neighbor& other) const {return dist > other.dist;}
};

#endif
