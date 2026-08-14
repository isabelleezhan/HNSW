#include "hnsw.hpp"
#include "types.hpp"
#include <queue>
#include <set>

HNSWIndex::HNSWIndex(int dim, int M, int ef_construction)
    : dim_(dim), M_(M), ef_construction_(ef_construction) {}

std::vector<Neighbor> HNSWIndex::greedy_search(const Vec &query, VecId entry,
                                               int ef) const {
  std::priority_queue<Neighbor> bestsofar;
  std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>>
      candidates;
  std::set<VecId> visited;

  float curr_dist = squared_l2(query, data_[entry]);
  candidates.push({entry, curr_dist});
  bestsofar.push({entry, curr_dist});
  visited.insert(entry);

  while (!candidates.empty()) {
    Neighbor c = candidates.top();
    candidates.pop();

    // if c farther than worst item in bestsofar (full) -> stop
    if (bestsofar.size() == ef && bestsofar.top() < c) {
      break;
    }

    std::vector<VecId> neighbors = nodes_[c.id].neighbors;
    for (VecId nb : neighbors) {
      if (visited.count(nb) != 0) {
        continue;
      }

      float d = squared_l2(query, data_[nb]);

      candidates.push({nb, d});
      visited.insert(nb);
      if (bestsofar.size() < ef) {
        bestsofar.push({nb, d});
      } else if (bestsofar.size() == ef && bestsofar.top().dist > d) {
        bestsofar.pop();
        bestsofar.push({nb, d});
      }
    }
  }
  // return bestsofar as vector, sorted nearest-first
  std::vector<Neighbor> result;
  result.reserve(bestsofar.size());
  while (!bestsofar.empty()) {
    result.push_back(bestsofar.top());
    bestsofar.pop();
  }
  std::reverse(result.begin(), result.end());
  return result;
}

void HNSWIndex::trim_neighbors(VecId id) {
  auto &neighbor_ids = nodes_[id].neighbors;
  if (neighbor_ids.size() <= static_cast<size_t>(M_))
    return; // nothing to do

  // Compute distance from `id` to each of its neighbors
  std::vector<Neighbor> scored;
  for (VecId nb : neighbor_ids) {
    float d = squared_l2(data_[id], data_[nb]);
    scored.push_back({nb, d});
  }

  // Sort nearest-first, keep only the M closest
  std::sort(
      scored.begin(), scored.end(),
      [](const Neighbor &a, const Neighbor &b) { return a.dist < b.dist; });

  std::vector<VecId> kept;
  for (int i = 0; i < M_; ++i)
    kept.push_back(scored[i].id);

  // For every neighbor we're dropping, remove the reverse edge back to `id`
  for (size_t i = M_; i < scored.size(); ++i) {
    VecId dropped = scored[i].id;
    auto &their_neighbors = nodes_[dropped].neighbors;
    their_neighbors.erase(
        std::remove(their_neighbors.begin(), their_neighbors.end(), id),
        their_neighbors.end());
  }

  neighbor_ids = kept;
}

void HNSWIndex::add(const Vec &v) {
  if (static_cast<int>(v.size()) != dim_) {
    throw std::invalid_argument(
        "Vector dimension does not match index dimension.");
  }
  VecId new_id = static_cast<VecId>(data_.size());
  data_.push_back(v);

  std::vector<VecId> neighbours;
  Node nd = {new_id, neighbours};
  nodes_.push_back(nd);
  // int lvl = random_level();

  // case 1: first node in graph
  if ((int)entry_point_ == -1) {
    entry_point_ = new_id;
    return;
  }

  // case 2: find candidates w/ greedy_search from curr entry point
  std::vector<Neighbor> candidates = greedy_search(v, entry_point_, ef_construction_);

  // Connect new_id to each candidate, both directions
  int connect_count = std::min(static_cast<int>(candidates.size()), M_);
  for (int i = 0; i < connect_count; ++i) {
    const auto &c = candidates[i];
    nodes_[new_id].neighbors.push_back(c.id);
    nodes_[c.id].neighbors.push_back(new_id);

    // If c now has too many neighbors, trim it back to its M closest
    trim_neighbors(c.id);
  }
}