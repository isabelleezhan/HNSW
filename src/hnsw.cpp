#include "hnsw.hpp"
#include "types.hpp"
#include <queue>
#include <set>

HNSWIndex::HNSWIndex(int dim, int M, int ef_construction)
    : dim_(dim), M_(M), ef_construction_(ef_construction),
      mL_(1.0f / std::log(static_cast<float>(M))) {}

int HNSWIndex::random_level() const {
  // generate random num 0 <= r < 1
  float r = level_dist_(rng_);
  // clamp to 1e-12 if r is roughly 0
  r = std::max(r, 1e-12f);
  return static_cast<int>(std::floor(-std::log(r) * mL_));
}

std::vector<Neighbor> HNSWIndex::greedy_search(const Vec &query, VecId entry,
                                               int ef, int lvl) const {
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
    if (bestsofar.size() == static_cast<size_t>(ef) && bestsofar.top() < c) {
      break;
    }

    std::vector<VecId> neighbors = nodes_[c.id].neighbors[lvl];
    for (VecId nb : neighbors) {
      if (visited.count(nb) != 0) {
        continue;
      }

      float d = squared_l2(query, data_[nb]);

      candidates.push({nb, d});
      visited.insert(nb);
      if (bestsofar.size() < static_cast<size_t>(ef)) {
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

void HNSWIndex::trim_neighbors(VecId id, int lvl) {
  auto &neighbor_ids = nodes_[id].neighbors[lvl];
  if (neighbor_ids.size() <= static_cast<size_t>(M_))
    return;

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

  // For every neighbor dropping, remove the reverse edge back to `id`
  for (size_t i = M_; i < scored.size(); ++i) {
    VecId dropped = scored[i].id;
    auto &their_neighbors = nodes_[dropped].neighbors[lvl];
    their_neighbors.erase(
        std::remove(their_neighbors.begin(), their_neighbors.end(), id),
        their_neighbors.end());
  }

  neighbor_ids = kept;
}

std::vector<Neighbor> HNSWIndex::search(const Vec &query, int k,
                                        int ef_search) const {
  if (static_cast<int>(query.size()) != dim_) {
    throw std::invalid_argument(
        "Query dimension does not match index dimension.");
  }
  if (entry_point_ == -1) {
    return {};
  }

  VecId curr_entry = entry_point_;
  int entry_layer = nodes_[entry_point_].top_layer;

  // phase 1: descend through upper layers with cheap ef=1 greedy search
  for (int i = entry_layer; i > 0; i--) {
    auto result = greedy_search(query, curr_entry, /*ef=*/1, i);
    curr_entry = result[0].id;
  }

  // phase 2: real ef_search-width search at layer 0
  auto results = greedy_search(query, curr_entry, ef_search, /*layer=*/0);

  if (results.size() > static_cast<size_t>(k)) {
    results.resize(k);
  }
  return results;
}

void HNSWIndex::add(const Vec &v) {
  if (static_cast<int>(v.size()) != dim_) {
    throw std::invalid_argument(
        "Vector dimension does not match index dimension.");
  }
  int lvl = random_level();
  VecId new_id = static_cast<VecId>(data_.size());
  data_.push_back(v);

  std::vector<std::vector<VecId>> neighbours;
  Node nd = {new_id, lvl, neighbours};
  nd.neighbors.resize(lvl + 1);
  nodes_.push_back(nd);
  // int lvl = random_level();

  // case 1: first node in graph
  if ((int)entry_point_ == -1) {
    entry_point_ = new_id;
    return;
  }

  VecId curr_entry = entry_point_;
  int entry_layer = nodes_[entry_point_].top_layer;

  // phase 1: descend through layers ABOVE the new node's level
  for (int i = entry_layer; i > lvl; i--) {
    auto result = greedy_search(v, curr_entry, /*ef=*/1, i);
    curr_entry = result[0].id;
  }

  // phase 2: from min(lvl, entry_layer) down to 0:
  // ef_construction-width search AND connect the new node in, layer by layer.
  int start_layer = std::min(lvl, entry_layer);
  for (int i = start_layer; i >= 0; i--) {
    auto candidates = greedy_search(v, curr_entry, ef_construction_, i);

    // connect new_id to each candidate, both directions
    int connect_count = std::min(static_cast<int>(candidates.size()), M_);
    for (int j = 0; j < connect_count; j++) {
      const auto &c = candidates[j];
      nodes_[new_id].neighbors[i].push_back(c.id);
      nodes_[c.id].neighbors[i].push_back(new_id);
      trim_neighbors(c.id, i);
    }

    // Use this layer's best-found candidate as the entry point one layer down.
    if (!candidates.empty()) {
      curr_entry = candidates[0].id;
    }
  }

  // If the new node reaches higher than anything currently in the graph,
  // it becomes the new entry point.
  if (lvl > entry_layer) {
    entry_point_ = new_id;
  }
}