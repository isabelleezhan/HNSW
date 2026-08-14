// placeholder
#include <iostream>
#include <random>
#include "brute_force.hpp"
#include "types.hpp"

int main() {
    int dim = 128;
    int n = 1000;

    // TODO 1: generate `n` random Vecs of dimension `dim`
    // hint: std::mt19937 rng(42); std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // TODO 2: construct a BruteForceIndex(dim) and add() each vector

    // TODO 3: pick a query (e.g. the first vector you generated),
    //         call index.search(query, k) for some k like 10

    // TODO 4: loop over the results and print neighbor.id and neighbor.dist

    return 0;
}