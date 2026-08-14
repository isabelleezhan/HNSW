#pragma once
#include "types.hpp"

// STUB -- brute_force.cpp calls squared_l2(query, data_[i]) but doesn't
// #include this file, so you likely already have your own distance.hpp
// (or it's declared elsewhere). Replace this with your actual version.
// Flagging so you don't lose track of this before it's wired together.

inline float squared_l2(const Vec& a, const Vec& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}
