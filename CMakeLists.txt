cmake_minimum_required(VERSION 3.15)
project(hnsw_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

include_directories(include)

# Core library sources -- add hnsw.cpp here once it exists
add_library(hnsw_core
    src/brute_force.cpp
)

add_executable(hnsw_cli src/main.cpp)
target_link_libraries(hnsw_cli hnsw_core)

add_executable(run_tests tests/test_hnsw.cpp)
target_link_libraries(run_tests hnsw_core)

add_executable(run_bench bench/benchmark.cpp)
target_link_libraries(run_bench hnsw_core)
