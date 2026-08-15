cmake_minimum_required(VERSION 3.15)
project(hnsw_cpp LANGUAGES CXX)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

include_directories(include)

# Core library sources 
add_library(hnsw_core
    src/brute_force.cpp
    src/hnsw.cpp
)

add_executable(hnsw_cli src/main.cpp)
target_link_libraries(hnsw_cli hnsw_core)

add_executable(run_tests tests/test_hnsw.cpp)
target_link_libraries(run_tests hnsw_core)

add_executable(run_bench bench/benchmark.cpp)
target_link_libraries(run_bench hnsw_core)

add_executable(run_bench_large bench/benchmark_large_scale.cpp)
target_link_libraries(run_bench_large hnsw_core)

pybind11_add_module(
    hnsw_cpp
    bindings.cpp
    src/hnsw.cpp
)