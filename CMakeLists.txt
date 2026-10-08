cmake_minimum_required(VERSION 3.25)
project(ortools_test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ortools REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main ortools::ortools)

add_executable(bench_coeff_changes bench_coeff_changes.cpp)
target_link_libraries(bench_coeff_changes ortools::ortools)

add_executable(bench_coeff_changes_mpsolver bench_coeff_changes_mpsolver.cpp)
target_link_libraries(bench_coeff_changes_mpsolver ortools::ortools)

add_executable(incremental_bench incremental_bench.cpp)
target_link_libraries(incremental_bench ortools::ortools)

add_executable(bfloat16_demo bfloat16_demo.cpp)
target_include_directories(bfloat16_demo PRIVATE $ENV{HOME}/code/ortools_deps/include/eigen3)
