cmake_minimum_required(VERSION 3.16)
project(qicert_cpp LANGUAGES CXX)

# Phase-2 contraction-kernel skeleton. Builds a static lib + self-test.
# No timing targets, no benchmark binaries — measurement lives in
# bench/latency_microbench.py against the shared fixture (cpp/README.md).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(qicert_cpp STATIC src/tt_matvec.cpp)
target_include_directories(qicert_cpp PUBLIC include)

add_executable(qicert_cpp_selftest src/selftest_main.cpp)
target_link_libraries(qicert_cpp_selftest PRIVATE qicert_cpp)
