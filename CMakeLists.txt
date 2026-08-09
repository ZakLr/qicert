cmake_minimum_required(VERSION 3.20)
project(qicert_kernels LANGUAGES CXX)

option(QICERT_BUILD_CPP "Build the C++/CUDA-Q kernel stubs" ON)
find_package(CUDAQ QUIET)

if(QICERT_BUILD_CPP)
  add_library(qicert_kernels STATIC
    src/tt_cross.cpp
    src/pauli_family.cpp
    src/iqae.cpp
    src/shadow_monitor.cpp
  )
  target_include_directories(qicert_kernels PUBLIC include)
  target_compile_features(qicert_kernels PUBLIC cxx_std_17)
  if(CUDAQ_FOUND)
    message(STATUS "CUDA-Q found — linking simulators/noise models into kernels.")
    target_link_libraries(qicert_kernels PUBLIC CUDAQ::cudaq)
  else()
    message(STATUS "CUDA-Q not found — kernels build as plain C++ stubs (Q19 pin pending).")
  endif()
endif()
