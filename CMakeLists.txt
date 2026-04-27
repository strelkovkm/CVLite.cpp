cmake_minimum_required(VERSION 3.10)
project(CVLite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CVLITE_BUILD_TESTS "Build CVLite tests" ON)
option(CVLITE_BUILD_EXAMPLES "Build CVLite examples" ON)
option(CVLITE_BUILD_BENCHMARKS "Build CVLite benchmarks" ON)

add_subdirectory(src)

if (CVLITE_BUILD_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()

if (CVLITE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if (CVLITE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
