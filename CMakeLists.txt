cmake_minimum_required(VERSION 3.10)
project(simple-tad VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64"))
    message(FATAL_ERROR "Only AMD64 or x86_64 is supported.")
endif()

option(BUILD_MAIN "Build main." ON)
option(BUILD_PYBIND11 "Build pybind11." OFF)
option(BUILD_TESTS "Build tests." OFF)
option(BUILD_BENCHMARKS "Build benchmarks." OFF)

set(CMAKE_CXX_FLAGS "-Wall -std=c++17 -O3 -mavx2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

if (BUILD_MAIN)
    add_executable(simple-tad ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
    target_include_directories(simple-tad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/lib ${CMAKE_CURRENT_SOURCE_DIR}/libsimdpp)
endif()

if (BUILD_PYBIND11)
    add_subdirectory(pybind11)
    pybind11_add_module(simple_tad ${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper.cpp)
    target_include_directories(simple_tad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/lib ${CMAKE_CURRENT_SOURCE_DIR}/libsimdpp)
    
    set_target_properties(
        simple_tad PROPERTIES 
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/simple-tad
    )
endif()

if (BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    add_executable(simple-tad-tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/test.cpp)
    target_include_directories(simple-tad-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/lib ${CMAKE_CURRENT_SOURCE_DIR}/libsimdpp)
    
    target_link_libraries(
        simple-tad-tests
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(simple-tad-tests)
endif()

if (NOT BUILD_TESTS AND BUILD_BENCHMARKS)
    message(FATAL_ERROR "BUILD_TESTS must be enabled to build benchmarks.")
endif()

if (BUILD_BENCHMARKS)
    include(FetchContent)
    FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.7.1.zip
    )

    FetchContent_MakeAvailable(googlebenchmark)

    add_executable(simple-tad-benchmarks ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/benchmark.cpp)
    target_include_directories(simple-tad-benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/lib ${CMAKE_CURRENT_SOURCE_DIR}/libsimdpp)
    
    target_link_libraries(
        simple-tad-benchmarks
        benchmark::benchmark
    )
endif()
