cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
project(neuronet)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
add_subdirectory(src)

################################
# GTest
################################
add_subdirectory (gtest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR} ${PROJECT_SOURCE_DIR})
# Add test cpp file
add_executable( unittests ${PROJECT_SOURCE_DIR}/tests/test.cpp)
# Link test executable against gtest & gtest_main
target_link_libraries(unittests gtest gtest_main)
add_test( unittests unittests )

