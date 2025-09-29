cmake_minimum_required(VERSION 3.10)
project(trace_replay)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../CacheLib_2023/cachelib/cmake")
set(CMAKE_PREFIX_PATH "/proj/cac101-PG0/CacheLib_2023/opt/cachelib/" ${CMAKE_PREFIX_PATH})
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(Eigen3 REQUIRED)
find_package(mlpack REQUIRED)
find_package(Sodium REQUIRED)
find_package(cachelib REQUIRED)

add_executable(trace_replay trace_replay.cpp)
target_link_libraries(trace_replay PRIVATE
    cachelib
    mlpack::mlpack
    Eigen3::Eigen
    Boost::serialization
    ${sodium_LIBRARY}
)

target_include_directories(trace_replay PRIVATE ${sodium_INCLUDE_DIR})
