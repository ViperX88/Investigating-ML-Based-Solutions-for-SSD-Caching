cmake_minimum_required(VERSION 3.16)  # or whatever the project minimum really is
project(trace_replay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# If CacheLib installs its own package config file, just point CMAKE_PREFIX_PATH
# to the installation prefix (do this on the command line: -DCMAKE_PREFIX_PATH=/path/to/install)
find_package(cachelib REQUIRED CONFIG)

# Sodium ships a modern imported target named Sodium::Sodium
find_package(Sodium REQUIRED)

add_executable(trace_replay
    trace_replay.cpp
)

target_link_libraries(trace_replay
    PRIVATE
        cachelib::cachelib     # or whatever target CacheLib exports
        Sodium::Sodium
)

# Include directories, compile defs, etc. should be propagated through the imported targets;
# only add extra directories if you really need custom headers:
# target_include_directories(trace_replay PRIVATE "${PROJECT_SOURCE_DIR}/include")

