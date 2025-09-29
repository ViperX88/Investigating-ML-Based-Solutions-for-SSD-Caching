cmake_minimum_required(VERSION 3.10)
project(trace_replay)
set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../CacheLib/cachelib/cmake")
set(CMAKE_PREFIX_PATH "/proj/cac101-PG0/CacheLib/opt/cachelib/" ${CMAKE_PREFIX_PATH})
find_package(cachelib REQUIRED)
find_package(Sodium REQUIRED)
find_package(LightGBM CONFIG QUIET)
if (LightGBM_FOUND)
  set(LIGHTGBM_LIB LightGBM::LightGBM)
else()
  find_library(LIGHTGBM_LIB NAMES lightgbm _lightgbm lib_lightgbm)
  if (NOT LIGHTGBM_LIB)
    message(FATAL_ERROR "LightGBM shared library not found. Install liblightgbm-dev or build from source and ensure lib>  endif()
  include_directories(/usr/include/LightGBM)
endif()
find_library(JSONCPP_LIB NAMES jsoncpp)
if (NOT JSONCPP_LIB)
  message(FATAL_ERROR "JsonCpp not found. Install libjsoncpp-dev.")
endif()
find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
  set(OPENMP_LIB OpenMP::OpenMP_CXX)
endif()
add_executable(trace_replay trace_replay.cpp)
target_link_libraries(trace_replay PRIVATE
  cachelib
  ${sodium_LIBRARY}
  ${LIGHTGBM_LIB}
  ${JSONCPP_LIB}
  ${OPENMP_LIB}
)

target_include_directories(trace_replay PRIVATE
  ${sodium_INCLUDE_DIR}
  /usr/include            
  /usr/include/LightGBM   
)
