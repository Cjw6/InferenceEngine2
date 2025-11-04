cmake_minimum_required(VERSION 3.10.0)
project(InferenceEngine2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug) # 或者 Release, RelWithDebInfo, MinSizeRel 等
else()
    message("CMAKE_BUILD_TYPE is set to ${CMAKE_BUILD_TYPE}")
endif()


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(REGEX REPLACE "-O[0-9s]?" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REGEX REPLACE "-Ofast" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_compile_options("/utf-8")
endif()

# 设置生成目录
if(MSVC)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
message("CMAKE_CXX_FLAGS is set to ${CMAKE_CXX_FLAGS}")

set(ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

# options
option(ENABLE_CUDA "use cuda" ON)
# 使用cuda后应该禁用 address sanitize, 否则会发生错误，只对主机代码检查内存错误
option(ENABLE_ADDRESS_SANITIZER "use address sanitizer" OFF)

if(ENABLE_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

find_package(spdlog REQUIRED)
find_package(cpptrace REQUIRED)

if(ENABLE_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        message(STATUS "cuda found!!!!")
        add_definitions("-DUSE_CUDA")
    endif()
endif()

add_subdirectory(inference)
add_subdirectory(modelzoo)
add_subdirectory(test)
