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

set(INFERENCE_INC_DIRS ${ROOT_PATH})
set(INFERENCE_LINK_DIRS)
set(INFERENCE_DEP_LIBS)
set(INFERENCE_COMPILE_DEFS)

# options
option(ENABLE_EXPERIMENT "use experiment" OFF)
# 使用cuda后应该禁用 address sanitize, 否则会发生错误，只对主机代码检查内存错误
option(ENABLE_ADDRESS_SANITIZER "use address sanitizer" OFF)
option(ENABLE_STACKTRACE "use stacktrace exception" OFF)
option(ENABLE_ASSERTS "use asserts" OFF)
option(ENABLE_CUDA "use cuda" ON)
option(ENABLE_XTENSOR "use onnxruntime" OFF)

message(STATUS "ENABLE_EXPERIMENT: ${ENABLE_EXPERIMENT}")
message(STATUS "ENABLE_STACKTRACE: ${ENABLE_STACKTRACE}")
message(STATUS "ENABLE_ASSERTS: ${ENABLE_ASSERTS}")
message(STATUS "ENABLE_CUDA: ${ENABLE_CUDA}")
message(STATUS "ENABLE_XTENSOR: ${ENABLE_XTENSOR}")

add_subdirectory(3rd/cpptoolkit)
list(APPEND INFERENCE_INC_DIRS ${ROOT_PATH}/3rd/cpptoolkit)
list(APPEND INFERENCE_DEP_LIBS cpptoolkit)

# if(ENABLE_ADDRESS_SANITIZER)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
# endif()

# if(ENABLE_STACKTRACE)
#     find_package(cpptrace REQUIRED)
#     add_definitions("-DENABLE_CPPTRACE_STACKTRACE")
#     list(APPEND INFERENCE_DEP_LIBS cpptrace::cpptrace)
# endif()

# if(ENABLE_ASSERTS)
#     add_definitions("-DENABLE_ASSERTS")
# endif()

# find_package(spdlog REQUIRED)
# list(APPEND INFERENCE_DEP_LIBS spdlog::spdlog)

if(ENABLE_XTENSOR)
    # xtensor
    find_package(xtl REQUIRED)
    find_package(xtensor REQUIRED)
    list(APPEND INFERENCE_DEP_LIBS xtensor)
    list(APPEND INFERENCE_COMPILE_DEFS "USE_XTENSOR")
endif()

if(ENABLE_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        message(STATUS "cuda found!!!!")
        add_definitions("-DUSE_CUDA")
        list(APPEND INFERENCE_INC_DIRS ${CUDA_INCLUDE_DIRS})
        list(APPEND INFERENCE_LINK_DIRS ${CUDA_LIBRARY_DIRS})
        list(APPEND INFERENCE_DEP_LIBS ${CUDA_LIBRARIES})
    endif()
endif()

# onnxruntime
list(APPEND INFERENCE_INC_DIRS ${ONNXRUNTIME_DIR}/include)
list(APPEND INFERENCE_LINK_DIRS ${ONNXRUNTIME_DIR}/lib)
list(APPEND INFERENCE_DEP_LIBS onnxruntime)


add_subdirectory(inference)
add_subdirectory(modelzoo)
add_subdirectory(test)
if(ENABLE_EXPERIMENT)
    add_subdirectory(experiment)
endif()
