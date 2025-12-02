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
# 使用cuda后应该禁用 address sanitize, 否则会发生错误，只对主机代码检查内存错误
option(ENABLE_ADDRESS_SANITIZER "use address sanitizer" OFF)
option(ENABLE_STACKTRACE "use stacktrace exception" OFF)
option(ENABLE_ASSERTS "use asserts" OFF)
option(ENABLE_CUDA "use cuda" ON)
option(ENABLE_XTENSOR "use onnxruntime" OFF)

# inference backend
option(ENABLE_ONNXRUNTIME "use onnxruntime" OFF)
option(ENABLE_TENSORRT "use tensorrt" OFF)

option(ENABLE_EXPERIMENT "use experiment" OFF)
option(ENABLE_MODEL_ZOO "use model zoo" OFF)
option(ENABLE_TEST "use test" OFF)

message(STATUS "ENABLE_STACKTRACE: ${ENABLE_STACKTRACE}")
message(STATUS "ENABLE_ASSERTS: ${ENABLE_ASSERTS}")
message(STATUS "ENABLE_CUDA: ${ENABLE_CUDA}")
message(STATUS "ENABLE_XTENSOR: ${ENABLE_XTENSOR}")
message(STATUS "ENABLE_ONNXRUNTIME: ${ENABLE_ONNXRUNTIME}")
message(STATUS "ENABLE_TENSORRT: ${ENABLE_TENSORRT}")
message(STATUS "ENABLE_EXPERIMENT: ${ENABLE_EXPERIMENT}")
message(STATUS "ENABLE_MODEL_ZOO: ${ENABLE_MODEL_ZOO}")
message(STATUS "ENABLE_TEST: ${ENABLE_TEST}")

if(ENABLE_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

if(ENABLE_STACKTRACE)
    set(CPP_TK_ENABLE_CPPTRACE ON)
endif()

if(ENABLE_ASSERTS)
    set(CPP_TK_ENABLE_ASSERT ON)
endif()

if(ENABLE_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        message(STATUS "cuda found!!!!")
        add_definitions("-DUSE_CUDA")
        list(APPEND INFERENCE_COMPILE_DEFS "INFERENCE_USE_CUDA")
        list(APPEND INFERENCE_INC_DIRS ${CUDA_INCLUDE_DIRS})
        list(APPEND INFERENCE_LINK_DIRS ${CUDA_LIBRARY_DIRS})
        list(APPEND INFERENCE_DEP_LIBS ${CUDA_LIBRARIES})
    endif()

    set(CPP_TK_ENABLE_CUDA ON)
endif()

if(ENABLE_XTENSOR)
    # xtensor
    find_package(xtl REQUIRED)
    find_package(xtensor REQUIRED)
    list(APPEND INFERENCE_DEP_LIBS xtensor)
    list(APPEND INFERENCE_COMPILE_DEFS "USE_XTENSOR")
endif()

add_subdirectory(3rd/cpptoolkit)
list(APPEND INFERENCE_INC_DIRS ${ROOT_PATH}/3rd/cpptoolkit)
list(APPEND INFERENCE_DEP_LIBS cpptoolkit)

if(ENABLE_ONNXRUNTIME)
    message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
    list(APPEND INFERENCE_INC_DIRS ${ONNXRUNTIME_DIR}/include)
    list(APPEND INFERENCE_LINK_DIRS ${ONNXRUNTIME_DIR}/lib)
    list(APPEND INFERENCE_DEP_LIBS onnxruntime)
    list(APPEND INFERENCE_COMPILE_DEFS "INFER_USE_ONNXRUNTIME")
endif()


if(ENABLE_TENSORRT AND ENABLE_CUDA)
    message(STATUS "TENSORRT_DIR: ${TENSORRT_DIR}")
    list(APPEND INFERENCE_INC_DIRS ${TENSORRT_DIR}/include)
    list(APPEND INFERENCE_LINK_DIRS ${TENSORRT_DIR}/lib)
    list(APPEND INFERENCE_DEP_LIBS "nvinfer")
    list(APPEND INFERENCE_COMPILE_DEFS "INFER_USE_TENSORRT")
endif()

add_subdirectory(inference)

if(ENABLE_MODEL_ZOO)
    add_subdirectory(modelzoo)
endif()

if(ENABLE_TEST)
    add_subdirectory(test)
endif()

if(ENABLE_EXPERIMENT)
    add_subdirectory(experiment)
endif()
