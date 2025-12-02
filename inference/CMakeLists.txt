set(INFER_SRCS)
file(GLOB INFER_BASE_SRC "*.h" "*.hpp" "*.cpp" "tensor/*.h" "tensor/*.hpp" "tensor/*.cpp")
list(APPEND INFER_SRCS ${INFER_BASE_SRC})

if(ENABLE_ONNXRUTIME)
    file(GLOB ONNXRUNTIME_SRC "onnxruntime/*.h" "onnxruntime/*.hpp" "onnxruntime/*.cpp")
    list(APPEND INFER_SRCS ${ONNXRUNTIME_SRC})
endif()

if(ENABLE_TENSORRT)
    file(GLOB TENSORRT_SRC "tensorrt/*.h" "tensorrt/*.hpp" "tensorrt/*.cpp")
    list(APPEND INFER_SRCS ${TENSORRT_SRC})
endif()


message("[INFERENCE_ENGINE] INFERENCE_SRC: ${INFERENCE_SRC}")
message("[INFERENCE_ENGINE] INFERENCE_INC_DIRS: ${INFERENCE_INC_DIRS}")
message("[INFERENCE_ENGINE] INFERENCE_LINK_DIRS: ${INFERENCE_LINK_DIRS}")
message("[INFERENCE_ENGINE] INFERENCE_DEP_LIBS: ${INFERENCE_DEP_LIBS}")
message("[INFERENCE_ENGINE] INFERENCE_COMPILE_DEFS: ${INFERENCE_COMPILE_DEFS}")

add_library(inference STATIC ${INFER_SRCS})
target_include_directories(inference PUBLIC ${INFERENCE_INC_DIRS})
target_link_directories(inference PUBLIC ${INFERENCE_LINK_DIRS})
target_link_libraries(inference PUBLIC ${INFERENCE_DEP_LIBS})
target_compile_definitions(inference PUBLIC ${INFERENCE_COMPILE_DEFS})
