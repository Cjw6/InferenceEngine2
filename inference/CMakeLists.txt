file(GLOB_RECURSE ONNXRUNTIME_SRC
    "*.h" "*.hpp" "*.cpp"
)

add_library(inference STATIC ${ONNXRUNTIME_SRC})
target_include_directories(inference PUBLIC ${CUDA_INCLUDE_DIRS} ${ONNXRUNTIME_DIR}/include ${ROOT_PATH})
target_link_directories(inference PUBLIC ${ONNXRUNTIME_DIR}/lib)
target_link_libraries(inference PUBLIC onnxruntime ${CUDA_LIBRARIES} spdlog::spdlog cpptrace::cpptrace)
