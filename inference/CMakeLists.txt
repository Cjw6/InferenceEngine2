file(GLOB_RECURSE ONNXRUNTIME_SRC
    "*.h" "*.hpp" "*.cpp"
)

add_library(inference STATIC ${ONNXRUNTIME_SRC})
target_include_directories(inference PUBLIC ${INFERENCE_INC_DIRS})
target_link_directories(inference PUBLIC ${INFERENCE_LINK_DIRS})
target_link_libraries(inference PUBLIC ${INFERENCE_DEP_LIBS})
