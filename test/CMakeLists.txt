file(GLOB_RECURSE TEST_SRC "test_main.cpp" "test_*.cpp")
add_executable(test_inference ${TEST_SRC})
target_link_libraries(test_inference  ${ALL_MODEL_ZOO_LIBS} common inference gtest)
