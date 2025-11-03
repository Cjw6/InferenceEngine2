find_package(OpenCV REQUIRED)
file(GLOB_RECURSE COMMON_SRC "common/*.cpp" "common/*.h" "common/*.hpp")
add_library(common ${COMMON_SRC})
target_include_directories(common PRIVATE ${ROOT_PATH} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(common PUBLIC inference ${OpenCV_LIBS})


function(add_compie_exe dir_name inc_dirs dep_libs)
    file(GLOB_RECURSE SRC_FILES "${dir_name}/*.cpp" "${dir_name}/*.h" "${dir_name}/*.hpp")
    set(exe_name "exe_${dir_name}")
    add_executable(${exe_name} ${SRC_FILES})
    target_include_directories(${exe_name} PRIVATE ${ROOT_PATH} ${inc_dirs})
    target_link_libraries(${exe_name} ${dep_libs})
    message(STATUS "add exe : ${exe_name}")
endfunction()

# 这里使用分号分隔库名表示数组
add_compie_exe(mnist "" "common;gflags")
add_compie_exe(mnist_dynamic "" "common;gflags")
