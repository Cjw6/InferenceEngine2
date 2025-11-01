function(add_compie_exe dir_name inc_dirs dep_libs)
    file(GLOB_RECURSE SRC_FILES "${dir_name}/*.cpp" "${dir_name}/*.h" "${dir_name}/*.hpp")
    set(exe_name "exe_${dir_name}")
    add_executable(${exe_name} ${SRC_FILES})
    target_include_directories(${exe_name} PRIVATE ${ROOT_PATH} ${inc_dirs})
    target_link_libraries(${exe_name} ${dep_libs})
    message(STATUS "add exe : ${exe_name}")
endfunction()


# 这里使用分号分隔库名
add_compie_exe(mnist "" "inference;gflags")
