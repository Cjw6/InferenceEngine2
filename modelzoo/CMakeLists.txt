find_package(OpenCV REQUIRED)
file(GLOB_RECURSE COMMON_SRC "common/*.cpp" "common/*.h" "common/*.hpp")
add_library(common ${COMMON_SRC})
target_include_directories(common PRIVATE ${ROOT_PATH} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(common PUBLIC inference ${OpenCV_LIBS})

set(exe_prefix "example_")
set(lib_name_prefix "modelzoo_")
set(default_dep_libs "common;inference")

set(ALL_MODEL_ZOO_LIBS)

function(add_compie_static_lib dir_name inc_dirs dep_libs)
    file(GLOB_RECURSE SRC_FILES "${dir_name}/*.cpp")
    list(FILTER SRC_FILES EXCLUDE REGEX "example_*")
    # file(GLOB_RECURSE COMPILE_FILES "${dir_name}/*.cpp")
    if(SRC_FILES)
        file(GLOB_RECURSE HEADER_FILES "${dir_name}/*.h" "${dir_name}/*.hpp")
        list(APPEND SRC_FILES ${HEADER_FILES})
        # message(STATUS "SRC_FILES: ${SRC_FILES}")
        set(lib_name "${lib_name_prefix}${dir_name}")
        add_library(${lib_name} STATIC ${SRC_FILES})
        target_include_directories(${lib_name} PUBLIC ${ROOT_PATH} ${inc_dirs})
        target_link_libraries(${lib_name} PUBLIC ${dep_libs} ${default_dep_libs})
        message(STATUS "add lib: [${lib_name}] success, src_files: ${SRC_FILES}")
        list(APPEND ALL_MODEL_ZOO_LIBS ${lib_name})
    else()
        message(STATUS "add lib: [dirname:${dir_name}] fail")
    endif()
endfunction()

function(add_compie_exe dir_name inc_dirs dep_libs)
    file(GLOB_RECURSE SRC_FILES "${dir_name}/*.cpp" "${dir_name}/*.h" "${dir_name}/*.hpp")
    list(FILTER SRC_FILES INCLUDE REGEX "example_*")
    if(SRC_FILES)
        set(exe_name "${exe_prefix}${dir_name}")
        add_executable(${exe_name} ${SRC_FILES})
        target_include_directories(${exe_name} PRIVATE ${ROOT_PATH} ${inc_dirs})
        target_link_libraries(${exe_name} ${dep_libs} ${default_dep_libs})
        message(STATUS "add_exe: [${exe_name}] success, src_files: ${SRC_FILES}")
    else()
        message(STATUS "add_exe: [dirname:${dir_name}] fail")
    endif()
endfunction()

# 编译一个目录下的所有源文件为静态库和可执行文件 
# 静态库名称为 lib_name_prefix + dir_name
# 可执行文件名称为 exe_prefix + dir_name
# inc_dirs: 头文件目录
# dep_libs: 静态库依赖
# exe_dep_libs: 可执行文件依赖
function(add_model dir_name inc_dirs dep_libs exe_dep_libs)
    add_compie_static_lib("${dir_name}" "${inc_dirs}" "${dep_libs}")
    if(TARGET ${lib_name_prefix}${dir_name})
        list(APPEND exe_dep_libs ${lib_name_prefix}${dir_name})
    endif()
    add_compie_exe("${dir_name}" "${inc_dirs}" "${exe_dep_libs}")
endfunction()

# 这里使用分号分隔库名表示数组
add_model(mnist "" "" "gflags")
add_model(mnist_dynamic "" "" "")
add_model(mnist_add_process "" "" "")
add_model(yolov8n "" "" "gflags")
