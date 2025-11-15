message(STATUS "add experiment !!!")

function(add_experiment dir_name dep_libs)
    file(GLOB_RECURSE SRC_FILES "${dir_name}/*.cpp")
    set(exe_name "experiment_${dir_name}")
    add_executable(${exe_name} ${SRC_FILES})
    target_include_directories(${exe_name} PRIVATE ${dir_name} ${ROOT_PATH})
    target_link_libraries(${exe_name} ${dep_libs})
    message(STATUS "add experiment [${exe_name}] !!!")
endfunction()

# add_experiment(xtensor_seg "inference;xtensor")

add_experiment(npy_data "inference")
