import argparse
import subprocess
import os


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", dest="task", type=str, default="all")
    parser.add_argument("--cmake_dir", dest="cmake_dir", type=str, default=None)
    parser.add_argument("--build_dir", dest="build_dir", type=str, default=None)
    parser.add_argument("--build_type", dest="build_type", type=str, default="Debug")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--install_dir", dest="install_dir", type=str, default=None)
    parser.add_argument("--target", dest="target", type=str, default=None)
    # advance option
    parser.add_argument(
        "--onnxruntime_dir", dest="onnxruntime_dir", type=str, default=None
    )
    parser.add_argument("--tensorrt_dir", dest="tensorrt_dir", type=str, default=None)
    parser.add_argument("--nvcc", dest="nvcc", type=str, default=None)
    parser.add_argument("--cuda_dir", dest="cuda_dir", type=str, default=None)
    parser.add_argument("--cudnn_dir", dest="cudnn_dir", type=str, default=None)
    return parser.parse_args()


def task_map():
    task_2_func = {
        "cmake": cmake_task_func,
        "build": build_task_func,
        "install": install_task_func,
        "all": all_task_func,
    }
    return task_2_func


def null_task_func():
    print("null task")
    pass


def get_cmake_and_build_dir(args):
    if args.cmake_dir is not None:
        cmake_source_dir = args.cmake_dir
    else:
        cmake_source_dir = os.getcwd()

    if args.build_dir is not None:
        if os.path.isabs(args.build_dir):
            build_dir = args.build_dir
        else:
            build_dir = os.path.join(os.getcwd(), args.build_dir)
    else:
        build_dir = os.path.join(cmake_source_dir, "build")

    return cmake_source_dir, build_dir


def cmake_task_func(args):
    cmake_source_dir, build_dir = get_cmake_and_build_dir(args)
    cmd_args = [
        "cmake",
        "-DCMAKE_BUILD_TYPE={}".format(args.build_type),
        "-G",
        "Ninja",
        "-S",
        cmake_source_dir,
        "-B",
        build_dir,
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]

    if args.install_dir is not None:
        cmd_args.append("-DCMAKE_INSTALL_PREFIX={}".format(args.install_dir))

    if args.onnxruntime_dir:
        cmd_args.append("-DONNXRUNTIME_DIR={}".format(args.onnxruntime_dir))

    if args.tensorrt_dir:
        cmd_args.append("-DTENSORRT_DIR={}".format(args.tensorrt_dir))

    if args.nvcc is not None:
        cmd_args.append("-DCMAKE_CUDA_COMPILER={}".format(args.nvcc))

    if args.cuda_dir is not None:
        cmd_args.append("-DCUDA_DIR={}".format(args.cuda_dir))

    if args.cudnn_dir is not None:
        cmd_args.append("-DCUDNN_DIR={}".format(args.cudnn_dir))

    result = subprocess.run(cmd_args, text=True)
    if result.returncode != 0:
        raise Exception("cmake failed!!!. {}".format(result.returncode))


def build_task_func(args):
    cmake_source_dir, build_dir = get_cmake_and_build_dir(args)
    cmd_args = [
        "ninja",
    ]
    if args.target is not None:
        cmd_args.append(args.target)
    if args.verbose:
        cmd_args.append("-v")
    print("build cmd: {}".format(" ".join(cmd_args)))
    result = subprocess.run(cmd_args, text=True, cwd=build_dir)
    if result.returncode != 0:
        raise Exception("build failed!!!. {}".format(result.returncode))


def install_task_func(args):
    cmake_source_dir, build_dir = get_cmake_and_build_dir(args)
    cmd_args = [
        "ninja",
        "install",
    ]
    if args.verbose:
        cmd_args.append("-v")
    result = subprocess.run(cmd_args, text=True, cwd=build_dir)
    if result.returncode != 0:
        raise Exception("install failed!!!. {}".format(result.returncode))


def all_task_func(args):
    cmake_task_func(args)
    build_task_func(args)
    if args.install_dir is not None:
        install_task_func(args)


if __name__ == "__main__":
    args = build_argparser()
    task_map()[args.task](args)
