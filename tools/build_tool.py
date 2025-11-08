import argparse
from operator import le
import subprocess
import os
import sys

"""
Usage:
    python build_tool.py <config> [args]
"""

EXTRA_ARGS: list[str] = []


def config_map():
    conf_map = {
        # cmake
        "cmake_debug": [
            "--task",
            "cmake",
            "--build_dir",
            "build/debug",
            "--enable_debug_mode",
            "--onnxruntime_dir",
            "/home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1",
        ],
        "cmake_release": [
            "--task",
            "cmake",
            "--build_dir",
            "build/release",
            "--build_type",
            "Release",
            "--onnxruntime_dir",
            "/home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1",
        ],
        "cmake_relwithdebinfo": [
            "--task",
            "cmake",
            "--build_dir",
            "build/relwithdebinfo",
            "--build_type",
            "RelWithDebInfo",
            "--onnxruntime_dir",
            "/home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1",
        ],
        # build
        "build_debug": [
            "--task",
            "build",
            "--build_dir",
            "build/debug",
        ],
        "build_release": [
            "--task",
            "build",
            "--build_dir",
            "build/release",
        ],
        "build_relwithdebinfo": [
            "--task",
            "build",
            "--build_dir",
            "build/relwithdebinfo",
        ],
        # cmake + build
        "debug": [
            "--task",
            "all",
            "--build_dir",
            "build/debug",
            "--enable_debug_mode",
            "--onnxruntime_dir",
            "/home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1",
        ],
        "release": [
            "--task",
            "all",
            "--build_dir",
            "build/release",
            "--build_type",
            "Release",
            "--onnxruntime_dir",
            "/home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1",
        ],
        "relwithdebinfo": [
            "--task",
            "all",
            "--build_dir",
            "build/relwithdebinfo",
            "--build_type",
            "RelWithDebInfo",
            "--onnxruntime_dir",
            "/home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1",
        ],
    }
    return conf_map


# Basic ANSI color codes
class TermColors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def build_argparser():
    parser = argparse.ArgumentParser()
    # task select
    parser.add_argument("--task", dest="task", type=str, default="all")

    # CMakeLists.txt pos
    parser.add_argument("--cmake_dir", dest="cmake_dir", type=str, default=None)

    # build binary pos
    parser.add_argument("--build_dir", dest="build_dir", type=str, default=None)

    # build config
    parser.add_argument("--build_type", dest="build_type", type=str, default="Debug")

    # build target
    parser.add_argument("--target", dest="target", type=str, default=None)

    # build verbose
    parser.add_argument("--verbose", dest="verbose", action="store_true")

    # install pos
    parser.add_argument("--install_dir", dest="install_dir", type=str, default=None)

    # debug option
    parser.add_argument(
        "--enable_debug_mode", dest="enable_debug_mode", action="store_true"
    )

    # onnxruntime
    parser.add_argument(
        "--onnxruntime_dir", dest="onnxruntime_dir", type=str, default=None
    )

    # tensorrt
    parser.add_argument("--tensorrt_dir", dest="tensorrt_dir", type=str, default=None)
    parser.add_argument("--nvcc", dest="nvcc", type=str, default=None)
    parser.add_argument("--cuda_dir", dest="cuda_dir", type=str, default=None)
    parser.add_argument("--cudnn_dir", dest="cudnn_dir", type=str, default=None)

    return parser


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

    if args.enable_debug_mode:
        arg_list = ["-DENABLE_STACKTRACE=ON", "-DENABLE_ASSERTS=ON"]
        cmd_args.extend(arg_list)

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

    if args.install_dir is not None:
        cmd_args.append("-DCMAKE_INSTALL_PREFIX={}".format(args.install_dir))

    print(f"{TermColors.GREEN}cmake cmd: {cmd_args}{TermColors.END}")
    result = subprocess.run(cmd_args, text=True)
    if result.returncode != 0:
        raise Exception(
            f"{TermColors.RED}cmake failed!!!{result.returncode=}{TermColors.END}"
        )

    print(f"{TermColors.BLUE}cmake success!!!{TermColors.END}")


def build_task_func(args):
    cmake_source_dir, build_dir = get_cmake_and_build_dir(args)
    cmd_args = [
        "ninja",
    ]
    if args.target is not None:
        cmd_args.append(args.target)
    if args.verbose:
        cmd_args.append("-v")

    print(f"{TermColors.GREEN}build cmd: {cmd_args}{TermColors.END}")
    result = subprocess.run(cmd_args, text=True, cwd=build_dir)
    if result.returncode != 0:
        raise Exception("build failed!!!. {}".format(result.returncode))

    print(f"{TermColors.BLUE}build success!!!{TermColors.END}")


def install_task_func(args):
    cmake_source_dir, build_dir = get_cmake_and_build_dir(args)
    cmd_args = [
        "ninja",
        "install",
    ]
    if args.verbose:
        cmd_args.append("-v")

    print(f"{TermColors.GREEN}install cmd: {cmd_args}{TermColors.END}")
    result = subprocess.run(cmd_args, text=True, cwd=build_dir)
    if result.returncode != 0:
        raise Exception("install failed!!!. {}".format(result.returncode))

    print(f"{TermColors.BLUE}install success!!!{TermColors.END}")


def all_task_func(args):
    cmake_task_func(args)
    build_task_func(args)
    if args.install_dir is not None:
        install_task_func(args)


def run_build_pipeline():
    config = sys.argv[1]
    configure = config_map()[config]
    if len(sys.argv) > 2:
        configure.extend(sys.argv[2:])

    print(f"{configure=}")

    parser = build_argparser()
    args = parser.parse_args(configure)

    task_map()[args.task](args)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_tool.py <config> [args]")
        sys.exit(1)

    run_build_pipeline()
