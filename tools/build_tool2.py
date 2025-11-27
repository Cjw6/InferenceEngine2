"""
example：
    conda activate my_cpp_env
    python ./tools/build_tool2.py -c linux_cmake_debug
    python ./tools/build_tool2.py -c linux_build_debug -- --target all
"""

import argparse
import subprocess
import os
import time


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
    parser.add_argument("-c", "--configure", dest="configure", type=str, default=None)
    parser.add_argument("extra_args", nargs="*")
    return parser


def config_map():
    conf_map = {
        "linux_cmake_debug": [
            "cmake",
            "-DCMAKE_BUILD_TYPE=Debug",
            "-S",
            ".",
            "-B",
            "./build/debug",
            "-G",
            "Ninja",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DENABLE_STACKTRACE=ON",
            "-DENABLE_ASSERTS=ON",
            "-DONNXRUNTIME_DIR=/home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1",
            "-DCUDA_PATH=/usr/local/cuda",
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
        ],
        "linux_build_debug": [
            "cmake",
            "--build",
            "./build/debug",
        ],
    }
    return conf_map


def run_process(cmd_args: list[str]):
    start_time = time.time()
    print(f"{TermColors.BLUE}run cmd: {cmd_args}{TermColors.END}")
    result = subprocess.run(cmd_args, cwd=os.getcwd(), text=True, check=False)
    if result.returncode != 0:
        tips = f"{TermColors.RED}run failed!!! {cmd_args=}, {result.returncode=} cost_time:{time.time() - start_time:.4f}s{TermColors.END}"
        raise RuntimeError(tips)

    print(
        f"{TermColors.GREEN}run success!!! {cmd_args=}, cost_time:{time.time() - start_time:.4f}s{TermColors.END}"
    )


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.configure is not None:
        cmd_args = config_map()[args.configure]
        if args.extra_args:
            cmd_args.extend(args.extra_args)
        run_process(cmd_args)
    else:
        print("Please specify a configure option")
