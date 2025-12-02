# InferenceEngine

## how to build

### build all(onnxruntime + cuda + tensorrt)  debug

```bash
conda activate my_cpp_env
python ./tools/build_tool2.py -c linux_cmake_debug
python ./tools/build_tool2.py -c linux_build_debug -- --target all
```
## build tensorrt debug

```bash
conda activate my_cpp_env
python ./tools/build_tool2.py   -c  linux_cmake_debug_trt
python ./tools/build_tool2.py   -c  linux_build_debug_trt
```

## set env
- solve: FAIL : Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.9: cannot open shared object file: No such file or directory

```bash
export LD_LIBRARY_PATH=/home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1/lib:/home/cjw/lib/cudnn-linux-x86_64-9.14.0.64_cuda13-archive/lib
```
