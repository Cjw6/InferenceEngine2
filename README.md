# InferenceEngine

## how to build

### build debug

```bash
conda activate my_cpp_env
python ./tools/build_tool.py  \
--task all \
--build_dir build/debug \
--enable_debug_mode \
--onnxruntime_dir /home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1
```

### build debug with debug_info
```bash
conda activate my_cpp_env
python ./tools/build_tool.py  \
--task all \
--build_dir build/relwithdebinfo \
--build_type Relwithdebinfo \
--onnxruntime_dir /home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1
```

### build release
```bash
conda activate my_cpp_env
python ./tools/build_tool.py  \
--task all \
--build_dir build/release \
--build_type Release \
--onnxruntime_dir /home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1
```


## FAIL : Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.9: cannot open shared object file: No such file or directory

```bash
export LD_LIBRARY_PATH=/home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1/lib:/home/cjw/lib/cudnn-linux-x86_64-9.14.0.64_cuda13-archive/lib
```
