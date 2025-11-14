# InferenceEngine

## how to build

### activate conda env

```bash
conda activate my_cpp_env

```

### build release
```bash
python ./tools/build_tool.py  release
```

### build debug

```bash
python ./tools/build_tool.py debug
```

### build release with debug_info
```bash
python ./tools/build_tool.py  relwithdebinfo
```

## set env
- solve: FAIL : Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.9: cannot open shared object file: No such file or directory

```bash
export LD_LIBRARY_PATH=/home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1/lib:/home/cjw/lib/cudnn-linux-x86_64-9.14.0.64_cuda13-archive/lib
```
