# InferenceEngine

## how to build
```shell
conda activate my_cpp_env
python ./tools/build_tool.py  \
--task cmake \
--build_dir build/debug \
--onnxruntime_dir /home/cjw/lib/onnxruntime-linux-x64-gpu-1.23.1
```