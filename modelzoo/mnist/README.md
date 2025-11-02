# 推理引擎

## CPU 推理fp32

```bash
./build/debug/bin/exe_mnist --device cpu --model_path modelzoo/mnist/mnist.onnx
```

## GPU 推理fp32

```bash
./build/debug/bin/exe_mnist --device gpu --model_path modelzoo/mnist/mnist.onnx
```

## GPU 推理fp16

```bash
./build/debug/bin/exe_mnist --device gpu --model_path modelzoo/mnist/mnist_fp16.onnx
```
