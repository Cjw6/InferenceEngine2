#pragma once

#include "inference/tensor/tensor.h"
#include <cpptoolkit/exception/exception.h>
#include <span>
#include <unordered_map>

namespace inference {

// 支持 ONNXRuntime
struct InferenceParams {
  // device
  DeviceType device_type = kCPU;
  int device_id = 0;

  // model
  std::string model_path;
  std::string weight_path;

  int log_level = -1;

  // max batch size for inference, only used when model is dynamic
  int max_batch_size = 1;

  // onnxruntime
  int intra_op_num_threads = 1;
  int inter_op_num_threads = 1;
  int graph_optimize_level = 0;
  int exe_mode = 0;

  std::unordered_map<std::string, std::string> ext_params;
};

InferenceParams GetDefaultOnnxRuntimeEngineParams();

// 未来迁移到这里， 支持 TensorRT
struct InferenceParamsV2 {
  std::string model_path;     // 模型文件的路径
  std::span<char> model_data; // 模型文件的数据
  DeviceType device_type = kCPU;
  int device_id = 0;
  int batch_size = 1;
  std::map<const char *, std::string> params;
};

InferenceParamsV2 GetDefaultTensorRTEngineParams();

} // namespace inference
