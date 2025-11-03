#include "inference/tensor/tensor.h"
#include <unordered_map>

namespace inference {

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
  int graph_opt_level = 0;
  int exe_mode = 0;

  std::unordered_map<std::string, std::string> ext_params;
};

} // namespace inference
