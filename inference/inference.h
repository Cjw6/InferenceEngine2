#include "inference/tensor/tensor.h"
#include <unordered_map>

namespace inference {

struct InferenceParams {
  DeviceType device_type = kCPU;
  int device_id = 0;
  std::string model_path;
  std::string weight_path;

  int intra_op_num_threads = 1;
  int inter_op_num_threads = 1;
  int graph_opt_level = 0;
  int exe_mode = 0;

  std::unordered_map<std::string, std::string> ext_params;
};

} // namespace inference
