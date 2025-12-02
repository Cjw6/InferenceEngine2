#include "inference/inference.h"

namespace inference {

InferenceParams GetDefaultOnnxRuntimeEngineParams() {
  inference::InferenceParams params;
  params.log_level = 2;
  return params;
}

InferenceParamsV2 GetDefaultTensorRTEngineParams() {
  inference::InferenceParamsV2 params;
  params.device_type = kGPU;
  params.device_id = 0;
  return params;
}

} // namespace inference
