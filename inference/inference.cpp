#include "inference/inference.h"

namespace inference {

InferenceParams GetDefaultOnnxRuntimeEngineParams() {
  inference::InferenceParams params;
  params.log_level = 2;
  return params;
}

} // namespace inference
