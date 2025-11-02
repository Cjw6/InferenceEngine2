#include <gflags/gflags.h>

#include "inference/onnxruntime/onnxruntime.h"
#include "inference/utils/log.h"

DEFINE_string(model_path, "modelzoo/mnist/mnist.onnx",
              "Path to the ONNX model file.");
DEFINE_string(device, "cpu", "Device to run the inference on.");
DEFINE_int32(device_id, 0, "Number of threads to use.");

inference::InferenceParams GetParams() {
  inference::InferenceParams params;
  if (FLAGS_device == "cpu") {
    params.device_type = inference::DeviceType::kCPU;
  } else if (FLAGS_device == "gpu") {
    params.device_type = inference::DeviceType::kGPU;
  }
  params.device_id = FLAGS_device_id;
  params.model_path = FLAGS_model_path;
  return params;
}

int main(int argc, char *argv[]) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  LogInit();
  LOG_INFO("use model_path: {}", FLAGS_model_path);
  inference::InferenceParams params = GetParams();
  inference::OnnxRuntimeEngine engine;
  int ret = engine.Init(params);
  if (ret != 0) {
    LOG_ERROR("Failed to init engine with error: {}", ret);
    return -1;
  }
  engine.Run();
}
