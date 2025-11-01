#include <filesystem>
#include <gflags/gflags.h>

#include "inference/onnxruntime/onnxruntime.h"
#include "inference/utils/log.h"

DEFINE_string(model_path, "modelzoo/mnist/mnist.onnx", "Path to the ONNX model file.");

int main(int argc, char *argv[]) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  LogInit();
  LOG_INFO("use model_path: {}", FLAGS_model_path);
  inference::OnnxRuntimeEngine engine;
  engine.Init(FLAGS_model_path.c_str());
  engine.Run();
}
