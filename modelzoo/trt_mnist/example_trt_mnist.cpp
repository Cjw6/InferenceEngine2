#include "inference/inference.h"
#include "inference/tensorrt/tensorrt.h"
#include "modelzoo/common/img_common.hpp"
#include <cpptoolkit/log/log.h>

int main() {
  cpptoolkit::LogConfig log_config;
  log_config.console_log_level = cpptoolkit::LogLevel::kDebug;
  // log_config.file_log_level = cpptoolkit::LogLevel::kDebug;
  cpptoolkit::LogInit();
  auto infer_params = inference::GetDefaultTensorRTEngineParams();
  infer_params.model_path = "modelzoo/data/mnist/mnist.fp16.engine";
  inference::TensorRTEngine engine;
  if (engine.Init(infer_params) != 0) {
    LOG_ERROR("Init engine failed.");
    return -1;
  }

  if (!engine.IsReady()) {
    LOG_ERROR("Engine is not ready.");
    return -1;
  }

  LOG_INFO("Model info: {}", engine.DumpModelInfo());
  auto ret = engine.Warmup();
  if (ret != 0) {
    LOG_ERROR("Warmup failed. {}", ret);
    return -1;
  }

  std::string in_name = engine.GetInputNodeNames().front();
  std::string out_name = engine.GetOutputNodeNames().front();

  auto in_pointer = engine.GetInputTensors().at(in_name);

  std::string test_img_path = "modelzoo/data/mnist/3.pgm";
  std::vector<uint8_t> fileData(28 * 28);
  imgutils::readPGMFile(test_img_path, fileData.data(), 28, 28);

  // for (int i = 0; i < 28 * 28; i++) {
  // ((float *)in_pointer.p)[i] = 1.0f - (float)(((uint8_t *)in_pointer.p)[i]) /
  // 255.0;
  // }

  float *hostDataBuffer = static_cast<float *>(in_pointer.p);
  std::transform(fileData.begin(), fileData.end(), hostDataBuffer,
                 [](uint8_t x) { return 1.0 - static_cast<float>(x / 255.0); });

  engine.CopyInputToDevice();
  ret = engine.Run();
  if (ret != 0) {
    LOG_ERROR("Run failed. {}", ret);
    return -1;
  }
  engine.CopyOutputToHost();

  auto out_pointer = engine.GetOutputTensors().at(out_name);
  auto p = (float *)out_pointer.p;
  float *fdata = (float *)out_pointer.p;
  for (int i = 0; i < out_pointer.shape.at(1); i++) {
    LOG_INFO("{}", fdata[i]);
  }

  return 0;
}