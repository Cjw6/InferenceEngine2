#include "inference/onnxruntime/onnxruntime.h"
#include "inference/utils/elapse_time.hpp"
#include "inference/utils/log.h"
#include "modelzoo/common/img_common.hpp"
#include <gtest/gtest.h>

namespace {

const std::string fp32_model_path = "modelzoo/yolov8n/yolov8n.onnx";

void RunYoloV8Model(const std::string &model_path,
                    inference::DeviceType device_type) {

  ::inference::InferenceParams params;
  params.device_type = device_type;
  params.model_path = model_path;
  params.log_level = 2;

  ::inference::OnnxRuntimeEngine engine;
  int ret = engine.Init(params);
  ASSERT_TRUE(ret == 0) << "Failed to init engine: " << ret;

  cpputils::ElapseTime elapes_time;

  engine.Run();
  LOG_INFO("cost time 1 : {}ms", elapes_time.DurationMs());
  elapes_time.Restart();
  engine.Run();
  LOG_INFO("cost time 2 : {}ms", elapes_time.DurationMs());
  elapes_time.Restart();
  engine.Run();
  LOG_INFO("cost time 3 : {}ms", elapes_time.DurationMs());
  elapes_time.Restart();
}
} // namespace

TEST(YoloV8, CPU_FP32) { RunYoloV8Model(fp32_model_path, inference::kCPU); }

TEST(YoloV8, GPU_FP32) { RunYoloV8Model(fp32_model_path, inference::kGPU); }