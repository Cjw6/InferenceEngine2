#include "inference/onnxruntime/onnxruntime.h"
#include <cpptoolkit/time/elapse_time.hpp>
#include <cpptoolkit/log/log.h>
#include "modelzoo/common/img_common.hpp"
#include <gtest/gtest.h>

namespace {

const std::string fp32_model_path = "modelzoo/yolov8n/data/yolov8n.onnx";
const std::string fp16_model_path = "modelzoo/yolov8n/data/yolov8n_fp16.onnx";

void RunYoloV8Model(const std::string &model_path,
                    inference::DeviceType device_type) {

  auto params = inference::GetDefaultOnnxRuntimeEngineParams();
  params.device_type = device_type;
  params.model_path = model_path;
  // params.log_level = 2;

  ::inference::OnnxRuntimeEngine engine;
  int ret = engine.Init(params);
  ASSERT_TRUE(ret == 0) << "Failed to init engine: " << ret;

  cpptoolkit::ElapsedTime elapse_time;;

  ret = engine.Run();
  ASSERT_TRUE(ret == 0) << "Failed to run engine: " << ret;
  LOG_INFO("cost time 1 : {}ms", elapse_time.DurationMs());
  elapse_time.Restart();
  ret = engine.Run();
  ASSERT_TRUE(ret == 0) << "Failed to run engine: " << ret;
  LOG_INFO("cost time 2 : {}ms", elapse_time.DurationMs());
  elapse_time.Restart();
  ret = engine.Run();
  ASSERT_TRUE(ret == 0) << "Failed to run engine: " << ret;
  LOG_INFO("cost time 3 : {}ms", elapse_time.DurationMs());
  elapse_time.Restart();
}
} // namespace

TEST(YoloV8, CPU_FP32) { RunYoloV8Model(fp32_model_path, inference::kCPU); }
TEST(YoloV8, CPU_FP16) { RunYoloV8Model(fp16_model_path, inference::kCPU); }

TEST(YoloV8, GPU_FP32) { RunYoloV8Model(fp32_model_path, inference::kGPU); }
TEST(YoloV8, GPU_FP16) { RunYoloV8Model(fp16_model_path, inference::kGPU); }
