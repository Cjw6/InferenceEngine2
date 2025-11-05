#include "inference/onnxruntime/onnxruntime.h"
#include "inference/utils/log.h"
#include "modelzoo/common/img_common.hpp"
#include <gtest/gtest.h>

namespace {

const std::string fp32_model_path = "modelzoo/mnist/mnist.onnx";
const std::string fp16_model_path = "modelzoo/mnist/mnist_fp16.onnx";
const std::string test_img_path = "modelzoo/mnist/0001-0.jpg";
const std::string label_path = "modelzoo/mnist/labels.txt";

void RunMnistModel(const std::string &model_path,
                   inference::DeviceType device_type) {
  cv::Mat img = cv::imread(test_img_path);
  ASSERT_FALSE(img.empty()) << "Failed to read image: " << test_img_path;
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

  auto params = inference::GetDefaultOnnxRuntimeEngineParams();
  params.device_type = device_type;
  params.model_path = model_path;

  ::inference::OnnxRuntimeEngine engine;
  int ret = engine.Init(params);
  ASSERT_TRUE(ret == 0) << "Failed to init engine: " << ret;

  auto intput_tensor = engine.GetInputTensors().at("x");
  img_utils::BlobNormalizeFromImage(img, intput_tensor.p, intput_tensor.data_type);

  ret = engine.Run();
  ASSERT_TRUE(ret == 0) << "Failed to run engine: " << ret;

  auto output_tensor_ = engine.GetOutputTensors().at("linear_2");
  auto result = img_utils::Softmax(output_tensor_.p, output_tensor_.mem_size,
                                   output_tensor_.data_type);
  ASSERT_TRUE(result.size() == 10) << "Invalid output size: " << result.size();

  auto max_iter = std::max_element(result.begin(), result.end());
  ASSERT_TRUE(max_iter != result.end()) << "Failed to find max element";

  int max_index = std::distance(result.begin(), max_iter);
  ASSERT_TRUE(max_index == 0) << "classify result error: " << max_index;
}

} // namespace

TEST(Mnist, CPU_FP32) { RunMnistModel(fp32_model_path, inference::kCPU); }

TEST(Mnist, CPU_FP16) { RunMnistModel(fp16_model_path, inference::kCPU); }

TEST(Mnist, GPU_FP32) { RunMnistModel(fp32_model_path, inference::kGPU); }

TEST(Mnist, GPU_FP16) { RunMnistModel(fp16_model_path, inference::kGPU); }
