#include <gflags/gflags.h>

#include "inference/onnxruntime/onnxruntime.h"
#include "inference/utils/log.h"
#include "inference/utils/to_string.h"
#include "modelzoo/common/img_common.hpp"

DEFINE_string(img_path, "modelzoo/mnist/0001-0.jpg", "Path to the image file.");
DEFINE_string(model_path, "modelzoo/mnist/mnist.onnx",
              "Path to the ONNX model file.");
DEFINE_string(device, "cpu", "Device to run the inference on.");
DEFINE_int32(device_id, 0, "gpu device id to use.");
DEFINE_string(label_path, "modelzoo/mnist/labels.txt", "");
DEFINE_int32(max_batch_size, 5, "max batch size for inference.");

inference::InferenceParams GetParams() {
  inference::InferenceParams params;
  if (FLAGS_device == "cpu") {
    params.device_type = inference::DeviceType::kCPU;
  } else if (FLAGS_device == "gpu") {
    params.device_type = inference::DeviceType::kGPU;
  }
  params.device_id = FLAGS_device_id;
  params.model_path = FLAGS_model_path;
  params.max_batch_size = FLAGS_max_batch_size;
  return params;
}

int main(int argc, char *argv[]) {
  ::google::ParseCommandLineFlags(&argc, &argv, true);
  LogInit();
  LOG_INFO("use model_path: {}", FLAGS_model_path);
  LOG_INFO("use img_path: {}", FLAGS_img_path);

  inference::InferenceParams params = GetParams();
  inference::OnnxRuntimeEngine engine;
  int ret = engine.Init(params);
  if (ret != 0) {
    LOG_ERROR("Failed to init engine with error: {}", ret);
    return -1;
  }

  auto labels = img_utils::ReadLabelsFromFile(FLAGS_label_path);
  LOG_INFO("labels:{}", cpputils::VectorToString(labels));

  cv::Mat img = cv::imread(FLAGS_img_path);
  if (img.empty()) {
    LOG_ERROR("Failed to read image: {}", FLAGS_img_path);
    return -2;
  }

  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

  auto input_names = engine.GetInputNodeNames();
  for (const auto &name : input_names) {
    LOG_INFO("input name: {}", name);
  }

  auto intput_tensor = engine.GetInputTensors().at("x");
  LOG_INFO("input tensor desc: {}", cpputils::ToString(intput_tensor));

  img_utils::BlobNormalizeFromImage(img, intput_tensor.p, intput_tensor.data_type);

  auto output_names = engine.GetOutputNodeNames();
  for (const auto &name : output_names) {
    LOG_INFO("output name: {}", name);
  }
  std::cout << "input tensor shape: "
            << cpputils::VectorToString(intput_tensor.shape) << std::endl;

  ret = engine.Run();
  if (ret != 0) {
    LOG_ERROR("Failed to run engine with error: {}", ret);
    return -3;
  }

  auto output_tensor_ = engine.GetOutputTensors().at("linear_2");
  LOG_INFO("output tensor desc: {}", cpputils::ToString(output_tensor_));

  auto result = img_utils::Softmax(output_tensor_.p, output_tensor_.mem_size,
                                   output_tensor_.data_type);
  for (int i = 0; i < result.size(); ++i) {
    LOG_INFO("index: {}, value: {}", i, result[i]);
  }
  auto max_iter = std::max_element(result.begin(), result.end());
  if (max_iter != result.end()) {
    int max_index = std::distance(result.begin(), max_iter);
    LOG_INFO("max index: {}, max value: {}, label:{}", max_index, *max_iter,
             labels[max_index]);
  } else {
    LOG_ERROR("Failed to get max element from softmax result");
  }
}
