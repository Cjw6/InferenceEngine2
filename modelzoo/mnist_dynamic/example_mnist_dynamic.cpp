#include "modelzoo/common/filesystem_common.hpp"
#include "modelzoo/common/img_common.hpp"
#include "modelzoo/mnist_dynamic/mnist_dynamic.hpp"
#include "inference/utils/backtrace.h"

int main(int argc, char **argv) {
  LogInit();

  const std::string model_path =
      "modelzoo/mnist_dynamic/data/mnist_dynamic.onnx";
  const std::string jpg_dir = "modelzoo/mnist_dynamic/data";
  const std::string jpg_ext = ".jpg";
  const std::string label_file = "modelzoo/mnist_dynamic/data/labels.txt";

  auto labels = img_utils::ReadLabelsFromFile(label_file);
  auto img_paths = cpputils::GetAllFilesWithExt(jpg_dir.c_str(), jpg_ext);
  LOG_INFO("img_paths:{}", cpputils::VectorToString(img_paths));

  std::vector<cv::Mat> imgs;
  for (auto &f : img_paths) {
    auto img = cv::imread(f.string());
    if (!img.empty()) {
      if (img.type() != CV_8UC1) {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
      }
      imgs.push_back(img);
    }
  }
  int batch_size = 32;

  batch_size = std::min((int)imgs.size(), batch_size);
  LOG_INFO("batch_size:{}", batch_size);

  imgs.resize(batch_size);

  auto params = inference::GetDefaultOnnxRuntimeEngineParams();
  params.device_type = inference::kCPU;
  params.model_path = model_path;
  params.max_batch_size = batch_size;

  modelzoo::MnistDynamic mnist_dynamic;
  mnist_dynamic.Init(params);
  auto results = mnist_dynamic.Classify(imgs);

  for (int i = 0; i < imgs.size(); i++) {
    LOG_INFO("idx:{} img_path:{} class_id:{}, label:{}, confidence:{}", i,
             img_paths[i].string(), results[i].first, labels[results[i].first],
             results[i].second);
  }
}