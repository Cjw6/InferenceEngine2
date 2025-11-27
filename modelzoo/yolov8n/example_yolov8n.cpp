#include <gflags/gflags.h>

#include "inference/inference.h"
#include <cpptoolkit/assert/assert.h>
#include <cpptoolkit/exception/exception.h>
#include <cpptoolkit/log/log.h>
#include <cpptoolkit/strings/to_string.h>
#include "modelzoo/common/filesystem_common.hpp"
#include "modelzoo/common/img_common.hpp"
#include "modelzoo/yolov8n/yolov8n.hpp"

DEFINE_string(img_path, "modelzoo/yolov8n/data/img/bus.jpg", "image path");
DEFINE_string(model_path, "modelzoo/yolov8n/data/yolov8n.onnx", "model path");
DEFINE_string(label_path, "modelzoo/yolov8n/data/labels.txt", "label path");

int main(int argc, char **argv) {
  cpptoolkit::LogInit();
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG_INFO("img_path: {}", FLAGS_img_path);
  LOG_INFO("model_path: {}", FLAGS_model_path);
  LOG_INFO("label_path: {}", FLAGS_label_path);

  auto labels = imgutils::ReadLabelsFromFile(FLAGS_label_path);

  auto img_paths = cpptoolkit::GetImgDataPaths(FLAGS_img_path, ".jpg");
  std::vector<cv::Mat> img_datas;
  for (auto &f : img_paths) {
    cv::Mat img = cv::imread(f);
    if (img.empty() || img.type() != CV_8UC3) {
      LOG_ERROR("invalid image: {}", f);
      img_datas.emplace_back();
    } else {
      img_datas.push_back(img);
    }
  }
  LOG_DEBUG("img_datas.size(): {}", img_datas.size());

  auto infer_params = inference::GetDefaultOnnxRuntimeEngineParams();
  infer_params.device_type = inference::kGPU;
  infer_params.model_path = FLAGS_model_path;

  CHECK_MSG(img_datas.size() == img_paths.size(),
            fmt::format("img_datas.size():{} != img_paths.size():{}",
                        img_datas.size(), img_paths.size()));

  modelzoo::YoloV8N yolov8n;
  yolov8n.SetClassNum(labels.size());
  int ret = yolov8n.Init(infer_params);
  if (ret != 0) {
    LOG_ERROR("init yolov8n failed");
    return 1;
  }

  ret = yolov8n.Warmup();
  if (ret != 0) {
    LOG_ERROR("warmup yolov8n failed");
    return 1;
  }

  auto random_colors = imgutils::GetRandomColor(labels.size());

  for (int i = 0; i < img_datas.size(); i++) {
    auto &img = img_datas[i];
    auto &img_path = img_paths[i];
    modelzoo::YoloV8N::Result result;
    ret = yolov8n.Detect(img, result);
    if (ret != 0) {
      LOG_ERROR("detect yolov8n failed, img_path: {}", img_path);
      continue;
    }

    LOG_DEBUG("{}: detect result: {}", i, result.size());
    imgutils::VisualDetectBox(img, result, random_colors, labels);
    cv::imshow("result", img);
    cv::waitKey(0);

  }
}