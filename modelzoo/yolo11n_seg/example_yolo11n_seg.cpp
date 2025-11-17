#include <gflags/gflags.h>

#include "inference/inference.h"
#include "inference/utils/assert.h"
#include "inference/utils/exception.h"
#include "inference/utils/log.h"
#include "inference/utils/to_string.h"
#include "modelzoo/common/filesystem_common.hpp"
#include "modelzoo/common/img_common.hpp"
#include "modelzoo/yolo11n_seg/yolo11n_seg.h"

DEFINE_string(img_path, "modelzoo/yolo11n_seg/data/img", "image path");
DEFINE_string(model_path, "modelzoo/yolo11n_seg/data/yolo11n-seg.onnx",
              "model path");
DEFINE_string(label_path, "modelzoo/yolo11n_seg/data/labels.txt", "label path");

int main(int argc, char **argv) {
  LogInit();
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG_INFO("img_path: {}", FLAGS_img_path);
  LOG_INFO("model_path: {}", FLAGS_model_path);
  LOG_INFO("label_path: {}", FLAGS_label_path);

  // auto labels = imgutils::ReadLabelsFromFile(FLAGS_label_path);

  auto img_paths = cpputils::GetImgDataPaths(FLAGS_img_path, ".jpg");
  LOG_INFO("img_paths:{}", cpputils::VectorToString(img_paths));

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

  auto infer_params = inference::GetDefaultOnnxRuntimeEngineParams();
  infer_params.device_type = inference::kGPU;
  infer_params.model_path = FLAGS_model_path;

  CHECK_MSG(img_datas.size() == img_paths.size(),
            fmt::format("img_datas.size():{} != img_paths.size():{}",
                        img_datas.size(), img_paths.size()));

  modelzoo::Yolo11NSeg yolov11n_seg;
  int ret = yolov11n_seg.Init(infer_params);
  if (ret != 0) {
    LOG_ERROR("init yolov8n failed");
    return 1;
  }

  LOG_INFO("init yolov11n_pose success, dump:{}", yolov11n_seg.DumpModel());
  ret = yolov11n_seg.Warmup();
  if (ret != 0) {
    LOG_ERROR("warmup yolov8n failed");
    return 1;
  }

  for (int i = 0; i < img_datas.size(); i++) {
    modelzoo::Yolo11NSeg::Result result;
    ret = yolov11n_seg.Segment(img_datas[i], result);
    break;
  }
}