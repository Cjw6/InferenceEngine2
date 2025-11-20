#include <gflags/gflags.h>

#include "inference/inference.h"
#include <cpptoolkit/assert/assert.h>
#include <cpptoolkit/exception/exception.h>
#include <cpptoolkit/log/log.h>
#include <cpptoolkit/strings/to_string.h>
#include "modelzoo/common/filesystem_common.hpp"
#include "modelzoo/common/img_common.hpp"
#include "modelzoo/yolo11n_obb/yolo11n_obb.h"

DEFINE_string(img_path, "modelzoo/yolo11n_obb/data/img", "image path");
DEFINE_string(model_path, "modelzoo/yolo11n_obb/data/yolo11n-obb.onnx",
              "model path");
DEFINE_int32(class_cnt, 15, "class count");

int main(int argc, char **argv) {
  LogInit();
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG_INFO("img_path: {}", FLAGS_img_path);
  LOG_INFO("model_path: {}", FLAGS_model_path);

  auto img_paths = cpptoolkit::GetImgDataPaths(FLAGS_img_path, ".jpg");
  LOG_INFO("img_paths:{}", cpptoolkit::ToString(img_paths));

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

  modelzoo::Yolo11NObb yolov11n_obb;
  int ret = yolov11n_obb.Init(infer_params);
  if (ret != 0) {
    LOG_ERROR("init yolov8n failed");
    return 1;
  }

  yolov11n_obb.SetClassNum(FLAGS_class_cnt);

  ret = yolov11n_obb.Warmup();
  if (ret != 0) {
    LOG_ERROR("warmup yolov8n failed");
    return 1;
  }

  for (int i = 0; i < img_datas.size(); i++) {
    modelzoo::Yolo11NObb::Result result;
    ret = yolov11n_obb.DetectObb(img_datas[i], result);
    auto src_img = img_datas[i].clone();
    modelzoo::Yolo11NObb::DrawObb(src_img, result);

    cv::imshow("result", src_img);
    cv::waitKey(0);
  }
}