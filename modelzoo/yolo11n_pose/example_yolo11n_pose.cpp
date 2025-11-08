// TODO optimize:

// clang-format off
/*
2025-11-07 00:16:30.740754226 [W:onnxruntime:, transformer_memcpy.cc:111 ApplyImpl] 4 Memcpy nodes are added to the graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
2025-11-07 00:16:30.741726124 [W:onnxruntime:, session_state.cc:1316 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
2025-11-07 00:16:30.741751302 [W:onnxruntime:, session_state.cc:1318 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
*/
// clang-format on

#include <gflags/gflags.h>

#include "inference/inference.h"
#include "inference/utils/assert.h"
#include "inference/utils/exception.h"
#include "inference/utils/log.h"
#include "inference/utils/to_string.h"
#include "modelzoo/common/filesystem_common.hpp"
#include "modelzoo/common/img_common.hpp"
#include "modelzoo/yolo11n_pose/yolo11n_pose.hpp"

DEFINE_string(img_path, "modelzoo/yolo11n_pose/data/img", "image path");
DEFINE_string(model_path, "modelzoo/yolo11n_pose/data/yolo11n-pose.onnx",
              "model path");
DEFINE_string(label_path, "modelzoo/yolo11n_pose/data/labels.txt",
              "label path");

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

  modelzoo::Yolo11NSeg yolov11n_pose;
  yolov11n_pose.SetKptShapes({17, 3});
  int ret = yolov11n_pose.Init(infer_params);
  if (ret != 0) {
    LOG_ERROR("init yolov8n failed");
    return 1;
  }

  LOG_INFO("init yolov11n_pose success, dump:{}", yolov11n_pose.DumpModel());
  ret = yolov11n_pose.Warmup();
  if (ret != 0) {
    LOG_ERROR("warmup yolov8n failed");
    return 1;
  }

  auto random_color = imgutils::GetRandomColor(40);

  for (int i = 0; i < img_datas.size(); i++) {
    modelzoo::Yolo11NSeg::Result result;
    ret = yolov11n_pose.DetectPose(img_datas[i], result);

    std::vector<imgutils::KeyPointList *> ps;
    for (auto &obj : result) {
      ps.push_back(&obj.kps);
    }

    imgutils::DrawKeyPointList(img_datas[i], ps, random_color);
    cv::imshow("result", img_datas[i]);
    cv::waitKey(0);
  }

  // for (int i = 0; i < img_datas.size(); i++) {
  //   auto &img = img_datas[i];
  //   auto &img_path = img_paths[i];
  //   modelzoo::YoloV8N::Result result;
  //   ret = yolov11n_pose.Detect(img, result);
  //   if (ret != 0) {
  //     LOG_ERROR("detect yolov8n failed, img_path: {}", img_path);
  //     continue;
  //   }
  // }
}