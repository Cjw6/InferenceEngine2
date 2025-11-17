#pragma once

#include "inference/inference.h"
#include "modelzoo/common/detect_common.hpp"
#include "modelzoo/common/segment_common.hpp"

namespace inference {
class OnnxRuntimeEngine;
}

namespace modelzoo {

class Yolo11NSeg {
public:
  using Threshold = imgutils::Threshold;
  struct Object {
    imgutils::DetectBox box;
    imgutils::MaskRegion mask;
  };
  using Result = std::vector<Object>;

  Yolo11NSeg();
  ~Yolo11NSeg();

  int Init(const inference::InferenceParams &params);
  void Deinit();
  std::string DumpModel();
  bool IsReady();

  int Warmup();
  int Segment(const cv::Mat &img, Result &result);

private:
  int Preprocess(const cv::Mat &img);
  int Postprocess(Result &result);

  std::unique_ptr<inference::OnnxRuntimeEngine> engine_;
  float img_scales_ = 0.0f;
};

} // namespace modelzoo
