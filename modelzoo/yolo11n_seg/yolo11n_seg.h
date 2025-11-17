#pragma once

#include "inference/inference.h"

#include <opencv2/opencv.hpp>

namespace inference {
class OnnxRuntimeEngine;
}

namespace modelzoo {

class Yolo11NSeg {
public:
  struct ResultObj {
    int id = 0;
    float accu = 0.0;
    cv::Rect bound;
    cv::Mat mask;
  };

  using Result = std::vector<ResultObj>;

  Yolo11NSeg();
  ~Yolo11NSeg();

  int Init(const inference::InferenceParams &params);
  void Deinit();
  std::string DumpModel();
  bool IsReady();

  int Warmup();
  int Segment(const cv::Mat &img, Result &result);

  static void DrawResult(cv::Mat &img, std::vector<ResultObj> &result,
                         std::vector<cv::Scalar> color);

private:
  int Preprocess(const cv::Mat &img);
  int Postprocess(Result &result);

  std::unique_ptr<inference::OnnxRuntimeEngine> engine_;
  float img_scales_ = 0.0f;
};

} // namespace modelzoo
