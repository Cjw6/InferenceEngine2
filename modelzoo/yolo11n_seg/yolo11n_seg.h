#pragma once

#include "inference/inference.h"

#include <opencv2/opencv.hpp>

namespace inference {
class OnnxRuntimeEngine;
}

namespace modelzoo {

class Yolo11NSeg {
public:
  struct ImageInfo {
    cv::Size raw_size;
    cv::Vec4d trans;
  };

  struct ResultObj {
    int id = 0;
    float accu = 0.0;
    cv::Rect bound;                        // 图像位置
    cv::Mat mask;                          // 相对Rect 位置
    std::vector<cv::Point> mask_countours; // 图像位置
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
  ImageInfo img_info_;
};

} // namespace modelzoo
