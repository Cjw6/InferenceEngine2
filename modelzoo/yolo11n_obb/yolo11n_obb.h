#pragma once

#include "inference/inference.h"
#include <opencv2/opencv.hpp>

namespace inference {
class OnnxRuntimeEngine;
}

namespace modelzoo {

class Yolo11NObb {
public:
  Yolo11NObb();
  ~Yolo11NObb();

  struct ImageInfo {
    cv::Size raw_size;
    cv::Vec4d trans;
  };

  struct Thresholds {
    float det_threshold = 0.1;
    float nms_threshold = 0.45;
  };

  struct Box {
    cv::Vec4f box; // [center_x, center_y, w, h]
    float angle;
    float score;
    int class_id;

    std::array<cv::Point, 4> ToXYXY() const;
  };

  using Result = std::vector<Box>;

  void SetClassNum(int class_num);

  int Init(const inference::InferenceParams &params);
  void Deinit();
  std::string DumpModel();
  bool IsReady();

  int Warmup();
  int DetectObb(const cv::Mat &img, Result &result);

  static void DrawObb(cv::Mat &images, const Result &yolo_out);

private:
  int Preprocess(const cv::Mat &img);
  int Postprocess(Result &result);

  std::unique_ptr<inference::OnnxRuntimeEngine> engine_;
  ImageInfo image_info_;
  int class_num_ = 0;
  Thresholds threshold_;
};

} // namespace modelzoo
