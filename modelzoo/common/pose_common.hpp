#pragma once

#include <cpptoolkit/log/log.h>
#include <fmt/format.h>
#include <opencv2/opencv.hpp>
#include <sstream>

namespace imgutils {

struct KeyPoint {
  int x;
  int y;
  float confidence = -.0f;
};

using KeyPointList = std::vector<KeyPoint>;

inline std::ostream &operator<<(std::ostream &s, const KeyPoint &kp) {
  s << "DetectBox{x:" << kp.x << " y:" << kp.y << " y:" << kp.y
    << "  confidence:" << kp.confidence << "}";
  return s;
}

inline void DrawKeyPoint(cv::Mat &img, const std::vector<KeyPoint> &kps,
                         cv::Scalar color,
                         const std::vector<std::string> &labels = {}) {
  for (int i = 0; i < kps.size(); i++) {
    auto &kp = kps[i];
    cv::circle(img, cv::Point(kp.x, kp.y), 3, color, -1);
    std::string label;
    if (labels.empty()) {
      label = fmt::format("{} {}", i, kp.confidence);
    } else {
      label = fmt::format("{} {}", labels[i], kp.confidence);
    }
    cv::putText(img, label, cv::Point(kp.x, kp.y), cv::FONT_HERSHEY_SIMPLEX,
                0.5, color, 1);
  }
}

inline void DrawKeyPointList(cv::Mat &img,
                             const std::vector<KeyPointList *> &kps,
                             const std::vector<cv::Scalar> &colors,
                             const std::vector<std::string> &labels = {}) {
  if (colors.size() < kps.size()) {
    LOG_ERROR("colors.size():{} < kps.size():{}", colors.size(), kps.size());
  }
  for (int i = 0; i < kps.size(); i++) {
    DrawKeyPoint(img, *kps[i], colors[i], labels);
  }
}

} // namespace imgutils