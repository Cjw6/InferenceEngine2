#pragma once

#include "inference/utils/log.h"
#include <opencv2/opencv.hpp>

namespace detect_utils {

struct Threshold {
  float det_threshold = 0.5f;
  float iou_threshold = 0.5f;
};

struct DetectBox {
  int x = -1;
  int y = -1;
  int w = -1;
  int h = -1;
  int class_id = -1;
  float confidence = -1.0f;
};

inline std::ostream &operator<<(std::ostream &s, const DetectBox &tensor_desc) {
  s << "DetectBox{x:" << tensor_desc.x << " y:" << tensor_desc.y
    << " w:" << tensor_desc.w << " h:" << tensor_desc.h
    << " class_id:" << tensor_desc.class_id
    << "  confidence:" << tensor_desc.confidence << "}";
  return s;
}

inline void VisualDetectBox(cv::Mat &image, const std::vector<DetectBox> &boxes,
                            const std::vector<std::string> &class_names) {
  for (size_t i = 0; i < boxes.size(); ++i) {
    const auto &box = boxes[i];
    int x1 = box.x;
    int y1 = box.y;
    int x2 = box.x + box.w;
    int y2 = box.y + box.h;
    int class_id = box.class_id;
    float confidence = box.confidence;
    std::string label = class_names[class_id];
    std::string label_text = label + " " + cv::format("%.3f", confidence);

    // 绘制矩形和标签
    int base_line;
    cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX,
                                          0.6, 1, &base_line);
    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2),
                  cv::Scalar(251, 81, 163), 2, cv::LINE_AA);
    cv::rectangle(image, cv::Point(x1, y1 - label_size.height),
                  cv::Point(x1 + label_size.width, y1), cv::Scalar(125, 40, 81),
                  -1);
    cv::putText(image, label_text, cv::Point(box.x, box.y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1);
  }
}

} // namespace detect_utils