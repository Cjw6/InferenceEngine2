#pragma once

#include "inference/onnxruntime/onnxruntime.h"
#include "inference/utils/assert.h"
#include "inference/utils/exception.h"
#include "inference/utils/log.h"
#include "inference/utils/map.h"
#include "inference/utils/to_string.h"

#include "modelzoo/common/detect_common.hpp"
#include "modelzoo/common/img_common.hpp"
#include "modelzoo/common/pose_common.hpp"

#define DUMP_MODEL_IO

// clang-format off
/*

[2025-11-05 17:51:22.294] [info] [Yolo11NPose.hpp:26 Init] model io desc: model info:
dynamic model: false
input nums: 1
input: images
TensorDesc {data_type:TensorDataType::FP32, shape:[1, 3, 640, 640], element_size:1228800}
output nums: 1
output: output0
TensorDesc {data_type:TensorDataType::FP32, shape:[1, 84, 8400], element_size:705600}

*/
// clang-format on

namespace modelzoo {

class Yolo11NPose {
public:
  using Threshold = detect_utils::Threshold;
  struct Object {
    detect_utils::DetectBox box;
    std::vector<pose_utils::KeyPoint> kps;
  };
  using Result = std::vector<Object>;

  Yolo11NPose() = default;
  ~Yolo11NPose() = default;

  void SetThreshold(const Threshold &threshold) { threshold_ = threshold; }
  void SetKptShapes(const std::vector<int> &kpt_shapes) {
    kpt_shapes_ = kpt_shapes;
  }

  int Init(const inference::InferenceParams &params) {
    int ret = engine.Init(params);
    if (engine.IsDynamicModel()) {
      LOG_ERROR("must be static model");
      Deinit();
      return -1;
    }

    return ret;
  }

  std::string DumpModel() { return engine.DumpModelInfo(); }

  void Deinit() { engine.Deinit(); }

  bool IsReady() { return engine.IsReady(); }

  int Warmup() { return engine.Warmup(); }

  int DetectPose(const cv::Mat &img, Result &result) {
    result.clear();
    if (img.empty() || img.type() != CV_8UC3) {
      LOG_ERROR("invalid image");
      return -1;
    }

    int ret = Preprocess(img);
    if (ret != 0) {
      LOG_ERROR("preprocess failed: {}", ret);
      return -2;
    }

    ret = engine.Run();
    if (ret != 0) {
      LOG_ERROR("run model failed: {}", ret);
      return -3;
    }

    ret = Postprocess(result);
    if (ret != 0) {
      LOG_ERROR("postprocess failed: {}", ret);
      return -4;
    }

    return 0;
  }

private:
  int Preprocess(const cv::Mat &img) {
    if (img.type() != CV_8UC3) {
      LOG_ERROR("input img.type():{}, need CV_8UC3", img.type());
      return -1;
    }

    auto i_tensor = engine.GetInputTensors().at("images");
    auto [dst_img, img_scale] =
        img_utils::LetterBoxPadImage(img, cv::Size(640, 640));
    img_scales_ = img_scale;
    img_utils::BlobNormalizeFromImage(dst_img, i_tensor.p, i_tensor.data_type);
    return 0;
  }

  int Postprocess(Result &result) {
    auto o_tensor = engine.GetOutputTensors().at("output0");
    const auto &o_shape = o_tensor.shape;
    const auto &o_data = o_tensor.p;
    const auto &o_data_type = o_tensor.data_type;
    int signalResultNum = o_shape[1]; // 4 + 1 + kpt_tensor_size
    int class_cnt = 1;
    int strideNum = o_shape[2]; // 8400

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    cv::Mat rawData;

    if (o_data_type == inference::TensorDataType::kFP32) {
      rawData = cv::Mat(o_shape[1], o_shape[2], CV_32F, o_data);
    } else if (o_data_type == inference::TensorDataType::kFP16) {
      rawData = cv::Mat(o_shape[1], o_shape[2], CV_16F, o_data);
      rawData.convertTo(rawData, CV_32F);
    }
    rawData = rawData.t();
    float *data = (float *)rawData.data;

    std::vector<float *> key_pointers;
    key_pointers.reserve(strideNum);
    confidences.reserve(strideNum);
    class_ids.reserve(strideNum);

    for (int i = 0; i < strideNum; ++i) {
      float *classesScores = data + 4;
      cv::Mat scores(1, class_cnt, CV_32FC1, classesScores);
      cv::Point class_id;
      double maxClassScore;
      cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
      if (maxClassScore > threshold_.det_threshold) {
        confidences.push_back(maxClassScore);
        class_ids.push_back(class_id.x);
        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];
        int left = int((x - 0.5 * w) * img_scales_);
        int top = int((y - 0.5 * h) * img_scales_);
        int width = int(w * img_scales_);
        int height = int(h * img_scales_);
        boxes.push_back(cv::Rect(left, top, width, height));
        key_pointers.push_back(data + 5);
      }
      data += signalResultNum;
    }
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, threshold_.det_threshold,
                      threshold_.iou_threshold, nmsResult);
    for (int i = 0; i < nmsResult.size(); ++i) {
      int idx = nmsResult[i];

      // detect box
      auto &box = boxes[idx];
      detect_utils::DetectBox result_box;
      result_box.class_id = class_ids[idx];
      result_box.confidence = confidences[idx];
      result_box.x = box.x;
      result_box.y = box.y;
      result_box.w = box.width;
      result_box.h = box.height;

      auto kps = DecodeKetPoints(key_pointers, idx);
      result.push_back({result_box, kps});
    }

    return 0;
  }

  pose_utils::KeyPointList
  DecodeKetPoints(const std::vector<float *> &kp_tensor, int idx) {
    auto p = kp_tensor[idx];
    int point_num = kpt_shapes_[0];
    int point_tensor_len = kpt_shapes_[1];
    pose_utils::KeyPointList kps;
    for (int i = 0; i < point_num; i++) {
      pose_utils::KeyPoint k;
      k.x = p[0] * img_scales_;
      k.y = p[1] * img_scales_;
      if (point_tensor_len == 3) {
        k.confidence = p[2];
      }
      kps.push_back(k);
      p += 3;
    }
    return kps;
  }

  inference::OnnxRuntimeEngine engine;
  Threshold threshold_ = {0.1, 0.5};
  float img_scales_ = 0.0f;
  std::vector<int> kpt_shapes_;
};

// inline std::ostream &operator<<(std::ostream &os,
//                                 const Yolo11NPose::Result &result) {
//   return os << cpputils::VectorToString(result);
// }

// inline std::ostream &operator<<(std::ostream &os,
                                // const Yolo11NPose::Object &result) {
  // return os << cpputils::VectorToString(result);
// }



} // namespace modelzoo
