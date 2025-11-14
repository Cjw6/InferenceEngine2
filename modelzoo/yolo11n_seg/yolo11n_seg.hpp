#pragma once

#include "inference/onnxruntime/onnxruntime.h"
#include "inference/utils/assert.h"
#include "inference/utils/exception.h"
#include "inference/utils/log.h"
#include "inference/utils/map.h"
#include "inference/utils/to_string.h"

#include "modelzoo/common/detect_common.hpp"
#include "modelzoo/common/img_common.hpp"
#include "modelzoo/common/segment_common.hpp"

namespace modelzoo {

class Yolo11NSeg {
public:
  using Threshold = imgutils::Threshold;
  struct Object {
    imgutils::DetectBox box;
    imgutils::MaskRegion mask;
  };
  using Result = std::vector<Object>;

  Yolo11NSeg() = default;
  ~Yolo11NSeg() = default;

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

  int Segment(const cv::Mat &img, Result &result) {
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
  int Preprocess(const cv::Mat &img) { return 0; }

  int Postprocess(Result &result) { return 0; }

  inference::OnnxRuntimeEngine engine;
};

} // namespace modelzoo
