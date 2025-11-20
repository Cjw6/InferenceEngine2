#pragma once

#include "inference/onnxruntime/onnxruntime.h"
#include <cpptoolkit/exception/exception.h>
#include <cpptoolkit/log/log.h>
#include <cpptoolkit/strings/to_string.h>
#include "modelzoo/common/img_common.hpp"

namespace modelzoo {

using ::cpptoolkit::ToString;
using ::cpptoolkit::ToString;

class MnistDynamic {
public:
  using BatchResult = std::vector<std::pair<int, float>>;

  MnistDynamic() {}
  ~MnistDynamic() { Deinit(); }

  int Init(const inference::InferenceParams &params) {
    int ret = engine.Init(params);
    if (!engine.IsDynamicModel()) {
      LOG_ERROR("Not a dynamic model");
      Deinit();
      return -1;
    }
    return ret;
  }

  void Deinit() { engine.Deinit(); }

  bool IsReady() { return engine.IsReady(); }

  int GetMaxBatchSize() { return engine.GetMaxBatchSize(); }

  BatchResult Classify(const std::vector<cv::Mat> &imgs) {
    if (!engine.IsReady()) {
      THROW_RUNTIME_EXCEPTION("Engine not ready");
    }

    int batch_size = engine.GetMaxBatchSize();
    if (imgs.size() > batch_size) {
      THROW_RUNTIME_EXCEPTION(
          fmt::format("input imgs.size:{} > enfine max_batch_size:{}",
                      imgs.size(), engine.GetMaxBatchSize()));
    }
    batch_size = imgs.size();
    LOG_INFO("batch_size:{}", batch_size);

    auto i_tensor = engine.GetInputTensors().at("input");
    LOG_INFO("input tensor size:{}", i_tensor.p_arr.size());

    for (int i = 0; i < batch_size; i++) {
      if (imgs[i].type() != CV_8UC1) {
        THROW_RUNTIME_EXCEPTION(fmt::format(
            "img idx:{} type error, cur_img_type:{}, need_img_type:{}", i,
            imgs[i].type(), CV_8UC1));
      }

      auto p = i_tensor.p_arr[i];
      imgutils::BlobNormalizeFromImage(imgs[i], p, i_tensor.data_type);
    }

    int ret = engine.Run(batch_size);
    if (ret != 0) {
      THROW_RUNTIME_EXCEPTION(fmt::format("Failed to run engine, ret:{}", ret));
    }

    BatchResult batch_result;
    batch_result.reserve(batch_size);
    auto output_tensor = engine.GetOutputTensors();
    for (int i = 0; i < batch_size; i++) {
      auto &batch_tensor = output_tensor.at("output");
      auto p = batch_tensor.p_arr[i];
      auto result =
          imgutils::Softmax(p, batch_tensor.mem_size, batch_tensor.data_type);
      auto max_iter = std::max_element(result.begin(), result.end());
      int max_idx = std::distance(result.begin(), max_iter);
      batch_result.emplace_back(max_idx, *max_iter);
    }
    return batch_result;
  }

private:
  inference::OnnxRuntimeEngine engine;
};

} // namespace modelzoo
