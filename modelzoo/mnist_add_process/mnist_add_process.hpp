#pragma once

#include "inference/onnxruntime/onnxruntime.h"
#include "inference/utils/assert.h"
#include "inference/utils/exception.h"
#include "inference/utils/log.h"
#include "inference/utils/map.h"
#include "inference/utils/to_string.h"
#include "modelzoo/common/img_common.hpp"

namespace modelzoo {

using ::cpputils::ToString;
using ::cpputils::VectorToString;

class MnistAddProcess {
public:
  using Result = std::pair<int, float>;

  MnistAddProcess() {}
  ~MnistAddProcess() { Deinit(); }

  int Init(const inference::InferenceParams &params) {
    int ret = engine.Init(params);
    if (engine.IsDynamicModel()) {
      CHECK_MSG(!engine.IsDynamicModel(), "model must be static")
      LOG_ERROR("model must be static");
      Deinit();
      return -1;
    }
    return ret;
  }

  void Deinit() { engine.Deinit(); }

  bool IsReady() { return engine.IsReady(); }

  Result Classify(const cv::Mat &img) {
    if (!engine.IsReady()) {
      THROW_RUNTIME_EXCEPTION("Engine not ready");
    }

    // auto i_tensor = engine.GetInputTensors().at("input");
    // LOG_INFO("input tensor size:{}", i_tensor.p_arr.size());

    if (img.type() != CV_8UC1) {
      THROW_RUNTIME_EXCEPTION(
          fmt::format("img type error!!! cur_img_type:{}, need_img_type:{}",
                      img.type(), CV_8UC1));
    }

    auto i_desc = engine.GetInputTensorDescs();
    auto o_desc = engine.GetOutputTensorDescs();

    // LOG_INFO("i_desc:\n{}", cpputils::MapToString(i_desc));
    // LOG_INFO("o_desc:\n{}", cpputils::MapToString(o_desc));

    auto i_tensor = engine.GetInputTensors();
    auto x_tensor = cpputils::MapGet(i_tensor, "x");
    if (!x_tensor) {
      THROW_RUNTIME_EXCEPTION("x_tensor is null");
    }

    CHECK_MSG(x_tensor->data_type == inference::kUint8,
              fmt::format("x_tensor data_type error!!!, x_tensor->data_type:{}",
                          cpputils::ToString(x_tensor->data_type)));
    memcpy(x_tensor->p, img.data, img.total());

    engine.Run();

    // [2025-11-05 15:39:09.960] [info] [mnist_add_process.hpp:54 Classify]
    // i_desc: {x:TensorDesc {data_type:TensorDataType::Uint8, shape:[1, 1, 28,
    // 28], element_size:784}}
    //
    // [2025-11-05 15:39:09.960] [info] [mnist_add_process.hpp:55 Classify]
    // o_desc: {y:TensorDesc {data_type:TensorDataType::FP32, shape:[1, 10],
    // element_size:10}, z:TensorDesc {data_type:TensorDataType::Int64,
    // shape:[1], element_size:1}}

    auto o_tensor = engine.GetOutputTensors();
    auto y_tensor = cpputils::MapGet(o_tensor, "y");
    if (!y_tensor) {
      THROW_RUNTIME_EXCEPTION("y_tensor is null");
    }
    auto z_tensor = cpputils::MapGet(o_tensor, "z");
    if (!z_tensor) {
      THROW_RUNTIME_EXCEPTION("z_tensor is null");
    }

    std::span<float> confidence((float *)y_tensor->p, 10);
    int64_t idx = *(int64_t *)(z_tensor->p);

    // LOG_INFO("confidence:{}", cpputils::SpanToString(confidence));
    // LOG_INFO("idx:{}", idx);

    if (idx < 0 || idx > 9) {
      THROW_RUNTIME_EXCEPTION(fmt::format("idx error!!! idx:{}", idx));
    }

    return {idx, confidence[idx]};
  }

private:
  inference::OnnxRuntimeEngine engine;
};

} // namespace modelzoo
