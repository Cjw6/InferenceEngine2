#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "inference/inference.h"
#include "inference/inference_engine.h"
#include "inference/tensor/tensor.h"
#include "inference/utils/construct.h"

namespace inference {

class OnnxRuntimeEngineImpl;

class OnnxRuntimeEngine : public InferenceEngine {
public:
  OnnxRuntimeEngine();
  virtual ~OnnxRuntimeEngine();

  NON_COPY_CONSTRUCT(OnnxRuntimeEngine);
  NON_MOVE_CONSTRUCT(OnnxRuntimeEngine);

  int Init(const InferenceParams &params = {});
  void Deinit();

  int Warmup();

  /*batch_size 为 -1 时，
  如果是静态张量模型，默认使用固定shape推理
  如果是动态张量模型，默认使用单batch推理
  batch_size 为其他值时，在动态张量模型使用指定batch_size推理*/
  int Run(int batch_size = -1);

  bool IsReady() const;
  std::string DumpModelInfo() const;
  bool IsDynamicModel() const;
  int GetMaxBatchSize() const;

  int InputsNums() const;
  const InputNodeNames &GetInputNodeNames() const;
  const InputTensorDescs &GetInputTensorDescs() const;
  InputTensorPointers GetInputTensors();

  int OutputsNums() const;
  const OutputNodeNames &GetOutputNodeNames() const;
  const OutputTensorDescs &GetOutputTensorDescs() const;
  OutputTensorPointers GetOutputTensors();

private:
  OnnxRuntimeEngineImpl *impl_ = nullptr;
};

} // namespace inference