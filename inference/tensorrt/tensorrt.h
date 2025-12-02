#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "inference/inference.h"
#include "inference/inference_engine.h"
#include "inference/tensor/tensor.h"

#include <cpptoolkit/construct/construct.h>

namespace inference {

class TensorRTEngineImpl;

class TensorRTEngine : public InferenceEngine {
public:
  TensorRTEngine();
  virtual ~TensorRTEngine();

  CPP_TK_NON_COPY_CONSTRUCT(TensorRTEngine);
  CPP_TK_NON_MOVE_CONSTRUCT(TensorRTEngine);

  int Init(const InferenceParams &params = {}) override;
  int Init(const InferenceParamsV2 &params = {}) override;
  void Deinit() override;

  int Warmup() override;

  void CopyInputToDevice() override;
  void CopyOutputToHost() override;
  int Run(int batch_size = -1) override;

  bool IsReady() const override;
  std::string DumpModelInfo() const override;
  bool IsDynamicModel() const override;
  int GetMaxBatchSize() const override;

  int InputsNums() const override;
  const InputNodeNames &GetInputNodeNames() const override;
  const InputTensorDescs &GetInputTensorDescs() const override;
  InputTensorPointers GetInputTensors(DeviceType device = kCPU) override;

  int OutputsNums() const override;
  const OutputNodeNames &GetOutputNodeNames() const override;
  const OutputTensorDescs &GetOutputTensorDescs() const override;
  OutputTensorPointers GetOutputTensors(DeviceType device = kCPU) override;

private:
  std::unique_ptr<TensorRTEngineImpl> impl_;
};

} // namespace inference