#pragma once

#include "inference/inference.h"

namespace inference {

class InferenceEngine {
public:
  InferenceEngine() {}
  ~InferenceEngine() {}

  virtual int Init(const InferenceParams &params = {}) = 0;
  virtual void Deinit() = 0;

  virtual int Warmup() = 0;

  virtual int Run(int batch_size = -1) = 0;

  virtual bool IsReady() const = 0;
  virtual std::string DumpModelInfo() const = 0;
  virtual bool IsDynamicModel() const = 0;
  virtual int GetMaxBatchSize() const = 0;

  virtual int InputsNums() const = 0;
  virtual const InputNodeNames &GetInputNodeNames() const = 0;
  virtual const InputTensorDescs &GetInputTensorDescs() const = 0;
  virtual InputTensorPointers GetInputTensors() = 0;

  virtual int OutputsNums() const = 0;
  virtual const OutputNodeNames &GetOutputNodeNames() const = 0;
  virtual const OutputTensorDescs &GetOutputTensorDescs() const = 0;
  virtual OutputTensorPointers GetOutputTensors() = 0;
};

} // namespace inference
