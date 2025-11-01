#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "inference/inference.h"
#include "inference/tensor/tensor.h"
#include "inference/utils/construct.h"

namespace inference {

class OnnxRuntimeEngineImpl;

class OnnxRuntimeEngine {
public:
  OnnxRuntimeEngine();
  ~OnnxRuntimeEngine();

  NON_COPY_CONSTRUCT(OnnxRuntimeEngine);
  NON_MOVE_CONSTRUCT(OnnxRuntimeEngine);

  int Init(const std::string &model_path, const std::string &log_id = "",
           const ParamMap &params = {});
  void Deinit();

  int Run();

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