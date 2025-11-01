#pragma once

#include "inference/tensor/tensor.h"
#include <onnxruntime_cxx_api.h>

namespace inference {

TensorDataType
OnnxTensorDataTypeToTensorDataType(ONNXTensorElementDataType onnx_data_type);

// TensorShape OnnxTensorShapeToTensorShape(const ONNXTensorShape& onnx_shape);

} // namespace inference