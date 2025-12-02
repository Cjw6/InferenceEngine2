#pragma once

#include "inference/tensor/tensor.h"

#include <NvInfer.h>

namespace inference {

TensorShape TensorRTConvertShape(const nvinfer1::Dims &dims);
TensorDataType TensorRTConvertDataType(nvinfer1::DataType data_type);

} // namespace inference
