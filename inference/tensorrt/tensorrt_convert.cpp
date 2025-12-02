#include "tensorrt_convert.h"

#include <cpptoolkit/exception/exception.h>
#include <fmt/format.h>

namespace inference {

TensorShape TensorRTConvertShape(const nvinfer1::Dims &dims) {
  TensorShape shape;
  shape.reserve(dims.nbDims);
  for (int i = 0; i < dims.nbDims; i++) {
    shape.push_back(dims.d[i]);
  }
  return shape;
}

TensorDataType TensorRTConvertDataType(nvinfer1::DataType data_type) {
  switch (data_type) {
  case nvinfer1::DataType::kFLOAT:
    return TensorDataType::kFP32;
  case nvinfer1::DataType::kHALF:
    return TensorDataType::kFP16;
  case nvinfer1::DataType::kINT64:
    return TensorDataType::kInt64;
  default:
    THROW_RUNTIME_EXCEPTION(
        fmt::format("Unsupported data type, {}", (int)data_type));
  }
}

} // namespace inference