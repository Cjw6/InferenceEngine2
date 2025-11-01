#include "onnxruntime_convert.h"

namespace inference {

TensorDataType
OnnxTensorDataTypeToTensorDataType(ONNXTensorElementDataType onnx_data_type) {
  switch (onnx_data_type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return kFloat32;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return kFloat16;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return kInt8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return kUint8;
  default:
    throw std::runtime_error("Unsupported ONNX tensor element data type");
  }
}

} // namespace inference
