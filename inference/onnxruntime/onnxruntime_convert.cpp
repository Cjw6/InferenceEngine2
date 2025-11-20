#include "inference/onnxruntime/onnxruntime_convert.h"
#include <cpptoolkit/strings/to_string.h>

namespace inference {

TensorDataType
OnnxTensorDataTypeToTensorDataType(ONNXTensorElementDataType onnx_data_type) {
  switch (onnx_data_type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return kFP32;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return kFP16;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return kInt8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return kUint8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return kInt64;
  default:
    throw std::runtime_error("Unsupported ONNX tensor element data type" +
                             cpptoolkit::ToString(onnx_data_type));
  }
}

} // namespace inference
