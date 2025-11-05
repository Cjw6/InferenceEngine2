#include "inference/tensor/tensor.h"
#include "inference/tensor/buffer.h"
#include "inference/utils/half.hpp"
#include "inference/utils/log.h"
#include "inference/utils/to_string.h"

#include <numeric>

using half_float::half;

namespace inference {

std::ostream &operator<<(std::ostream &s, const DeviceType &device_type) {
  s << "DeviceType::";
  switch (device_type) {
  case kCPU:
    return s << "CPU";
  case kGPU:
    return s << "GPU";
  case kNPU:
    return s << "NPU";
  default:
    return s << "Unknown";
  }
}

std::ostream &operator<<(std::ostream &s, const TensorDataType &data_type) {
  s << "TensorDataType::";
  switch (data_type) {
  case kFP32:
    return s << "FP32";
  case kFP16:
    return s << "FP16";
  case kInt8:
    return s << "Int8";
  case kUint8:
    return s << "Uint8";
  case kInt64:
    return s << "Int64";
  default:
    return s << "Unknown";
  }
}

std::ostream &operator<<(std::ostream &s, const TensorDesc &tensor_desc) {
  return s << "TensorDesc {" << "data_type:" << tensor_desc.data_type
           << ", shape:" << cpputils::VectorToString(tensor_desc.shape)
           << ", element_size:" << tensor_desc.element_size << "}";
}

std::ostream &operator<<(std::ostream &s,
                         const TensorDataPointer &tensor_pointer) {
  return s << "TensorDataPointer {" << "p:" << tensor_pointer.p << ", "
           << "size:" << tensor_pointer.mem_size
           << ", shape:" << cpputils::VectorToString(tensor_pointer.shape)
           << ", data_type:" << tensor_pointer.data_type
           << ", device_type:" << tensor_pointer.device_type << "}";
}

size_t GetDataTypeSize(TensorDataType data_type) {
  switch (data_type) {
  case kFP32:
    return sizeof(float);
  case kFP16:
    return sizeof(half);
  case kInt8:
    return sizeof(int8_t);
  case kUint8:
    return sizeof(uint8_t);
  case kInt64:
    return sizeof(int64_t);
  default:
    return 0;
  }
}

size_t GetElemMemSize(TensorDataType data_type, size_t element_size) {
  return element_size * GetDataTypeSize(data_type);
}

int64_t GetElemCntFromShape(const std::vector<int64_t> &v, int64_t batch_size) {
  int64_t cnt = 1;
  for (auto dim : v) {
    if (dim > 0) {
      cnt *= dim;
    } else {
      cnt *= batch_size;
    }
  }
  return cnt;
}

int64_t GetSingleBatchElemCntFromShape(const std::vector<int64_t> &v) {
  int64_t sum_size = 1;
  for (int i = 1; i < v.size(); i++) {
    sum_size *= v[i];
  }
  return sum_size;
}

// namespace {

// BufferType SelectBufferTypeByDevice(DeviceType device_type) {
//   switch (device_type) {
//   case kCPU:
//     return BufferType::Host;
//   case kGPU:
//     return BufferType::Device;
//   default:
//     throw std::runtime_error("Invalid device type");
//   }
// }

// } // namespace

int64_t GetMemSizeFromShape(const std::vector<int64_t> &v,
                            TensorDataType data_type, int batch_size) {
  int64_t elem_cnt = GetElemCntFromShape(v, batch_size);
  return GetElemMemSize(data_type, elem_cnt);
}

int64_t GetSingleBatchMemSizeFromShape(const std::vector<int64_t> &v,
                                       TensorDataType data_type) {
  int64_t elem_cnt = GetSingleBatchElemCntFromShape(v);
  return GetElemMemSize(data_type, elem_cnt);
}

TensorBufferUPtr CreateTensorBufferCPU(TensorDataType data_type,
                                       size_t mem_size) {
  auto *buffer = BufferFactory::createBuffer(BufferType::Host);
  buffer->allocate(mem_size);
  return TensorBufferUPtr(buffer);
}

} // namespace inference
