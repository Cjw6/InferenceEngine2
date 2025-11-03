#include "inference/tensor/tensor.h"
#include "inference/tensor/buffer.h"
#include "inference/utils/log.h"
#include "inference/utils/to_string.h"

#include <numeric>

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

size_t GetTensorMemSize(TensorDataType data_type, size_t element_size) {
  switch (data_type) {
  case kFP32:
    return element_size * sizeof(float);
  case kFP16:
    return element_size * sizeof(float) / 2;
  case kInt8:
    return element_size * sizeof(int8_t);
  case kUint8:
    return element_size * sizeof(uint8_t);
  default:
    return 0;
  }
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

int64_t GetSingleBatchEleCntFromShape(const std::vector<int64_t> &v) {
  for (int i = 1; i < v.size(); i++) {
    if (v[i] < 0) {
      return v[i] * -1;
    }
  }
  return -1;
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
  return GetTensorMemSize(data_type, elem_cnt);
}

int64_t GetSingleBatchMemSizeFromShape(const std::vector<int64_t> &v,
                                       TensorDataType data_type) {
  int64_t elem_cnt = GetSingleBatchEleCntFromShape(v);
  return GetTensorMemSize(data_type, elem_cnt);
}

TensorBufferUPtr CreateTensorBufferCPU(TensorDataType data_type,
                                       size_t mem_size) {
  // LOG_DEBUG("CreateTensorBufferCPU, data_type: {}, mem_size: {}",
            // cpputils::ToString(data_type), mem_size);
  auto *buffer = BufferFactory::createBuffer(BufferType::Host);
  buffer->allocate(mem_size);
  return TensorBufferUPtr(buffer);
}

} // namespace inference
