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

std::ostream &operator<<(std::ostream &s,
                         const TensorDataPointer &tensor_pointer) {
  return s << "TensorDataPointer {" << "p:" << tensor_pointer.p << ", "
           << "size:" << tensor_pointer.mem_size
           << ", shape:" << cpputils::VectorToString(tensor_pointer.shape)
           << ", data_type:" << tensor_pointer.data_type
           << ", device_type:" << tensor_pointer.device_type << "}";
}

int64_t GetTensorSize(TensorDataType data_type, int32_t element_size) {
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

int64_t ShapeElemNum(const std::vector<int64_t> &v) {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<int64_t>());
};

namespace {

BufferType SelectBufferTypeByDevice(DeviceType device_type) {
  switch (device_type) {
  case kCPU:
    return BufferType::Host;
  case kGPU:
    return BufferType::Device;
  default:
    throw std::runtime_error("Invalid device type");
  }
}

} // namespace

TensorBufferUPtr CreateTensorHostBuffer(TensorDataType data_type,
                                        int64_t element_size,
                                        DeviceType device_type) {
  int64_t size = GetTensorSize(data_type, element_size);
  LOG_INFO("Create tensor buffer with size: {}", size);
  auto *buffer = BufferFactory::createBuffer(BufferType::Host);
  buffer->allocate(size);
  return TensorBufferUPtr(buffer);
}

} // namespace inference
