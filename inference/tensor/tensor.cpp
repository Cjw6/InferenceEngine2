#include "inference/tensor/tensor.h"
#include "inference/tensor/buffer.h"
#include "inference/utils/log.h"

#include <numeric>

namespace inference {

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
  auto *buffer =
      BufferFactory::createBuffer(BufferType::Host);
  buffer->allocate(size);
  return TensorBufferUPtr(buffer);
}

} // namespace inference
