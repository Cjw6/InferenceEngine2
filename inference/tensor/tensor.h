#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "inference/tensor/buffer.h"

namespace inference {

enum DeviceType { kCPU = 0, kGPU = 1, kNPU = 2 };

enum TensorDataType {
  kFP32 = 0,
  kFP16 = 1,
  kInt8 = 2,
  kUint8 = 3,
};

using TensorShape = std::vector<int64_t>;

struct TensorDesc {
  // std::string name;
  // TensorShape min_shape;
  TensorDataType data_type = kFP32;
  TensorShape shape;
  int64_t element_size = 0;
  // TensorShape max_shape;
};

struct TensorDataPointer {
  TensorDataPointer() : p(nullptr), size(0), device_type(kCPU) {}
  TensorDataPointer(void *p, int64_t size, const TensorShape &shape,
                    DeviceType device_type)
      : p(p), size(size), shape(shape), device_type(device_type) {}
  void *p;
  int64_t size;
  TensorShape shape;
  DeviceType device_type;
};

using InputNodeNames = std::vector<std::string>;
using InputNodeNamePointers = std::vector<const char *>;
using InputTensorDescs = std::map<std::string, TensorDesc>;
using InputTensorPointers = std::map<std::string, TensorDataPointer>;

using OutputNodeNames = std::vector<std::string>;
using OutputNodeNamePointers = std::vector<char *>;
using OutputTensorDescs = std::map<std::string, TensorDesc>;
using OutputTensorPointers = std::map<std::string, TensorDataPointer>;

int64_t ShapeElemNum(const std::vector<int64_t> &v);

int64_t GetTensorSize(TensorDataType data_type, int64_t element_size,
                      DeviceBuffer buffer_type);

TensorBufferUPtr CreateTensorHostBuffer(TensorDataType data_type,
                                        int64_t element_size,
                                        DeviceType device_type);
} // namespace inference