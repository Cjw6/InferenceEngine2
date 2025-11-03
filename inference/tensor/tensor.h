#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "inference/tensor/buffer.h"

namespace inference {

enum DeviceType { kCPU = 0, kGPU = 1, kNPU = 2 };

std::ostream &operator<<(std::ostream &s, const DeviceType &device_type);

enum TensorDataType {
  kFP32 = 0,
  kFP16 = 1,
  kInt8 = 2,
  kUint8 = 3,
};

std::ostream &operator<<(std::ostream &s, const TensorDataType &data_type);

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
  TensorDataPointer() {}
  TensorDataPointer(void *p, int64_t size, int64_t elem_cnt,
                    const TensorShape &shape, TensorDataType data_type,
                    DeviceType device_type)
      : p(p), mem_size(size), elem_cnt(elem_cnt), shape(shape),
        data_type(data_type), device_type(device_type) {}
  void *p = nullptr;
  int64_t mem_size = 0; // 指定的是内存大小
  int64_t elem_cnt = 0;
  TensorShape shape = {};
  TensorDataType data_type = kFP32;
  DeviceType device_type = kCPU;
};

std::ostream &operator<<(std::ostream &s,
                         const TensorDataPointer &tensor_pointer);

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