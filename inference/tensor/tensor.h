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
  TensorDataType data_type = kFP32;
  TensorShape shape;
  int64_t element_size = 0; // 这里如果等于-1，说明是动态张量

  bool IsDynamic() const { return element_size == -1; }
};

std::ostream &operator<<(std::ostream &s, const TensorDesc &tensor_desc);

// 如果是 dynamic 模型，这里保存每个batch的指针，数据指的是的单个batch的大小
struct TensorDataPointer {
  TensorDataPointer() {}
  TensorDataPointer(void *p, int64_t size, int64_t elem_cnt,
                    const TensorShape &shape, TensorDataType data_type,
                    DeviceType device_type)
      : p(p), mem_size(size), elem_cnt(elem_cnt), shape(shape),
        data_type(data_type), device_type(device_type) {}
  void *p = nullptr; // 指向第一个batch的指针, static model的tensor指针头
  std::vector<void *> p_arr = {}; // 指向每个batch的指针, 数量是max batch size
  int64_t mem_size = 0;           // 指定的是内存大小
  int64_t elem_cnt = 0;
  TensorShape shape = {};
  TensorDataType data_type = kFP32;
  DeviceType device_type = kCPU;

  int64_t GetBatchSize() const { return p_arr.size(); }
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

size_t GetDataTypeSize(TensorDataType data_type);

size_t GetElemMemSize(TensorDataType data_type, size_t element_size);

// elem_size

// 这里为了处理动态大小的模型， 所以需要根据batch_size来计算元素数量
int64_t GetElemCntFromShape(const std::vector<int64_t> &v,
                            int64_t batch_size = 1);
int64_t GetSingleBatchElemCntFromShape(const std::vector<int64_t> &v);

// mem_size
int64_t GetMemSizeFromShape(const std::vector<int64_t> &v,
                            TensorDataType data_type, int batch_size = 1);
int64_t GetSingleBatchMemSizeFromShape(const std::vector<int64_t> &v,
                                       TensorDataType data_type);

TensorBufferUPtr CreateTensorBufferCPU(TensorDataType data_type,
                                       size_t mem_size);
} // namespace inference