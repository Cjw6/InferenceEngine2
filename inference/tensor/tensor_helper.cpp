#include "inference/tensor/tensor_helper.h"
#include "inference/tensor/tensor.h"
#include "inference/utils/assert.h"
#include "inference/utils/half.hpp"
#include "inference/utils/log.h"
#include "inference/utils/to_string.h"

#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xnpy.hpp>

#include <filesystem>
namespace fs = std::filesystem;

namespace inference {

int SaveTensorDataToFile(TensorDataPointer *data,
                         const std::string &file_path) {
  if (!data) {
    LOG_ERROR("buffer is nullptr");
    return -1;
  }

  if (file_path.empty()) {
    LOG_ERROR("file_path is empty");
    return -1;
  }

  if (data->data_type == TensorDataType::kFP32) {
    auto t = xt::adapt((float *)data->p, data->shape);
    xt::dump_npy(file_path, t);
  } else {
    LOG_ERROR("unsupported data type {}", cpputils::ToString(data->data_type));
    return -1;
  }

  return 0;
}

TensorData LoadTensorDataFromFile(const std::string &file_path) {
  xt::xarray<float> float_array{xt::load_npy<float>(file_path)};
  auto shape = float_array.shape();

  TensorShape shape_dst;
  shape_dst.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_dst.push_back(static_cast<int64_t>(shape[i]));
  }
  size_t mem_size = GetMemSizeFromShape(shape_dst, TensorDataType::kFP32);
  auto t_data = CreateTensorBufferCPU(TensorDataType::kFP32, mem_size);
  std::memcpy(t_data->host(), float_array.data(), mem_size);

  TensorData data;
  data.pointer.p = t_data->host();
  data.pointer.shape = shape_dst;
  data.pointer.data_type = TensorDataType::kFP32;
  data.pointer.mem_size = mem_size;
  data.data = std::move(t_data);
  return data;
}

} // namespace inference
