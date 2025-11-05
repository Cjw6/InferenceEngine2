#include "img_common.hpp"

#include "inference/utils/half.hpp"

namespace img_utils {

std::vector<float> Softmax(const void *input, int len,
                           inference::TensorDataType data_type) {
  if (data_type == inference::TensorDataType::kFP32) {
    const float *input_fp32 = (const float *)input;
    return Softmax(input_fp32, len / sizeof(float));
  } else if (data_type == inference::TensorDataType::kFP16) {
    return Softmax((const half_float::half *)input,
                   len / sizeof(half_float::half));
  } else {
    throw std::runtime_error("data type not supported");
  }
  return {};
}

int GetMaxFromSoftmax(const void *input, int len,
                      inference::TensorDataType data_type) {
  if (data_type == inference::TensorDataType::kFP32) {
    const float *input_fp32 = (const float *)input;
    return GetMaxFromSoftmax(input_fp32, len / sizeof(float));
  } else if (data_type == inference::TensorDataType::kFP16) {
    return GetMaxFromSoftmax((const half_float::half *)input,
                             len / sizeof(half_float::half));
  } else {
    throw std::runtime_error("data type not supported");
  }
}

void BlobNormalizeFromImage(const cv::Mat &img, void *blob,
                   inference::TensorDataType data_type) {
  if (data_type == inference::TensorDataType::kFP32) {
    BlobNormalizeFromImage(img, (float *)blob);
  } else if (data_type == inference::TensorDataType::kFP16) {
    BlobNormalizeFromImage(img, (half_float::half *)blob);
  } else {
    throw std::runtime_error("data type not supported");
  }
}

} // namespace img_utils
