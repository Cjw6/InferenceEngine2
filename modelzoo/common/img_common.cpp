#include "img_common.hpp"

#include "inference/utils/half.hpp"

namespace imgutils {

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

std::tuple<cv::Mat, float> LetterBoxPadImage(const cv::Mat &image,
                                             const cv::Size &new_shape) {
  cv::Mat iImg, oImg;
  iImg = image;
  if (image.channels() == 3) {
    oImg = iImg.clone();
    cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
  } else {
    cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
  }

  std::vector<int> iImgSize = {new_shape.width, new_shape.height};
  float resizeScales_ = 1.0;
  if (iImg.cols >= iImg.rows) {
    resizeScales_ = iImg.cols / (float)iImgSize.at(0);
    cv::resize(oImg, oImg,
               cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales_)));
  } else {
    resizeScales_ = iImg.rows / (float)iImgSize.at(0);
    cv::resize(oImg, oImg,
               cv::Size(int(iImg.cols / resizeScales_), iImgSize.at(1)));
  }
  cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
  oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
  oImg = tempImg;
  return {oImg, resizeScales_};
}

static int GetRandomLimit(int min, int max) {
  int range = max - min;
  int rand_num = rand() % range + min;
  return rand_num;
}

std::vector<cv::Scalar> GetRandomColor(int num) {
  std::vector<cv::Scalar> colors;
  for (int i = 0; i < num; i++) {
    int b = GetRandomLimit(50, 200);
    int g = GetRandomLimit(50, 200);
    int r = GetRandomLimit(50, 200);
    colors.push_back(cv::Scalar(b, g, r));
  }
  return colors;
}

} // namespace imgutils
