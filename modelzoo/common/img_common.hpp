#pragma once

#include <fstream>
#include <opencv2/opencv.hpp>

#include "inference/tensor/tensor.h"

namespace img_utils {

template <typename T> int BlobNormalizeFromImage(const cv::Mat &img, T *blob) {
  int channels = img.channels();
  int imgHeight = img.rows;
  int imgWidth = img.cols;

  if (img.type() == CV_8UC3) {
    for (int c = 0; c < channels; c++) {
      for (int h = 0; h < imgHeight; h++) {
        for (int w = 0; w < imgWidth; w++) {
          blob[c * imgWidth * imgHeight + h * imgWidth + w] =
              img.at<cv::Vec3b>(h, w)[c] / 255.0f;
        }
      }
    }
  } else if (img.type() == CV_8UC1) {
    for (int h = 0; h < imgHeight; h++) {
      for (int w = 0; w < imgWidth; w++) {
        blob[h * imgWidth + w] = img.at<uchar>(h, w) / 255.0f;
      }
    }
  } else {
    throw std::runtime_error("img type must be CV_8UC1 or CV_8UC3");
  }
  return 0;
}

template <typename T> std::vector<float> Softmax(const T *input, int len) {
  T max_val = *std::max_element(input, input + len);
  T sum = static_cast<T>(0.0f);
  std::vector<float> result(len);

  for (size_t i = 0; i < len; ++i) {
    result[i] = std::exp(input[i] - max_val);
    sum += result[i];
  }

  for (size_t i = 0; i < len; ++i) {
    result[i] /= sum;
  }

  return result;
}

template <typename T> int GetMaxFromSoftmax(const T *input, int len) {
  auto result = Softmax(input, len);
  return std::distance(result.begin(),
                       std::max_element(result.begin(), result.end()));
}

inline std::vector<std::string>
ReadLabelsFromFile(const std::string &label_file) {
  std::ifstream file(label_file);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open labels file: " + label_file);
  }

  std::vector<std::string> labels;
  std::string label;
  while (std::getline(file, label)) {
    labels.emplace_back(label);
  }
  return labels;
}

std::vector<float> Softmax(const void *input, int len,
                           inference::TensorDataType data_type);

int GetMaxFromSoftmax(const void *input, int len,
                      inference::TensorDataType data_type);

void BlobNormalizeFromImage(const cv::Mat &img, void *blob,
                            inference::TensorDataType data_type);

inline std::tuple<cv::Mat, float> LetterBoxPadImage(const cv::Mat &image,
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

} // namespace img_utils
