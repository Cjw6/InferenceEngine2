#include "inference/utils/log.h"
#include "inference/utils/to_string.h"

#include <xtensor/containers/xadapt.hpp>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/io/xnpy.hpp>

// namespace infer = inference;

using cpputils::ToString;

template <class T> std::string XtToString(const T &arr) {
  std::ostringstream ss;
  ss << xt::adapt(arr);
  return ss.str();
}

int main() {
  LogInit();
  std::string yolo_seg_prediction_npy =
      "experiment/xtensor_seg/data/prediction0.npy";
  xt::xarray<float> arr = xt::load_npy<float>(yolo_seg_prediction_npy);

  // std::cout << xt::adapt(arr.shape());
  // std::cout << xt::adapt(yolo_seg_prediction.shape()) << std::endl;
  // std::ostringstream ss;
  // ss << xt::adapt(arr.shape());
  // std::cout << xt::adapt(yolo_seg_prediction.shape());
  LOG_INFO("yolo_seg_prediction shape: {}", ::ToString(xt::adapt(arr.shape())));
  int class_cnt = 80;

  // auto result = xt::hsplit(arr, 3);
  // LOG_INFO("result size: {}", result.size());
  // for (int i = 0; i < result.size(); ++i) {
    // LOG_INFO("result[{}] shape: {}", i, XtToString(result[i].shape()));
  // }
  // std::cout <<   result[0].shape() << std::endl;
}
