#include "inference/utils/backtrace.h"
#include "inference/utils/exception.h"
#include "inference/utils/log.h"
#include "modelzoo/common/filesystem_common.hpp"
#include "modelzoo/common/img_common.hpp"
#include "modelzoo/mnist_add_process/mnist_add_process.hpp"

void ClassifyOneImage() {
  CPP_UTILS_TRY {
    const std::string model_path =
        "modelzoo/mnist_add_process/data/mnist_add_process.onnx";
    const std::string img_path = "modelzoo/mnist_add_process/data/0001-0.jpg";

    LogInit();

    auto params = inference::GetDefaultOnnxRuntimeEngineParams();
    params.model_path = model_path;
    params.log_level = 2;

    modelzoo::MnistAddProcess mnist;
    int ret = mnist.Init(params);
    if (ret != 0) {
      LOG_CRITICAL("mnist init failed");
      return;
    }
    if (!mnist.IsReady()) {
      LOG_CRITICAL("mnist not ready");
      return;
    }

    cv::Mat img = cv::imread(img_path);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    auto [idx, confidence] = mnist.Classify(img);
    LOG_INFO("idx: {}, confidence: {}", idx, confidence);
  }
  CPP_UTILS_CATCH(const std::exception &e) {
    cpputils::PrettyPrintException(e);
  }
}

void ClassifyDir() {
  CPP_UTILS_TRY {
    const std::string model_path =
        "modelzoo/mnist_add_process/data/mnist_add_process.onnx";
    const std::string jpg_dir = "modelzoo/mnist_dynamic/data";
    const std::string jpg_ext = ".jpg";

    LogInit();

    auto img_paths = cpputils::GetAllFilesWithExt(jpg_dir.c_str(), jpg_ext);
    LOG_INFO("img_paths:{}", cpputils::VectorToString(img_paths));

    std::vector<cv::Mat> imgs;
    for (auto &f : img_paths) {
      auto img = cv::imread(f.string());
      if (!img.empty()) {
        if (img.type() != CV_8UC1) {
          cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }
        imgs.push_back(img);
      }
    }

    auto params = inference::GetDefaultOnnxRuntimeEngineParams();
    params.model_path = model_path;
    params.log_level = 2;

    modelzoo::MnistAddProcess mnist;
    int ret = mnist.Init(params);
    if (ret != 0) {
      LOG_CRITICAL("mnist init failed");
      return;
    }
    if (!mnist.IsReady()) {
      LOG_CRITICAL("mnist not ready");
      return;
    }

    for (int i = 0; i < img_paths.size(); i++) {
      auto &img = imgs[i];
      auto [idx, confidece] = mnist.Classify(img);
      LOG_INFO("img_path: {}, idx: {}, confidence: {}", img_paths[i].string(),
               idx, confidece);
    }
  }
  CPP_UTILS_CATCH(const std::exception &e) {
    cpputils::PrettyPrintException(e);
  }
}

int main() {
  ClassifyOneImage();
  ClassifyDir();
}