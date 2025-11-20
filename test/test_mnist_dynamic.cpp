#include "inference/onnxruntime/onnxruntime.h"
#include <cpptoolkit/exception/exception.h>
#include <cpptoolkit/log/log.h>
#include <cpptoolkit/strings/pystring.h>
#include <cpptoolkit/strings/to_string.h>
#include "modelzoo/common/img_common.hpp"
#include <gtest/gtest.h>

#include <filesystem>
#include <regex>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::ostream &operator<<(std::ostream &s, const fs::path &path) {
  return s << path.string();
}

inline std::vector<fs::path> GetAllFilesWithExt(const char *path,
                                                std::string ext) {
  std::vector<fs::path> files;
  // std::regex re(regex, std::regex::icase);
  for (const auto &entry : fs::directory_iterator(path)) {
    if (entry.is_regular_file()) {
      // LOG_DEBUG("entry.path(): {}, ext: {}", entry.path().string(),
      //           entry.path().extension().string());
      if (entry.path().extension() == ext) {
        files.push_back(entry.path());
      }
    }
  }
  return files;
}

} // namespace

// namespace fs = std::filesystem;

using cpptoolkit::ToString;

namespace {

const std::string fp32_model_path =
    "modelzoo/mnist_dynamic/data/mnist_dynamic.onnx";
// const std::string fp16_model_path = "modelzoo/mnist/mnist_fp16.onnx";
const std::string test_img_path = "modelzoo/mnist_dynamic/data/0001-0.jpg";
// const std::string label_path = "modelzoo/mnist/labels.txt";

void RunMnistModel(const std::string &model_path,
                   inference::DeviceType device_type) {
  cv::Mat img = cv::imread(test_img_path);
  ASSERT_FALSE(img.empty()) << "Failed to read image: " << test_img_path;
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

  int batch_size = 32;

  auto params = inference::GetDefaultOnnxRuntimeEngineParams();
  params.device_type = device_type;
  params.model_path = model_path;
  // params.log_level = 3;
  params.max_batch_size = batch_size;

  ::inference::OnnxRuntimeEngine engine;
  int ret = engine.Init(params);
  ASSERT_TRUE(ret == 0) << "Failed to init engine: " << ret;

  auto input_tensors = engine.GetInputTensors();
  ASSERT_TRUE(input_tensors.at("input").p_arr.size() == batch_size)
      << "Invalid input size: " << input_tensors.at("input").p_arr.size();
  ASSERT_TRUE(input_tensors.at("input").elem_cnt == 1 * 28 * 28)
      << "Invalid input mem size: " << input_tensors.at("input").elem_cnt;
  ASSERT_TRUE(
      input_tensors.at("input").mem_size ==
      1 * 28 * 28 *
          inference::GetDataTypeSize(input_tensors.at("input").data_type))
      << "Invalid input mem size: " << input_tensors.at("input").mem_size;

  if (model_path == fp32_model_path) {
    ASSERT_TRUE(input_tensors.at("input").data_type == inference::kFP32)
        << "Invalid input data type: "
        << ToString(input_tensors.at("input").data_type);
  } else {
    THROW_RUNTIME_EXCEPTION("Invalid model path: " + model_path);
  }
  inference::TensorShape shape = {1, 1, 28, 28};
  ASSERT_EQ(input_tensors.at("input").shape, shape)
      << "Invalid input shape: "
      << ToString(input_tensors.at("input").shape);

  for (int i = 0; i < batch_size; i++) {
    auto p = input_tensors.at("input").p_arr[i];
    ASSERT_TRUE(p != nullptr) << "Invalid input ptr: " << p;
    imgutils::BlobNormalizeFromImage(img, p,
                                      input_tensors.at("input").data_type);
  }

  ret = engine.Run(32);
  ASSERT_TRUE(ret == 0) << "Failed to run engine: " << ret;

  auto output_tensor = engine.GetOutputTensors();
  for (int i = 0; i < batch_size; i++) {
    auto &batch_tensor = output_tensor.at("output");
    auto p = batch_tensor.p_arr[i];
    ASSERT_TRUE(p != nullptr) << "Invalid output ptr: " << p;
    int max_idx = imgutils::GetMaxFromSoftmax(p, batch_tensor.mem_size,
                                               batch_tensor.data_type);
    LOG_INFO("batch {}, max idx: {}", i, max_idx);
    ASSERT_TRUE(max_idx == 0) << "Invalid max idx: " << max_idx;
  }
}

} // namespace

TEST(Mnist_Dynamic, CPU_FP32) {
  RunMnistModel(fp32_model_path, inference::kCPU);
}

TEST(Mnist_Dynamic, GPU_FP32) {
  RunMnistModel(fp32_model_path, inference::kGPU);
}

namespace {

class MnistDynamic {
public:
  void Init(const inference::InferenceParams &params) {
    int ret = engine.Init(params);
    ASSERT_TRUE(ret == 0) << "Failed to init engine: " << ret;
  }

  std::vector<int> Classify(const std::vector<cv::Mat> &imgs, int batch_size) {
    if (!engine.IsReady()) {
      THROW_RUNTIME_EXCEPTION("Engine not ready");
    }

    if (imgs.size() < batch_size) {
      THROW_RUNTIME_EXCEPTION("Invalid batch size");
    }

    if (batch_size > engine.GetMaxBatchSize()) {
      THROW_RUNTIME_EXCEPTION("Batch size exceeds max batch size");
    }

    auto i_tensor = engine.GetInputTensors().at("input");

    for (int i = 0; i < batch_size; i++) {
      auto p = i_tensor.p_arr[i];
      if (!p) {
        THROW_RUNTIME_EXCEPTION("Invalid input ptr: " + ToString(p));
      }
      imgutils::BlobNormalizeFromImage(imgs[i], p, i_tensor.data_type);
    }

    int ret = engine.Run(batch_size);
    if (ret != 0) {
      THROW_RUNTIME_EXCEPTION("Failed to run engine: " + ToString(ret));
    }

    std::vector<int> max_idxs;
    max_idxs.reserve(batch_size);
    auto output_tensor = engine.GetOutputTensors();
    for (int i = 0; i < batch_size; i++) {
      auto &batch_tensor = output_tensor.at("output");
      auto p = batch_tensor.p_arr[i];
      if (!p) {
        THROW_RUNTIME_EXCEPTION("Invalid output ptr: " + ToString(p));
      }
      // ASSERT_TRUE(p != nullptr) << "Invalid output ptr: " << p;
      int max_idx = imgutils::GetMaxFromSoftmax(p, batch_tensor.mem_size,
                                                 batch_tensor.data_type);
      max_idxs.push_back(max_idx);
    }

    return max_idxs;
  }

private:
  inference::OnnxRuntimeEngine engine;
};

const std::string data_dir = "modelzoo/mnist_dynamic/data";

struct TestImgSample {
  std::string img_path;
  cv::Mat img_data;
  int result_idx = 0;
};

std::ostream &operator<<(std::ostream &s, const TestImgSample &sample) {
  return s << "TestImgSample(img_path: " << sample.img_path
           << ", result_idx: " << sample.result_idx << ")";
}

using TestImgSampleArr = std::vector<TestImgSample>;

int GetResultFromFilename(const fs::path &filename) {
  auto file_name = filename.stem().string();
  auto splits = pystring::split(file_name, "-");
  return atoi(splits.back().c_str());
}

TestImgSampleArr PrepareTestImgSamples() {
  auto files = GetAllFilesWithExt(data_dir.c_str(), ".jpg");
  LOG_INFO("files size: {}", files.size());
  LOG_INFO("files: {}", ToString(files));

  TestImgSampleArr samples;

  for (auto &f : files) {
    TestImgSample s;
    s.img_path = f.string();
    s.result_idx = GetResultFromFilename(f);
    if (s.result_idx < 0) {
      LOG_ERROR("Invalid result idx: {}", s.result_idx);
      continue;
    }

    s.img_data = cv::imread(s.img_path);
    if (s.img_data.empty()) {
      LOG_ERROR("Failed to read image: {}", s.img_path);
      continue;
    }
    cv::cvtColor(s.img_data, s.img_data, cv::COLOR_BGR2GRAY);
    // LOG_DEBUG("{}", ToString(s));
    samples.push_back(s);
  }

  return samples;
}

void RunMnistBatch(const std::string &model_path,
                   inference::DeviceType device_type, int batch_size) {
  auto samples = PrepareTestImgSamples();
  ASSERT_TRUE(!samples.empty()) << "No test samples";

  {
    auto params = inference::GetDefaultOnnxRuntimeEngineParams();
    params.device_type = device_type;
    params.model_path = model_path;
    params.max_batch_size = batch_size;
    MnistDynamic mnist;
    mnist.Init(params);

    for (int i = 0; i < samples.size(); i++) {
      auto max_idxs = mnist.Classify({samples[i].img_data}, 1);
      ASSERT_TRUE(max_idxs.size() == 1)
          << "Invalid max idxs size: " << max_idxs.size();
      samples[i].result_idx = max_idxs[0];
    }
  }

  auto params = inference::GetDefaultOnnxRuntimeEngineParams();
  params.device_type = device_type;
  params.model_path = model_path;
  params.max_batch_size = batch_size;

  inference::OnnxRuntimeEngine engine;
  int ret = engine.Init(params);
  ASSERT_TRUE(ret == 0) << "Failed to init engine: " << ret;

  auto input_tensors = engine.GetInputTensors();
  auto i_tensor = input_tensors.at("input");

  std::vector<int> batch_sample_idx;
  for (int i = 0; i < samples.size(); i++) {
    batch_sample_idx.push_back(i % samples.size());
  }

  for (int i = 0; i < batch_size; i++) {
    auto p = i_tensor.p_arr[i];
    ASSERT_TRUE(p != nullptr) << "Invalid input ptr: " << p;
    imgutils::BlobNormalizeFromImage(samples[batch_sample_idx[i]].img_data, p,
                                      i_tensor.data_type);
  }

  ret = engine.Run(batch_size);
  ASSERT_TRUE(ret == 0) << "Failed to run engine: " << ret;

  auto output_tensor = engine.GetOutputTensors();
  auto o_tensor = output_tensor.at("output");
  ASSERT_TRUE(o_tensor.p_arr.size() == batch_size)
      << "Invalid output size: " << o_tensor.p_arr.size();

  for (int i = 0; i < batch_size; i++) {
    auto p = o_tensor.p_arr[i];
    ASSERT_TRUE(p != nullptr) << "Invalid output ptr: " << p;
    int max_idx =
        imgutils::GetMaxFromSoftmax(p, o_tensor.mem_size, o_tensor.data_type);
    LOG_INFO("batch {}, max idx: {}, expect: {}", i, max_idx,
             samples[batch_sample_idx[i]].result_idx);
    // ASSERT_TRUE(max_idx == samples[batch_sample_idx[i]].result_idx)
    //     << "Invalid max idx: " << max_idx;
  }
}

} // namespace

TEST(Mnist_Batch32, GPU_FP32) {
  RunMnistBatch(fp32_model_path, inference::kGPU, 32);
}
