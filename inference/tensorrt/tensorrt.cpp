#include "tensorrt.h"
#include "inference/tensor/buffer.h"
#include "tensorrt_common.h"
#include "tensorrt_convert.h"
#include "tensorrt_to_string.h"

#include <cpptoolkit/exception/exception.h>
#include <cpptoolkit/fs/file.h>
#include <cpptoolkit/log/log.h>
#include <cpptoolkit/strings/to_string.h>
#include <cpptoolkit_cuda/cuda_device.hpp>

#include <NvInfer.h>

using cpptoolkit::ToString;

namespace inference {

class TensorRTEngineImpl {
public:
  int Init(const InferenceParamsV2 &params) {

    int ret = 0;
    do {
      logger_ =
          std::make_unique<TRTLogger>(nvinfer1::ILogger::Severity::kWARNING);

      // 创建 TensorRT runtime
      runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
          nvinfer1::createInferRuntime(*logger_));
      if (!runtime_) {
        LOG_ERROR("Failed to create TensorRT runtime.");
        ret = -1;
        break;
      }

      if (params.model_path.empty() && params.model_data.empty()) {
        LOG_ERROR("model path  is empty, model data is empty!!!!");
        ret = -1;
        break;
      }

      if (!params.model_path.empty()) {
        auto model_data = cpptoolkit::ReadAllFileData(params.model_path);
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(model_data.data(),
                                            model_data.size()));
        if (!engine_) {
          LOG_ERROR("Failed to deserialize CUDA engine. model path: {}",
                    params.model_path);
        }
      } else {
        auto model_data = cpptoolkit::ReadAllFileData(params.model_path);
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(model_data.data(),
                                            model_data.size()));
        if (!engine_) {
          LOG_ERROR("Failed to deserialize CUDA engine. model mem error!!! "
                    "model data size: {}",
                    params.model_data.size());
        }
      }
      if (!engine_) {
        ret = -1;
        break;
      }

      // 创建执行上下文
      context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
          engine_->createExecutionContext());
      if (!context_) {
        LOG_ERROR("Failed to create execution context.");
        ret = -1;
        break;
      }

      ret = InitIO();
      if (ret != 0) {
        LOG_ERROR("Failed to init IO.");
        break;
      }

    } while (0);

    if (ret != 0) {
      Deinit();
    }

    is_ready_ = true;
    return 0;
  }

  void Deinit() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
    logger_.reset();

    input_node_names_.clear();
    input_tensor_buffers_.clear();
    input_tensor_descs_.clear();

    output_node_names_.clear();
    output_tensor_buffers_.clear();
    output_tensor_descs_.clear();

    is_ready_ = false;
    is_dynamic_ = false;
  }

  int Warmup() {
    if (!is_ready_) {
      return -1;
    }

    auto ret = context_->enqueueV3(stream_.get());
    LOG_INFO("enqueue inference ret: {}", ret);

    if (!ret) {
      LOG_ERROR("Failed to warmup inference.");
      return -1;
    }
    cudaStreamSynchronize(stream_.get());
    return 0;
  }

  int Run() {
    if (!is_ready_) {
      return -1;
    }

    auto ret = context_->enqueueV3(stream_.get());
    LOG_INFO("enqueue inference ret: {}", ret);

    if (!ret) {
      LOG_ERROR("Failed to enqueue inference.");
      return -1;
    }
    cudaStreamSynchronize(stream_.get());
    return 0;
  }

  bool IsReady() const { return is_ready_; }

  std::string DumpModelInfo() const {
    if (!is_ready_) {
      return "model not ready.";
    }

    std::string msg;
    for (auto &[k, v] : input_tensor_descs_) {
      msg += "input node name: " + k + "\n";
      msg += "input node shape: " + ToString(v.shape) + "\n";
      msg += "input node dtype: " + ToString(v.data_type) + "\n";
    }
    for (auto &[k, v] : output_tensor_descs_) {
      msg += "output node name: " + k + "\n";
      msg += "output node shape: " + ToString(v.shape) + "\n";
      msg += "output node dtype: " + ToString(v.data_type) + "\n";
    }

    return msg;
  }

  const InputNodeNames &GetInputNodeNames() const { return input_node_names_; }

  const InputTensorDescs &GetInputTensorDescs() const {
    return input_tensor_descs_;
  }

  InputTensorPointers GetInputTensors(DeviceType device = kCPU) {
    InputTensorPointers itp;
    for (auto &[k, v] : input_tensor_buffers_) {
      const auto &itd = input_tensor_descs_.at(k);
      TensorDataPointer tdp;

      if (device == kCPU) {
        tdp.p = v->host();
      } else if (device == kGPU) {
        tdp.p = v->device();
      } else {
        THROW_RUNTIME_EXCEPTION(fmt::format(
            "device type error, {} not support!!", ToString(device)));
      }
      tdp.mem_size = GetMemSizeFromShape(itd.shape, itd.data_type);
      tdp.elem_cnt = GetElemCntFromShape(itd.shape);
      tdp.shape = itd.shape;
      tdp.data_type = itd.data_type;
      tdp.device_type = device;
      itp[k] = tdp;
    }
    return itp;
  }

  const OutputNodeNames &GetOutputNodeNames() const {
    return output_node_names_;
  }

  const OutputTensorDescs &GetOutputTensorDescs() const {
    return output_tensor_descs_;
  }

  InputTensorPointers GetOutputTensors(DeviceType device = kCPU) {
    OutputTensorPointers otp;
    for (auto &[k, v] : output_tensor_buffers_) {
      const auto &otd = output_tensor_descs_.at(k);
      TensorDataPointer tdp;

      if (device == kCPU) {
        tdp.p = v->host();
      } else if (device == kGPU) {
        tdp.p = v->device();
      } else {
        THROW_RUNTIME_EXCEPTION(fmt::format(
            "device type error, {} not support!!", ToString(device)));
      }
      tdp.mem_size = GetMemSizeFromShape(otd.shape, otd.data_type);
      tdp.elem_cnt = GetElemCntFromShape(otd.shape);
      tdp.shape = otd.shape;
      tdp.data_type = otd.data_type;
      tdp.device_type = device;
      otp[k] = tdp;
    }
    return otp;
  }

  void CopyInputToDevice() {
    if (!is_ready_) {
      return;
    }
    for (auto &[k, v] : input_tensor_buffers_) {
      LOG_INFO("Copy input to device: {}", k);
      v->hostToDevice(stream_.get());
    }
  }

  void CopyOutputToHost() {
    if (!is_ready_) {
      return;
    }
    for (auto &[k, v] : output_tensor_buffers_) {
      LOG_INFO("Copy output to host: {}", k);
      v->deviceToHost(stream_.get());
    }
    cudaStreamSynchronize(stream_.get());
  }

private:
  int InitIO() {
    auto io_nums = engine_->getNbIOTensors();
    for (int i = 0; i < io_nums; i++) {
      auto io_name = engine_->getIOTensorName(i);
      auto io_tensor_mode = engine_->getTensorIOMode(io_name);
      auto io_tensor_dtype = engine_->getTensorDataType(io_name);
      auto io_tensor_dims = engine_->getTensorShape(io_name);

      LOG_INFO("dtype: {}", (int)io_tensor_dtype);

      auto t_shape = TensorRTConvertShape(io_tensor_dims);
      auto t_dtype = TensorRTConvertDataType(io_tensor_dtype);

      // 这里只支持 动态维度
      TensorDesc desc;
      desc.data_type = std::move(t_dtype);
      desc.shape = t_shape;
      desc.element_size = GetElemCntFromShape(t_shape);

      auto mem_size = GetMemSizeFromShape(t_shape, t_dtype);
      // 我这里使用的是 RTX显卡测试，使用分离内存
      auto t_buffer = CreateTensorBuffer(mem_size, BufferType::Discrete);

      context_->setTensorAddress(io_name, t_buffer->device());

      if (io_tensor_mode == nvinfer1::TensorIOMode::kINPUT) {
        input_node_names_.push_back(io_name);
        input_tensor_descs_[io_name] = std::move(desc);
        input_tensor_buffers_[io_name] = std::move(t_buffer);
      } else if (io_tensor_mode == nvinfer1::TensorIOMode::kOUTPUT) {
        output_node_names_.push_back(io_name);
        output_tensor_descs_[io_name] = std::move(desc);
        output_tensor_buffers_[io_name] = std::move(t_buffer);
      }
    }
    return 0;
  }

  bool is_ready_ = false;
  bool is_dynamic_ = false;

  ::std::unique_ptr<TRTLogger> logger_;
  ::std::unique_ptr<nvinfer1::IRuntime> runtime_;
  ::std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  ::std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // input
  InputNodeNames input_node_names_;
  TensorBuffers input_tensor_buffers_;
  InputTensorDescs input_tensor_descs_;

  // output
  OutputNodeNames output_node_names_;
  TensorBuffers output_tensor_buffers_;
  OutputTensorDescs output_tensor_descs_;

  cpptoolkit::CudaStream stream_;
};

TensorRTEngine::TensorRTEngine() {
  impl_ = std::make_unique<TensorRTEngineImpl>();
}

TensorRTEngine::~TensorRTEngine() {}

int TensorRTEngine::Init(const InferenceParams &params) {
  LOG_ERROR("TensorRT not support!!!");
  return -1;
}

int TensorRTEngine::Init(const InferenceParamsV2 &params) {
  return impl_->Init(params);
}

void TensorRTEngine::Deinit() {}

int TensorRTEngine::Warmup() { return impl_->Warmup(); }

int TensorRTEngine::Run(int batch_size) { return impl_->Run(); }

bool TensorRTEngine::IsReady() const { return impl_->IsReady(); }

std::string TensorRTEngine::DumpModelInfo() const {
  return impl_->DumpModelInfo();
}

bool TensorRTEngine::IsDynamicModel() const { return false; }

int TensorRTEngine::GetMaxBatchSize() const { return 0; }

int TensorRTEngine::InputsNums() const {
  return impl_->GetInputTensorDescs().size();
}

const InputNodeNames &TensorRTEngine::GetInputNodeNames() const {
  return impl_->GetInputNodeNames();
}

const InputTensorDescs &TensorRTEngine::GetInputTensorDescs() const {
  return impl_->GetInputTensorDescs();
}

InputTensorPointers TensorRTEngine::GetInputTensors(DeviceType device) {
  LOG_INFO("device {}", ToString(device));
  return impl_->GetInputTensors(device);
}

int TensorRTEngine::OutputsNums() const {
  return impl_->GetOutputNodeNames().size();
}

const OutputNodeNames &TensorRTEngine::GetOutputNodeNames() const {
  return impl_->GetOutputNodeNames();
}

const OutputTensorDescs &TensorRTEngine::GetOutputTensorDescs() const {
  return impl_->GetOutputTensorDescs();
}

OutputTensorPointers TensorRTEngine::GetOutputTensors(DeviceType device) {
  LOG_INFO("device {}", ToString(device));
  return impl_->GetOutputTensors(device);
}

void TensorRTEngine::CopyInputToDevice() { impl_->CopyInputToDevice(); }

void TensorRTEngine::CopyOutputToHost() { impl_->CopyOutputToHost(); }

} // namespace inference