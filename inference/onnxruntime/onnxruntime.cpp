#include "onnxruntime.h"

#include <any>

#include "inference/onnxruntime/onnxruntime_convert.h"
#include "inference/tensor/buffer.h"
#include "inference/utils/log.h"
#include "inference/utils/to_string.h"

#include <onnxruntime_cxx_api.h>

namespace inference {

namespace {

void ParseParams(const ParamMap &params, Ort::SessionOptions &options) {}

std::string ToString(Ort::ConstTensorTypeAndShapeInfo info) {
  auto elem_type = info.GetElementType();
  auto elem_count = info.GetElementCount();
  auto dims_count = info.GetDimensionsCount();
  auto symbolic_dims = info.GetSymbolicDimensions();
  auto shape = info.GetShape();

  return fmt::format("elem_type {}, elem_count {}, dims_count {}, "
                     "symbolic_dims {}, shape {}",
                     (int)elem_type, elem_count, dims_count,
                     cpputils::VectorToString(symbolic_dims),
                     cpputils::VectorToString(shape));
}

} // namespace

class OnnxRuntimeEngineImpl {
public:
  OnnxRuntimeEngineImpl() {}
  ~OnnxRuntimeEngineImpl() {}

  int Init(const std::string &model_path, const std::string &log_id,
           const ParamMap &params);
  void Deinit();

  int Run();

  int InputsNums() const;
  const InputNodeNames &GetInputNodeNames() const;
  const InputTensorDescs &GetInputTensorDescs() const;
  InputTensorPointers GetInputTensors();

  int OutputsNums() const;
  const OutputNodeNames &GetOutputNodeNames() const;
  const OutputTensorDescs &GetOutputTensorDescs() const;
  OutputTensorPointers GetOutputTensors();

private:
  bool ready_ = false;

  Ort::AllocatorWithDefaultOptions allocator;
  std::unique_ptr<Ort::Env> env_ = nullptr;
  std::unique_ptr<Ort::Session> session_ = nullptr;

  TensorDeviceType device_type_ = kCPU;

  InputNodeNames input_node_names_;
  InputNodeNamePointers input_node_names_pointers_;
  std::vector<Ort::Value> input_ort_tensors_;
  std::vector<Ort::Value *> input_ort_tensor_pointers_;
  InputTensorDescs input_tensor_descs_;
  TensorBuffers input_tensor_buffers_;

  OutputNodeNames output_node_names_;
  OutputNodeNamePointers output_node_names_pointers_;
  std::vector<Ort::Value> output_ort_tensors_;
  std::vector<Ort::Value *> output_ort_tensor_pointers_;
  OutputTensorDescs output_tensor_descs_;
  TensorBuffers output_tensor_buffers_;
};

int OnnxRuntimeEngineImpl::Init(const std::string &model_path,
                                const std::string &log_id,
                                const ParamMap &params) {
  try {
    env_ =
        std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, log_id.c_str());

    // TODO 这里添加推理框架选项
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session_ =
        std::make_unique<Ort::Session>(*env_, model_path.c_str(), options);

    auto input_nums = session_->GetInputCount();
    input_node_names_.reserve(input_nums);
    input_node_names_pointers_.reserve(input_nums);
    input_ort_tensors_.reserve(input_nums);
    input_ort_tensor_pointers_.reserve(input_nums);

    for (int i = 0; i < input_nums; ++i) {
      auto input_name = session_->GetInputNameAllocated(i, allocator);
      input_node_names_.push_back(input_name.get());
      // input_node_names_pointers_.push_back(input_name.get());

      auto input_type_info = session_->GetInputTypeInfo(i);
      auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

      auto &tensor_desc = input_tensor_descs_[input_name.get()];
      tensor_desc.data_type = OnnxTensorDataTypeToTensorDataType(
          input_tensor_info.GetElementType());
      tensor_desc.shape = input_tensor_info.GetShape();
      tensor_desc.element_size = input_tensor_info.GetElementCount();

      auto tensor_buffer = CreateTensorBuffer(
          tensor_desc.data_type, tensor_desc.element_size, device_type_);
      input_tensor_buffers_[input_name.get()] = std::move(tensor_buffer);

      // LOG_INFO("input {}, {}", input_name.get(),
      // ToString(input_tensor_info));

      auto memory_info = Ort::MemoryInfo::CreateCpu(
          OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
      input_ort_tensors_.push_back(Ort::Value::CreateTensor(
          memory_info, (float *)input_tensor_buffers_[input_name.get()]->host(),
          tensor_desc.element_size, tensor_desc.shape.data(),
          tensor_desc.shape.size()));
    }

    auto output_nums = session_->GetOutputCount();
    output_node_names_.reserve(output_nums);
    output_node_names_pointers_.reserve(output_nums);
    output_ort_tensors_.reserve(output_nums);
    output_ort_tensor_pointers_.reserve(output_nums);
    for (int i = 0; i < output_nums; ++i) {
      auto output_name = session_->GetOutputNameAllocated(i, allocator);
      output_node_names_.push_back(output_name.get());
      // output_node_names_pointers_.push_back(output_name.get());

      auto output_type_info = session_->GetOutputTypeInfo(i);
      auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

      auto &tensor_desc = output_tensor_descs_[output_name.get()];
      tensor_desc.data_type = OnnxTensorDataTypeToTensorDataType(
          output_tensor_info.GetElementType());
      tensor_desc.shape = output_tensor_info.GetShape();
      tensor_desc.element_size = output_tensor_info.GetElementCount();

      auto tensor_buffer = CreateTensorBuffer(
          tensor_desc.data_type, tensor_desc.element_size, device_type_);
      output_tensor_buffers_[output_name.get()] = std::move(tensor_buffer);

      // LOG_INFO("output {}, {}",
      // output_name.get(),ToString(output_tensor_info));

      auto memory_info = Ort::MemoryInfo::CreateCpu(
          OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
      output_ort_tensors_.push_back(Ort::Value::CreateTensor(
          memory_info,
          (float *)output_tensor_buffers_[output_name.get()]->host(),
          tensor_desc.element_size, tensor_desc.shape.data(),
          tensor_desc.shape.size()));
    }

    // for (auto &name : output_node_names_) {
    //   output_node_names_pointers_.push_back(&name);
    // }

    for (auto &name : input_node_names_) {
      input_node_names_pointers_.push_back(name.data());
    }
    for (auto &name : output_node_names_) {
      output_node_names_pointers_.push_back(name.data());
    }

    // for (auto &ort_tensor : output_ort_tensors_) {
    //   output_ort_tensor_pointers_.push_back(&ort_tensor);
    // }
    // for (auto &ort_tensor : input_ort_tensors_) {
    //   input_ort_tensor_pointers_.push_back(&ort_tensor);
    // }

    ready_ = true;
    return 0;
  } catch (const Ort::Exception &e) {
    Deinit();
    LOG_ERROR("Ort::Session init failed: {}", e.what());
    return -1;
  }
}

void OnnxRuntimeEngineImpl::Deinit() {
  session_.reset();
  env_.reset();
  ready_ = false;
}

int OnnxRuntimeEngineImpl::Run() {
  try {
    Ort::RunOptions ops;
    session_->Run(ops, input_node_names_pointers_.data(),
                  input_ort_tensors_.data(), input_node_names_.size(),
                  output_node_names_pointers_.data(),
                  output_ort_tensors_.data(), output_node_names_.size());
    return 0;
  } catch (const Ort::Exception &e) {
    LOG_ERROR("Ort::Session run failed: {}", e.what());
    return -1;
  }
}

int OnnxRuntimeEngineImpl::InputsNums() const {
  return input_node_names_.size();
}

const InputNodeNames &OnnxRuntimeEngineImpl::GetInputNodeNames() const {
  return input_node_names_;
}

const InputTensorDescs &OnnxRuntimeEngineImpl::GetInputTensorDescs() const {
  return input_tensor_descs_;
}

InputTensorPointers OnnxRuntimeEngineImpl::GetInputTensors() {
  InputTensorPointers input_tensors;
  for (auto &[k, v] : input_tensor_descs_) {
    input_tensors.emplace(
        std::piecewise_construct, std::forward_as_tuple(k),
        std::forward_as_tuple(input_tensor_buffers_[k]->host(), v.element_size,
                              v.shape, kCPU));
  }
  return input_tensors;
}

int OnnxRuntimeEngineImpl::OutputsNums() const {
  return output_node_names_.size();
}

const OutputNodeNames &OnnxRuntimeEngineImpl::GetOutputNodeNames() const {
  return output_node_names_;
}

const OutputTensorDescs &OnnxRuntimeEngineImpl::GetOutputTensorDescs() const {
  return output_tensor_descs_;
}

OutputTensorPointers OnnxRuntimeEngineImpl::GetOutputTensors() {
  OutputTensorPointers output_tensors;
  for (auto &[k, v] : output_tensor_descs_) {
    output_tensors.emplace(
        std::piecewise_construct, std::forward_as_tuple(k),
        std::forward_as_tuple(output_tensor_buffers_[k]->host(), v.element_size,
                              v.shape, kCPU));
  }
  return output_tensors;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
OnnxRuntimeEngine::OnnxRuntimeEngine() { impl_ = new OnnxRuntimeEngineImpl(); }

OnnxRuntimeEngine::~OnnxRuntimeEngine() {
  Deinit();
  delete impl_;
}

int OnnxRuntimeEngine::Init(const std::string &model_path,
                            const std::string &log_id, const ParamMap &params) {
  return impl_->Init(model_path, log_id, params);
}

void OnnxRuntimeEngine::Deinit() { impl_->Deinit(); }

int OnnxRuntimeEngine::InputsNums() const { return impl_->InputsNums(); }

const InputNodeNames &OnnxRuntimeEngine::GetInputNodeNames() const {
  return impl_->GetInputNodeNames();
}

const InputTensorDescs &OnnxRuntimeEngine::GetInputTensorDescs() const {
  return impl_->GetInputTensorDescs();
}

InputTensorPointers OnnxRuntimeEngine::GetInputTensors() {
  return impl_->GetInputTensors();
}

int OnnxRuntimeEngine::OutputsNums() const { return impl_->OutputsNums(); }

const OutputNodeNames &OnnxRuntimeEngine::GetOutputNodeNames() const {
  return impl_->GetOutputNodeNames();
}

const OutputTensorDescs &OnnxRuntimeEngine::GetOutputTensorDescs() const {
  return impl_->GetOutputTensorDescs();
}

OutputTensorPointers OnnxRuntimeEngine::GetOutputTensors() {
  return impl_->GetOutputTensors();
}

int OnnxRuntimeEngine::Run() { return impl_->Run(); }

} // namespace inference
