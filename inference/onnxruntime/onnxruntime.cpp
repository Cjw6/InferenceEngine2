#include "onnxruntime.h"

#include "inference/onnxruntime/onnxruntime_convert.h"
#include "inference/tensor/buffer.h"
#include "inference/utils/log.h"
#include "inference/utils/map.h"
#include "inference/utils/to_string.h"

#include <onnxruntime_cxx_api.h>

namespace inference {

namespace {

using ParamsTable = std::unordered_map<std::string, std::string>;

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

// ParamsTable GetORT_CUDA_ProviderOptions(const InferenceParams &params) {
//   ParamsTable params_table;
//   for (auto &[key, value] : params.ext_params) {
//     params_table[key] = value;
//   }
//   return params_table;
// }

} // namespace

class OnnxRuntimeEngineImpl {
public:
  OnnxRuntimeEngineImpl() {}
  ~OnnxRuntimeEngineImpl() {}

  int Init(const InferenceParams &params);
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
  void ParseSetParams(const InferenceParams &params);
  void CreateTensorBuffer(const char *name, bool input, bool output);

  Ort::SessionOptions sess_options_;
  Ort::RunOptions run_options_;

  bool ready_ = false;

  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Env> env_ = nullptr;
  std::unique_ptr<Ort::Session> session_ = nullptr;

  DeviceType inference_device_type_ = kCPU;
  TensorDataType inference_tensor_type_ = kFP32;

  InputNodeNames input_node_names_;
  InputNodeNamePointers input_node_names_pointers_;
  std::vector<Ort::Value> input_ort_tensors_;
  // std::vector<Ort::Value *> input_ort_tensor_pointers_;
  InputTensorDescs input_tensor_descs_;
  TensorBuffers input_tensor_buffers_;

  OutputNodeNames output_node_names_;
  OutputNodeNamePointers output_node_names_pointers_;
  std::vector<Ort::Value> output_ort_tensors_;
  // std::vector<Ort::Value *> output_ort_tensor_pointers_;
  OutputTensorDescs output_tensor_descs_;
  TensorBuffers output_tensor_buffers_;
};

namespace {}

void OnnxRuntimeEngineImpl::ParseSetParams(const InferenceParams &params) {
  inference_device_type_ = params.device_type;
  sess_options_.SetIntraOpNumThreads(params.intra_op_num_threads);
  sess_options_.SetInterOpNumThreads(params.inter_op_num_threads);
  sess_options_.SetGraphOptimizationLevel(
      (GraphOptimizationLevel)params.graph_opt_level);
  sess_options_.SetExecutionMode((ExecutionMode)params.exe_mode);

  if (inference_device_type_ == kCPU) {
    LOG_INFO("use cpu inference");
  } else if (inference_device_type_ == kGPU) {
    LOG_INFO("use gpu inference, device id {}", params.device_id);
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = params.device_id;
    // https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
    sess_options_.AppendExecutionProvider_CUDA(cuda_options);
  } else {
    throw std::runtime_error(fmt::format(
        "OnnxRuntimeEngineImpl::ParseParams: unknown device type {}",
        (int)inference_device_type_));
  }
}

void OnnxRuntimeEngineImpl::CreateTensorBuffer(const char *name, bool input,
                                               bool output) {
  auto memory_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

  TensorBuffers *tensor_buffers;
  std::map<std::string, TensorDesc> *desc;
  std::vector<Ort::Value> *dst;
  if (input) {
    dst = &input_ort_tensors_;
    desc = &input_tensor_descs_;
    tensor_buffers = &input_tensor_buffers_;
  } else if (output) {
    dst = &output_ort_tensors_;
    desc = &output_tensor_descs_;
    tensor_buffers = &output_tensor_buffers_;
  } else {
    throw std::runtime_error(fmt::format(
        "OnnxRuntimeEngineImpl::CreateTensorBuffer: unknown input/output {}",
        name));
  }

  auto *t_desc = &(*desc)[name];
  Ort::Value v;

  if (t_desc->data_type == kFP32) {
    v = Ort::Value::CreateTensor<float>(
        memory_info, (float *)tensor_buffers->at(name)->host(),
        t_desc->element_size, t_desc->shape.data(), t_desc->shape.size());
  } else if (t_desc->data_type == kFP16) {
    v = Ort::Value::CreateTensor(
        memory_info, (Ort::Float16_t *)tensor_buffers->at(name)->host(),
        t_desc->element_size, t_desc->shape.data(), t_desc->shape.size());
  }
  dst->push_back(std::move(v));
}

int OnnxRuntimeEngineImpl::Init(const InferenceParams &params) {
  try {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ort");

    ParseSetParams(params);

    session_ = std::make_unique<Ort::Session>(*env_, params.model_path.c_str(),
                                              sess_options_);

    auto input_nums = session_->GetInputCount();
    input_node_names_.reserve(input_nums);
    input_node_names_pointers_.reserve(input_nums);
    input_ort_tensors_.reserve(input_nums);
    // input_ort_tensor_pointers_.reserve(input_nums);

    for (int i = 0; i < input_nums; ++i) {
      auto input_name = session_->GetInputNameAllocated(i, allocator_);
      input_node_names_.push_back(input_name.get());
      // input_node_names_pointers_.push_back(input_name.get());

      auto input_type_info = session_->GetInputTypeInfo(i);
      auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();

      auto &tensor_desc = input_tensor_descs_[input_name.get()];
      tensor_desc.data_type = OnnxTensorDataTypeToTensorDataType(
          input_tensor_info.GetElementType());
      tensor_desc.shape = input_tensor_info.GetShape();
      tensor_desc.element_size = input_tensor_info.GetElementCount();

      // TODO 如果是其他数据类型呢？
      auto tensor_buffer = CreateTensorHostBuffer(tensor_desc.data_type,
                                                  tensor_desc.element_size,
                                                  inference_device_type_);
      input_tensor_buffers_[input_name.get()] = std::move(tensor_buffer);

      CreateTensorBuffer(input_name.get(), true, false);

      // auto memory_info = Ort::MemoryInfo::CreateCpu(
      //     OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
      // input_ort_tensors_.push_back(Ort::Value::CreateTensor(
      //     memory_info, (float
      //     *)input_tensor_buffers_[input_name.get()]->host(),
      //     tensor_desc.element_size, tensor_desc.shape.data(),
      //     tensor_desc.shape.size()));
    }

    auto output_nums = session_->GetOutputCount();
    output_node_names_.reserve(output_nums);
    output_node_names_pointers_.reserve(output_nums);
    output_ort_tensors_.reserve(output_nums);
    // output_ort_tensor_pointers_.reserve(output_nums);
    for (int i = 0; i < output_nums; ++i) {
      auto output_name = session_->GetOutputNameAllocated(i, allocator_);
      output_node_names_.push_back(output_name.get());
      // output_node_names_pointers_.push_back(output_name.get());

      auto output_type_info = session_->GetOutputTypeInfo(i);
      auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();

      auto &tensor_desc = output_tensor_descs_[output_name.get()];
      tensor_desc.data_type = OnnxTensorDataTypeToTensorDataType(
          output_tensor_info.GetElementType());
      tensor_desc.shape = output_tensor_info.GetShape();
      tensor_desc.element_size = output_tensor_info.GetElementCount();

      auto tensor_buffer = CreateTensorHostBuffer(tensor_desc.data_type,
                                                  tensor_desc.element_size,
                                                  inference_device_type_);
      output_tensor_buffers_[output_name.get()] = std::move(tensor_buffer);

      // TODO 如果是其他数据类型呢？
      // auto memory_info = Ort::MemoryInfo::CreateCpu(
      //     OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
      // output_ort_tensors_.push_back(Ort::Value::CreateTensor(
      //     memory_info,
      //     (float *)output_tensor_buffers_[output_name.get()]->host(),
      //     tensor_desc.element_size, tensor_desc.shape.data(),
      //     tensor_desc.shape.size()));

      CreateTensorBuffer(output_name.get(), false, true);
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

int OnnxRuntimeEngine::Init(const InferenceParams &params) {
  return impl_->Init(params);
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
