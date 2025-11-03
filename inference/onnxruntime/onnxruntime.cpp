#include "onnxruntime.h"

#include "inference/onnxruntime/onnxruntime_convert.h"
#include "inference/tensor/buffer.h"
#include "inference/utils/log.h"
#include "inference/utils/to_string.h"

#include <onnxruntime_cxx_api.h>

namespace inference {

namespace {

// using ParamsTable = std::unordered_map<std::string, std::string>;

bool IsDynamic(const std::vector<int64_t> &shape) {
  return std::any_of(shape.begin(), shape.end(),
                     [](int64_t dim) { return dim == -1; });
}

std::ostream &operator<<(std::ostream &s,
                         const Ort::ConstTensorTypeAndShapeInfo &info) {
  auto elem_type = info.GetElementType();
  auto elem_count = info.GetElementCount();
  auto dims_count = info.GetDimensionsCount();
  auto symbolic_dims = info.GetSymbolicDimensions();
  auto shape = info.GetShape();

  s << fmt::format("Ort::ConstTensorTypeAndShapeInfo {{elem_type {}, "
                   "elem_count {}, dims_count {}, "
                   "symbolic_dims {}, shape {}}}",
                   (int)elem_type, elem_count, dims_count,
                   cpputils::VectorToString(symbolic_dims),
                   cpputils::VectorToString(shape));
  return s;
}

TensorDesc OrtTypeInfoToTensorDesc(const Ort::TypeInfo &info) {
  TensorDesc t_desc;
  auto input_tensor_info = info.GetTensorTypeAndShapeInfo();
  auto elem_type = input_tensor_info.GetElementType();
  t_desc.data_type =
      OnnxTensorDataTypeToTensorDataType(input_tensor_info.GetElementType());
  t_desc.shape = input_tensor_info.GetShape();
  if (IsDynamic(t_desc.shape)) {
    t_desc.element_size = -1;
  } else {
    t_desc.element_size = input_tensor_info.GetElementCount();
  }
  return t_desc;
}

Ort::Value CreateOrtTensorCPU(TensorDataType data_type, void *p_data,
                              size_t p_data_element_count, const int64_t *shape,
                              size_t shape_len) {
  auto memory_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

  if (data_type == kFP32) {
    return Ort::Value::CreateTensor<float>(
        memory_info, (float *)p_data, p_data_element_count, shape, shape_len);
  } else if (data_type == kFP16) {
    return Ort::Value::CreateTensor(memory_info, (Ort::Float16_t *)p_data,
                                    p_data_element_count, shape, shape_len);
  } else if (data_type == kUint8) {
    return Ort::Value::CreateTensor<uint8_t>(
        memory_info, (uint8_t *)p_data, p_data_element_count, shape, shape_len);
  } else {
    throw std::runtime_error(fmt::format(
        "OnnxRuntimeEngineImpl::CreateTensorBuffer: unknown data type {}",
        (int)data_type));
  }
}

} // namespace

class OnnxRuntimeEngineImpl {
public:
  OnnxRuntimeEngineImpl() {}
  ~OnnxRuntimeEngineImpl() {}

  int Init(const InferenceParams &params);
  void Deinit();

  int Run(int batch_size);
  int RunStaticModel();
  int RunDynamicModel(int batch_size);

  std::string DumpModelInfo() const;
  bool IsDynamicModel() const { return dynamic_model_; }

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

  bool ready_ = false;
  bool dynamic_model_ = false;
  int max_batch_size_ = 1; // 如果为动态模型，最大batch size

  Ort::SessionOptions sess_options_;
  Ort::RunOptions run_options_;

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

  max_batch_size_ = params.max_batch_size;
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
  } else {
    throw std::runtime_error(fmt::format(
        "OnnxRuntimeEngineImpl::CreateTensorBuffer: unknown data type {}",
        (int)t_desc->data_type));
  }
  dst->push_back(std::move(v));
}

int OnnxRuntimeEngineImpl::Init(const InferenceParams &params) {
  // try {
  if (ready_) {
    LOG_WARN("OnnxRuntimeEngineImpl::Init: engine is ready, deinit first");
    Deinit();
  }

  ParseSetParams(params);
  OrtLoggingLevel ort_log_level = ORT_LOGGING_LEVEL_WARNING;
  if (params.log_level >= ORT_LOGGING_LEVEL_VERBOSE ||
      params.log_level <= ORT_LOGGING_LEVEL_FATAL) {
    ort_log_level = (OrtLoggingLevel)params.log_level;
  }

  env_ = std::make_unique<Ort::Env>(ort_log_level, "ort");
  session_ = std::make_unique<Ort::Session>(*env_, params.model_path.c_str(),
                                            sess_options_);

  auto input_nums = session_->GetInputCount();
  input_node_names_.reserve(input_nums);
  input_node_names_pointers_.reserve(input_nums);
  input_ort_tensors_.reserve(input_nums);

  for (int i = 0; i < input_nums; ++i) {
    auto input_name = session_->GetInputNameAllocated(i, allocator_);
    auto input_type_info = session_->GetInputTypeInfo(i);

    // LOG_DEBUG("input {}: {}: {}", i, input_name.get(),
    // cpputils::ToString(input_type_info.GetTensorTypeAndShapeInfo()));

    size_t mem_alloc_size = 0;
    auto tensor_desc = OrtTypeInfoToTensorDesc(input_type_info);
    if (tensor_desc.IsDynamic()) {
      dynamic_model_ = true;
      int64_t max_element_cnt =
          GetElemCntFromShape(tensor_desc.shape, max_batch_size_);
      mem_alloc_size = GetTensorMemSize(tensor_desc.data_type, max_element_cnt);
    } else {
      mem_alloc_size =
          GetTensorMemSize(tensor_desc.data_type, tensor_desc.element_size);
    }
    auto tensor_buffer =
        CreateTensorBufferCPU(tensor_desc.data_type, mem_alloc_size);

    Ort::Value ort_tensor;
    if (!tensor_desc.IsDynamic()) {
      ort_tensor =
          CreateOrtTensorCPU(tensor_desc.data_type, tensor_buffer->host(),
                             tensor_desc.element_size, tensor_desc.shape.data(),
                             tensor_desc.shape.size());
    }

    input_node_names_.push_back(input_name.get());
    input_tensor_descs_[input_name.get()] = std::move(tensor_desc);
    input_tensor_buffers_[input_name.get()] = std::move(tensor_buffer);
    input_ort_tensors_.push_back(std::move(ort_tensor));
  }

  auto output_nums = session_->GetOutputCount();
  output_node_names_.reserve(output_nums);
  output_node_names_pointers_.reserve(output_nums);
  output_ort_tensors_.reserve(output_nums);

  for (int i = 0; i < output_nums; ++i) {
    auto output_name = session_->GetOutputNameAllocated(i, allocator_);
    auto output_type_info = session_->GetOutputTypeInfo(i);

    // LOG_DEBUG("output {}: {}: {}", i, output_name.get(),
    //           cpputils::ToString(output_type_info.GetTensorTypeAndShapeInfo()));

    auto tensor_desc = OrtTypeInfoToTensorDesc(output_type_info);

    size_t mem_alloc_size = 0;
    if (tensor_desc.IsDynamic()) {
      dynamic_model_ = true;
      int64_t max_element_cnt =
          GetElemCntFromShape(tensor_desc.shape, max_batch_size_);
      mem_alloc_size = GetTensorMemSize(tensor_desc.data_type, max_element_cnt);
    } else {
      mem_alloc_size =
          GetTensorMemSize(tensor_desc.data_type, tensor_desc.element_size);
    }
    auto tensor_buffer =
        CreateTensorBufferCPU(tensor_desc.data_type, mem_alloc_size);

    Ort::Value ort_tensor;
    if (!tensor_desc.IsDynamic()) {
      ort_tensor =
          CreateOrtTensorCPU(tensor_desc.data_type, tensor_buffer->host(),
                             tensor_desc.element_size, tensor_desc.shape.data(),
                             tensor_desc.shape.size());
    }

    output_node_names_.push_back(output_name.get());
    output_tensor_descs_[output_name.get()] = std::move(tensor_desc);
    output_tensor_buffers_[output_name.get()] = std::move(tensor_buffer);
    output_ort_tensors_.push_back(std::move(ort_tensor));
  }

  for (auto &name : input_node_names_) {
    input_node_names_pointers_.push_back(name.data());
  }
  for (auto &name : output_node_names_) {
    output_node_names_pointers_.push_back(name.data());
  }

  if (inference_device_type_ == kCPU && dynamic_model_) {
    LOG_WARN("cpu inference use dynamic!!!!, It is recommended to use GPU for "
             "inference.");
  }

  ready_ = true;
  return 0;
}

void OnnxRuntimeEngineImpl::Deinit() {
  input_node_names_.clear();
  input_node_names_pointers_.clear();
  input_ort_tensors_.clear();
  input_tensor_buffers_.clear();
  input_tensor_descs_.clear();

  output_node_names_.clear();
  output_node_names_pointers_.clear();
  output_ort_tensors_.clear();
  output_tensor_buffers_.clear();
  output_tensor_descs_.clear();

  session_.reset();
  env_.reset();
  ready_ = false;
}

int OnnxRuntimeEngineImpl::RunStaticModel() {
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

int OnnxRuntimeEngineImpl::RunDynamicModel(int batch_size) {
  try {
    if (batch_size < 1 || batch_size > max_batch_size_) {
      LOG_ERROR("batch_size:{} is invalid, max_batch_size:{}", batch_size,
                max_batch_size_);
      return -1;
    }

    for (int i = 0; i < input_node_names_.size(); i++) {
      const auto &name = input_node_names_.at(i);
      const auto &tensor_desc = input_tensor_descs_.at(name);
      const auto &tensor_buffer = input_tensor_buffers_.at(name);
      auto shape = tensor_desc.shape;
      if (tensor_desc.IsDynamic()) {
        shape[0] = batch_size;
      }
      LOG_DEBUG("shape:{}", cpputils::VectorToString(shape));
      auto ort_tensor = CreateOrtTensorCPU(
          tensor_desc.data_type, tensor_buffer->host(),
          tensor_desc.element_size, shape.data(), shape.size());
      input_ort_tensors_[i] = std::move(ort_tensor);
    }

    for (int i = 0; i < output_node_names_.size(); i++) {
      const auto &name = output_node_names_.at(i);
      const auto &tensor_desc = output_tensor_descs_.at(name);
      const auto &tensor_buffer = output_tensor_buffers_.at(name);

      auto shape = tensor_desc.shape;
      if (tensor_desc.IsDynamic()) {
        shape[0] = batch_size;
      }
      LOG_DEBUG("shape:{}", cpputils::VectorToString(shape));

      auto ort_tensor = CreateOrtTensorCPU(
          tensor_desc.data_type, tensor_buffer->host(),
          tensor_desc.element_size, shape.data(), shape.size());
      output_ort_tensors_[i] = std::move(ort_tensor);
    }

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

int OnnxRuntimeEngineImpl::Run(int batch_size) {
  if (!dynamic_model_) {
    return RunStaticModel();
  } else {
    return RunDynamicModel(batch_size);
  }
}

std::string OnnxRuntimeEngineImpl::DumpModelInfo() const {
  if (!ready_) {
    LOG_ERROR("OnnxRuntimeEngineImpl::DumpModelInfo: engine is not ready");
    return "NULL, model is not ready";
  }

  std::string model_info = fmt::format("model info:\n");
  model_info += fmt::format("dynamic model: {}\n", dynamic_model_);
  model_info += fmt::format("input nums: {}\n", input_node_names_.size());
  for (auto &i_names : input_node_names_) {
    model_info +=
        fmt::format("input: {}\n{}\n", i_names,
                    cpputils::ToString(input_tensor_descs_.at(i_names)));
  }
  model_info += fmt::format("output nums: {}\n", output_node_names_.size());
  for (auto &o_names : output_node_names_) {
    model_info +=
        fmt::format("output: {}\n{}\n", o_names,
                    cpputils::ToString(output_tensor_descs_.at(o_names)));
  }
  return model_info;
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
  for (auto &[k, t_desc] : input_tensor_descs_) {
    TensorDataPointer tensor_pointer(input_tensor_buffers_.at(k)->host(),
                                     input_tensor_buffers_.at(k)->size(),
                                     t_desc.element_size, t_desc.shape,
                                     t_desc.data_type, kCPU);

    auto single_batch_elem_cnt = GetSingleBatchEleCntFromShape(t_desc.shape);
    if (t_desc.IsDynamic()) {
      int64_t single_batch_elem_cnt =
          GetSingleBatchEleCntFromShape(t_desc.shape);
      int64_t single_batch_mem_size =
          GetTensorMemSize(t_desc.data_type, single_batch_elem_cnt);
      tensor_pointer.shape[0] = 1;
      tensor_pointer.elem_cnt = single_batch_elem_cnt;
      tensor_pointer.mem_size = single_batch_mem_size;
      for (int i = 0; i < max_batch_size_; i++) {
        tensor_pointer.p_arr.push_back((char *)tensor_pointer.p +
                                       i * single_batch_mem_size);
      }
    }
    input_tensors[k] = std::move(tensor_pointer);
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
  for (auto &[k, t_desc] : output_tensor_descs_) {
    TensorDataPointer tensor_pointer(output_tensor_buffers_.at(k)->host(),
                                     output_tensor_buffers_.at(k)->size(),
                                     t_desc.element_size, t_desc.shape,
                                     t_desc.data_type, kCPU);

    auto single_batch_elem_cnt = GetSingleBatchEleCntFromShape(t_desc.shape);
    if (t_desc.IsDynamic()) {
      int64_t single_batch_elem_cnt =
          GetSingleBatchEleCntFromShape(t_desc.shape);
      int64_t single_batch_mem_size =
          GetTensorMemSize(t_desc.data_type, single_batch_elem_cnt);
      tensor_pointer.shape[0] = 1;
      tensor_pointer.elem_cnt = single_batch_elem_cnt;
      tensor_pointer.mem_size = single_batch_mem_size;
      for (int i = 0; i < max_batch_size_; i++) {
        tensor_pointer.p_arr.push_back((char *)tensor_pointer.p +
                                       i * single_batch_mem_size);
      }
    }
    output_tensors[k] = std::move(tensor_pointer);
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

int OnnxRuntimeEngine::Run(int batch_size) { return impl_->Run(batch_size); }

std::string OnnxRuntimeEngine::DumpModelInfo() const {
  return impl_->DumpModelInfo();
}

bool OnnxRuntimeEngine::IsDynamicModel() const {
  return impl_->IsDynamicModel();
}

} // namespace inference
