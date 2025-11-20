#include "buffer.h"
#include <cpptoolkit/cuda/cuda_error.hpp>
#include <cpptoolkit/assert/assert.h>

namespace inference {

HostBuffer::HostBuffer(HostBuffer &&other) noexcept
    : size_(other.size_), host_(other.host_) {
  other.host_ = nullptr;
  other.size_ = 0;
}

HostBuffer &HostBuffer::operator=(HostBuffer &&other) noexcept {
  if (this != &other) {
    free();
    size_ = other.size_;
    host_ = other.host_;
    other.host_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void HostBuffer::allocate(size_t size) {
  if (size > size_) {
    free();
    CHECK(host_ = malloc(size)); // < 分配主机内存
    size_ = size;
  }
}

void HostBuffer::free() {
  if (host_)
    ::free(host_); // < 释放主机内存
  host_ = nullptr;
  size_ = 0;
}

void *HostBuffer::device() { return nullptr; }

void *HostBuffer::host() { return host_; }

size_t HostBuffer::size() const { return size_; }

void HostBuffer::hostToDevice(cudaStream_t stream) {}

void HostBuffer::deviceToHost(cudaStream_t stream) {}

#ifdef USE_CUDA

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : size_(other.size_), device_(other.device_) {
  other.device_ = nullptr;
  other.size_ = 0;
}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
  if (this != &other) {
    free();
    size_ = other.size_;
    device_ = other.device_;
    other.device_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void DeviceBuffer::allocate(size_t size) {
  if (size > size_) {
    free();
    CUDA_CHECK(cudaMalloc(&device_, size)); // < 分配设备内存
    size_ = size;
  }
}

void DeviceBuffer::free() {
  if (device_)
    CUDA_CHECK(cudaFree(device_)); // < 释放设备内存
  device_ = nullptr;
  size_ = 0;
}

void *DeviceBuffer::device() { return device_; }

void *DeviceBuffer::host() { return nullptr; }

size_t DeviceBuffer::size() const { return size_; }

void DeviceBuffer::hostToDevice(cudaStream_t stream) {}

void DeviceBuffer::deviceToHost(cudaStream_t stream) {}

DiscreteBuffer::DiscreteBuffer(DiscreteBuffer &&other) noexcept
    : size_(other.size_), host_(other.host_), device_(other.device_) {
  other.host_ = nullptr;
  other.device_ = nullptr;
  other.size_ = 0;
}

DiscreteBuffer &DiscreteBuffer::operator=(DiscreteBuffer &&other) noexcept {
  if (this != &other) {
    free();
    size_ = other.size_;
    host_ = other.host_;
    device_ = other.device_;
    other.host_ = nullptr;
    other.device_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void DiscreteBuffer::allocate(size_t size) {
  if (size > size_) {
    free();
    CUDA_CHECK(cudaMallocHost(&host_, size)); // < 分配主机内存
    CUDA_CHECK(cudaMalloc(&device_, size));   // < 分配设备内存
    size_ = size;
  }
}

void DiscreteBuffer::free() {
  if (host_) {
    CUDA_CHECK(cudaFreeHost(host_)); // < 释放主机内存
    host_ = nullptr;                 // < 将指针置为 nullptr，避免双重释放
  }
  if (device_) {
    CUDA_CHECK(cudaFree(device_)); // < 释放设备内存
    device_ = nullptr;             // < 将指针置为 nullptr，避免双重释放
  }
  size_ = 0;
}

void *DiscreteBuffer::device() { return device_; }

void *DiscreteBuffer::host() { return host_; }

size_t DiscreteBuffer::size() const { return size_; }

void DiscreteBuffer::hostToDevice(cudaStream_t stream) {
  if (stream) {
    CUDA_CHECK(cudaMemcpyAsync(device_, host_, size_, cudaMemcpyHostToDevice,
                               stream)); // < 异步拷贝主机到设备
  } else {
    CUDA_CHECK(cudaMemcpy(device_, host_, size_,
                          cudaMemcpyHostToDevice)); // < 同步拷贝主机到设备
  }
}

void DiscreteBuffer::deviceToHost(cudaStream_t stream) {
  if (stream) {
    CUDA_CHECK(cudaMemcpyAsync(host_, device_, size_, cudaMemcpyDeviceToHost,
                               stream)); // < 异步拷贝设备到主机
  } else {
    CUDA_CHECK(cudaMemcpy(host_, device_, size_,
                          cudaMemcpyDeviceToHost)); // < 同步拷贝设备到主机
  }
}

UnifiedBuffer::UnifiedBuffer(UnifiedBuffer &&other) noexcept
    : size_(other.size_), host_(other.host_), device_(other.device_) {
  other.host_ = nullptr;
  other.device_ = nullptr;
  other.size_ = 0;
}

UnifiedBuffer &UnifiedBuffer::operator=(UnifiedBuffer &&other) noexcept {
  if (this != &other) {
    free();
    size_ = other.size_;
    host_ = other.host_;
    device_ = other.device_;
    other.host_ = nullptr;
    other.device_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void UnifiedBuffer::allocate(size_t size) {
  if (size > size_) {
    free();
    CUDA_CHECK(cudaMallocManaged(&host_, size)); // < 分配统一内存
    device_ = host_; // < 设备内存和主机内存共享同一指针
    size_ = size;
  }
}

void UnifiedBuffer::free() {
  if (host_)
    CUDA_CHECK(cudaFree(host_)); // < 释放统一内存
  host_ = nullptr;
  device_ = nullptr;
  size_ = 0;
}

void *UnifiedBuffer::device() { return device_; }

void *UnifiedBuffer::host() { return host_; }

size_t UnifiedBuffer::size() const { return size_; }

void UnifiedBuffer::hostToDevice(cudaStream_t stream) {}

void UnifiedBuffer::deviceToHost(cudaStream_t stream) {}

MappedBuffer::MappedBuffer(MappedBuffer &&other) noexcept
    : size_(other.size_), host_(other.host_), device_(other.device_) {
  other.host_ = nullptr;
  other.device_ = nullptr;
  other.size_ = 0;
}

MappedBuffer &MappedBuffer::operator=(MappedBuffer &&other) noexcept {
  if (this != &other) {
    free();
    size_ = other.size_;
    host_ = other.host_;
    device_ = other.device_;
    other.host_ = nullptr;
    other.device_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void MappedBuffer::allocate(size_t size) {
  if (size > size_) {
    free();
    CUDA_CHECK(
        cudaHostAlloc(&host_, size, cudaHostAllocMapped));    // < 分配映射内存
    CUDA_CHECK(cudaHostGetDevicePointer(&device_, host_, 0)); // < 获取设备指针
    size_ = size;
  }
}

void MappedBuffer::free() {
  if (host_)
    CUDA_CHECK(cudaFreeHost(host_)); // < 释放映射内存
  host_ = nullptr;
  device_ = nullptr;
  size_ = 0;
}

void *MappedBuffer::device() { return device_; }

void *MappedBuffer::host() { return host_; }

size_t MappedBuffer::size() const { return size_; }

void MappedBuffer::hostToDevice(cudaStream_t stream) {}

void MappedBuffer::deviceToHost(cudaStream_t stream) {}

TensorBuffer *BufferFactory::createBuffer(BufferType type) {
  switch (type) {
  case BufferType::Host:
    return new HostBuffer(); // < 创建主机内存
  case BufferType::Device:
    return new DeviceBuffer(); // < 创建设备内存
  case BufferType::Discrete:
    return new DiscreteBuffer(); // < 创建分离内存
  case BufferType::Unified:
    return new UnifiedBuffer(); // < 创建统一内存
  case BufferType::Mapped:
    return new MappedBuffer(); // < 创建映射内存
  default:
    throw std::invalid_argument("Unknown buffer type"); // < 未知的缓冲区类型
  }
}

#endif

} // namespace inference
