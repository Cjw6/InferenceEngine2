#pragma once

#include <map>
#include <memory>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace inference {

/**
 * @brief 抽象基类 Buffer，用于管理内存操作
 *
 */
class TensorBuffer {
public:
  virtual ~TensorBuffer() = default;

  /**
   * @brief 分配内存
   *
   * @param size 要分配的内存大小
   */
  virtual void allocate(size_t size) = 0;

  /**
   * @brief 释放内存
   *
   */
  virtual void free() = 0;

  /**
   * @brief 获取设备内存指针
   *
   * @return void* 设备内存指针
   */
  virtual void *device() = 0;

  /**
   * @brief 获取主机内存指针
   *
   * @return void* 主机内存指针
   */
  virtual void *host() = 0;

  /**
   * @brief 获取内存大小
   *
   * @return size_t 内存大小
   */
  virtual size_t size() const = 0;

#ifdef USE_CUDA
  /**
   * @brief 从主机到设备拷贝数据
   *
   * @param stream CUDA流
   */
  virtual void hostToDevice(cudaStream_t stream = nullptr) = 0;

  /**
   * @brief 从设备到主机拷贝数据
   *
   * @param stream CUDA流
   */
  virtual void deviceToHost(cudaStream_t stream = nullptr) = 0;
#endif
};

/**
 * @brief HostBuffer 类，表示内存，主要用于CPU推理
 *
 */
class HostBuffer : public TensorBuffer {
public:
  HostBuffer() : size_(0), host_(nullptr) {}
  HostBuffer(const HostBuffer &) = delete;
  HostBuffer &operator=(const HostBuffer &) = delete;
  HostBuffer(HostBuffer &&other) noexcept;
  HostBuffer &operator=(HostBuffer &&other) noexcept;
  ~HostBuffer() { free(); }

  void allocate(size_t size) override;
  void free() override;
  void *device() override;
  void *host() override;
  size_t size() const override;
#ifdef USE_CUDA
  void hostToDevice(cudaStream_t stream = nullptr) override;
  void deviceToHost(cudaStream_t stream = nullptr) override;
#endif
private:
  void *host_;  // < 设备内存指针
  size_t size_; // < 内存大小
};

#ifdef USE_CUDA

class DeviceBuffer : public TensorBuffer {
public:
  DeviceBuffer() : size_(0), device_(nullptr) {}
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept;
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;
  ~DeviceBuffer() { free(); }

  void allocate(size_t size) override;
  void free() override;
  void *device() override;
  void *host() override;
  size_t size() const override;

  void hostToDevice(cudaStream_t stream = nullptr) override;
  void deviceToHost(cudaStream_t stream = nullptr) override;

private:
  void *device_; // < 设备内存指针
  size_t size_;  // < 内存大小
};

/**
 * @brief DiscreteBuffer 类，表示具有主机和设备内存的分离内存
 *
 */
class DiscreteBuffer : public TensorBuffer {
public:
  DiscreteBuffer() : size_(0), host_(nullptr), device_(nullptr) {}
  DiscreteBuffer(const DiscreteBuffer &) = delete;
  DiscreteBuffer &operator=(const DiscreteBuffer &) = delete;
  DiscreteBuffer(DiscreteBuffer &&other) noexcept;
  DiscreteBuffer &operator=(DiscreteBuffer &&other) noexcept;
  ~DiscreteBuffer() { free(); }

  void allocate(size_t size) override;
  void free() override;
  void *device() override;
  void *host() override;
  size_t size() const override;
  void hostToDevice(cudaStream_t stream = nullptr) override;
  void deviceToHost(cudaStream_t stream = nullptr) override;

private:
  void *host_;   // < 主机内存指针
  void *device_; // < 设备内存指针
  size_t size_;  // < 内存大小
};

/**
 * @brief UnifiedBuffer 类，表示统一内存（主机和设备共享内存）
 *
 */
class UnifiedBuffer : public TensorBuffer {
public:
  UnifiedBuffer() : size_(0), host_(nullptr), device_(nullptr) {}
  UnifiedBuffer(const UnifiedBuffer &) = delete;
  UnifiedBuffer &operator=(const UnifiedBuffer &) = delete;
  UnifiedBuffer(UnifiedBuffer &&other) noexcept;
  UnifiedBuffer &operator=(UnifiedBuffer &&other) noexcept;
  ~UnifiedBuffer() { free(); }

  void allocate(size_t size) override;
  void free() override;
  void *device() override;
  void *host() override;
  size_t size() const override;
  void hostToDevice(cudaStream_t stream = nullptr) override;
  void deviceToHost(cudaStream_t stream = nullptr) override;

private:
  void *host_;   // < 主机内存指针
  void *device_; // < 设备内存指针
  size_t size_;  // < 内存大小
};

#endif

/**
 * @brief MappedBuffer 类，表示映射内存（主机和设备共享映射内存）
 *
 */
class MappedBuffer : public TensorBuffer {
public:
  MappedBuffer() : size_(0), host_(nullptr), device_(nullptr) {}
  MappedBuffer(const MappedBuffer &) = delete;
  MappedBuffer &operator=(const MappedBuffer &) = delete;
  MappedBuffer(MappedBuffer &&other) noexcept;
  MappedBuffer &operator=(MappedBuffer &&other) noexcept;
  ~MappedBuffer() { free(); }

  void allocate(size_t size) override;
  void free() override;
  void *device() override;
  void *host() override;
  size_t size() const override;
  void hostToDevice(cudaStream_t stream = nullptr) override;
  void deviceToHost(cudaStream_t stream = nullptr) override;

private:
  void *host_;   // < 主机内存指针
  void *device_; // < 设备内存指针
  size_t size_;  // < 内存大小
};

/**
 * @brief Buffer 类型枚举，用于选择不同类型的 Buffer
 *
 */
enum class BufferType {
  Host,
  Device,   // < 设备内存
  Discrete, // < 分离内存（主机和设备都有内存）
  Unified,  // < 统一内存（设备和主机共享内存）
  Mapped    // < 映射内存（用于NVIDIA集成设备）
};

/**
 * @brief Buffer 工厂类，根据类型创建不同的 Buffer
 *
 */
class BufferFactory {
public:
  /**
   * @brief 创建指定类型的 Buffer
   *
   * @param type 要创建的 Buffer 类型
   * @return 指定类型的 Buffer 智能指针
   */
  static TensorBuffer *createBuffer(BufferType type);
};

using TensorBufferUPtr = std::unique_ptr<TensorBuffer>;
using TensorBuffers = std::map<std::string, TensorBufferUPtr>;

} // namespace inference
