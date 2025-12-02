#pragma once

#include <NvInfer.h>
#include <map>
#include <string>

namespace inference {
/**
 * @class TRTLogger
 * @brief TensorRT 日志记录器类，继承自 nvinfer1::ILogger。
 * 用于管理 TensorRT 的日志记录行为，支持设置日志级别并重写日志记录方法。
 */
class TRTLogger : public nvinfer1::ILogger {
public:
  /**
   * @brief 构造函数，允许设置日志级别，默认为 INFO。
   * @param severity 日志级别，默认为 nvinfer1::ILogger::Severity::kINFO。
   */
  explicit TRTLogger(nvinfer1::ILogger::Severity severity =
                         nvinfer1::ILogger::Severity::kINFO);

  // 禁用拷贝和移动语义
  TRTLogger(const TRTLogger &) = delete;            // < 禁用拷贝构造函数
  TRTLogger &operator=(const TRTLogger &) = delete; // < 禁用拷贝赋值运算符
  TRTLogger(TRTLogger &&) = delete;                 // < 禁用移动构造函数
  TRTLogger &operator=(TRTLogger &&) = delete;      // < 禁用移动赋值运算符

  /**
   * @brief 重写 TensorRT 的日志记录方法。
   * @param severity 日志级别。
   * @param msg 日志消息。
   */
  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override;

private:
  nvinfer1::ILogger::Severity severity_; // < 当前日志级别

  /**
   * @brief 日志级别与前缀的映射表。
   * 用于将日志级别转换为可读的字符串前缀。
   */
  static const std::map<nvinfer1::ILogger::Severity, std::string> severity_map_;
};
} // namespace inference