#include "tensorrt_common.h"

#include <iostream>

namespace inference {

const std::map<nvinfer1::ILogger::Severity, std::string>
    TRTLogger::severity_map_ = {
        {nvinfer1::ILogger::Severity::kINTERNAL_ERROR, "INTERNAL_ERROR: "},
        {nvinfer1::ILogger::Severity::kERROR, "ERROR: "},
        {nvinfer1::ILogger::Severity::kWARNING, "WARNING: "},
        {nvinfer1::ILogger::Severity::kINFO, "INFO: "},
        {nvinfer1::ILogger::Severity::kVERBOSE, "VERBOSE: "}};

// 构造函数
TRTLogger::TRTLogger(nvinfer1::ILogger::Severity severity)
    : severity_(severity) {}

// 实现 log 方法
void TRTLogger::log(nvinfer1::ILogger::Severity severity,
                    const char *msg) noexcept {
  // 如果当前日志级别高于设置的日志级别，则忽略该日志
  if (severity > severity_)
    return;

  // 根据日志级别选择输出流（INFO 及以上输出到标准输出，其他输出到标准错误）
  std::ostream &stream =
      severity >= nvinfer1::ILogger::Severity::kINFO ? std::cout : std::cerr;

  // 查找日志级别的映射表，获取对应的前缀
  auto it = severity_map_.find(severity);
  if (it != severity_map_.end()) {
    stream << it->second << msg << '\n'; // 输出日志
  }
}

} // namespace inference