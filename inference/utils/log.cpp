#include "log.h"

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <cstdarg>
#include <cstring>

void LogInit() {
  // 设置日志打印到控制台
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::trace);

  // 设置日志旋转
  auto max_size = 1024 * 1024 * 5;
  auto max_files = 3;
  auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
      "logs/inference_engine.log", max_size, max_files);
  file_sink->set_level(spdlog::level::trace);

  std::vector<spdlog::sink_ptr> sinks = {console_sink, file_sink};

  auto logger =
      std::make_shared<spdlog::logger>("", sinks.begin(), sinks.end());
  logger->set_level(spdlog::level::trace);
  logger->flush_on(spdlog::level::trace);

  spdlog::flush_every(std::chrono::milliseconds(20));
  spdlog::set_default_logger(logger);
}
