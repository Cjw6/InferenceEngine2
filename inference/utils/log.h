#pragma once

#include "inference/utils/macro.h"
#include <spdlog/spdlog.h>

void LogInit();

static constexpr const char *___past_last_slash(const char *str,
                                                const char *last_slash) {
#if CPP_UTILS_OS_LINUX
  return *str == '\0'  ? last_slash
         : *str == '/' ? ___past_last_slash(str + 1, str + 1)
                       : ___past_last_slash(str + 1, last_slash);
#else
  return *str == '\0'   ? last_slash
         : *str == '\\' ? ___past_last_slash(str + 1, str + 1)
                        : ___past_last_slash(str + 1, last_slash);
#endif
}

static constexpr const char *___past_last_slash(const char *str) {
  return ___past_last_slash(str, str);
}

#define LOG_TRACE(format, ...)                                                 \
  spdlog::trace("[{}:{} {}] " format, ___past_last_slash(__FILE__), __LINE__,  \
                __func__, ##__VA_ARGS__)

#define LOG_DEBUG(format, ...)                                                 \
  spdlog::debug("[{}:{} {}] " format, ___past_last_slash(__FILE__), __LINE__,  \
                __func__, ##__VA_ARGS__)

#define LOG_INFO(format, ...)                                                  \
  spdlog::info("[{}:{} {}] " format, ___past_last_slash(__FILE__), __LINE__,   \
               __func__, ##__VA_ARGS__)

#define LOG_WARN(format, ...)                                                  \
  spdlog::warn("[{}:{} {}] " format, ___past_last_slash(__FILE__), __LINE__,   \
               __func__, ##__VA_ARGS__)

#define LOG_ERROR(format, ...)                                                 \
  spdlog::error("[{}:{} {}] " format, ___past_last_slash(__FILE__), __LINE__,  \
                __func__, ##__VA_ARGS__)

#define LOG_CRITICAL(format, ...)                                              \
  spdlog::critical("[{}:{} {}] " format, ___past_last_slash(__FILE__),         \
                   __LINE__, __func__, ##__VA_ARGS__)
