#pragma once

#include "inference/utils/backtrace.h"
#include "inference/utils/log.h"

#ifdef ENABLE_ASSERTS

#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      LOG_CRITICAL("CHECK_ERROR\n{}\nCHECK:{} error!!!",                       \
                   cpputils::StacktraceToString(), #x);                        \
      assert(0);                                                               \
    }                                                                          \
  } while (0);

#define CHECK_MSG(x, msg)                                                      \
  do {                                                                         \
    if (!(x)) {                                                                \
      LOG_CRITICAL("CHECK_ERROR\n{}\nCHECK:{} error!!! tips: {}",              \
                   cpputils::StacktraceToString(), #x, msg);                   \
      assert(0);                                                               \
    }                                                                          \
  } while (0);

#else
#define CHECK(x)
#define CHECK_MSG(x, msg)
#endif

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    if (x != cudaSuccess) {                                                    \
      LOG_CRITICAL("CUDA_CHECK {} error!!!", #x);                              \
    }                                                                          \
  } while (0);
