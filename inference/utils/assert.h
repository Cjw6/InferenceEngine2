#pragma once

#include "inference/utils/log.h"

#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      LOG_CRITICAL("CHECK {} error!!!", #x);                                   \
    }                                                                          \
  } while (0);

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    if (x != cudaSuccess) {                                                    \
      LOG_CRITICAL("CUDA_CHECK {} error!!!", #x);                              \
    }                                                                          \
  } while (0);
