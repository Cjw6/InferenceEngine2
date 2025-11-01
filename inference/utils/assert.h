#pragma once

#include "inference/utils/log.h"

#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                 \
      LOG_CRITICAL("CHECK {} error!!!", #x);                                   \
    }                                                                          \
  } while (0);
