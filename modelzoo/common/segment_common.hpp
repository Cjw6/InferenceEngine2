#pragma once

#include <vector>
#include <string>

namespace imgutils {

struct MaskPoints {
  int x;
  int y;
};

using MaskRegion = std::vector<MaskPoints>;

} // namespace imgutils