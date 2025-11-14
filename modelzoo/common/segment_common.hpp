#pragma once

#include "inference/utils/to_string.h"
#include <ostream>
#include <vector>

namespace imgutils {

struct MaskPoints {
  int x;
  int y;
};

using MaskRegion = std::vector<MaskPoints>;

inline std::ostream &operator<<(std::ostream &s, const MaskPoints &mask) {
  return s << "MaskPoints(x=" << mask.x << " y=" << mask.y << ")";
}

inline std::ostream &operator<<(std::ostream &s, const MaskRegion &mask) {
  return s << cpputils::VectorToString(mask);
}

} // namespace imgutils