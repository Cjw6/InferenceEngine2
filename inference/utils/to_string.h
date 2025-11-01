#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace cpputils {
template <typename T> std::string ToString(T value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

template <typename T> std::string VectorToString(std::vector<T> val) {
  if (val.empty()) {
    return "";
  }

  std::stringstream stream;
  stream << "[";
  for (int i = 0; i < val.size(); ++i) {
    stream << val[i];
    if (i != val.size() - 1)
      stream << ",";
  }
  stream << "]";
  return stream.str();
}

} // namespace cpputils