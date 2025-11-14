#pragma once

#include <map>
#include <sstream>
#include <string>
#include <vector>

#if __cplusplus > 201703L
#include <span>
#endif

namespace cpputils {
template <typename T> std::string ToString(const T &value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

template <typename T> std::string VectorToString(const std::vector<T> &val) {
  if (val.empty()) {
    return "";
  }

  std::stringstream stream;
  stream << "[";
  for (int i = 0; i < val.size(); ++i) {
    stream << val[i];
    if (i != val.size() - 1)
      stream << ", ";
  }
  stream << "]";
  return stream.str();
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    os << vec[i];
    if (i != vec.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

template <typename K, typename V>
std::string MapToString(const std::map<K, V> &val) {
  if (val.empty()) {
    return "";
  }
  std::stringstream stream;
  stream << "{";
  for (auto it = val.begin(); it != val.end(); ++it) {
    stream << it->first << ":" << it->second;
    if (std::next(it) != val.end())
      stream << ", ";
  }
  stream << "}";
  return stream.str();
}

template <typename K, typename V>
std::ostream &operator<<(std::ostream &os, const std::map<K, V> &map) {
  os << "{";
  for (auto it = map.begin(); it != map.end(); ++it) {
    os << it->first << ":" << it->second;
    if (std::next(it) != map.end()) {
      os << ", ";
    }
  }
  os << "}";
  return os;
}

#if __cplusplus > 201703L

template <typename T> std::string SpanToString(const std::span<T> &val) {
  if (val.empty()) {
    return "";
  }

  std::stringstream stream;
  stream << "[";
  for (int i = 0; i < val.size(); ++i) {
    stream << val[i];
    if (i != val.size() - 1)
      stream << ", ";
  }
  stream << "]";
  return stream.str();
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::span<T> &span) {
  os << "[";
  for (size_t i = 0; i < span.size(); ++i) {
    os << span[i];
    if (i != span.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

#endif

} // namespace cpputils