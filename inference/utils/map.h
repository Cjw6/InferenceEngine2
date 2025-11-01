#pragma once

#include <map>

namespace cpputils {

template <typename K, typename V>
V *MapGetValue(const std::map<K, V> &map, const K &key) {
  auto it = map.find(key);
  if (it == map.end()) {
    return nullptr;
  }
  return &it->second;
}

} // namespace cpputils