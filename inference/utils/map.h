#pragma once

#include <map>
#include <string>

namespace cpputils {

template <typename K, typename V>
const V *MapGet(const std::map<K, V> &src_map, const K &key) {
  auto iter = src_map.find(key);
  if (iter == src_map.end()) {
    return nullptr;
  }
  return &(iter->second);
}

template <typename V>
const V *MapGet(const std::map<std::string, V> &src_map,
                const std::string &key) {
  auto iter = src_map.find(key);
  if (iter == src_map.end()) {
    return nullptr;
  }
  return &(iter->second);
}

} // namespace cpputils