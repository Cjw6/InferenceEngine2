#pragma once

#include <chrono>

namespace cpputils {

class ElapseTime {
public:
  ElapseTime() = default;
  ~ElapseTime() = default;

  void Restart();
  float DurationSec();
  float DurationMs();
  float DurationUs();

private:
  std::chrono::time_point<std::chrono::steady_clock> _tp =
      std::chrono::steady_clock::now();
};

inline void ElapseTime::Restart() { _tp = std::chrono::steady_clock::now(); }

inline float ElapseTime::DurationSec() {
  auto now_time = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(now_time - _tp)
             .count() /
         1000000.0f;
}

inline float ElapseTime::DurationMs() {
  auto now_time = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(now_time - _tp)
             .count() /
         1000.0f;
}

inline float ElapseTime::DurationUs() {
  auto now_time = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(now_time - _tp)
      .count();
}

} // namespace cpputils
