#pragma once

#include <cstdint>
#include <exception>
#include <string>

namespace cpputils {

class RuntimeException : public std::exception {
public:
  RuntimeException(const std::string &msg, const char *file, const char *func,
                   uint32_t line);

  virtual ~RuntimeException() {}

  virtual const char *what(void) const noexcept override {
    return what_.c_str();
  }

private:
  std::string what_;
};

} // namespace cpputils

#define THROW_RUNTIME_EXCEPTION(x)                                             \
  throw cpputils::RuntimeException(x, __FILE__, __func__, __LINE__);
