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

void PrettyPrintException(const std::exception &e);

} // namespace cpputils

#define THROW_RUNTIME_EXCEPTION(x)                                             \
  throw cpputils::RuntimeException(x, __FILE__, __func__, __LINE__);

// 定义是否开启异常栈跟踪
// #define ENABLE_CPPTRACE_EXCEPTION_STACKTRACE

#ifndef ENABLE_CPPTRACE_EXCEPTION_STACKTRACE
#define CPP_UTILS_TRY try
#define CPP_UTILS_CATCH catch
#else
#include <cpptrace/from_current.hpp>
#define CPP_UTILS_TRY CPPTRACE_TRY
#define CPP_UTILS_CATCH CPPTRACE_CATCH
#endif
