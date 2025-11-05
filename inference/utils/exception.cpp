#include "inference/utils/exception.h"
#include "inference/utils/log.h"

#include <fmt/format.h>

#ifdef ENABLE_CPPTRACE_EXCEPTION_STACKTRACE
#include "inference/utils/backtrace.h"
#include <cpptrace/cpptrace.hpp>
#endif

namespace cpputils {

RuntimeException::RuntimeException(const std::string &msg, const char *file,
                                   const char *func, uint32_t line) {
  what_ = "\n----------[RuntimeException]------------\n";
  what_ += "############################################\n";
  what_ += "tips: " + msg + "\n";
  what_ += fmt::format("pos: {}:{}, func: {}\n", file, line, func);
#ifdef ENABLE_CPPTRACE_EXCEPTION_STACKTRACE
  what_ += cpputils::StacktraceToString();
  what_ += "\n";
#endif
  what_ += "############################################\n\n";
}

void PrettyPrintException(const std::exception &e) {
#ifdef ENABLE_CPPTRACE_EXCEPTION_STACKTRACE
  if (typeid(e) == typeid(RuntimeException)) {
    LOG_CRITICAL("exception info:\n{}", e.what());
  } else {
    LOG_CRITICAL("exception info:\n{}\n{}", cpputils::StacktraceToString(),
                 e.what());
  }
#else
  LOG_ERROR("{}", e.what());
#endif
}

}; // namespace cpputils
