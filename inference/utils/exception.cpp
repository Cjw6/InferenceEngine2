#include "inference/utils/exception.h"
#include "inference/utils/log.h"

#include <fmt/format.h>

#ifdef ENABLE_CPPTRACE_STACKTRACE
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
#ifdef ENABLE_CPPTRACE_STACKTRACE
  what_ += cpputils::StacktraceToString();
  what_ += "\n";
#endif
  what_ += "############################################\n\n";
}

void PrettyPrintException(const std::exception &e) {
  LOG_CRITICAL("exception info:\n{}", e.what());
}

}; // namespace cpputils
