#include "inference/utils/exception.h"
#include "inference/utils/backtrace.h"

#include <fmt/format.h>

namespace cpputils {

RuntimeException::RuntimeException(const std::string &msg, const char *file,
                                   const char *func, uint32_t line) {
  what_ = msg + "\n";
  what_ += fmt::format("pos: {}:{}, func: {}\n", file, line, func);
  what_ += "stacktrace:\n";
  what_ += cpputils::StacktraceToString();
}

}; // namespace cpputils
