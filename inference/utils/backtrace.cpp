#include "inference/utils/backtrace.h"

#include <cpptrace/cpptrace.hpp>
#include <cpptrace/from_current.hpp>
#include <fmt/format.h>

namespace cpputils {

#ifdef ENABLE_CPPTRACE_STACKTRACE

void StacktracePrint() { cpptrace::generate_trace().print(); }

std::string StacktraceToString() {
  return cpptrace::generate_trace().to_string();
}

void ExceptionStacktracePrint() { cpptrace::from_current_exception().print(); }

std::string ExceptionStacktraceToString() {
  return cpptrace::from_current_exception().to_string();
}

#else

void StacktracePrint() {}

std::string StacktraceToString() {
  return fmt::format("{}: null stacktrace", __func__);
}

void ExceptionStacktracePrint() {}

std::string ExceptionStacktraceToString() {
  return fmt::format("{}: null stacktrace", __func__);
}

#endif

} // namespace cpputils
