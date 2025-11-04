#include "inference/utils/backtrace.h"

#include <cpptrace/cpptrace.hpp>
#include <cpptrace/from_current.hpp>

namespace cpputils {

void StacktracePrint() { cpptrace::generate_trace().print(); }

std::string StacktraceToString() {
  return cpptrace::generate_trace().to_string();
}

void ExceptionStacktracePrint() { cpptrace::from_current_exception().print(); }

std::string ExceptionStacktraceToString() {
  return cpptrace::from_current_exception().to_string();
}

} // namespace cpputils
