#include "inference/utils/backtrace.h"

#include <cpptrace/cpptrace.hpp>

namespace cpputils {

std::string StacktraceToString() { return cpptrace::stacktrace().to_string(); }

} // namespace cpputils
