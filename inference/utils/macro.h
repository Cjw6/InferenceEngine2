
#ifndef _CPP_UTILS_BASE_MACRO_H_
#define _CPP_UTILS_BASE_MACRO_H_

/**
 * @brief api
 *
 */
#if defined _WIN32 || defined __CYGWIN__
#ifdef ENABLE_CPP_UTILS_BUILDING_DLL
#ifdef __GNUC__
#define CPP_UTILS_CC_API __attribute__((dllexport))
#else // __GNUC__
#define CPP_UTILS_CC_API __declspec(dllexport)
#endif // __GNUC__
#else  // CPP_UTILS_BUILDING_DLL
#ifdef __GNUC__
#define CPP_UTILS_CC_API __attribute__((dllimport))
#else
#define CPP_UTILS_CC_API __declspec(dllimport)
#endif // __GNUC__
#endif // CPP_UTILS_BUILDING_DLL
#else  // _WIN32 || __CYGWIN__
#if __GNUC__ >= 4
#define CPP_UTILS_CC_API __attribute__((visibility("default")))
#else
#define CPP_UTILS_CC_API
#endif
#endif

#ifdef __cplusplus
#define CPP_UTILS_C_API extern "C" CPP_UTILS_CC_API
#endif

/**
 * @brief program
 *
 */
#ifdef _MSC_VER
#define CPP_UTILS_PRAGMA(X) __pragma(X)
#else
#define CPP_UTILS_PRAGMA(X) _Pragma(#X)
#endif

/**
 * @brief deprecated
 *
 */
#if defined(__GNUC__) || defined(__clang__)
#define CPP_UTILS_DEPRECATED(msg) __attribute__((deprecated(msg)))
#elif defined(_MSC_VER)
#define CPP_UTILS_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define CPP_UTILS_DEPRECATED
#endif

/**
 * @brief string
 *
 */
#define CPP_UTILS_DEFAULT_STR "CPP_UTILS_default_str"
#define CPP_UTILS_TO_STR(x) #x
#define CPP_UTILS_NAMESPACE_PLUS_TO_STR(x) CPP_UTILS_namespace##x
#define CPP_UTILS_GENERATE_DEFAULT_STR()                                       \
  std::string file = __FILE__;                                                 \
  std::string function = __FUNCTION__;                                         \
  std::string line = std::to_string(__LINE__);                                 \
  return CPP_UTILS_DEFAULT_STR + "_" file + "_" + function + "_" + line;

/**
 * @brief math
 *
 */
#ifndef CPP_UTILS_UP_DIV
#define CPP_UTILS_UP_DIV(x, y)                                                 \
  ((static_cast<int>(x) + static_cast<int>(y) - (1)) / static_cast<int>(y))
#endif

#ifndef CPP_UTILS_ROUND_UP
#define CPP_UTILS_ROUND_UP(x, y)                                               \
  ((static_cast<int>(x) + static_cast<int>(y) - (1)) / static_cast<int>(y) *   \
   static_cast<int>(y))
#endif

#ifndef CPP_UTILS_ALIGN_UP4
#define CPP_UTILS_ALIGN_UP4(x) CPP_UTILS_ROUND_UP((x), 4)
#endif

#ifndef CPP_UTILS_ALIGN_UP8
#define CPP_UTILS_ALIGN_UP8(x) CPP_UTILS_ROUND_UP((x), 8)
#endif

#ifndef CPP_UTILS_ALIGN_PTR
#define CPP_UTILS_ALIGN_PTR(x, y)                                              \
  (void *)(x & ~static_cast<size_t>(CPP_UTILS_ABS(y) - 1))
#endif

#ifndef CPP_UTILS_MIN
#define CPP_UTILS_MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef CPP_UTILS_MAX
#define CPP_UTILS_MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef CPP_UTILS_ABS
#define CPP_UTILS_ABS(x) ((x) > (0) ? (x) : (-(x)))
#endif

#ifdef CPP_UTILS_XADD
// allow to use user-defined macro
#elif defined __GNUC__ || defined __clang__
#if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ &&       \
    !defined __EMSCRIPTEN__ && !defined(__CUDACC__) &&                         \
    !defined __INTEL_COMPILER
#ifdef __ATOMIC_ACQ_REL
#define CPP_UTILS_XADD(addr, delta)                                            \
  __c11_atomic_fetch_add((_Atomic(int) *)(addr), delta, __ATOMIC_ACQ_REL)
#else
#define CPP_UTILS_XADD(addr, delta)                                            \
  __atomic_fetch_add((_Atomic(int) *)(addr), delta, 4)
#endif
#else
#if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#define CPP_UTILS_XADD(addr, delta)                                            \
  (int)__atomic_fetch_add((unsigned *)(addr), (unsigned)(delta),               \
                          __ATOMIC_ACQ_REL)
#else
#define CPP_UTILS_XADD(addr, delta)                                            \
  (int)__sync_fetch_and_add((unsigned *)(addr), (unsigned)(delta))
#endif
#endif
#elif defined _MSC_VER && !defined RC_INVOKED
#include <intrin.h>
#define CPP_UTILS_XADD(addr, delta)                                            \
  (int)_InterlockedExchangeAdd((long volatile *)addr, delta)
#else
#ifdef CPP_UTILS_FORCE_UNSAFE_XADD
static inline int CPP_UTILS_XADD(int *addr, int delta) {
  int tmp = *addr;
  *addr += delta;
  return tmp;
}
#else
#error                                                                         \
    "CPP_UTILS: can't define safe CPP_UTILS_XADD macro for current platform (unsupported). Define CPP_UTILS_XADD macro through custom port header"
#endif
#endif

#define CPP_UTILS_OS_LINUX 0
#define CPP_UTILS_OS_ANDROID 0
#define CPP_UTILS_OS_DARWIN 0
#define CPP_UTILS_OS_IOS 0
#define CPP_UTILS_OS_WINDOWS 0
#define CPP_UTILS_OS_UNIX 0
#define CPP_UTILS_OS_UNKNOWN 0

#if (defined __linux)
#undef CPP_UTILS_OS_LINUX
#define CPP_UTILS_OS_LINUX 1
#elif (defined __linux__)
#undef CPP_UTILS_OS_LINUX
#define CPP_UTILS_OS_LINUX 1
#else
#endif

#if (defined __android__)
#undef CPP_UTILS_OS_ANDROID
#define CPP_UTILS_OS_ANDROID 1
#endif

#if (defined __APPLE__)
#include <TargetConditionals.h>
#if defined(TARGET_OS_MAC)
#undef CPP_UTILS_OS_DARWIN
#define CPP_UTILS_OS_DARWIN 1
#elif defined(TARGET_OS_IPHONE)
#undef CPP_UTILS_OS_IOS
#define CPP_UTILS_OS_IOS 1
#endif
#endif

#ifdef _WIN32
#undef CPP_UTILS_OS_WINDOWS
#define CPP_UTILS_OS_WINDOWS 1
#endif

#ifdef __CYGWIN__
#undef CPP_UTILS_OS_WINDOWS
#define CPP_UTILS_OS_WINDOWS 1
#endif

#if (1 != CPP_UTILS_OS_LINUX + CPP_UTILS_OS_ANDROID + CPP_UTILS_OS_DARWIN +    \
              CPP_UTILS_OS_IOS + CPP_UTILS_OS_WINDOWS)
#define CPP_UTILS_OS_UNKNOWN 1
#endif

#if CPP_UTILS_OS_LINUX || CPP_UTILS_OS_ANDROID || CPP_UTILS_OS_DARWIN ||       \
    CPP_UTILS_OS_IOS
#undef CPP_UTILS_OS_UNIX
#define CPP_UTILS_OS_UNIX 1
#endif

// ARCHITECTURE
#define CPP_UTILS_ARCHITECTURE_X86 0
#define CPP_UTILS_ARCHITECTURE_ARM 0
#define CPP_UTILS_ARCHITECTURE_CPU 1

#if (defined ENABLE_CPP_UTILS_DEVICE_X86)
#undef CPP_UTILS_ARCHITECTURE_X86
#define CPP_UTILS_ARCHITECTURE_X86 1
#endif

#if (defined ENABLE_CPP_UTILS_DEVICE_ARM)
#undef CPP_UTILS_ARCHITECTURE_ARM
#define CPP_UTILS_ARCHITECTURE_ARM 1
#endif

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define CPP_UTILS_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define CPP_UTILS_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define CPP_UTILS_LIKELY(expr) (expr)
#define CPP_UTILS_UNLIKELY(expr) (expr)
#endif

#endif // _CPP_UTILS_BASE_MACRO_H_
