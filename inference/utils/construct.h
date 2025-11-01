#pragma once

#define NON_COPY_CONSTRUCT(x)                                                  \
  x(const x &) = delete;                                                       \
  x &operator=(const x &) = delete

#define NON_MOVE_CONSTRUCT(x)                                                  \
  x(const x &&) = delete;                                                      \
  x &operator=(x &&) = delete

/*
https://gcc.gnu.org/onlinedocs/gcc-4.7.0/gcc/Function-Attributes.html

constructor
destructor
constructor (priority)
destructor (priority)
The constructor attribute causes the function to be called automatically before
execution enters main (). Similarly, the destructor attribute causes the
function to be called automatically after main () has completed or exit () has
been called. Functions with these attributes are useful for initializing data
that will be used implicitly during the execution of the program. You may
provide an optional integer priority to control the order in which constructor
and destructor functions are run.

A constructor with a smaller priority number runs before a constructor with a
larger priority number; the opposite relationship holds for destructors.

So, if you have a constructor that allocates a
resource and a destructor that deallocates the same resource, both functions
typically have the same priority. The priorities for constructor and destructor
functions are the same as those specified for namespace-scope C++ objects (see
C++ Attributes).

These attributes are not currently implemented for Objective-C.

examples:

__attribute__((constructor)) 与 __attribute__((destructor)) 是 GCC
中用来修饰函数的，constructor 可以使被修饰的函数在 main()
执行前被调用，destructor 可以使被修饰的函数在 main() 执行结束或 exit()
调用结束后被执行。

__attribute__((constructor)) void constructor_func() {
    // ...
}

__attribute__((destructor)) void destructor_func() {
    // ...

原文链接：https://blog.csdn.net/stone8761/article/details/122498016

*/

#define BEFORE_MAIN __attribute__((constructor))
#define BEFORE_MAIN_PRIO(x) __attribute__((constructor(x)))

#define AFTER_MAIN __attribute__((destructor))
#define AFTER_MAIN_PRIO(x) __attribute__((destructor(x)))
