set(CMAKE_C_COMPILER gcc)
add_library(fakegettid SHARED fake_syscall_gettid.c)
