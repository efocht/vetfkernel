#ifndef KERNEL_H
#define KERNEL_H

extern void register_kernel(char const* name, char const* func);

class Register {
  public:
    typedef void (*Func)(void);
    Register(Func func) {
      func();
    }
};

#define MAKE_NAME(a, b) a##b

#define REGISTER_KERNEL_HELPER(ctr, name, func) \
  static Register MAKE_NAME(__register, ctr)([]() { \
    register_kernel(name, func); \
  });

#define REGISTER_KERNEL(name, func) \
  REGISTER_KERNEL_HELPER(__COUNTER__, name, func)

#endif
