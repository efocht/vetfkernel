#ifndef KERNEL_H
#define KERNEL_H

extern void register_kernel(char const* name, char const* func);

#define REGISTER_KERNEL(name, func) \
    class __register_##__COUNTER__ { \
        public: \
            __register_##__COUNTER__() { \
                register_kernel(name, func); \
            } \
    } __register_##__COUNTER__; \

#endif
