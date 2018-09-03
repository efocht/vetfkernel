#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>

#include "kernel.h"

#define MAX_KERNEL 1024

extern "C" {
    int get_num_kernels();
    uint64_t get_kernel_table_addr();
}

struct Kernel
{
    char name[256];
    char func[256];
} table_[MAX_KERNEL];

static int kernel_index = 0;

int get_num_kernels()
{
    return kernel_index;
}

uint64_t get_kernel_table_addr()
{
    return (uint64_t)table_;
}

void register_kernel(char const* name, char const* func)
{
    // FIXME: check MAX_KERNEL
    fprintf(stderr, "libvetfkernel::register_kernel: kernel_index=%d name=%s func=%p\n", kernel_index, name, func);
    Kernel& k = table_[kernel_index++];

    strcpy(k.name, name);
    strcpy(k.func, func);
}
