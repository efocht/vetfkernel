#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>

#include "kernel.h"

#define MAX_KERNEL 1024

extern "C" {
    int get_num_kernels();
    uint64_t get_kernel_table_addr();

    int vetfkl_entry(const void* arg, size_t len);

    int op_Assign(const void* arg, size_t len);
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
    fprintf(stderr, "libvetfkernel::register_kernel: kernel_index=%d name=%s func=%s\n", kernel_index, name, func);
    Kernel& k = table_[kernel_index++];

    strcpy(k.name, name);
    strcpy(k.func, func);
}

int vetfkl_entry(const void* arg, size_t len)
{
#if 0
  fprintf(stderr, "vetfkl_entry: len=%lu\n", len);
#endif

  //const void* curr = arg;

  uint64_t end = reinterpret_cast<uintptr_t>(arg) + len;
  uintptr_t curr = reinterpret_cast<uintptr_t>(arg);

  int32_t num_kernels = *reinterpret_cast<int32_t*>(curr);
  curr += sizeof(int32_t);

#if 0
  fprintf(stderr, "%s: num_kernels=%d\n", __FUNCTION__, num_kernels);
#endif

  typedef int (*func_t)(const void* arg, size_t len);

  for (int i = 0; i < num_kernels; ++i) {
    uint64_t sym = *reinterpret_cast<uint64_t*>(curr);
    curr += sizeof(uint64_t);
    func_t func = reinterpret_cast<func_t>(sym);
    size_t len0 = *reinterpret_cast<size_t*>(curr);
    curr += sizeof(size_t);
    const void* arg0 = reinterpret_cast<const void*>(curr);
    curr += len0;

#if 0
    fprintf(stderr, "vetfkl_entry: i=%d/%d func=%p args0=%p len=%lu\n",
            i, num_kernels, func, arg0, len0);
#endif
    int ret = func(arg0, len0);
#if 0
    fprintf(stderr, "vetfkl_entry: ret=%d\n", ret);
#endif
  }

#if 0
  fprintf(stderr, "vetfkl_entry: end\n");
#endif
}

