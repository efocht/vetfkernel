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
    int vetfkl_entry_prof(const void* argIn, size_t lenIn, void* argOut, size_t lenOut);
    int vetfkl_get_timestamp(void* arg, size_t len);

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
  return 0;
}

static inline unsigned long int __ve_get_usrcc() {
  unsigned long int val;
  asm volatile("smir %0,%usrcc\n" : "=r"(val));
  return val;
}

static inline unsigned long long __veperf_get_stm() {
  void *vehva = (void *)0x1000;
  unsigned long long val;
  asm volatile ("lhm.l %0,0(%1)":"=r"(val):"r"(vehva));
  return val;
}

static inline unsigned long long get_timestamp() {
  return __veperf_get_stm();
}

static inline double get_resolution() {
  return 800e6;
}

int vetfkl_get_timestamp(void* arg, size_t len)
{
  struct tmp {
    uint64_t ts;
    double resolution;
  }* p = reinterpret_cast<tmp*>(arg);

  //fprintf(stderr, "vetfkl_get_timestamp: len=%lu sizeof(tmp)=%lu\n", len, sizeof(tmp));

  if (len < sizeof(tmp))
    return 1;

  p->ts = get_timestamp();
  p->resolution = get_resolution();

  //fprintf(stderr, "vetfkl_get_timestamp: resolution=%lf\n", p->resolution);

  return 0;

#if 0
#if 1
  *reinterpret_cast<uint64_t*>(arg) = get_timestamp();
  return 0;
#else
  uint64_t t = get_timestamp();
  fprintf(stderr, "vetfkl_get_timestamp: t=%lu\n", t);
  *reinterpret_cast<uint64_t*>(arg) = t;
  return 0;
#endif
#endif
}

int vetfkl_entry_prof(const void* argIn, size_t lenIn, void* argOut, size_t lenOut)
{
#if 0
  fprintf(stderr, "vetfkl_entry_prof: argIn=%p lenIn=%lu argOut=%p lenOut=%lu\n",
          argIn, lenIn, argOut, lenOut);
#endif

  uint64_t end = reinterpret_cast<uintptr_t>(argIn) + lenIn;
  uintptr_t curr = reinterpret_cast<uintptr_t>(argIn);

  int32_t num_kernels = *reinterpret_cast<int32_t*>(curr);
  curr += sizeof(int32_t);

#if 0
  fprintf(stderr, "%s: num_kernels=%d\n", __FUNCTION__, num_kernels);
#endif

  typedef int (*func_t)(const void* arg, size_t len);

  uint64_t* pcyc = nullptr;
  if (lenOut > 0) {
    *reinterpret_cast<double*>(argOut) = 1.4*1e9;
    pcyc = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(argOut) + sizeof(double));
  }

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
    uint64_t t0 = get_timestamp();
    int ret = func(arg0, len0);
    uint64_t t1 = get_timestamp();
    if (pcyc) {
      pcyc[i*2] = t0;
      pcyc[i*2+1] = t1;
    }
    //fprintf(stderr, "vetfkl_entry_prof: i=%d cyc=%llu\n", i, pcyc[i]);
#if 0
    fprintf(stderr, "vetfkl_entry: ret=%d\n", ret);
#endif
  }

#if 0
  fprintf(stderr, "vetfkl_entry: end\n");
#endif

  return 0;
}

