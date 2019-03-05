#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>

#include <sched.h>
#include <omp.h>

#include "asl.h"
#include "kernel.h"
#include "log.h"

#define MAX_KERNEL 1024

#define USE_DMA
#ifdef USE_DMA
extern "C" {
#include <vhshm.h>
#include <vedma.h>
}
#endif

class InitVETFKernel
{
public :
  InitVETFKernel() {
    setaffinity() ;
    ASL::initialize() ;
  }

  ~InitVETFKernel() {
    ASL::finalize() ;
  }

private :
  void setaffinity() {
#pragma omp parallel
    {
      int threadid = omp_get_thread_num() ;
      cpu_set_t mask ;
      CPU_ZERO(&mask) ;
      CPU_SET(threadid, &mask) ;
      sched_setaffinity(0, sizeof(mask), &mask ) ;
    }
  }
} _InitVETFKernel ; 

extern "C" {
    int get_num_kernels();
    uint64_t get_kernel_table_addr();

    uint64_t vetfkl_entry(const void* arg, size_t len);
    uint64_t vetfkl_entry_prof(const void* argIn, size_t lenIn, void* argOut, size_t lenOut);
    int vetfkl_get_timestamp(void* arg, size_t len);

#ifdef USE_DMA
    int vetfkl_init_dma(void* arg, size_t len);
    int vetfkl_write_mem(void* arg, size_t len);
#endif

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
  LOG(1) << __FUNCTION__ << ":"
    << " kernel_index=" << kernel_index
    << " kernel_name=" << name
    << " func=" << func;
    Kernel& k = table_[kernel_index++];

    strcpy(k.name, name);
    strcpy(k.func, func);
}

uint64_t vetfkl_entry(const void* arg, size_t len)
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
    if (ret != 0) {
      uint64_t retval = ((uint64_t)i) << 32 | ret;
      return retval;
    }
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

#ifdef USE_DMA
void* vemva_ = NULL;
void* vehva_vh_;
uint64_t vehva_ve_;

int init_dma(int32_t shmid, size_t size)
{
    LOG(1) << "init_dma: shmid=" << shmid << " size=" << size;

    ve_dma_init();

    size_t align = 64 * 1024 * 1024;
    if (posix_memalign(&vemva_, align, size) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        return 1;
    }
    LOG(2) << "init_dma: vemva=" << vemva_;
    vehva_ve_ = ve_register_mem_to_dmaatb(vemva_, size);
    LOG(2) << "init_dma: vehva_ve_=" << (void*)vehva_ve_;
    if (vehva_ve_ == (uint64_t)-1) {
        fprintf(stderr, "ve_register_mem_to_dmaatb failed\n");
        return 1;
    }

    void* tmp = vh_shmat(shmid, NULL, 0, &vehva_vh_);
    if (tmp == (void*)-1) {
        perror("vh_shmget");
        return 1;
    }

    // TODO: vh_shmdt

    return 0;
}

int vetfkl_init_dma(void* arg, size_t len)
{
    struct tmp {
        int32_t shmid;
        uint64_t size;
    }* p = reinterpret_cast<tmp*>(arg);

    return init_dma(p->shmid, p->size);
}

int vetfkl_write_mem(void* arg, size_t len)
{
    struct tmp {
        uint64_t size;
        uint64_t vevma;
    }* p = reinterpret_cast<tmp*>(arg);

    // TODO: 
    // 1. DMA to pre-allocated buffer, then memcpy to destination address.
    // 2. register destination address, and DMA, and unregister. 
    //    (Address have to be 64b aligned. Difficult condition)
    // Currently 1 is used. which is faster?

#if 0
    fprintf(stderr, "%16llu %s: size=%lu vevma=%x (vehva_ve_=%p)\n",
            __veperf_get_stm(), __FUNCTION__, p->size, p->vevma, vehva_ve_);
#endif

#if 1 // method 1
    int ret = ve_dma_post_wait(vehva_ve_, (uint64_t)vehva_vh_, p->size);
#if 0
    fprintf(stderr, "%16llu %s: size=%lu ret=%d\n", __veperf_get_stm(), __FUNCTION__, p->size, ret);
#endif

    if (ret != 0)
        return 1;

#if 0
    fprintf(stderr, "%16llu %s: call memcpy\n", __veperf_get_stm(), __FUNCTION__);
#endif

    memcpy(reinterpret_cast<void*>(p->vevma), (void const*)vemva_, p->size);

#else // method 2
    uint64_t vehva_ve = ve_register_mem_to_dmaatb((void*)p->vevma, p->size);
    if (vehva_ve == (uint64_t)-1)
        return 1;

    int ret = ve_dma_post_wait(vehva_ve, (uint64_t)vehva_vh_, p->size);
#if 1
    fprintf(stderr, "%16llu %s: size=%lu ret=%d\n", __veperf_get_stm(), __FUNCTION__, p->size, ret);
#endif

    ve_unregister_mem_from_dmaatb(vehva_ve);
#endif

#if 0
    fprintf(stderr, "%16llu %s: done\n", __veperf_get_stm(), __FUNCTION__);
#endif
    return 0;
}
#endif

uint64_t vetfkl_entry_prof(const void* argIn, size_t lenIn, 
                           void* argOut, size_t lenOut)
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
    pcyc = reinterpret_cast<uint64_t*>(reinterpret_cast<uintptr_t>(argOut));
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
    if (ret != 0) {
      uint64_t retval = ((uint64_t)i) << 32 | ret;
      return retval;
    }
  }

#if 0
  fprintf(stderr, "vetfkl_entry: end\n");
#endif

  return 0;
}

