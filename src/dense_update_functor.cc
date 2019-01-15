#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "kernel.h"
#include "types.h"
#include "log.h"

#include <omp.h>

REGISTER_KERNEL("DenseUpdateAssign", "op_DenseUpdateAssign");

#define CHECK_ARG_LEN(l0, l1) \
  if ((l0) != (l1)) { \
      fprintf(stderr, "%s: illegal argument length: %ld expected but %ld\n", (l1), (l0)); \
      return 1; \
  }

extern "C" {
  int op_DenseUpdateAssign(const void* arg, size_t len);
}


//
// DenseUpdateAssign
//

namespace {

template <typename T>
int dense_update_assign(int64_t num_elements,
                    uint64_t dst_ptr, uint64_t src_ptr )
{
  T* dst = reinterpret_cast<T*>(dst_ptr);
  const T* src = reinterpret_cast<const T*>(src_ptr);

#pragma omp parallel for
  for(int64_t i=0; i<num_elements; i++) {
    dst[i] = src[i] ;
  }
  return 0 ;
}
}

int op_DenseUpdateAssign(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__ << " begin";

  struct Args {
    int dtype;
    int64_t num_elements ;
    uint64_t dst_ptr, src_ptr ;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  int ret = 1;

  if (p->dtype == DT_FLOAT) {
    ret = dense_update_assign<float> (p->num_elements,
                                      p->dst_ptr, p->src_ptr) ;
  }
  else if (p->dtype == DT_DOUBLE) {
    ret = dense_update_assign<double>(p->num_elements,
                                      p->dst_ptr, p->src_ptr) ;
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

