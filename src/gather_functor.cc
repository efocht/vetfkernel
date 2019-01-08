#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "kernel.h"
#include "types.h"
#include "log.h"

#include <omp.h>

REGISTER_KERNEL("Gather", "op_Gather");

#define CHECK_ARG_LEN(l0, l1) \
  if ((l0) != (l1)) { \
      fprintf(stderr, "%s: illegal argument length: %ld expected but %ld\n", (l1), (l0)); \
      return 1; \
  }

extern "C" {
  int op_Gather(const void* arg, size_t len);
}


//
// Gather
//

namespace {

template <typename T, typename Index>
int gather(int64_t outer_size, int64_t inner_size, int64_t nindex,
           uint64_t src_ptr, uint64_t idx_ptr, uint64_t dst_ptr) 
{
  const T* src = reinterpret_cast<const T*>(src_ptr);
  const Index* idx = reinterpret_cast<const Index*>(idx_ptr);
  T* dst = reinterpret_cast<T*>(dst_ptr);

  for(int64_t i=0; i<nindex; i++) {
    const int64_t k = idx[i] ;
    for(int64_t j=0; j<inner_size; j++) {
      dst[i*inner_size+j] = src[k*inner_size+j] ;
    }
  }

  return 0 ;
}
}

int op_Gather(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__ << " begin";

  struct Args {
    int dtype, idxtype;
    int64_t outer_size ;
    int64_t inner_size ;
    int64_t nindex ;
    uint64_t src_ptr, idx_ptr, dst_ptr ;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  int ret = 1;

  if (p->dtype == DT_FLOAT) {
    if ( p->idxtype == DT_INT32 ) {
      ret = gather<float, int32_t> (p->outer_size, p->inner_size, p->nindex,
                                    p->src_ptr, p->idx_ptr, p->dst_ptr) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = gather<float, int64_t> (p->outer_size, p->inner_size, p->nindex,
                                    p->src_ptr, p->idx_ptr, p->dst_ptr) ;
    }
  }
  else if (p->dtype == DT_DOUBLE) {
    if ( p->idxtype == DT_INT32 ) {
      ret = gather<double, int32_t> (p->outer_size, p->inner_size, p->nindex,
                                     p->src_ptr, p->idx_ptr, p->dst_ptr) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = gather<double, int64_t> (p->outer_size, p->inner_size, p->nindex,
                                     p->src_ptr, p->idx_ptr, p->dst_ptr) ;
    }
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

