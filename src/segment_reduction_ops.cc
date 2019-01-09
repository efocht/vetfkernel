#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "kernel.h"
#include "types.h"
#include "log.h"

#include <omp.h>

REGISTER_KERNEL("UnsortedSegmentSum", "op_UnsortedSegmentSum");

#define CHECK_ARG_LEN(l0, l1) \
  if ((l0) != (l1)) { \
      fprintf(stderr, "%s: illegal argument length: %ld expected but %ld\n", (l1), (l0)); \
      return 1; \
  }

extern "C" {
  int op_UnsortedSegmentSum(const void* arg, size_t len);
}


//
// UnsortedSegmentSum
//

namespace {

template <typename T, typename Index>
int unsorted_segment_sum(int64_t num_idx, int64_t num_segments, int64_t segment_size,
                         uint64_t src_ptr, uint64_t idx_ptr, uint64_t dst_ptr,
                         T initial_value ) 
{
  const T* src = reinterpret_cast<const T*>(src_ptr);
  const Index* idx = reinterpret_cast<const Index*>(idx_ptr);
  T* dst = reinterpret_cast<T*>(dst_ptr);

  for(int64_t i=0; i < num_segments * segment_size; i++) {
    dst[i] = initial_value ;
  }

  for(int64_t i=0; i<num_idx; i++) {
    const int64_t k = idx[i] ;
    for(int64_t j=0; j<segment_size; j++) {
      dst[k*segment_size+j] += src[i*segment_size+j] ;
    }
  }

  return 0 ;
}
}

int op_UnsortedSegmentSum(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__ << " begin";

  struct Args {
    int dtype, idxtype;
    int64_t num_idx, num_segments, segment_size ;
    union { float f32; double f64; } initial_value ;
    uint64_t src_ptr, idx_ptr, dst_ptr ;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  int ret = 1;

  if (p->dtype == DT_FLOAT) {
    if ( p->idxtype == DT_INT32 ) {
      ret = unsorted_segment_sum<float, int32_t> (p->num_idx, p->num_segments, p->segment_size,
                                                  p->src_ptr, p->idx_ptr, p->dst_ptr,
                                                  p->initial_value.f32 ) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = unsorted_segment_sum<float, int64_t> (p->num_idx, p->num_segments, p->segment_size,
                                                  p->src_ptr, p->idx_ptr, p->dst_ptr,
                                                  p->initial_value.f32 ) ;
    }
  }
  else if (p->dtype == DT_DOUBLE) {
    if ( p->idxtype == DT_INT32 ) {
      ret = unsorted_segment_sum<double, int32_t> (p->num_idx, p->num_segments, p->segment_size,
                                                   p->src_ptr, p->idx_ptr, p->dst_ptr,
                                                   p->initial_value.f64 ) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = unsorted_segment_sum<double, int64_t> (p->num_idx, p->num_segments, p->segment_size,
                                                   p->src_ptr, p->idx_ptr, p->dst_ptr,
                                                   p->initial_value.f64 ) ;
    }
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

