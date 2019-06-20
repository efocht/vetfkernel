#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "kernel.h"
#include "types.h"
#include "log.h"

#include <omp.h>

REGISTER_KERNEL("ArgMax", "op_ArgMax");
REGISTER_KERNEL("ArgMin", "op_ArgMin");

#define CHECK_ARG_LEN(l0, l1) \
  if ((l0) != (l1)) { \
      fprintf(stderr, "%s: illegal argument length: %ld expected but %ld\n", (l1), (l0)); \
      return 1; \
  }

extern "C" {
  int op_ArgMax(const void* arg, size_t len);
  int op_ArgMin(const void* arg, size_t len);
}


#define VE_ARGOP_MAX_HANDLE_DIM 4

//
// ArgMax
//

namespace {

template <typename T, typename Index>
int argmax_10(const T* in, Index* out, const int64_t* dim_size)
{
  const int64_t d0 = dim_size[0] ;

  Index idx = 0 ;
  for(int64_t i=0; i<d0; i++) {
    if( in[idx] < in[i] ) idx = i ;
  }
  out[0] = idx ;
  return 0 ;
}


template <typename T, typename Index>
int argmax_20(const T* in, Index* out, const int64_t* dim_size) 
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;

#pragma novector
  for(int64_t j=0; j<d1; j++) {
    Index idx = 0 ;
#pragma vector
    for(int64_t i=0; i<d0; i++) {
      if( in[idx*d1+j] < in[i*d1+j] ) idx = i ;
    }
    out[j] = idx ;
  }

  return 0 ;
}

template <typename T, typename Index>
int argmax_21(const T* in, Index* out, const int64_t* dim_size) 
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;

  for(int64_t i=0; i<d0; i++) {
    Index idx = 0 ;
    for(int64_t j=0; j<d1; j++) {
      if( in[idx] < in[j] ) idx = j ;
    }
    out[i] = idx ;
    in+=d1 ;
  }

  return 0 ;
}

template <typename T, typename Index>
int argmax_30(const T* in, Index* out, const int64_t* dim_size) 
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;
  const int64_t d2 = dim_size[2] ;

  for(int64_t j=0; j<d1; j++) {
    for(int64_t k=0; k<d2; k++) {
      Index idx = 0 ;
#pragma vector
      for(int64_t i=0; i<d0; i++) {
        if( in[idx*d1*d2+j*d2+k] < in[i*d1*d2+j*d2+k] ) idx = i ;
      }
      out[j*d2+k] = idx ;
    }
  }

  return 0 ;
}

template <typename T, typename Index>
int argmax_31(const T* in, Index* out, const int64_t* dim_size) 
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;
  const int64_t d2 = dim_size[2] ;

  for(int64_t i=0; i<d0; i++) {
    for(int64_t k=0; k<d2; k++) {
      Index idx = 0 ;
#pragma vector
      for(int64_t j=0; j<d1; j++) {
        if( in[i*d1*d2+idx*d2+k] < in[i*d1*d2+j*d2+k] ) idx = j ;
      }
      out[i*d2+k] = idx ;
    }
  }
  return 0 ;
}

template <typename T, typename Index>
int argmax_32(const T* in, Index* out, const int64_t* dim_size)
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;
  const int64_t d2 = dim_size[2] ;

  for(int64_t i=0; i<d0; i++) {
    for(int64_t j=0; j<d1; j++) {
      Index idx = 0 ;
#pragma vector
      for(int64_t k=0; k<d2; k++) {
        if( in[idx] < in[k] ) idx = k ;
      }
      *out = idx ;
      in+=d2 ;
      out++ ;
    }
  }
  return 0 ;
}

#if 0
template <typename T, typename Index>
int argmax_43(const T* in, Index* out, const int64_t* dim_size)
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;
  const int64_t d2 = dim_size[2] ;
  const int64_t d3 = dim_size[3] ;
  int64_t oi, oo, oo0, oo1;

  for(int64_t i0=0; i0<d0; i0++) {
    oo0 = i0*d1*d2 ;
    for(int64_t i1=0; i1<d1; i1++) {
      oo1 = oo0 + i1*d2 ;
      for(int64_t i2=0; i2<d2; i2++) {
        oo = oo1 + i2 ;
        oi = oo*d3 ;
        Index idx = 0 ;
#pragma vector
        for(int64_t i3=0; i3<d3; i3++) {
          if( in[oi+idx] < in[oi+i3] ) idx = i3 ;
        }
        out[oo] = idx ;
      }
    }
  }
  return 0 ;
}
#else
#define MAXSTRIP 10240
#define MIN(a,b) ((a) < (b) ? (a) : (b))

template <typename T, typename Index>
int argmax_43(const T* in, Index* out, const int64_t* dim_size)
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;
  const int64_t d2 = dim_size[2] ;
  const int64_t d3 = dim_size[3] ;
  int64_t osize = d0*d1*d2 ;
  T omax[MAXSTRIP];

  LOG(2) << "argmax_43" << " begin";
  for (int64_t i=0; i<osize; i+=MAXSTRIP) {
    for (int64_t j=0; j<MIN(MAXSTRIP,osize-i); j++) {
      omax[j] = in[(i+j)*d3] ;
      out[i+j] = 0;
    }
    for(int64_t i3=1; i3<d3; i3++) {
#pragma _NEC ivdep
#pragma _NEC packed_vector
      for (int64_t j=0; j<MIN(MAXSTRIP,osize-i); j++) {
        if (omax[j] < in[(i+j)*d3+i3]) {
          out[i+j] = i3 ;
          omax[j] = in[(i+j)*d3+i3] ;
        }
      }
    }
  }
  LOG(2) << "argmax_43" << " end";
  return 0 ;
}
#undef MAXSTRIP
#undef MIN
#endif


template <typename T, typename Index>
int argmax(uint64_t in_ptr, uint64_t out_ptr,
           int64_t axis, int64_t input_dims, int64_t* dim_size)
{
  const T* in = reinterpret_cast<const T*>(in_ptr);
  Index* out = reinterpret_cast<Index*>(out_ptr);

  int ret = 1 ;

  if( input_dims == 1 ) {
    if( axis == 0 ) {
      ret = argmax_10<T, Index>(in, out, dim_size) ;
    }
  }
  else if( input_dims == 2 ) {
    if( axis == 0 ) {
      ret = argmax_20<T, Index>(in, out, dim_size) ;
    }
    else if( axis == 1 ) {
      ret = argmax_21<T, Index>(in, out, dim_size) ;
    }
  }
  else if( input_dims == 3 ) {
    if( axis == 0 ) {
      ret = argmax_30<T, Index>(in, out, dim_size) ;
    }
    else if( axis == 1 ) {
      ret = argmax_31<T, Index>(in, out, dim_size) ;
    }
    else if( axis == 2 ) {
      ret = argmax_32<T, Index>(in, out, dim_size) ;
    }
  }
  else if( input_dims == 4 ) {
    if( axis == 3 ) {
      ret = argmax_43<T, Index>(in, out, dim_size) ;
    }
    else
      ret = 1 ;
  }
  return ret ;
}
}

int op_ArgMax(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__ << " begin";

  struct Args {
    int dtype, idxtype ;
    int64_t axis ;
    uint64_t in_ptr, out_ptr ;
    int64_t input_dims ;
    int64_t dim_size[VE_ARGOP_MAX_HANDLE_DIM] ;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  int ret = 1;

  if (p->dtype == DT_FLOAT) {
    if ( p->idxtype == DT_INT32 ) {
      ret = argmax<float, int32_t> (p->in_ptr, p->out_ptr,
                                    p->axis, p->input_dims, 
                                    (int64_t*) p->dim_size) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = argmax<float, int64_t> (p->in_ptr, p->out_ptr,
                                    p->axis, p->input_dims, 
                                    (int64_t*) p->dim_size) ;
    }
  }
  else if (p->dtype == DT_DOUBLE) {
    if ( p->idxtype == DT_INT32 ) {
      ret = argmax<double, int32_t>(p->in_ptr, p->out_ptr,
                                    p->axis, p->input_dims, 
                                    (int64_t*) p->dim_size) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = argmax<double, int64_t>(p->in_ptr, p->out_ptr,
                                    p->axis, p->input_dims, 
                                    (int64_t*) p->dim_size) ;
    }
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

//
// ArgMin
//

namespace {

template <typename T, typename Index>
int argmin_10(const T* in, Index* out, const int64_t* dim_size)
{
  const int64_t d0 = dim_size[0] ;

  Index idx = 0 ;
  for(int64_t i=0; i<d0; i++) {
    if( in[idx] > in[i] ) idx = i ;
  }
  out[0] = idx ;
  return 0 ;
}


template <typename T, typename Index>
int argmin_20(const T* in, Index* out, const int64_t* dim_size)
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;

#pragma novector
  for(int64_t j=0; j<d1; j++) {
    Index idx = 0 ;
#pragma vector
    for(int64_t i=0; i<d0; i++) {
      if( in[idx*d1+j] > in[i*d1+j] ) idx = i ;
    }
    out[j] = idx ;
  }

  return 0 ;
}

template <typename T, typename Index>
int argmin_21(const T* in, Index* out, const int64_t* dim_size)
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;

  for(int64_t i=0; i<d0; i++) {
    Index idx = 0 ;
    for(int64_t j=0; j<d1; j++) {
      if( in[idx] > in[j] ) idx = j ;
    }
    out[i] = idx ;
    in+=d1 ;
  }

  return 0 ;
}

template <typename T, typename Index>
int argmin_30(const T* in, Index* out, const int64_t* dim_size)
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;
  const int64_t d2 = dim_size[2] ;

  for(int64_t j=0; j<d1; j++) {
    for(int64_t k=0; k<d2; k++) {
      Index idx = 0 ;
#pragma vector
      for(int64_t i=0; i<d0; i++) {
        if( in[idx*d1*d2+j*d2+k] > in[i*d1*d2+j*d2+k] ) idx = i ;
      }
      out[j*d2+k] = idx ;
    }
  }

  return 0 ;
}

template <typename T, typename Index>
int argmin_31(const T* in, Index* out, const int64_t* dim_size)
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;
  const int64_t d2 = dim_size[2] ;

  for(int64_t i=0; i<d0; i++) {
    for(int64_t k=0; k<d2; k++) {
      Index idx = 0 ;
#pragma vector
      for(int64_t j=0; j<d1; j++) {
        if( in[i*d1*d2+idx*d2+k] > in[i*d1*d2+j*d2+k] ) idx = j ;
      }
      out[i*d2+k] = idx ;
    }
  }
  return 0 ;
}

template <typename T, typename Index>
int argmin_32(const T* in, Index* out, const int64_t* dim_size) 
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;
  const int64_t d2 = dim_size[2] ;

  for(int64_t i=0; i<d0; i++) {
    for(int64_t j=0; j<d1; j++) {
      Index idx = 0 ;
#pragma vector
      for(int64_t k=0; k<d2; k++) {
        if( in[idx] > in[k] ) idx = k ;
      }
      *out = idx ;
      in+=d2 ;
      out++ ;
    }
  }
  return 0 ;
}

#if 0
template <typename T, typename Index>
int argmin_43(const T* in, Index* out, const int64_t* dim_size)
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;
  const int64_t d2 = dim_size[2] ;
  const int64_t d3 = dim_size[3] ;
  int64_t oi, oo, oo0, oo1;

  for(int64_t i0=0; i0<d0; i0++) {
    oo0 = i0*d1*d2 ;
    for(int64_t i1=0; i1<d1; i1++) {
      oo1 = oo0 + i1*d2 ;
      for(int64_t i2=0; i2<d2; i2++) {
        oo = oo1 + i2 ;
        oi = oo * d3 ;
        Index idx = 0 ;
#pragma vector
        for(int64_t i3=0; i3<d3; i3++) {
          if( in[oi+idx] > in[oi+i3] ) idx = i3 ;
        }
        out[oo] = idx ;
      }
    }
  }
  return 0 ;
}
#else
#define MAXSTRIP 10240
#define MIN(a,b) ((a) < (b) ? (a) : (b))

template <typename T, typename Index>
int argmin_43(const T* in, Index* out, const int64_t* dim_size)
{
  const int64_t d0 = dim_size[0] ;
  const int64_t d1 = dim_size[1] ;
  const int64_t d2 = dim_size[2] ;
  const int64_t d3 = dim_size[3] ;
  int64_t osize = d0*d1*d2 ;
  T omax[MAXSTRIP];

  for (int64_t i=0; i<osize; i+=MAXSTRIP) {
    for (int64_t j=0; j<MIN(MAXSTRIP,osize-i); j++) {
      omax[j] = in[(i+j)*d3] ;
      out[i+j] = 0;
    }
    for(int64_t i3=1; i3<d3; i3++) {
#pragma _NEC ivdep
#pragma _NEC packed_vector
      for (int64_t j=0; j<MIN(MAXSTRIP,osize-i); j++) {
        if (omax[j] > in[(i+j)*d3+i3]) {
          out[i+j] = i3 ;
          omax[j] = in[(i+j)*d3+i3] ;
        }
      }
    }
  }
  return 0 ;
}
#undef MAXSTRIP
#endif

template <typename T, typename Index>
int argmin(uint64_t in_ptr, uint64_t out_ptr,
           int64_t axis, int64_t input_dims, int64_t* dim_size)
{
  const T* in = reinterpret_cast<const T*>(in_ptr);
  Index* out = reinterpret_cast<Index*>(out_ptr);

  int ret = 1 ;

  if( input_dims == 1 ) {
    if( axis == 0 ) {
      ret = argmin_10<T, Index>(in, out, dim_size) ;
    }
  }
  else if( input_dims == 2 ) {
    if( axis == 0 ) {
      ret = argmin_20<T, Index>(in, out, dim_size) ;
    }
    else if( axis == 1 ) {
      ret = argmin_21<T, Index>(in, out, dim_size) ;
    }
  }
  else if( input_dims == 3 ) {
    if( axis == 0 ) {
      ret = argmin_30<T, Index>(in, out, dim_size) ;
    }
    else if( axis == 1 ) {
      ret = argmin_31<T, Index>(in, out, dim_size) ;
    }
    else if( axis == 2 ) {
      ret = argmin_32<T, Index>(in, out, dim_size) ;
    }
  }
  else if( input_dims == 4 ) {
    if( axis == 3 ) {
      ret = argmin_43<T, Index>(in, out, dim_size) ;
    }
    else
      ret = 1 ;
  }
  return ret ;
}
}

int op_ArgMin(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__ << " begin";

  struct Args {
    int dtype, idxtype ;
    int64_t axis ;
    uint64_t in_ptr, out_ptr ;
    int64_t input_dims ;
    int64_t dim_size[VE_ARGOP_MAX_HANDLE_DIM] ;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  int ret = 1;

  if (p->dtype == DT_FLOAT) {
    if ( p->idxtype == DT_INT32 ) {
      ret = argmin<float, int32_t> (p->in_ptr, p->out_ptr,
                                    p->axis, p->input_dims, 
                                    (int64_t*) p->dim_size) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = argmin<float, int64_t> (p->in_ptr, p->out_ptr,
                                    p->axis, p->input_dims, 
                                    (int64_t*) p->dim_size) ;
    }
  }
  else if (p->dtype == DT_DOUBLE) {
    if ( p->idxtype == DT_INT32 ) {
      ret = argmin<double, int32_t>(p->in_ptr, p->out_ptr,
                                    p->axis, p->input_dims, 
                                    (int64_t*) p->dim_size) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = argmin<double, int64_t>(p->in_ptr, p->out_ptr,
                                    p->axis, p->input_dims, 
                                    (int64_t*) p->dim_size) ;
    }
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}


