#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <stdlib.h>
#include <float.h>

#include <vednn.h>

#include "kernel.h"

REGISTER_KERNEL("MaxPooling", "maxpooling");

extern "C" {
    int maxpooling(const void* arg, size_t len);
}

struct TensorParam {
    int w,h,c,n ;
} ;

struct PoolingParam {
    uint64_t in;
    uint64_t out;
    TensorParam in_param;
    TensorParam out_param;

    int row_window;
    int col_window;
    int row_stride;
    int col_stride;

    int data_format;
    int data_type;
};

int maxpooling(const void* arg, size_t len)
{

#ifdef _DEBUG
    fprintf(stderr, "[start]maxpooling\n");
#endif
    assert(len == sizeof(PoolingParam));
    const PoolingParam& p = *(PoolingParam*)arg;

#ifdef _DEBUG
    fprintf(stderr, "maxpooling: data_format=%d data_type=%d\n", p.data_format, p.data_type);
    // assert(p.data_type   == 1 ) ; // float
    // assert(p.data_format == 1 ) ; // NHWC

    fprintf(stderr, "maxpooling: input   (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.in_param.n, p.in_param.c, p.in_param.h, p.in_param.w ) ;
    fprintf(stderr, "maxpooling: outnput (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.out_param.n, p.out_param.c, p.out_param.h, p.out_param.w ) ;
    
    fprintf(stderr, "maxpooling: window=%dx%d stride=%dx%d\n",
            p.col_window,  p.row_window,
            p.col_stride,  p.row_stride);
#endif
     
    {
#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))
      float *pIn  = (float*)p.in ;
      float *pOut = (float*)p.out ;
 
      // [todo] too slow
      for(int64_t n=0; n<p.out_param.n; n++) {
        for(int64_t c=0; c<p.out_param.c; c++) {
          for(int64_t h=0; h<p.out_param.h; h++) {
            for(int64_t w=0; w<p.out_param.w; w++) {
              float max_val = -FLT_MAX ;
              for(int64_t ph=0; ph<p.col_window; ph++) {
                for(int64_t pw=0; pw<p.row_window; pw++) {
                  const int64_t in_idx = NCHW_IDX(n,c,h*p.col_stride+ph,w*p.row_stride+pw,p.in_param.c,p.in_param.h,p.in_param.w) ;
                  const float   in_val = pIn[in_idx] ; 
                  if( in_val > max_val ) max_val = in_val ;
                }
              }
              const int64_t out_idx = NCHW_IDX(n,c,h,w,p.out_param.c,p.out_param.h,p.out_param.w) ;
              pOut[out_idx] = max_val ;
            }
          }
        }
      }
    }

#ifdef _DEBUG
    fprintf(stderr, "[end]maxpooling\n");
#endif
    return 0;
}
