#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <stdlib.h>
#include <float.h>

#include <vednn.h>

#include "kernel.h"

REGISTER_KERNEL("MaxPoolingBackprop", "maxpooling_backprop");

extern "C" {
    int maxpooling_backprop(const void* arg, size_t len);
}

struct TensorParam {
    int w,h,c,n ;
} ;

struct PoolingGradParam {
    uint64_t in;
    uint64_t out;
    uint64_t out_bp;
    uint64_t in_bp;

    TensorParam in_param;
    TensorParam out_param;
    TensorParam out_bp_param;
    TensorParam in_bp_param;

    int row_window;
    int col_window;
    int row_stride;
    int col_stride;
    int row_padding;
    int col_padding;

    int data_format;
    int data_type;
};

int maxpooling_backprop(const void* arg, size_t len)
{

#ifdef _DEBUG
    fprintf(stderr, "[start]maxpooling_backprop\n");
#endif
    assert(len == sizeof(PoolingGradParam));
    const PoolingGradParam& p = *(PoolingGradParam*)arg;

#ifdef _DEBUG
    fprintf(stderr, "maxpooling_backprop: data_format=%d data_type=%d\n", p.data_format, p.data_type);
    // assert(p.data_type   == 1 ) ; // float
    // assert(p.data_format == 1 ) ; // NHWC

    fprintf(stderr, "maxpooling_backprop: input   (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.in_param.n, p.in_param.c, p.in_param.h, p.in_param.w ) ;
    fprintf(stderr, "maxpooling_backprop: outnput (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.out_param.n, p.out_param.c, p.out_param.h, p.out_param.w ) ;
    fprintf(stderr, "maxpooling_backprop: out_bp  (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.out_bp_param.n, p.out_bp_param.c, p.out_bp_param.h, p.out_bp_param.w ) ;
    fprintf(stderr, "maxpooling_backprop: in_bp   (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.in_bp_param.n, p.in_bp_param.c, p.in_bp_param.h, p.in_bp_param.w ) ;

    fprintf(stderr, "maxpooling_backprop: window=%dx%d stride=%dx%d\n",
            p.col_window,  p.row_window,
            p.col_stride,  p.row_stride);
#endif
    
    {
#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))
      float *pIn      = (float*)p.in ;
      float *pOut     = (float*)p.out ;
      float *pGradOut = (float*)p.out_bp ;
      float *pGradIn  = (float*)p.in_bp ;

      // [todo] too slow
      for(int64_t n=0; n<p.out_param.n; n++) {
        for(int64_t c=0; c<p.out_param.c; c++) {
          for(int64_t h=0; h<p.out_param.h; h++) {
            for(int64_t w=0; w<p.out_param.w; w++) {
              const int64_t out_idx = NCHW_IDX(n,c,h,w,p.out_param.c,p.out_param.h,p.out_param.w) ;
              const float   out_val = pOut[out_idx] ;
              int found = 0;
              for(int64_t ph=0; ph<p.col_window; ph++) {
                for(int64_t pw=0; pw<p.row_window; pw++) {
                  const int64_t in_idx = NCHW_IDX(n,c,h*p.col_stride+ph,w*p.row_stride+pw,p.in_param.c,p.in_param.h,p.in_param.w) ;
                  if( !found && (out_val == pIn[in_idx])) {
                    pGradIn[in_idx] = pGradOut[out_idx] ;
                    found = 1 ;
                  }
                  else {
                    pGradIn[in_idx] = 0.f ;
                  }
                }
              }
            }
          }
        }
      } 
    }     

#ifdef _DEBUG
    fprintf(stderr, "[end]maxpooling_backprop\n");
#endif
    return 0;
}
