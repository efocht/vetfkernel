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

    fprintf(stderr, "maxpooling: window=%dx%d stride=%dx%d, padding=%dx%d\n",
            p.col_window,   p.row_window,
            p.col_stride,   p.row_stride,
            p.col_padding,  p.row_padding );
#endif
    
    {
      void *pGradOut  = (void *) p.out_bp ;
      void *pOut      = (void *) p.out ;
      void *pIn       = (void *) p.in ;
      void *pGradIn   = (void *) p.in_bp ;
    
      vednnTensorParam_t ParamGradOut ;
      vednnTensorParam_t ParamOut ;
      vednnTensorParam_t ParamIn ;
      vednnTensorParam_t ParamGradIn ;

      vednnPoolingParam_t ParamPool ;

      ParamGradOut.dtype   = ParamOut.dtype   = DTYPE_FLOAT ;
      ParamGradOut.batch   = ParamOut.batch   = p.out_param.n ;
      ParamGradOut.channel = ParamOut.channel = p.out_param.c ;
      ParamGradOut.width   = ParamOut.width   = p.out_param.w ;
      ParamGradOut.height  = ParamOut.height  = p.out_param.h ;
      
      ParamGradIn.dtype   = ParamIn.dtype   = DTYPE_FLOAT ;
      ParamGradIn.batch   = ParamIn.batch   = p.in_param.n ;
      ParamGradIn.channel = ParamIn.channel = p.in_param.c ;
      ParamGradIn.height  = ParamIn.height  = p.in_param.h ;
      ParamGradIn.width   = ParamIn.width   = p.in_param.w ;

      ParamPool.windowWidth  = p.col_window ; 
      ParamPool.windowHeight = p.row_window ; 
      ParamPool.strideWidth  = p.col_stride ; 
      ParamPool.strideHeight = p.row_stride ; 
      ParamPool.padWidth     = p.col_padding / 2 ; 
      ParamPool.padHeight    = p.row_padding / 2 ; 

      vednnMaxPoolingBackward(&ParamGradOut,   pGradOut, 
                       	      &ParamOut,       pOut, 
                              &ParamIn,        pIn,
                       	      &ParamGradIn,    pGradIn, 
                     	      &ParamPool ) ;
    }     

#ifdef _DEBUG
    fprintf(stderr, "[end]maxpooling_backprop\n");
#endif
    return 0;
}
