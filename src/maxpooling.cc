#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <stdlib.h>

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
    int row_padding;
    int col_padding;

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
    
    fprintf(stderr, "maxpooling: window=%dx%d stride=%dx%d, padding=%dx%d\n",
            p.col_window,   p.row_window,
            p.col_stride,   p.row_stride,
            p.col_padding,  p.row_padding );
#endif
     
    {
      void *pIn     = (void *) p.in ;
      void *pOut    = (void *) p.out ;
    
      vednnTensorParam_t ParamIn ;
      vednnTensorParam_t ParamOut ;

      vednnPoolingParam_t ParamPool ;

      ParamIn.dtype   = DTYPE_FLOAT ;
      ParamIn.batch   = p.in_param.n ;
      ParamIn.channel = p.in_param.c ;
      ParamIn.height  = p.in_param.h ;
      ParamIn.width   = p.in_param.w ;

      ParamOut.dtype   = DTYPE_FLOAT ;
      ParamOut.batch   = p.out_param.n ;
      ParamOut.channel = p.out_param.c ;
      ParamOut.width   = p.out_param.w ;
      ParamOut.height  = p.out_param.h ;

      ParamPool.windowWidth  = p.col_window ; 
      ParamPool.windowHeight = p.row_window ; 
      ParamPool.strideWidth  = p.col_stride ; 
      ParamPool.strideHeight = p.row_stride ; 
      ParamPool.padWidth     = p.col_padding / 2 ; 
      ParamPool.padHeight    = p.row_padding / 2 ; 

      vednnMaxPoolingForward(&ParamIn,     pIn,
                       	     &ParamOut,    pOut, 
                     	     &ParamPool ) ;
    }

#ifdef _DEBUG
    fprintf(stderr, "[end]maxpooling\n");
#endif
    return 0;
}
