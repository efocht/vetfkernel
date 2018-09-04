#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <stdlib.h>

#include <vednn.h>

#include "kernel.h"

REGISTER_KERNEL("Conv2D", "conv2d");

extern "C" {
    int conv2d(const void* arg, size_t len);
}

struct TensorParam {
    int w,h,c,n ;
} ;

struct ConvParam {
    uint64_t in;
    uint64_t filter;
    uint64_t out;
    TensorParam in_param;
    TensorParam filter_param;
    TensorParam out_param;

    int row_stride;
    int col_stride;
    int row_dilation;
    int col_dilation;
    int row_padding;
    int col_padding;

    int data_format;
    int data_type;
};

int conv2d(const void* arg, size_t len)
{

#ifdef _DEBUG
    fprintf(stderr, "[start]conv2d\n");
#endif
    assert(len == sizeof(ConvParam));
    const ConvParam& p = *(ConvParam*)arg;

#ifdef _DEBUG
    fprintf(stderr, "conv2d: data_format=%d data_type=%d\n", p.data_format, p.data_type);
    // assert(p.data_type   == 1 ) ; // float
    // assert(p.data_format == 1 ) ; // NHWC

    fprintf(stderr, "conv2d: input   (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.in_param.n, p.in_param.c, p.in_param.h, p.in_param.w ) ;
    fprintf(stderr, "conv2d: outnput (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.out_param.n, p.out_param.c, p.out_param.h, p.out_param.w ) ;
    fprintf(stderr, "conv2d: filter  (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.filter_param.n, p.filter_param.c, p.filter_param.h, p.filter_param.w ) ;

    fprintf(stderr, "conv2d: stride=%dx%d dilation=%dx%d padding=%dx%d\n", 
            p.col_stride,   p.row_stride,
            p.col_dilation, p.row_dilation,
            p.col_padding,   p.row_padding);
#endif
     
    float * transformed_filter = NULL ;
    if( p.filter_param.n > 1 || p.filter_param.c > 1 ) {
      const int N = p.filter_param.n ;
      const int C = p.filter_param.c ;
      const int H = p.filter_param.h ;
      const int W = p.filter_param.w ;

      float * filter = (float *) p.filter ;     

      transformed_filter = (float *) malloc(sizeof(float)*N*C*H*W) ;

      for(int n=0; n<N ; n++) {
        for(int c=0; c<C ; c++) {
          for(int h=0; h<H ; h++) {
            for(int w=0; w<W ; w++) {
              transformed_filter[((n*C+c)*H+h)*W+w] = filter[((h*W+w)*C+c)*N+n] ; 
 	    }
          }
        }
      }
    }
    
    void *pIn     = (void *) p.in ;
    void *pOut    = (void *) p.out ;
    void *pFilter = (transformed_filter != NULL) ? (void*)transformed_filter : (void*)p.filter  ;
    
    vednnTensorParam_t ParamIn ;
    vednnFilterParam_t ParamFilter ;
    vednnTensorParam_t ParamOut ;

    vednnConvolutionParam_t ParamConv ;

    ParamIn.dtype   = DTYPE_FLOAT ;
    ParamIn.batch   = p.in_param.n ;
    ParamIn.channel = p.in_param.c ;
    ParamIn.height  = p.in_param.h ;
    ParamIn.width   = p.in_param.w ;

    ParamFilter.dtype      = DTYPE_FLOAT ;
    ParamFilter.inChannel  = p.in_param.c ;
    ParamFilter.outChannel = p.out_param.c ;
    ParamFilter.width      = p.filter_param.w ;
    ParamFilter.height     = p.filter_param.h ;
     
    ParamOut.dtype   = DTYPE_FLOAT ;
    ParamOut.batch   = p.out_param.n ;
    ParamOut.channel = p.out_param.c ;
    ParamOut.width   = p.out_param.w ;
    ParamOut.height  = p.out_param.h ;

    ParamConv.group          = 1 ;
    ParamConv.strideWidth    = p.col_stride ; 
    ParamConv.strideHeight   = p.row_stride ; 
    ParamConv.padWidth       = p.col_padding / 2 ; 
    ParamConv.padHeight      = p.row_padding / 2 ; 
    ParamConv.dilationWidth  = p.col_dilation ; 
    ParamConv.dilationHeight = p.row_dilation ; 

    vednnConvolutionForward(&ParamIn,     pIn,
                     	    &ParamFilter, pFilter,
                     	    &ParamOut,    pOut, 
                     	    &ParamConv,
                     	    VEDNN_CONV_ALGORITHM_DIRECT );
    

    if( transformed_filter != NULL ) free(transformed_filter) ;

#ifdef _DEBUG
    fprintf(stderr, "[end]conv2d\n");
#endif
    return 0;
}
