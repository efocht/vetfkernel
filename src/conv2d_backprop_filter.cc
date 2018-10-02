#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <stdlib.h>

#include <vednn.h>

#include "kernel.h"

REGISTER_KERNEL("Conv2DBackpropFilter", "conv2d_backprop_filter");

extern "C" {
    int conv2d_backprop_filter(const void* arg, size_t len);
}

struct TensorParam {
    int w,h,c,n ;
} ;

struct ConvParam {
    uint64_t out_bp;
    uint64_t in;
    uint64_t filter_bp;
    TensorParam out_bp_param;
    TensorParam in_param;
    TensorParam filter_bp_param;

    int row_stride;
    int col_stride;
    int row_dilation;
    int col_dilation;
    int row_padding;
    int col_padding;

    int data_format;
    int data_type;
};

int conv2d_backprop_filter(const void* arg, size_t len)
{

#ifdef _DEBUG
    fprintf(stderr, "[start]conv2d_backprop_filter\n");
#endif
    assert(len == sizeof(ConvParam));
    const ConvParam& p = *(ConvParam*)arg;

#ifdef _DEBUG
    fprintf(stderr, "conv2d_backprop_filter: data_format=%d data_type=%d\n", p.data_format, p.data_type);
    // assert(p.data_type   == 1 ) ; // float
    // assert(p.data_format == 1 ) ; // NHWC

    fprintf(stderr, "conv2d_backprop_filter: out_bp     (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.out_bp_param.n, p.out_bp_param.c, p.out_bp_param.h, p.out_bp_param.w ) ;
    fprintf(stderr, "conv2d_backprop_filter: input      (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.in_param.n, p.in_param.c, p.in_param.h, p.in_param.w ) ;
    fprintf(stderr, "conv2d_backprop_filter: filter_bp  (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.filter_bp_param.n, p.filter_bp_param.c, p.filter_bp_param.h, p.filter_bp_param.w ) ;

    fprintf(stderr, "conv2d_backprop_filter: stride=%dx%d dilation=%dx%d padding=%dx%d\n", 
            p.col_stride,   p.row_stride,
            p.col_dilation, p.row_dilation,
            p.col_padding,   p.row_padding);
#endif


    const int N = p.filter_bp_param.n ;
    const int C = p.filter_bp_param.c ;
    const int H = p.filter_bp_param.h ;
    const int W = p.filter_bp_param.w ;
    
    float * transformed_bp_filter = NULL ;  
    if( p.filter_bp_param.n > 1 || p.filter_bp_param.c > 1 ) {
      transformed_bp_filter = (float *) malloc(sizeof(float)*N*C*H*W) ;
      for(size_t i=0; i<N*C*H*W; i++) transformed_bp_filter[i] = 0.f ;
    }
    else { 
      float * filter_bp = (float *) p.filter_bp ;     
      for(size_t i=0; i<N*C*H*W; i++) filter_bp[i] = 0.f ;
    }
    
    void *pIn         = (void *) p.in ;
    void *pGradOut    = (void *) p.out_bp ;
    void *pGradFilter = (transformed_bp_filter != NULL) ? (void*)transformed_bp_filter : (void*)p.filter_bp  ;
    
    vednnTensorParam_t ParamIn ;
    vednnTensorParam_t ParamGradOut ;
    vednnFilterParam_t ParamGradFilter ;

    vednnConvolutionParam_t ParamConv ;

    ParamIn.dtype   = DTYPE_FLOAT ;
    ParamIn.batch   = p.in_param.n ;
    ParamIn.channel = p.in_param.c ;
    ParamIn.height  = p.in_param.h ;
    ParamIn.width   = p.in_param.w ;

    ParamGradOut.dtype   = DTYPE_FLOAT ;
    ParamGradOut.batch   = p.out_bp_param.n ;
    ParamGradOut.channel = p.out_bp_param.c ;
    ParamGradOut.width   = p.out_bp_param.w ;
    ParamGradOut.height  = p.out_bp_param.h ;
    
    ParamGradFilter.dtype      = DTYPE_FLOAT ;
    ParamGradFilter.inChannel  = p.in_param.c ;
    ParamGradFilter.outChannel = p.out_bp_param.c ;
    ParamGradFilter.width      = p.filter_bp_param.w ;
    ParamGradFilter.height     = p.filter_bp_param.h ;

    ParamConv.group          = 1 ;
    ParamConv.strideWidth    = p.col_stride ; 
    ParamConv.strideHeight   = p.row_stride ; 
    ParamConv.padWidth       = p.col_padding / 2 ; 
    ParamConv.padHeight      = p.row_padding / 2 ; 
    ParamConv.dilationWidth  = p.col_dilation ; 
    ParamConv.dilationHeight = p.row_dilation ; 

    vednnConvolutionBackwardFilter(&ParamIn,         pIn,
                     	           &ParamGradOut,    pGradOut, 
                     	           &ParamGradFilter, pGradFilter,
                     	           &ParamConv,
                     	           VEDNN_CONV_ALGORITHM_DIRECT );

    if( transformed_bp_filter != NULL ) {
      float * filter_bp = (float *) p.filter_bp ;     

#if 0
      for(int n=0; n<N ; n++) {
        for(int c=0; c<C ; c++) {
          for(int h=0; h<H ; h++) {
            for(int w=0; w<W ; w++) {
              filter_bp[((h*W+w)*C+c)*N+n] = transformed_bp_filter[((n*C+c)*H+h)*W+w] ;
 	    }
          }
        }
      }
#else
      for(int n=0; n<N ; n++) {
        for(int c=0; c<C ; c++) {
          for(int hw=0; hw<H*W ; hw++) {
            filter_bp[((hw)*C+c)*N+n] = transformed_bp_filter[((n*C+c)*H)*W+hw] ;
          }
        }
      }
#endif
      free(transformed_bp_filter) ;
    }
#ifdef _DEBUG
    fprintf(stderr, "[end]conv2d_backprop_filter\n");
#endif
    return 0;
}
