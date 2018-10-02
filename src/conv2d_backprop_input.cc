#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include <stdlib.h>

#include <vednn.h>

#include "kernel.h"

REGISTER_KERNEL("Conv2DBackpropInput", "conv2d_backprop_input");

extern "C" {
    int conv2d_backprop_input(const void* arg, size_t len);
}

struct TensorParam {
    int w,h,c,n ;
} ;

struct ConvParam {
    uint64_t out_bp;
    uint64_t filter;
    uint64_t in_bp;
    TensorParam out_bp_param;
    TensorParam filter_param;
    TensorParam in_bp_param;

    int row_stride;
    int col_stride;
    int row_dilation;
    int col_dilation;
    int row_padding;
    int col_padding;

    int data_format;
    int data_type;
};

int conv2d_backprop_input(const void* arg, size_t len)
{

#ifdef _DEBUG
    fprintf(stderr, "[start] conv2d_backprop_input\n");
#endif
    assert(len == sizeof(ConvParam));
    const ConvParam& p = *(ConvParam*)arg;

#ifdef _DEBUG
    fprintf(stderr, "conv2d_backprop_input: data_format=%d data_type=%d\n", p.data_format, p.data_type);
    // assert(p.data_type   == 1 ) ; // float
    // assert(p.data_format == 1 ) ; // NHWC

    fprintf(stderr, "conv2d_backprop_input: out_bp   (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.out_bp_param.n, p.out_bp_param.c, p.out_bp_param.h, p.out_bp_param.w ) ;
    fprintf(stderr, "conv2d_backprop_input: filter   (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.filter_param.n, p.filter_param.c, p.filter_param.h, p.filter_param.w ) ;
    fprintf(stderr, "conv2d_backprop_input: input_bp (N,C,H,W) = (%d,%d,%d,%d)\n",
            p.in_bp_param.n, p.in_bp_param.c, p.in_bp_param.h, p.in_bp_param.w ) ;

    fprintf(stderr, "conv2d_backprop_filter: stride=%dx%d dilation=%dx%d padding=%dx%d\n", 
            p.col_stride,   p.row_stride,
            p.col_dilation, p.row_dilation,
            p.col_padding,  p.row_padding);
#endif

    const int N = p.filter_param.n ;
    const int C = p.filter_param.c ;
    const int H = p.filter_param.h ;
    const int W = p.filter_param.w ;
    
    float * transformed_filter = NULL ;  
    if( p.filter_param.n > 1 || p.filter_param.c > 1 ) {
      
      transformed_filter = (float *) malloc(sizeof(float)*N*C*H*W) ;
      
      float * filter = (float *) p.filter ;     

#if 0
      for(int n=0; n<N ; n++) {
        for(int c=0; c<C ; c++) {
          for(int h=0; h<H ; h++) {
            for(int w=0; w<W ; w++) {
              transformed_filter[((n*C+c)*H+h)*W+w] = filter[((h*W+w)*C+c)*N+n] ;
 	    }
          }
        }
      }
#else
      for(int n=0; n<N ; n++) {
        for(int c=0; c<C ; c++) {
          for(int hw=0; hw<H*W ; hw++) {
            transformed_filter[((n*C+c)*H)*W+hw] = filter[((hw)*C+c)*N+n] ;
          }
        }
      }
#endif
    }
    
    void *pGradOut    = (void *) p.out_bp ;
    void *pFilter     = (transformed_filter != NULL) ? (void*)transformed_filter : (void*)p.filter  ;
    void *pGradIn     = (void *) p.in_bp ;
    
    vednnTensorParam_t ParamGradOut ;
    vednnFilterParam_t ParamFilter ;
    vednnTensorParam_t ParamGradIn ;

    vednnConvolutionParam_t ParamConv ;

    ParamGradOut.dtype   = DTYPE_FLOAT ;
    ParamGradOut.batch   = p.out_bp_param.n ;
    ParamGradOut.channel = p.out_bp_param.c ;
    ParamGradOut.width   = p.out_bp_param.w ;
    ParamGradOut.height  = p.out_bp_param.h ;
    
    ParamFilter.dtype      = DTYPE_FLOAT ;
    ParamFilter.inChannel  = p.in_bp_param.c ;
    ParamFilter.outChannel = p.out_bp_param.c ;
    ParamFilter.width      = p.filter_param.w ;
    ParamFilter.height     = p.filter_param.h ;
    
    ParamGradIn.dtype   = DTYPE_FLOAT ;
    ParamGradIn.batch   = p.in_bp_param.n ;
    ParamGradIn.channel = p.in_bp_param.c ;
    ParamGradIn.height  = p.in_bp_param.h ;
    ParamGradIn.width   = p.in_bp_param.w ;

    ParamConv.group          = 1 ;
    ParamConv.strideWidth    = p.col_stride ; 
    ParamConv.strideHeight   = p.row_stride ; 
    ParamConv.padWidth       = p.col_padding / 2 ; 
    ParamConv.padHeight      = p.row_padding / 2 ; 
    ParamConv.dilationWidth  = p.col_dilation ; 
    ParamConv.dilationHeight = p.row_dilation ; 

    vednnConvolutionBackwardData(&ParamGradOut,  pGradOut, 
                     	         &ParamFilter,   pFilter,
                                 &ParamGradIn,   pGradIn,
                     	         &ParamConv,
                     	         VEDNN_CONV_ALGORITHM_DIRECT );

    if( transformed_filter != NULL ) {
      free(transformed_filter) ;
    }
#ifdef _DEBUG
    fprintf(stderr, "[end] conv2d_backprop_input\n");
#endif
    return 0;
}
