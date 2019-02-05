#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "libvetfkernel.h"

#include "veintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))



int transpose4_0231(uint64_t out, uint64_t in, const int32_t* dim_size)
{
  float* po = (float*)(out);
  const float* pi = (float*)(in);


#if 0
  uint64_t si2 = dim_size[3];
  uint64_t si1 = si2 * dim_size[2];
  uint64_t si0 = si1 * dim_size[1];

  uint64_t so2 = dim_size[1];
  uint64_t so1 = so2 * dim_size[3];
  uint64_t so0 = so1 * dim_size[2];

  for (int64_t i0 = 0; i0 < dim_size[0]; ++i0) {
    for (int64_t i1 = 0; i1 < dim_size[2]; ++i1) {
      for (int64_t i2 = 0; i2 < dim_size[3]; ++i2) {
        for (int64_t i3 = 0; i3 < dim_size[1]; ++i3) {
          po[i0 * so0 + i1 * so1 + i2 * so2 + i3]
            = pi[i0 * si0 + i1 * si2 + i2 + i3 * si1];
        }
      }
    }
  }
#else
  const uint64_t d0 = dim_size[0] ;
  const uint64_t d1 = dim_size[1] ;
  const uint64_t d2 = dim_size[2] ;
  const uint64_t d3 = dim_size[3] ;

  const uint64_t d23 = d2 * d3 ;

  if( d1 < (VLEN>>1) && d23 > (d1<<2) ) {
    if( (d23&0x1)==0 && (in&0x7)==0          // d23 is even and input-array is 8-bytes aligned.
         && (d1&0x1)==0 && (out&0x7)==0 ) {  // d1 is even and output-array is 8-bytes aligned.
      const int64_t d23_half = d23 >> 1 ;
      for (int64_t i0 = 0; i0 < d0; ++i0) {
        for (int64_t i3 = 0; i3 < d1; i3+=2) {
          for (int64_t i12 = 0; i12 < d23_half; i12+=VLEN) {
            const int64_t vl = d23_half - i12 < VLEN ? d23_half - i12 : VLEN ;
            _ve_lvl(vl) ;
            __vr vr_in_0 = _ve_vld_vss(8, pi+2*i12+(i3  )*d23) ;
            __vr vr_in_1 = _ve_vld_vss(8, pi+2*i12+(i3+1)*d23) ;
            _ve_vst_vss(_ve_vshf_vvvs(vr_in_0, vr_in_1, VE_VSHUFFLE_ZLYL), 8*d1, po+d1*(2*i12  )+i3) ;
            _ve_vst_vss(_ve_vshf_vvvs(vr_in_0, vr_in_1, VE_VSHUFFLE_ZUYU), 8*d1, po+d1*(2*i12+1)+i3) ;
          }
        }
        pi += d1*d2*d3 ;
        po += d1*d2*d3 ;
      }
    }
    else {
      for (int64_t i0 = 0; i0 < d0; ++i0) {
        for (int64_t i3 = 0; i3 < d1; ++i3) {
          for (int64_t i12 = 0; i12 < d23; i12+=VLEN) {
            const int64_t vl = d23 - i12 < VLEN ? d23 - i12 : VLEN ;
            _ve_lvl(vl) ;
            __vr vr_in = _ve_vldu_vss(4, pi+i12+i3*d23) ;
            _ve_vstu_vss(vr_in, 4*d1, po+d1*i12+i3) ;
          }
        }
        pi += d1*d2*d3 ;
        po += d1*d2*d3 ;
      }
    }
  }
  else {
    for (int64_t i0 = 0; i0 < d0; ++i0) {
      for (int64_t i12 = 0; i12 < d23; ++i12) {
        for (int64_t i3 = 0; i3 < d1; i3+=VLEN) {
          const int64_t vl = d1 - i3 < VLEN ? d1 - i3 : VLEN ;
          _ve_lvl(vl) ;
          __vr vr_in = _ve_vldu_vss(4*d23, pi+i12+i3*d23) ;
          _ve_vstu_vss(vr_in, 4, po+d1*i12+i3) ;
        }
      }
      pi += d1*d2*d3 ;
      po += d1*d2*d3 ;
    }
  }
#endif

  return 0;
}

int transpose4_0312(uint64_t out, uint64_t in, const int32_t* dim_size)
{
  float* po = (float*)(out);
  const float* pi = (float*)(in);


#if 0
  uint64_t si2 = dim_size[3];
  uint64_t si1 = si2 * dim_size[2];
  uint64_t si0 = si1 * dim_size[1];

  uint64_t so2 = dim_size[2];
  uint64_t so1 = so2 * dim_size[1];
  uint64_t so0 = so1 * dim_size[3];

  for (int64_t i0 = 0; i0 < dim_size[0]; ++i0) {
    for (int64_t i1 = 0; i1 < dim_size[3]; ++i1) {
      for (int64_t i2 = 0; i2 < dim_size[1]; ++i2) {
        for (int64_t i3 = 0; i3 < dim_size[2]; ++i3) {
          po[i0 * so0 + i1 * so1 + i2 * so2 + i3]
            = pi[i0 * si0 + i1 + i2 * si1 + i3 * si2];
        }
      }
    }
  }
#else
  const uint64_t d0 = dim_size[0] ;
  const uint64_t d1 = dim_size[1] ;
  const uint64_t d2 = dim_size[2] ;
  const uint64_t d3 = dim_size[3] ;

  const uint64_t d12 = d1 * d2 ;

  if( (d3&0x1)==0 && (in&0x7)==0 ) { // d3 is even and input-array is 8-bytes aligned.
    if ( (d12&0x1)==0 && (out&0x7)==0 ) { // d12 is even and output-array is 8-bytes aligned.
      const int64_t d12_half = d12 >> 1 ;
      for (int64_t i0 = 0; i0 < d0; ++i0) {
        for (int64_t i1 = 0; i1 < d3; i1+=2) {
          for (int64_t i23 = 0; i23 < d12_half; i23+=VLEN) {
            const int64_t vl = d12_half - i23 < VLEN ? d12_half - i23 : VLEN ;
            _ve_lvl(vl) ;
            __vr vr_in_0 = _ve_vld_vss(8*d3, pi+i1+(2*i23  )*d3) ;
            __vr vr_in_1 = _ve_vld_vss(8*d3, pi+i1+(2*i23+1)*d3) ;
            _ve_vst_vss(_ve_vshf_vvvs(vr_in_0, vr_in_1, VE_VSHUFFLE_ZLYL ), 8, po+(i1+0)*d12+2*i23) ;
            _ve_vst_vss(_ve_vshf_vvvs(vr_in_0, vr_in_1, VE_VSHUFFLE_ZUYU ), 8, po+(i1+1)*d12+2*i23) ;
          }
        }
        pi += d1*d2*d3 ;
        po += d1*d2*d3 ;
      }
    }
    else {
      for (int64_t i0 = 0; i0 < d0; ++i0) {
        for (int64_t i1 = 0; i1 < d3; i1+=2) {
          for (int64_t i23 = 0; i23 < d12; i23+=VLEN) {
            const int64_t vl = d12 - i23 < VLEN ? d12 - i23 : VLEN ;
            _ve_lvl(vl) ;
            __vr vr_in = _ve_vld_vss(4*d3, pi+i1+i23*d3) ;
            _ve_vstl_vss(vr_in, 4, po+(i1+0)*d12+i23) ;
            _ve_vstu_vss(vr_in, 4, po+(i1+1)*d12+i23) ;
          }
        }
        pi += d1*d2*d3 ;
        po += d1*d2*d3 ;
      }
    }
  }
  else {
    for (int64_t i0 = 0; i0 < d0; ++i0) {
      for (int64_t i1 = 0; i1 < d3; ++i1) {
        for (int64_t i23 = 0; i23 < d12; i23+=VLEN) {
          const int64_t vl = d12 - i23 < VLEN ? d12 - i23 : VLEN ;
          _ve_lvl(vl) ;
          __vr vr_in = _ve_vldu_vss(4*d3, pi+i1+i23*d3) ;
          _ve_vstu_vss(vr_in, 4, po+i1*d12+i23) ;
        }
      }
      pi += d1*d2*d3 ;
      po += d1*d2*d3 ;
    }
  }
#endif

  return 0;
}



