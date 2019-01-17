#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "libvetfkernel.h"

#include "veintrin.h"
#define VLEN	(256)


static inline void _apply_adam_f32_packed(
  float* var, float* m, float *v,
  const float beta1, const float beta2, 
  const float epsilon, const float k,
  const int64_t numElement, 
  const float *grd )
{
  const float one_minus_beta1 = 1.f - beta1 ;
  const float one_minus_beta2 = 1.f - beta2 ;
  const float minus_k         = -k ;

  const uint64_t one_minus_beta1_packed = _ve_pack_f32a(&one_minus_beta1) ;
  const uint64_t one_minus_beta2_packed = _ve_pack_f32a(&one_minus_beta2) ;
  const uint64_t minus_k_packed         = _ve_pack_f32a(&minus_k) ;
  const uint64_t epsilon_packed         = _ve_pack_f32a(&epsilon) ;

  const uint64_t alignVar = ((const uint64_t)var) & 0x07;
  if( alignVar ) {
    m[0] = m[0] + one_minus_beta1 * (grd[0] - m[0]) ;
    v[0] = v[0] + one_minus_beta2 * (grd[0]*grd[0] - v[0]) ;
    var[0] -= k * m[0] / (epsilon + sqrtf(v[0])) ;    
  }

  const int64_t j = alignVar ? 1 : 0 ;
  const int64_t halfElement = (numElement - j) >> 1 ;  
  for(int64_t i=0; i<halfElement; i+=VLEN) {
    const int64_t vl = halfElement - i < VLEN ? halfElement - i : VLEN ;

    _ve_lvl(vl) ;

    __vr vrm   = _ve_vld_vss(8, m+2*i+j) ;
    __vr vrv   = _ve_vld_vss(8, v+2*i+j) ;
    __vr vrgrd = _ve_vld_vss(8, grd+2*i+j) ;
    __vr vrvar = _ve_vld_vss(8, var+2*i+j) ;

    vrm = _ve_pvfmad_vvsv(vrm,
                          one_minus_beta1_packed,
                          _ve_pvfsub_vvv(vrgrd, vrm)) ;
    vrv = _ve_pvfmad_vvsv(vrv,
                          one_minus_beta2_packed,
                          _ve_pvfmsb_vvvv(vrv, vrgrd,vrgrd)) ;

    __vr sqrt_vrv = _ve_vshf_vvvs(_ve_vfsqrts_vv(vrv),
                                  _ve_vfsqrts_vv(_ve_vsll_vvs(vrv,32)) ,
                                  VE_VSHUFFLE_YUZU ) ;

    vrvar = _ve_pvfmad_vvsv(vrvar,
                            minus_k_packed,
                            _ve_pvfdivA_vvv(vrm,
                                            _ve_pvfadd_vsv(epsilon_packed,
                                                           sqrt_vrv))) ;
   
    _ve_vst_vss(vrm, 8, m+2*i+j) ; 
    _ve_vst_vss(vrv, 8, v+2*i+j) ; 
    _ve_vst_vss(vrvar, 8, var+2*i+j) ; 
  }
  
  if( ( !alignVar && (numElement & 0x01)==1 )
      || ( alignVar && (numElement & 0x01)==0 ) ) {
    const int64_t idx = numElement - 1 ; 
    m[idx] = m[idx] + one_minus_beta1 * (grd[idx] - m[idx]) ;
    v[idx] = v[idx] + one_minus_beta2 * (grd[idx]*grd[idx] - v[idx]) ;
    var[idx] -= k * m[idx] / (epsilon + sqrtf(v[idx])) ;    
  }
}

static inline void _apply_adam_f32_defualt(
  float* var, float* m, float *v,
  const float beta1, const float beta2, 
  const float epsilon, const float k,
  const int64_t numElement, 
  const float *grd )
{
  for(int64_t i=0; i<numElement; i+=VLEN) {
    const int64_t vl = numElement - i < VLEN ? numElement - i : VLEN ;

    _ve_lvl(vl) ;

    __vr vrm   = _ve_vldu_vss(4, m+i) ;
    __vr vrv   = _ve_vldu_vss(4, v+i) ;
    __vr vrgrd = _ve_vldu_vss(4, grd+i) ;
    __vr vrvar = _ve_vldu_vss(4, var+i) ;

    vrm = _ve_vfmads_vvsv(vrm,
                          1.f - beta1,
                          _ve_vfsubs_vvv(vrgrd, vrm)) ;
    vrv = _ve_vfmads_vvsv(vrv,
                          1.f - beta2,
                          _ve_vfmsbs_vvvv(vrv, vrgrd,vrgrd)) ;
    vrvar = _ve_vfmads_vvsv(vrvar,
                            -k,
                            _ve_vfdivsA_vvv(vrm,
                                            _ve_vfadds_vsv(epsilon,
                                                           _ve_vfsqrts_vv(vrv)))) ;
   
    _ve_vstu_vss(vrm, 4, m+i) ; 
    _ve_vstu_vss(vrv, 4, v+i) ; 
    _ve_vstu_vss(vrvar, 4, var+i) ; 

  }
}

void _apply_adam_f32(float* var, float* m, float *v,
                     const float beta1, const float beta2, 
                     const float epsilon, const float k,
                     const int64_t numElement, 
                     const float *grd ) 
{
  const uint64_t alignVar = ((const uint64_t)var) & 0x07;
  const uint64_t alignM   = ((const uint64_t)m  ) & 0x07;
  const uint64_t alignV   = ((const uint64_t)v  ) & 0x07;
  const uint64_t alignGrd = ((const uint64_t)grd) & 0x07;
  
  if ( (numElement >= 2*VLEN)
       && (alignVar == alignM) 
       && (alignVar == alignV)
       && (alignVar == alignGrd) )
  {
     _apply_adam_f32_packed(var, m, v, beta1, beta2, epsilon, k, 
                            numElement, grd) ;

  }
  else {
     _apply_adam_f32_defualt(var, m, v, beta1, beta2, epsilon, k, 
                             numElement, grd) ;
  }
}
