#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "libvetfkernel.h"

#include "veintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))


int sub_nn_f32(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
	float* po = (float*)(out);
	const float* pi0 = (const float*)(in0);
	const float* pi1 = (const float*)(in1);

	const uint64_t alignIn0 = ((const uint64_t)in0) & 0x07;
	const uint64_t alignIn1 = ((const uint64_t)in1) & 0x07;
	const uint64_t alignOut = ((const uint64_t)out) & 0x07;


	if((alignIn0==0)&&(alignIn1==0)&&(alignOut==0)&&(n%2==0)){
		for (size_t i = 0; i < n; i+=2*VLEN) {
			const int64_t vlen = (n-i < 2*VLEN ? n-i : 2*VLEN) >> 1;
			_ve_lvl(vlen) ;
			__vr vr_pin0 = _ve_vld_vss(8,pi0+i);
			__vr vr_pin1 = _ve_vld_vss(8,pi1+i);
			__vr vr_sub = _ve_pvfsub_vvv(vr_pin0, vr_pin1);
			_ve_vst_vss(vr_sub,8,po+i);
		}
	}else if(n>7){
		for (size_t i = 0; i < n; i+=VLEN) {
			const int64_t vlen = n-i < VLEN ? n-i : VLEN;
			_ve_lvl(vlen) ;
			__vr vr_pin0 = _ve_vldu_vss(4,pi0+i);
			__vr vr_pin1 = _ve_vldu_vss(4,pi1+i);
			__vr vr_sub = _ve_vfsubs_vvv(vr_pin0, vr_pin1);
			_ve_vstu_vss(vr_sub,4,po+i);
		}
	}else{
		for (size_t i = 0; i < n; i++) {
			po[i] = pi0[i] - pi1[i];
		}
	}	



	return 0;
}

