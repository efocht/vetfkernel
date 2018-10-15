#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "libvetfkernel.h"

#include "veintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))


void neg(uint64_t out, uint64_t in, size_t n)
{

	const float* pi = (const float*)(in);
	float* po = (float*)(out);


	if(VLEN<n){
		const uint64_t alignIn = ((const uint64_t)in) & 0x07;
		const uint64_t alignOut = ((const uint64_t)out) & 0x07;


		if((alignIn==0)&&(alignOut==0)&&(n%2==0)){
			__vr vr_zero = _ve_vbrdu_vs_f32(0.f);
			for (size_t i = 0; i < n; i+=2*VLEN) {
				const int64_t vlen = (n-i < 2*VLEN ? n-i : 2*VLEN) >> 1;
				_ve_lvl(vlen);
				__vr vr_pin = _ve_vld_vss(8,pi+i);
				__vr vr_sub = _ve_pvfsub_vvv(vr_zero, vr_pin);
				_ve_vst_vss(vr_sub,8,po+i);
			}
		}else{
			for (size_t i = 0; i < n; i+=VLEN) {
				const int64_t vlen = n-i < VLEN ? n-i : VLEN;
				_ve_lvl(vlen);
				__vr vr_pin = _ve_vldu_vss(4,pi+i);
				__vr vr_sub = _ve_vfsubs_vsv(0.f, vr_pin);
				_ve_vstu_vss(vr_sub,4,po+i);
			}
		}
	}else if(n<=VLEN){
		_ve_lvl(n);
		__vr vr_pin = _ve_vldu_vss(4,pi);
		__vr vr_sub = _ve_vfsubs_vsv(0.f, vr_pin);
		_ve_vstu_vss(vr_sub,4,po);
	}else if(n<17){
		for (size_t i = 0; i < n; i++)
			po[i] = - pi[i];
	}

	return;
}


