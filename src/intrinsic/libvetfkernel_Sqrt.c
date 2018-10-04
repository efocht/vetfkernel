#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "libvetfkernel.h"

#include "veintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))


int sqrt_(uint64_t out, uint64_t in, size_t n)
{

	const float* pi = (const float*)(in);
	float* po = (float*)(out);


	if(VLEN<n){
		for (size_t i = 0; i < n; i+=VLEN) {
			const int64_t vlen = n-i < VLEN ? n-i : VLEN;
			_ve_lvl(vlen);
			__vr vr_pin = _ve_vldu_vss(4,pi+i);
			__vr vr_s = _ve_vfsqrts_vv(vr_pin);
			_ve_vstu_vss(vr_s,4,po+i);
		}
	}else if(n<17){
		for (size_t i = 0; i < n; i++)
			po[i] = sqrtf(pi[i]);

	}else{
		_ve_lvl(n);
		__vr vr_pin = _ve_vldu_vss(4,pi);
		__vr vr_s = _ve_vfsqrts_vv(vr_pin);
		_ve_vstu_vss(vr_s,4,po);
	}
	return 0;
}


