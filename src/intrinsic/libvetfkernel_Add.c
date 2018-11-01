#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "libvetfkernel.h"

#include "veintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))

int add_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
	float* po = (float*)(out);
	const float* pi0 = (const float*)(in0);
	float i1 = *((const float*)(in1));



	const uint64_t alignIn = ((const uint64_t)in0) & 0x07;
	const uint64_t alignOut = ((const uint64_t)out) & 0x07;


	if((alignIn==0)&&(alignOut==0)&&(n%2==0)&&(n>256)){
		unsigned long int li1 = _ve_pack_f32a(&i1);
		if(n%(8*VLEN)==0){
			_ve_lvl(VLEN);
			for (size_t i = 0; i < n; i+=8*VLEN) {
				__vr vr_pin1 = _ve_vld_vss(8,pi0+i+2*VLEN*0);
				__vr vr_pin2 = _ve_vld_vss(8,pi0+i+2*VLEN*1);
				__vr vr_pin3 = _ve_vld_vss(8,pi0+i+2*VLEN*2);
				__vr vr_pin4 = _ve_vld_vss(8,pi0+i+2*VLEN*3);
				__vr vr_sum1 = _ve_pvfadd_vsv(li1, vr_pin1);
				__vr vr_sum2 = _ve_pvfadd_vsv(li1, vr_pin2);
				__vr vr_sum3 = _ve_pvfadd_vsv(li1, vr_pin3);
				__vr vr_sum4 = _ve_pvfadd_vsv(li1, vr_pin4);
				_ve_vst_vss(vr_sum1,8,po+i+2*VLEN*0);
				_ve_vst_vss(vr_sum2,8,po+i+2*VLEN*1);
				_ve_vst_vss(vr_sum3,8,po+i+2*VLEN*2);
				_ve_vst_vss(vr_sum4,8,po+i+2*VLEN*3);
			}
		}else if(n%8==0){
			for (size_t i = 0; i < n; i+=8*VLEN) {
				const int64_t vlen = (n-i < 8*VLEN ? n-i : 8*VLEN) >> 3;
				_ve_lvl(vlen);
				__vr vr_pin1 = _ve_vld_vss(8,pi0+i+2*vlen*0);
				__vr vr_pin2 = _ve_vld_vss(8,pi0+i+2*vlen*1);
				__vr vr_pin3 = _ve_vld_vss(8,pi0+i+2*vlen*2);
				__vr vr_pin4 = _ve_vld_vss(8,pi0+i+2*vlen*3);
				__vr vr_sum1 = _ve_pvfadd_vsv(li1, vr_pin1);
				__vr vr_sum2 = _ve_pvfadd_vsv(li1, vr_pin2);
				__vr vr_sum3 = _ve_pvfadd_vsv(li1, vr_pin3);
				__vr vr_sum4 = _ve_pvfadd_vsv(li1, vr_pin4);
				_ve_vst_vss(vr_sum1,8,po+i+2*vlen*0);
				_ve_vst_vss(vr_sum2,8,po+i+2*vlen*1);
				_ve_vst_vss(vr_sum3,8,po+i+2*vlen*2);
				_ve_vst_vss(vr_sum4,8,po+i+2*vlen*3);
			}
		}else{
			for (size_t i = 0; i < n; i+=2*VLEN) {
				const int64_t vlen = (n-i < 2*VLEN ? n-i : 2*VLEN) >> 1;
				_ve_lvl(vlen);
				__vr vr_pin = _ve_vld_vss(8,pi0+i);
				__vr vr_sum = _ve_pvfadd_vsv(li1, vr_pin);
				_ve_vst_vss(vr_sum,8,po+i);
			}
		}
	}else if(n>256){
		for (size_t i = 0; i < n; i+=VLEN) {
			const int64_t vlen = n-i < VLEN ? n-i : VLEN;
			_ve_lvl(vlen);
			__vr vr_pin = _ve_vldu_vss(4,pi0+i);
			__vr vr_sum = _ve_vfadds_vsv(i1, vr_pin);
			_ve_vstu_vss(vr_sum,4,po+i);
		}
	}else if(n<17){
		for (size_t i = 0; i < n; i++)
			po[i] = pi0[i] + i1;

	}else{
		_ve_lvl(n);
		__vr vr_pin = _ve_vldu_vss(4,pi0);
		__vr vr_sum = _ve_vfadds_vsv(i1, vr_pin);
		_ve_vstu_vss(vr_sum,4,po);
	}
	return 0;
}

int add_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
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
			__vr vr_sum = _ve_pvfadd_vvv(vr_pin0, vr_pin1);
			_ve_vst_vss(vr_sum,8,po+i);
		}
	}else{
		for (size_t i = 0; i < n; i+=VLEN) {
			const int64_t vlen = n-i < VLEN ? n-i : VLEN;
			_ve_lvl(vlen) ;
			__vr vr_pin0 = _ve_vldu_vss(4,pi0+i);
			__vr vr_pin1 = _ve_vldu_vss(4,pi1+i);
			__vr vr_sum = _ve_vfadds_vvv(vr_pin0, vr_pin1);
			_ve_vstu_vss(vr_sum,4,po+i);
		}
	}





	return 0;
}


