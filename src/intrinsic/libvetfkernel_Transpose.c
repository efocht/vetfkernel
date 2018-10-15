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

	uint64_t si2 = dim_size[3];
	uint64_t si1 = si2 * dim_size[2];
	uint64_t si0 = si1 * dim_size[1];

	uint64_t so2 = dim_size[1];
	uint64_t so1 = so2 * dim_size[3];
	uint64_t so0 = so1 * dim_size[2];





	if(dim_size[1]*2<VLEN){
		_ve_lvl(VLEN);
		__vr vr_256 = _ve_vseq_v();
		__vr vr_i1,vr_i3;
		int d_num = VLEN/dim_size[1]; 	
		if(dim_size[1]==64){
			vr_i1 = _ve_vsrl_vvs(vr_256,6); // /64
			vr_i3 = _ve_vand_vsv(63,vr_256); // %64
		}else{
			vr_i1 = _ve_vdivul_vvs(vr_256,dim_size[1]);
			vr_i3 = _ve_vsubul_vvv(vr_256,_ve_vmulul_vsv(dim_size[1],vr_i1));
		}
		__vr vr_pos = _ve_vaddul_vvv(vr_i1,_ve_vmulul_vsv(si1,vr_i3));



		for (int64_t i0 = 0; i0 < dim_size[0]; ++i0) {
			for (int64_t i1 = 0; i1 < dim_size[2]*dim_size[3]; i1+=d_num) {
				const int64_t vlen = dim_size[1]*(dim_size[2]*dim_size[3]-i1 < d_num ? dim_size[2]*dim_size[3]-i1:d_num);
				_ve_lvl(vlen);
				__vr vr_ad = _ve_vaddul_vsv((unsigned long)(pi+i0*si0+i1),_ve_vmulul_vsv(4, vr_pos));
				__vr vr_pi = _ve_vgtu_vv(vr_ad);
				_ve_vstu_vss(vr_pi,4,po+i0*so0+i1*dim_size[1]);
			}
		}
	}else{
		for (int64_t i0 = 0; i0 < dim_size[0]; ++i0) {
			for (int64_t i1 = 0; i1 < dim_size[2]*dim_size[3]; ++i1) {
				for (int64_t i3 = 0; i3 < dim_size[1]; i3+=VLEN) {
					const int64_t vlen = dim_size[1]-i3 < VLEN ? dim_size[1]-i3:VLEN;
					_ve_lvl(vlen);
					__vr vr_pi = _ve_vldu_vss(4*si1,pi+i0*si0+i1+i3);
					_ve_vstu_vss(vr_pi,4,po+i0*so0+i1*so2+i3);
				}
			}
		}

	}





	return 0;
}

int transpose4_0312(uint64_t out, uint64_t in, const int32_t* dim_size)
{
	float* po = (float*)(out);
	const float* pi = (float*)(in);

	uint64_t si2 = dim_size[3];
	uint64_t si1 = si2 * dim_size[2];
	uint64_t si0 = si1 * dim_size[1];

	uint64_t so2 = dim_size[2];
	uint64_t so1 = so2 * dim_size[1];
	uint64_t so0 = so1 * dim_size[3];



	if(dim_size[1]*dim_size[2]*2<VLEN){
		_ve_lvl(VLEN);
		__vr vr_256 = _ve_vseq_v();
		__vr vr_i2,vr_i3;
		int d_num = VLEN/(dim_size[1]*dim_size[2]);
		if((dim_size[1]*dim_size[2])==64){
			vr_i3 = _ve_vsrl_vvs(vr_256,6); // /64
			vr_i2 = _ve_vand_vsv(63,vr_256); // %64
		}else{
			vr_i3 = _ve_vdivul_vvs(vr_256,(dim_size[1]*dim_size[2]));
			vr_i2 = _ve_vsubul_vvv(vr_256,_ve_vmulul_vsv((dim_size[1]*dim_size[2]),vr_i3));
		}
		__vr vr_pos = _ve_vaddul_vvv(vr_i3,_ve_vmulul_vsv(si2,vr_i2));



		for (int64_t i0 = 0; i0 < dim_size[0]; ++i0) {
			for (int64_t i1 = 0; i1 < dim_size[3]; i1+=d_num) {
				const int64_t vlen = dim_size[1]*dim_size[2]*(dim_size[3]-i1 < d_num ? dim_size[3]-i1:d_num);
				_ve_lvl(vlen);
				__vr vr_ad = _ve_vaddul_vsv((unsigned long)(pi + i0 * si0 + i1),_ve_vmulul_vsv(4, vr_pos));
				__vr vr_pi = _ve_vgtu_vv(vr_ad);
				_ve_vstu_vss(vr_pi,4,po + i0 * so0 + i1 * so1);
			}
		}

	}else if(dim_size[1]*dim_size[2]<=VLEN){

#if 1
		_ve_lvl(dim_size[1]*dim_size[2]);

		for (int64_t i0 = 0; i0 < dim_size[0]; ++i0) {
			for (int64_t i1 = 0; i1 < dim_size[3]; i1+=2) {
				__vr vr_pi0 = _ve_vld_vss(4*si2,pi + i0 * si0 + i1);
				__vr vr_pi1 = _ve_vsll_vvs(vr_pi0,32);
				_ve_vstu_vss(vr_pi1,4,po + i0 * so0 + (i1+0) * so1);
				_ve_vstu_vss(vr_pi0,4,po + i0 * so0 + (i1+1) * so1);
			}
		}

#endif


	}else if(dim_size[1]*dim_size[2]%2==0){

#if 1
		for (int64_t i0 = 0; i0 < dim_size[0]; ++i0) {
			for (int64_t i1 = 0; i1 < dim_size[3]; i1+=2) {
				for (int64_t i3 = 0; i3 < dim_size[1]*dim_size[2]; i3+=VLEN) {
					const int64_t vlen = dim_size[1]*dim_size[2]-i3 < VLEN ? dim_size[1]*dim_size[2]-i3:VLEN;
					_ve_lvl(vlen);
					__vr vr_pi0 = _ve_vld_vss(4*si2,pi + i0 * si0 + si2*i3 + i1);
					__vr vr_pi1 = _ve_vsll_vvs(vr_pi0,32);
					_ve_vstu_vss(vr_pi1,4,po + i0 * so0 + (i1+0) * so1 + i3);
					_ve_vstu_vss(vr_pi0,4,po + i0 * so0 + (i1+1) * so1 + i3);
				}
			}
		}
#endif


	}else{
		for (int64_t i0 = 0; i0 < dim_size[0]; ++i0) {
			for (int64_t i1 = 0; i1 < dim_size[3]; i1++) {
				for (int64_t i3 = 0; i3 < dim_size[1]*dim_size[2]; i3+=VLEN) {
					const int64_t vlen = dim_size[1]*dim_size[2]-i3 < VLEN ? dim_size[1]*dim_size[2]-i3:VLEN;
					_ve_lvl(vlen);
					__vr vr_pi = _ve_vldu_vss(4*si2,pi + i0 * si0 + si2*i3 + i1);
					_ve_vstu_vss(vr_pi,4,po + i0 * so0 + i1 * so1 + i3);
				}
			}
		}


	}


	return 0;
}



