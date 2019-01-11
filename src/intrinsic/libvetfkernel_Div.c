#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "libvetfkernel.h"

#include "veintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))


int div_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
	float* po = (float*)(out);
	const float* pi0 = (const float*)(in0);
	float i1 = 1.f/(*(const float*)(in1));


	if(n>VLEN*2){
		const uint64_t alignIn = ((const uint64_t)in0) & 0x07;
		const uint64_t alignOut = ((const uint64_t)out) & 0x07;
		if((alignIn==0)&&(alignOut==0)&&(n%2==0)){
			unsigned long int li1 = _ve_pack_f32a(&i1);
			if(n%(8*VLEN)==0){
				_ve_lvl(VLEN);
				for (size_t i = 0; i < n; i+=8*VLEN) {
					__vr vr_pin1 = _ve_vld_vss(8,pi0+i+2*VLEN*0);
					__vr vr_pin2 = _ve_vld_vss(8,pi0+i+2*VLEN*1);
					__vr vr_pin3 = _ve_vld_vss(8,pi0+i+2*VLEN*2);
					__vr vr_pin4 = _ve_vld_vss(8,pi0+i+2*VLEN*3);
					__vr vr_mul1 = _ve_pvfmul_vsv(li1, vr_pin1);
					__vr vr_mul2 = _ve_pvfmul_vsv(li1, vr_pin2);
					__vr vr_mul3 = _ve_pvfmul_vsv(li1, vr_pin3);
					__vr vr_mul4 = _ve_pvfmul_vsv(li1, vr_pin4);
					_ve_vst_vss(vr_mul1,8,po+i+2*VLEN*0);
					_ve_vst_vss(vr_mul2,8,po+i+2*VLEN*1);
					_ve_vst_vss(vr_mul3,8,po+i+2*VLEN*2);
					_ve_vst_vss(vr_mul4,8,po+i+2*VLEN*3);
				}
			}else if((n%8==0)&&(n>VLEN*8)){
				for (size_t i = 0; i < n; i+=8*VLEN) {
					const int64_t vlen = (n-i < 8*VLEN ? n-i : 8*VLEN) >> 3;
					_ve_lvl(vlen);
					__vr vr_pin1 = _ve_vld_vss(8,pi0+i+2*vlen*0);
					__vr vr_pin2 = _ve_vld_vss(8,pi0+i+2*vlen*1);
					__vr vr_pin3 = _ve_vld_vss(8,pi0+i+2*vlen*2);
					__vr vr_pin4 = _ve_vld_vss(8,pi0+i+2*vlen*3);
					__vr vr_mul1 = _ve_pvfmul_vsv(li1, vr_pin1);
					__vr vr_mul2 = _ve_pvfmul_vsv(li1, vr_pin2);
					__vr vr_mul3 = _ve_pvfmul_vsv(li1, vr_pin3);
					__vr vr_mul4 = _ve_pvfmul_vsv(li1, vr_pin4);
					_ve_vst_vss(vr_mul1,8,po+i+2*vlen*0);
					_ve_vst_vss(vr_mul2,8,po+i+2*vlen*1);
					_ve_vst_vss(vr_mul3,8,po+i+2*vlen*2);
					_ve_vst_vss(vr_mul4,8,po+i+2*vlen*3);
				}
			}else{
				for (size_t i = 0; i < n; i+=2*VLEN) {
					const int64_t vlen = (n-i < 2*VLEN ? n-i : 2*VLEN) >> 1;
					_ve_lvl(vlen);
					__vr vr_pin = _ve_vld_vss(8,pi0+i);
					__vr vr_mul = _ve_pvfmul_vsv(li1, vr_pin);
					_ve_vst_vss(vr_mul,8,po+i);
				}
			}
		}else{
			for (size_t i = 0; i < n; i+=VLEN) {
				const int64_t vlen = n-i < VLEN ? n-i : VLEN;
				_ve_lvl(vlen);
				__vr vr_pin = _ve_vldu_vss(4,pi0+i);
				__vr vr_mul = _ve_vfmuls_vsv(i1, vr_pin);
				_ve_vstu_vss(vr_mul,4,po+i);
			}
		}
	}else if(n>VLEN){
		for (size_t i = 0; i < n; i+=VLEN) {
			const int64_t vlen = n-i < VLEN ? n-i : VLEN;
			_ve_lvl(vlen);
			__vr vr_pin = _ve_vldu_vss(4,pi0+i);
			__vr vr_mul = _ve_vfmuls_vsv(i1, vr_pin);
			_ve_vstu_vss(vr_mul,4,po+i);
		}
	}else if(n<17){
		for (size_t i = 0; i < n; i++)
			po[i] = pi0[i] * i1;
	}else{
		_ve_lvl(n);
		__vr vr_pin = _ve_vldu_vss(4,pi0);
		__vr vr_mul = _ve_vfmuls_vsv(i1, vr_pin);
		_ve_vstu_vss(vr_mul,4,po);
	}

	return 0;
}


int div2_nn_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n0, size_t n1)
{
	float* po = (float*)(out);
	const float* pi0 = (const float*)(in0);
	const float* pi1 = (const float*)(in1);

	if((n0>VLEN/3)&&(n0%4==0)){
		if(n1<=VLEN){
			if(n0<=VLEN){
				float temp[VLEN];
				_ve_lvl(n0);
				__vr vr_pin = _ve_vldu_vss(4,pi1);
				__vr vr_div = _ve_vfdivs_vsv(1.0f, vr_pin);
				_ve_vstu_vss(vr_div,4,temp);

				for (int ii = 0 ; ii < n0; ii+=4) {
					float dval1 = temp[ii+0];
					float dval2 = temp[ii+1];
					float dval3 = temp[ii+2];
					float dval4 = temp[ii+3];
					_ve_lvl(n1);
					for (size_t j = 0; j < n1; j+=VLEN) {
						__vr vr_pin1 = _ve_vldu_vss(4,pi0+(ii+0)*n1+j);
						__vr vr_pin2 = _ve_vldu_vss(4,pi0+(ii+1)*n1+j);
						__vr vr_pin3 = _ve_vldu_vss(4,pi0+(ii+2)*n1+j);
						__vr vr_pin4 = _ve_vldu_vss(4,pi0+(ii+3)*n1+j);
						__vr vr_mul1 = _ve_vfmuls_vsv(dval1, vr_pin1);
						__vr vr_mul2 = _ve_vfmuls_vsv(dval2, vr_pin2);
						__vr vr_mul3 = _ve_vfmuls_vsv(dval3, vr_pin3);
						__vr vr_mul4 = _ve_vfmuls_vsv(dval4, vr_pin4);
						_ve_vstu_vss(vr_mul1,4,po+(ii+0)*n1+j);
						_ve_vstu_vss(vr_mul2,4,po+(ii+1)*n1+j);
						_ve_vstu_vss(vr_mul3,4,po+(ii+2)*n1+j);
						_ve_vstu_vss(vr_mul4,4,po+(ii+3)*n1+j);
					}
				}
#if 1
                }else{
                                float temp[VLEN];
                                for (size_t i = 0; i < n0; i+=VLEN) {
                                        const int64_t vlen = n0-i < VLEN ? n0-i : VLEN;
                                        _ve_lvl(vlen);
                                        __vr vr_pin = _ve_vldu_vss(4,pi1+i);
                                        __vr vr_div = _ve_vfdivs_vsv(1.0f, vr_pin);
                                        _ve_vstu_vss(vr_div,4,temp);

                                        for (int ii = i ; ii < vlen+i; ii+=4) {
                                                float dval1 = temp[ii-i+0];
                                                float dval2 = temp[ii-i+1];
                                                float dval3 = temp[ii-i+2];
                                                float dval4 = temp[ii-i+3];
                                                _ve_lvl(n1);
                                                for (size_t j = 0; j < n1; j+=VLEN) {
                                                        __vr vr_pin1 = _ve_vldu_vss(4,pi0+(ii+0)*n1+j);
                                                        __vr vr_pin2 = _ve_vldu_vss(4,pi0+(ii+1)*n1+j);
                                                        __vr vr_pin3 = _ve_vldu_vss(4,pi0+(ii+2)*n1+j);
                                                        __vr vr_pin4 = _ve_vldu_vss(4,pi0+(ii+3)*n1+j);
                                                        __vr vr_mul1 = _ve_vfmuls_vsv(dval1, vr_pin1);
                                                        __vr vr_mul2 = _ve_vfmuls_vsv(dval2, vr_pin2);
                                                        __vr vr_mul3 = _ve_vfmuls_vsv(dval3, vr_pin3);
                                                        __vr vr_mul4 = _ve_vfmuls_vsv(dval4, vr_pin4);
                                                        _ve_vstu_vss(vr_mul1,4,po+(ii+0)*n1+j);
                                                        _ve_vstu_vss(vr_mul2,4,po+(ii+1)*n1+j);
                                                        _ve_vstu_vss(vr_mul3,4,po+(ii+2)*n1+j);
                                                        _ve_vstu_vss(vr_mul4,4,po+(ii+3)*n1+j);
                                                }
                                        }
                                }
                        }

#else
			}else{
				float temp[VLEN];
				for (size_t i = 0; i < n0; i+=VLEN) {
					const int64_t vlen = n0-i < VLEN ? n0-i : VLEN;
					_ve_lvl(vlen);
					__vr vr_pin = _ve_vldu_vss(4,pi1+i);
					__vr vr_div = _ve_vfdivs_vsv(1.0f, vr_pin);
					_ve_vstu_vss(vr_div,4,temp);

					for (int ii = i ; ii < vlen+i; ii+=4) {
						float dval1 = temp[ii+0];
						float dval2 = temp[ii+1];
						float dval3 = temp[ii+2];
						float dval4 = temp[ii+3];
						_ve_lvl(n1);
						for (size_t j = 0; j < n1; j+=VLEN) {
							__vr vr_pin1 = _ve_vldu_vss(4,pi0+(ii+0)*n1+j);
							__vr vr_pin2 = _ve_vldu_vss(4,pi0+(ii+1)*n1+j);
							__vr vr_pin3 = _ve_vldu_vss(4,pi0+(ii+2)*n1+j);
							__vr vr_pin4 = _ve_vldu_vss(4,pi0+(ii+3)*n1+j);
							__vr vr_mul1 = _ve_vfmuls_vsv(dval1, vr_pin1);
							__vr vr_mul2 = _ve_vfmuls_vsv(dval2, vr_pin2);
							__vr vr_mul3 = _ve_vfmuls_vsv(dval3, vr_pin3);
							__vr vr_mul4 = _ve_vfmuls_vsv(dval4, vr_pin4);
							_ve_vstu_vss(vr_mul1,4,po+(ii+0)*n1+j);
							_ve_vstu_vss(vr_mul2,4,po+(ii+1)*n1+j);
							_ve_vstu_vss(vr_mul3,4,po+(ii+2)*n1+j);
							_ve_vstu_vss(vr_mul4,4,po+(ii+3)*n1+j);
						}
					}
				}
			}
#endif
		}else{
			float temp[VLEN];
			for (size_t i = 0; i < n0; i+=VLEN) {
				const int64_t vlen = n0-i < VLEN ? n0-i : VLEN;
				_ve_lvl(vlen);
				__vr vr_pin = _ve_vldu_vss(4,pi1+i);
				__vr vr_div = _ve_vfdivs_vsv(1.0f, vr_pin);
				_ve_vstu_vss(vr_div,4,temp);

				for (int ii = i ; ii < vlen+i; ii+=4) {
					float dval1 = temp[ii+0];
					float dval2 = temp[ii+1];
					float dval3 = temp[ii+2];
					float dval4 = temp[ii+3];
					for (size_t j = 0; j < n1; j+=VLEN) {
						const int64_t vlen1 = n1-j < VLEN ? n1-j : VLEN;
						_ve_lvl(vlen1);

						__vr vr_pin1 = _ve_vldu_vss(4,pi0+(ii+0)*n1+j);
						__vr vr_pin2 = _ve_vldu_vss(4,pi0+(ii+1)*n1+j);
						__vr vr_pin3 = _ve_vldu_vss(4,pi0+(ii+2)*n1+j);
						__vr vr_pin4 = _ve_vldu_vss(4,pi0+(ii+3)*n1+j);
						__vr vr_mul1 = _ve_vfmuls_vsv(dval1, vr_pin1);
						__vr vr_mul2 = _ve_vfmuls_vsv(dval2, vr_pin2);
						__vr vr_mul3 = _ve_vfmuls_vsv(dval3, vr_pin3);
						__vr vr_mul4 = _ve_vfmuls_vsv(dval4, vr_pin4);
						_ve_vstu_vss(vr_mul1,4,po+(ii+0)*n1+j);
						_ve_vstu_vss(vr_mul2,4,po+(ii+1)*n1+j);
						_ve_vstu_vss(vr_mul3,4,po+(ii+2)*n1+j);
						_ve_vstu_vss(vr_mul4,4,po+(ii+3)*n1+j);
					}
				}
			}
		}
	}else if(n0<n1){
		for (size_t i = 0; i < n0; ++i) {
			float dval = 1.0f / pi1[i];
			for (size_t j = 0; j < n1; j+=VLEN) {
				const int64_t vlen = n1-j < VLEN ? n1-j : VLEN;
				_ve_lvl(vlen);
				__vr vr_pin = _ve_vldu_vss(4,pi0+i*n1+j);
				__vr vr_mul = _ve_vfmuls_vsv(dval, vr_pin);
				_ve_vstu_vss(vr_mul,4,po+i*n1+j);
			}
		}
	}else{
		for (size_t i = 0; i < n0; i+=VLEN) {
			const int64_t vlen = n0-i < VLEN ? n0-i : VLEN;
			_ve_lvl(vlen);
			__vr vr_pi1 = _ve_vldu_vss(4,pi1+i);
			__vr dval = _ve_vfdivs_vsv(1.0f,vr_pi1);
			for (size_t j = 0; j < n1; j++) {
				__vr vr_pin = _ve_vldu_vss(4*n1,pi0+i*n1+j);
				__vr vr_mul = _ve_vfmuls_vvv(dval, vr_pin);
				_ve_vstu_vss(vr_mul,4*n1,po+i*n1+j);
			}
		}
	}

	return 0;
}


