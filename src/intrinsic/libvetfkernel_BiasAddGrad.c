#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "libvetfkernel.h"

#include "veintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))




int BiasAddGrad_NHWC(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel)
{
	float* pout = (float*)(output);
	const float* pin = (const float*)(output_backprop);


	const uint64_t alignOut = ((const uint64_t)output) & 0x07;
	const uint64_t alignIn = ((const uint64_t)output_backprop) & 0x07;

	if((alignIn==0)&&(alignOut==0)&&(channel%2==0)&&(channel>256)){
		for (int c = 0; c < channel; c+=2*VLEN) {
			const int64_t vlen = (channel-c < 2*VLEN ? channel-c : 2*VLEN) >> 1;
			_ve_lvl(vlen);
			__vr vr_sum = _ve_vbrd_vs_f64(0.f);
			for (int b = 0; b < batch; ++b) {
				for (int xy = 0; xy < width*height; ++xy) {
					int pos = b * height * width * channel
						+ xy * channel;
					__vr vr_pin = _ve_vld_vss(8,pin+pos+c);
					vr_sum = _ve_pvfadd_vvv(vr_sum, vr_pin);
				}
			}
			_ve_vst_vss(vr_sum,8,pout+c);
		}

	}else{
		for (int c = 0; c < channel; c+=VLEN) {
			const int64_t vlen = channel-c < VLEN ? channel-c : VLEN;
			_ve_lvl(vlen);
			__vr vr_sum = _ve_vbrdu_vs_f32(0.f);
			for (int b = 0; b < batch; ++b) {
				for (int xy = 0; xy < width*height; ++xy) {
					int pos = b * height * width * channel
						+ xy * channel;
					__vr vr_pin = _ve_vldu_vss(4,pin+pos+c);
					vr_sum = _ve_vfadds_vvv(vr_sum, vr_pin);
				}
			}
			_ve_vstu_vss(vr_sum,4,pout+c);
		}
	}



	return 0;
}


int BiasAddGrad_NCHW(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel)
{
	float* pout = (float*)(output);
	const float* pin = (const float*)(output_backprop);

	const uint64_t alignIn = ((const uint64_t)output_backprop) & 0x07;

	if(((width*height)<64)&&(width*height<channel)){
		for (int c = 0; c < channel; c+=VLEN) {
			const int64_t vlen = channel-c < VLEN ? channel-c : VLEN;
			_ve_lvl(vlen) ;
			__vr vr_sum = _ve_vbrdu_vs_f32(0.f);
			for (int b = 0; b < batch; ++b) {
				int pos = b * channel * height * width + c * height * width;
				for (int i = 0; i < width * height; i++) {
					__vr vr_pin = _ve_vldu_vss(4*width*height,pin+pos+i);
					vr_sum = _ve_vfadds_vvv(vr_sum, vr_pin);
				}
			}
			_ve_vstu_vss(vr_sum,4,pout+c);
		}
	}else{
		if((alignIn==0)&&(width*height%2==0)){
			if(channel%4==0){
				for (int c = 0; c < channel; c+=4) {
					_ve_lvl(256);
					__vr vr_sum1 = _ve_vbrd_vs_f64(0);
					__vr vr_sum2 = _ve_vbrd_vs_f64(0);
					__vr vr_sum3 = _ve_vbrd_vs_f64(0);
					__vr vr_sum4 = _ve_vbrd_vs_f64(0);
					for (int b = 0; b < batch; ++b) {
						int pos1 = b * channel * height * width + (c+0) * height * width;
						int pos2 = b * channel * height * width + (c+1) * height * width;
						int pos3 = b * channel * height * width + (c+2) * height * width;
						int pos4 = b * channel * height * width + (c+3) * height * width;
						for (int i = 0; i < width * height; i+=2*VLEN) {
							const int64_t vlen = (width*height-i < 2*VLEN ? width*height-i : 2*VLEN) >> 1;
							_ve_lvl(vlen) ;
							__vr vr_pin1 = _ve_vld_vss(8,pin+pos1+i);
							__vr vr_pin2 = _ve_vld_vss(8,pin+pos2+i);
							__vr vr_pin3 = _ve_vld_vss(8,pin+pos3+i);
							__vr vr_pin4 = _ve_vld_vss(8,pin+pos4+i);
							vr_sum1 = _ve_pvfadd_vvv(vr_sum1, vr_pin1);
							vr_sum2 = _ve_pvfadd_vvv(vr_sum2, vr_pin2);
							vr_sum3 = _ve_pvfadd_vvv(vr_sum3, vr_pin3);
							vr_sum4 = _ve_pvfadd_vvv(vr_sum4, vr_pin4);
						}
					}
					_ve_lvl(256);
					vr_sum1 = _ve_vfadds_vvv(vr_sum1, _ve_vsll_vvs(vr_sum1,32));
					vr_sum2 = _ve_vfadds_vvv(vr_sum2, _ve_vsll_vvs(vr_sum2,32));
					vr_sum3 = _ve_vfadds_vvv(vr_sum3, _ve_vsll_vvs(vr_sum3,32));
					vr_sum4 = _ve_vfadds_vvv(vr_sum4, _ve_vsll_vvs(vr_sum4,32));
					vr_sum1 = _ve_vfsums_vv(vr_sum1);
					vr_sum2 = _ve_vfsums_vv(vr_sum2);
					vr_sum3 = _ve_vfsums_vv(vr_sum3);
					vr_sum4 = _ve_vfsums_vv(vr_sum4);
					_ve_lvl(1);
					_ve_vstu_vss(vr_sum1,4,pout+c+0);
					_ve_vstu_vss(vr_sum2,4,pout+c+1);
					_ve_vstu_vss(vr_sum3,4,pout+c+2);
					_ve_vstu_vss(vr_sum4,4,pout+c+3);
				}

			}else{
				for (int c = 0; c < channel; ++c) {
					_ve_lvl(256);
					__vr vr_sum = _ve_vbrd_vs_f64(0);
					for (int b = 0; b < batch; ++b) {
						int pos = b * channel * height * width + c * height * width;
						for (int i = 0; i < width * height; i+=2*VLEN) {
							const int64_t vlen = (width*height-i < 2*VLEN ? width*height-i : 2*VLEN) >> 1;
							_ve_lvl(vlen) ;
							__vr vr_pin = _ve_vld_vss(8,pin+pos+i);
							vr_sum = _ve_pvfadd_vvv(vr_sum, vr_pin);
						}
					}
					_ve_lvl(256);
					vr_sum = _ve_vfadds_vvv(vr_sum, _ve_vsll_vvs(vr_sum,32));
					vr_sum = _ve_vfsums_vv(vr_sum);
					_ve_lvl(1);
					_ve_vstu_vss(vr_sum,4,pout+c);
				}
			}
		}else{
			for (int c = 0; c < channel; ++c) {
				_ve_lvl(256);		
				__vr vr_sum = _ve_vbrdu_vs_f32(0.f);
				for (int b = 0; b < batch; ++b) {
					int pos = b * channel * height * width + c * height * width;
					for (int i = 0; i < width * height; i+=VLEN) {
						const int64_t vlen = width*height-i < VLEN ? width*height-i : VLEN;
						_ve_lvl(vlen) ;
						__vr vr_pin = _ve_vldu_vss(4,pin+pos+i);
						vr_sum = _ve_vfadds_vvv(vr_sum, vr_pin);
					}
				}
				_ve_lvl(256);
				vr_sum = _ve_vfsums_vv(vr_sum);
				_ve_lvl(1);
				_ve_vstu_vss(vr_sum,4,pout+c);
			}
		}
	}

	return 0;
}



