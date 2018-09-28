#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "libvetfkernel.h"

#include "veintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))



int BiasAdd_NHWC(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
	float* pout = (float*)(out);
	const float* pin = (const float*)(in);
	const float* pbias = (const float*)(bias);

	if((channel<64)&&(width*height>channel)){
		for (int xy = 0; xy < width*height; xy+=VLEN) {
			const int64_t vlen = width*height-xy < VLEN ? width*height-xy : VLEN;
			_ve_lvl(vlen) ;
			for (int b = 0; b < batch; ++b) {
				int i = b * height * width * channel
					+ xy * channel;
				for (int c = 0; c < channel; ++c) {
					__vr vr_pin = _ve_vldu_vss(4*channel,pin+i+c);
					__vr vr_sum = _ve_vfadds_vsv(pbias[c], vr_pin);
					_ve_vstu_vss(vr_sum,4*channel,pout+i+c);
				}
			}
		}
	}else{
		for (int c = 0; c < channel; c+=VLEN) {
			const int64_t vlen = channel-c < VLEN ? channel-c : VLEN;
			_ve_lvl(vlen) ;
			__vr vr_pbias = _ve_vldu_vss(4,pbias+c);
			for (int b = 0; b < batch; ++b) {
				for (int xy = 0; xy < width*height; xy++) {
					int i = b * height * width * channel
						+ xy * channel;
					__vr vr_pin = _ve_vldu_vss(4,pin+i+c);
					__vr vr_sum = _ve_vfadds_vvv(vr_pbias, vr_pin);
					_ve_vstu_vss(vr_sum,4,pout+i+c);
				}
			}
		}

	}

#if 0
	fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
	return 0;
}






int BiasAdd_NCHW(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
	float* pout = (float*)(out);
	const float* pin = (const float*)(in);
	const float* pbias = (const float*)(bias);

	int wh = width*height;
	const uint64_t alignIn = ((const uint64_t)in) & 0x07;
	const uint64_t alignOut = ((const uint64_t)out) & 0x07;


	if((alignIn==0)&&(alignOut==0)&&(wh%2==0)){
		if(channel%5==0){
			for (int b = 0; b < batch; ++b) {
				for (int c = 0; c < channel; c+=5) {
					for (int xy = 0; xy < width*height; xy+=2*VLEN) {
						const int64_t vlen = (width*height-xy < 2*VLEN ? width*height-xy : 2*VLEN) >> 1;
						_ve_lvl(vlen) ;
						int i1 = b * height * width * channel + c * height * width;
						int i2 = b * height * width * channel + (c+1) * height * width;
						int i3 = b * height * width * channel + (c+2) * height * width;
						int i4 = b * height * width * channel + (c+3) * height * width;
						int i5 = b * height * width * channel + (c+4) * height * width;
						__vr vr_pin1 = _ve_vld_vss(8,pin+i1+xy);
						__vr vr_pin2 = _ve_vld_vss(8,pin+i2+xy);
						__vr vr_pin3 = _ve_vld_vss(8,pin+i3+xy);
						__vr vr_pin4 = _ve_vld_vss(8,pin+i4+xy);
						__vr vr_pin5 = _ve_vld_vss(8,pin+i5+xy);

						__vr vr_sum1 = _ve_pvfadd_vsv(_ve_pack_f32a(pbias+c), vr_pin1);
						__vr vr_sum2 = _ve_pvfadd_vsv(_ve_pack_f32a(pbias+c+1), vr_pin2);
						__vr vr_sum3 = _ve_pvfadd_vsv(_ve_pack_f32a(pbias+c+2), vr_pin3);
						__vr vr_sum4 = _ve_pvfadd_vsv(_ve_pack_f32a(pbias+c+3), vr_pin4);
						__vr vr_sum5 = _ve_pvfadd_vsv(_ve_pack_f32a(pbias+c+4), vr_pin5);

						_ve_vst_vss(vr_sum1,8,pout+i1+xy);
						_ve_vst_vss(vr_sum2,8,pout+i2+xy);
						_ve_vst_vss(vr_sum3,8,pout+i3+xy);
						_ve_vst_vss(vr_sum4,8,pout+i4+xy);
						_ve_vst_vss(vr_sum5,8,pout+i5+xy);
					}
				}
			}

		}
		else if(channel%4==0){
			for (int b = 0; b < batch; ++b) {
				for (int c = 0; c < channel; c+=4) {
					for (int xy=0; xy < width*height; xy+=2*VLEN) {
						const int64_t vlen = (width*height-xy < 2*VLEN ? width*height-xy : 2*VLEN) >> 1;
						_ve_lvl(vlen);
						int i1 = b * height * width * channel + c * height * width;
						int i2 = b * height * width * channel + (c+1) * height * width;
						int i3 = b * height * width * channel + (c+2) * height * width;
						int i4 = b * height * width * channel + (c+3) * height * width;
						__vr vr_pin1 = _ve_vld_vss(8,pin+i1+xy);
						__vr vr_pin2 = _ve_vld_vss(8,pin+i2+xy);
						__vr vr_pin3 = _ve_vld_vss(8,pin+i3+xy);
						__vr vr_pin4 = _ve_vld_vss(8,pin+i4+xy);
						__vr vr_sum1 = _ve_pvfadd_vsv(_ve_pack_f32a(pbias+c), vr_pin1);
						__vr vr_sum2 = _ve_pvfadd_vsv(_ve_pack_f32a(pbias+c+1), vr_pin2);
						__vr vr_sum3 = _ve_pvfadd_vsv(_ve_pack_f32a(pbias+c+2), vr_pin3);
						__vr vr_sum4 = _ve_pvfadd_vsv(_ve_pack_f32a(pbias+c+3), vr_pin4);
						_ve_vst_vss(vr_sum1,8,pout+i1+xy);
						_ve_vst_vss(vr_sum2,8,pout+i2+xy);
						_ve_vst_vss(vr_sum3,8,pout+i3+xy);
						_ve_vst_vss(vr_sum4,8,pout+i4+xy);
					}
				}
			}
		}else{

			for (int b = 0; b < batch; ++b) {
				for (int c = 0; c < channel; ++c) {
					_ve_lvl(256) ;
					unsigned long p_bias= _ve_pack_f32a(pbias+c);
					for (int xy = 0; xy < width*height; xy+=2*VLEN) {
						const int64_t vlen = (width*height-xy < 2*VLEN ? width*height-xy : 2*VLEN) >> 1;
						_ve_lvl(vlen) ;
						int i = b * height * width * channel
							+ c * height * width;
						__vr vr_pin = _ve_vld_vss(8,pin+i+xy);
						__vr vr_sum = _ve_pvfadd_vsv(p_bias, vr_pin);
						_ve_vst_vss(vr_sum,8,pout+i+xy);
					}
				}
			}
		}
	}else{
		for (int b = 0; b < batch; ++b) {
			for (int c = 0; c < channel; ++c) {
				for (int xy = 0; xy < width*height; xy+=VLEN) {
					const int64_t vlen = width*height-xy < VLEN ? width*height-xy : VLEN;
					_ve_lvl(vlen) ;
					int i = b * height * width * channel
						+ c * height * width;
					__vr vr_pin = _ve_vldu_vss(4,pin+i+xy);
					__vr vr_sum = _ve_vfadds_vsv(pbias[c], vr_pin);
					_ve_vstu_vss(vr_sum,4,pout+i+xy);
				}
			}
		}
	}









#if 0
	fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
	return 0;
}





