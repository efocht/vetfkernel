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



	if(wh%2==0){
		for (int b = 0; b < batch; ++b) {
			for (int xy = 0; xy < width*height; xy+=2*VLEN) {
				const int64_t vlen = width*height-xy < VLEN ? width*height-xy : VLEN;
				_ve_lvl(vlen) ;
				for (int c = 0; c < channel; ++c) {
					int i = b * height * width * channel
						+ c * height * width;
					__vr vr_pin = _ve_vld_vss(8,pin+i+xy);
					__vr vr_sum = _ve_pvfadd_vsv(_ve_pack_f32a(pbias+c), vr_pin);
					_ve_vst_vss(vr_sum,8,pout+i+xy);

				}
			}
		}
	}else{
		for (int b = 0; b < batch; ++b) {
			for (int xy = 0; xy < width*height; xy+=VLEN) {
				const int64_t vlen = width*height-xy < VLEN ? width*height-xy : VLEN;
				_ve_lvl(vlen) ;
				for (int c = 0; c < channel; ++c) {
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





