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


#if 1
	const uint64_t alignOut = ((const uint64_t)output) & 0x07;
	const uint64_t alignIn = ((const uint64_t)output_backprop) & 0x07;

	if((alignIn==0)&&(alignOut==0)&&(channel%2==0)){
		for (int c = 0; c < channel; c+=2*VLEN) {
			const int64_t vlen = (channel-c < 2*VLEN ? channel : 2*VLEN) >> 1;
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
			const int64_t vlen = channel-c < VLEN ? channel : VLEN;
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

#elif 0
	memset(pout, 0, sizeof(float) * channel);

	for (int b = 0; b < batch; ++b) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				for (int c = 0; c < channel; ++c) {
					int i
						= b * height * width * channel
						+ y * width * channel
						+ x * channel;
					pout[c] += pin[i + c];
				}
			}
		}
	}
#endif


#if 0
	fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
	return 0;
}


int BiasAddGrad_NCHW(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel)
{
	float* pout = (float*)(output);
	const float* pin = (const float*)(output_backprop);

#if 1
	const uint64_t alignIn = ((const uint64_t)output_backprop) & 0x07;

	if((alignIn==0)&&(width*height%2==0)){
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
#elif 1


	for (int c = 0; c < channel; ++c) {
		pout[c] = 0;
		for (int b = 0; b < batch; ++b) {
			for (int i = 0; i < width * height; ++i) {
				pout[c] += pin[b * channel * height * width + c * height * width + i];
			}
		}
	}

#elif 0

	memset(pout, 0, sizeof(float) * channel);

	for (int b = 0; b < batch; ++b) {
		for (int c = 0; c < channel; ++c) {
			for (int i = 0; i < width * height; ++i) {
				pout[c] += pin[b * channel * height * width + c * height * width + i];
			}
		}
	}


#endif


#if 0
	fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
	return 0;
}


#if 0
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
		for (int b = 0; b < batch; ++b) {
			for (int c = 0; c < channel; ++c) {
				for (int xy = 0; xy < width*height; xy+=2*VLEN) {
					const int64_t vlen = width*height-xy < VLEN ? width*height-xy : VLEN;
					_ve_lvl(vlen) ;
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

#endif



