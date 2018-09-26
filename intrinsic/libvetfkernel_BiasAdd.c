#include <stdint.h>
#include <float.h>

#include <stdio.h>

#include "libvetfkernel.h"

#include "veintrin.h"
#define VLEN	(256)

#define NCHW_IDX(n,c,h,w,cl,hl,wl) ((((n)*(cl)+(c))*(hl)+(h))*(wl)+(w))


//#define SET_TIMER

#ifdef __ve__
static inline unsigned long long __veperf_get_stm() {
	void *vehva = (void *)0x1000;
	unsigned long long val;
	asm volatile ("lhm.l %0,0(%1)":"=r"(val):"r"(vehva));
	return val;
}
#endif


#if 1
int BiasAdd_NHWC(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
	float* pout = (float*)(out);
	const float* pin = (const float*)(in);
	const float* pbias = (const float*)(bias);
#ifdef SET_TIMER
	unsigned long long start = __veperf_get_stm();
#endif

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
#ifdef SET_TIMER
	unsigned long long end = __veperf_get_stm();

	printf("1 time %lf, nhwc %d %d %d %d\n", (end-start)/(800e6),batch,height,width,channel);
#endif


#if 0
	fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
	return 0;
}

#endif



#if 0
int BiasAdd_NHWC(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
	float* pout = (float*)(out);
	const float* pin = (const float*)(in);
	const float* pbias = (const float*)(bias);

	for (int b = 0; b < batch; ++b) {
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				for (int c = 0; c < channel; ++c) {
					int i
						= b * height * width * channel
						+ y * width * channel
						+ x * channel;
					pout[i + c] = pin[i + c] + pbias[c];
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


#if 1

int BiasAdd_NCHW(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
	float* pout = (float*)(out);
	const float* pin = (const float*)(in);
	const float* pbias = (const float*)(bias);
#ifdef SET_TIMER
	unsigned long long start = __veperf_get_stm();
#endif

	int wh = width*height;




#if 1
	if(wh%2==0){
		for (int xy = 0; xy < width*height; xy+=2*VLEN) {
			const int64_t vlen = width*height-xy < VLEN ? width*height-xy : VLEN;
			_ve_lvl(vlen) ;
			for (int b = 0; b < batch; ++b) {
				for (int c = 0; c < channel; ++c) {
					int i = b * height * width * channel
						+ c * height * width;
					__vr vr_pin = _ve_vld_vss(8,pin+i+xy);
					__vr vr_sum = _ve_pvfadd_vsv(pbias[c], vr_pin);
					_ve_vst_vss(vr_sum,8,pout+i+xy);

				}
			}
		}
	}else{
		for (int xy = 0; xy < width*height; xy+=VLEN) {
			const int64_t vlen = width*height-xy < VLEN ? width*height-xy : VLEN;
			_ve_lvl(vlen) ;
			for (int b = 0; b < batch; ++b) {	
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
#endif



#if 0
	if((width*height)%2==0){
		for (int b = 0; b < batch; ++b) {
			for (int c = 0; c < channel; ++c) {
				int i = b * height * width * channel
					+ c * height * width;
				for (int xy = 0; xy < width*height; xy+=2*VLEN) {
					const int64_t vlen = width*height-xy < VLEN ? width*height-xy : VLEN;
					_ve_lvl(vlen) ;
					__vr vr_pin = _ve_vld_vss(8,pin+i+xy);
					__vr vr_sum = _ve_pvfadd_vsv(pbias[c], vr_pin);
					_ve_vst_vss(vr_sum,8,pout+i+xy);
				}
			}
		}
	}else{
		for (int b = 0; b < batch; ++b) {
			for (int c = 0; c < channel; ++c) {
				int i = b * height * width * channel
					+ c * height * width;
				for (int xy = 0; xy < width*height; xy+=VLEN) {
					const int64_t vlen = width*height-xy < VLEN ? width*height-xy : VLEN;
					_ve_lvl(vlen) ;
					__vr vr_pin = _ve_vldu_vss(4,pin+i+xy);
					__vr vr_sum = _ve_vfadds_vsv(pbias[c], vr_pin);
					_ve_vstu_vss(vr_sum,4,pout+i+xy);
				}
			}
		}

	}
#endif


#if 0
	for (int b = 0; b < batch; ++b) {
		for (int c = 0; c < channel; ++c) {
			int i = b * height * width * channel
				+ c * height * width;
			for (int xy = 0; xy < width*height; xy+=VLEN) {
				const int64_t vlen = width*height-xy < VLEN ? width*height-xy : VLEN;
				_ve_lvl(vlen) ;
				__vr vr_pin = _ve_vldu_vss(4,pin+i+xy);
				__vr vr_sum = _ve_vfadds_vsv(pbias[c], vr_pin);
				_ve_vstu_vss(vr_sum,4,pout+i+xy);
			}
		}
	}
#endif



#ifdef SET_TIMER
	unsigned long long end = __veperf_get_stm();

	printf("2 time %lf, nchw %d %d %d %d\n", (end-start)/(800e6),batch,channel,height,width);
#endif



#if 0
	fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
	return 0;
}
#endif



#if 0

int BiasAdd_NCHW(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
	float* pout = (float*)(out);
	const float* pin = (const float*)(in);
	const float* pbias = (const float*)(bias);

	for (int b = 0; b < batch; ++b) {
		for (int c = 0; c < channel; ++c) {
			for (int xy = 0; xy < width*height; ++xy) {
				int i
					= b * height * width * channel
					+ c * height * width ;
				pout[i + xy] = pin[i + xy] + pbias[c];
			}
		}
	}

#if 0
	fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
	return 0;
}
#endif




#if 0

vednnError_t vednnMaxPoolingBackward_regular(
		const vednnTensorParam_t 		*pParamGradOut,
		const void 				*pDataGradOut,
		const vednnTensorParam_t 		*pParamOut,
		const void 				*pDataOut,
		const vednnTensorParam_t 		*pParamIn,
		const void 				*pDataIn,
		const vednnTensorParam_t 		*pParamGradIn,
		void 				*pDataGradIn,
		const vednnPoolingParam_t		*pParamPool
		)
{
	const int64_t batch      = pParamIn->batch;
	const int64_t inChannel  = pParamIn->channel;
	const int64_t inWidth    = pParamIn->width;
	const int64_t inHeight   = pParamIn->height;
	const int64_t outChannel = pParamOut->channel;
	const int64_t outWidth   = pParamOut->width;
	const int64_t outHeight  = pParamOut->height;

	const int64_t windowWidth  = pParamPool->windowWidth;
	const int64_t windowHeight = pParamPool->windowHeight;
	const int64_t strideWidth  = pParamPool->strideWidth;;
	const int64_t strideHeight = pParamPool->strideHeight;
	//  const int64_t padWidth     = pParamPool->padWidth;		// must be 0
	//  const int64_t padHeight    = pParamPool->padHeight;		// must be 0

	const float * restrict pGOut   = pDataGradOut;
	const float * restrict pOut    = pDataOut;
	const float * restrict pIn     = pDataIn;
	float * restrict const pGIn    = pDataGradIn ;

	{
		for(int64_t n=0; n<batch; n++) {
			for(int64_t c=0; c<outChannel; c++) {
				for(int64_t h=0; h<outHeight; h++) {
					for(int64_t w=0; w<outWidth; w+=VLEN) {
						const int64_t vlen = outWidth-w < VLEN ? outWidth-w : VLEN ;

						const int64_t outIndex  = NCHW_IDX(n,c,h,w,outChannel,outHeight,outWidth) ;

						_ve_lvl(vlen) ;

						__vr vrout  = _ve_vldu_vss(4, pOut+outIndex) ;
						__vr vrgout = _ve_vldu_vss(4, pGOut+outIndex) ;

						__vm256 vm_not_found = _ve_vfmkat_m() ;

						for(int64_t ph=0; ph<windowHeight; ph++) {
							const int64_t y = h*strideHeight + ph ;

							for(int64_t pw=0; pw<windowWidth; pw++) {
								const int64_t x = w*strideWidth + pw ;
								const int64_t inIndex = NCHW_IDX(n,c,y,x,inChannel,inHeight,inWidth) ;

								__vr vrin = _ve_vldu_vss(4*strideWidth,pIn+inIndex) ;

								__vm256 vm_equal = _ve_vfmks_mcv(VECC_EQ, _ve_vfcmps_vvv(vrout,vrin)) ;
								__vm256 vm_and   = _ve_andm_mmm(vm_equal, vm_not_found) ;
								vm_not_found = _ve_nndm_mmm(vm_equal, vm_not_found) ;

								__vr vrgin = _ve_vmrg_vvvm(_ve_vbrdu_vs_f32(0.f), vrgout, vm_and) ;

								_ve_vstu_vss(vrgin, 4*strideWidth, pGIn+inIndex) ;

							} // windowWidth
						} // windowHeight
					} // outWidth
				} // outHeight
			} // channel
		} // batch
	}

	return VEDNN_SUCCESS ;
}

#endif

