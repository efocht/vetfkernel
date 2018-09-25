#ifndef __LIBVETFKERNEL__
#define __LIBVETFKERNEL__


#ifdef __cplusplus
extern "C" {
#endif

int BiasAdd_NHWC(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel);
int BiasAdd_NCHW(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel);

#ifdef __cplusplus
}
#endif


#endif
