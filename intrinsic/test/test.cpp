#include <cstdint>
#include <stdio.h>
#include "libvetfkernel.h"


#define SPF_USE_FTRACE
#if defined(_SX)||defined(__ve) && defined(SPF_USE_FTRACE)
#include <ftrace.h>
#define FTRACE_A(name) ftrace_region_begin(name)
#define FTRACE_B(name) ftrace_region_end(name)
#else
#define FTRACE_A(name)
#define FTRACE_B(name)
#endif

#define SET_TIMER

#ifdef __ve__
static inline unsigned long long __veperf_get_stm() {
        void *vehva = (void *)0x1000;
        unsigned long long val;
        asm volatile ("lhm.l %0,0(%1)":"=r"(val):"r"(vehva));
        return val;
}
#endif


int main(){

	int batch=128;
	int channel=32;
	int height=64;
	int width=64;
	float *in = new float[batch*channel*width*height];
	float *bias = new float[batch*channel*width*height];
	float *out = new float[batch*channel*width*height];

#ifdef SET_TIMER
        unsigned long long start = __veperf_get_stm();
#endif
FTRACE_A("BiasAdd");

	for(int i=0;i<10000;i++)
		BiasAdd_NCHW((uint64_t)out,(uint64_t)in,(uint64_t)bias,batch,width,height,channel);

FTRACE_B("BiasAdd");


#ifdef SET_TIMER
        unsigned long long end = __veperf_get_stm();

        printf("test time %lf, nchw %d %d %d %d\n", (end-start)/(800e6),batch,channel,height,width);
#endif



}
