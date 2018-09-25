#include <cstdint>
#include "libvetfkernel.h"



int main(){

	int batch=128;
	int channel=64;
	int height=24;
	int width=24;
	float *in = new float[batch*channel*width*height];
	float *bias = = new float[batch*channel*width*height];
	float *out = new float[batch*channel*width*height];


	for(int i=0;i<100000;i++)
		BiasAdd_NCHW((uint64_t)out,(uint64_t)in,(uint64_t)bias,batch,width,height,channel);



}
