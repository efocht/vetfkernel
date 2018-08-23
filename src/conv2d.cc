#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>

#include "kernel.h"

REGISTER_KERNEL("Conv2D", "conv2d");

extern "C" {
    int conv2d(const void* arg, size_t len);
}

struct TensorParam {
    int w, h, c, n;
};

struct ConvParam {
    uint64_t in;
    uint64_t filter;
    uint64_t out;
    TensorParam in_param;
    TensorParam filter_param;
    TensorParam out_param;

    int row_stride;
    int col_stride;
    int row_dilation;
    int col_dilation;
    int row_padding;
    int col_padding;

    int data_format;
    int data_type;
};

int conv2d(const void* arg, size_t len)
{
    fprintf(stderr, "conv2d\n");
    assert(len == sizeof(ConvParam));
    const ConvParam& p = *(ConvParam*)arg;

    fprintf(stderr, "conv2d: data_format=%d data_type=%d\n", p.data_format, p.data_type);
    fprintf(stderr, "conv2d: in=%dx%d out=%dx%d filter=%dx%d\n", 
            p.in_param.w, p.in_param.h,
            p.out_param.w, p.out_param.h,
            p.filter_param.w, p.filter_param.h);
    fprintf(stderr, "conv2d: stride=%dx%d dilation=%dx%d padding=%dx%d\n", 
            p.row_stride, p.col_stride,
            p.row_dilation, p.col_dilation,
            p.row_padding, p.col_padding);

    int W = p.out_param.w;
    int H = p.out_param.h;
    int KW = p.filter_param.w;
    int KH = p.filter_param.h;

    fprintf(stderr, "conv2d: W=%d H=%d P=%d Q=%d\n", W, H, KW, KH);

    if (p.data_type == 1) { // float
        const float* in = (float*)p.in;
        const float* filter = (float*)p.filter;
        float* out = (float*)p.out;

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float s = 0.0f;
                for (int ky = 0; ky < KH; ++ky) { // height
                    for (int kx = 0; kx < KW; ++kx) { // width
                        int x0 = x + kx;
                        int y0 = y + ky;
                        float inv = 0.0f;
                        if (x0 <  W)
                            inv = in[y0 * W + x0];
                        s += inv * filter[kx * KH + ky];
                        fprintf(stderr, "(%d,%d) * (%d,%d) = %f * %f\n", x0, y0, kx, ky, inv, filter[kx*KH+ky]);
                    }
                }
                out[y * W + x] = s;

                fprintf(stderr, "[%d,%d]=%f\n", x, y, s);
            }
        }

    }

    printf("libvetfkernel: conv2d: out=%p %dx%d\n", (void*)p.out, p.out_param.w, p.out_param.h);
#if 0
    const char* tmp = "hello VE";
    memcpy((void*)p.out, tmp, 9);
#endif

    return 0;
}
