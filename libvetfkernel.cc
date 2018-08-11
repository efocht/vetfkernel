#include <cstdio>
#include <cstdint>
#include <cstring>

int conv2d(const void* arg);

extern "C" {
    //int conv2d(void* arg);
    int compute(int kernelId, const void* arg, size_t len);
}

#if 0
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

    int data_type;
    int data_format;
};

int conv2d(const void* arg)
{
    fprintf(stderr, "conv2d\n");
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
    int P = p.filter_param.h;
    int Q = p.filter_param.w;

    fprintf(stderr, "conv2d: W=%d H=%d P=%d Q=%d\n", W, H, P, Q);

    if (p.data_type == 0) { // float
        const float* in = (float*)p.in;
        const float* filter = (float*)p.filter;
        float* out = (float*)p.out;

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float s = 0.0f;
                for (int p = 0; p < P; ++p) { // height
                    for (int q = 0; q < Q; ++q) { // width
                        int x0 = x + P - p - 1;
                        int y0 = y + Q - q - 1;
                        float inv = 0.0f;
                        if (x0 <  W)
                            inv = in[(y + Q - q - 1) * W + (x + P - p - 1)];
                        s += inv * filter[p * Q + q];
                        fprintf(stderr, "(%d,%d) * (%d,%d) = %f * %f\n", (x+q), (y+p), q, p, in[(y+p)*W+(x+q)], filter[p*Q+q]);
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


    return 1976;
}
#endif

int compute(int kernelId, const void* arg, size_t len)
{
    fprintf(stderr, "libvetfkernel::compute: kernelId=%d\n", kernelId);
    conv2d(arg);
    return 0;
}
