#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "kernel.h"
#include "types.h"
#include "log.h"

#include <omp.h>

#include "vednn.h"


#define LIBVETF_INTRINSIC

#ifdef LIBVETF_INTRINSIC
#include "libvetfkernel.h"
#endif


//#define SET_TIMER


#ifdef SET_TIMER
#ifdef __ve__
static inline unsigned long long __veperf_get_stm() {
        void *vehva = (void *)0x1000;
        unsigned long long val;
        asm volatile ("lhm.l %0,0(%1)":"=r"(val):"r"(vehva));
        return val;
}
#endif
#endif




#define ADD_
#include <cblas_f77.h>
#undef ADD_

REGISTER_KERNEL("Fill", "op_fill");
REGISTER_KERNEL("AddN", "op_AddN");
REGISTER_KERNEL("BiasAdd", "op_BiasAdd");
REGISTER_KERNEL("BiasAddGrad", "op_BiasAddGrad");
REGISTER_KERNEL("Relu", "op_Relu");
REGISTER_KERNEL("ReluGrad", "op_ReluGrad");
REGISTER_KERNEL("Snapshot", "op_Snapshot")
REGISTER_KERNEL("Sum", "op_Sum");
REGISTER_KERNEL("Transpose", "op_Transpose");
REGISTER_KERNEL("MatMul", "op_MatMul");
REGISTER_KERNEL("Softmax", "op_Softmax");

// Unary
REGISTER_KERNEL("Neg", "op_Neg");
REGISTER_KERNEL("Sqrt", "op_Sqrt");
REGISTER_KERNEL("Rsqrt", "op_Rsqrt");
REGISTER_KERNEL("Square", "op_Square");
REGISTER_KERNEL("Floor", "op_Floor");

#define CHECK_ARG_LEN(l0, l1) \
  if ((l0) != (l1)) { \
      fprintf(stderr, "%s: illegal argument lenght: %ld expected but %ld\n", (l1), (l0)); \
      return 1; \
  }

extern "C" {
  int op_fill(const void* arg, size_t len);
  int op_AddN(const void* arg, size_t len);
  int op_BiasAdd(const void* arg, size_t len);
  int op_BiasAddGrad(const void* arg, size_t len);
  int op_Relu(const void* arg, size_t len);
  int op_ReluGrad(const void* arg, size_t len);
  int op_Snapshot(const void* arg, size_t len);
  int op_Sum(const void* arg, size_t len);
  int op_Neg(const void* arg, size_t len);
  int op_Floor(const void* arg, size_t len);
  int op_Transpose(const void* arg, size_t len);
  int op_MatMul(const void* arg, size_t len);
  int op_Softmax(const void* arg, size_t len);
  int op_Sqrt(const void* arg, size_t len);
  int op_Rsqrt(const void* arg, size_t len);
  int op_Square(const void* arg, size_t len);
}

namespace {
template <typename T>
  int fill(const void* args, size_t len)
  {
    LOG(2) << __PRETTY_FUNCTION__;
    struct Args {
      int data_type;
      uint64_t in;
      uint64_t out;
      size_t num_elems;
    } const* p;

    if (len != sizeof(*p)) {
      fprintf(stderr, "%s: illegal argument lenght: %ld expected but %ld\n",
              sizeof(*p), len);
      return 1;
    }

    p = reinterpret_cast<const Args*>(args);
#if 0
    LOG(2, "%s: data_type=%d out=%016lx num_elems=%lu\n",
          __FUNCTION__, p->data_type, p->out, p->num_elems);
#endif

    const T* in = (T*)p->in;
    T* out = (T*)p->out;

    LOG(2) << __FUNCTION__ << ": value=" << *in;

    for (size_t i = 0; i < p->num_elems; ++i)
      out[i] = *in;

    LOG(2) << __PRETTY_FUNCTION__ << ": done";
    return 0;
  }
}

#define DATA_TYPE(p) *(int*)(p)

int op_fill(const void* args, size_t len)
{
  LOG(1) << __FUNCTION__;
  int dtype = DATA_TYPE(args);
  LOG(2) << __FUNCTION__ << ": dtype=" << dtype;

  if (dtype == DT_FLOAT) {
    return fill<float>(args, len);
  } else {
    return 1;
  }
}

namespace {
template <typename T>
void AddNOp(T* out, T** in, size_t num_elems, size_t num_inputs)
{
#if 0
  for (size_t j = 0; j < num_inputs; ++j) {
    std::cerr << __FUNCTION__ << ": in[" << j << "]=" << in[j] << std::endl;
  }
#endif

#if 1
  memset(out, 0, sizeof(T) * num_elems);
  for (size_t j = 0; j < num_inputs; ++j) {
    for (size_t i = 0; i < num_elems; ++i) {
      out[i] += in[j][i];
    }
  }

#else
  for (size_t i = 0; i < num_elems; ++i) {
    T s = 0;
    for (size_t j = 0; j < num_inputs; ++j) {
#if 0
      std::cerr << __FUNCTION__ << " in[" << j << "][" << i << "]=" << in[j][i] << std::endl;
#endif
      s += in[j][i];
    }
    out[i] = s;
  }
#endif
}
};

int op_AddN(const void* args, size_t len)
{
  LOG(1) << __FUNCTION__;
#define MAX_INPUTS 16
  struct Args {
    int output_type;
    uint64_t out;
    size_t num_elems;
    size_t num_inputs;
    uint64_t in[MAX_INPUTS];
  } const* p;

  if (len != sizeof(*p)) {
      fprintf(stderr, "%s: illegal argument lenght: %ld expected but %ld\n", sizeof(*p), len);
      return 1;
  }

  p = reinterpret_cast<const Args*>(args);

  LOG(2) << __FUNCTION__ << "num_elems=" << p->num_elems << " num_inputs=" << p->num_inputs;

  if (p->output_type == DT_FLOAT) {
    AddNOp<float>((float*)p-> out, (float**)p->in, p->num_elems, p->num_inputs);
  } else {
    return 1;
  }

  return 0;
}
#ifndef LIBVETF_INTRINSIC
namespace {

template<typename T>
int BiasAdd_NHWC(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
  T* pout = reinterpret_cast<T*>(out);
  const T* pin = reinterpret_cast<const T*>(in);
  const T* pbias = reinterpret_cast<const T*>(bias);

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

template<typename T>
int BiasAdd_NCHW(uint64_t out, uint64_t in, uint64_t bias, int batch, int width, int height, int channel)
{
  T* pout = reinterpret_cast<T*>(out);
  const T* pin = reinterpret_cast<const T*>(in);
  const T* pbias = reinterpret_cast<const T*>(bias);

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

};
#endif
int op_BiasAdd(const void* args, size_t len)
{
  LOG(1) << __FUNCTION__;
  struct Args {
    int dtype;
    int data_format;
    uint64_t in;
    uint64_t bias;
    uint64_t out;
    int batch;
    int width;
    int height;
    int channel;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));

  p = reinterpret_cast<const Args*>(args);

#if 0
  fprintf(stderr, "%s dtype=%d data_format=%d batch=%d width=%d height=%d channel=%d\n", 
          __FUNCTION__, p->dtype, p->data_format, p->batch, p->width, p->height, p->channel);
#endif
  int r=0;
#ifdef SET_TIMER
  unsigned long long start = __veperf_get_stm();
#endif

  if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NHWC) {
#ifndef LIBVETF_INTRINSIC
    r = BiasAdd_NHWC<float>(p->out, p->in, p->bias, p->batch, p->width, p->height, p->channel);
#else
    r = BiasAdd_NHWC(p->out, p->in, p->bias, p->batch, p->width, p->height, p->channel);
#endif
#ifdef SET_TIMER
  unsigned long long end = __veperf_get_stm();
  printf("add hwc, nchw %d %d %d %d:%lfms\n",p->batch,p->channel,p->width, p->height,(end-start)/(800e3));
#endif
  } else if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NCHW) {
#ifndef LIBVETF_INTRINSIC
    r = BiasAdd_NCHW<float>(p->out, p->in, p->bias, p->batch, p->width, p->height, p->channel);
#else
    r = BiasAdd_NCHW(p->out, p->in, p->bias, p->batch, p->width, p->height, p->channel);
#endif
#ifdef SET_TIMER
  unsigned long long end = __veperf_get_stm();
  printf("add chw, nchw %d %d %d %d:%lfms\n",p->batch,p->channel,p->width, p->height,(end-start)/(800e3));
#endif

  }



  return r;


#if 0
  fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
  return 1;
}

#ifndef LIBVETF_INTRINSIC
namespace {

template<typename T>
int BiasAddGrad_NHWC(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel)
{
  T* pout = reinterpret_cast<T*>(output);
  const T* pin = reinterpret_cast<const T*>(output_backprop);

  memset(pout, 0, sizeof(T) * channel);

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

#if 0
  fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
  return 0;
}


template<typename T>
int BiasAddGrad_NCHW(uint64_t output, uint64_t output_backprop, int batch, int width, int height, int channel)
{
  T* pout = reinterpret_cast<T*>(output);
  const T* pin = reinterpret_cast<const T*>(output_backprop);

  memset(pout, 0, sizeof(T) * channel);

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channel; ++c) {
      for (int i = 0; i < width * height; ++i) {
        pout[c] += pin[b * channel * height * width + c * height * width + i];
      }
    }
  }

#if 0
  fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
  return 0;
}

};
#endif
int op_BiasAddGrad(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__;
  struct Args{
    int dtype;
    int data_format;
    uint64_t output_backprop;
    uint64_t output;
    int batch;
    int width;
    int height;
    int channel;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));

  p = reinterpret_cast<const Args*>(args);

#if 0
  fprintf(stderr, "%s dtype=%d data_format=%d batch=%d width=%d height=%d channel=%d\n", 
          __FUNCTION__, p->dtype, p->data_format, p->batch, p->width, p->height, p->channel);
#endif

#ifndef LIBVETF_INTRINSIC
  if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NHWC) {
    return BiasAddGrad_NHWC<float>(p->output, p->output_backprop, p->batch, p->width, p->height, p->channel);
  } else if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NCHW) {
    return BiasAddGrad_NCHW<float>(p->output, p->output_backprop, p->batch, p->width, p->height, p->channel);
  }
#else
int r;
  if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NHWC) {
#ifdef SET_TIMER
  unsigned long long start = __veperf_get_stm();
#endif
  r = BiasAddGrad_NHWC(p->output, p->output_backprop, p->batch, p->width, p->height, p->channel);
#ifdef SET_TIMER
  unsigned long long end = __veperf_get_stm();
  printf("grad hwc, nchw %d %d %d %d:%lfms\n",p->batch,p->channel,p->width, p->height,(end-start)/(800e3));    
#endif
  } else if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NCHW) {
#ifdef SET_TIMER
  unsigned long long start = __veperf_get_stm();
#endif
    r =  BiasAddGrad_NCHW(p->output, p->output_backprop, p->batch, p->width, p->height, p->channel);
#ifdef SET_TIMER
  unsigned long long end = __veperf_get_stm();
  printf("grad chw, nchw %d %d %d %d:%lfms\n",p->batch,p->channel,p->width, p->height,(end-start)/(800e3));
#endif
  }
  return r;
#endif

  LOG(2) << __FUNCTION__ << " done";
  return 1;
}

//
// Relu and ReluGrad
//

int op_Relu(const void* args, size_t len)
{
  struct Args {
    int dtype;
    uint64_t in;
    uint64_t out;
    uint64_t num_elems;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  if (p->dtype == DT_FLOAT) {
    return vednnActivationForward(
      VEDNN_ACTIVATION_RELU,
      (void*)(p->in), (void*)(p->out),
      p->num_elems 
    ) ;
  }
  return 1;
}

int op_ReluGrad(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__;
  struct Args {
    int dtype;
    uint64_t g;
    uint64_t a;
    uint64_t output;
    uint64_t num_elems;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  if (p->dtype == DT_FLOAT) {
    return vednnActivationBackward(
      VEDNN_ACTIVATION_RELU,
      (void*)(p->g), (void*)(p->a), (void*)(p->output),
      p->num_elems 
    ) ;

  }
  return 1;
}


int op_Snapshot(const void* arg, size_t len)
{
  LOG(2) << __FUNCTION__;
  struct Arg {
    uint64_t dst;
    uint64_t src;
    size_t size;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Arg));
  p = reinterpret_cast<const Arg*>(arg);

  LOG(3) << __FUNCTION__ 
    << " dst=" << p->dst << " src=" << p->src << " size=" << p->size; 

  memcpy(reinterpret_cast<void*>(p->dst),
         reinterpret_cast<const void*>(p->src),
         p->size);
  LOG(2) << __FUNCTION__ << ": done";

  return 0;
}

//
// Neg
//
#ifndef LIBVETF_INTRINSIC
namespace {
template<typename Tin, typename Tout>
  void neg(uint64_t out, uint64_t in, size_t nelems)
  {
    Tout* po = reinterpret_cast<Tout*>(out);
    const Tin* pi = reinterpret_cast<Tin*>(in);

    for (int64_t i = 0; i < nelems; ++i) {
      po[i] = - pi[i];
    }
  }
}
#endif
int op_Neg(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__ << " begin";

  struct _Tensor {
    int dtype;
    int data_format;
    uint64_t addr;
    int32_t dims;
    int64_t nelems;
    int64_t dim_size[8];
  };

  struct Args {
    _Tensor in;
    _Tensor out;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  if (p->in.dtype == DT_FLOAT || p->out.dtype == DT_FLOAT) {
#ifdef SET_TIMER
  unsigned long long start = __veperf_get_stm();
#endif
#ifndef LIBVETF_INTRINSIC
    neg<float, float>(p->out.addr, p->in.addr, p->in.nelems);
#else
    neg(p->out.addr, p->in.addr, p->in.nelems);
#endif
#ifdef SET_TIMER
  unsigned long long end = __veperf_get_stm();
  printf("neg, len %d:%lfms\n",p->in.nelems,(end-start)/(800e3));
#endif


  } else {
    return 1;
  }

  LOG(2) << __FUNCTION__ << " end";
  return 0;
}

namespace {
template <typename T>
int reduction_d2a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi = reinterpret_cast<const T*>(in);

  for (size_t i = 0; i < dim0; ++i) {
    T s = T(0);
    for (size_t j = 0; j < dim1; ++j) {
      s += pi[i * dim1 + j];
    }
    po[i] = s;
  }

  return 0;
}
}

int op_Sum(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__ << " begin";

  struct Args {
    int dtype;
    int ndims;
    uint64_t in;
    uint64_t out;
    int64_t dim_size[2];
    int axis;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  int ret = 1;

  LOG(3) << __FUNCTION__ << ": ndims=" << p->ndims << " axis=" << p->axis;

  if (p->dtype == DT_FLOAT) {
    if (p->ndims == 2 && p->axis == 1) {
      ret = reduction_d2a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
    }
  }


  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}


//
// Floor
//

namespace {
template<typename Tin, typename Tout>
  void op_floor(uint64_t out, uint64_t in, size_t nelems)
  {
    Tout* po = reinterpret_cast<Tout*>(out);
    const Tin* pi = reinterpret_cast<Tin*>(in);

    for (int64_t i = 0; i < nelems; ++i) {
      po[i] = std::floor(pi[i]);
    }
  }
}

int op_Floor(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__ << " begin";

  struct _Tensor {
    int dtype;
    int data_format;
    uint64_t addr;
    int32_t dims;
    int64_t nelems;
    int64_t dim_size[8];
  };

  struct Args {
    _Tensor in;
    _Tensor out;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  if (p->in.dtype == DT_FLOAT || p->out.dtype == DT_FLOAT) {
    op_floor<float, float>(p->out.addr, p->in.addr, p->in.nelems);
  } else {
    return 1;
  }

  LOG(2) << __FUNCTION__ << " end";
  return 0;
}

//
// Transpose
//
#ifndef LIBVETF_INTRINSIC

namespace {
template<typename Tin, typename Tout = Tin>
  int transpose4_0231(uint64_t out, uint64_t in, const int32_t* dim_size)
  {
    Tout* po = reinterpret_cast<Tout*>(out);
    const Tin* pi = reinterpret_cast<Tin*>(in);

    uint64_t si2 = dim_size[3];
    uint64_t si1 = si2 * dim_size[2];
    uint64_t si0 = si1 * dim_size[1];

    uint64_t so2 = dim_size[1];
    uint64_t so1 = so2 * dim_size[3];
    uint64_t so0 = so1 * dim_size[2];

    for (int64_t i0 = 0; i0 < dim_size[0]; ++i0) {
      for (int64_t i1 = 0; i1 < dim_size[2]; ++i1) {
        for (int64_t i2 = 0; i2 < dim_size[3]; ++i2) {
          for (int64_t i3 = 0; i3 < dim_size[1]; ++i3) {
            po[i0 * so0 + i1 * so1 + i2 * so2 + i3]
              = pi[i0 * si0 + i1 * si2 + i2 + i3 * si1];
          }
        }
      }
    }

    return 0;
  }

template<typename Tin, typename Tout = Tin>
  int transpose4_0312(uint64_t out, uint64_t in, const int32_t* dim_size)
  {
    Tout* po = reinterpret_cast<Tout*>(out);
    const Tin* pi = reinterpret_cast<Tin*>(in);

    uint64_t si2 = dim_size[3];
    uint64_t si1 = si2 * dim_size[2];
    uint64_t si0 = si1 * dim_size[1];

    uint64_t so2 = dim_size[2];
    uint64_t so1 = so2 * dim_size[1];
    uint64_t so0 = so1 * dim_size[3];

    for (int64_t i0 = 0; i0 < dim_size[0]; ++i0) {
      for (int64_t i1 = 0; i1 < dim_size[3]; ++i1) {
        for (int64_t i2 = 0; i2 < dim_size[1]; ++i2) {
          for (int64_t i3 = 0; i3 < dim_size[2]; ++i3) {
            po[i0 * so0 + i1 * so1 + i2 * so2 + i3]
              = pi[i0 * si0 + i1 + i2 * si1 + i3 * si2];
          }
        }
      }
    }

    return 0;
  }
}
#endif

int op_Transpose(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__ << " begin";

  struct Args {
    int dtype;
    uint64_t in;
    uint64_t out;
    int size;
    int32_t dim_size[4]; // in
    int32_t perm[4];
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  LOG(3) << __FUNCTION__ << " size=" << p->size
    << " perm=(" << p->perm[0]
    << " " << p->perm[1]
    << " " << p->perm[2]
    << " " << p->perm[3]
    << ")";

  int ret = 1;
#ifdef SET_TIMER
  unsigned long long start = __veperf_get_stm();
#endif

  if (p->dtype == DT_FLOAT) {
    if (p->size == 4) {
      if (p->perm[0] == 0 && p->perm[1] == 2 
          && p->perm[2] == 3 && p->perm[3] == 1) {
#ifndef LIBVETF_INTRINSIC
        ret = transpose4_0231<float>(p->out, p->in, p->dim_size);
#else
        ret = transpose4_0231(p->out, p->in, p->dim_size);
#endif

#ifdef SET_TIMER
  unsigned long long end = __veperf_get_stm();
  printf("transpose231, len %d %d %d %d:%lfms\n",p->dim_size[0],p->dim_size[1],p->dim_size[2],p->dim_size[3],(end-start)/(800e3));
#endif

      } else if (p->perm[0] == 0 && p->perm[1] == 3 
                 && p->perm[2] == 1 && p->perm[3] == 2) {
#ifndef LIBVETF_INTRINSIC
        ret = transpose4_0312<float>(p->out, p->in, p->dim_size);
#else
        ret = transpose4_0312(p->out, p->in, p->dim_size);
#endif

#ifdef SET_TIMER
  unsigned long long end = __veperf_get_stm();
  printf("transpose312, len %d %d %d %d:%lfms\n",p->dim_size[0],p->dim_size[1],p->dim_size[2],p->dim_size[3],(end-start)/(800e3));
#endif

      }
    }
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

//
// MatMul
//

namespace {
#define GEMM_ARGS(T) \
char* transa, char* transb, \
const int* N, const int* M, const int* K, \
const T* alpha, \
const T* A, const int* lda, \
const T* B, const int* ldb, \
const T* beta, \
T* C, const int* ldc

#define GEMM_REAL_ARGS \
transa, transb, N, M, K, alpha, A, lda, B, ldb, beta, C, ldc

template <typename T> void blas_gemm(GEMM_ARGS(T)) { assert(false && "blas_gemm: not implemented"); }
template<> void blas_gemm<float>(GEMM_ARGS(float)) { sgemm_(GEMM_REAL_ARGS); }

// C[M x N] = A[M x K] * B[K x N] ([rows x cols] in row-major)
//
// M[H x W] (rows x cols in row-major) = M[W x H] (rows x cols in col-major)
//
// C[M x N] (RxC in RM)
//   = C[N x M] (RxC in CM)
//   = B[N x K] (RxC in CM) * A[K x M] (RxC in CM)
//   = B[K x N] (RxC in RM) * A[M x K] (RxC in RM)
//                
template<typename T, char TransA, char TransB>
  int matmul(uint64_t c, uint64_t a, uint64_t b, int M, int N, int K)
  {
    LOG(3) << __FUNCTION__ << " begin: (" << M << "," << N << "," << K << ")";
    T* C = reinterpret_cast<T*>(c);
    const T* A = reinterpret_cast<const T*>(a);
    const T* B = reinterpret_cast<const T*>(b);

    T alpha = T(1);
    T beta = T(0);

    char transa = TransA;
    char transb = TransB;
    int lda = TransA == 'N' ? K : M;
    int ldb = TransB == 'N' ? N : K;

#pragma omp parallel
    {
      int nthreads = omp_get_num_threads() ;
      int threadid = omp_get_thread_num() ;

      int chunkSize = M / nthreads ;
      int remain    = M % nthreads ;

      int chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
      int myChunk    = chunkSize + ( threadid < remain ? 1 : 0 ) ;

      int offset    = TransA == 'N' ? K : 1 ;

      if( myChunk > 0 ) {
        blas_gemm<T>(&transb, &transa, &N, &myChunk, &K, &alpha, B, &ldb, A+offset*chunkBegin, &lda, &beta, C+N*chunkBegin, &N);
      }
    }

    LOG(2) << __FUNCTION__ << " end";
    return 0;
  }
}

int op_MatMul(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__ << " begin";

  struct Args {
    int dtype;
    uint64_t a;
    uint64_t b;
    uint64_t out;
    int64_t dim_size_a[2];
    int64_t dim_size_b[2];
    int32_t transpose_a;
    int32_t transpose_b;
  } const *p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  LOG(3) << __FUNCTION__
    << " a=(" << p->dim_size_a[0] << ", " << p->dim_size_a[1] << ")"
    << " b=(" << p->dim_size_b[0] << ", " << p->dim_size_b[1] << ")"
    << " transpose_a=" << p->transpose_a
    << " transpose_b=" << p->transpose_b;

  int ret = 0;
  if (p->dtype == DT_FLOAT) {
    if (!p->transpose_a && !p->transpose_b) {
      assert(p->dim_size_a[1] == p->dim_size_b[0]);
#if 0
      ret = matmul<float, 'N', 'N'>(
          p->out, p->a, p->b, p->dim_size_a[0], p->dim_size_b[1], p->dim_size_a[1]);
#else
      /* vednn version */
      const uint64_t inDim  = p->dim_size_a[1] ;
      const uint64_t outDim = p->dim_size_b[1] ;
      const uint64_t nBatch = p->dim_size_a[0] ;
      const void* pDataIn     = reinterpret_cast<const void*>(p->a);
      const void* pDataWeight = reinterpret_cast<const void*>(p->b);
      void*       pDataOut    = reinterpret_cast<void*>(p->out);
      ret = vednnLinearForward(inDim,outDim,nBatch, pDataIn, pDataWeight, pDataOut) ;
#endif
    } else if (!p->transpose_a && p->transpose_b) {
      assert(p->dim_size_a[1] == p->dim_size_b[1]);
      ret = matmul<float, 'N', 'T'>(
          p->out, p->a, p->b, p->dim_size_a[0], p->dim_size_b[0], p->dim_size_a[1]);
    } else if (p->transpose_a && !p->transpose_b) {
      assert(p->dim_size_a[0] == p->dim_size_b[0]);
      ret = matmul<float, 'T', 'N'>(
          p->out, p->a, p->b, p->dim_size_a[1], p->dim_size_b[1], p->dim_size_a[0]);
    }
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

//
// Softmax
//

int op_Softmax(const void* args, size_t len)
{
  struct Args {
    int dtype;
    int bool_log;
    uint64_t in;
    uint64_t out;
    uint64_t batch_size;
    uint64_t num_classes;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  if (p->dtype == DT_FLOAT) {
    const float* in = reinterpret_cast<const float*>(p->in);
    float* out = reinterpret_cast<float*>(p->out);

// [todo] use vednn
    if( p->bool_log ) {
      // LogSoftmax
      for(uint64_t b=0; b<p->batch_size; b++) {
        float max = -FLT_MAX ;
        for(uint64_t i=0; i<p->num_classes; i++) {
           if( max < in[i] ) max = in[i] ;
        }

        float sum = 0.f ;
        for(uint64_t i=0; i<p->num_classes; i++) {
          const float shifted_in = in[i] - max ; 
          sum += expf(shifted_in) ;
          out[i] = shifted_in ;
        }

        float log_sum = logf(sum) ;
        for(uint64_t i=0; i<p->num_classes; i++) {
          out[i] -= log_sum ;
        }

        in  += p->num_classes ; 
        out += p->num_classes ; 
      }
    }
    else {
      // Softmax
      for(uint64_t b=0; b<p->batch_size; b++) {
        float max = -FLT_MAX ;
        for(uint64_t i=0; i<p->num_classes; i++) {
          if( max < in[i] ) max = in[i] ;
        }

        float sum = 0.f ;
        for(uint64_t i=0; i<p->num_classes; i++) {
          sum += (out[i] = expf(in[i]-max)) ;
        }

        float inv_sum = 1.f / sum ;
#pragma _NEC novector  // compiler bug ??
        for(uint64_t i=0; i<p->num_classes; i++) {
          out[i] *= inv_sum ;
        }

        in  += p->num_classes ; 
        out += p->num_classes ; 
      }
    }
  }
  
  return 1;
}

namespace {

int unary_op(const void* args, size_t len,
             int (*func_f32_f32)(uint64_t, uint64_t, size_t))
{
  LOG(2) << __FUNCTION__ << " begin";

  struct _Tensor {
    int dtype;
    int data_format;
    uint64_t addr;
    int32_t dims;
    int64_t nelems;
    int64_t dim_size[8];
  };
  
  struct Args {
    _Tensor in;
    _Tensor out;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  int ret = 1;
  if (p->in.dtype == DT_FLOAT || p->out.dtype == DT_FLOAT) {
    ret = func_f32_f32(p->out.addr, p->in.addr, p->in.nelems);
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}
#ifndef LIBVETF_INTRINSIC

template<typename Tin, typename Tout>
int sqrt_(uint64_t out, uint64_t in, size_t nelems)
{
  Tout* po = reinterpret_cast<Tout*>(out);
  const Tin* pi = reinterpret_cast<Tin*>(in);

  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = std::sqrt(pi[i]);
  }
  return 0;
}

template<typename Tin, typename Tout>
int rsqrt(uint64_t out, uint64_t in, size_t nelems)
{
  Tout* po = reinterpret_cast<Tout*>(out);
  const Tin* pi = reinterpret_cast<Tin*>(in);

  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = Tin(1) / std::sqrt(pi[i]);
  }
  return 0;
}


template<typename Tin, typename Tout>
int square(uint64_t out, uint64_t in, size_t nelems)
{
  Tout* po = reinterpret_cast<Tout*>(out);
  const Tin* pi = reinterpret_cast<Tin*>(in);

  for (int64_t i = 0; i < nelems; ++i) {
    po[i] = pi[i] * pi[i];
  }
  return 0;
}
#endif
}

int op_Sqrt(const void* args, size_t len)
{
  int r = 0;
#ifdef SET_TIMER
  unsigned long long start = __veperf_get_stm();
#endif
#ifndef LIBVETF_INTRINSIC
  r = unary_op(args, len, sqrt_<float, float>);
#else
  r = unary_op(args, len, sqrt_);
#endif
#ifdef SET_TIMER
  unsigned long long end = __veperf_get_stm();
  printf("sqrt, len %d: %lf ms\n",len,(end-start)/(800e3));
#endif
  return r;
}

int op_Rsqrt(const void* args, size_t len)
{
  int r = 0;
#ifdef SET_TIMER
  unsigned long long start = __veperf_get_stm();
#endif
#ifndef LIBVETF_INTRINSIC
  r = unary_op(args, len, rsqrt<float, float>);
#else
  r = unary_op(args, len, rsqrt);
#endif
#ifdef SET_TIMER
  unsigned long long end = __veperf_get_stm();
  printf("rsqrt, len %d: %lf ms\n",len,(end-start)/(800e3));
#endif
  return r;
}

int op_Square(const void* args, size_t len)
{
  int r = 0;
#ifdef SET_TIMER
  unsigned long long start = __veperf_get_stm();
#endif
#ifndef LIBVETF_INTRINSIC
  r = unary_op(args, len, square<float, float>);
#else
  r = unary_op(args, len, square);
#endif
#ifdef SET_TIMER
  unsigned long long end = __veperf_get_stm();
  printf("square, len %d: %lf ms\n",len,(end-start)/(800e3));
#endif
  return r;
}
