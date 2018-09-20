#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "kernel.h"
#include "types.h"
#include "log.h"

REGISTER_KERNEL("Snapshot", "op_Snapshot")
REGISTER_KERNEL("Fill", "op_fill");
REGISTER_KERNEL("AddN", "op_AddN");
REGISTER_KERNEL("BiasAdd", "op_BiasAdd");
REGISTER_KERNEL("BiasAddGrad", "op_BiasAddGrad");
REGISTER_KERNEL("Relu", "op_Relu");
REGISTER_KERNEL("ReluGrad", "op_ReluGrad");
REGISTER_KERNEL("Mul", "op_Mul");

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
  int op_Mul(const void* arg, size_t len);
  int op_Snapshot(const void* arg, size_t len);
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

  if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NHWC) {
    return BiasAdd_NHWC<float>(p->out, p->in, p->bias, p->batch, p->width, p->height, p->channel);
  } else if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NCHW) {
    return BiasAdd_NCHW<float>(p->out, p->in, p->bias, p->batch, p->width, p->height, p->channel);
  }
#if 0
  fprintf(stderr, "%s done\n", __PRETTY_FUNCTION__);
#endif
  return 1;
}

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

  if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NHWC) {
    return BiasAddGrad_NHWC<float>(p->output, p->output_backprop, p->batch, p->width, p->height, p->channel);
  } else if (p->dtype == DT_FLOAT && p->data_format == FORMAT_NCHW) {
    return BiasAddGrad_NCHW<float>(p->output, p->output_backprop, p->batch, p->width, p->height, p->channel);
  }
  LOG(2) << __FUNCTION__ << " done";
  return 1;
}

//
// Relu and ReluGrad
//

namespace {
template<typename T>
  int Relu(uint64_t in, uint64_t out, uint64_t num_elems)
  {
    const T* pi = reinterpret_cast<const T*>(in);
    T* po = reinterpret_cast<T*>(out);

    for (uint64_t i = 0; i < num_elems; ++i) {
      po[i] = pi[i] > T(0.0) ? pi[i] : T(0.0);
      //po[i] = std::max(T(0), pi[i]); // unvectorized. why?
    }

    return 0;
  }

template<typename T>
  int ReluGrad(uint64_t g, uint64_t a, uint64_t out, uint64_t num_elems)
  {
    const T* pg = reinterpret_cast<const T*>(g);
    const T* pa = reinterpret_cast<const T*>(a);
    T* po = reinterpret_cast<T*>(out);

    for (uint64_t i = 0; i < num_elems; ++i) {
      po[i] = pg[i] * (pa[i] > 0);
    }

    return 0;
  }
}

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
    return Relu<float>(p->in, p->out, p->num_elems);
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
    return ReluGrad<float>(p->g, p->a, p->output, p->num_elems);
  }
  return 1;
}

//
// Mul
// 

namespace {
template <typename T>
mul_nx1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] * i1;
  }
}

template <typename T>
mul_nxn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] * pi1[i];
  }
}
}

int op_Mul(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__;
  struct Args {
    int dtype;
    uint64_t in0;
    uint64_t in1;
    uint64_t out;
    int32_t dims_in0;
    int32_t dims_in1;
    int32_t dims_out;
    int64_t nelems_in0;
    int64_t nelems_in1;
    int64_t nelems_out;
    int64_t dim_size_in0[8];
    int64_t dim_size_in1[8];
    int64_t dim_size_out[8];
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  LOG(3) << "op_Mul:"
    << " dims_in0=" << p->dims_in0
    << " dims_in1=" << p->dims_in1
    << " dims_out=" << p->dims_out
    << " nelems_in0=" << p->nelems_in0
    << " nelems_in1=" << p->nelems_in1
    << " nelems_out=" << p->nelems_out;

  if (p->dtype == DT_FLOAT) {
    if (p->nelems_in0 == 1) {
      assert(p->nelems_in1 == p->nelems_out);
      mul_nx1<float>(p->out, p->in1, p->in0, p->nelems_out);
    } else if (p->nelems_in1 == 1) {
      assert(p->nelems_in0 == p->nelems_out);
      mul_nx1<float>(p->out, p->in0, p->in1, p->nelems_out);
    } else if (p->nelems_in0 == p->nelems_in1) {
      assert(p->nelems_in0 == p->nelems_out);
      mul_nxn<float>(p->out, p->in0, p->in1, p->nelems_out);
    } else {
      return 1;
    }
  } else {
    return 1;
  }

  LOG(2) << __FUNCTION__ << " end";

  return 0;
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
  LOG(2) << __FUNCTION__ << " done";

  return 0;
}

