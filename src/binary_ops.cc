#include <cstdint>
#include "kernel.h"
#include "types.h"
#include "log.h"
#include <sstream>

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




REGISTER_KERNEL("Add", "op_Add");
REGISTER_KERNEL("Sub", "op_Sub");
REGISTER_KERNEL("Mul", "op_Mul");
REGISTER_KERNEL("Div", "op_Div");
REGISTER_KERNEL("DivNoNan", "op_DivNoNan");
REGISTER_KERNEL("Minimum", "op_Minimum");
REGISTER_KERNEL("Maximum", "op_Maximum");
REGISTER_KERNEL("LessEqual", "op_LessEqual");
REGISTER_KERNEL("GreaterEqual", "op_GreaterEqual");

extern "C" {
  int op_Add(const void* arg, size_t len);
  int op_Sub(const void* arg, size_t len);
  int op_Mul(const void* arg, size_t len);
  int op_Div(const void* arg, size_t len);
  int op_DivNoNan(const void* arg, size_t len);
  int op_Minimum(const void* arg, size_t len);
  int op_Maximum(const void* arg, size_t len);
  int op_LessEqual(const void* arg, size_t len);
  int op_GreaterEqual(const void* arg, size_t len);
}

namespace {

struct _Tensor {
  int dtype;
  uint64_t addr;
  int32_t dims;
  int64_t nelems;
  int64_t dim_size[8];

  std::string to_s() const {
    std::stringstream s;

    s << "[dtype=" << dtype
      << ",dims=" << dims
      << ",nelems=" << nelems
      << "]";
    return s.str();
  }
};

struct BinaryOpArgs {
  _Tensor in0;
  _Tensor in1;
  _Tensor out;
};

bool CheckTypes(const BinaryOpArgs& args, int dt0, int dt1, int dt2)
{
  return args.in0.dtype == dt0
    && args.in1.dtype == dt1
    && args.out.dtype == dt2;
}

bool CheckTypesAll(const BinaryOpArgs& args, int dtype) {
  return CheckTypes(args, dtype, dtype, dtype);
}


int op_Binary(const void* args, size_t len, 
              int (*func)(const BinaryOpArgs&),
              const char* name)
{
  LOG(2) << name << ": begin";
  int ret = 1;

  if (sizeof(BinaryOpArgs) == len) {
    const BinaryOpArgs* p = reinterpret_cast<const BinaryOpArgs*>(args);

    LOG(3) << name << ":"
      << " in0=" << p->in0.to_s()
      << " in1=" << p->in1.to_s()
      << " out=" << p->out.to_s();

    if (func)
      ret = func(*p);
  } else {
    LOG(3) << name << ": illegal args size " << len
      << " bytes. but " << sizeof(BinaryOpArgs) << " bytes expected";
  }

  LOG(2) << name << ": end. ret=" << ret;
  return ret;
}

// Add
#ifndef LIBVETF_INTRINSIC

template <typename T>
int add_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] + i1;
  }

  return 0;
}

template <typename T>
int add_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] + pi1[i];
  }

  return 0;
}
#endif

int op_add(const BinaryOpArgs& args) {
  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

#ifdef SET_TIMER
  unsigned long long start = __veperf_get_stm();
#endif

    if (args.in0.nelems == 1) {
#ifndef LIBVETF_INTRINSIC
      r = add_n1<float>(args.out.addr, args.in1.addr, args.in0.addr,
                           args.out.nelems);
#else
     r = add_n1(args.out.addr, args.in1.addr, args.in0.addr,
                           args.out.nelems);
#endif

#ifdef SET_TIMER
      unsigned long long end = __veperf_get_stm();
      printf("add n1: %d, 		%lf ms\n",args.out.nelems,(end-start)/(800e3));
#endif
    } else if (args.in1.nelems == 1) {
#ifndef LIBVETF_INTRINSIC
      r = add_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.out.nelems);
#else
   r = add_n1(args.out.addr, args.in0.addr, args.in1.addr,
                           args.out.nelems);
#endif

#ifdef SET_TIMER
      unsigned long long end = __veperf_get_stm();
      printf("add n1: %d, 		%lf ms\n",args.out.nelems,(end-start)/(800e3));
#endif
    } else if (args.in0.nelems == args.in1.nelems) {
#ifndef LIBVETF_INTRINSIC
      r = add_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.in0.nelems);
#else
   r = add_nn(args.out.addr, args.in0.addr, args.in1.addr,
                           args.in0.nelems);
#endif

#ifdef SET_TIMER
      unsigned long long end = __veperf_get_stm();
      printf("add nn: %d, 		%lf ms\n",args.out.nelems,(end-start)/(800e3));
#endif
    }
  
  return r;
  }
  return 1;
}

// Sub

template <typename T>
int sub_1n(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  T i0 = *reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    po[i] = i0 - pi1[i];
  }
  return 0;
}

template <typename T>
int sub_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    po[i] = pi0[i] - i1;
  }
  return 0;
}

#ifndef LIBVETF_INTRINSIC
template <typename T>
int sub_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    po[i] = pi0[i] - pi1[i];
  }
  return 0;
}
#endif

template <typename T>
int sub2_nn_n1(uint64_t out, 
               uint64_t in0,
               uint64_t in1, 
               size_t n0,
               size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n0; ++i) {
    for (size_t j = 0; j < n1; ++j) {
      po[i * n1 + j] = pi0[i * n1 + j] - pi1[i];
    }
  }
  return 0;
}

int op_sub(const BinaryOpArgs& args) {
  if (CheckTypesAll(args, DT_FLOAT)) {
    int r=1;
    if (args.in0.nelems == 1) {
      r = sub_1n<float>(args.out.addr, args.in0.addr, args.in1.addr,
                        args.out.nelems);
    }
    else if(args.in1.nelems == 1) {
      r = sub_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                        args.out.nelems);
    }
    else if (args.in0.nelems == args.in1.nelems) {
#ifdef SET_TIMER
  unsigned long long start = __veperf_get_stm();
#endif

#ifndef LIBVETF_INTRINSIC
      r = sub_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.in0.nelems);
#else
      r = sub_nn(args.out.addr, args.in0.addr, args.in1.addr,
                           args.in0.nelems);
#endif

#ifdef SET_TIMER
      unsigned long long end = __veperf_get_stm();
      printf("sub nn: %d, 		%lf ms\n",args.in0.nelems,(end-start)/(800e3));
#endif
    }
    else if (args.in0.dims == 2
               && args.in1.dims == 2
               && args.in0.dim_size[0] == args.in1.dim_size[0]
               && args.in1.dim_size[1] == 1) {
      r = sub2_nn_n1<float>(args.out.addr,
                               args.in0.addr,
                               args.in1.addr,
                               args.in0.dim_size[0],
                               args.in0.dim_size[1]);
    }
    return r;
  }
  return 1;
}

// Mul
#ifndef LIBVETF_INTRINSIC

template <typename T>
int mul_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] * i1;
  }

  return 0;
}

template <typename T>
int mul_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] * pi1[i];
  }

  return 0;
}
#endif

int op_mul(const BinaryOpArgs& args) {
  if (CheckTypesAll(args, DT_FLOAT)) {

#ifdef SET_TIMER
  unsigned long long start = __veperf_get_stm();
#endif
    int r=1;


    if (args.in0.nelems == 1) {
#ifndef LIBVETF_INTRINSIC
     r = mul_n1<float>(args.out.addr, args.in1.addr, args.in0.addr,
                           args.out.nelems);
#else
     r = mul_n1(args.out.addr, args.in1.addr, args.in0.addr,
                           args.out.nelems);
#endif

#ifdef SET_TIMER
      unsigned long long end = __veperf_get_stm();
      printf("mul n1: %d, 		%lf ms\n",args.out.nelems,(end-start)/(800e3));
#endif

    } else if (args.in1.nelems == 1) {
#ifndef LIBVETF_INTRINSIC
     r = mul_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.out.nelems);
#else
     r = mul_n1(args.out.addr, args.in0.addr, args.in1.addr,
                           args.out.nelems);
#endif


#ifdef SET_TIMER
      unsigned long long end = __veperf_get_stm();
      printf("mul n1: %d, 		%lf ms\n",args.out.nelems,(end-start)/(800e3));
#endif

    } else if (args.in0.nelems == args.in1.nelems) {
#ifndef LIBVETF_INTRINSIC
     r = mul_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.in0.nelems);
#else
     r = mul_nn(args.out.addr, args.in0.addr, args.in1.addr,
                           args.in0.nelems);
#endif

#ifdef SET_TIMER
      unsigned long long end = __veperf_get_stm();
      printf("mul nn: %d, 		%lf ms\n",args.out.nelems,(end-start)/(800e3));
#endif

    }
    return r;
  }
  return 1;
}


// LessEqual

template <typename T>
int lessEqual_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] <= i1;
  }
  return 0;
}

int op_lessEqual(const BinaryOpArgs& args) {
  if (CheckTypes(args, DT_FLOAT, DT_FLOAT, DT_BOOL)) {
    if (args.in1.nelems == 1) {
      return lessEqual_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                                 args.in0.nelems);
    }
  }
  return 1;
}

// Div

template <typename T>
int div_1n(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  T i0 = *reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    po[i] = i0 / pi1[i];
  }
  return 0;
}

#ifndef LIBVETF_INTRINSIC
template <typename T>
int div_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    po[i] = pi0[i] / i1;
  }
  return 0;
}

// nelems_in0 > nelems_in1
template <typename T>
int div2_nn_n1(uint64_t out, 
               uint64_t in0,
               uint64_t in1, 
               size_t n0,
               size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n0; ++i) {
    for (size_t j = 0; j < n1; ++j) {
      po[i * n1 + j] = pi0[i * n1 + j] / pi1[i];
    }
  }
  return 0;
}
#endif

int op_div(const BinaryOpArgs& args) {
  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

#ifdef SET_TIMER
  unsigned long long start = __veperf_get_stm();
#endif

    if (args.in0.nelems == 1) {
      /* TODO : impl intrinsic */
      r = div_1n<float>(args.out.addr, args.in0.addr, args.in1.addr,
                        args.out.nelems);

#ifdef SET_TIMER
      unsigned long long end = __veperf_get_stm();
      printf("div n1: %d, 		%lf ms\n",args.out.nelems,(end-start)/(800e3));
#endif

    } else if (args.in1.nelems == 1) {
      /* FIXME : modify a bug in intrinsic version */
#ifndef LIBVETF_INTRINSIC
      r = div_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.out.nelems);
#else
      r = div_n1(args.out.addr, args.in0.addr, args.in1.addr,
                           args.out.nelems);
#endif

#ifdef SET_TIMER
      unsigned long long end = __veperf_get_stm();
      printf("div n1: %d, 		%lf ms\n",args.out.nelems,(end-start)/(800e3));
#endif

    } else if (args.in0.dims == 2
               && args.in1.dims == 2
               && args.in0.dim_size[0] == args.in1.dim_size[0]
               && args.in1.dim_size[1] == 1) {
#ifndef LIBVETF_INTRINSIC
      r = div2_nn_n1<float>(args.out.addr,
                               args.in0.addr,
                               args.in1.addr,
                               args.in0.dim_size[0],
                               args.in0.dim_size[1]);
#else
      r = div2_nn_n1(args.out.addr,
                               args.in0.addr,
                               args.in1.addr,
                               args.in0.dim_size[0],
                               args.in0.dim_size[1]);
#endif
#ifdef SET_TIMER
      unsigned long long end = __veperf_get_stm();
      printf("div2 nn n1: %d %d, 		%lf ms\n",args.in0.dim_size[0],args.in0.dim_size[1],(end-start)/(800e3));
#endif

    }
    return r;
  }
  return 1;
}

// DivNoNan

template <typename T>
int divnonan_1n(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  T i0 = *reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    if( pi1[i] == T(0.) ) po[i] = T(0.) ;
    else                  po[i] = i0 / pi1[i];
  }
  return 0;
}

template <typename T>
int divnonan_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  if( i1 == T(0.) ) { 
    for (size_t i = 0; i < nelems; ++i) {
      po[i] = T(0.) ;
    }
  }
  else {
    for (size_t i = 0; i < nelems; ++i) {
      po[i] = pi0[i] / i1;
    }
  }
  return 0;
}

template <typename T>
int divnonan2_nn_n1(uint64_t out, 
                    uint64_t in0,
                    uint64_t in1, 
                    size_t n0,
                    size_t n1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n0; ++i) {
    if( pi1[i] == T(0.) ) {
      for (size_t j = 0; j < n1; ++j) {
        po[i * n1 + j] = T(0.) ;
      }
    }
    else { 
      for (size_t j = 0; j < n1; ++j) {
        po[i * n1 + j] = pi0[i * n1 + j] / pi1[i];
      }
    }
  }
  return 0;
}

int op_divnonan(const BinaryOpArgs& args) {
  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    if (args.in0.nelems == 1) {
      r = divnonan_1n<float>(args.out.addr, args.in0.addr, args.in1.addr,
                            args.out.nelems);
    } else if (args.in1.nelems == 1) {
      r = divnonan_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                             args.out.nelems);
    } else if (args.in0.dims == 2
               && args.in1.dims == 2
               && args.in0.dim_size[0] == args.in1.dim_size[0]
               && args.in1.dim_size[1] == 1) {
      r = divnonan2_nn_n1<float>(args.out.addr,
                               args.in0.addr,
                               args.in1.addr,
                               args.in0.dim_size[0],
                               args.in0.dim_size[1]);
    }
    return r;
  }
  return 1;
}

// Minimum

template <typename T>
int minimum_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] < i1 ? pi0[i] : i1;
  }
  return 0;
}

int op_minimum(const BinaryOpArgs& args)
{
  if (CheckTypesAll(args, DT_FLOAT)) {
    if (args.in1.nelems == 1) {
      return minimum_n1<float>(args.out.addr, args.in0.addr, args.in1.addr, args.out.nelems);
    }
  }
  return 1;
}

// Maximum

template <typename T>
int maximum_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] > i1 ? pi0[i] : i1;
  }
  return 0;
}

int op_maximum(const BinaryOpArgs& args)
{
  if (CheckTypesAll(args, DT_FLOAT)) {
    if (args.in1.nelems == 1) {
      return maximum_n1<float>(args.out.addr, args.in0.addr, args.in1.addr, args.out.nelems);
    }
  }
  return 1;
}

// GreaterEqual

template <typename T>
int greaterEqual_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] >= i1;
  }
  return 0;
}

int op_greaterEqual(const BinaryOpArgs& args) {
  if (CheckTypes(args, DT_FLOAT, DT_FLOAT, DT_BOOL)) {
    if (args.in1.nelems == 1) {
      return greaterEqual_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                                    args.in0.nelems);
    }
  }
  return 1;
}

} // namespace

int op_Add(const void* args, size_t len)
{
  return op_Binary(args, len, op_add, "op_Add");
}

int op_Sub(const void* args, size_t len)
{
  return op_Binary(args, len, op_sub, "op_Sub");
}

int op_Mul(const void* args, size_t len)
{
  return op_Binary(args, len, op_mul, "op_Mul");
}

int op_Div(const void* args, size_t len)
{
  return op_Binary(args, len, op_div, "op_Div");
}

int op_DivNoNan(const void* args, size_t len)
{
  return op_Binary(args, len, op_divnonan, "op_DivNoNan");
}

int op_Minimum(const void* args, size_t len)
{
  return op_Binary(args, len, op_minimum, "op_Minimum");
}

int op_Maximum(const void* args, size_t len)
{
  return op_Binary(args, len, op_maximum, "op_Maximum");
}

int op_LessEqual(const void* args, size_t len)
{
  return op_Binary(args, len, op_lessEqual, "op_LessEqual");
}

int op_GreaterEqual(const void* args, size_t len)
{
  return op_Binary(args, len, op_greaterEqual, "op_GreaterEqual");
}
