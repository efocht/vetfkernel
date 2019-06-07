#include <cstdint>
#include <cassert>
#include "kernel.h"
#include "types.h"
#include "log.h"
#include <sstream>

#define LIBVETF_INTRINSIC

#ifdef LIBVETF_INTRINSIC
#include "libvetfkernel.h"
#endif

REGISTER_KERNEL("Add", "op_Add");
REGISTER_KERNEL("Sub", "op_Sub");
REGISTER_KERNEL("Mul", "op_Mul");
REGISTER_KERNEL("Div", "op_Div");
REGISTER_KERNEL("DivNoNan", "op_DivNoNan");
REGISTER_KERNEL("Pow", "op_Pow");
REGISTER_KERNEL("SquaredDifference", "op_SquaredDifference")
REGISTER_KERNEL("RsqrtGrad", "op_RsqrtGrad")
REGISTER_KERNEL("Minimum", "op_Minimum");
REGISTER_KERNEL("Maximum", "op_Maximum");
REGISTER_KERNEL("Equal", "op_Equal");
REGISTER_KERNEL("NotEqual", "op_NotEqual");
REGISTER_KERNEL("LessEqual", "op_LessEqual");
REGISTER_KERNEL("GreaterEqual", "op_GreaterEqual");

extern "C" {
  int op_Add(const void* arg, size_t len);
  int op_Sub(const void* arg, size_t len);
  int op_Mul(const void* arg, size_t len);
  int op_Div(const void* arg, size_t len);
  int op_DivNoNan(const void* arg, size_t len);
  int op_Pow(const void* arg, size_t len);
  int op_SquaredDifference(const void* arg, size_t len);
  int op_RsqrtGrad(const void* arg, size_t len);
  int op_Minimum(const void* arg, size_t len);
  int op_Maximum(const void* arg, size_t len);
  int op_Equal(const void* arg, size_t len);
  int op_NotEqual(const void* arg, size_t len);
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
      << "[";
    for (int i = 0; i < dims; ++i)
        s << " " << dim_size[i];
    s  << " ],nelems=" << nelems
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

bool CheckDimsAll(const BinaryOpArgs& args, size_t dims)
{
    return args.in0.dims == dims
        && args.in1.dims == dims
        && args.out.dims == dims;
}

bool IsSameDims(const BinaryOpArgs& args)
{
    return args.in0.dims == args.in1.dims
        && args.in0.dims == args.out.dims;
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

#ifdef LIBVETF_INTRINSIC
template <>
int add_n1<float>(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  return add_n1_f32(out,in0,in1,n) ;
}
#endif

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

#ifdef LIBVETF_INTRINSIC
template <>
inline int add_nn<float>(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  return add_nn_f32(out,in0,in1,n) ;
}
#endif

template <typename T>
int add2_nn_1n(uint64_t out,
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
      po[i * n1 + j] = pi0[i * n1 + j] + pi1[j];
    }
  }
  return 0;
}

// X = Y op Z
template<typename T, typename F>
int binop_dim3(_Tensor const& X, _Tensor const& Y, _Tensor const& Z, F op)
{
    LOG(3) << __FUNCTION__;
    T* px = reinterpret_cast<T*>(X.addr);
    T const* py = reinterpret_cast<T*>(Y.addr);
    T const* pz = reinterpret_cast<T*>(Z.addr);

    int64_t const* sx = X.dim_size;
    int64_t const* sy = Y.dim_size;
    int64_t const* sz = Z.dim_size;

    for (size_t i0 = 0; i0 < sx[0]; ++i0) {
        for (size_t i1 = 0; i1 < sx[1]; ++i1) {
            for (size_t i2 = 0; i2 < sx[2]; ++i2) {
                px[i0 * sx[1] * sx[2] + i1 * sx[2] + i2]
                    = op(py[(i0 % sy[0]) * sy[1] * sy[2] + (i1 % sy[1]) * sy[2] + (i2 % sy[2])],
                         pz[(i1 % sz[0]) * sz[1] * sz[2] + (i1 % sz[1]) * sz[2] + (i2 % sz[2])]);
            }
        }
    }

    return 0;
}

// X = Y op Z
template<typename T, typename F>
int binop(_Tensor const& X, _Tensor const& Y, _Tensor const& Z, F op)
{
    T* px = reinterpret_cast<T*>(X.addr);
    T const* py = reinterpret_cast<T*>(Y.addr);
    T const* pz = reinterpret_cast<T*>(Z.addr);

    assert(X.dims == Y.dims && X.dims == Z.dims);

    if (X.dims == 3)
        return binop_dim3<T>(X, Y, Z, op);

    LOG(3) << __FUNCTION__;

    size_t dims = X.dims;

    for (size_t ix = 0; ix < X.nelems; ++ix) {
        size_t ix0[dims];
        size_t tmp = ix;
        for (int64_t dim = dims - 1; dim >= 0; --dim) {
            ix0[dim] = tmp % X.dim_size[dim];
            tmp /= X.dim_size[dim];
        }
        size_t iy = 0;
        size_t iz = 0;
        for (size_t dim = 0; dim < dims; ++dim) {
            iy = (iy * Y.dim_size[dim]) + ix0[dim] % Y.dim_size[dim];
            iz = (iz * Z.dim_size[dim]) + ix0[dim] % Z.dim_size[dim];
        }

        px[ix] = op(py[iy], pz[iz]);
#ifdef DEBUG
        fprintf(stderr, "ix=%lu[", ix);
        for (int64_t dim = 0; dim < dims; ++dim)
            fprintf(stderr, " %lu", ix0[dim]);
        fprintf(stderr, " ]");
        fprintf(stderr, " iy=%lu iz=%lu\n", iy, iz);
#endif
    }

    return 0;
}

int op_add(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    if (args.in0.nelems == 1) {
      r = add_n1<float>(args.out.addr, args.in1.addr, args.in0.addr,
                           args.out.nelems);
    } else if (args.in1.nelems == 1) {
      r = add_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.out.nelems);
    } else if (args.in0.nelems == args.in1.nelems) {
      r = add_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.in0.nelems);
    } else if (args.in0.dims == 2 && args.in1.dims == 1
        && args.in0.dim_size[1] == args.in1.dim_size[0] ) {
      r = add2_nn_1n<float>(args.out.addr,
			    args.in0.addr,
			    args.in1.addr,
			    args.in0.dim_size[0],
			    args.in0.dim_size[1]) ;
    } else if (IsSameDims(args)) {
      r = binop<float>(args.out, args.in0, args.in1,
                       [](float y, float z) -> float { return y + z; });
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

#ifdef LIBVETF_INTRINSIC
template <>
inline int sub_nn<float>(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  return sub_nn_f32(out,in0,in1,nelems) ;
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

template <typename T>
int sub2_nn_1n(uint64_t out,
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
      po[i * n1 + j] = pi0[i * n1 + j] - pi1[j];
    }
  }
  return 0;
}

int op_sub(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

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
      r = sub_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.in0.nelems);
    }
    else if (args.in0.dims == 2 && args.in1.dims == 2
               && args.in0.dim_size[0] == args.in1.dim_size[0]
               && args.in1.dim_size[1] == 1) {
      r = sub2_nn_n1<float>(args.out.addr,
                               args.in0.addr,
                               args.in1.addr,
                               args.in0.dim_size[0],
                               args.in0.dim_size[1]);
    }
    else if (args.in0.dims == 2 && args.in1.dims == 2
               && args.in0.dim_size[1] == args.in1.dim_size[1]
               && args.in1.dim_size[0] == 1) {
      r = sub2_nn_1n<float>(args.out.addr,
                               args.in0.addr,
                               args.in1.addr,
                               args.in0.dim_size[0],
                               args.in0.dim_size[1]);
    } else if (IsSameDims(args)) {
      r = binop<float>(args.out, args.in0, args.in1,
                       [](float y, float z) -> float { return y - z; });
    }
    return r;
  }
  return 1;
}

// Mul
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

#ifdef LIBVETF_INTRINSIC
template <>
inline int mul_n1<float>(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  return mul_n1_f32(out, in0, in1, n) ;
}
#endif

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

#ifdef LIBVETF_INTRINSIC
template <>
inline int mul_nn<float>(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  return mul_nn_f32(out, in0, in1, n) ;
}
#endif

// nelems_in0 > nelems_in1
template <typename T>
int mul2_nn_n1(uint64_t out, 
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
      po[i * n1 + j] = pi0[i * n1 + j] * pi1[i];
    }
  }
  return 0;
}

template <typename T>
int mul2_nn_1n(uint64_t out,
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
      po[i * n1 + j] = pi0[i * n1 + j] * pi1[j];
    }
  }
  return 0;
}

int op_mul(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    if (args.in0.nelems == 1) {
     r = mul_n1<float>(args.out.addr, args.in1.addr, args.in0.addr,
                           args.out.nelems);
    } else if (args.in1.nelems == 1) {
     r = mul_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.out.nelems);
    } else if (args.in0.nelems == args.in1.nelems) {
     r = mul_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.in0.nelems);
    } else if (args.in0.dims == 2 && args.in1.dims == 2 
               && args.in0.dim_size[0] == args.in1.dim_size[0] ) {
      if( args.in1.dim_size[1] == 1 ) {
        r = mul2_nn_n1<float>(args.out.addr,
                               args.in0.addr,
                               args.in1.addr,
                               args.in0.dim_size[0],
                               args.in0.dim_size[1]);
      }
      else if( args.in0.dim_size[1] == 1 ) {
        r = mul2_nn_n1<float>(args.out.addr,
                               args.in1.addr,
                               args.in0.addr,
                               args.in1.dim_size[0],
                               args.in1.dim_size[1]);
      }
    } else if (args.in0.dims == 2 && args.in1.dims == 1
	        && args.in0.dim_size[1] == args.in1.dim_size[0] ) {
      r = mul2_nn_1n<float>(args.out.addr,
	                    args.in0.addr,
			    args.in1.addr,
			    args.in0.dim_size[0],
			    args.in0.dim_size[1]) ;
    } else if (args.in0.dims == args.in1.dims
               && args.in0.dims == args.out.dims) {
        r = binop<float>(args.out, args.in0, args.in1,
                         [](float a, float b) -> float { return a * b; });
    }

    return r;
  }
  return 1;
}

// Equal

template <typename T>
int equal_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = (pi0[i] == i1);
  }
  return 0;
}

template <typename T>
int equal_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = (pi0[i] == pi1[i]);
  }
  return 0;
}

int op_equal(const BinaryOpArgs& args) {
  if (CheckTypes(args, DT_FLOAT, DT_FLOAT, DT_BOOL)) {
    if (args.in1.nelems == 1) {
      return equal_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                             args.in0.nelems);
    }
    else if( args.in0.nelems == args.in1.nelems ) {
      return equal_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                              args.in0.nelems);
    }
  }
  else if (CheckTypes(args, DT_DOUBLE, DT_DOUBLE, DT_BOOL)) {
    if (args.in1.nelems == 1) {
      return equal_n1<double>(args.out.addr, args.in0.addr, args.in1.addr,
                              args.in0.nelems);
    }
    else if( args.in0.nelems == args.in1.nelems ) {
      return equal_nn<double>(args.out.addr, args.in0.addr, args.in1.addr,
                              args.in0.nelems);
    }
  }
  else if (CheckTypes(args, DT_INT64, DT_INT64, DT_BOOL)) {
    if (args.in1.nelems == 1) {
      return equal_n1<int64_t>(args.out.addr, args.in0.addr, args.in1.addr,
                              args.in0.nelems);
    }
    else if( args.in0.nelems == args.in1.nelems ) {
      return equal_nn<int64_t>(args.out.addr, args.in0.addr, args.in1.addr,
                              args.in0.nelems);
    }
  }
  
  return 1;
}

// NotEqual

template <typename T>
int notEqual_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = (pi0[i] != i1);
  }
  return 0;
}

template <typename T>
int notEqual_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  bool* po = reinterpret_cast<bool*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = (pi0[i] != pi1[i]);
  }
  return 0;
}

int op_notEqual(const BinaryOpArgs& args) {
  if (CheckTypes(args, DT_FLOAT, DT_FLOAT, DT_BOOL)) {
    if (args.in1.nelems == 1) {
      return notEqual_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                             args.in0.nelems);
    }
    else if( args.in0.nelems == args.in1.nelems ) {
      return notEqual_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                              args.in0.nelems);
    }
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

template <typename T>
int div_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < nelems; ++i) {
    po[i] = pi0[i] / pi1[i];
  }
  return 0;
}

#ifdef LIBVETF_INTRINSIC
template <>
inline int div_n1<float>(uint64_t out, uint64_t in0, uint64_t in1, size_t nelems)
{
  return div_n1_f32(out, in0, in1, nelems) ;
}
#endif

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

#ifdef LIBVETF_INTRINSIC
template <>
inline int div2_nn_n1<float>(uint64_t out,
                             uint64_t in0,
			     uint64_t in1,
			     size_t n0,
			     size_t n1)
{
  return div2_nn_n1_f32(out, in0, in1, n0, n1) ;
}
#endif

int op_div(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;


  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    if (args.in0.nelems == 1) {
      /* TODO : impl intrinsic */
      r = div_1n<float>(args.out.addr, args.in0.addr, args.in1.addr,
                        args.out.nelems);

    } else if (args.in1.nelems == 1) {
      r = div_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.out.nelems);
    } else if (args.in0.nelems == args.in1.nelems ) {
      r = div_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                        args.out.nelems);
    } else if (args.in0.dims == 2
               && args.in1.dims == 2
               && args.in0.dim_size[0] == args.in1.dim_size[0]
               && args.in1.dim_size[1] == 1) {
      r = div2_nn_n1<float>(args.out.addr,
                               args.in0.addr,
                               args.in1.addr,
                               args.in0.dim_size[0],
                               args.in0.dim_size[1]);
    } else if (IsSameDims(args)) {
      r = binop<float>(args.out, args.in0, args.in1,
                       [](float y, float z) -> float { return y / z; });
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
int divnonan_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pi1[i] == T(0.) ? T(0.) : pi0[i]/pi1[i] ;
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

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    if (args.in0.nelems == args.in1.nelems) {
     r = divnonan_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                            args.in0.nelems);
    } else if (args.in0.nelems == 1) {
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


// Pow
template <typename T>
int pow_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    po[i] = std::pow(pi0[i],pi1[i]) ;
  }

  return 0;
}

int op_pow(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    // TODO : impl other patterns
    if (args.in0.nelems == args.in1.nelems) {
     r = pow_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.in0.nelems);
    }

    return r;
  }
  return 1;
}


// SquaredDifference
template <typename T>
int sqdiff_n1(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  T i1 = *reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    T diff = pi0[i] - i1 ;
    po[i] = diff * diff;
  }

  return 0;
}

template <typename T>
int sqdiff_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0 = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    T diff = pi0[i] - pi1[i];
    po[i] = diff * diff ;
  }

  return 0;
}

// nelems_in0 > nelems_in1
template <typename T>
int sqdiff2_nn_n1(uint64_t out,
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
      T diff = pi0[i * n1 + j] - pi1[i];
      po[i * n1 + j] = diff * diff ;
    }
  }
  return 0;
}

template <typename T>
int sqdiff2_nn_1n(uint64_t out,
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
      T diff = pi0[i * n1 + j] - pi1[j];
      po[i * n1 + j] = diff * diff ;
    }
  }
  return 0;
}

int op_sqdiff(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    if (args.in0.nelems == 1) {
     r = sqdiff_n1<float>(args.out.addr, args.in1.addr, args.in0.addr,
                           args.out.nelems);
    } else if (args.in1.nelems == 1) {
     r = sqdiff_n1<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.out.nelems);
    } else if (args.in0.nelems == args.in1.nelems) {
     r = sqdiff_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.in0.nelems);
    } else if (args.in0.dims == 2 && args.in1.dims == 2
               && args.in0.dim_size[0] == args.in1.dim_size[0] ) {
      if( args.in1.dim_size[1] == 1 ) {
        r = sqdiff2_nn_n1<float>(args.out.addr,
                               args.in0.addr,
                               args.in1.addr,
                               args.in0.dim_size[0],
                               args.in0.dim_size[1]);
      }
      else if( args.in0.dim_size[1] == 1 ) {
        r = sqdiff2_nn_n1<float>(args.out.addr,
                               args.in1.addr,
                               args.in0.addr,
                               args.in1.dim_size[0],
                               args.in1.dim_size[1]);
      }
    } else if (args.in0.dims == 2 && args.in1.dims == 2
	        && args.in0.dim_size[1] == args.in1.dim_size[1]
		&& args.in1.dim_size[0] == 1 ) {
      r = sqdiff2_nn_1n<float>(args.out.addr,
	                    args.in0.addr,
			    args.in1.addr,
			    args.in0.dim_size[0],
			    args.in0.dim_size[1]) ;
    }

    return r;
  }
  return 1;
}


// RsqrtGrad
template <typename T>
int rsqrt_grad_nn(uint64_t out, uint64_t in0, uint64_t in1, size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi0  = reinterpret_cast<const T*>(in0);
  const T* pi1 = reinterpret_cast<const T*>(in1);

  for (size_t i = 0; i < n; ++i) {
    T out     = pi0[i] ;
    T gradout = pi1[i] ;
    po[i] = T(-0.5) * gradout * out * out * out ;
  }

  return 0;
}
int op_rsqrt_grad(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    // TODO : impl other patterns
    if (args.in0.nelems == args.in1.nelems) {
     r = rsqrt_grad_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                           args.in0.nelems);
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

#if 0 // original ( partialy vectorized )
  for (size_t i = 0; i < n; ++i) {
    po[i] = pi0[i] >= i1;
  }
#else
  const size_t vloop_begin =  out & 0x3 ;
  const size_t vloop_end   =  n   & 0xFFFFFFFFFFFFFFFC ;

#pragma novector
  for(size_t i=0; i < vloop_begin ; i++) {
    po[i] = pi0[i] >= i1;
  }

  int*  po_i = reinterpret_cast<int*>(&po[vloop_begin]);
  for(size_t j=0; j < (vloop_end - vloop_begin)>>2 ; j++) {
    const int32_t b0 = pi0[vloop_begin+4*j+0] >= i1 ? 1 : 0 ;
    const int32_t b1 = pi0[vloop_begin+4*j+1] >= i1 ? 1 : 0 ;
    const int32_t b2 = pi0[vloop_begin+4*j+2] >= i1 ? 1 : 0 ;
    const int32_t b3 = pi0[vloop_begin+4*j+3] >= i1 ? 1 : 0 ;

    const int32_t b  = (b3 << 24) | (b2 << 16) | (b1 <<8) | b0 ;
    po_i[j] = b ;  
  }

#pragma novector
  for(size_t i=vloop_end; i < n ; i++) {
    po[i] = pi0[i] >= i1;
  }
#endif
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

int op_Pow(const void* args, size_t len)
{
  return op_Binary(args, len, op_pow, "op_Pow");
}

int op_SquaredDifference(const void* args, size_t len)
{
  return op_Binary(args, len, op_sqdiff, "op_SquaredDifference");
}

int op_RsqrtGrad(const void* args, size_t len)
{
  return op_Binary(args, len, op_rsqrt_grad, "op_RsqrtGrad");
}


int op_Minimum(const void* args, size_t len)
{
  return op_Binary(args, len, op_minimum, "op_Minimum");
}

int op_Maximum(const void* args, size_t len)
{
  return op_Binary(args, len, op_maximum, "op_Maximum");
}
int op_Equal(const void* args, size_t len)
{
  return op_Binary(args, len, op_equal, "op_Equal");
}
int op_NotEqual(const void* args, size_t len)
{
  return op_Binary(args, len, op_notEqual, "op_NotEqual");
}
int op_LessEqual(const void* args, size_t len)
{
  return op_Binary(args, len, op_lessEqual, "op_LessEqual");
}

int op_GreaterEqual(const void* args, size_t len)
{
  return op_Binary(args, len, op_greaterEqual, "op_GreaterEqual");
}
