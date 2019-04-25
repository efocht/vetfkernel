#include <cstdint>
#include "kernel.h"
#include "types.h"
#include "log.h"
#include <sstream>

REGISTER_KERNEL("SigmoidGrad", "op_SigmoidGrad");
REGISTER_KERNEL("TanhGrad",    "op_TanhGrad");


extern "C" {
int op_SigmoidGrad(const void* arg, size_t len);
int op_TanhGrad(const void* arg, size_t len);

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

static inline
bool CheckTypes(const BinaryOpArgs& args, int dt0, int dt1, int dt2)
{
  return args.in0.dtype == dt0
    && args.in1.dtype == dt1
    && args.out.dtype == dt2;
}

static inline
bool CheckTypesAll(const BinaryOpArgs& args, int dtype) {
  return CheckTypes(args, dtype, dtype, dtype);
}


static
int op_CwiseGradients(const void* args, size_t len,
                      int (*func)(const BinaryOpArgs&),
		      const char* name)
{
  LOG(2) << name << ": begin";
  int ret = 1;

  if (sizeof(BinaryOpArgs) == len) {
    const BinaryOpArgs* p = reinterpret_cast<const BinaryOpArgs*>(args);

    LOG(3) << name << ":"
      << " out="  << p->in0.to_s()
      << " gout=" << p->in1.to_s()
      << " gin="  << p->out.to_s();

    if (func)
      ret = func(*p);
  } else {
    LOG(3) << name << ": illegal args size " << len
      << " bytes. but " << sizeof(BinaryOpArgs) << " bytes expected";
  }

  LOG(2) << name << ": end. ret=" << ret;
  return ret;
}



// SigmoidGrad

template <typename T>
int sigmoid_grad_nn(uint64_t gin, uint64_t out, uint64_t gout, size_t n)
{
  T* gi = reinterpret_cast<T*>(gin);
  const T* oo = reinterpret_cast<const T*>(out);
  const T* go = reinterpret_cast<const T*>(gout);

  for (size_t i = 0; i < n; ++i) {
    gi[i] = go[i] * oo[i] * (T(1.) - oo[i]) ;
  }

  return 0;
}

int op_sigmoidGrad(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    // TODO : impl other patterns
    if (args.in0.nelems == args.in1.nelems) {
     r = sigmoid_grad_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                               args.in0.nelems);
    }

    return r;
  }
  return 1;
}

// TanhGrad

template <typename T>
int tanh_grad_nn(uint64_t gin, uint64_t out, uint64_t gout, size_t n)
{
  T* gi = reinterpret_cast<T*>(gin);
  const T* oo = reinterpret_cast<const T*>(out);
  const T* go = reinterpret_cast<const T*>(gout);

  for (size_t i = 0; i < n; ++i) {
    gi[i] = go[i] * (T(1.) - oo[i] * oo[i]) ;
  }

  return 0;
}

int op_tanhGrad(const BinaryOpArgs& args) {

//  printf("args.in0.dims = %ld\n", args.in0.dims) ;
//  for(int i=0; i<args.in0.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in0.dim_size[i]) ;
//  printf("args.in1.dims = %ld\n", args.in1.dims) ;
//  for(int i=0; i<args.in1.dims ; i++ ) printf(" [%d] = %ld\n", i, args.in1.dim_size[i]) ;

  if (CheckTypesAll(args, DT_FLOAT)) {

    int r=1;

    // TODO : impl other patterns
    if (args.in0.nelems == args.in1.nelems) {
     r = tanh_grad_nn<float>(args.out.addr, args.in0.addr, args.in1.addr,
                               args.in0.nelems);
    }

    return r;
  }
  return 1;
}

} // namespace


int op_SigmoidGrad(const void* args, size_t len)
{
  return op_CwiseGradients(args, len, op_sigmoidGrad, "op_SigmoidGrad");
}
int op_TanhGrad(const void* args, size_t len)
{
  return op_CwiseGradients(args, len, op_tanhGrad, "op_TanhGrad");
}
