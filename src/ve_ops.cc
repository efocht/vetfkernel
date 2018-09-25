#include <cstdint>
#include "kernel.h"
#include "types.h"
#include "log.h"
#include <sstream>

#include <asl.h>

REGISTER_KERNEL("Select", "op_Select");
REGISTER_KERNEL("RandomUniform", "op_RandomUniform");

extern "C" {
  int op_Select(const void* args, size_t len);
  int op_RandomUniform(const void* args, size_t len);
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

struct VEOpArgs {
  int ninputs;
  int noutputs;
  _Tensor input[4];
  _Tensor output[4];
};

int op_Kernel(const void* args, size_t len, 
              int (*func)(const VEOpArgs&),
              const char* name)
{
  LOG(2) << name << ": begin";
  int ret = 1;

  if (sizeof(VEOpArgs) == len) {
    const VEOpArgs* p = reinterpret_cast<const VEOpArgs*>(args);

    for (int i = 0; i < p->ninputs; ++i)
      LOG(3) << name << ": in[" << i << "]=" << p->input[i].to_s();
    for (int i = 0; i < p->noutputs; ++i)
      LOG(3) << name << ": out[" << i << "]=" << p->output[i].to_s();

    if (func)
      ret = func(*p);
  } else {
    LOG(3) << name << ": illegal args size " << len
      << " bytes. but " << sizeof(VEOpArgs) << " bytes expected";
  }

  LOG(2) << name << ": end. ret=" << ret;
  return ret;
}

template <typename T>
int op_select_nn(uint64_t out,
                 uint64_t cond,
                 uint64_t then,
                 uint64_t else_,
                 size_t n)
{
  T* po = reinterpret_cast<T*>(out);
  const bool* pc = reinterpret_cast<const bool*>(cond);
  const T* pt = reinterpret_cast<const T*>(then);
  const T* pe = reinterpret_cast<const T*>(else_);

  for (size_t i = 0; i < n; ++i) {
    po[i] = pc[i] ? pt[i] : pe[i];
  }
  return 0;
}

int op_select(const VEOpArgs& args)
{
  if (args.ninputs != 3 && args.noutputs != 1)
    return 1;

  if (args.input[0].dtype == DT_BOOL
      && args.input[1].dtype == DT_FLOAT
      && args.input[2].dtype == DT_FLOAT
      && args.output[0].dtype == DT_FLOAT) {
    if (args.input[0].nelems == args.input[1].nelems
        && args.input[0].nelems == args.input[2].nelems) {
      return op_select_nn<float>(args.output[0].addr,
                                 args.input[0].addr,
                                 args.input[1].addr,
                                 args.input[2].addr,
                                 args.input[0].nelems);
    }
  }

  return 0;
}

int op_randomUniform(const VEOpArgs& args)
{
  if (args.ninputs != 0 && args.noutputs != 1)
    return 1;

  LOG(3) << "op_RandomUniform: nelems=" << args.output[0].nelems;

  asl_random_t hnd;

  if (asl_random_create(&hnd, ASL_RANDOMMETHOD_AUTO) != ASL_ERROR_OK) {
    fprintf(stderr, "asl_random_create failed\n");
    exit(-1);
  }

  if (args.output[0].dtype == DT_FLOAT) {
    float* p = reinterpret_cast<float*>(args.output[0].addr);
    if (asl_random_generate_s(hnd, args.output[0].nelems, p) != ASL_ERROR_OK) {
      fprintf(stderr, "asl_random_generate_d failed\n");
      exit(-1);
    }
  }

  if (asl_random_destroy(hnd) != ASL_ERROR_OK) {
    fprintf(stderr, "asl_random_destroy failed\n");
    exit(-1);
  }

  return 0;
}

} // namespace

int op_Select(const void* args, size_t len)
{
  return op_Kernel(args, len, op_select, "op_Select");
}

int op_RandomUniform(const void* args, size_t len)
{
  return op_Kernel(args, len, op_randomUniform, "op_RandomUniform");
}

