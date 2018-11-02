#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "kernel.h"
#include "types.h"
#include "log.h"

#include <omp.h>

REGISTER_KERNEL("Sum", "op_Sum");
REGISTER_KERNEL("Prod", "op_Prod");

#define CHECK_ARG_LEN(l0, l1) \
  if ((l0) != (l1)) { \
      fprintf(stderr, "%s: illegal argument lenght: %ld expected but %ld\n", (l1), (l0)); \
      return 1; \
  }

extern "C" {
  int op_Sum(const void* arg, size_t len);
  int op_Prod(const void* arg, size_t len);
}


//
// Sum
//

namespace {
template <typename T>
int sum_d2a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
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
      ret = sum_d2a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
    }
  }
  else if (p->dtype == DT_DOUBLE) {
    if (p->ndims == 2 && p->axis == 1) {
      ret = sum_d2a1<double>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
    }
  }


  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

//
// Prod
//

namespace {
template <typename T>
int prod_d2a1(uint64_t out, uint64_t in, size_t dim0, size_t dim1)
{
  T* po = reinterpret_cast<T*>(out);
  const T* pi = reinterpret_cast<const T*>(in);

  for (size_t i = 0; i < dim0; ++i) {
    T s = T(1);
    for (size_t j = 0; j < dim1; ++j) {
      s *= pi[i * dim1 + j];
    }
    po[i] = s;
  }

  return 0;
}
}

int op_Prod(const void* args, size_t len)
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
      ret = prod_d2a1<float>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
    }
  }
  else if (p->dtype == DT_DOUBLE) {
    if (p->ndims == 2 && p->axis == 1) {
      ret = prod_d2a1<double>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
    }
  }
  else if (p->dtype == DT_INT32) {
    if (p->ndims == 2 && p->axis == 1) {
      ret = prod_d2a1<int32_t>(p->out, p->in, p->dim_size[0], p->dim_size[1]);
    }
  }


  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

