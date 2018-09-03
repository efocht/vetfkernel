#include <cstdio>
#include <cstdint>
#include <iostream>
#include "kernel.h"
#include "types.h"

REGISTER_KERNEL("Fill", "ops_fill");
REGISTER_KERNEL("AddN", "ops_AddN");

extern "C" {
  int ops_fill(const void* arg, size_t len);
  int ops_AddN(const void* arg, size_t len);
}

namespace {
template <typename T>
  int fill(const void* args, size_t len)
  {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    struct Args {
      int data_type;
      uint64_t in;
      uint64_t out;
      size_t num_elems;
    } const* p;

    if (len != sizeof(*p)) {
      fprintf(stderr, "%s: illegal argument lenght: %ld expected but %ld\n", sizeof(*p), len);
      return 1;
    }

    p = reinterpret_cast<const Args*>(args);
    fprintf(stderr, "%s: data_type=%d out=%016lx num_elems=%lu\n", __FUNCTION__, p->data_type, p->out, p->num_elems);

    const T* in = (T*)p->in;
    T* out = (T*)p->out;

    std::cerr << __FUNCTION__ << ": value=" << *in << std::endl;

    for (size_t i = 0; i < p->num_elems; ++i)
      out[i] = *in;

    fprintf(stderr, "%s: done\n", __PRETTY_FUNCTION__);
    return 0;
  }
}

#define DATA_TYPE(p) *(int*)(p)

int ops_fill(const void* args, size_t len)
{
  fprintf(stderr, "%s\n", __FUNCTION__);
  int dtype = DATA_TYPE(args);
  fprintf(stderr, "%s: dtype=%d\n", __FUNCTION__, dtype);

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
}
};

int ops_AddN(const void* args, size_t len)
{
  std::cerr << __FUNCTION__ << std::endl;
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

  if (p->output_type == DT_FLOAT) {
    AddNOp<float>((float*)p-> out, (float**)p->in, p->num_elems, p->num_inputs);
  } else {
    return 1;
  }

  return 0;
}
