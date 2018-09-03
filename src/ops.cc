#include <cstdio>
#include <cstdint>
#include <iostream>
#include "kernel.h"
#include "types.h"

REGISTER_KERNEL("Fill", "ops_fill");

extern "C" {
  int ops_fill(const void* arg, size_t len);
}

namespace {
template <typename T>
  int fill(const void* args, size_t len)
  {
    fprintf(stderr, "%s\n", __PRETTY_FUNCTION__);
    struct Arg {
      int data_type;
      uint64_t in;
      uint64_t out;
      size_t num_elems;
    } const* p;

    if (len != sizeof(*p)) {
      fprintf(stderr, "%s: illegal argument lenght: %ld expected but %ld\n", sizeof(*p), len);
      return 1;
    }

    p = reinterpret_cast<const Arg*>(args);
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
