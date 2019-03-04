#ifndef VE_OPS_COMMON_H_
#define VE_OPS_COMMON_H_

#include <cstdint>
#include "log.h"
#include "kernel.h"

#define DEFINE_KERNEL(NAME, FUNC) \
  REGISTER_KERNEL(#NAME, "op_" # NAME); \
  extern "C" { \
    int op_##NAME(const void* args, size_t len) { \
      return op_Kernel(args, len, FUNC, "op_" # NAME); \
    } \
  }

namespace {

struct Tensor {
  int32_t dtype;
  uint64_t addr;
  int32_t dims;
  int64_t nelems;
  int64_t dim_size[1];

  size_t size() const {
    return sizeof(Tensor) + sizeof(int64_t) * (dims - 1);
  }

  std::string to_s() const {
    std::stringstream s;

    s << "[dtype=" << dtype
      << ",dims=" << dims
      << ",nelems=" << nelems
      << ",dim_size=[";

    for (size_t i = 0; i < dims; ++i) {
      s << dim_size[i];
      if (i < dims - 1)
        s << ",";
    }
    s << "]]";
    return s.str();
  }
} __attribute__((__packed__));

class VEOpArgs  {
  public:
    VEOpArgs(const void* buf) : buf_(buf) {
      pHeader_ = reinterpret_cast<const Header*>(buf);
      pVariable_ = reinterpret_cast<uintptr_t>(buf) + sizeof(Header);

#if 0
      fprintf(stderr, "%s: buf=%p pHeader_=%p pTensor_=%p (%d)\n", __FUNCTION__,
              buf, pHeader_, pTensor_, pTensor_ - reinterpret_cast<uintptr_t>(pHeader_));
#endif

      const int* p = reinterpret_cast<const int*>(pVariable_);
#if 0
      fprintf(stderr, "*p=%d\n", *p);
#endif

#if 0
      tensor_size_ = sizeof(Tensor) + sizeof(int64_t) * (pHeader_->max_dim_size - 1);
#endif
    }

    int64_t nVariables() const { return pHeader_->nVariables; }

    template<typename T>
    const T* arg(int i) const {
      uintptr_t p = pVariable_;
      for (int j = 0; j < i; ++j) {
        const size_t size  = *reinterpret_cast<size_t*>(p);
        p += sizeof(size_t) + size;
      }
      return reinterpret_cast<const T*>(p+sizeof(size_t));
    }

  private:
    const void* buf_;
    uintptr_t pVariable_;
    struct Header {
      int64_t nVariables;
    };
    const Header* pHeader_;

    size_t tensor_size_;
};

int op_Kernel(const void* args, size_t len,
              int (*func)(const VEOpArgs&),
              const char* name)
{
  LOG(2) << name << ": begin";
  int ret = 1;

  VEOpArgs tmp(args);

  LOG(2) << name << ": nVariable=" << tmp.nVariables();

  // TODO: check length

  if (func)
    ret = func(tmp);

  LOG(2) << name << ": end. ret=" << ret;
  return ret;
}

}



#endif /* VE_OPS_COMMON_H_ */
