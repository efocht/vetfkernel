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
      //pHeader_ = reinterpret_cast<const Header*>(buf);
      pHeader2_ = reinterpret_cast<const Header2*>(buf);
      pTensor_ = reinterpret_cast<uintptr_t>(buf) + sizeof(Header2);

#if 0
      fprintf(stderr, "%s: buf=%p pHeader_=%p pTensor_=%p (%d)\n", __FUNCTION__,
              buf, pHeader_, pTensor_, pTensor_ - reinterpret_cast<uintptr_t>(pHeader_));
#endif

      const int* p = reinterpret_cast<const int*>(pTensor_);
#if 0
      fprintf(stderr, "*p=%d\n", *p);
#endif

#if 0
      tensor_size_ = sizeof(Tensor) + sizeof(int64_t) * (pHeader_->max_dim_size - 1);
#endif
    }

    int64_t nTensors() const { return pHeader2_->nTensors; }

    const Tensor* tensor(int i) const {
      uintptr_t p = pTensor_;
      //const Tensor* p = reinterpret_cast<const Tensor*>(pTensor_);
      for (int j = 0; j < i; ++j) {
        const Tensor* t = reinterpret_cast<const Tensor*>(p);
        p += t->size();
      }
      return reinterpret_cast<const Tensor*>(p);
    }

    int max_dim_size() const { return pHeader_->max_dim_size; }
    int ninputs() const { return pHeader_->ninputs; }
    int noutputs() const { return pHeader_->noutputs; }

    const Tensor& input(int i) const {
      return *reinterpret_cast<const Tensor*>(pTensor_ + tensor_size_ * i);
    }

    const Tensor& output(int i) const {
      return *reinterpret_cast<const Tensor*>(pTensor_ + tensor_size_ * (pHeader_->ninputs + i));
    }

  private:
    const void* buf_;
    uintptr_t pTensor_;
    struct Header2 {
      int64_t nTensors;
    };
    struct Header {
      int max_dim_size;
      int ninputs;
      int noutputs;
    };
    const Header* pHeader_;
    const Header2* pHeader2_;

    size_t tensor_size_;
};

int op_Kernel(const void* args, size_t len,
              int (*func)(const VEOpArgs&),
              const char* name)
{
  LOG(2) << name << ": begin";
  int ret = 1;

  VEOpArgs tmp(args);

  LOG(2) << name << ": nTensor=" << tmp.nTensors();

  // TODO: check length

  if (func)
    ret = func(tmp);

  LOG(2) << name << ": end. ret=" << ret;
  return ret;
}

}



#endif /* VE_OPS_COMMON_H_ */
