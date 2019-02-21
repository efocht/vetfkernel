#include <cstdint>
#include "asl.h"
#include "kernel.h"
#include "types.h"
#include "log.h"
#include <sstream>

#define DEFINE_KERNEL(NAME, FUNC) \
  REGISTER_KERNEL(#NAME, "op_" # NAME); \
  extern "C" { \
    int op_##NAME(const void* args, size_t len) { \
      return op_Kernel(args, len, FUNC, "op_" # NAME); \
    } \
  }

REGISTER_KERNEL("Select", "op_Select");
REGISTER_KERNEL("RandomUniform", "op_RandomUniform");

extern "C" {
  int op_Select(const void* args, size_t len);
  int op_RandomUniform(const void* args, size_t len);
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
#if 0
  //fprintf(stderr, "%s: ninputs=%d noutputs=%d\n", __FUNCTION__, args.ninputs(), args.noutputs());
  if (args.ninputs() != 3 && args.noutputs() != 1)
    return 1;

#if 0
  fprintf(stderr, "%s: input(0).dtype=%d\n", __FUNCTION__, args.input(0).dtype);
  fprintf(stderr, "%s: input(1).dtype=%d\n", __FUNCTION__, args.input(1).dtype);
  fprintf(stderr, "%s: input(2).dtype=%d\n", __FUNCTION__, args.input(2).dtype);
  fprintf(stderr, "%s: output(0).dtype=%d\n", __FUNCTION__, args.output(0).dtype);
#endif

  if (args.input(0).dtype == DT_BOOL
      && args.input(1).dtype == DT_FLOAT
      && args.input(2).dtype == DT_FLOAT
      && args.output(0).dtype == DT_FLOAT) {
    if (args.input(0).nelems == args.input(1).nelems
        && args.input(0).nelems == args.input(2).nelems) {
      return op_select_nn<float>(args.output(0).addr,
                                 args.input(0).addr,
                                 args.input(1).addr,
                                 args.input(2).addr,
                                 args.input(0).nelems);
    }
  }

#if 0
  fprintf(stderr, "%s: return 1\n", __FUNCTION__);
#endif
#else
  //fprintf(stderr, "%s: ninputs=%d noutputs=%d\n", __FUNCTION__, args.ninputs(), args.noutputs());
  if (args.nTensors() != 4)
    return 1;

#if 0
  fprintf(stderr, "%s: input(0).dtype=%d\n", __FUNCTION__, args.input(0).dtype);
  fprintf(stderr, "%s: input(1).dtype=%d\n", __FUNCTION__, args.input(1).dtype);
  fprintf(stderr, "%s: input(2).dtype=%d\n", __FUNCTION__, args.input(2).dtype);
  fprintf(stderr, "%s: output(0).dtype=%d\n", __FUNCTION__, args.output(0).dtype);
#endif

  if (args.tensor(0)->dtype == DT_BOOL
      && args.tensor(1)->dtype == DT_FLOAT
      && args.tensor(2)->dtype == DT_FLOAT
      && args.tensor(3)->dtype == DT_FLOAT) {
    if (args.tensor(0)->nelems == args.tensor(1)->nelems
        && args.tensor(0)->nelems == args.tensor(2)->nelems) {
      return op_select_nn<float>(args.tensor(3)->addr,
                                 args.tensor(0)->addr,
                                 args.tensor(1)->addr,
                                 args.tensor(2)->addr,
                                 args.tensor(0)->nelems);
    }
  }

#if 0
  fprintf(stderr, "%s: return 1\n", __FUNCTION__);
#endif
#endif
  return 1;
}

int op_randomUniform(const VEOpArgs& args)
{
  if (args.nTensors() != 1)
    return 1;

  const Tensor* t = args.tensor(0);

  LOG(3) << "op_RandomUniform: nelems=" << t->nelems;

  if (t->dtype == DT_FLOAT) {
    float* p = reinterpret_cast<float*>(t->addr);
    ASL::getRandom(t->nelems, p) ;
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

//
// Cast
//

namespace {

template <typename TO, typename TI>
  void cast(const Tensor* to, const Tensor* ti) {
    TO* po = reinterpret_cast<TO*>(to->addr);
    const TI* pi = reinterpret_cast<const TI*>(ti->addr);

    for (size_t i = 0; i < ti->nelems; ++i)
      po[i] = pi[i];
  }

template <typename TO>
  void cast2bool(const Tensor* to, const Tensor* ti) {
    TO* po = reinterpret_cast<TO*>(to->addr);
    const bool* pi = reinterpret_cast<const bool*>(ti->addr);

#if 0 // original ( partially vectorized ) 
    for (size_t i = 0; i < ti->nelems; ++i)
      po[i] = pi[i];
#else
  const size_t n = ti->nelems ;

  const size_t vloop_begin =  (to->addr) & 0x3 ;
  const size_t vloop_end   =  n   & 0xFFFFFFFFFFFFFFFC ;

#pragma novector
  for(size_t i=0; i < vloop_begin ; i++) {
    po[i] = pi[i] ;
  }
  
  const int*  pi_i = reinterpret_cast<const int*>(&pi[vloop_begin]);
  for(size_t j=0; j < (vloop_end - vloop_begin)>>2 ; j++) {
    const int32_t b  = pi_i[j] ;

    const int32_t b0 =   b        & 0xFF ; 
    const int32_t b1 = ( b >>  8) & 0xFF ;
    const int32_t b2 = ( b >> 16) & 0xFF ;
    const int32_t b3 = ( b >> 24)        ;

    po[vloop_begin+4*j+0] = b0 ;
    po[vloop_begin+4*j+1] = b1 ;
    po[vloop_begin+4*j+2] = b2 ;
    po[vloop_begin+4*j+3] = b3 ;
  }


#pragma novector
  for(size_t i=vloop_end; i < n ; i++) {
    po[i] = pi[i] ;
  }
#endif
  }

int op_cast(const VEOpArgs& args)
{
  if (args.nTensors() != 2)
    return 1;
  const Tensor* ti = args.tensor(0);
  const Tensor* to = args.tensor(1);

  LOG(3) << __FUNCTION__ << " ti=" << ti << " to=" << to;

  if (!ti || !to)
    return 1;

  LOG(3) << __FUNCTION__ << " ti=" << ti->to_s() << " to=" << to->to_s();

  if (ti->nelems != to->nelems)
    return 1;

  if (ti->dtype == DT_BOOL && to->dtype == DT_FLOAT) {
    cast2bool<float>(to, ti);
  } else if (ti->dtype == DT_INT32 && to->dtype == DT_FLOAT) {
    cast<float, int32_t>(to, ti);
  } else if (ti->dtype == DT_BOOL && to->dtype == DT_INT32) {
    cast2bool<int32_t>(to, ti);
  } else if (ti->dtype == DT_UINT16 && to->dtype == DT_INT32) {
    cast<int32_t, uint16_t>(to, ti);
  } else if (ti->dtype == DT_INT8 && to->dtype == DT_BOOL) {
    cast<bool, int8_t>(to, ti);
  } else {
    return 1;
  }

  return 0;
}

} // namespace

DEFINE_KERNEL(Cast, op_cast);

//
// Tile
//

namespace {
int op_tile(const VEOpArgs& args)
{
  if (args.nTensors() != 2)
    return 1;
  const Tensor* ti = args.tensor(0);
  const Tensor* to = args.tensor(1);

  LOG(3) << __FUNCTION__ 
    << " ti=" << ti->to_s()
    << " to=" << to->to_s();

  if (ti->dtype == DT_FLOAT && to->dtype == DT_FLOAT) {
    const float* pi = reinterpret_cast<const float*>(ti->addr);
    float* po = reinterpret_cast<float*>(to->addr);
    if (ti->dims == 1 && to->dims == 1 && ti->nelems == 1) {
      for (size_t i = 0; i < to->nelems; ++i) {
        po[i] = pi[0];
      }
    } else if (ti->dims == 2 && to->dims == 2
               && ti->dim_size[0] == to->dim_size[0]
               && ti->dim_size[1] == 1) {
      for (size_t i = 0; i < ti->dim_size[0]; ++i) {
        for (size_t j = 0; j < to->dim_size[1]; ++j) {
          po[i * to->dim_size[1] + j] = pi[i];
        }
      }
    } else 
      return 1;
  } else {
    return 1;
  }

  return 0;
}
} // namespace

DEFINE_KERNEL(Tile, op_tile);
