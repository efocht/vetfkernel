#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "kernel.h"
#include "types.h"
#include "log.h"

#include <omp.h>

REGISTER_KERNEL("ApplyAdam", "op_ApplyAdam");

#define CHECK_ARG_LEN(l0, l1) \
  if ((l0) != (l1)) { \
      fprintf(stderr, "%s: illegal argument length: %ld expected but %ld\n", (l1), (l0)); \
      return 1; \
  }

extern "C" {
  int op_ApplyAdam(const void* arg, size_t len);
}


//
// ApplyAdam
//

namespace {

template <typename T>
int apply_adam(bool use_nesterov, int64_t num_elements,
               uint64_t var_ptr, uint64_t m_ptr, uint64_t v_ptr,
               uint64_t beta1_power_ptr, uint64_t beta2_power_ptr,
               uint64_t lr_ptr,
               uint64_t beta1_ptr, uint64_t beta2_ptr, uint64_t epsilon_ptr,
               uint64_t grd_ptr )
{
  T* var = reinterpret_cast<T*>(var_ptr);
  T* m   = reinterpret_cast<T*>(m_ptr);
  T* v   = reinterpret_cast<T*>(v_ptr);

  const T* grd = reinterpret_cast<const T*>(grd_ptr);

  const T beta1_power = reinterpret_cast<const T*>(beta1_power_ptr)[0];
  const T beta2_power = reinterpret_cast<const T*>(beta2_power_ptr)[0];
  const T lr = reinterpret_cast<const T*>(lr_ptr)[0];
  const T beta1 = reinterpret_cast<const T*>(beta1_ptr)[0];
  const T beta2 = reinterpret_cast<const T*>(beta2_ptr)[0];
  const T epsilon = reinterpret_cast<const T*>(epsilon_ptr)[0];

  const T one = T(1.) ; 

#if 1 // optimized
 
 
  const T k = (lr * std::sqrt( one - beta2_power) / ( one - beta1_power)) ;

#pragma omp parallel
  { 
    int64_t nthreads = omp_get_num_threads() ;
    int64_t threadid = omp_get_thread_num() ;

    int64_t eachNElement = num_elements / nthreads ;
    int64_t remain       = num_elements % nthreads ;

    int64_t elementBegin = eachNElement * threadid + ( threadid < remain ? threadid : remain ) ;
    int64_t myElement    = eachNElement + ( threadid < remain ? 1 : 0 ) ;

    if( use_nesterov ) {
      for(int64_t i=elementBegin; i<elementBegin+myElement; i++) {
        m[i] = m[i] + (one - beta1) * (grd[i] - m[i]) ;
        v[i] = v[i] + (one - beta2) * (grd[i]*grd[i] - v[i]) ;
        var[i] -= k * ( m[i] * beta1 + (one-beta1) * grd[i] ) / ( epsilon + std::sqrt(v[i])) ;
      }
    }
    else {
      for(int64_t i=elementBegin; i<elementBegin+myElement; i++) {
        m[i] = m[i] + (one - beta1) * (grd[i] - m[i]) ;
        v[i] = v[i] + (one - beta2) * (grd[i]*grd[i] - v[i]) ;
        var[i] -= k * m[i] / (epsilon + std::sqrt(v[i])) ;
      }
    }
  }
#else // original
  for(int64_t i=0; i<num_elements; i++) {
    m[i] = m[i] + (one - beta1) * (grd[i] - m[i]) ;
  }
  for(int64_t i=0; i<num_elements; i++) {
    v[i] = v[i] + (one - beta2) * (grd[i]*grd[i] - v[i]) ;
  }
  
  const T k = (lr * std::sqrt( one - beta2_power) / ( one - beta1_power)) ;
  if( use_nesterov ) {
    for(int64_t i=0; i<num_elements; i++) {
      var[i] -= k * ( m[i] * beta1 + (one-beta1) * grd[i] ) / ( epsilon + std::sqrt(v[i])) ;
    }
  }
  else {
    for(int64_t i=0; i<num_elements; i++) {
      var[i] -= k * m[i] / (epsilon + std::sqrt(v[i])) ;
    }
  }
#endif

  return 0 ;
}
}

int op_ApplyAdam(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__ << " begin";

  struct Args {
    int dtype;
    bool use_nesterov_ ;
    int64_t num_elements ;
    uint64_t var_ptr, m_ptr, v_ptr ;
    uint64_t beta1_power_ptr, beta2_power_ptr ;
    uint64_t lr ;
    uint64_t beta1_ptr, beta2_ptr, epsilon_ptr ;
    uint64_t grad_ptr;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  int ret = 1;

  if (p->dtype == DT_FLOAT) {
    ret = apply_adam<float> (p->use_nesterov_, p->num_elements,
                             p->var_ptr, p->m_ptr, p->v_ptr,
                             p->beta1_power_ptr, p->beta2_power_ptr, p->lr,
                             p->beta1_ptr, p->beta2_ptr, p->epsilon_ptr,
                             p->grad_ptr ) ;
  }
  else if (p->dtype == DT_DOUBLE) {
    ret = apply_adam<double>(p->use_nesterov_, p->num_elements,
                             p->var_ptr, p->m_ptr, p->v_ptr,
                             p->beta1_power_ptr, p->beta2_power_ptr, p->lr,
                             p->beta1_ptr, p->beta2_ptr, p->epsilon_ptr,
                             p->grad_ptr ) ;
  }


  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

