#include <cstdio>
#include <cstdint>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include "kernel.h"
#include "types.h"
#include "log.h"

#include <omp.h>

REGISTER_KERNEL("SparseSoftmaxXentWithLogits", "op_SparseSoftmaxXentWithLogits");

#define CHECK_ARG_LEN(l0, l1) \
  if ((l0) != (l1)) { \
      fprintf(stderr, "%s: illegal argument length: %ld expected but %ld\n", (l1), (l0)); \
      return 1; \
  }

extern "C" {
  int op_SparseSoftmaxXentWithLogits(const void* arg, size_t len);
}


//
// SparseSoftmaxXentWithLogits
//

namespace {

template <typename T, typename Index>
int SparseSoftmaxXentWithLogits(int64_t batch_size, int64_t num_classes, 
                                uint64_t logits_ptr, uint64_t labels_ptr,
                                uint64_t scratch_ptr, uint64_t loss_ptr, uint64_t backprop_ptr) 
{
  const T* logits = reinterpret_cast<const T*>(logits_ptr);
  const Index* labels = reinterpret_cast<const Index*>(labels_ptr);
  T* scratch = reinterpret_cast<T*>(scratch_ptr);
  T* loss = reinterpret_cast<T*>(loss_ptr);
  T* backprop = reinterpret_cast<T*>(backprop_ptr);

#if 1
  if( num_classes == 2 ) {
    // vectorize  batch_loop
    for(int64_t i=0; i<batch_size; i++) {
      const T logits0 = logits[2*i+0] ; 
      const T logits1 = logits[2*i+1] ; 

      const T max_logits = logits0 > logits1 ? logits0 : logits1 ;
       
      const T backprop0 = logits0 - max_logits ;
      const T backprop1 = logits1 - max_logits ;

      const T exp_backprop0 = std::exp(backprop0) ;
      const T exp_backprop1 = std::exp(backprop1) ;

      const T sum_exp_logits = exp_backprop0 + exp_backprop1 ;

      const T log_sum_exp_logits = std::log(sum_exp_logits) ;

      const Index label = labels[i] ;

      loss[i] = label == 0 ? log_sum_exp_logits - backprop0 : log_sum_exp_logits - backprop1 ;

      backprop[2*i+0] = exp_backprop0 / sum_exp_logits - ( 0 == label ? T(1.) : T(0.) ) ;
      backprop[2*i+1] = exp_backprop1 / sum_exp_logits - ( 1 == label ? T(1.) : T(0.) ) ;
      
    }
  }
  else {
    // omp-parallelize batch loop
#pragma omp parallel for
    for(int64_t i=0; i<batch_size; i++) {
      T max_logits = T(0.) ;
      for(int64_t j=0; j<num_classes; j++) {
        if( max_logits < logits[i*num_classes+j]) {
          max_logits = logits[i*num_classes+j] ;
        }
      }
      for(int64_t j=0; j<num_classes; j++) {
        backprop[i*num_classes+j] = logits[i*num_classes+j] - max_logits ;
      }
   
      T sum_exp_logits = T(0.) ;
      for(int64_t j=0; j<num_classes; j++) {
        sum_exp_logits += std::exp(backprop[i*num_classes+j]) ;
      }

      const T log_sum_exp_logits = std::log(sum_exp_logits) ;
      const Index label = labels[i] ;
      T sum = T(0.) ; 
      for(int64_t j=0; j<num_classes; j++) {
        sum += ( j == label ? 
                     log_sum_exp_logits - backprop[i*num_classes+j] :
                     T(0.) 
               ) ;  
        backprop[i*num_classes+j] = 
          std::exp(backprop[i*num_classes+j]) / sum_exp_logits
            - ( j == label ? T(1.) : T(0.) ) ;
      }
      loss[i] = sum ;
    }
  }
#else // original
  // scratch = max_logits along classes.
  for(int64_t i=0; i<batch_size; i++) {
    T max = T(0.) ;
    for(int64_t j=0; j<num_classes; j++) {
      if( max < logits[i*num_classes+j]) {
        max = logits[i*num_classes+j] ;
      }
    }
    scratch[i] = max ;
  }

  // backprop = logits - max_logits.
  for(int64_t i=0; i<batch_size; i++) {
    const T max = scratch[i] ;
    for(int64_t j=0; j<num_classes; j++) {
      backprop[i*num_classes+j] = logits[i*num_classes+j] - max ;
    }
  }

  // scratch = sum(exp(logits - max_logits)) along classes.
  for(int64_t i=0; i<batch_size; i++) {
    T sum_exp_logits = T(0.) ;
    for(int64_t j=0; j<num_classes; j++) {
      sum_exp_logits += std::exp(backprop[i*num_classes+j]) ;
    }
    scratch[i] = sum_exp_logits ;
  }

  //  sum(-labels *
  //     ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
  //  along classes
  for(int64_t i=0; i<batch_size; i++) {
    const T log_sum_exp_logits = std::log(scratch[i]) ;
    const Index label = labels[i] ;
    T sum = T(0.) ; 
    for(int64_t j=0; j<num_classes; j++) {
      sum += ( j == label ? 
                     log_sum_exp_logits - backprop[i*num_classes+j] :
                     T(0.) 
                 ) ;  
    }
    loss[i] = sum ;
  }

  // backprop: prob - labels, where
  //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
  for(int64_t i=0; i<batch_size; i++) {
    const T sum_exp_logits = scratch[i] ;
    const Index label = labels[i] ;
    for(int64_t j=0; j<num_classes; j++) {
      backprop[i*num_classes+j] = 
        std::exp(backprop[i*num_classes+j]) / sum_exp_logits
          - ( j == label ? T(1.) : T(0.) ) ;
    }
  }
#endif

  return 0 ;
}
}

int op_SparseSoftmaxXentWithLogits(const void* args, size_t len)
{
  LOG(2) << __FUNCTION__ << " begin";

  struct Args {
    int dtype, idxtype;
    int64_t batch_size, num_classes ;
    uint64_t logits_ptr, labels_ptr ;
    uint64_t scratch_ptr, loss_ptr, backprop_ptr ;
  } const* p;

  CHECK_ARG_LEN(len, sizeof(Args));
  p = reinterpret_cast<const Args*>(args);

  int ret = 1;

  if (p->dtype == DT_FLOAT) {
    if ( p->idxtype == DT_INT32 ) {
      ret = SparseSoftmaxXentWithLogits<float, int32_t> 
              (p->batch_size, p->num_classes, 
               p->logits_ptr, p->labels_ptr,
               p->scratch_ptr, p->loss_ptr, p->backprop_ptr) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = SparseSoftmaxXentWithLogits<float, int64_t> 
              (p->batch_size, p->num_classes, 
               p->logits_ptr, p->labels_ptr,
               p->scratch_ptr, p->loss_ptr, p->backprop_ptr) ;
    }
  }
  else if (p->dtype == DT_DOUBLE) {
    if ( p->idxtype == DT_INT32 ) {
      ret = SparseSoftmaxXentWithLogits<double, int32_t> 
              (p->batch_size, p->num_classes, 
               p->logits_ptr, p->labels_ptr,
               p->scratch_ptr, p->loss_ptr, p->backprop_ptr) ;
    }
    else if ( p->idxtype == DT_INT64 ) {
      ret = SparseSoftmaxXentWithLogits<double, int64_t> 
              (p->batch_size, p->num_classes, 
               p->logits_ptr, p->labels_ptr,
               p->scratch_ptr, p->loss_ptr, p->backprop_ptr) ;
    }
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;
  return ret;
}

