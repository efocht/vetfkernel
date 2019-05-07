#include <cstdint>
#include "asl.h"
#include "types.h"
#include <sstream>
#include "ve_ops_common.h"

namespace {

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
  //fprintf(stderr, "%s: ninputs=%d noutputs=%d\n", __FUNCTION__, args.ninputs(), args.noutputs());
  if (args.nVariables() != 4)
    return 1;

  const Tensor *t0 = args.arg<Tensor>(0) ;
  const Tensor *t1 = args.arg<Tensor>(1) ;
  const Tensor *t2 = args.arg<Tensor>(2) ;
  const Tensor *t3 = args.arg<Tensor>(3) ;

#if 0
  fprintf(stderr, "%s: input(0).dtype=%d\n", __FUNCTION__, t0->dtype);
  fprintf(stderr, "%s: input(1).dtype=%d\n", __FUNCTION__, t1->dtype);
  fprintf(stderr, "%s: input(2).dtype=%d\n", __FUNCTION__, t2->dtype);
  fprintf(stderr, "%s: output(0).dtype=%d\n", __FUNCTION__, t3->dtype);
#endif

  if (t0->dtype == DT_BOOL
      && t1->dtype == DT_FLOAT
      && t2->dtype == DT_FLOAT
      && t3->dtype == DT_FLOAT) {
    if (t0->nelems == t1->nelems
        && t0->nelems == t2->nelems) {
      return op_select_nn<float>(t3->addr,
                                 t0->addr,
                                 t1->addr,
                                 t2->addr,
                                 t0->nelems);
    }
  }

#if 0
  fprintf(stderr, "%s: return 1\n", __FUNCTION__);
#endif
  return 1;
}

int op_randomUniform(const VEOpArgs& args)
{
  if (args.nVariables() != 1)
    return 1;

  const Tensor* t = args.arg<Tensor>(0);

  LOG(3) << "op_RandomUniform: nelems=" << t->nelems;

  if (t->dtype == DT_FLOAT) {
    float* p = reinterpret_cast<float*>(t->addr);
    ASL::getRandom(t->nelems, p) ;
  }

  return 0;
}

} // namespace

DEFINE_KERNEL(Select, op_select);
DEFINE_KERNEL(RandomUniform, op_randomUniform);

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
  if (args.nVariables() != 2)
    return 1;
  const Tensor* ti = args.arg<Tensor>(0);
  const Tensor* to = args.arg<Tensor>(1);

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
  if (args.nVariables() != 2)
    return 1;
  const Tensor* ti = args.arg<Tensor>(0);
  const Tensor* to = args.arg<Tensor>(1);

  LOG(3) << __FUNCTION__ 
    << " ti=" << ti->to_s()
    << " to=" << to->to_s();

//  printf("ti->dims = %ld\n", ti->dims) ;
//  for(int i=0; i<ti->dims ; i++ ) printf(" [%d] = %ld\n", i, ti->dim_size[i]) ;
//  printf("to->dims = %ld\n", to->dims) ;
//  for(int i=0; i<to->dims ; i++ ) printf(" [%d] = %ld\n", i, to->dim_size[i]) ;

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
    } else if (ti->dims == 2 && to->dims == 2
               && ti->dim_size[1] == to->dim_size[1]
               && ti->dim_size[0] == 1) {
      for (size_t i = 0; i < to->dim_size[0]; ++i) {
        for (size_t j = 0; j < to->dim_size[1]; ++j) {
          po[i * to->dim_size[1] + j] = pi[j];
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


//
// SoftmaxXentWithLogits
//

template<typename T>
int softmax_xent_with_logits_same_shape(
  int64_t logits_addr,
  int64_t labels_addr,
  int64_t scratch_addr,
  int64_t loss_addr,
  int64_t back_addr,
  size_t batch_size,
  size_t num_classes )
{
  T* logits  = reinterpret_cast<T*>(logits_addr);
  T* labels  = reinterpret_cast<T*>(labels_addr);
  T* scratch = reinterpret_cast<T*>(scratch_addr);
  T* loss    = reinterpret_cast<T*>(loss_addr);
  T* back    = reinterpret_cast<T*>(back_addr);

#if 1 /* optimized version */
  for(int64_t i=0; i<batch_size; i++) {
    T max_logits = T(0.) ;
    for(int64_t j=0; j<num_classes; j++) {
      if(max_logits < logits[i*num_classes+j]) max_logits = logits[i*num_classes+j] ;
    }

    T sum_exp_logits = T(0.) ;
    for(int64_t j=0; j<num_classes; j++) {
      const T logit = logits[i*num_classes+j] - max_logits;
      sum_exp_logits += std::exp(logit) ;
      back[i*num_classes+j] = logit ;
    }

    T l = T(0.) ;
    for(int64_t j=0; j<num_classes; j++) {
      const T logit = back[i*num_classes+j] ;
      const T label = labels[i*num_classes+j] ;

      l += label * (std::log(sum_exp_logits) - logit);
      back[i*num_classes+j] = std::exp(logit) / sum_exp_logits - label ;
    }
    loss[i] = l ;
  }
#else /* original version */
  // max_logits along classes.
  for(int64_t i=0; i<batch_size; i++) {
    T max_logits = T(0.) ;
    for(int64_t j=0; j<num_classes; j++) {
      if(max_logits < logits[i*num_classes+j]) max_logits = logits[i*num_classes+j] ;
    }
    scratch[i] = max_logits ;
  }

  // logits - max_logits.
  for(int64_t i=0; i<batch_size; i++) {
    const T max_logits = scratch[i] ;
    for(int64_t j=0; j<num_classes; j++) {
      back[i*num_classes+j] = logits[i*num_classes+j] - max_logits;
    }
  }

  // sum(exp(logits - max_logits)) along classes.
  for(int64_t i=0; i<batch_size; i++) {
    T sum_exp_logits = T(0.) ;
    for(int64_t j=0; j<num_classes; j++) {
      sum_exp_logits += std::exp(back[i*num_classes+j]) ;
    }
    scratch[i] = sum_exp_logits ;
  }


  //  sum(-labels *
  //     ((logits - max_logits) - log(sum(exp(logits - max_logits)))))
  //  along classes
  for(int64_t i=0; i<batch_size; i++) {
    const T sum_exp_logits = scratch[i] ;
    T l = T(0.) ;
    for(int64_t j=0; j<num_classes; j++) {
      l += labels[i*num_classes+j] * (std::log(sum_exp_logits) - back[i*num_classes+j]);
    }
    loss[i] = l ;
  }

  // backprop: prob - labels, where
  //   prob = exp(logits - max_logits) / sum(exp(logits - max_logits))
  for(int64_t i=0; i<batch_size; i++) {
    const T sum_exp_logits = scratch[i] ;
    for(int64_t j=0; j<num_classes; j++) {
      back[i*num_classes+j] = std::exp(back[i*num_classes+j]) / sum_exp_logits - labels[i*num_classes+j] ;
    }
  }
#endif

  return 0 ;
}

namespace {
int op_softmax_xent_with_logits(const VEOpArgs& args)
{
  if (args.nVariables() != 5)
    return 5;

  const Tensor* logits_in = args.arg<Tensor>(0);
  const Tensor* labels_in = args.arg<Tensor>(1);
  const Tensor* scratch = args.arg<Tensor>(2);
  const Tensor* loss_out = args.arg<Tensor>(3);
  const Tensor* back_out = args.arg<Tensor>(4);

  LOG(3) << __FUNCTION__
    << " logits_in=" << logits_in->to_s()
    << " labels_in=" << labels_in->to_s()
    << " scratch="   << scratch->to_s()
    << " loss_out="  << loss_out->to_s()
    << " back_out="  << back_out->to_s() ;

  if ( logits_in->dtype == DT_FLOAT
      && labels_in->dtype == DT_FLOAT
      && scratch->dtype == DT_FLOAT
      && loss_out->dtype == DT_FLOAT
      && back_out->dtype == DT_FLOAT ) {

    int r=1;

    // TODO : add other patterns (ex:n1,1n)
    if (logits_in->dims == 2 && labels_in->dims == 2
	&& logits_in->dim_size[0] == labels_in->dim_size[0]
        && logits_in->dim_size[1] == labels_in->dim_size[1] ) {
      r = softmax_xent_with_logits_same_shape<float>(
   	    logits_in->addr, labels_in->addr,
	    scratch->addr, loss_out->addr, back_out->addr,
	    logits_in->dim_size[0], labels_in->dim_size[1] ) ;
    }

    return r;
  }
  return 1;
}
} // namespace

DEFINE_KERNEL(SoftmaxXentWithLogits, op_softmax_xent_with_logits);


//
// Concat
//
template<typename T>
  int concat(uint64_t n, uint64_t dim0, uint64_t dim1, uint64_t out,
             uint64_t *ins, uint64_t *dim1s )
  {
    const T** pi = reinterpret_cast<const T**>(ins);
    const uint64_t* in_dim1s = reinterpret_cast<const uint64_t*>(dim1s);
    T* po = reinterpret_cast<T*>(out);

    for(int64_t i=0; i<dim0; i++) {
      for(int64_t j=0; j<n; j++) {
        int64_t idim1 = idim1 = in_dim1s[j] ;
        for(int64_t k=0; k<idim1; k++) {
          po[k] = pi[j][i*idim1+k] ;
        }
        po+=idim1 ;
      }
    }
    return 0 ;
  }

namespace {
int op_Concat(const VEOpArgs& args)
{
  LOG(2) << __FUNCTION__ << " begin";

  int ret=1;

  int narg = 0 ;
  const int64_t dtype              = *args.arg<int64_t>(narg++) ;
  const uint64_t n_input           = *args.arg<uint64_t>(narg++) ;
  const uint64_t outputs_flat_dim0 = *args.arg<uint64_t>(narg++) ;
  const uint64_t outputs_flat_dim1 = *args.arg<uint64_t>(narg++) ;
  const uint64_t output_ptr        = *args.arg<uint64_t>(narg++) ;


  uint64_t ins[n_input] ;
  uint64_t dim1s[n_input] ;
  for(int64_t i=0; i<n_input; i++) {
    ins[i]   = *args.arg<uint64_t>(narg++) ;
    dim1s[i] = *args.arg<uint64_t>(narg++) ;
  }

  if (dtype == DT_FLOAT) {
    ret = concat<float>(n_input, outputs_flat_dim0, outputs_flat_dim1, output_ptr, ins, dim1s ) ;
  }
  else if (dtype == DT_INT32) {
    ret = concat<int32_t>(n_input, outputs_flat_dim0, outputs_flat_dim1, output_ptr, ins, dim1s ) ;
  }
  else if (dtype == DT_DOUBLE) {
    ret = concat<double>(n_input, outputs_flat_dim0, outputs_flat_dim1, output_ptr, ins, dim1s ) ;
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;

  return ret;
}
}

DEFINE_KERNEL(Concat, op_Concat);


//
// Split
//
template<typename T>
  int split(const int64_t  num_split,
            const int64_t  prefix_dim_size,
	    const int64_t  split_dim_size,
	    const int64_t  suffix_dim_size,
	    const uint64_t input_addr,
	    uint64_t       *output_addrs)
  {
    const T* pi = reinterpret_cast<const T*>(input_addr);
    T** po = reinterpret_cast<T**>(output_addrs);

    const int64_t split_dim_output_size = split_dim_size / num_split ;

    for(int64_t i=0; i<prefix_dim_size; i++) {
      for(int64_t n=0; n<num_split; n++) {
	for(int64_t j=0; j<split_dim_output_size*suffix_dim_size; j++) {
	  po[n][(i*split_dim_output_size)*suffix_dim_size+j]
		= pi[(i*split_dim_size+(n*split_dim_output_size))*suffix_dim_size+j] ;
	}
      }
    }
    return 0 ;
  }

namespace {
int op_Split(const VEOpArgs& args)
{
  LOG(2) << __FUNCTION__ << " begin";

  int ret=1;

  int narg = 0 ;
  const int64_t num_split       = *args.arg<int64_t>(narg++) ;
  const int64_t prefix_dim_size = *args.arg<int64_t>(narg++) ;
  const int64_t split_dim_size  = *args.arg<int64_t>(narg++) ;
  const int64_t suffix_dim_size = *args.arg<int64_t>(narg++) ;

  const Tensor *input_tensor = args.arg<Tensor>(narg++) ;
  const uint64_t input_addr  = input_tensor->addr ;

  uint64_t output_addrs[num_split] ;
  for(int64_t i=0; i<num_split; i++) {
    const Tensor *result = args.arg<Tensor>(narg++) ;
    output_addrs[i] = result->addr ;
  }

  const int dtype = input_tensor->dtype ;

  if (dtype == DT_FLOAT) {
    ret = split<float>(num_split, prefix_dim_size, split_dim_size, suffix_dim_size,
                       input_addr, output_addrs) ;
  }
  else if (dtype == DT_DOUBLE) {
    ret = split<double>(num_split, prefix_dim_size, split_dim_size, suffix_dim_size,
                       input_addr, output_addrs) ;
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;

  return ret;
}
}

DEFINE_KERNEL(Split, op_Split);


//
// SplitV
//
template<typename T>
  int split_v(const int64_t  num_split,
              const int64_t  prefix_dim_size,
	      const int64_t  split_dim_size,
	      const int64_t  suffix_dim_size,
	      const int64_t  *split_sizes,
	      const uint64_t input_addr,
	      uint64_t       *output_addrs)
  {
    const T* pi = reinterpret_cast<const T*>(input_addr);
    T** po = reinterpret_cast<T**>(output_addrs);


    for(int64_t i=0; i<prefix_dim_size; i++) {
      int64_t offset = 0 ;
      for(int64_t n=0; n<num_split; n++) {
	const int64_t split_size = split_sizes[n] ;
 	for(int64_t j=0; j<split_size*suffix_dim_size; j++) {
	  po[n][i*split_size*suffix_dim_size+j]
		= pi[(i*split_dim_size+offset)*suffix_dim_size+j] ;
	}
 	offset += split_size ;
      }
    }
    return 0 ;
  }

namespace {
int op_SplitV(const VEOpArgs& args)
{
  LOG(2) << __FUNCTION__ << " begin";

  int ret=1;

  int narg = 0 ;
  const int64_t num_split       = *args.arg<int64_t>(narg++) ;
  const int64_t prefix_dim_size = *args.arg<int64_t>(narg++) ;
  const int64_t split_dim_size  = *args.arg<int64_t>(narg++) ;
  const int64_t suffix_dim_size = *args.arg<int64_t>(narg++) ;

  const Tensor *input_tensor = args.arg<Tensor>(narg++) ;
  const uint64_t input_addr  = input_tensor->addr ;

  uint64_t output_addrs[num_split] ;
  int64_t  split_sizes[num_split] ;
  for(int64_t i=0; i<num_split; i++) {
    const Tensor *result = args.arg<Tensor>(narg++) ;
    output_addrs[i] = result->addr ;
    split_sizes[i] = *args.arg<int64_t>(narg++) ;
  }

  const int dtype = input_tensor->dtype ;

  if (dtype == DT_FLOAT) {
    ret = split_v<float>(num_split, prefix_dim_size, split_dim_size, suffix_dim_size,
	                 split_sizes, input_addr, output_addrs) ;
  }
  else if (dtype == DT_DOUBLE) {
    ret = split_v<double>(num_split, prefix_dim_size, split_dim_size, suffix_dim_size,
                          split_sizes, input_addr, output_addrs) ;
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;

  return ret;
}
}

DEFINE_KERNEL(SplitV, op_SplitV);



//
// StridedSliceGrad
//
#define STRIDED_SLICE_GRAD_MAX_HANDLE_DIM 3

template<typename T>
int strided_slice1(const int64_t* begin_di,
                   const int64_t* end_di,
		   const int64_t* stride_di,
		   const Tensor*  dy_tensor,
                   Tensor* result_tensor)
{
  const T* pi = reinterpret_cast<const T*>(dy_tensor->addr);
  T* po = reinterpret_cast<T*>(result_tensor->addr);

  const int64_t d0 = result_tensor->dim_size[0] ;

  for(int64_t i=0; i<d0; i++) po[i] = T(0.) ;

  for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
    po[i0] = *pi ; pi++ ;
  }

  return 0 ;
}


template<typename T>
int strided_slice2(const int64_t* begin_di,
                   const int64_t* end_di,
		   const int64_t* stride_di,
		   const Tensor*  dy_tensor,
                   Tensor* result_tensor)
{
  const T* pi = reinterpret_cast<const T*>(dy_tensor->addr);
  T* po = reinterpret_cast<T*>(result_tensor->addr);

  const int64_t d0 = result_tensor->dim_size[0] ;
  const int64_t d1 = result_tensor->dim_size[1] ;

  for(int64_t i=0; i<d0*d1; i++) po[i] = T(0.) ;

  for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
    for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
      po[i0*d1+i1] = *pi ; pi++ ;
    }
  }

  return 0 ;
}

template<typename T>
int strided_slice3(const int64_t* begin_di,
                   const int64_t* end_di,
		   const int64_t* stride_di,
		   const Tensor*  dy_tensor,
                   Tensor* result_tensor)
{
  const T* pi = reinterpret_cast<const T*>(dy_tensor->addr);
  T* po = reinterpret_cast<T*>(result_tensor->addr);

  const int64_t d0 = result_tensor->dim_size[0] ;
  const int64_t d1 = result_tensor->dim_size[1] ;
  const int64_t d2 = result_tensor->dim_size[2] ;

  for(int64_t i=0; i<d0*d1*d2; i++) po[i] = T(0.) ;

  for(int64_t i0=begin_di[0]; i0<end_di[0] ; i0+=stride_di[0] ) {
    for(int64_t i1=begin_di[1]; i1<end_di[1] ; i1+=stride_di[1] ) {
      for(int64_t i2=begin_di[2]; i2<end_di[2] ; i2+=stride_di[2] ) {
	po[(i0*d1+i1)*d2+i2] = *pi ; pi++ ;
      }
    }
  }

  return 0 ;
}

namespace {
int op_StridedSliceGrad(const VEOpArgs& args)
{
  LOG(2) << __FUNCTION__ << " begin";

  int ret=1;

  int narg = 0 ;
  const int64_t processing_dims = *args.arg<int64_t>(narg++) ;

  const Tensor *dy_tensor     = args.arg<Tensor>(narg++) ;
  const Tensor *result_tensor = args.arg<Tensor>(narg++) ;

  int64_t begin_di[STRIDED_SLICE_GRAD_MAX_HANDLE_DIM] ;
  int64_t end_di[STRIDED_SLICE_GRAD_MAX_HANDLE_DIM] ;
  int64_t stride_di[STRIDED_SLICE_GRAD_MAX_HANDLE_DIM] ;

  for(int64_t i=0; i<processing_dims; i++) {
    begin_di[i] = *args.arg<int64_t>(narg++) ;
  }
  for(int64_t i=0; i<processing_dims; i++) {
    end_di[i] = *args.arg<int64_t>(narg++) ;
  }
  for(int64_t i=0; i<processing_dims; i++) {
    stride_di[i] = *args.arg<int64_t>(narg++) ;
  }

  const int dtype = dy_tensor->dtype ;

  if (dtype == DT_FLOAT) {
    switch(processing_dims) {
    case 1:
      ret = strided_slice1<float>(begin_di, end_di, stride_di, dy_tensor, (Tensor*)result_tensor) ;
      break ;
    case 2:
      ret = strided_slice2<float>(begin_di, end_di, stride_di, dy_tensor, (Tensor*)result_tensor) ;
      break ;
    case 3 :
      ret = strided_slice3<float>(begin_di, end_di, stride_di, dy_tensor, (Tensor*)result_tensor) ;
      break ;
    default :
      break ;
    }
  }
  else if (dtype == DT_DOUBLE) {
    switch(processing_dims) {
    case 1:
      ret = strided_slice1<double>(begin_di, end_di, stride_di, dy_tensor, (Tensor*)result_tensor) ;
      break ;
    case 2:
      ret = strided_slice2<double>(begin_di, end_di, stride_di, dy_tensor, (Tensor*)result_tensor) ;
      break ;
    case 3 :
      ret = strided_slice3<double>(begin_di, end_di, stride_di, dy_tensor, (Tensor*)result_tensor) ;
      break ;
    default :
      break ;
    }
  }

  LOG(2) << __FUNCTION__ << " end. ret=" << ret;

  return ret;
}
}
#undef STRIDED_SLICE_GRAD_MAX_HANDLE_DIM

DEFINE_KERNEL(StridedSliceGrad, op_StridedSliceGrad);
