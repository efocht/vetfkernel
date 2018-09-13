#ifndef TYPES_H
#define TYPES_H

// from tensorflow/core/framework/types.proto
enum DataType {
  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  DT_INVALID = 0,

  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT = 1,
  DT_DOUBLE = 2,
  DT_INT32 = 3,
  DT_UINT8 = 4,
  DT_INT16 = 5,
  DT_INT8 = 6,
  DT_STRING = 7,
  DT_COMPLEX64 = 8,  // Single-precision complex
  DT_INT64 = 9,
  DT_BOOL = 10,
  DT_QINT8 = 11,     // Quantized int8
  DT_QUINT8 = 12,    // Quantized uint8
  DT_QINT32 = 13,    // Quantized int32
  DT_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16 = 15,    // Quantized int16
  DT_QUINT16 = 16,   // Quantized uint16
  DT_UINT16 = 17,
  DT_COMPLEX128 = 18,  // Double-precision complex
  DT_HALF = 19,
  DT_RESOURCE = 20,
  DT_VARIANT = 21,  // Arbitrary C++ data types
  DT_UINT32 = 22,
  DT_UINT64 = 23,
};

// from tensorflow/core/util/tensor_format.h
enum TensorFormat {
  // FORMAT_NHWC is the default format in TensorFlow.
  FORMAT_NHWC = 0,

  // FORMAT_NCHW often improves performance on GPUs.
  FORMAT_NCHW = 1,

  // NCHW_VECT_C is the most performant tensor format for cudnn6's quantized
  // int8 convolution and fused convolution. It is laid out in the same order
  // as NCHW, except that the size of the Channels dimension is divided by 4,
  // and a new dimension of size 4 is appended, which packs 4 adjacent channel
  // activations for the same pixel into an int32. Thus an NCHW format tensor
  // with dimensions [N, C, H, W] would have dimensions [N, C/4, H, W, 4] in
  // NCHW_VECT_C format.
  // A pre-condition of this format is that C must be a multiple of 4.
  FORMAT_NCHW_VECT_C = 2,

  // Similar to NHWC, but the size of the W dimension is divided by 4, and a
  // new dimension of size 4 is appended, which packs 4 adjacent activations
  // in the width dimension.
  FORMAT_NHWC_VECT_W = 3,

  // Note: although the current code in this file assumes VECT_C and VECT_W
  // enums imply int8x4 vectors, this should not be relied upon.
  // In the future we may change the meaning of these enums to include vectors
  // of other types such as int16x2, with op implementations automatically
  // determining which format is implied based on the datatype.

  // FORMAT_HWNC is for TPUs.
  FORMAT_HWNC = 4,

  // FORMAT_HWCN is for TPUs.
  FORMAT_HWCN = 5,
};

#endif
