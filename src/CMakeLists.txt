cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

set(CMAKE_CXX_FLAGS "-O4 -fdiag-vector=0")

#set(CMAKE_CXX_FLAGS "-D_DEBUG")

set(VEDNN_PATH ${CMAKE_SOURCE_DIR}/libs/vednn CACHE PATH "path to vednn")

add_library(vetfkernel STATIC
  asl.cc
  libvetfkernel.cc
  conv2d.cc
  conv2d_backprop_filter.cc
  conv2d_backprop_input.cc
  maxpooling.cc
  maxpooling_backprop.cc
  ops.cc
  ve_ops.cc
  binary_ops.cc
  reduction_ops.cc
  argmax.cc
  training_ops.cc
  dense_update_functor.cc
  gather_functor.cc
  segment_reduction_ops.cc
  sparse_xent_ops.cc
  revision.h
  $<TARGET_OBJECTS:vetfkernel_intrinsic>)

target_compile_options(vetfkernel PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)

target_include_directories(vetfkernel PRIVATE ${VEDNN_PATH}/src)
target_include_directories(vetfkernel PRIVATE /opt/nec/ve/nlc/1.0.0/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/intrinsic)
target_include_directories(vetfkernel PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(check_revision
        ALL
        DEPENDS revision.h.in
        BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/revision.h
        COMMAND ${CMAKE_COMMAND} -DTOPDIR=${CMAKE_SOURCE_DIR} -P ${CMAKE_SOURCE_DIR}/cmake/revision.cmake)

add_dependencies(vetfkernel check_revision)
