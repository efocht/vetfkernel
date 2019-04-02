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
  $<TARGET_OBJECTS:vetfkernel_intrinsic>)

execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE VETFKERNEL_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/libs/vednn
        OUTPUT_VARIABLE VEDNN_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

target_compile_options(vetfkernel PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
target_compile_definitions(vetfkernel PRIVATE VETFKERNEL_REVISION=${VETFKERNEL_REVISION})
target_compile_definitions(vetfkernel PRIVATE VEDNN_REVISION=${VEDNN_REVISION})

target_include_directories(vetfkernel PRIVATE ${VEDNN_PATH}/src)
target_include_directories(vetfkernel PRIVATE /opt/nec/ve/nlc/1.0.0/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/intrinsic)

