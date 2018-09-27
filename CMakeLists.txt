cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER /proj/share/local/x86_64/llvm/bin/clang)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target ve -O2 -fno-vectorize -fno-slp-vectorize -fno-crash-diagnostics")
#set(CMAKE_C_COMPILER /opt/nec/ve/bin/ncc)
set(CMAKE_CXX_COMPILER /opt/nec/ve/bin/nc++)

#set(CMAKE_CXX_FLAGS "-D_DEBUG")

set(VEDNN_PATH ${CMAKE_SOURCE_DIR}/vednn CACHE PATH "path to vednn")

add_library(vetfkernel STATIC
        src/asl.cc
	src/libvetfkernel.cc
	src/conv2d.cc
	src/conv2d_backprop_filter.cc
	src/conv2d_backprop_input.cc
	src/maxpooling.cc
	src/maxpooling_backprop.cc
        src/ops.cc
        src/ve_ops.cc
        src/binary_ops.cc
	src/intrinsic/libvetfkernel_BiasAdd.c
	src/intrinsic/libvetfkernel_BiasAddGrad.c)

target_compile_options(vetfkernel PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)

target_include_directories(vetfkernel PUBLIC ${VEDNN_PATH})
target_include_directories(vetfkernel PRIVATE /opt/nec/ve/nlc/1.0.0/include
				src/intrinsic)

target_link_libraries(vetfkernel ${vednn})
