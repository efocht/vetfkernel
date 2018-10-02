cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER /proj/share/local/x86_64/llvm/bin/clang)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target ve -O3 -fno-vectorize -fno-slp-vectorize -fno-crash-diagnostics")

add_library(vetfkernel_intrinsic OBJECT
	libvetfkernel_BiasAdd.c
	libvetfkernel_BiasAddGrad.c
	libvetfkernel_Add.c
	libvetfkernel_Sub.c
	libvetfkernel_Mul.c
	libvetfkernel_Div.c)
