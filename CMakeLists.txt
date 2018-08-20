cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER /opt/nec/ve/bin/ncc)
set(CMAKE_CXX_COMPILER /opt/nec/ve/bin/nc++)

add_library(vetfkernel SHARED 
        src/libvetfkernel.cc
        src/conv2d.cc)


add_subdirectory(fake)
