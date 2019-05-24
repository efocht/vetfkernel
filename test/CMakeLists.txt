cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER ${NCC})
set(CMAKE_CXX_COMPILER ${NCXX})

set(CMAKE_CXX_FLAGS "-O4 -fdiag-vector=0")

#set(CMAKE_CXX_FLAGS "-D_DEBUG")

set(VEDNN_PATH ${CMAKE_SOURCE_DIR}/libs/vednn CACHE PATH "path to vednn")

add_executable(test01 test01.cc)
target_link_libraries(test01 PRIVATE 
        vetfkernel
        vednn_openmp
        ${BLAS}
        ${ASL}
        ${CLANG_RUNTIME}
        -fopenmp
        -lveio
        -pthread
        -ldl)

