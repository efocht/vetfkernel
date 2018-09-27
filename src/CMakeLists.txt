cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER /opt/nec/ve/bin/ncc)
set(CMAKE_CXX_COMPILER /opt/nec/ve/bin/nc++)

set(CMAKE_CXX_FLAGS "-O4")

#set(CMAKE_CXX_FLAGS "-D_DEBUG")

set(VEDNN_PATH ${CMAKE_SOURCE_DIR}/vednn CACHE PATH "path to vednn")

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
        $<TARGET_OBJECTS:vetfkernel_intrinsic>)

target_compile_options(vetfkernel PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)

target_include_directories(vetfkernel PUBLIC ${VEDNN_PATH})
target_include_directories(vetfkernel PRIVATE /opt/nec/ve/nlc/1.0.0/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/intrinsic)

target_link_libraries(vetfkernel ${vednn})
