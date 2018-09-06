cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER /opt/nec/ve/bin/ncc)
set(CMAKE_CXX_COMPILER /opt/nec/ve/bin/nc++)

#set(CMAKE_CXX_FLAGS "-D_DEBUG")

set(VEDNN_PATH ${CMAKE_SOURCE_DIR}/vednn CACHE PATH "path to vednn")

add_library(vetfkernel SHARED 
	src/libvetfkernel.cc
	src/conv2d.cc
	src/conv2d_backprop_filter.cc
	src/conv2d_backprop_input.cc
	src/maxpooling.cc
	src/maxpooling_backprop.cc
        src/ops.cc)

target_include_directories(vetfkernel PUBLIC ${VEDNN_PATH})

find_library(vednn libvednn.so ${VEDNN_PATH})
target_link_libraries(vetfkernel ${vednn})

add_subdirectory(fake)
