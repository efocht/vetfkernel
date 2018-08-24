cmake_minimum_required(VERSION 3.0)

set(CMAKE_C_COMPILER /opt/nec/ve/bin/ncc)
set(CMAKE_CXX_COMPILER /opt/nec/ve/bin/nc++)

set(VEDNN_PATH ${CMAKE_SOURCE_DIR}/vednn)

add_library(vetfkernel SHARED 
	src/libvetfkernel.cc
	src/conv2d.cc)

target_include_directories(vetfkernel PUBLIC ${VEDNN_PATH})

find_library(vednn libvednn.a ${VEDNN_PATH})
target_link_libraries(vetfkernel ${vednn})

add_subdirectory(fake)
