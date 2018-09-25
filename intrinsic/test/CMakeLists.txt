cmake_minimum_required(VERSION 3.0)


#set(CMAKE_C_COMPILER /opt/nec/ve/bin/ncc)
set(CMAKE_C_COMPILER /proj/share/local/x86_64/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/nec/ve/bin/nc++)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -target ve -O2 -fno-vectorize -fno-slp-vectorize -fno-crash-diagnostics")

SET(VEPERF_PREFIX /usr/uhome/aurora/mpc/pub/veperf/latest)

#set(CMAKE_CXX_FLAGS "-D_DEBUG")

ADD_EXECUTABLE(ve_test ./test.cpp
	../libvetfkernel_BiasAdd.c)

TARGET_LINK_LIBRARIES(ve_test 
        -L${VEPERF_PREFIX}/lib -lveperf -lsysve -ftrace)

target_include_directories(ve_test PUBLIC ../)

target_include_directories(ve_test PUBLIC
        ${VEPERF_PREFIX}/include)

