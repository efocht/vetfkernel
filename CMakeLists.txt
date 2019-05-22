cmake_minimum_required(VERSION 3.0)

set(NCC "/opt/nec/ve/bin/ncc" CACHE FILEPATH "Path of ncc")
set(NCXX "/opt/nec/ve/bin/nc++" CACHE FILEPATH "Path of nc++")

find_package(LLVM REQUIRED CONFIG)

set(BLAS /opt/nec/ve/nlc/1.0.0/lib/libblas_sequential.a)
set(ASL /opt/nec/ve/nlc/1.0.0/lib/libasl_sequential.a)

message("llvm: ${LLVM_INSTALL_PREFIX}")
message("ncc:  ${NCC}")
message("nc++: ${NCXX}")
message("blas: ${BLAS}")
message("asl:  ${ASL}")

add_subdirectory(libs/vednn)
add_subdirectory(src/intrinsic)
add_subdirectory(src)

set(CLANG_RUNTIME ${LLVM_INSTALL_PREFIX}/lib/clang/9.0.0/lib/linux/libclang_rt.builtins-ve.a)

add_custom_target(veorun_tf ALL
        COMMAND ${CMAKE_COMMAND} -E env CC=${NCC} CXX=${NCXX} CFLAGS="-no-proginf" ${CMAKE_SOURCE_DIR}/mk_veorun_static veorun_tf src/libvetfkernel.a -- ${BLAS} ${ASL} ${CLANG_RUNTIME} libs/vednn/src/libvednn_openmp.a -fopenmp -lveio -pthread -ldl
  DEPENDS vednn_openmp vetfkernel)

# for make clean
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES veorun_tf)
