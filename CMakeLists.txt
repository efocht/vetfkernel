cmake_minimum_required(VERSION 3.0)

add_subdirectory(libs/vednn)
add_subdirectory(src/intrinsic)
add_subdirectory(src)

set(BLAS /opt/nec/ve/nlc/1.0.0/lib/libblas_sequential.a)
set(ASL /opt/nec/ve/nlc/1.0.0/lib/libasl_sequential.a)
set(CLANG_RUNTIME /proj/share/local/x86_64/llvm/lib/clang/8.0.0/lib/linux/libclang_rt.builtins-ve.a)

add_custom_target(veorun_tf ALL
  COMMAND ${CMAKE_SOURCE_DIR}/mk_veorun_static veorun_tf src/libvetfkernel.a -- ${BLAS} ${ASL} ${CLANG_RUNTIME} libs/vednn/src/libvednn_openmp.a -fopenmp
  DEPENDS vednn_openmp vetfkernel)

# for make clean
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES veorun_tf)

