#include <vector>
#include <omp.h>

#include "asl.h"
#include "log.h"

std::vector<asl_random_t> ASL::rnd ;

void ASL::initialize () {
  if (asl_library_initialize() != ASL_ERROR_OK) {
    fprintf(stderr, "asl_library_initialize failed\n");
    exit(1);
  }

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads() ;
    int threadid = omp_get_thread_num() ;

#pragma omp single
    {
       rnd.resize(nthreads) ;
    }

    if (asl_random_create(&rnd[threadid], ASL_RANDOMMETHOD_AUTO) != ASL_ERROR_OK) {
      fprintf(stderr, "asl_random_create failed\n");
      exit(-1);
    }

    const asl_uint32_t seed = time(NULL) >> threadid ;
    if (asl_random_initialize(rnd[threadid], 1, &seed) != ASL_ERROR_OK) {
      fprintf(stderr, "asl_random_initialize failed\n");
      exit(-1);
    }
  }
}

void ASL::finalize() {
#pragma omp parallel
  {
    int threadid = omp_get_thread_num() ;
    if (asl_random_destroy(rnd[threadid]) != ASL_ERROR_OK) {
      fprintf(stderr, "asl_random_destroy failed\n");
      exit(-1);
    }
  }

  if (asl_library_finalize() != ASL_ERROR_OK) {
    fprintf(stderr, "asl_library_finalize failed\n");
    exit(-1);
  }
}

int ASL::getRandom(size_t num, float *val) {

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads() ;
    int threadid = omp_get_thread_num() ;

    size_t chunkSize = num / nthreads ;
    size_t remain    = num % nthreads ;

    size_t chunkBegin = chunkSize * threadid + ( threadid < remain ? threadid : remain ) ;
    size_t myChunk    = chunkSize + ( threadid < remain ? 1 : 0 ) ;

    if( myChunk > 0 ) {
      if (asl_random_generate_s(rnd[threadid], myChunk, val+chunkBegin) != ASL_ERROR_OK) {
        fprintf(stderr, "asl_random_generate_d failed\n");
        exit(-1);
      }
    }
  }
  return 0 ;
} 

