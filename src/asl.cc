#include <vector>
#include <omp.h>

#include "asl.h"
#include "log.h"

// instance of the static member
asl_random_t* ASL::rnd ;
int64_t       ASL::asl_thread_num ;

void ASL::initialize () {
  if (asl_library_initialize() != ASL_ERROR_OK) {
    fprintf(stderr, "asl_library_initialize failed\n");
    exit(1);
  }

#pragma omp parallel
  {
#pragma omp single
    {
      asl_thread_num = omp_get_num_threads() ;
    }
  }

  rnd = new asl_random_t[asl_thread_num] ;

  for( int threadid = 0 ; threadid < asl_thread_num; threadid++) {
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

  for( int threadid = 0 ; threadid < asl_thread_num; threadid++) {
    if (asl_random_destroy(rnd[threadid]) != ASL_ERROR_OK) {
      fprintf(stderr, "asl_random_destroy failed\n");
      exit(-1);
    }
  }

  if (asl_library_finalize() != ASL_ERROR_OK) {
    fprintf(stderr, "asl_library_finalize failed\n");
    exit(-1);
  }

  delete [] rnd ;
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

