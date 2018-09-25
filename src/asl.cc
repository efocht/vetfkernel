#include <asl.h>
#include "log.h"

static void asl_init() {
  fprintf(stderr, "%s\n", __FUNCTION__);
  if (asl_library_initialize() != ASL_ERROR_OK) {
    fprintf(stderr, "asl_library_initialize failed\n");
    exit(1);
  }

#if 0
  asl_random_t hnd;
  if (asl_random_create(&hnd, ASL_RANDOMMETHOD_AUTO) != ASL_ERROR_OK) {
    fprintf(stderr, "asl_random_create failed\n");
    exit(-1);

  }
#endif
}

static void asl_fini() {
  fprintf(stderr, "%s\n", __FUNCTION__);
#if 0
  if (asl_random_destroy(hnd) != ASL_ERROR_OK) {
    fprintf(stderr, "asl_random_destroy failed\n");
    exit(-1);
  }
#endif

  if (asl_library_finalize() != ASL_ERROR_OK) {
    fprintf(stderr, "asl_library_finalize failed\n");
    exit(-1);
  }
}

class InitASL
{
  public:
    InitASL() { asl_init(); }
    ~InitASL() { asl_fini(); }
} _InitASL;
