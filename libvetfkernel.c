#include <stdio.h>
#include <stdint.h>

#if 0
int64_t buffer = 0xdeadbeefdeadbeef;

uint64_t hello(int i)
{
  printf("Hello, %d\n", i);
  fflush(stdout);
  return i + 1;
}

uint64_t print_buffer()
{
  printf("0x%016lx\n", buffer);
  fflush(stdout);
  return 1;
}
#endif

int conv2d(uint64_t addr, uint64_t len)
{
    void* p = (void*)addr;
    printf("libvetfkernel: conv2d: p=%p len=%lu\n", p, len);
    const char* tmp = "hello VE";
    memcpy(p, tmp, 9);
    return 1215;
}
