#ifdef __ve__
static inline unsigned long long __veperf_get_stm() {
        void *vehva = (void *)0x1000;
        unsigned long long val;
        asm volatile ("lhm.l %0,0(%1)":"=r"(val):"r"(vehva));
        return val;
}

static double second() 
{
  return __veperf_get_stm() / 800e6;
}
#endif

