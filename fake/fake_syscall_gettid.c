/*

gcc -shared -fPIC fake_syscall_gettid.c -o fakegettid.so -ldl

 *
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <stdlib.h>		/* for EXIT_FAILURE */
#include <unistd.h>		/* for _exit() */
#include <sys/syscall.h>
#include <sys/types.h>


#ifndef RTLD_NEXT
#define RTLD_NEXT      ((void *) -1l)
#endif

typedef long (*syscall_t) (long number, ...);

static void *get_libc_func(const char *funcname)
{
	void *func;
	char *error;

	func = dlsym(RTLD_NEXT, funcname);
	if ((error = dlerror()) != NULL) {
		fprintf(stderr, "I can't locate libc function `%s' error: %s",
				funcname, error);
		_exit(EXIT_FAILURE);
	}
	return func;
}

static syscall_t orig_syscall = (syscall_t)NULL;
static __thread long _fake_tid = 0L;

long syscall(long number, ...)
{
	long a[6];
	va_list ap;
	int i;

#if 0
	if (number != SYS_gettid)
		printf("syscall: %ld\n", number);
#endif
	if (orig_syscall == (syscall_t)NULL)
		orig_syscall = (syscall_t) get_libc_func("syscall");
		
	if (number == SYS_gettid) {
		if (_fake_tid != 0L)
			return _fake_tid;
	}
	/*
	 * hack to pass through args to variadic function,
	 * luckily syscall() just takes all (maximum) 6
	 * args as 64 bit integers.
	 */
	va_start(ap, number);
	for (i = 0; i < 6; i++)
		a[i] = va_arg(ap, long);
	va_end(ap);
	return orig_syscall(number, a[0], a[1], a[2], a[3], a[4], a[5]);
}

void set_fake_tid(long tid)
{
	_fake_tid = tid;
}

#if 0
int main(int argc, char *argv[])
{
	printf("orig_syscall: %p\n", (void *)*orig_syscall);

	printf("syscall(SYS_gettid) = %ld\n", syscall(SYS_gettid));
	printf("... setting _fake_tid to 17 ...");
	set_fake_tid(17);
	printf(" done\n");
	printf("syscall(SYS_gettid) = %ld\n", syscall(SYS_gettid));
	printf("... setting _fake_tid to 0\n");
	set_fake_tid(0);
	printf("syscall(SYS_gettid) = %ld\n", syscall(SYS_gettid));

	return 0;
}
#endif
