all:
	/opt/nec/ve/bin/ncc -shared -fpic -o libvetfkernel.so libvetfkernel.c

clean:
	rm -f libvetfkernel.so
