all:  libconv2d.so libvetfkernel.so

libvetfkernel.so:
	/opt/nec/ve/bin/nc++ -shared -fpic -o libvetfkernel.so libvetfkernel.cc -L. -lconv2d

libconv2d.so:
	/opt/nec/ve/bin/nc++ -shared -fpic -o libconv2d.so conv2d.cc

clean:
	rm -f libvetfkernel.so libconv2d.so
