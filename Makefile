CC=gcc
ISPC=ispc
CFLAGS=-std=c99 -O2 -march=native
ISPCFLAGS=-O2 --arch=x86-64 --target=avx

all: jacobian_sse jacobian_avx jacobian_ispc

jacobian_sse: jacobian_sse.c Makefile
	$(CC) $(CFLAGS) -msse4.1 time.c jacobian_sse.c -o jacobian_sse

jacobian_avx: jacobian_avx.c Makefile
	$(CC) $(CFLAGS) -mavx time.c jacobian_avx.c -o jacobian_avx

jacobian_ispc: routine_ispc.o jacobian_ispc.c jacobian_double.c Makefile
	$(CC) $(CFLAGS) -mavx time.c jacobian_ispc.c jacobian_double.c routine_ispc.o -o jacobian_ispc

routine_ispc.o: routine.ispc
	$(ISPC) $(ISPCFLAGS) routine.ispc -o routine_ispc.o -h routine_ispc.h

clean:
	rm -rf jacobian_avx jacobian_sse jacobian_ispc *.o
