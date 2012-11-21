CC=gcc

all: jacobian_sse jacobian_avx

jacobian_sse: jacobian_sse.c Makefile
	$(CC) -std=c99 -O2 -march=native -msse4.1 time.c jacobian_sse.c -o jacobian_sse

jacobian_avx: jacobian_avx.c Makefile
	$(CC) -std=c99 -O2 -march=native -mavx time.c jacobian_avx.c -o jacobian_avx

clean:
	rm -rf jacobian_avx jacobian_sse *.o
