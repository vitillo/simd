#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "routine_ispc.h"

void start_clock();
long stop_clock();
void routineU();

const unsigned int N = 1000000;

double P[42] = {1.,2.,3.,4.,5.,6.,7.,1.,2.,3.,4.,5.,6.,7.,1.,2.,3.,4.,5.,6.,7., \
                1.,2.,3.,4.,5.,6.,7.,1.,2.,3.,4.,5.,6.,7.,1.,2.,3.,4.,5.,6.,7.};

int main(){
  routineU(P, N); //warmup

  start_clock();
  routine_ispc(P, N);
  printf("Optimized (ispc): %ld miliseconds\n", stop_clock());

  start_clock();
  routineU(P, N);
  printf("Unoptimized: %ld miliseconds\n", stop_clock());

  return 0;
}
