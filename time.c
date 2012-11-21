#include <sys/time.h>
#include <unistd.h>

static struct timeval start, end;
static long seconds, useconds;

void start_clock(){
  gettimeofday(&start, NULL);
}

long stop_clock(){
  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  useconds = end.tv_usec - start.tv_usec;
  return ((seconds) * 1000 + useconds/1000.0) + 0.5;
}