#include <stdio.h>
#include <immintrin.h>
#include <sys/time.h>
#include <unistd.h>

#define CROSS_SHUFFLE_201(Y1) _mm256_shuffle_pd(_mm256_permute2f128_pd(Y1, Y1, 0x01), Y1, 0xC);
#define CROSS_SHUFFLE_120(Y1) _mm256_permute_pd(_mm256_shuffle_pd(Y1, _mm256_permute2f128_pd(Y1, Y1, 0x01), 0x5), 0x6);

void start_clock();
long stop_clock();

const unsigned int N = 1000000;

double P[42] = {1.,2.,3.,4.,5.,6.,7.,1.,2.,3.,4.,5.,6.,7.,1.,2.,3.,4.,5.,6.,7., \
                1.,2.,3.,4.,5.,6.,7.,1.,2.,3.,4.,5.,6.,7.,1.,2.,3.,4.,5.,6.,7.};

__attribute__((aligned(32))) double H0[] = {1.2, 3.4, 5.6, 0.};
__attribute__((aligned(32))) double H1[] = {1.2, 3.4, 5.6, 0.};
__attribute__((aligned(32))) double H2[] = {1.2, 3.4, 5.6, 0.};
__attribute__((aligned(32))) double A[] = {1.2, 3.4, 5.6, 0.};
__attribute__((aligned(32))) double V0[] = {1.2, 3.4, 5.6, 0.};
__attribute__((aligned(32))) double V3[] = {1.2, 3.4, 5.6, 0.};
__attribute__((aligned(32))) double V4[] = {1.2, 3.4, 5.6, 0.};
__attribute__((aligned(32))) double V6[] = {1.2, 3.4, 5.6, 0.};
__attribute__((aligned(32))) double S3_V[] = {1.2, 1.2, 1.2, 0.};
__attribute__((aligned(32))) double C_V[] = {0.33333333, 0.33333333, 0.33333333, 0.};

void routineI(){
  __m256d C_012 = _mm256_set1_pd(0.33333333);
  __m256d A_012 = _mm256_load_pd(A);
  __m256d S3_012 = _mm256_load_pd(S3_V);
  __m256d V0_012 = _mm256_load_pd(V0);
  __m256d V3_012 = _mm256_load_pd(V3);
  __m256d V4_012 = _mm256_load_pd(V4);
  __m256d V6_012 = _mm256_load_pd(V6);

  __m256d H0_012 = _mm256_load_pd(H0);
  __m256d H0_201 = CROSS_SHUFFLE_201(H0_012);
  __m256d H0_120 = CROSS_SHUFFLE_120(H0_012);

  __m256d H1_012 = _mm256_load_pd(H1);
  __m256d H1_201 = CROSS_SHUFFLE_201(H1_012);
  __m256d H1_120 = CROSS_SHUFFLE_120(H1_012);

  __m256d H2_012 = _mm256_load_pd(H1);
  __m256d H2_201 = CROSS_SHUFFLE_201(H2_012);
  __m256d H2_120 = CROSS_SHUFFLE_120(H2_012);

  for(int j = 0; j < N; j++){
    for(int i = 0; i < 42; i+=7){
      __m256d dR = _mm256_loadu_pd(&P[i]);
    
      __m256d dA = _mm256_loadu_pd(&P[i + 3]);
      __m256d dA_201 = CROSS_SHUFFLE_201(dA);
      __m256d dA_120 = CROSS_SHUFFLE_120(dA);

      __m256d d0 = _mm256_sub_pd(_mm256_mul_pd(H0_201, dA_120), _mm256_mul_pd(H0_120, dA_201));

      if(i==35){
        d0 = _mm256_add_pd(d0, V0_012);
      }
    
      __m256d d2 = _mm256_add_pd(d0, dA);
      __m256d d2_201 = CROSS_SHUFFLE_201(d2);
      __m256d d2_120 = CROSS_SHUFFLE_120(d2);
      
      __m256d d3 = _mm256_sub_pd(_mm256_add_pd(dA, _mm256_mul_pd(d2_120, H1_201)), _mm256_mul_pd(d2_201, H1_120));
      if(i==35){
        d3 = _mm256_add_pd(d3, _mm256_sub_pd(V3_012, A_012));
      }
      __m256d d3_201 = CROSS_SHUFFLE_201(d3);
      __m256d d3_120 = CROSS_SHUFFLE_120(d3);

      __m256d d4 = _mm256_sub_pd(_mm256_add_pd(dA, _mm256_mul_pd(d3_120, H1_201)), _mm256_mul_pd(d3_201, H1_120));

      if(i==35){
        d4 = _mm256_add_pd(d4, _mm256_sub_pd(V4_012, A_012));
      }

      __m256d d5 = _mm256_sub_pd(_mm256_add_pd(d4, d4), dA);
      __m256d d5_201 = CROSS_SHUFFLE_201(d5);
      __m256d d5_120 = CROSS_SHUFFLE_120(d5);

      __m256d d6 = _mm256_sub_pd(_mm256_mul_pd(d5_120, H2_201), _mm256_mul_pd(d5_201, H2_120));

      if(i==35){
        d6 = _mm256_add_pd(d6, V6_012);
      }

      _mm256_storeu_pd(&P[i], _mm256_add_pd(dR, _mm256_mul_pd(_mm256_add_pd(d2, _mm256_add_pd(d3, d4)), S3_012)));
      _mm256_storeu_pd(&P[i + 3], _mm256_mul_pd(C_012, _mm256_add_pd(d0, _mm256_add_pd(d3, _mm256_add_pd(d3, _mm256_add_pd(d5, d6))))));
    }
  }
}

void routineU(){
  double A0 = 1.2; double B0 = 3.4; double C0 = 5.6;
  double A3 = 1.2; double B3 = 3.4; double C3 = 5.6;
  double A4 = 1.2; double B4 = 3.4; double C4 = 5.6;
  double A6 = 1.2; double B6 = 3.4; double C6 = 5.6;
  double A00 = 1.2; double A11 = 3.4; double A22 = 5.6;
  double S3 = 1.2;
  
  for(int j = 0; j < N; j++){
    for(int i=0; i<42; i+=7) {
      double* dR   = &P[i];
      double* dA   = &P[i+3];
    
      double dA0   = H0[ 2]*dA[1]-H0[ 1]*dA[2];
      double dB0   = H0[ 0]*dA[2]-H0[ 2]*dA[0];
      double dC0   = H0[ 1]*dA[0]-H0[ 0]*dA[1];

      if(i==35) {dA0+=A0; dB0+=B0; dC0+=C0;}

      double dA2   = dA0+dA[0];               
      double dB2   = dB0+dA[1];          
      double dC2   = dC0+dA[2];   
    
      double dA3   = dA[0]+dB2*H1[2]-dC2*H1[1];
      double dB3   = dA[1]+dC2*H1[0]-dA2*H1[2];
      double dC3   = dA[2]+dA2*H1[1]-dB2*H1[0];

      if(i==35) {dA3+=A3-A00; dB3+=B3-A11; dC3+=C3-A22;}

      double dA4   = dA[0]+dB3*H1[2]-dC3*H1[1];
      double dB4   = dA[1]+dC3*H1[0]-dA3*H1[2];
      double dC4   = dA[2]+dA3*H1[1]-dB3*H1[0];

      if(i==35) {dA4+=A4-A00; dB4+=B4-A11; dC4+=C4-A22;}

      double dA5   = dA4+dA4-dA[0];          
      double dB5   = dB4+dB4-dA[1];           
      double dC5   = dC4+dC4-dA[2];          

      double dA6   = dB5*H2[2]-dC5*H2[1];
      double dB6   = dC5*H2[0]-dA5*H2[2];      
      double dC6   = dA5*H2[1]-dB5*H2[0];      

      if(i==35) {dA6+=A6; dB6+=B6; dC6+=C6;}

      dR[0]+=(dA2+dA3+dA4)*S3; dA[0]=(dA0+dA3+dA3+dA5+dA6)*.33333333;      
      dR[1]+=(dB2+dB3+dB4)*S3; dA[1]=(dB0+dB3+dB3+dB5+dB6)*.33333333; 
      dR[2]+=(dC2+dC3+dC4)*S3; dA[2]=(dC0+dC3+dC3+dC5+dC6)*.33333333;
    }
  }
}

int main(){
  start_clock();
  routineI();
  printf("Optimized (intrinsics): %ld miliseconds\n", stop_clock());
  
  start_clock();
  routineU();
  printf("Unoptimized: %ld miliseconds\n", stop_clock());
  
  return 0;
}
