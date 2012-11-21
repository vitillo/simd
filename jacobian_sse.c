#include <stdio.h>
#include <xmmintrin.h>

void start_clock();
long stop_clock();

const unsigned int N = 1000000;

float P[42] = {1.f,2.f,3.f,4.f,5.f,6.f,7.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f, \
               1.f,2.f,3.f,4.f,5.f,6.f,7.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f};

__attribute__((aligned(16))) float H0[] = {1.2f, 3.4f, 5.6f, 0.f};
__attribute__((aligned(16))) float H1[] = {1.2f, 3.4f, 5.6f, 0.f};
__attribute__((aligned(16))) float H2[] = {1.2f, 3.4f, 5.6f, 0.f};
__attribute__((aligned(16))) float A[] = {1.2f, 3.4f, 5.6f, 0.f};
__attribute__((aligned(16))) float V0[] = {1.2f, 3.4f, 5.6f, 0.f};
__attribute__((aligned(16))) float V3[] = {1.2f, 3.4f, 5.6f, 0.f};
__attribute__((aligned(16))) float V4[] = {1.2f, 3.4f, 5.6f, 0.f};
__attribute__((aligned(16))) float V6[] = {1.2f, 3.4f, 5.6f, 0.f};
__attribute__((aligned(16))) float S3_V[] = {1.2f, 1.2f, 1.2f, 0.f};
__attribute__((aligned(16))) float C_V[] = {0.33333333f, 0.33333333f, 0.33333333f, 0.f};

__attribute__((noinline)) void routineA(){
  __asm__ __volatile__ (
			"movapd (%0), %%xmm0;"
			"movapd (%0), %%xmm1;"
			"movapd (%1), %%xmm2;"
			"movapd (%1), %%xmm3;"
			"movapd (%2), %%xmm4;"
			"movapd (%2), %%xmm5;"
			"movss (%4), %%xmm6;"
			"movss (%10), %%xmm15;"
			"pshufd $0xD2, %%xmm0, %%xmm0;" //H0_201
			"pshufd $0xC9, %%xmm1, %%xmm1;" //H0_120
			"pshufd $0xD2, %%xmm2, %%xmm2;" //H1_201
			"pshufd $0xC9, %%xmm3, %%xmm3;" //H1_120
			"pshufd $0xD2, %%xmm4, %%xmm4;" //H2_201
			"pshufd $0xC9, %%xmm5, %%xmm5;" //H2_120
			"pshufd $0x0, %%xmm6, %%xmm6;" //S3
			"pshufd $0x0, %%xmm15, %%xmm15;"
			"OuterLoop:"
			"xor %%rdi, %%rdi;"
			"JacobianLoop:"
			"movups 0xC(%3, %%rdi, 0x4), %%xmm7;" //dA
			"pshufd $0xC9, %%xmm7, %%xmm8;" //dA_120
			"pshufd $0xD2, %%xmm7, %%xmm9;" //dA_201
			"mulps %%xmm0, %%xmm8;" //H0_201*dA_120
			"mulps %%xmm1, %%xmm9;" //H0_120*dA_201
			"subps %%xmm9, %%xmm8;" //d0
			"cmp $0x23, %%rdi;"
			"jnz D2;"
			"addps (%5), %%xmm8;"
			"D2:"
			"movaps %%xmm8, %%xmm9;"
			"addps %%xmm7, %%xmm9;" //d2
			"pshufd $0xC9, %%xmm9, %%xmm10;" //d2_120
			"pshufd $0xD2, %%xmm9, %%xmm11;" //d2_201
			"mulps %%xmm2, %%xmm10;" //H1_201*d2_120
			"mulps %%xmm3, %%xmm11;" //H1_120*d2_201
			"addps %%xmm7, %%xmm10;"
			"subps %%xmm11, %%xmm10;" //d3
			"jnz D3;"
		  "movaps (%9), %%xmm11;"
			"addps (%6), %%xmm10;"
      "subps %%xmm11, %%xmm10;"
			"D3:"
			"pshufd $0xC9, %%xmm10, %%xmm11;" //d3_120
			"pshufd $0xD2, %%xmm10, %%xmm12;" //d3_201
			"mulps %%xmm2, %%xmm11;" //H1_201*d3_120
			"mulps %%xmm3, %%xmm12;" //H1_120*d3_201
			"addps %%xmm7, %%xmm11;"
			"subps %%xmm12, %%xmm11;" //d4
			"jnz D5;"
			"movaps (%9), %%xmm12;"
			"addps (%7), %%xmm11;"
			"subps %%xmm12, %%xmm11;"
			"D5:"
			"movaps %%xmm11, %%xmm12;"
			"addps %%xmm11, %%xmm12;" 
			"subps %%xmm7, %%xmm12;" //d5 = d4+d4-dA
			"movups (%3, %%rdi, 0x4), %%xmm7;"
			"pshufd $0xC9, %%xmm12, %%xmm13;" //d5_120
			"pshufd $0xD2, %%xmm12, %%xmm14;" //d5_201
			"mulps %%xmm4, %%xmm13;" //H2_201*d5_120
			"mulps %%xmm5, %%xmm14;" //H2_120*d5_201
			"subps %%xmm14, %%xmm13;" //d6		
			"jnz FIN;"
			"addps (%8), %%xmm13;"
			"FIN:"
			"addps %%xmm10, %%xmm9;"
			"addps %%xmm11, %%xmm9;"
			"mulps %%xmm6, %%xmm9;"
			"addps %%xmm7, %%xmm9;"
			"movups %%xmm9, (%3, %%rdi, 0x4);" //dR += (d2+d3+d4)*S3 
			"addps %%xmm10, %%xmm8;" 
			"addps %%xmm10, %%xmm8;" 
			"addps %%xmm12, %%xmm8;" 
			"addps %%xmm13, %%xmm8;" 
			"mulps %%xmm15, %%xmm8;" 
			"movups %%xmm8, 0xC(%3, %%rdi, 0x4);" // dA = (d0+d3+d3+d5+d6)
			"add $0x7, %%edi;"
			"cmp $0x2A, %%edi;"
			"jnz JacobianLoop;"
			"sub $0x1, %11;"
			"cmp $0x0, %11;"
			"jnz OuterLoop;"
			:
			: "r"(H0), "r"(H1), "r"(H2), "r"(P), "r"(S3_V), "r"(V0), "r"(V3), "r"(V4), \
			  "r"(V6), "r"(A), "r"(C_V), "r"(N)
			: "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", \
			  "%xmm7", "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", \
			  "%xmm14", "%xmm15", "%rdi"
		       );
  return;
}

void routineI(){
  __m128 C_012 = _mm_set1_ps(0.33333333f);
  __m128 A_012 = _mm_load_ps(A);
  __m128 S3_012 = _mm_load_ps(S3_V);
  __m128 V0_012 = _mm_load_ps(V0);
  __m128 V3_012 = _mm_load_ps(V3);
  __m128 V4_012 = _mm_load_ps(V4);
  __m128 V6_012 = _mm_load_ps(V6);

  __m128 H0_012 = _mm_load_ps(H0);
  __m128 H0_201 = _mm_shuffle_ps(H0_012, H0_012, 0xD2);
  __m128 H0_120 = _mm_shuffle_ps(H0_012, H0_012, 0xC9);

  __m128 H1_012 = _mm_load_ps(H1);
  __m128 H1_201 = _mm_shuffle_ps(H1_012, H1_012, 0xD2);
  __m128 H1_120 = _mm_shuffle_ps(H1_012, H1_012, 0xC9);

  __m128 H2_012 = _mm_load_ps(H1);
  __m128 H2_201 = _mm_shuffle_ps(H2_012, H2_012, 0xD2);
  __m128 H2_120 = _mm_shuffle_ps(H2_012, H2_012, 0xC9);

  for(int j = 0; j < N; j++){
    for(int i = 0; i < 42; i+=7){
      __m128 dR = _mm_loadu_ps(&P[i]);
    
      __m128 dA = _mm_loadu_ps(&P[i + 3]);
      __m128 dA_201 = _mm_shuffle_ps(dA, dA, 0xD2);
      __m128 dA_120 = _mm_shuffle_ps(dA, dA, 0xC9);

      __m128 d0 = _mm_sub_ps(_mm_mul_ps(H0_201, dA_120), _mm_mul_ps(H0_120, dA_201));

      if(i==35){
        d0 = _mm_add_ps(d0, V0_012);
      }
    
      __m128 d2 = _mm_add_ps(d0, dA);
      __m128 d2_201 = _mm_shuffle_ps(d2, d2, 0xD2);
      __m128 d2_120 = _mm_shuffle_ps(d2, d2, 0xC9);
      
      __m128 d3 = _mm_sub_ps(_mm_add_ps(dA, _mm_mul_ps(d2_120, H1_201)), _mm_mul_ps(d2_201, H1_120));
      if(i==35){
        d3 = _mm_add_ps(d3, _mm_sub_ps(V3_012, A_012));
      }
      __m128 d3_201 = _mm_shuffle_ps(d3, d3, 0xD2);
      __m128 d3_120 = _mm_shuffle_ps(d3, d3, 0xC9);

      __m128 d4 = _mm_sub_ps(_mm_add_ps(dA, _mm_mul_ps(d3_120, H1_201)), _mm_mul_ps(d3_201, H1_120));

      if(i==35){
        d4 = _mm_add_ps(d4, _mm_sub_ps(V4_012, A_012));
      }

      __m128 d5 = _mm_sub_ps(_mm_add_ps(d4, d4), dA);
      __m128 d5_201 = _mm_shuffle_ps(d5, d5, 0xD2);
      __m128 d5_120 = _mm_shuffle_ps(d5, d5, 0xC9);

      __m128 d6 = _mm_sub_ps(_mm_mul_ps(d5_120, H2_201), _mm_mul_ps(d5_201, H2_120));

      if(i==35){
        d6 = _mm_add_ps(d6, V6_012);
      }

      _mm_storeu_ps(&P[i], _mm_add_ps(dR, _mm_mul_ps(_mm_add_ps(d2, _mm_add_ps(d3, d4)), S3_012)));
      _mm_storeu_ps(&P[i + 3], _mm_mul_ps(C_012, _mm_add_ps(d0, _mm_add_ps(d3, _mm_add_ps(d3, _mm_add_ps(d5, d6))))));
    }
  }
}

void routineU(){
  float A0 = 1.2f; float B0 = 3.4f; float C0 = 5.6f;
  float A3 = 1.2f; float B3 = 3.4f; float C3 = 5.6f;
  float A4 = 1.2f; float B4 = 3.4f; float C4 = 5.6f;
  float A6 = 1.2f; float B6 = 3.4f; float C6 = 5.6f;
  float A00 = 1.2f; float A11 = 3.4f; float A22 = 5.6f;
  float S3 = 1.2f;
  
  for(int j = 0; j < N; j++){
    for(int i=0; i<42; i+=7) {
      float* dR   = &P[i];
      float* dA   = &P[i+3];
    
      float dA0   = H0[ 2]*dA[1]-H0[ 1]*dA[2];
      float dB0   = H0[ 0]*dA[2]-H0[ 2]*dA[0];
      float dC0   = H0[ 1]*dA[0]-H0[ 0]*dA[1];

      if(i==35) {dA0+=A0; dB0+=B0; dC0+=C0;}

      float dA2   = dA0+dA[0];               
      float dB2   = dB0+dA[1];          
      float dC2   = dC0+dA[2];   
    
      float dA3   = dA[0]+dB2*H1[2]-dC2*H1[1];
      float dB3   = dA[1]+dC2*H1[0]-dA2*H1[2];
      float dC3   = dA[2]+dA2*H1[1]-dB2*H1[0];

      if(i==35) {dA3+=A3-A00; dB3+=B3-A11; dC3+=C3-A22;}

      float dA4   = dA[0]+dB3*H1[2]-dC3*H1[1];
      float dB4   = dA[1]+dC3*H1[0]-dA3*H1[2];
      float dC4   = dA[2]+dA3*H1[1]-dB3*H1[0];

      if(i==35) {dA4+=A4-A00; dB4+=B4-A11; dC4+=C4-A22;}

      float dA5   = dA4+dA4-dA[0];          
      float dB5   = dB4+dB4-dA[1];           
      float dC5   = dC4+dC4-dA[2];          

      float dA6   = dB5*H2[2]-dC5*H2[1];
      float dB6   = dC5*H2[0]-dA5*H2[2];      
      float dC6   = dA5*H2[1]-dB5*H2[0];      

      if(i==35) {dA6+=A6; dB6+=B6; dC6+=C6;}

      dR[0]+=(dA2+dA3+dA4)*S3; dA[0]=(dA0+dA3+dA3+dA5+dA6)*.33333333;      
      dR[1]+=(dB2+dB3+dB4)*S3; dA[1]=(dB0+dB3+dB3+dB5+dB6)*.33333333; 
      dR[2]+=(dC2+dC3+dC4)*S3; dA[2]=(dC0+dC3+dC3+dC5+dC6)*.33333333;
    }
  }
}

int main(){
  start_clock();
  routineA();
  printf("Optimized (assembly): %ld miliseconds\n", stop_clock());

  start_clock();
  routineI();
  printf("Optimized (intrinsics): %ld miliseconds\n", stop_clock());
  
  start_clock();
  routineU();
  printf("Unoptimized: %ld miliseconds\n", stop_clock());
  
  return 0;
}


