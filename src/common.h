#ifndef COMMON_H
#define COMMON_H

#include <intrin.h>

#define SIMD_REGISTERS 4

#ifdef _MSC_VER_ // for MSVC
#define forceinline __forceinline
#elif defined __GNUC__ // for gcc on Linux/Apple OS X
#define forceinline __inline__ __attribute__((always_inline))
#else
#define forceinline
#endif

static forceinline __m128 mm_fmadd_ps(const __m128 a, const __m128 b, __m128 c)
{
    return _mm_add_ps(_mm_mul_ps(a, b), c);
}

static forceinline __m256 mm256_fmadd_ps(const __m256 a, const __m256 b, __m256 c)
{
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}

void InitMatrix(float *p, int m, int ld);

void PrintMatrix(float *p, int m, int n);

void CopyMatrix(const int m, const int n,
                const float *src, const int lda,
                float *dst, const int ldb);

void cblas_sgemm_naive(const int m, const int n, const int k,
                       const float *A, const int lda,
                       const float *B, const int ldb,
                       float *C, const int ldc);

float CompareResult(const int m, const int n,
                   const float *A, const int lda,
                   const float *B, const int ldb);

#endif