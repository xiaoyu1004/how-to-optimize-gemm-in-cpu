#include "common.h"

void cblas_sgemm(const int m, const int n, const int k,
                 const float *A, const int lda,
                 const float *B, const int ldb,
                 float *C, const int ldc)
{
    int i = 0;
    int partM = m - m % SIMD_REGISTERS;
    for (; i < partM; i += SIMD_REGISTERS)
    {
        int j = 0;
        int partN = n - n % 4;
        for (; j < partN; j += 4)
        {
            __m128 re[SIMD_REGISTERS] = {_mm_setzero_ps()};
            
            for (int p = 0; p < k; ++p)
            {
                const __m128 b_v = _mm_load_ps(B + p * ldb + j);

                for (int r = 0; r < SIMD_REGISTERS; ++r)
                {
                    __m128 a_v = _mm_load_ps1(A + (i + r) * lda + p);
                    re[r] = _mm_fmadd_ps(a_v, b_v, re[r]);
                }
            }

            for (int r = 0; r < SIMD_REGISTERS; ++r)
            {
                __m128 c_v = _mm_load_ps(C + (i + r) * ldc + j);
                _mm_store_ps(C + (i + r) * ldc + j, _mm_add_ps(re[r], c_v));
            }
        }
        for (; j < n; ++j)
        {
            float v0 = 0.f;
            float v1 = 0.f;
            float v2 = 0.f;
            float v3 = 0.f;

            for (int p = 0; p < k; ++p)
            {
                float b_v = B[p * ldb + j];

                v0 += A[i * lda + p] * b_v;
                v1 += A[(i + 1) * lda + p] * b_v;
                v2 += A[(i + 2) * lda + p] * b_v;
                v3 += A[(i + 3) * lda + p] * b_v;
            }

            C[i * ldc + j] += v0;
            C[(i + 1) * ldc + j] += v1;
            C[(i + 2) * ldc + j] += v2;
            C[(i + 3) * ldc + j] += v3;
        }
    }

    for (; i < m; ++i)
    {
        int j = 0;
        int partN = n - n % 4;

        for (; j < partN; j += 4)
        {
            float v0 = 0.f;
            float v1 = 0.f;
            float v2 = 0.f;
            float v3 = 0.f;

            for (int p = 0; p < k; ++p)
            {
                float a_v = A[i * lda + p];

                v0 += a_v * B[p * ldb + j];
                v1 += a_v * B[p * ldb + j + 1];
                v2 += a_v * B[p * ldb + j + 2];
                v3 += a_v * B[p * ldb + j + 3];
            }

            C[i * ldc + j] += v0;
            C[i * ldc + j + 1] += v1;
            C[i * ldc + j + 2] += v2;
            C[i * ldc + j + 3] += v3;
        }
        for (; j < n; ++j)
        {
            for (int p = 0; p < k; ++p)
            {
                C[i * ldc + j] += A[i * lda + p] * B[p * ldb + j];
            }
        }
    }
}