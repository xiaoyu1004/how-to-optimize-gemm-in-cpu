#include "common.h"

#define KC 128
#define MC 8
#define NC 128

void PackMatrixA(int m, int pk, int kc, const float *from, int lda, float *to)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < pk; ++j)
        {
            to[i * kc + j] = from[i * lda + j];
        }
    }
}

void PackMatrixB(int pn, int nc, int pk, int kc, const float *from, int ldb, float *to)
{
    for (int i = 0; i < pk; ++i)
    {
        for (int j = 0; j < pn; ++j)
        {
            to[i * nc + j] = from[i * ldb + j];
        }
    }
}

void inner_gemm_kernel(const int m, const int n, const int k,
                       const float *A, const int lda,
                       const float *B, const int ldb,
                       float *C, const int ldc)
{
    int i = 0;
    int partM = m - m % 4;
    for (; i < partM; i += 4)
    {
        int j = 0;
        int partN = n - n % 4;
        for (; j < partN; j += 4)
        {
            __m128 c_0 = _mm_load_ps(C + i * ldc + j);
            __m128 c_1 = _mm_load_ps(C + (i + 1) * ldc + j);
            __m128 c_2 = _mm_load_ps(C + (i + 2) * ldc + j);
            __m128 c_3 = _mm_load_ps(C + (i + 3) * ldc + j);

            for (int p = 0; p < k; ++p)
            {
                const __m128 a_0 = _mm_load_ps1(A + i * lda + p);
                const __m128 a_1 = _mm_load_ps1(A + (i + 1) * lda + p);
                const __m128 a_2 = _mm_load_ps1(A + (i + 2) * lda + p);
                const __m128 a_3 = _mm_load_ps1(A + (i + 3) * lda + p);

                const __m128 b_v = _mm_load_ps(B + p * ldb + j);

                c_0 = mm_fmadd_ps(a_0, b_v, c_0);
                c_1 = mm_fmadd_ps(a_1, b_v, c_1);
                c_2 = mm_fmadd_ps(a_2, b_v, c_2);
                c_3 = mm_fmadd_ps(a_3, b_v, c_3);
            }

            _mm_store_ps(C + i * ldc + j, c_0);
            _mm_store_ps(C + (i + 1) * ldc + j, c_1);
            _mm_store_ps(C + (i + 2) * ldc + j, c_2);
            _mm_store_ps(C + (i + 3) * ldc + j, c_3);
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

void cblas_sgemm(const int m, const int n, const int k,
                 const float *A, const int lda,
                 const float *B, const int ldb,
                 float *C, const int ldc)
{
    float *packA = new float[m * KC]{0};
    float *packB = new float[KC * NC]{0};

    for (int p = 0; p < k; p += KC)
    {
        int inner_k = std::min(k - p, KC);
        PackMatrixA(m, inner_k, KC, A + p, lda, packA);
        for (int j = 0; j < n; j += NC)
        {
            int inner_n = std::min(n - j, NC);
            PackMatrixB(inner_n, NC, inner_k, KC, B + p * ldb + j, ldb, packB);
            for (int i = 0; i < m; i += MC)
            {
                int inner_m = std::min(m - i, MC);
                inner_gemm_kernel(inner_m, inner_n, inner_k,
                                  packA + i * KC, KC,
                                  packB, NC,
                                  C + i * ldc + j, ldc);
            }
        }
    }

    delete[] packA;
    delete[] packB;
}