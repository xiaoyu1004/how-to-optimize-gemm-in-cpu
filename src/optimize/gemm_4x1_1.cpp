void cblas_sgemm(const int m, const int n, const int k,
                 const float *A, const int lda,
                 const float *B, const int ldb,
                 float *C, const int ldc)
{
    int partM = m - m % 4;
    int i = 0;
    for (; i < partM; i += 4)
    {
        for (int j = 0; j < n; ++j)
        {
            float v0 = 0.f;
            float v1 = 0.f;
            float v2 = 0.f;
            float v3 = 0.f;

            for (int p = 0; p < k; ++p)
            {
                float b_v = B[p * ldb + j];

                // row 0
                v0 += A[i * lda + p] * b_v;
                // row 1
                v1 += A[(i + 1) * lda + p] * b_v;
                // row 2
                v2 += A[(i + 2) * lda + p] * b_v;
                // row 3
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
        for (int j = 0; j < n; ++j)
        {
            for (int p = 0; p < k; ++p)
            {
                C[i * ldc + j] += A[i * lda + p] * B[p * ldb + j];
            }
        }
    }
}