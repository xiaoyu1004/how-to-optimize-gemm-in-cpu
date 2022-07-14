void cblas_sgemm_naive(const int m, const int n, const int k,
                       const float *A, const int lda,
                       const float *B, const int ldb,
                       float *C, const int ldc)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float v = 0.f;
            for (int l = 0; l < k; ++l)
            {
                v += A[i * lda + l] * B[l * ldc + j];
            }
            C[i * ldc + j] = v;
        }
    }
}

void cblas_sgemm(const int m, const int n, const int k,
                 const float *A, const int lda,
                 const float *B, const int ldb,
                 float *C, const int ldc)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float v = 0.f;
            for (int l = 0; l < k; ++l)
            {
                v += A[i * lda + l] * B[l * ldc + j];
            }
            C[i * ldc + j] = v;
        }
    }
}