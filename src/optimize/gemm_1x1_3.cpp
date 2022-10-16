void cblas_sgemm(const int m, const int n, const int k,
                 const float *A, const int lda,
                 const float *B, const int ldb,
                 float *C, const int ldc)
{
    // n-k-m
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < m; ++i)
        {
            for (int l = 0; l < k; ++l)
            {
                C[i * ldc + j] += A[i * lda + l] * B[l * ldb + j];
            }
        }
    }
}