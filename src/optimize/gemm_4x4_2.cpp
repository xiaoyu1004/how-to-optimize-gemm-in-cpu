void cblas_sgemm(const int m, const int n, const int k,
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
            float v00 = 0.f;
            float v01 = 0.f;
            float v02 = 0.f;
            float v03 = 0.f;

            float v10 = 0.f;
            float v11 = 0.f;
            float v12 = 0.f;
            float v13 = 0.f;

            float v20 = 0.f;
            float v21 = 0.f;
            float v22 = 0.f;
            float v23 = 0.f;

            float v30 = 0.f;
            float v31 = 0.f;
            float v32 = 0.f;
            float v33 = 0.f;

            for (int p = 0; p < k; ++p)
            {
                float a_v00 = A[i * lda + p];
                float a_v10 = A[(i + 1) * lda + p];
                float a_v20 = A[(i + 2) * lda + p];
                float a_v30 = A[(i + 3) * lda + p];

                float b_v00 = B[p * ldb + j];
                float b_v01 = B[p * ldb + j + 1];
                float b_v02 = B[p * ldb + j + 2];
                float b_v03 = B[p * ldb + j + 3];

                // c row 0
                v00 += a_v00 * b_v00;
                v01 += a_v00 * b_v01;
                v02 += a_v00 * b_v02;
                v03 += a_v00 * b_v03;

                // c row 1
                v10 += a_v10 * b_v00;
                v11 += a_v10 * b_v01;
                v12 += a_v10 * b_v02;
                v13 += a_v10 * b_v03;

                // c row 2
                v20 += a_v20 * b_v00;
                v21 += a_v20 * b_v01;
                v22 += a_v20 * b_v02;
                v23 += a_v20 * b_v03;

                // c row 3
                v30 += a_v30 * b_v00;
                v31 += a_v30 * b_v01;
                v32 += a_v30 * b_v02;
                v33 += a_v30 * b_v03;
            }

            C[i * ldc + j] += v00;
            C[i * ldc + j + 1] += v01;
            C[i * ldc + j + 2] += v02;
            C[i * ldc + j + 3] += v03;

            C[(i + 1) * ldc + j] += v10;
            C[(i + 1) * ldc + j + 1] += v11;
            C[(i + 1) * ldc + j + 2] += v12;
            C[(i + 1) * ldc + j + 3] += v13;

            C[(i + 2) * ldc + j] += v20;
            C[(i + 2) * ldc + j + 1] += v21;
            C[(i + 2) * ldc + j + 2] += v22;
            C[(i + 2) * ldc + j + 3] += v23;

            C[(i + 3) * ldc + j] += v30;
            C[(i + 3) * ldc + j + 1] += v31;
            C[(i + 3) * ldc + j + 2] += v32;
            C[(i + 3) * ldc + j + 3] += v33;
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