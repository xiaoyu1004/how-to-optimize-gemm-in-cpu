#ifndef COMMON_H
#define COMMON_H

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