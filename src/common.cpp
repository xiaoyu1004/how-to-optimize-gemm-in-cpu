#include "common.h"

#include <random>
#include <iostream>

std::random_device rd;
std::default_random_engine r_eng(rd());
std::uniform_real_distribution<float> dis(-1, 1);

void InitMatrix(float *p, int m, int ld)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < ld; ++j)
        {
            p[i * ld + j] = dis(r_eng);
        }
    }
}

void PrintMatrix(float *p, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << p[i * n + j] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << std::flush;
}

void CopyMatrix(const int m, const int n,
                const float *src, const int lda,
                float *dst, const int ldb)
{
    for (int i = 0; i < m; ++i)
    {
        std::copy(src + i * lda, src + i * lda + n, dst + i * ldb);
    }
}

bool CompareResult(const int m, const int n, const int k,
                   const float *A, const int lda,
                   const float *B, const int ldb,
                   float *C, const int ldc,
                   const float error = 1e-4f)
{
    
}