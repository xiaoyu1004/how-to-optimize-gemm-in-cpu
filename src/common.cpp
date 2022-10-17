#include "common.h"

#include <random>
#include <iostream>

std::random_device rd;
std::default_random_engine r_eng(rd());
std::uniform_int_distribution<int> dis(-10, 10);

void InitMatrix(float *p, int m, int ld)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < ld; ++j)
        {
            p[i * ld + j] = static_cast<float>(dis(r_eng));
            // p[i * ld + j] = 1 + j + i;
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

    std::cout << std::flush << std::endl;
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

float CompareResult(const int m, const int n,
                    const float *A, const int lda,
                    const float *B, const int ldb)
{
    float diff = 0.f;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            diff += std::powf(A[i * lda + j] - B[i * ldb + j], 2);
        }
    }

    return std::sqrt(diff);
}