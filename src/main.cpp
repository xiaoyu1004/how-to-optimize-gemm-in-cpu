#include "common.h"
#include "timer.h"

#include <iostream>
#include <fstream>

#define CONTACT(x,y) x##y
#define STR(x) #x

void cblas_sgemm(const int m, const int n, const int k,
                 const float *A, const int lda,
                 const float *B, const int ldb,
                 float *C, const int ldc);

int main()
{
    // constexpr int LDA = 1000;
    // constexpr int LDB = 1000;
    // constexpr int LDC = 1000;

    constexpr int START = 64;
    constexpr int END = 2048;
    constexpr int STRIDE = 64;

    std::ofstream fs("../profile/gemm_cache_block_packA_packB_4x4_8.txt", std::ios::out);
    fs << "N" << "\t\t" << "gflops" << "\t\t" << "error" << std::endl;

    for (int i = START; i <= END; i += STRIDE)
    {
        int m = i;
        int n = i;
        int k = i;

        // matrix A: m * k
        int lda = k;
        // matrix B: k * n
        int ldb = n;
        // matrix C: m * n
        int ldc = n;

        double flop = 2 * (double)m * n * k;

        float *A = new float[m * lda];
        float *B = new float[k * ldb];
        float *C = new float[m * ldc];
        float *ref_C = new float[m * ldc];
        float *buffer = new float[m * ldc];

        InitMatrix(A, m, lda);
        InitMatrix(B, k, ldb);
        InitMatrix(C, m, ldb);
        CopyMatrix(m, n, C, ldc, ref_C, ldc);
        CopyMatrix(m, n, C, ldc, buffer, ldc);

        // PrintMatrix(A, m, k);
        // PrintMatrix(B, k, n);

        // ref
        cblas_sgemm_naive(m, n, k, A, lda, B, ldb, ref_C, ldc);

        // PrintMatrix(ref_C, m, n);

        constexpr int warm_count = 2;
        float err = 0.f;
        for (int w = 0; w < warm_count; ++w)
        {
            CopyMatrix(m, n, buffer, ldc, C, ldc);
            cblas_sgemm(m, n, k, A, lda, B, ldb, C, ldc);
            // PrintMatrix(C, m, n);
            err = CompareResult(m, n, C, ldc, ref_C, ldc);
            if (err > 1e-4f)
            {
                std::cout << "Error: compare result is fail! error: " << err << std::endl;
            }
        }

        timer t;
        double best_time = FLT_MAX;
        constexpr int loop_count = 10;
        for (int l = 0; l < loop_count; ++l)
        {
            CopyMatrix(m, n, buffer, ldc, C, ldc);
            t.start();
            cblas_sgemm(m, n, k, A, lda, B, ldb, C, ldc);
            t.stop();
            best_time = std::min(best_time, t.get_elapsed_nano_seconds());
        }

        double gflops = flop / best_time;

        fs << i << "\t\t" << gflops << "\t\t" << err << std::endl;

        delete[] A;
        delete[] B;
        delete[] C;
        delete[] buffer;
    }

    fs.close();
    std::cout << "gemm compute finish!" << std::endl;
}

// int main()
// {
//     int m = 3, n = 3;
//     float *p = new float[m * n];
//     InitMatrix(p, m, n);

//     PrintMatrix(p, m, n);

//     return 0;
// }