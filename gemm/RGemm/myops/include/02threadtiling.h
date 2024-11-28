#pragma once

namespace RGemm{
    void sgemm02(const float* A, const float *B, float *C, 
                int M, int N, int K, 
                float alpha, float beta);
}