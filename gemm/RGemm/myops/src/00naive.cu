#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL(x, y) (((x)+(y-1)) / (y))

namespace RGemm{
    template<int BLOCK_SIZE>
    __global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta){
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= N || y >= M) return;  
        
        float acc = 0.0;
        for (int i = 0; i < K; i++){
            acc += A[y*K + i] * B[i*N + x];
        }

        C[y*K+x] = acc * alpha + beta * C[y*K+x];
    }

    // launch kernel
    void naive_sgemm(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta){
        constexpr int BLOCK_SIZE = 32;
        dim3 BLOCKDIM(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 GRIDDIM(CEIL(M, BLOCK_SIZE), CEIL(N, BLOCK_SIZE), 1);
        sgemm_kernel<BLOCK_SIZE><<<GRIDDIM, BLOCKDIM>>>(A, B, C, M, N, K, alpha, beta);
    }
}