#include <cuda_runtime.h>
#include <stdio.h>

#define CEIL(x, y) (((x)+(y-1)) / (y))

namespace RGemm{
    template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
    __global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta){
        const uint outer_x = blockIdx.x * blockDim.x;
        const uint outer_y = blockIdx.y * blockDim.y;
        const uint inner_x = threadIdx.x;
        const uint inner_y = threadIdx.y;

        __shared__ float sA[BLOCK_M*BLOCK_K];
        __shared__ float sB[BLOCK_K*BLOCK_N];

        float acc = 0.0f;

        for(int bK_id = 0; bK_id < K; bK_id += BLOCK_K){
            // load data from global memory to shared memory
            for(int k = inner_x; k < BLOCK_K; k += BLOCK_N)
                sA[inner_y*BLOCK_K + k] = A[(outer_y+inner_y)*K + bK_id+k];

            for(int k = inner_y; k < BLOCK_K; k += BLOCK_M)
                sB[k*BLOCK_N + inner_x] = B[(bK_id + k)*N + outer_x+inner_x];
            
            __syncthreads();

            for(int kid = 0; kid < BLOCK_K; kid++){
                acc += sA[inner_y * BLOCK_K + kid]*sB[kid*BLOCK_N + inner_x];
            }

        }

        C[(outer_y+inner_y)*N + outer_x+inner_x] = acc * alpha + C[(outer_y+inner_y)*N + outer_x+inner_x] *beta;
    }

    // launch kernel
    void sgemm01(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta){
        constexpr int BLOCK_M = 32;
        constexpr int BLOCK_N = 32;
        constexpr int BLOCK_K = 32;

        dim3 BLOCKDIM(BLOCK_N, BLOCK_M, 1);
        dim3 GRIDDIM(CEIL(N, BLOCK_N), CEIL(M, BLOCK_M), 1);
        sgemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K><<<GRIDDIM, BLOCKDIM>>>(A, B, C, M, N, K, alpha, beta);
    }
}