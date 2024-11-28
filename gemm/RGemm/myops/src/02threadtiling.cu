# include <cuda_runtime.h>
# include <stdio.h>


#define CEIL(x, y) (((x)+(y-1)) / (y))

namespace RGemm{
    template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int TM, int TN>
    __global__ void sgemm_kernel(const float* A, const float *B, float* C, 
                            int M, int N, int K, 
                            float alpha, float beta){
        const uint inner_x = threadIdx.x;
        const uint inner_y = threadIdx.y;
        const uint outer_x = blockDim.x * blockIdx.x * TN;
        const uint outer_y = blockDim.y * blockIdx.y * TM;

        __shared__ float sA[BLOCK_M*TM*BLOCK_K];
        __shared__ float sB[BLOCK_K*BLOCK_N*TN];

        float acc[TM][TN] = {0.0f};

        // tiling K
        for(int kbid = 0; kbid < K; kbid += BLOCK_K){
            // load data 
            
            // tiling TM
            for (int i = 0; i < TM; i++){
                for (int k=inner_x; k < BLOCK_K; k += BLOCK_N){
                    sA[(inner_y*TM+i)*BLOCK_K + k] = A[(outer_y+inner_y*TM+i)*K + kbid+k];
                }
            }

            for (int i = 0; i < TN; i++){
                for (int k=inner_y; k < BLOCK_K; k+=BLOCK_M){
                   sB[k*BLOCK_N*TN + inner_x*TN+i] = B[(kbid+k)*N + outer_x+inner_x*TN+i];
                }
            }

            __syncthreads();

            // compute
            for(int i = 0; i < TM; i++){
                for(int j = 0; j < TN; j++)
                    for(int k = 0; k < BLOCK_K; k++)
                        acc[i][j] += sA[(inner_y*TM+i)*BLOCK_K + k] * sB[k*BLOCK_N*TN + inner_x*TN+j];
            }
        }

        for(int i = 0; i < TM; i++){
            for(int j = 0; j < TN; j++){
                C[(outer_y + inner_y*TM + i)*N + outer_x + inner_x*TN + j] = acc[i][j] * alpha + C[(outer_y + inner_y*TM + i)*N + outer_x + inner_x*TN + j] * beta;
            }
        }
    
    }

    void sgemm02(const float* A, const float *B, float *C, 
                int M, int N, int K, 
                float alpha, float beta){
        const int BLOCK_M = 16;
        const int BLOCK_N = 16;
        const int BLOCK_K = 16;
        const int TM = 4;
        const int TN = 4;

        dim3 BLOCKDIM(BLOCK_N, BLOCK_M, 1);
        dim3 GRIDDIM(CEIL(N, (BLOCK_N*TN)), CEIL(M, (BLOCK_M*TM)), 1);
        sgemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, TM, TN><<<GRIDDIM, BLOCKDIM>>>(A, B, C, M, N, K, alpha, beta); 
    }
} 
