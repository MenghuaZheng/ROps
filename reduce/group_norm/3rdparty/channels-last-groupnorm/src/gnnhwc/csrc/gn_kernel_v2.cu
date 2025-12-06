/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

 #include "gn_kernel_v2.h"
 #include <cub/cub.cuh>
 #include <ATen/cuda/CUDAContext.h> 
 

 template <int32_t tTHREADS_PER_BLOCK, typename Params>
 __global__ void groupNormNHWCSumKernel(Params params) {
     using DataType = typename Params::DataType;
     
     typedef cub::BlockScan<GroupSums, tTHREADS_PER_BLOCK> BlockScan;
     __shared__ typename BlockScan::TempStorage tempStorage;
     __shared__ float2 smem[tTHREADS_PER_BLOCK];
 
     int32_t ni = blockIdx.z;
     int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;
     int32_t hwBegin = blockIdx.y * params.hwPerBlock;
     int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);
 
     float sum = 0.0f;
     float sumSq = 0.0f;
 
     for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
         int64_t offset = static_cast<int64_t>(ni) * params.hwc + static_cast<int64_t>(hwi) * params.c + ci;
         float2 f2 = make_float2(0.0f, 0.0f);
 
         if (ci < params.c) {
             if constexpr (std::is_same<DataType, __half>::value) {
                 // FP16 path
                 __half2 h2 = *reinterpret_cast<const __half2*>(&params.src[offset]);
                 f2 = __half22float2(h2);
             } else if constexpr (std::is_same<DataType, double>::value) {
                 // FP64 path: Read double2, convert to float2 for accumulation
                 double2 d2 = *reinterpret_cast<const double2*>(&params.src[offset]);
                 f2.x = static_cast<float>(d2.x);
                 f2.y = static_cast<float>(d2.y);
             } else {
                 // FP32 path
                 f2 = *reinterpret_cast<const float2*>(&params.src[offset]);
             }
         }
 
         sum += f2.x + f2.y;
         sumSq += f2.x * f2.x + f2.y * f2.y;
     }
 
     int32_t gi = threadIdx.x * 2 / params.cPerGroup;  // in the i-th group (local)
     int32_t cj = threadIdx.x * 2 - params.cPerGroup * gi; 
     GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};
     GroupSums out;
     BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());
 
     if (cj == params.cPerGroup - 2) {
         smem[gi] = make_float2(out.sum, out.sumSq);
     }
     __syncthreads();
 
     int32_t gj = blockIdx.x * params.groupsPerBlock + threadIdx.x;  //
     if (threadIdx.x >= params.groupsPerBlock || gj >= params.groups) return;
 
     float2 sums = smem[threadIdx.x];
     atomicAdd(&(params.meanData[ni * params.groups + gj]), sums.x);
     atomicAdd(&(params.rstdData[ni* params.groups + gj]), sums.y);
 }
 
 // -------------------------------------------------------------------------
 // Scale Kernel
 // -------------------------------------------------------------------------
 
 template <int32_t tTHREADS_PER_BLOCK, typename Params>
 __global__ void groupNormNHWCScaleKernel(Params params) {
     using DataType = typename Params::DataType;
 
     int32_t ni = blockIdx.z;
     int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;
     int32_t gi = ci / params.cPerGroup;
 
     float sum = 0.0f, sumSq = 0.0f;
     if (gi < params.groups) {
         sum = params.meanData[ni * params.groups + gi];
         sumSq = params.rstdData[ni * params.groups + gi];
     }
 
     float2 gammaF2, betaF2;
     if (ci < params.c) {
         if constexpr (std::is_same<DataType, __half>::value) {
             __half2 gh2 = *reinterpret_cast<const __half2*>(&params.gamma[ci]);
             __half2 bh2 = *reinterpret_cast<const __half2*>(&params.beta[ci]);
             gammaF2 = __half22float2(gh2);
             betaF2 = __half22float2(bh2);
         } else if constexpr (std::is_same<DataType, double>::value) {
             double2 gh2 = *reinterpret_cast<const double2*>(&params.gamma[ci]);
             double2 bh2 = *reinterpret_cast<const double2*>(&params.beta[ci]);
             gammaF2.x = static_cast<float>(gh2.x); gammaF2.y = static_cast<float>(gh2.y);
             betaF2.x = static_cast<float>(bh2.x); betaF2.y = static_cast<float>(bh2.y);
         } else {
             gammaF2 = *reinterpret_cast<const float2*>(&params.gamma[ci]);
             betaF2 = *reinterpret_cast<const float2*>(&params.beta[ci]);
         }
     }
 
     float mean = sum * params.invHWC;
     float var = sumSq * params.invHWC - (mean * mean);
     float invStdDev = var <= 0.0f ? 1.0f : rsqrtf(var);
 
     int32_t hwBegin = blockIdx.y * params.hwPerBlock;
     int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);
 
     for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
         int64_t offset = (int64_t) ni * params.hwc + hwi * params.c + ci;
         float2 f2;
 
         if (ci < params.c) {
              if constexpr (std::is_same<DataType, __half>::value) {
                 __half2 h2 = *reinterpret_cast<const __half2*>(&params.src[offset]);
                 f2 = __half22float2(h2);
              } else if constexpr (std::is_same<DataType, double>::value) {
                 double2 d2 = *reinterpret_cast<const double2*>(&params.src[offset]);
                 f2.x = static_cast<float>(d2.x); f2.y = static_cast<float>(d2.y);
              } else {
                 f2 = *reinterpret_cast<const float2*>(&params.src[offset]);
              }
         }
 
         f2.x = (f2.x - mean) * invStdDev;
         f2.y = (f2.y - mean) * invStdDev;
         
         f2.x = gammaF2.x * f2.x + betaF2.x;
         f2.y = gammaF2.y * f2.y + betaF2.y;
 
         if (params.withSwish) {
             f2.x = f2.x * sigmoid(f2.x);
             f2.y = f2.y * sigmoid(f2.y);
         }
 
         if (ci < params.c) {
             if constexpr (std::is_same<DataType, __half>::value) {
                 *reinterpret_cast<__half2*>(&params.dst[offset]) = __float22half2_rn(f2);
             } else if constexpr (std::is_same<DataType, double>::value) {
                 double2 d2;
                 d2.x = static_cast<double>(f2.x);
                 d2.y = static_cast<double>(f2.y);
                 *reinterpret_cast<double2*>(&params.dst[offset]) = d2;
             } else {
                 *reinterpret_cast<float2*>(&params.dst[offset]) = f2;
             }
         }
     }
 }
 
 // -------------------------------------------------------------------------
 // Kernel Launchers
 // -------------------------------------------------------------------------
 
 template<typename T>
 void groupNormNHWCSum(GroupNormNHWCParams<T> const& params, cudaStream_t stream) {
     TORCH_INTERNAL_ASSERT(params.c % params.cPerBlock == 0 && params.hw % params.hwPerBlock == 0);
     dim3 grid;
     grid.x = divUp(params.c, params.cPerBlock);
     grid.y = divUp(params.hw, params.hwPerBlock);
     grid.z = params.n;
     switch (params.cPerBlock) {
         case 320: groupNormNHWCSumKernel<160><<<grid, 160, 0, stream>>>(params); break;
         case 480: groupNormNHWCSumKernel<256><<<grid, 256, 0, stream>>>(params); break;
         case 256: groupNormNHWCSumKernel<128><<<grid, 128, 0, stream>>>(params); break;
         case 128: groupNormNHWCSumKernel<64><<<grid, 64, 0, stream>>>(params); break;
         default: break;
     }
 }
 
 template<typename T>
 void groupNormNHWCScale(GroupNormNHWCParams<T> const& params, cudaStream_t stream) {
     dim3 grid;
     grid.x = divUp(params.c, params.cPerBlock);
     grid.y = divUp(params.hw, params.hwPerBlock);
     grid.z = params.n;

     switch (params.cPerBlock) {
         case 320: groupNormNHWCScaleKernel<160><<<grid, 160, 0, stream>>>(params); break;
         case 480: groupNormNHWCScaleKernel<256><<<grid, 256, 0, stream>>>(params); break;
         case 256: groupNormNHWCScaleKernel<128><<<grid, 128, 0, stream>>>(params); break;
         case 128: groupNormNHWCScaleKernel<64><<<grid, 64, 0, stream>>>(params); break;
         default: break;
     }
 }
 
 // -------------------------------------------------------------------------
 // Main Entry Point
 // -------------------------------------------------------------------------
 
 template <typename T>
 void run_gn_fwd_kernels_v2(
     const T *X_data,
     const T *weight_data,
     const T *bias_data,
     const int N,
     const int R,
     const int C,
     const int G,
     T eps,
     const int64_t act_fn_option,
     T *Y_data,
     float *mean_data,
     float *rstd_data) 
 {
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    using CType = typename CUDAType<T>::type;

    GroupNormNHWCParams<T> params;

    // Pointer casts: T (PyTorch type) -> CType (CUDA native type)
    params.src = reinterpret_cast<const CType*>(X_data);
    params.dst = reinterpret_cast<CType*>(Y_data);
    params.gamma = reinterpret_cast<const CType*>(weight_data);
    params.beta  = reinterpret_cast<const CType*>(bias_data);

    // params.redBuffer = red_buffer_ptr;
    params.meanData = mean_data;
    params.rstdData = rstd_data;
 
    params.n = N;
    params.c = C;
    params.groups = G;
    params.hw = R;
    params.hwc = R * C;
    params.cPerGroup = C / G;
    
    // Block sizing logic
    // params.cPerBlock = 128;
    // if (C % 256 == 0) params.cPerBlock = 256;
    // 1. 优先处理 10, 20, 40 这类特殊对齐
    if (params.cPerGroup % 10 == 0) {
        // 如果 C 很大，优先用 320 (160 threads) 以获得高 Occupancy
        // 320 能整除 10, 20, 40, 80, 160
        if (C >= 320) {
            params.cPerBlock = 320; 
        } 
        // 如果 C 比较小（例如 C=40, 80, 160），用 320 浪费太大，降级到 160 (80 threads)
        else {
            params.cPerBlock = 160; 
        }
    }
    // 2. 处理常规 2 的幂次 (32, 64, 128...)
    else {
        // 默认逻辑，优先 256 (128 threads)
        params.cPerBlock = 256; 
        
        // 如果 cPerGroup 很大（比如 512），或者很奇怪，保底处理：
        // 确保 cPerBlock 是 cPerGroup 的倍数
        if (params.cPerBlock < params.cPerGroup) {
             params.cPerBlock = params.cPerGroup;
             // 向上寻找最接近的 32 的倍数作为 block size (为了 warp 对齐)
             // ... 简单起见，这里可以不做太复杂的 fallback，
             // 假设用户输入的 cPerGroup 都在常见范围内
        }
    }
    
    // 再次确认对齐 (这一步是防止 cPerGroup=120 这种既是10倍数又不能被 320/160 整除的情况)
    // 如果 320 不能整除 cPerGroup，我们必须回退到 "One Block One Group" 模式
    if (params.cPerBlock % params.cPerGroup != 0) {
        params.cPerBlock = params.cPerGroup;
        // 如果算出来是 40，我们可以扩展到 160 以利用更多线程
        while (params.cPerBlock < 128) {
            params.cPerBlock += params.cPerGroup;
        }
    }
    
    params.hwPerBlock = 32;
    params.invHWC = 1.0f / (float)(R * C / G);
    params.withSwish = false;
    
    params.groupsPerBlock = params.cPerBlock / params.cPerGroup;

    groupNormNHWCSum(params, cuda_stream);
    groupNormNHWCScale(params, cuda_stream);
 }
 
 // -------------------------------------------------------------------------
 // Explicit Instantiations (Crucial for Linker!)
 // -------------------------------------------------------------------------
 
// 1. FP16
template void run_gn_fwd_kernels_v2<c10::Half>(
  const c10::Half *X_data, 
  const c10::Half *weight_data, 
  const c10::Half *bias_data, 
  const int N, const int R, const int C, const int G, 
  c10::Half eps, const int64_t act_fn_option, 
  c10::Half *Y_data, float *mean_data, float *rstd_data);

// 2. BF16 (REQUIRED FIX)
template void run_gn_fwd_kernels_v2<c10::BFloat16>(
  const c10::BFloat16 *X_data, 
  const c10::BFloat16 *weight_data, 
  const c10::BFloat16 *bias_data, 
  const int N, const int R, const int C, const int G, 
  c10::BFloat16 eps, const int64_t act_fn_option, 
  c10::BFloat16 *Y_data, float *mean_data, float *rstd_data);

// 3. FP32
template void run_gn_fwd_kernels_v2<float>(
  const float *X_data, 
  const float *weight_data, 
  const float *bias_data, 
  const int N, const int R, const int C, const int G, 
  float eps, const int64_t act_fn_option, 
  float *Y_data, float *mean_data, float *rstd_data);

// 4. FP64
template void run_gn_fwd_kernels_v2<double>(
  const double *X_data, 
  const double *weight_data, 
  const double *bias_data, 
  const int N, const int R, const int C, const int G, 
  double eps, const int64_t act_fn_option, 
  double *Y_data, float *mean_data, float *rstd_data);