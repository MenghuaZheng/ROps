/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <ATen/ATen.h>
 // -------------------------------------------------------------------------
 // Type Traits: Map PyTorch/c10 types to CUDA native types
 // -------------------------------------------------------------------------
 
 template<typename T>
 struct CUDAType;
 
 template<>
 struct CUDAType<float> {
     using type = float;
 };
 
 // 新增：支持 Double，满足链接器需求
 template<>
 struct CUDAType<double> {
     using type = double;
 };
 
 template<>
 struct CUDAType<c10::Half> {
     using type = __half;
 };
 
 template<>
 struct CUDAType<c10::BFloat16> {
     using type = __nv_bfloat16;
 };
 
 static inline int32_t divUp(int32_t m, int32_t n) {
     return (m + n - 1) / n;
 }
 
 static inline __device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// -------------------------------------------------------------------------
// Sum Kernel
// -------------------------------------------------------------------------

struct GroupSums {
    int32_t flag;
    float sum;
    float sumSq;
};

struct GroupSumsOp {
    inline __device__ GroupSums operator()(GroupSums const& a, GroupSums const& b) {
        GroupSums dst;
        dst.sum = b.flag ? b.sum : (a.sum + b.sum);
        dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
        dst.flag = a.flag + b.flag;
        return dst;
    }
};


 // -------------------------------------------------------------------------
 // Params Structure
 // -------------------------------------------------------------------------
 
 template <typename T>
 struct GroupNormNHWCParams {
     using DataType = typename CUDAType<T>::type;
 
     DataType* dst;
     DataType const* src;
     DataType const* gamma;
     DataType const* beta;
     float* redBuffer;
     float* meanData;
     float* rstdData;

     int32_t n;
     int32_t h;
     int32_t w;
     int32_t c;
     int32_t groups;
     bool withSwish;
 
     int32_t hw;
     int32_t hwPerBlock;
     int32_t cPerBlock;
     int32_t cPerGroup;
 
     int32_t hwc;
     float invHWC;
     int32_t groupsPerBlock;
 };
 
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
     float *rstd_data);