import torch
from torch import Tensor

__all__ = ["naive_gemm_cuda", "gemm01_cuda"]

def naive_gemm_cuda(A: Tensor, B: Tensor, alpha=1.0, beta=0.0):
    return torch.ops.RGemm.naive_gemm_cuda(A, B, alpha, beta)

def gemm01_cuda(A: Tensor, B: Tensor, alpha=1.0, beta=0.0):
    return torch.ops.RGemm.gemm01_cuda(A, B, alpha, beta)

def gemm02_cuda(A: Tensor, B: Tensor, alpha=1.0, beta=0.0):
    return torch.ops.RGemm.gemm02_cuda(A, B, alpha, beta)