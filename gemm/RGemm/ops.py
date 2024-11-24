import torch
from torch import Tensor

__all__ = ["naive_gemm_cuda"]

def naive_gemm_cuda(A: Tensor, B: Tensor, alpha=1.0, beta=0.0):
    return torch.ops.RGemm.naive_gemm_cuda(A, B, alpha, beta)