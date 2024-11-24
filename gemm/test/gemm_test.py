
import torch
import argparse
import sys
import RGemm.ops as ops
import performance
import sys
import os

test_cases = [
        {"A": ((1024, 1024), torch.float32, 'cuda'),
         "B": ((1024, 1024), torch.float32, 'cuda')},
        {"A": ((2048, 2048), torch.float32, 'cuda'),
         "B": ((2048, 2048), torch.float32, 'cuda')},
        {"A": ((4096, 4096), torch.float32, 'cuda'),
         "B": ((4096, 4096), torch.float32, 'cuda')},
        {"A": ((8192, 8192), torch.float32, 'cuda'),
         "B": ((8192, 8192), torch.float32, 'cuda')},
        {"A": ((16384, 16384), torch.float32, 'cuda'),
         "B": ((16384, 16384), torch.float32, 'cuda')}     
]

def test(test_cases):
    for test_case in test_cases:
        A_size = test_case["A"][0]
        B_size = test_case["B"][0]
        test_dtype = test_case["A"][1]
        device = test_case["A"][2]
        
        print(f"Testing Linear on {device} with A_size:{A_size} x B_size: {B_size}, dtype:{test_dtype}")
            
        A = torch.rand(A_size, device=device, dtype=test_dtype, requires_grad=False)
        B = torch.rand(B_size, device=device, dtype=test_dtype, requires_grad=False) 

        if test_dtype == torch.float32:
            if device == "cuda":
                # Precision Comparison
                torch_C = torch.mm(A, B)
                custom_C = ops.naive_gemm_cuda(A, B)
                
                print("absolute error:%.4e"%(torch.mean(abs(torch_C - custom_C))))
                print("relative error:%.4e"%(torch.mean(abs(torch_C - custom_C)) / (torch.mean(abs(torch_C) + 1e-8))))
                
                if torch.allclose(torch_C, custom_C,  rtol=1e-02):
                    print("Ressult CORRECT!")
                else:
                    print("Torch: ", torch_C)
                    print("Custom:", custom_C)
                    raise AssertionError("Ressult ERROR!")

                # perm
                torch_gemm_time = performance.CudaProfile((torch.mul, (A, B)))  # 以毫秒为单位
                custom_gemm_time = performance.CudaProfile((ops.naive_gemm_cuda, (A, B)))  # 以毫秒为单位
            
            if device == "cpu":
                pass
        elif test_dtype == torch.float16:
            pass
        
        cl = A_size[0] * A_size[1] * B_size[1]
        performance.logBenchmark(torch_gemm_time, custom_gemm_time, cl) # cl: compute load 

test(test_cases)
