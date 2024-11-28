#include <00naive.h>
#include <01blocktiling.h>
#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>
#include <torch/extension.h>
#include <torch/utils.h>
#include <stdio.h>


namespace RGemm{
  // An example of an operator that mutates one of its inputs.
  at::Tensor naive_gemm_cuda(const at::Tensor& A, const at::Tensor& B, double alpha=1.0, double beta=0.0) {
      TORCH_CHECK(A.dim() == 2, "A must be a 2-dimensional tensor");
      TORCH_CHECK(B.dim() == 2, "B must be a 2-dimensional tensor");
      TORCH_CHECK(A.size(1) == B.size(0), "The second dimension of A must be equal to the first dimension of B");
      TORCH_CHECK(A.dtype() == at::kFloat);
      TORCH_CHECK(B.dtype() == at::kFloat);
      TORCH_INTERNAL_ASSERT(A.device().type() == at::DeviceType::CUDA);
      TORCH_INTERNAL_ASSERT(B.device().type() == at::DeviceType::CUDA);
      int M = A.size(0);
      int N = B.size(1);
      int K = B.size(0);
      int _c_size[] = {M, N};
      c10::ArrayRef<int> c_size(_c_size); 
      at::Tensor a_contig = A.contiguous();
      at::Tensor b_contig = B.contiguous();
      at::Tensor C = torch::empty({M, N}, at::device(A.device().type()).dtype(A.dtype()));
      const float* a_ptr = a_contig.data_ptr<float>();
      const float* b_ptr = b_contig.data_ptr<float>();
      float* c_ptr = C.data_ptr<float>();

      naive_sgemm(a_ptr, b_ptr, c_ptr, M, N, K, alpha, beta);

      return C;
  }

    // An example of an operator that mutates one of its inputs.
  at::Tensor gemm01_cuda(const at::Tensor& A, const at::Tensor& B, double alpha=1.0, double beta=0.0) {
      TORCH_CHECK(A.dim() == 2, "A must be a 2-dimensional tensor");
      TORCH_CHECK(B.dim() == 2, "B must be a 2-dimensional tensor");
      TORCH_CHECK(A.size(1) == B.size(0), "The second dimension of A must be equal to the first dimension of B");
      TORCH_CHECK(A.dtype() == at::kFloat);
      TORCH_CHECK(B.dtype() == at::kFloat);
      TORCH_INTERNAL_ASSERT(A.device().type() == at::DeviceType::CUDA);
      TORCH_INTERNAL_ASSERT(B.device().type() == at::DeviceType::CUDA);
      int M = A.size(0);
      int N = B.size(1);
      int K = B.size(0);
      int _c_size[] = {M, N};
      c10::ArrayRef<int> c_size(_c_size); 
      at::Tensor a_contig = A.contiguous();
      at::Tensor b_contig = B.contiguous();
      at::Tensor C = torch::empty({M, N}, at::device(A.device().type()).dtype(A.dtype()));
      const float* a_ptr = a_contig.data_ptr<float>();
      const float* b_ptr = b_contig.data_ptr<float>();
      float* c_ptr = C.data_ptr<float>();

      sgemm01(a_ptr, b_ptr, c_ptr, M, N, K, alpha, beta);

      return C;
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

  TORCH_LIBRARY(RGemm, m) {
    m.def("naive_gemm_cuda(Tensor A, Tensor B, float alpha, float beta) -> Tensor");
    m.def("gemm01_cuda(Tensor A, Tensor B, float alpha, float beta) -> Tensor");
  }

  TORCH_LIBRARY_IMPL(RGemm, CUDA, m) {
      m.impl("naive_gemm_cuda", &naive_gemm_cuda);
      m.impl("gemm01_cuda", &gemm01_cuda);
  }
}