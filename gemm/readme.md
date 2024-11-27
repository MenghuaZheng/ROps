# Gemm

设备：RTX 4090
矩阵：A x B = C 
A: (M, K)
B: (K, N)
C: (M, N)
dtype: float32
## 00 Naive Gemm

### 思路
一个thread block 计算 ${BLOCK\_SIZE \times BLOCK\_SIZE}$个C矩阵中的元素，每个线程计算1个C矩阵中的元素，共launch ${M\times N/ BLOCK\_SIZE^2}$个Grid。每个线程遍历A和B的K维度[0,K-1]共K个元素。

### 结果
当M=N=K=1024时，Pytorch: 92.18 TFLOPS, kernel: 4.3 TFLOPS [+4.66%]

当M=N=K=2048时，Pytorch: 669.75 TFLOPS(这个数据似乎有点点问题), kernel: 4.46 TFLOPS [+0.67%]

### 分析
先分析设备的算力和带宽的比值，RTX 4090显存带宽为1008GB/s，float32的算力为：82TFLOPS

$ratio_{device}=\frac{82\times10^3 GFLOPS}{1008GB/s} = 81.3$

$ratio_{device}$表示：

1. 在带宽打满的情况下，每从显存中搬运1B数据，cuda core可以进行81.3次浮点运算。即算子理论的计算访存比大于该ratio，表示该kernel为coumpute bound算子，否则是memory bound算子。

2. $ratio_{device}$其实还表示，当kernel计算访存比大于$ratio_{device}$时，计算所需要的时间大于访存的时间，当kernel为compute bound时，计算时间可以cover掉访存的时间。

NOTE: 
1. 算子理论的计算访存比 表示该算子在最低（理论上）的访存量情况下计算的计算访存比（例如，gemm的理论最低访存量为$(MK + NK + 2MN)\times sizeof(float)$，下文有解释）
2. Kernel计算访存比 表示该Kernel实际的访存量计算出的访存比。

再计算该Gemm算子的计算访存比：

计算量：2MNK (FFM算两次计算，一次乘法，一次加法)

访存量：$(MK + NK + 2MN)\times sizeof(float)$ (A: MK, B: NK, C: 2MN)

$ratio_{gemm}(M, N, K)=\frac{2MNK}{(MK+NK+2MN)\times sizeof(float)}$

$ratio_{gemm}(1024,1024,1024)=128 > 81.3$

$ratio_{gemm}(2048,2048,2048)=256 > 81.3$

当M=N=K=1024(2048)时，Gemm算子是compute bound算子。

而该nvive Gemm中，每个thread 读取A中K个元素，B中K个元素，C中1个元素，写入C中1个元素，共launch了MN个元素。
故放存量为：$MN\times(K+K+1+1) \times sizeof(float)=2MN(K+1) \times sizeof(float)$

计算量仍为：$2MNK$

$ratio_{kernel}(1024, 1024, 1024) = 0.25 < 81.3$

由于访存和计算是异步的，所以我们希望计算的时间能够cover掉访存的时间，即$ratio_{kernel} > ratio_{device}$，但是$ratio_{kernel} << ratio_{device} < ratio_{gemm}$，表示就算显存带宽打满的情况下，访存时间也远远大于计算时间，即cuda core一直在空闲在等待数据来计算。

所以下一步的优化策略是提升$ratio_{kernel}$，也就是提升计算强度。

## 01 Block tiling