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
当M=N=K=1024时，Pytorch: 41.72 TFLOPS, kernel: 4.69 TFLOPS [+11.25%]

当M=N=K=2048时，Pytorch: 54.71 TFLOPS, kernel: 4.78 TFLOPS [+8.74%]

当M=N=K=3072时，Pytorch: 54.33 TFLOPS, kernel: 4.96 TFLOPS [+9.12%]

当M=N=K=4096时，Pytorch: 56.42 TFLOPS, kernel: 4.9 TFLOPS [+8.68%]

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

而该nvive Gemm中，每个thread 读取A中K个元素，B中K个元素，C中1个元素，写入C中1个元素，共launch了MN个thread。
故放存量为：$MN\times(K+K+1+1) \times sizeof(float)=2MN(K+1) \times sizeof(float)$

计算量仍为：$2MNK$

$ratio_{kernel}(1024, 1024, 1024) = 0.25 < 81.3$

由于访存和计算是异步的，所以我们希望计算的时间能够cover掉访存的时间，即$ratio_{kernel} > ratio_{device}$，但是$ratio_{kernel} << ratio_{device} < ratio_{gemm}$，表示就算显存带宽打满的情况下，访存时间也远远大于计算时间，即cuda core一直在空闲在等待数据来计算。

所以下一步的优化策略是提升$ratio_{kernel}$，也就是提升计算强度。

有两个思路：
1. 在单个thread计算量不变的情况下，提高数据复用率 （Block tiling）
2. 在单个thread访存量不变的情况下，提升单个thread线程的计算量 （Thread tiling）

## 01 Block tiling
### 思路
将A 中$BM \times K$个数据读入shared memory中，B中$BN \times K$个数据读入shared memory中。一个thread block计算$BM \times BN$个元素，每个thread 计算一个C中元素。由于$BM \times K$个数据太大，不能一次性读入shared memory中，所以每次load $BM \times BK$ 个数据到shared memory中，共循环K/BK次。

### 结果
当M=N=K=1024时，Pytorch: 41.31 TFLOPS, kernel: 5.91 TFLOPS [+14.31%]

当M=N=K=2048时，Pytorch: 54.91 TFLOPS, kernel: 6.06 TFLOPS [+11.04%]

当M=N=K=3072时，Pytorch: 54.41 TFLOPS, kernel: 6.08 TFLOPS [+11.18%]

当M=N=K=4096时，Pytorch: 58.43 TFLOPS, kernel: 5.7 TFLOPS [+9.76%]

### 分析
在block tiling Gemm算子中，每个thread block 读取A中$BLOCK\_M \times K$个元素，B中$K \times BLOCK\_N $个元素，C中$BLOCK\_M \times BLOCK\_N$个元素，写入C中$BLOCK\_M \times BLOCK\_N$个元素，共launch了$M \times N/BLOCK\_M/BLOCK\_N$个thread blocks。

故访存量为：
$$\begin{align*}
(BLOCK\_M \times K + K \times BLOCK\_N + 2BLOCK\_M \times BLOCK\_N) \times M \times N/BLOCK\_M/BLOCK\_N \times \text{sizeof}(float) &\\ = MNK\left(\frac{1}{BLOCK\_M} + \frac{1}{BLOCK\_N}\right) 
+ 2MN \times \text{sizeof}(float)
\end{align*}
$$

计算量仍为：$2MNK$

$ratio_{kernel}(1024, 1024, 1024) = 8 < 81.3$(BLOCK_M=BLOCK_N=32)

## 02 Thread tiling
### 思路
将A中$BM \times TM \times K$个数据读入shared memory中，B中$BN \times TN \times K$个数据读入shared memory中。一个thread block launch $BM \times BN$个thread，一个thread block计算$BM \times TM \times BN \times TN$个元素，每个thread 计算$TM \times TN$个C中元素。

### 结果
当M=N=K=1024时，Pytorch: 40.91 TFLOPS, kernel: 6.03 TFLOPS [+14.73%]

当M=N=K=2048时，Pytorch: 54.91 TFLOPS, kernel: 11.63 TFLOPS [+21.19%]

当M=N=K=3072时，Pytorch: 54.41 TFLOPS, kernel: 10.57 TFLOPS [+19.43%]

当M=N=K=4096时，Pytorch: 57.77 TFLOPS, kernel: 11.48 TFLOPS [+19.87%]

### 分析
thread tiling每个thread 计算$TN\times TM$个C矩阵中的元素，block 处理$(BLOCK\_M \times TM) \times (BLOCK\_N \times TN)$个元素，每个thread block 从A中load $BLOCK\_M \times TM \times K$个元素，从B中load $BLOCK\_N \times TN \times K$个元素，从C中load和write 共$2(BLOCK\_M \times TM) \times (BLOCK\_N \times TN)$个元素。一共launch了$\frac{M}{BLOCK\_M\times TM} \times \frac{N}{BLOCK\_N \times TN}$个block。

访存量为：$[(BLOCK\_M \times TM \times K) + (BLOCK\_N \times TN \times K) + 2(BLOCK\_M \times TM) \times (BLOCK\_N \times TN)] \times \frac{M}{BLOCK\_M\times TM} \times \frac{N}{BLOCK\_N \times TN} \times sizeof(float)$

计算量：$2MNK$

$ratio_{kernel}= \frac{1}{\frac{1}{2BN\times TN} + \frac{1}{2BM\times TM} + \frac{1}{K}} \approx \frac{2BM\times TM \times BN \times TN}{BM \times TM + BN \times TN}$

$ratio_{kernel}(1024, 1024, 1024) = 15.1 < 81.3$(BLOCK_M=BLOCK_N=16, TM=TN=4)