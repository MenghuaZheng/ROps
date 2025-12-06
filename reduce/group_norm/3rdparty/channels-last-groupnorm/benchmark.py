# benchmarks various groupnorm kernels
from gnnhwc import GN_NHWC, GN_NHWC_V2
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, datetime, time, os, itertools
import pandas as pd

# make strings different colors
def red(text): return '\033[91m' + str(text) + '\033[0m'
def green(text): return '\033[92m' + str(text) + '\033[0m'
def yellow(text): return '\033[93m' + str(text) + '\033[0m'
def blue(text): return '\033[94m' + str(text) + '\033[0m'

def config_filter(x): # returns true if config is valid
    DTYPE, B, C, R, G = x
    if C % G != 0:
        return False
    if R == 1: # this causes an autograd problem where it gets confused since the tensor is both contiguous in NCHW/NHWC format 
        return False

    dtype_size = torch.finfo(DTYPE).bits / 8
    estimated_mem_usage_gib = (5 * dtype_size * B * C * R * R) / 2**30 #  this is just a rough estimate, likely wrong
    if estimated_mem_usage_gib > 3: # vram filter
        return False
    return True

if __name__ == '__main__':
    INCLUDE_BWD = False # benchmark forward and backward pass
    ACT_FN = 'identity'
    act_fn = {
        'identity': lambda x: x,
        'silu': F.silu,
        'relu': F.relu,
        'gelu': F.gelu,
        'gelu_tanh': lambda x: F.gelu(x, approximate='tanh'),
    }[ACT_FN]

    NSEC = 1 # number of seconds that each kernel runs for on a certain input
    # NSEC = 0 # number of seconds that each kernel runs for on a certain input
    DTYPES = [torch.float16]

    BATCHES = [1, 4, 8]
    CHANNELS = [320, 640, 1280, 2560]
    RESOLUTIONS = [16, 32, 64, 128]
    NUM_GROUPS = [32]

    # BATCHES = [1]
    # CHANNELS = [256]
    # RESOLUTIONS = [64]
    # NUM_GROUPS = [32]

    GN_KERNELS = [
        (nn.GroupNorm, 'torch.nn GN NCHW'),
        (nn.GroupNorm, 'torch.nn GN NHWC'),
        (GN_NHWC, 'GN NHWC'),
        (GN_NHWC_V2, 'GN NHWC V2'),
    ]

    os.makedirs('csvs', exist_ok=True)
    fname = datetime.datetime.now().strftime(os.path.join('csvs', '%Y-%m-%d-%H-%M-%S.csv'))
    print(f'Writing to {fname}')
    outfile = open(fname, 'w')
    outfile.write('Kernel,B (batch),C (num channels),R (resolution),G (num groups), D (C/G),Bandwidth (GB/s),Speed (it/s), is_correct\n')
    
    configs = list(filter(config_filter, itertools.product(DTYPES, BATCHES, CHANNELS, RESOLUTIONS, NUM_GROUPS)))
    print('Estimated time (seconds) to complete:', NSEC * len(configs) * len(GN_KERNELS))
    print(f'Activation fn: {ACT_FN}')
    print(f'Include bwd pass: {INCLUDE_BWD}')

    case_in_sd = [
        (2560, 16),
        (1280, 32),
        (640, 64),
        (320, 128),
    ]
    for DTYPE, B, C, R, G in configs:
        
        if(C, R) not in case_in_sd:
            print('Case in SD')
            continue

        print(blue(f'DTYPE: {DTYPE} | B: {B} | C: {C} | R: {R} | G: {G}'))
        for gn_class, desc in GN_KERNELS:
            # 为了减少L2 cache的影响，每次都新生成数据
            x_nchw = torch.randn((B, C, R, R), dtype=DTYPE, device='cuda', requires_grad=True)
            x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last).detach().requires_grad_(True)

            pad_len = max(len(kernel[1]) for kernel in GN_KERNELS)
            desc = f'{desc:<{pad_len}}' # right pad desc with spaces

            gn_input = x_nchw if 'NCHW' in desc else x_nhwc
            grad = torch.ones_like(gn_input)

            try:
                gn_layer = gn_class(G, C).to(DTYPE).cuda()
                gn_layer_baseline = nn.GroupNorm(G, C).to(DTYPE).cuda()
                # warmup iter
                g = gn_layer(gn_input)
                baseline = gn_layer_baseline(gn_input)

                is_correct = torch.allclose(g, baseline, atol=1e-3, rtol=1e-2)

                if not isinstance(gn_layer, GN_NHWC):
                    g = act_fn(g)
                if INCLUDE_BWD:
                    torch.autograd.grad(g, gn_input, grad_outputs=grad, retain_graph=True)

                tic = time.perf_counter()
                tic_sec = time.perf_counter()
                ntrials = 0

                while time.perf_counter() - tic < NSEC:
                    g = gn_layer(gn_input)
                    if not isinstance(gn_layer, GN_NHWC):
                        g = act_fn(g)
                    if INCLUDE_BWD:
                        torch.autograd.grad(g, gn_input, grad_outputs=grad, retain_graph=True)
                    torch.cuda.synchronize()
                    ntrials += 1

                    if time.perf_counter() - tic_sec > 0.1:
                        speed = round(ntrials / (time.perf_counter() - tic), 2)
                        bw = ntrials * gn_input.nbytes * (3+5 if INCLUDE_BWD else 3) / (time.perf_counter() - tic)
                        bw = round(bw / 1e9, 2)
                        print(f'\t{desc} | {time.perf_counter() - tic:.1f}/{NSEC} sec, bandwidth: {green(bw)} GB/s, speed: {yellow(speed)} it/s, is_correct: {is_correct}         \r', end='')
                        tic_sec = time.perf_counter()

                speed = round(ntrials / (time.perf_counter() - tic), 2)
                bw = ntrials * gn_input.nbytes * (3+5 if INCLUDE_BWD else 3) / (time.perf_counter() - tic)
                bw = round(bw / 1e9, 2)
                print(f'\t{desc} | bandwidth: {green(bw)} GB/s, speed: {yellow(speed)} it/s                             ')
            except KeyboardInterrupt:
                print(f'Keyboard interrupt, closing {fname}                               ')
                outfile.close()
                raise
            except Exception as e:
                print(red(f'\t{desc} | FAILED; err msg:'), str(e).strip())
                raise
                speed = '-1 (failed)'
            
            outfile.write(f'{desc},{B},{C},{R},{G},{C//G},{bw},{speed},{is_correct}\n')
    print(f'All tests done, closing {fname}')
    outfile.close()

    # 1. 使用 Pandas 读取刚才写好的 CSV
    try:
        df = pd.read_csv(fname)
        
        # 2. 定义分组长度
        group_size = len(GN_KERNELS)
        
        # ANSI 颜色代码
        RED = '\033[91m'
        RESET = '\033[0m'
        BOLD = '\033[1m'

        print("\n" + "="*80)
        print(f"{BOLD}Summary (Best Bandwidth per Group){RESET}")
        print("="*80)

        # 用来存储处理后的行（如果需要回写文件）
        new_lines = []
        # 保留 Header
        new_lines.append(','.join(df.columns.tolist())) 

        # 3. 按步长分组遍历
        for i in range(0, len(df), group_size):
            # 获取当前组的数据切片
            group = df.iloc[i : i + group_size]
            
            if group.empty: continue

            # 找到 Bandwidth 最大的那一行的索引
            # 假设 CSV 列名是 'Bandwidth(GB/s)'，需要根据实际 header 修改
            # 如果列名有空格，最好 strip 一下
            bw_col = [c for c in df.columns if 'Bandwidth' in c or 'GB/s' in c][0]
            
            # 找到最大值的行索引
            max_idx = group[bw_col].idxmax()
            
            # 4. 遍历当前组并打印/处理
            for idx, row in group.iterrows():
                line_str = ','.join([str(x) for x in row.values])
                
                if idx == max_idx:
                    # 终端打印：标红
                    print(f"{RED}{line_str}  <-- MAX BANDWIDTH{RESET}")
                    # 文件回写：添加标记 (可选)
                    new_lines.append(f"{line_str}, *BEST*") 
                else:
                    # 普通打印
                    print(line_str)
                    new_lines.append(line_str)
            
            # 组间加个分隔符，方便看
            print("-" * 80)

        # 5. (可选) 将标记后的结果写回文件
        # 这样你打开 CSV 就能看到哪一行后面多了个 *BEST*
        with open(fname, 'w') as f:
            f.write('\n'.join(new_lines))
        print(f"Results with highlights saved to {fname}")

    except Exception as e:
        print(f"Error processing CSV for statistics: {e}")
        # 可能原因：文件为空，或者列名对不上