import torch
import time
import logging


def CudaProfile(*function_with_args):
    times = 20
    for _ in range(times):
        for func, args in function_with_args:
            func(*args)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(times):
        for func, args in function_with_args:
            func(*args)
    end_event.record()
    # 等待事件完成
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)  # 以毫秒为单位        
    return elapsed_time/times


def CpuProfile(*function_with_args):
    times = 20
    for _ in range(times):
        for func, args in function_with_args:
            func(*args)
    start = time.time()
    for _ in range(times):
        for func, args in function_with_args:
            func(*args)
    
    elapsed_time = time.time() - start  # 以毫秒为单位        
    return 1000 * elapsed_time/times

def BangProfile(*function_with_args):
    times = 20
    for _ in range(times):
        for func, args in function_with_args:
            func(*args)
    start = time.time()
    for _ in range(times):
        for func, args in function_with_args:
            func(*args)
    
    elapsed_time = time.time() - start  # 以毫秒为单位        
    return 1000 * elapsed_time/times

def logBenchmark(baseline, time, cl):
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    msg = "Pytorch: " + str(round(cl/baseline/1000/1000/1000,2)) + " TFLOPS, kernel: " + str(round(cl/time/1000/1000/1000,2)) + " TFLOPS "
    percentage = "{:.2f}%".format(abs(baseline/time * 100))

    logging.info(msg + "\033[32m" + "[+" + percentage + "]" +"\033[0m")