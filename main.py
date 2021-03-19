import time
import os
import json
import pickle
import torch
import ofa

from torch.multiprocessing import Process, Queue, set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass

from read_power import read_sysfs
from build_model import build_block
from tqdm import tqdm
from time import sleep


# configurations
times = 1000 # times to run the blocks

def load_config_list():
    block_config_pkl = './config_list.pkl'
    # read in block configurations
    with open(block_config_pkl, 'rb') as f:
        block_config = pickle.load(f)
    return block_config

block_config = load_config_list()

def static_power():
    print("sleeping to stablize system ...")
    sleep(10)
    print("measuring static power ...")

    static_power = 0
    for _ in range(1000):
        power = read_sysfs()
        static_power += float(power)
    static_power /= 1000
    
    with open('./static_power.txt', 'w') as f:
        f.write(str(static_power))

    print('static power measurement done')

# ------- power measurement thread ---------
def power_thread(power_q, interval_q, in_q):
    # t_list = list() # time interval list
    # p_list = list() # power(mW) list
    
    while True:
        start = time.perf_counter() # time interval in second

        power = read_sysfs()

        end = time.perf_counter() 
        interval = end - start

        # p_list.append(power)
        # t_list.append(interval)
        
        interval_q.put(interval)
        power_q.put(power)

        if not in_q.empty():
            print("power thread exited")
            break        
    return 

# -------- latency measurement thread --------
_sentinel = object()
def latency_thread(model_queue, input_tensor_queue, return_q, out_q):
    """
    we return a list of latnecy data, becasue I'm not sure if 
    there would a warm up period when we measure with pytorch
    """
    input_tensor = input_tensor_queue.get()
    model = model_queue.get()
    model = model.to('cuda')
    input_tensor = input_tensor.to('cuda')

    # test latency here
    for _ in range(times):
        start = time.perf_counter()
        model(input_tensor)
        end = time.perf_counter()
        latency = (end - start)
        return_q.put(latency)
    
    
    del input_tensor

    out_q.put(_sentinel)
    print("latency_thread exited")
    return


# launches both latency and power threads
# to measure dynamic power and latency for one model
def dynamic_power(model, input_shape):
    q = Queue()
    power_return = Queue()
    interval_return = Queue()
    latency_return = Queue()
    input_tensor_queue = Queue()
    model_queue = Queue()
    
    input_tensor = torch.ones([*input_shape])
    input_tensor_queue.put(input_tensor)
    
    model.share_memory()

    model_queue.put(model)
    
    context = torch.multiprocessing.get_context('spawn')

    p_thread = context.Process(target=power_thread, args=(power_return, interval_return, q))
    l_thread = context.Process(target=latency_thread, args=(model_queue, input_tensor_queue, latency_return, q))

    l_thread.start()
    p_thread.start()

    power_l = list() # GPU power list
    interval_l = list() # power interval list
    latency_l = list() # latency list

    l_thread.join()

    while True:
        if not power_return.empty():
            power_l.append(power_return.get())
        if not interval_return.empty():
            interval_l.append(interval_return.get())
        if not latency_return.empty():
            latency_l.append(latency_return.get())
        if power_return.empty() and interval_return.empty() and latency_return.empty():
            break

    power_return.close()
    interval_return.close()
    latency_return.close()
    q.close()

    del q
    del power_return
    del latency_return
    del interval_return

    return latency_l, power_l, interval_l 


# main function
def main():
    # measure static power
    static_power()

    # take care of the result folder
    result_dir = './results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print('made new directory: ' + result_dir)

    # measure dynamic power
    for idx, b_conf in enumerate(tqdm(block_config)):
        in_channel = b_conf['mobile_inverted_conv']['in_channels']
        h, w = b_conf['input_size']
        model = build_block(b_conf)
        latency_l, power_l, interval_l = dynamic_power(model, (1, in_channel, h, w))
        print(f"latency_l length = {len(latency_l)}, power_l length={len(power_l)}, interval_l length={len(interval_l)}")
        # save each list into a txt file
        with open(os.path.join(result_dir, str(idx) + '_latency_torch.txt'), 'w') as f:
            for v in latency_l: f.write(str(v) + '\n')
        with open(os.path.join(result_dir, str(idx) + '_power.txt'), 'w') as f:
            for v in power_l: f.write(str(v) + '\n')
        with open(os.path.join(result_dir, str(idx) + '_interval.txt'), 'w') as f:
            for v in interval_l: f.write(str(v) + '\n')
        print("saved results to file.")

if __name__ == "__main__":
    main()
