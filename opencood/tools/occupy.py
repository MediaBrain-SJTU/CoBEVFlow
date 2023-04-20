# Refer: https://sparkydogx.github.io/2019/03/16/occupy-gpu-memory-in-advance/
from multiprocessing import Queue, Process
import os, json,ipdb
import torch, multiprocessing
import socket
import math
import time
import sys
import signal
card = [
        (0, 22000, 400),
        (1, 22000, 400),
        (2, 22000, 400),
        (3, 22000, 400),
        # (4, 22000, 400),
        # (5, 22000, 400),
        # (6, 22000, 400),
        # (7, 22000, 400),
    ]
# last_used = [0,0,0,0,0,0,0,0]
last_used = [0,0,0,0]
space = [None , None, None, None] #, None, None, None, None]

def get_pid_memory(card_id):
    pid = os.getpid()
    gpustat = json.loads(os.popen('gpustat --json').read())
    memory_process_list = []

    gpu = gpustat['gpus'][card_id]
    for process in gpu['processes']:
        if process['username'] == 'root':
            memory_process_list.append(process['gpu_memory_usage'])
    
    # assert len(memory_process_list) <= 1, f'Now only SINGLE occupy is supported: \n{memory_process_list}'
    # assert len(memory_process_list) != 0, 'There are NO cuda process right now'
    occupy_used = 0
    for process in gpu['processes']:
        if process['pid'] == pid:
            occupy_used = process['gpu_memory_usage']
    return sum(memory_process_list)-occupy_used, occupy_used

def occupy_mem(args):
    global last_used,space
    card_id = args[0]
    max_memory = args[1]
    K = args[2]
    used, occupy_used = get_pid_memory(card_id)
    print('card_id:',card_id,'max_memory:', max_memory, 'now used:', used)
    block_mem = max_memory - used - 3000
    #ipdb.set_trace()
    device = torch.device('cuda', card_id)
    if block_mem < 0 or used == 0:
        if space[card_id] is not None:
            del space[card_id]
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            space[card_id] = torch.ones(math.ceil(1024 * 1024 * 100 * 0/K), dtype=torch.float32, device=device)
        last_used[card_id] = used
        time.sleep(10)
        if not get_pid_memory(card_id)[0]:
            os.kill(os.getpid(), signal.SIGTERM)
        return True
    if abs(used - last_used[card_id]) > 2000:
        last_used[card_id] = used
        if space[card_id] is not None:
            del space[card_id]
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            space[card_id] = torch.ones(math.ceil(1024 * 1024 * 100 * block_mem/K), dtype=torch.float32, device=device)
        return False

def works(*args):
    while True:
        flag = occupy_mem(args)
            
if __name__ == '__main__':
    # Your training starts
    #ipdb.set_trace()
    #while True:
        #occupy_mem(card[0])
    #ipdb.set_trace()
    #works(*card[0])
    for car in card:
        work = Process(target=works, args=car)
        work.start()
        # occupy_mem(card[0])
        # occupy_mem(card[1])
    # with multiprocessing.Pool(processes=8) as pool:
    #     results = pool.map(works, card)
    
  
        # time.sleep(60)