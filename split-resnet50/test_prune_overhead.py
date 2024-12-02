# pruning되지 않은 일부 채널만 전송할 때 송수신 전후로 발생하는 추가 오버헤드를 측정하는 스크립트 (약 1분 소요)
# 실행 방법: docker run --rm -dt -p 29500:29500 -v log:/log test python test_prune_overhead.py; docker run --rm -dt test python test_prune_overhead.py --rank 1
# 결과 확인: docker desktop -> volumes 탭 -> 'log' 볼륨 -> runtime.txt (종류별 평균 실행시간이 여기에 append됨)

import torch
import torch.distributed.rpc as rpc
import os
import argparse
import time
import random

x_shape = [1, 100, 256, 256]
is_input_channel_unpruned = [True] * x_shape[1]
num_measures = 100
cached_restored_x = torch.zeros(x_shape)


def record_unpruned_channel_indices(indices):
    global is_input_channel_unpruned
    is_input_channel_unpruned = indices


def prune_input_channels(prune_ratio: float):
    num_channels = x_shape[1]
    num_channels_pruned = int(num_channels * prune_ratio)
    pruned_channel_indices = random.sample(range(num_channels), k = num_channels_pruned)
    
    global is_input_channel_unpruned
    is_input_channel_unpruned = [i not in pruned_channel_indices for i in range(num_channels)]
    rpc.rpc_async('worker1', func = record_unpruned_channel_indices, args = (is_input_channel_unpruned, )).wait()


def normal_echo_request(x: torch.Tensor):
    return x


def pruned_echo_request(x: torch.Tensor):
    return x


def pruned_echo_request_with_size_restoration(x: torch.Tensor):
    original_shape = list(x.shape)
    original_shape[1] = len(is_input_channel_unpruned)
    restored_x = torch.zeros(original_shape)
    restored_x[:, is_input_channel_unpruned, :, :] = x
    return restored_x


def pruned_echo_request_with_cached_size_restoration(x: torch.Tensor):
    cached_restored_x[:, is_input_channel_unpruned, :, :] = x
    return cached_restored_x


def log_average_time(task, task_name, log_file):
    net_time = 0
    for _ in range(num_measures):
        x = torch.rand(x_shape)
        start = time.time()

        task(x)

        end = time.time()
        net_time += end - start
    log_file.write(f'{task_name}: {net_time / num_measures}s\n')


def task1(x):
    """
    텐서를 그대로 주고받는 작업
    """
    y = rpc.rpc_async('worker1', func = normal_echo_request, args = (x,)).wait()


def task2(x):
    """
    보낼 때 일부 입력 채널을 제거한 뒤 주고받는 작업
    """
    y = rpc.rpc_async('worker1', func = pruned_echo_request, args = (x[:, is_input_channel_unpruned, :, :],)).wait()


def task3(x):
    """
    task2처럼 입력 채널을 제거한 뒤 전송하지만 받는 쪽에서 원래 사이즈를 복원한 뒤 응답하는 작업
    """
    y = rpc.rpc_async('worker1', func = pruned_echo_request_with_size_restoration, args = (x[:, is_input_channel_unpruned, :, :],)).wait()


def task4(x):
    """
    task3와 유사하지만 요청 데이터의 크기가 일정하다고 가정하여 미리 준비해둔 torch.zeros에 값만 복사한 뒤 응답하는 작업
    """
    y = rpc.rpc_async('worker1', func = pruned_echo_request_with_cached_size_restoration, args = (x[:, is_input_channel_unpruned, :, :],)).wait()


def main(rank: int):
    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=2,
    )

    if rank == 0:
        log_file = open('/log/runtime.txt', mode = 'a')
        log_file.write('\n\n### RPC Tensor Send/Recv Overhead Measurement ###\n# no pruning\n')
        log_average_time(task1, 'normal echo request', log_file)

        for prune_ratio in [0.5, 0.9, 1.0]:
            prune_input_channels(prune_ratio)

            log_file.write(f'# prune ratio: {100 * prune_ratio}%\n')
            log_average_time(task2, 'pruned echo request', log_file)
            log_average_time(task3, 'pruned echo request with size restoration', log_file)
            log_average_time(task4, 'pruned echo request with cached size restoration', log_file)

    rpc.shutdown()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--rank", type=int, default=0)
    args.add_argument("--addr", type=str, default='host.docker.internal')
    args = args.parse_args()
    rank = args.rank
    master_addr = args.addr

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '29500'

    main(rank=rank)