import torch
import torch.nn as nn
import time
from typing import List
import torch.distributed.rpc as rpc

class RPCWorkerNode:
    def __init__(self, rank: int):
        self.rank = rank

    def request_inference(self, x: torch.Tensor, original_layer_name: str, part_index: int):
        self.future_result = rpc.rpc_async(
            f"worker{self.rank}",
            func="rpc_worker_node_inference",
            args=(x, original_layer_name, part_index),
        )

    def receive_inference_result(self):
        return self.future_result.wait()



def rpc_worker_node_inference(x: torch.Tensor, original_layer_name: str, part_index: int):
    with torch.no_grad():
        start = time.time()
        result = x * 0.5  # 예제 계산
        end = time.time()
        return (end - start, result.shape)


class SplitResnet(nn.Module):
    def __init__(self, model: nn.Module, num_splits: int):
        super(SplitResnet, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class DistributedResnet(nn.Module):
    def __init__(self, split_model: SplitResnet, worker_nodes: List[RPCWorkerNode], is_sequential: bool = False):
        super(DistributedResnet, self).__init__()
        self.split_model = split_model
        self.worker_nodes = worker_nodes
        self.is_sequential = is_sequential

    def forward(self, x):
        return self.split_model(x)
