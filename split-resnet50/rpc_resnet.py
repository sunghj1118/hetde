from test import *
import torch.distributed.rpc as rpc
import os
import argparse

num_workers = 3
world_size = num_workers + 1 # 마스터 서버까지 포함하니까 +1

# 마스터, 워커 모두 필요한 모델 만들어두기
orig = models.resnet50(pretrained = True)
split = SplitResnet(orig, num_workers)


def rpc_worker_node_inference(x: torch.Tensor, original_layer_name: str, part_index: int):
    with torch.no_grad():
        start = time.time()
        y = split.split_conv_dict[original_layer_name].partial_convs[part_index](x)
        end = time.time()
        return ((end - start), y)


class RPCWorkerNode:
    def __init__(self, rank: int):
        self.rank = rank
    
    def request_inference(self, x: torch.Tensor, original_layer_name: str, part_index: int):
        self.future_result = rpc.rpc_async(f'worker{self.rank}', func = rpc_worker_node_inference, args = (x, original_layer_name, part_index))
    
    def receive_inference_result(self):
        return self.future_result.wait()


def main(rank: int):
    # Initialize RPC for each device role
    print(f"Rank {rank} initialized RPC")

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
    )
    print('All workers initialized RPC', time.time())

    # 마스터 서버만 실행하는 구간
    if rank == 0:
        print('Started master server routine')
        worker_nodes = [RPCWorkerNode(i + 1) for i in range(num_workers)]
        distributed = DistributedResnet(split, worker_nodes, is_sequential = False)
        
        input_shape = [1, 3, 256, 256]
        assert_model_equality(orig, distributed, input_shape, num_tests = 5)

        distributed.analyze_overheads(input_shape, num_tests = 5, outer_tqdm_progress = None)

        print(f"parallel version distributed part total runtime: {distributed.runtime_record.net_runtime_per_category('partial convs'):.7f}")
        for i in range(len(worker_nodes)):
            print(f"worker node {i} total runtime: {distributed.runtime_record.net_runtime_per_category(f'worker {i}'):.7f}")

        print('Finished master server routine')

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