from test import *
import torch.distributed.rpc as rpc
import os
import argparse

num_workers = 3
world_size = num_workers + 1 # 마스터 서버까지 포함하니까 +1

# 마스터, 워커 모두 필요한 모델 만들어두기
orig = models.resnet50(pretrained = True)
prune_all_conv_layers(orig, 0.95)

split = SplitResnet(orig, num_workers)


def rpc_worker_node_inference(x: torch.Tensor, original_layer_name: str, part_index: int):
    with torch.no_grad():
        start = time.time()

        # 1. prune된 채널에 0 채워넣기
        split_conv = split.split_conv_dict[original_layer_name]

        if split_conv.restored_x is None:
            original_shape = list(x.shape)
            original_shape[1] = len(split_conv.is_input_channel_unpruned)
            split_conv.restored_x = torch.zeros(original_shape)
        
        split_conv.restored_x[:, split_conv.is_input_channel_unpruned, :, :] = x


        # 2. 원래 입력 사이즈로 계산하기
        y = split_conv.partial_convs[part_index](split_conv.restored_x)

        end = time.time()

        # 혹시 데이터 전송 부분에서 실수했나 싶어서 확인하려고 놔둔 출력
        # 분명 prune된 입력 채널은 다 빼고 전송했는데 왜 속도가 전이랑 비슷하지???
        # print(f'{original_layer_name}:{part_index} received unpruned {x.shape[1] / split_conv.restored_x.shape[1] * 100}% of original input')
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
        distributed = DistributedResnet(split, worker_nodes, is_sequential = True)
        
        input_shape = [1, 3, 256, 256]
        # assert_model_equality(orig, distributed, input_shape, num_tests = 5)

        distributed.analyze_overheads(input_shape, num_tests = 5, outer_tqdm_progress = None)

        print(f"distributed part total runtime: {distributed.runtime_record.net_runtime_per_category('partial convs'):.7f}")
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