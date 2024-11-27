import torch.distributed.rpc as rpc
import argparse
import os
import time
import torch
import torchvision.models as models
import copy
from typing import List
import functools
from runtime import RuntimeRecord



class SplitConv2d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d, out_channels_per_part: list[int]):
        """
        :param conv: 원본 레이어
        :param out_channels_per_part: 쪼개진 레이어가 각각 담당할 출력 채널의 수

        .. example::
        SplitConv2d(conv, [2, 3, 10]) => 출력 채널의 수가 각각 2, 3, 10인 세 개의 conv 레이어로 쪼개기
        """
        super(SplitConv2d, self).__init__()

        # 사전 조건:
        # - 총 파트 수는 2 이상
        # - 각 파트가 분담할 출력 채널 수의 합이 원본과 동일
        # - 각 파트가 1개 이상의 출력 채널을 분담
        assert(len(out_channels_per_part) >= 2)
        assert(sum(out_channels_per_part) == conv.out_channels)
        for out_channel in out_channels_per_part:
            assert(out_channel > 0)

        self.out_channels_per_part = out_channels_per_part
        self.partial_convs = torch.nn.ModuleList()

        # 이 값은 루프가 끝날 때마다 직전 파트가 담당하는 마지막 채널의 인덱스 + 1으로 설정됨.
        # 그러므로, 이번 파트의 첫 번째 채널 인덱스로 해석할 수 있음.
        out_channel_begin = 0
        for num_channels in out_channels_per_part:
            # 이번 파트가 담당할 채널의 경계 (마지막 채널 인덱스 + 1)
            out_channel_end = out_channel_begin + num_channels

            self.partial_convs.append(PartialConv2d(conv, out_channel_begin, out_channel_end))

            # 다음 파트의 첫 채널은 내 마지막 채널의 바로 다음
            out_channel_begin = out_channel_end

        # 사후 조건:
        # - 채널의 경계 인덱스가 총 채널 수와 일치해야 함
        assert(out_channel_begin == conv.out_channels)
    
    def forward(self, x: torch.Tensor):
        # partial_convs의 원소들은 각각 원본 모델의 출력 채널 중 일부 범위를 담당함.
        # x는 4차원 batch형태로 주어지기 때문에 dim=1이 출력 채널을 의미함.
        # 즉, 쪼개진 conv들이 각각 계산한 출력 채널을 차곡차곡 포갠 것을 최종 출력으로 사용하는 것.
        return torch.cat([conv(x) for conv in self.partial_convs], dim=1)

class PartialConv2d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d, out_channel_begin: int, out_channel_end: int):
        """
        :param conv: 원본 레이어

        .. note::
        생성된 분할 레이어는 원본 출력 채널의 [out_channel_begin, out_channel_end) 구간을 담당한다.  
        ex) PartialConv2d(conv, 4, 10) => 5 ~ 10번째 출력 채널을 담당 (i.e., conv.weight[4:10])
        """
        super(PartialConv2d, self).__init__()
        # 출력 채널의 일정 범위만 담당하는 작은 conv 레이어 만들기
        out_channels = out_channel_end - out_channel_begin
        self.conv = torch.nn.Conv2d(conv.in_channels, out_channels, conv.kernel_size, conv.stride, conv.padding, bias = conv.bias is not None)

        # 원본 레이어의 가중치 복사해서 넣기 (bias는 없는 경우도 있음)
        self.conv.weight.data = conv.weight[out_channel_begin:out_channel_end]
        if conv.bias is not None:
            self.conv.bias.data = conv.bias[out_channel_begin:out_channel_end]

    def forward(self, x: torch.Tensor):
        return self.conv(x)



class PartialConv2dProxy(torch.nn.Module):
    """
    WorkerNode를 통해 연산을 처리하는 PartialConv2d의 프록시
    """
    def __init__(self, original_layer_name: str, part_index: int, worker_node: 'WorkerNode', runtime_record: RuntimeRecord):
        super(PartialConv2dProxy, self).__init__()
        self.original_layer_name = original_layer_name
        self.part_index = part_index
        self.worker_node = worker_node
        self.runtime_record = runtime_record
        self.local_computation = runtime_record.create_subcategory('local computation')
        self.network_overhead = runtime_record.create_subcategory('network overhead')

    def forward(self, x: torch.Tensor):
        start = time.time()
        self.worker_node.request_inference(x, self.original_layer_name, self.part_index)
        runtime, result = self.worker_node.receive_inference_result()
        end = time.time()
        self.record_runtime(total_runtime=end - start, local_computation_runtime=runtime)
        return result

    def request_inference(self, x: torch.Tensor):
        self.request_start_time = time.time()
        self.worker_node.request_inference(x, self.original_layer_name, self.part_index)

    def receive_inference_result(self):
        runtime, result = self.worker_node.receive_inference_result()
        request_end_time = time.time()
        self.record_runtime(total_runtime=request_end_time - self.request_start_time, local_computation_runtime=runtime)
        return result

    def record_runtime(self, total_runtime: float, local_computation_runtime: float):
        self.runtime_record.total_runtime = total_runtime
        self.local_computation.total_runtime = local_computation_runtime
        self.network_overhead.total_runtime = total_runtime - local_computation_runtime


class SplitResnet(torch.nn.Module):
    """
    각 노드마다 생성되는 resnet 분할 버전.
    마스터 서버의 WorkerNode 클래스가 여기에 접속해서 추론 요청을 보냄.
    """
    def __init__(self, model: models.ResNet, num_splits: int):
        super(SplitResnet, self).__init__()
        self.model = copy.deepcopy(model)
        self.split_conv_dict = {}
        self.num_splits = num_splits

        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                self.split_conv_dict[name] = self.create_split_conv(layer)

        for name, layer in self.split_conv_dict.items():
            rsetattr(self.model, name, layer)
    
    def create_split_conv(self, layer: torch.nn.Conv2d):
        """
        원본 Conv2d 레이어를 여러개로 분할.
        총 출력 채널 수와 원하는 분할 수를 기반으로 각 부분의 채널 수를 계산하고,
        이를 토대로 SplitConv2d 인스턴스를 생성 및 반환.
        """
        total_channels = layer.out_channels
        out_channels_per_part = self.distribute_channels(total_channels, self.num_splits)
        return SplitConv2d(layer, out_channels_per_part)

    @staticmethod
    def distribute_channels(total_channels: int, num_splits: int) -> List[int]:
        """
        주어진 총 채널 수를 지정된 분할 수에 따라 최대한 균등하게 분배.
        :param total_channels: 분배할 총 채널의 수
        :param num_splits: 채널을 분배할 분할의 수

        동작방법: 모든 분할에 기본적인 채널 수를 할당하고 (총 채널/분할 수), 
        이후 앞에서부터 나머지 채널 수를 하나씩 할당.
        """
        base_channels = total_channels // num_splits
        remainder = total_channels % num_splits

        out_channels_per_part = [base_channels] * num_splits
        
        # 나머지 채널 수를 앞에서부터 하나씩 할당
        for i in range(remainder):
            out_channels_per_part[i] += 1

        return out_channels_per_part

    def forward(self, x: torch.Tensor):
        return self.model(x)


class DistributedConv2d(torch.nn.Module):
    """
    SplitConv2d의 분산 처리 버전.
    PartialConv2d를 직접 계산하는 대신 대응되는 worker node에
    요청을 보내는 PartialConv2dProxy를 사용한다.
    """
    def __init__(self, original_layer_name: str, worker_nodes: List['WorkerNode'], is_sequential: bool, runtime_record: RuntimeRecord):
        super(DistributedConv2d, self).__init__()
        self.is_sequential = is_sequential
        self.partial_convs = torch.nn.ModuleList()
        self.runtime_record = runtime_record
        self.partial_convs_runtime = runtime_record.create_subcategory('partial convs')
        self.concat_runtime = runtime_record.create_subcategory('concat')
        for part_index, worker_node in enumerate(worker_nodes):
            worker_runtime = self.partial_convs_runtime.create_subcategory(f'worker {part_index}')
            self.partial_convs.append(PartialConv2dProxy(original_layer_name, part_index, worker_node, worker_runtime))
    
    def forward(self, x: torch.Tensor):
        start = time.time()
        if self.is_sequential:
            partial_outputs = [conv(x) for conv in self.partial_convs]
        else:
            for conv in self.partial_convs:
                conv.request_inference(x)
            partial_outputs = [conv.receive_inference_result() for conv in self.partial_convs]
        
        middle = time.time()
        net_output = torch.cat(partial_outputs, dim=1)
        end = time.time()
        self.runtime_record.total_runtime = end - start
        self.partial_convs_runtime.total_runtime = middle - start
        self.concat_runtime.total_runtime = end - middle
        return net_output


class DistributedResnet(torch.nn.Module):
    def __init__(self, split_model: 'SplitResnet', worker_nodes: List[WorkerNode], is_sequential: bool = False):
        super(DistributedResnet, self).__init__()
        self.split_model = copy.deepcopy(split_model)
        self.runtime_record = RuntimeRecord('DistributedResnet')
        self.worker_nodes = worker_nodes
        self.is_sequential = is_sequential

        for name, layer in self.split_model.split_conv_dict.items():
            rsetattr(self.split_model.model, name, self.create_distributed_conv(name, layer, is_sequential))
    
    def create_distributed_conv(self, name: str, layer: 'SplitConv2d', is_sequential: bool):
        num_splits = len(layer.out_channels_per_part)
        if num_splits > len(self.worker_nodes):
            raise ValueError(f"Not enough worker nodes for layer {name}. Required: {num_splits}, Available: {len(self.worker_nodes)}")
        
        return DistributedConv2d(name, self.worker_nodes[:num_splits], is_sequential, self.runtime_record.create_subcategory(name))

    def forward(self, x: torch.Tensor):
        start = time.time()
        result = self.split_model(x)
        end = time.time()
        self.runtime_record.total_runtime = end - start
        return result



def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))




num_workers = 3
world_size = num_workers + 1 # 마스터 서버까지 포함하니까 +1

# 마스터, 워커 모두 필요한 모델 만들어두기
orig = models.resnet50(pretrained = True)
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