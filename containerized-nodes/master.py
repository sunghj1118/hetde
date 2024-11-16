import torch
import copy
import time
import sys
from tqdm import tqdm
from typing import List
import tcp
import torchvision.models as models
from runtime import RuntimeRecord
import functools
from typing import Callable


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


class WorkerNode:
    """
    마스터 노드에서 각 워커 노드와 통신을 담당하는 클래스
    """
    def __init__(self, host: str, port: int):
        print(f"Attempting to connect to {host}:{port}...")
        self.sock = tcp.connect_server(host, port)
        print(f"Successfully connected to {host}:{port}")

    #어디서부터 어디까지 계산할지 보내는 부분 추가
    def request_inference(self, x: torch.Tensor, original_layer_name: str, part_index: int):
        tcp.send_utf8(self.sock, original_layer_name)
        tcp.send_u32(self.sock, part_index)
        tcp.send_tensor(self.sock, x)

    def receive_inference_result(self):
        return tcp.recv_f64(self.sock), tcp.recv_tensor(self.sock)


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



def measure_distributed_resnet_overheads(worker_nodes: list[WorkerNode], orig: models.ResNet, split: SplitResnet):
    progress = tqdm(total = 2, desc = 'initializing distributed models', file = sys.stdout, leave = False)

    progress.set_postfix_str('sequential version')
    distributed_sequential = DistributedResnet(split, worker_nodes, is_sequential = True)
    progress.update()

    progress.set_postfix_str('parallel version')
    distributed_parallel = DistributedResnet(split, worker_nodes, is_sequential = False)
    progress.update()
    progress.close()


    # 네트워크 오버헤드 분석
    input_shape = [1, 3, 256, 256]

    progress = tqdm(total = 2, desc = 'measuring distributed model overhead', file = sys.stdout, position = 0, leave = False)
    progress.set_postfix_str('running sequential version')
    distributed_sequential.analyze_overheads(input_shape, num_tests = 5, outer_tqdm_progress = progress)

    progress.set_postfix_str('running parallel version')
    distributed_parallel.analyze_overheads(input_shape, num_tests = 5, outer_tqdm_progress = progress)
    progress.close()

    print(f"sequential version distributed part total runtime: {distributed_sequential.runtime_record.net_runtime_per_category('partial convs'):.7f}")
    for i in range(len(worker_nodes)):
        print(f"worker node {i} total runtime: {distributed_sequential.runtime_record.net_runtime_per_category(f'worker {i}'):.7f}")

    print(f"parallel version distributed part total runtime: {distributed_parallel.runtime_record.net_runtime_per_category('partial convs'):.7f}")
    for i in range(len(worker_nodes)):
        print(f"worker node {i} total runtime: {distributed_parallel.runtime_record.net_runtime_per_category(f'worker {i}'):.7f}")

def test_worker_node_server(port: int, split: SplitResnet):
    try:
        with tcp.create_server('localhost', port) as server:
            client_sock, client_addr = server.accept()

            while True:
                original_layer_name = tcp.recv_utf8(client_sock)
                if original_layer_name == 'terminate':
                    break
                
                part_index = tcp.recv_u32(client_sock)
                x = tcp.recv_tensor(client_sock)
                with torch.no_grad():
                    start = time.time()
                    y = split.split_conv_dict[original_layer_name].partial_convs[part_index](x)
                    end = time.time()

                    # 네트워크 딜레이 재현
                    # time.sleep(0.02)
                    
                    tcp.send_f64(client_sock, end - start)
                    tcp.send_tensor(client_sock, y)

            client_sock.close()
    except:
        print('[Exception from worker node]')
        tcp.traceback.print_exc()


#원래 함수랑 달리 수정함 마스터노드에서 보내는걸로
def run_test_on_distributed_env(num_workers: int, test: Callable[[List[WorkerNode], models.ResNet, SplitResnet], None], port_offset: int = 1001, use_pretrained_resnet=False):
    tcp.supress_immutable_tensor_warning()

    progress = tqdm(total=2, desc='initializing worker node model', file=sys.stdout, leave=False)
    orig = models.resnet50(pretrained=use_pretrained_resnet)
    progress.update()
    split = SplitResnet(orig, num_workers)
    progress.update()

    try:
        worker_nodes = [WorkerNode(f'worker_node_{i + 1}', port_offset + i) for i in tqdm(range(num_workers), desc='connecting to worker nodes', file=sys.stdout, leave=False)]

        test(worker_nodes, orig, split)

        # 모든 worker node 종료
        for worker_node in worker_nodes:
            tcp.send_utf8(worker_node.sock, 'terminate')
    except Exception as e:
        print(f'[Exception from main server]: {e}')
        tcp.traceback.print_exc()

def test_master_worker_communication(worker_nodes: List[WorkerNode], orig: models.ResNet, split: SplitResnet):
    """
    마스터와 워커 노드 간의 기본 통신을 테스트하기 위한 함수
    각 워커 노드로 데이터를 보내고, 올바른 결과를 받는지 확인
    """
    print("\nStarting Master-Worker Communication Test...\n")

    # 임의의 데이터 생성
    x = torch.randn((1, 3, 224, 224))

    for worker_node in worker_nodes:
        try:
            worker_node.request_inference(x, "conv1", 0)  # conv1에서 첫 번째 부분 계산 요청
            runtime, result = worker_node.receive_inference_result()

            # 테스트 결과 출력
            print(f"Received result from worker node: Runtime = {runtime:.4f} sec, Result Shape = {result.shape}")
        except Exception as e:
            print(f"Error during communication with worker node: {e}")


if __name__ == '__main__':
    run_test_on_distributed_env(num_workers=2, test=test_master_worker_communication)