import copy
import torch
import torchvision.models as models


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

import tcp

class WorkerNode:
    """
    실제로 연산을 진행할 장치와의 통신 로직을 감싸는 클래스
    """
    def __init__(self, host: str, port: int):
        self.sock = tcp.connect_server(host, port)

    def request_inference(self, x: torch.Tensor, original_layer_name: str, part_index: int):
        tcp.send_json(self.sock, {'original_layer_name' : original_layer_name, 'part_index' : part_index})
        tcp.send_tensor(self.sock, x)
        return tcp.recv_json(self.sock), tcp.recv_tensor(self.sock)


class PartialConv2dProxy(torch.nn.Module):
    """
    WorkerNode를 통해 연산을 처리하는 PartialConv2d의 프록시
    """
    def __init__(self, original_layer_name: str, part_index: int, worker_node: WorkerNode):
        """
        :param original_layer_name: 쪼개지기 전 원본 conv 레이어가 모델 전체 시점에서 갖는 이름.
        :param part_index: 쪼개진 부분 중에서 몇 번째 출력 채널 범위를 계산할 것인지.

        .. example::
        split = SplitConv2d(model.conv1, [2, 4, 10])가 있다고 하면  
        partial = PartialConv2dProxy('conv1', 1, worker_node)는  
        split의 출력 채널 4개짜리 부분을 worker_node에서 계산하겠다는 것

        .. note::
        original_layer_name은 model.named_modules()를 순회하면 얻을 수 있다
        """
        super(PartialConv2dProxy, self).__init__()
        self.original_layer_name = original_layer_name
        self.part_index = part_index
        self.worker_node = worker_node

    def forward(self, x: torch.Tensor):
        start = time.time()
        header, result = self.worker_node.request_inference(x, self.original_layer_name, self.part_index)
        end = time.time()
        total_time = end - start
        network_overhead = total_time - header['time']
        print('PartialConv2dProxy [{}]: {:.7f} (network overhead: {:.2f}%)'.format(self.original_layer_name, total_time, network_overhead / total_time * 100))
        return result


class DistributedConv2d(torch.nn.Module):
    def __init__(self, original_layer_name: str, worker_nodes: list[WorkerNode]):
        super(DistributedConv2d, self).__init__()

        self.partial_convs = torch.nn.ModuleList()
        for part_index, worker_node in enumerate(worker_nodes):
            self.partial_convs.append(PartialConv2dProxy(original_layer_name, part_index, worker_node))
    
    def forward(self, x: torch.Tensor):
        start = time.time()
        partial_outputs = [conv(x) for conv in self.partial_convs]
        middle = time.time()
        net_output = torch.cat(partial_outputs, dim=1)
        end = time.time()
        print('DistributedConv2D [{}]: {:.7f} (partial convs: {:.2f}%, concat: {:.2f}%)'.format(self.partial_convs[0].original_layer_name, end - start, (middle - start) / (end - start) * 100, (end - middle) / (end - start) * 100))
        return net_output



# ResNet 모델의 모든 Conv2d 멤버 변수를 SplitConv2d로 교체하기 위한 recursive setattr()
# 출처: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
import functools

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class SplitResnet(torch.nn.Module):
    """
    각 노드마다 생성되는 resnet 분할 버전.
    마스터 서버의 WorkerNode 클래스가 여기에 접속해서 추론 요청을 보냄.
    """
    def __init__(self, model: models.ResNet):
        super(SplitResnet, self).__init__()
        self.model = copy.deepcopy(model)
        self.split_conv_dict = {}

        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                # TODO: 정확히 반반 말고 임의의 수와 비율로 쪼갤 수 있게 만들기
                self.split_conv_dict[name] = SplitConv2d(layer, out_channels_per_part = [layer.out_channels // 2, layer.out_channels // 2])

        for name, layer in self.split_conv_dict.items():
            rsetattr(self.model, name, layer)

    def forward(self, x: torch.Tensor):
        return self.model(x)


    
class DistributedResnet(torch.nn.Module):
    def __init__(self, split_model: SplitResnet, worker_nodes: list[WorkerNode]):
        super(DistributedResnet, self).__init__()
        self.split_model = split_model

        for name, layer in self.split_model.split_conv_dict.items():
            rsetattr(self.split_model.model, name, DistributedConv2d(name, worker_nodes))

    def forward(self, x: torch.Tensor):
        return self.split_model(x)


from tqdm import tqdm
from multiprocessing import Process
import time

def assert_model_equality(model1: torch.nn.Module, model2: torch.nn.Module, input_shape: torch.Size, num_tests: int = 100):
    model1.eval()
    model2.eval()
    with torch.no_grad():
        desc = 'model equality assertion ({} vs {})'.format(model1._get_name(), model2._get_name())
        for _ in tqdm(range(num_tests), desc):
            x = torch.rand(input_shape)
            y1 = model1(x)
            y2 = model2(x)
            assert(torch.equal(y1, y2))


def assert_split_conv_correctness():
    orig = torch.nn.Conv2d(3, 64, 7) # 랜덤 초기화된 conv 레이어
    split1 = SplitConv2d(orig, [32, 32])
    split2 = SplitConv2d(orig, [2, 30, 32])
    split3 = SplitConv2d(orig, [2, 1, 3, 22, 4, 16, 16])

    input_shape = [1, 3, 32, 32]
    assert_model_equality(orig, split1, input_shape)
    assert_model_equality(orig, split2, input_shape)
    assert_model_equality(orig, split3, input_shape)


def assert_split_resnet_correctness():
    orig = models.resnet50()
    split = SplitResnet(orig)
    assert_model_equality(orig, split, input_shape = [1, 3, 256, 256], num_tests = 20)

def test_worker_node_server(port: int):
    with tcp.create_server('localhost', port) as server:
        client_sock, client_addr = server.accept()

        orig = models.resnet50()
        split = SplitResnet(orig)

        while True:
            header = tcp.recv_json(client_sock)
            if 'terminate' in header:
                break

            x = tcp.recv_tensor(client_sock)
            with torch.no_grad():
                start = time.time()
                y = split.split_conv_dict[header['original_layer_name']].partial_convs[header['part_index']](x)
                end = time.time()
                tcp.send_json(client_sock, {'time' : end - start})
                tcp.send_tensor(client_sock, y)

        client_sock.close()

def assert_distributed_resnet_correctness():
    tcp.supress_immutable_tensor_warning()

    progress = tqdm(total = 1, desc = 'initializing worker nodes')
    worker1 = Process(target = test_worker_node_server, args = (1111,))
    worker2 = Process(target = test_worker_node_server, args = (2222,))
    worker1.start()
    worker2.start()

    time.sleep(1)

    progress.update()
    progress.close()

    orig = models.resnet50()
    split = SplitResnet(orig)

    worker_nodes = [
        WorkerNode('localhost', 1111),
        WorkerNode('localhost', 2222),
    ]
    distributed = DistributedResnet(split, worker_nodes)

    assert_model_equality(orig, distributed, [1, 3, 256, 256], num_tests = 5)

    tcp.send_json(worker_nodes[0].sock, {'terminate' : True})
    tcp.send_json(worker_nodes[1].sock, {'terminate' : True})

    worker1.join()
    worker2.join()


if __name__ == '__main__':
    tcp.assert_tcp_communication()
    assert_distributed_resnet_correctness()
    assert_split_resnet_correctness()
    assert_split_conv_correctness()