import copy
import torch
import torchvision.models as models
from runtime import RuntimeRecord

 

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
    실제로 연산을 진행할 장치와의 통신 로직을 감싸는 클래스.
    마스터 서버에서 이 객체를 통해 입력 데이터를 전달하고 추론 결과를 수신한다.
    """
    def __init__(self, host: str, port: int):
        self.sock = tcp.connect_server(host, port)

    def request_inference(self, x: torch.Tensor, original_layer_name: str, part_index: int):
        tcp.send_json(self.sock, {'original_layer_name' : original_layer_name, 'part_index' : part_index})
        tcp.send_tensor(self.sock, x)

    def receive_inference_result(self):
        return tcp.recv_json(self.sock), tcp.recv_tensor(self.sock)


class PartialConv2dProxy(torch.nn.Module):
    """
    WorkerNode를 통해 연산을 처리하는 PartialConv2d의 프록시
    """
    def __init__(self, original_layer_name: str, part_index: int, worker_node: WorkerNode, runtime_record: RuntimeRecord):
        """
        :param original_layer_name: 쪼개지기 전 원본 conv 레이어가 모델 전체 시점에서 갖는 이름.
        :param part_index: 쪼개진 부분 중에서 몇 번째 출력 채널 범위를 계산할 것인지.
        :param runtime_record: 해당 레이어의 실행 시간 세부 사항을 기록할 구조체.

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
        self.runtime_record = runtime_record
        self.local_computation = runtime_record.create_subcategory('local computation')
        self.network_overhead = runtime_record.create_subcategory('network overhead')

    def forward(self, x: torch.Tensor):
        """
        worker node에 요청을 보내고 응답을 기다리는 blocking 방식의 함수.
        이 함수를 사용해 inference를 진행하면 병렬 처리가 불가능하니
        여러 worker를 사용하는 경우 request_inference()를 사용하는 것을 권장함.
        """
        start = time.time()

        self.worker_node.request_inference(x, self.original_layer_name, self.part_index)
        header, result = self.worker_node.receive_inference_result()

        end = time.time()
        self.record_runtime(total_runtime = end - start, local_computation_runtime = header['time'])
        return result
    
    def request_inference(self, x: torch.Tensor):
        """
        forward()는 worker node에서 응답이 올 때까지 실행이 멈추는 blocking operation이라서
        병렬로 모든 worker에 요청만 보내는 함수가 필요함.
        여기서 요청을 보내면 receive_inference_result() 함수로 응답을 기다릴 수 있다.
        """
        self.request_start_time = time.time()
        self.worker_node.request_inference(x, self.original_layer_name, self.part_index)

    def receive_inference_result(self):
        """
        request_inference()와 한 쌍으로 사용되는 함수.
        조금 전에 보낸 요청에 대한 응답을 받고 실행 시간을 기록한다.
        """
        header, result = self.worker_node.receive_inference_result()

        request_end_time = time.time()
        self.record_runtime(total_runtime = request_end_time - self.request_start_time, local_computation_runtime = header['time'])

        return result

    def record_runtime(self, total_runtime: float, local_computation_runtime: float):
        """
        이번 inference에 소요된 시간을 기록한다.

        :param total_runtime: 요청 시작부터 응답을 받은 시점까지의 시간
        :param local_computation_runtime: worker node에서 요청을 받은 뒤부터 응답을 전송하기 직전까지 걸린 순수 연산 시간
        """
        self.runtime_record.total_runtime = total_runtime
        self.local_computation.total_runtime = local_computation_runtime
        self.network_overhead.total_runtime = total_runtime - local_computation_runtime



class DistributedConv2d(torch.nn.Module):
    """
    SplitConv2d의 분산 처리 버전.
    PartialConv2d를 직접 계산하는 대신 대응되는 worker node에
    요청을 보내는 PartialConv2dProxy를 사용한다.
    """
    def __init__(self, original_layer_name: str, worker_nodes: list[WorkerNode], runtime_record: RuntimeRecord):
        super(DistributedConv2d, self).__init__()

        self.partial_convs = torch.nn.ModuleList()
        self.runtime_record = runtime_record
        self.partial_convs_runtime = runtime_record.create_subcategory('partial convs')
        self.concat_runtime = runtime_record.create_subcategory('concat')
        for part_index, worker_node in enumerate(worker_nodes):
            worker_runtime = self.partial_convs_runtime.create_subcategory('worker {}'.format(part_index))
            self.partial_convs.append(PartialConv2dProxy(original_layer_name, part_index, worker_node, worker_runtime))
    
    def forward(self, x: torch.Tensor):
        start = time.time()

        # 연결된 워커 노드에 부분적인 inference를 요청
        sequential = False
        if sequential:
            # case 1) 순차적 요청: 1번 워커의 응답이 온 뒤에 2번 워커에 요청을 보냄
            partial_outputs = [conv(x) for conv in self.partial_convs]
        else:
            # case 2) 병렬적 요청: 요청을 일단 전부 보내놓고 응답은 모든 요청이 전송된 뒤에 순차적으로 수신
            for conv in self.partial_convs:
                conv.request_inference(x)

            partial_outputs = [conv.receive_inference_result() for conv in self.partial_convs]
        
        middle = time.time()

        # 부분적인 inference 결과를 포개서 최종 출력 재현
        net_output = torch.cat(partial_outputs, dim=1)

        end = time.time()

        # 실행 시간 기록
        self.runtime_record.total_runtime = end - start
        self.partial_convs_runtime.total_runtime = middle - start
        self.concat_runtime.total_runtime = end - middle
        
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
        self.runtime_record = RuntimeRecord('DistributedResnet')

        for name, layer in self.split_model.split_conv_dict.items():
            rsetattr(self.split_model.model, name, DistributedConv2d(name, worker_nodes, self.runtime_record.create_subcategory(name)))

    def forward(self, x: torch.Tensor):
        start = time.time()
        result = self.split_model(x)
        end = time.time()
        self.runtime_record.total_runtime = end - start
        return result


from tqdm import tqdm
from multiprocessing import Process
import time
import os
from PIL import Image
import torchvision.transforms as transforms

image_repo_path = "imagenet-sample-images"
class_mapping_path = "imagenet1000_clsidx_to_labels.txt"

def load_class_mapping(mapping_path):
    with open(mapping_path, 'r') as f:
        class_mapping = eval(f.read())
    return class_mapping

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(280),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor


def calculate_accuracy_and_latency(model, image_files, class_mapping):
    correct = 0
    total = 0
    latencies = []
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(total=len(image_files), desc="Calculating accuracy and latency")

        for image_file in image_files:
            input_tensor = load_image(image_file)

            start_time = time.time()
            outputs = model(input_tensor)
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)

            # 파일 이름에서 WordNet ID와 실제 정답 클래스 추출
            file_name = os.path.basename(image_file)  # 파일명 예: 'n01440764_tench.JPEG' 또는 'n01622779_great_grey_owl.JPEG'
            true_label = '_'.join(file_name.split('_')[1:]).split('.')[0]  # 'tench' 또는 'great_grey_owl'
            true_label = true_label.replace('_', ' ')

            # 예측한 클래스 선택
            _, predicted = torch.max(outputs, 1)
            predicted_label = predicted.item()

            # 예측한 클래스가 실제 클래스와 일치하는지 확인 (여기서는 정답 클래스 true_label과 비교)
            if true_label in class_mapping[predicted_label]:
                correct += 1

            total += 1
            progress_bar.update(1)

        progress_bar.close()

    # 정확도 및 평균 latency 계산
    accuracy = 100 * correct / total
    average_latency = sum(latencies) / len(latencies)

    return accuracy, average_latency



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
    assert_model_equality(orig, split, input_shape = [1, 3, 256, 256], num_tests = 1)

def test_worker_node_server(port: int):
    try:
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

                    # 네트워크 딜레이 재현
                    # time.sleep(0.02)
                    
                    tcp.send_json(client_sock, {'time' : end - start})
                    tcp.send_tensor(client_sock, y)

            client_sock.close()
    except:
        print('[Exception from worker node]')
        tcp.traceback.print_exc()

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

    try:
        orig = models.resnet50()
        split = SplitResnet(orig)

        worker_nodes = [
            WorkerNode('localhost', 1111),
            WorkerNode('localhost', 2222),
        ]
        distributed = DistributedResnet(split, worker_nodes)

        class_mapping = load_class_mapping(class_mapping_path)
        image_files = [os.path.join(image_repo_path, f) for f in os.listdir(image_repo_path) if f.endswith('.JPEG')]
    
    
        # 원본 ResNet의 정확도 및 latency 측정
        orig_accuracy, orig_latency = calculate_accuracy_and_latency(orig, image_files, class_mapping)
        print(f"Original ResNet Accuracy: {orig_accuracy:.2f}%")
        print(f"Original ResNet Latency: {orig_latency:.6f} seconds")

        # Distributed ResNet의 정확도 및 latency 측정
        distributed_accuracy, distributed_latency = calculate_accuracy_and_latency(distributed, image_files, class_mapping)
        print(f"Distributed ResNet Accuracy: {distributed_accuracy:.2f}%")
        print(f"Distributed ResNet Latency: {distributed_latency:.6f} seconds")
        print("Completed testing for all classes.")
      

        distributed.runtime_record.print_runtime()
        print('net local computation:', distributed.runtime_record.net_runtime_per_category('local computation'))
        print('net network overhead:', distributed.runtime_record.net_runtime_per_category('network overhead'))
        print('net concat:', distributed.runtime_record.net_runtime_per_category('concat'))

        tcp.send_json(worker_nodes[0].sock, {'terminate' : True})
        tcp.send_json(worker_nodes[1].sock, {'terminate' : True})
    except:
        print('[Exception from main server]')
        tcp.traceback.print_exc()

    worker1.join()
    worker2.join()


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('assert_distributed_resnet_correctness()', sort = 'tottime')
    tcp.assert_tcp_communication()
    assert_distributed_resnet_correctness()
    assert_split_resnet_correctness()
    assert_split_conv_correctness()