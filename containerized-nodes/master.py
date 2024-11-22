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

    # 워커 노드 연결
    worker_nodes = []
    max_retries = 2  # 재시도 최대 횟수
    retry_delay = 5  # 각 재시도 사이의 대기 시간

    for i in range(num_workers):
        host = f'worker_node_{i + 1}'
        port = port_offset + i
        connected = False

        for attempt in range(max_retries):
            try:
                tqdm.write(f"Attempting to connect to {host}:{port} (Attempt {attempt + 1}/{max_retries})...")
                worker_node = WorkerNode(host, port)
                worker_nodes.append(worker_node)
                connected = True
                tqdm.write(f"Successfully connected to {host}:{port}.")
                break
            except Exception as e:
                tqdm.write(f"Connection failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        if not connected:
            tqdm.write(f"Failed to connect to worker node {host}:{port} after {max_retries} attempts. Exiting.")
            return

    print("모든 워커 노드에 성공적으로 연결되었습니다.")
    progress.update()

    try:
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


#여기서부터 이미지 latency test
import time
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

image_repo_path = "imagenet-sample-images"
class_mapping_path = "942d3a0ac09ec9e5eb3a/imagenet1000_clsidx_to_labels.txt"

def load_class_mapping(mapping_path):
    with open(mapping_path, 'r') as f:
        class_mapping = eval(f.read())
    return class_mapping

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor


# ImageNet1K 테스트 데이터 불러오기 (클래스마다 이미지 1개)
class_mapping = load_class_mapping(class_mapping_path)
assert(len(class_mapping) == 1000)

image_files = [os.path.join(image_repo_path, f) for f in os.listdir(image_repo_path) if f.endswith('.JPEG')]
assert(len(image_files) == 1000)

# 1000개 다 돌리니까 너무 오래걸려서 앞부분만 테스트
image_files = image_files[:20]


def calculate_latency(model: torch.nn.Module, image_files: list[str]):
    latencies = []
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(total=len(image_files), desc="Calculating latency", file=sys.stdout, position=1, leave=False)

        for image_file in image_files:
            input_tensor = load_image(image_file)

            # 측정 시작
            start_time = time.time()
            outputs = model(input_tensor)
            # 측정 종료
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)

            progress_bar.update(1)

        progress_bar.close()

    average_latency = sum(latencies) / len(latencies) if latencies else 0
    return average_latency


def test_image_latency(worker_nodes: List[WorkerNode], orig: models.ResNet, split: SplitResnet):
    """
    분산 ResNet 모델로 이미지의 추론 지연 시간을 측정합니다.
    """
    print("\nStarting Image Latency Test...\n")
    distributed_resnet = DistributedResnet(split, worker_nodes, is_sequential=False)
    avg_latency = calculate_latency(distributed_resnet, image_files)
    print(f"\nAverage Latency for {len(image_files)} images: {avg_latency:.4f} seconds")

def measure_distributed_resnet_accuracy_and_latency(worker_nodes: list[WorkerNode], orig: models.ResNet, split: SplitResnet):
    progress = tqdm(total = 2, desc = 'measuring distributed model accuracy and latency', file = sys.stdout, position = 0)


    # Distributed ResNet의 정확도 및 latency 측정
    progress.set_postfix_str('running distributed model')
    distributed = DistributedResnet(split, worker_nodes, is_sequential = False)
    distributed_latency = test_image_latency(distributed, image_files, class_mapping)
    progress.update()
    progress.close()

    print(f"- Distributed ResNet Latency: {distributed_latency:.6f} seconds")

#원래 있던 함수 
def calculate_accuracy_and_latency(model: torch.nn.Module, image_files: list[str], class_mapping: dict):
    correct = 0
    total = 0
    latencies = []
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(total=len(image_files), desc="Calculating accuracy and latency", file = sys.stdout, position = 1, leave = False)

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


    accuracy = 100 * correct / total
    average_latency = sum(latencies) / len(latencies)

    return accuracy, average_latency

def measure_distributed_resnet_accuracy_and_latency(worker_nodes: list[WorkerNode], orig: models.ResNet, split: SplitResnet):
    progress = tqdm(total = 1, desc = 'measuring distributed model accuracy and latency', file = sys.stdout, position = 0)


    # Distributed ResNet의 정확도 및 latency 측정
    progress.set_postfix_str('running distributed model')
    distributed = DistributedResnet(split, worker_nodes, is_sequential = False)
    distributed_accuracy, distributed_latency = calculate_accuracy_and_latency(distributed, image_files, class_mapping)
    progress.update()
    progress.close()

    print(f"- Distributed ResNet Accuracy: {distributed_accuracy:.2f}%")
    print(f"- Distributed ResNet Latency: {distributed_latency:.6f} seconds")

#밑에 3개 한세트 랜덤으로 이미지 20개 생성후 마스터->워커->마스터 걸리는 시간 측정
def generate_random_images(num_images: int, input_shape: List[int]) -> List[torch.Tensor]:
    """
    주어진 개수와 크기로 랜덤 이미지를 생성합니다.
    :param num_images: 생성할 랜덤 이미지 개수
    :param input_shape: 각 이미지의 입력 크기 (e.g., [1, 3, 256, 256])
    :return: 랜덤 이미지 텐서 리스트
    """
    return [torch.rand(*input_shape) for _ in range(num_images)]


def calculate_latency_random_images(model: torch.nn.Module, random_images: List[torch.Tensor]):
    """
    랜덤 이미지를 사용하여 latency를 측정합니다.
    :param model: 분산 ResNet 모델
    :param random_images: 랜덤 이미지 리스트
    :return: 평균 latency
    """
    latencies = []
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(total=len(random_images), desc="Calculating latency", file=sys.stdout, position=1, leave=False)

        for input_tensor in random_images:
            # 시작 시간 기록
            start_time = time.time()
            
            # 모델 추론 수행
            outputs = model(input_tensor)
            
            # 종료 시간 기록
            end_time = time.time()
            latencies.append(end_time - start_time)

            progress_bar.update(1)

        progress_bar.close()

    # 평균 latency 계산
    average_latency = sum(latencies) / len(latencies) if latencies else 0
    return average_latency


def test_random_image_latency(worker_nodes: List['WorkerNode'], orig: models.ResNet, split: 'SplitResnet'):
    """
    분산 ResNet 모델로 랜덤 이미지의 latency를 측정합니다.
    """
    print("\nStarting Random Image Latency Test...\n")
    distributed_resnet = DistributedResnet(split, worker_nodes, is_sequential=False)

    # 랜덤 이미지 생성
    num_images = 20
    input_shape = [1, 3, 256, 256]
    random_images = generate_random_images(num_images, input_shape)

    # Latency 측정
    avg_latency = calculate_latency_random_images(distributed_resnet, random_images)
    print(f"\nAverage Latency for {num_images} random images: {avg_latency:.4f} seconds")

if __name__ == '__main__':
    #run_test_on_distributed_env(num_workers=2, test=test_master_worker_communication)
    #run_test_on_distributed_env(num_workers=2, test=measure_distributed_resnet_accuracy_and_latency, port_offset=1001, use_pretrained_resnet=True)
    run_test_on_distributed_env(num_workers=2, test=test_random_image_latency, port_offset=1001, use_pretrained_resnet=True)