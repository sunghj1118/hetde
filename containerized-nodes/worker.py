import torch
import tcp
import time
import os
import socket
from typing import List
import torchvision.models as models

class PartialConv2d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d, out_channel_begin: int, out_channel_end: int):
        super(PartialConv2d, self).__init__()
        out_channels = out_channel_end - out_channel_begin
        self.conv = torch.nn.Conv2d(conv.in_channels, out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
        self.conv.weight.data = conv.weight[out_channel_begin:out_channel_end].clone()  # 데이터를 복사하여 텐서를 생성
        if conv.bias is not None:
            self.conv.bias.data = conv.bias[out_channel_begin:out_channel_end].clone()  # 데이터를 복사하여 텐서를 생성

    def forward(self, x: torch.Tensor):
        return self.conv(x)

class SplitResnet:
    """
    ResNet 모델을 여러 개의 PartialConv2d 레이어로 분할하여 관리하는 클래스.
    """
    def __init__(self, model: models.ResNet, num_splits: int):
        self.split_conv_dict = {}
        self.num_splits = num_splits

        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                self.split_conv_dict[name] = self.create_split_conv(layer)

    def create_split_conv(self, layer: torch.nn.Conv2d):
        """
        원본 Conv2d 레이어를 여러 개로 분할.
        총 출력 채널 수와 원하는 분할 수를 기반으로 각 부분의 채널 수를 계산하고,
        이를 토대로 PartialConv2d 인스턴스를 생성 및 반환.
        """
        total_channels = layer.out_channels
        out_channels_per_part = self.distribute_channels(total_channels, self.num_splits)
        partial_convs = []

        out_channel_begin = 0
        for num_channels in out_channels_per_part:
            out_channel_end = out_channel_begin + num_channels
            partial_convs.append(PartialConv2d(layer, out_channel_begin, out_channel_end))
            out_channel_begin = out_channel_end

        return partial_convs

    @staticmethod
    def distribute_channels(total_channels: int, num_splits: int) -> List[int]:
        """
        주어진 총 채널 수를 지정된 분할 수에 따라 최대한 균등하게 분배.
        """
        base_channels = total_channels // num_splits
        remainder = total_channels % num_splits

        out_channels_per_part = [base_channels] * num_splits
        for i in range(remainder):
            out_channels_per_part[i] += 1

        return out_channels_per_part

class WorkerNodeServer:
    """
    Worker 노드에서 마스터 노드로부터 연산 요청을 수신하고, 연산 후 결과를 송신하는 서버 역할을 하는 클래스.
    """
    def __init__(self, host: str, port: int, model: models.ResNet, num_splits: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)  # 최대 대기 중인 연결 수를 5로 설정
        print(f"WorkerNodeServer listening on {self.host}:{self.port}")

        # SplitResnet 초기화
        self.split_resnet = SplitResnet(model, num_splits)

    def start(self):
        try:
            while True:
                client_socket, client_address = self.sock.accept()
                print(f"Connection established with {client_address}")

                try:
                    while True:
                        original_layer_name = tcp.recv_utf8(client_socket)
                        if original_layer_name == 'terminate':
                            break

                        part_index = tcp.recv_u32(client_socket)
                        x = tcp.recv_tensor(client_socket).detach().clone().contiguous()  # 데이터를 완전히 새로운 쓰기 가능한 텐서로 생성

                        # 연산 수행
                        with torch.no_grad():
                            start = time.time()
                            # 분할된 모델의 해당 부분 연산 수행
                            result = self.perform_inference(original_layer_name, part_index, x)
                            end = time.time()
                            computation_time = end - start

                            # 결과 전송
                            tcp.send_f64(client_socket, computation_time)
                            tcp.send_tensor(client_socket, result)
                except Exception as e:
                    print(f"[Exception during communication with {client_address}]: {str(e)}")
                finally:
                    client_socket.close()

        except Exception as e:
            print(f"[Exception in WorkerNodeServer]: {str(e)}")
        finally:
            self.sock.close()

    def perform_inference(self, layer_name: str, part_index: int, x: torch.Tensor):
        # 분할된 레이어의 일부를 사용하여 연산을 수행하는 함수
        partial_conv = self.split_resnet.split_conv_dict[layer_name][part_index]
        return partial_conv(x)


# 서버 실행 예시 (호스트와 포트는 환경 변수로 받을 수 있음)
if __name__ == "__main__":
    # 환경 변수에서 포트 번호를 받아 설정 (기본 포트는 10000번)
    port = int(os.getenv("PORT", 10000))
    print(f"Port value read from environment: {port}")
    # Pre-trained ResNet 모델 로드 및 분할 설정
    model = models.resnet50(weights=None)  # 사전 학습된 가중치를 사용하지 않음 (랜덤 초기화).
    num_splits = 2  # 원하는 분할 수 설정
    worker_server = WorkerNodeServer('0.0.0.0', port, model, num_splits)
    worker_server.start()
