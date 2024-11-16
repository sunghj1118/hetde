import torch
import tcp
import time
import os
import socket
import sys
from typing import List
import functools

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


class WorkerNodeServer:
    """
    Worker 노드에서 마스터 노드로부터 연산 요청을 수신하고, 연산 후 결과를 송신하는 서버 역할을 하는 클래스.
    """
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)  # 최대 대기 중인 연결 수를 5로 설정
        print(f"WorkerNodeServer listening on {self.host}:{self.port}")

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
                        conv_params = tcp.recv_tensor(client_socket).detach().clone().contiguous()  # 데이터를 완전히 새로운 쓰기 가능한 텐서로 생성
                        x = tcp.recv_tensor(client_socket).detach().clone().contiguous()  # 데이터를 완전히 새로운 쓰기 가능한 텐서로 생성
                        
                        # 연산 수행
                        with torch.no_grad():
                            start = time.time()
                            # 분할된 모델의 해당 부분 연산 수행
                            result = self.perform_inference(conv_params, x)
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

    def perform_inference(self, conv_params: torch.Tensor, x: torch.Tensor):
        # 분할된 레이어의 일부를 사용하여 연산을 수행하는 함수
        conv = PartialConv2d(conv_params, 0, conv_params.shape[0])
        return conv(x)


# 서버 실행 예시 (호스트와 포트는 환경 변수로 받을 수 있음)
if __name__ == "__main__":
    # 환경 변수에서 포트 번호를 받아 설정 (기본 포트는 10000번)
    port = int(os.getenv("PORT", 10000))
    print(f"Port value read from environment: {port}")
    worker_server = WorkerNodeServer('0.0.0.0', port)
    worker_server.start()
