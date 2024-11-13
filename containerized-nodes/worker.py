import torch
import tcp
import time
from typing import Callable

class WorkerNode:
    """
    실제로 연산을 진행할 장치와의 통신 로직을 감싸는 클래스.
    마스터 서버에서 이 객체를 통해 입력 데이터를 전달하고 추론 결과를 수신한다.
    """
    def __init__(self, host: str, port: int):
        self.sock = tcp.connect_server(host, port)

    def request_inference(self, x: torch.Tensor, original_layer_name: str, part_index: int):
        tcp.send_utf8(self.sock, original_layer_name)
        tcp.send_u32(self.sock, part_index)
        tcp.send_tensor(self.sock, x)

    def receive_inference_result(self):
        return tcp.recv_f64(self.sock), tcp.recv_tensor(self.sock)

class PartialConv2d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d, out_channel_begin: int, out_channel_end: int):
        super(PartialConv2d, self).__init__()
        out_channels = out_channel_end - out_channel_begin
        self.conv = torch.nn.Conv2d(conv.in_channels, out_channels, conv.kernel_size, conv.stride, conv.padding, bias = conv.bias is not None)
        self.conv.weight.data = conv.weight[out_channel_begin:out_channel_end]
        if conv.bias is not None:
            self.conv.bias.data = conv.bias[out_channel_begin:out_channel_end]

    def forward(self, x: torch.Tensor):
        return self.conv(x)

