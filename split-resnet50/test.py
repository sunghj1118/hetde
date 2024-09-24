import torch
import torchvision.models as models

def conv2d_with_partial_output(conv: torch.nn.Conv2d, out_channels: int):
    return torch.nn.Conv2d(conv.in_channels, out_channels, conv.kernel_size, conv.stride, conv.padding, bias = conv.bias is not None)

class SplitConv2d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d, out_channels_per_part: list[int]):
        """
        @param conv 원본 레이어
        @out_channels_per_part 쪼개진 레이어가 각각 담당할 출력 채널의 수

        ex) SplitConv2d(conv, [2, 3, 10]) => 출력 채널의 수가 각각 2, 3, 10인 세 개의 conv 레이어로 쪼개기
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
        self.partial_convs = [conv2d_with_partial_output(conv, out_channels) for out_channels in out_channels_per_part]

        # 이 값은 루프가 끝날 때마다 직전 파트가 담당하는 마지막 채널의 인덱스 + 1으로 설정됨.
        # 그러므로, 이번 파트의 첫 번째 채널 인덱스로 해석할 수 있음.
        out_channel_begin = 0
        for i in range(len(self.partial_convs)):
            # 이번 파트가 담당할 채널의 경계 (마지막 채널 인덱스 + 1)
            out_channel_end = out_channel_begin + out_channels_per_part[i]

            # 원본 레이어의 가중치 복사해서 넣기 (bias는 없는 경우도 있음)
            self.partial_convs[i].weight.data = conv.weight[out_channel_begin:out_channel_end]
            if conv.bias is not None:
                self.partial_convs[i].bias.data = conv.bias[out_channel_begin:out_channel_end]

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
    
    def __str__(self):
        # 각 partial conv 레이어를 줄마다 하나씩 보여주는 형식
        return '[\n{}\n]'.format('\n'.join(['   {}'.format(conv) for conv in self.partial_convs]))

if __name__ == "__main__":
    orig = models.resnet50(pretrained=True).conv1 # <-- ResNet50의 첫 번째 conv로 테스트
    # orig = torch.nn.Conv2d(3, 64, 7) # <-- 랜덤 초기화된 conv로 테스트

    split1 = SplitConv2d(orig, [32, 32])
    split2 = SplitConv2d(orig, [2, 30, 32])
    split3 = SplitConv2d(orig, [2, 1, 3, 22, 4, 16, 16])

    print('orig:', orig)
    print('split1:', split1)
    print('split2:', split2)
    print('split3:', split3)

    for i in range(100):
        x = torch.rand(1, 3, 32, 32)
        y = orig(x)
        y1 = split1(x)
        y2 = split2(x)
        y3 = split3(x)
        assert(torch.equal(y, y1))
        assert(torch.equal(y, y2))
        assert(torch.equal(y, y3))
    print('split model equality assertion passed for 100 random inputs')
