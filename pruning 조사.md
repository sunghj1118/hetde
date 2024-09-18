## 참고 자료
- [한국어 pytorch pruning 튜토리얼](https://tutorials.pytorch.kr/intermediate/pruning_tutorial.html)
- [prune.remove를 왜 하는가?](https://computing-jhson.tistory.com/42)
- [pytorch structured pruning 문서](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.ln_structured.html)
- [convex optimization이란?](https://audrb1999.tistory.com/70)

## Pruning
- pruning에는 두 가지 선택지가 있음

||재학습 o|재학습 x|
|-|-|-|
|structured|A|B|
|unstructured|C|D|

- unstructured: 무작위로 weight를 0으로 만드는거라 연산량 변화 x
- structured: channel 단위로 제거하면서 실제로 일부 연산이 사라지는거라 연산량 약간 감소
- unstructured를 예시로 들자면 일부 weight를 제거했다고 해도 사라진 부분만 빼고 전송한 뒤 이를 원래 shape로 복구할 방법이 필요함
    - ex) CSR format
- structured 방식으로 제거한다고 해도 어느 channel이 비활성화된 것인지 구분할 방법이 필요함

- 다행히 pytorch에 pruning 기능이 있어서 weight 제거 로직을 구현할 필요는 없어보임


## pytorch pruning
```python
# pytorch 튜토리얼에서 복붙한 코드
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1개 채널 수의 이미지를 입력값으로 이용하여 6개 채널 수의 출력값을 계산하는 방식
        # Convolution 연산을 진행하는 커널(필터)의 크기는 5x5 을 이용
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Convolution 연산 결과 5x5 크기의 16 채널 수의 이미지
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device=device)

# 'conv1.bias', 'fc2.weight' 등 nn.Module의 모든 파라미터를
# (멤버 이름).(weight|bias) 형태로 관리한다는 것을 확인할 수 있음.
print(list(model.named_parameters()))

# nn.Module의 임의의 named parameter를 일정 비율로 prune해주는 함수.
#
# prune할 파라미터의 이름이 XXX인 경우:
# 1. XXX_mask라는 named buffer를 생성
# 2. XXX_orig라는 파라미터를 새로 만들어서 원본을 보존
# 3. XXX는 prune된 파라미터로 교체
#
# model.fc1.weight_orig를 출력해보면 원본 weight가 남아있는 반면
# model.fc1.weight를 출력해보면 weight_mask가 적용된 값이 나옴!
prune.random_unstructured(model.fc1, name='weight', amount=0.3)


# prune.random_unstructured(model, name='fc1.weight', amount=0.3)처럼
# nested module에 대한 pruning은 아쉽게도 불가능함
#
# 그러므로 전체 모델에 대해 pruning을 하려면
# 직접 아래와 같이 모든 Module에 대해 순회해야 함
for name, module in model.named_modules():
    # 모든 2D-conv 층의 20% 연결에 대해 가지치기 기법을 적용
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.2)
    # 모든 선형 층의 40% 연결에 대해 가지치기 기법을 적용
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.4)

# structured pruning은 기준이 될 텐서 차원을 추가로 지정해줘야 함.
#
# 예를 들어, 위 모델의 conv1.weight.shape는 [6, 1, 5, 5]임.
# 여기서 conv 필터는 0번 차원이므로 아래 코드를 실행하면 6개의 필터 중에서 몇 개가 통째로 0으로 변함.
#
# 우리는 conv 레이어의 필터 단위로 분산 처리를 할 예정이므로 dim=0을 사용하면 될듯?
prune.random_structured(model.conv1, 'weight', 0.3, dim=0)


# prune한 파라미터는 forward과정에서 매번 mask와 orig값을 참조해 실제 weight를 새로 계산하게 됨.
# * 출처: https://computing-jhson.tistory.com/42
# 
# 이 과정을 없애고 prune 결과를 영구적으로 반영하려면 prune.remove를 해주면 됨.
prune.remove(model.conv1, 'weight')
```


## [Communication Aware DNN Pruning 논문](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10229043)
- 모델의 분할이 이미 이루어졌다고 가정하고 pruning (재학습 o)을 할 때 penalty term으로 device 사이의 communication cost를 추가함
    - 새로운 loss = 기존 loss + communication cost
- 그냥 학습시키면 통신이 불가능한 device 사이의 weight가 살아남는 등의 결과가 나올 수 있음  
=> convex optimization으로 취급해서 communication cost 관련 제약 조건을 만족하는 파라미터 theta를 탐색
    - 여기에 쓰인 최적화 알고리즘이 [ADMM](https://convex-optimization-for-all.github.io/contents/chapter21/2021/03/29/21_01_Last_time_Dual_method,_Augmented_Lagrangian_method,_ADMM,_ADMM_in_scaled_form/)
    - 공수에서 잠깐 나온 라그랑주 승수법이랑 비슷한 맥락인듯?
