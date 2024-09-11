## 참고 자료
- [한국어 pytorch pruning 튜토리얼](https://tutorials.pytorch.kr/intermediate/pruning_tutorial.html)
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

## [Communication Aware DNN Pruning 논문](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10229043)
- 모델의 분할이 이미 이루어졌다고 가정하고 pruning (재학습 o)을 할 때 penalty term으로 device 사이의 communication cost를 추가함
    - 새로운 loss = 기존 loss + communication cost
- 그냥 학습시키면 통신이 불가능한 device 사이의 weight가 살아남는 등의 결과가 나올 수 있음  
=> convex optimization으로 취급해서 communication cost 관련 제약 조건을 만족하는 파라미터 theta를 탐색
    - 여기에 쓰인 최적화 알고리즘이 [ADMM](https://convex-optimization-for-all.github.io/contents/chapter21/2021/03/29/21_01_Last_time_Dual_method,_Augmented_Lagrangian_method,_ADMM,_ADMM_in_scaled_form/)
    - 공수에서 잠깐 나온 라그랑주 승수법이랑 비슷한 맥락인듯?