FROM pytorch/pytorch

RUN pip install Flask
RUN apt-get update && apt-get -y install net-tools git

# ImageNet1K 테스트 데이터
RUN git clone https://github.com/EliSchwartz/imagenet-sample-images.git
RUN git clone https://gist.github.com/942d3a0ac09ec9e5eb3a.git

COPY test.py test.py
COPY runtime.py runtime.py
COPY tcp.py tcp.py
COPY pruning_test.py pruning_test.py
COPY rpc_test.py rpc_test.py

CMD ["python", "test.py"]

# rpc_test.py에서 MASTER_PORT로 사용하는 포트 개방
EXPOSE 29500

# test.py 즉시 실행 (모든 유닛 테스트 돌리기):
#   docker build -t test .; docker run --rm test
#
# cmd로 들어가기:
#   docker build -t test .; docker run -it --rm test bash
#
# 띄워놓고 vscode remote로 접속하는 용도:
#   docker build -t test .; docker run -dt --rm test bash

# rpc_test.py 테스트하는 방법:
# docker run -dt -p 29500:29500 test python rpc_test.py --rank 0 --addr host.docker.internal
# docker run -dt -p 29501:29500 test python rpc_test.py --rank 1 --addr host.docker.internal
# docker run -dt -p 29502:29500 test python rpc_test.py --rank 2 --addr host.docker.internal
# docker run -dt -p 29503:29500 test python rpc_test.py --rank 3 --addr host.docker.internal
#
# 주의사항: 이건 로그로 결과 확인해야해서 --rm 옵션 안 들어있음!!! 끝나면 종료된 컨테이너 직접 삭제할 것