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

# test.py 즉시 실행 (모든 유닛 테스트 돌리기):
#   docker build -t test .; docker run --rm test
#
# cmd로 들어가기:
#   docker build -t test .; docker run -it --rm test bash
#
# 띄워놓고 vscode remote로 접속하는 용도:
#   docker build -t test .; docker run -dt --rm test bash