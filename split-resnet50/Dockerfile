FROM pytorch/pytorch

RUN pip install Flask

COPY test.py test.py
COPY tcp.py tcp.py

CMD ["python", "test.py"]

# test.py 즉시 실행 (모든 유닛 테스트 돌리기):
#   docker build -t test .; docker run --rm test
#
# cmd로 들어가기:
#   docker build -t test .; docker run -it --rm test bash
#
# 띄워놓고 vscode remote로 접속하는 용도:
#   docker build -t test .; docker run -dt --rm test bash