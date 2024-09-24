FROM pytorch/pytorch

COPY test.py test.py

CMD ["python", "test.py"]

# test.py 즉시 실행:
#   docker build -t test .; docker run -it --rm test
#
# cmd로 들어가기:
#   docker build -t test .; docker run -it --rm test bash
