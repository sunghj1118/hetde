import torchvision.models as models

# 도커 이미지 빌드할 때 실행되는 함수로,
# pretrained weight 다운로드만 미리 해놓는 역할을 맡음
if __name__ == '__main__':
    models.resnet50(pretrained = True)