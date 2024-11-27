import torchvision.models as models
from torchvision import datasets

# 도커 이미지 빌드할 때 실행됨
if __name__ == '__main__':
    # pretrained weight 다운로드만 미리 해놓는 역할
    models.resnet50(pretrained = True)
    
    # 이건 cifar10 데이터셋 다운로드만 미리 해놓는 역할
    train_data = datasets.CIFAR10(
        root="data", # where to download data to?
        train=True, # get training data
        download=True, # download data if it doesn't exist on disk
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False, # get test data
        download=True
    )
