# 참고자료:
# - https://medium.com/@thatchawin.ler/cifar10-with-resnet-in-pytorch-a86fe18049df
# - https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
# 실행 명령:
# 일단 빌드) docker build -t test .
# 새롭게 학습) docker run --rm --gpus=all -it -v savevolume:/save test python cifar_resnet.py --mode train --epoch 20
# 이어서 학습) docker run --rm --gpus=all -it -v savevolume:/save test python cifar_resnet.py --mode train --epoch 20 --resume True
# 추론) docker run --rm --gpus=all -it -v savevolume:/save test python cifar_resnet.py --mode test
import torch
from torch.utils.data import DataLoader

import torchvision.models as models
from torchvision import datasets
from torchvision import transforms

import os
import argparse
from tqdm import tqdm

train_data_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_data_transform = transforms.Compose([  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


"""
cifar10 데이터셋의 학습 데이터와 테스트 데이터의 Dataloader를 튜플로 반환

ex)
train_dataloader, test_dataloader = prepare_cifar10_dataloader(batch_size = 100, num_workers = os.cpu_count())
"""
def prepare_cifar10_dataloader(batch_size: int, num_workers: int):
    train_data = datasets.CIFAR10(
        root="data", # where to download data to?
        train=True, # get training data
        download=True, # download data if it doesn't exist on disk
        transform=train_data_transform, # images come as PIL format, we want to turn into Torch tensors
        target_transform=None # you can transform labels as well
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False, # get test data
        download=True,
        transform=test_data_transform
    )
    
    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False)

    return train_dataloader, test_dataloader


"""
test_dataloader의 모든 데이터에 대해 모델을 돌려보고
전체 테스트 데이터 수와 정답을 맞춘 횟수를 튜플로 반환

ex)
num_corrects, num_tests = test_model(model, test_dataloader)
print(f'accuracy: {num_corrects / num_tests * 100}%')
"""
def test_model(model: torch.nn.Module, test_dataloader: DataLoader, device: str = 'cpu'):
    assert(device in ('cpu', 'cuda'))

    # Note: len(test_dataloader)는 데이터가 아니라 batch가 몇개나 있는지 반환함
    num_tests = len(test_dataloader.dataset)

    model.eval()
    with torch.no_grad():
        num_corrects = 0
        for batch, (x, y) in enumerate(tqdm(test_dataloader)):
            # optional: CPU <-> GPU 변환
            x, y = x.to(device), y.to(device)

            pred = model(x)
            pred_labels = torch.argmax(pred, dim=1)
            
            num_corrects += torch.sum(pred_labels == y).item()
    return num_corrects / num_tests


"""
train_dataloader의 모든 데이터에 대해 한 epoch 학습하고 net loss를 반환
"""
def train_model(model: torch.nn.Module, train_dataloader: DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: str = 'cpu'):
    assert(device in ('cpu', 'cuda'))


    net_loss = 0
    num_tests = len(train_dataloader.dataset)
    num_corrects = 0
    
    model.train()
    for batch, (x, y) in enumerate(tqdm(train_dataloader)):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        pred_labels = torch.argmax(pred, dim=1)
        num_corrects += torch.sum(pred_labels == y).item()
        net_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return net_loss, num_corrects / num_tests


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--savepath", type=str, default='/save') # 학습될 weight를 저장할 위치 (컨테이너 실행할 때 주어진 볼륨 옵션 -v [볼륨 이름]:[경로]의 경로와 일치해야 함)
    args.add_argument("--mode", type=str, default='train')
    args.add_argument("--epoch", type=int, default=1)
    args.add_argument("--resume", type=bool, default=False)
    args = args.parse_args()
    
    weight_path = args.savepath + '/trained_weight.pth'

    # GPU 사용 가능한지 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'currently using {device} to run model')

    # cifar10 데이터 불러오기
    train_dataloader, test_dataloader = prepare_cifar10_dataloader(batch_size = 128, num_workers = os.cpu_count())

    if args.mode == 'test':
        # 학습된 모델을 불러와 정확도 확인 (저장 전과 동일한지 확인 필요)
        model = models.resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        model.load_state_dict(torch.load(weight_path, weights_only = True))
        model.to(device)

        accuracy = test_model(model, test_dataloader, device)
        print(f'accuracy of saved model: {accuracy * 100}%')

    elif args.mode == 'train':
        # 원본은 클래스가 1000개인 imagenet에 대해 학습되었으므로
        # cifar10에 사용하려면 마지막 classifier를 출력 차원이 10인 새로운 레이어로 교체해야 함
        model = models.resnet50(pretrained = True)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)

        # 마지막 상태로부터 학습 재개
        if args.resume:
            model.load_state_dict(torch.load(weight_path, weights_only = True))

        model.to(device)

        # 학습하기 전 정확도 확인
        best_accuracy = test_model(model, test_dataloader, device)
        print(f'accuracy before training: {best_accuracy * 100}%')

        # cifar10에 대해 새롭게 학습
        learning_rate = 0.01
        weight_decay = 15e-5
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(args.epoch):
            net_loss, train_accuracy = train_model(model, train_dataloader, loss_fn, optimizer, device)
            test_accuracy = test_model(model, test_dataloader, device)

            print(f'epoch#{epoch} net loss: {net_loss}, train accuracy: {train_accuracy * 100}%, test accuracy: {test_accuracy * 100}%')

            # 테스트 데이터 정확도가 제일 높게 나오는 모델 저장
            if test_accuracy > best_accuracy:
                print(f'new best test accuracy! saving the model\'s weight to {weight_path}')
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), weight_path)
            
