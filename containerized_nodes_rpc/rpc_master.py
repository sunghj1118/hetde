import torch.distributed.rpc as rpc
import argparse
import os
import torch
import torchvision.models as models
from rpc_distributed_utils import DistributedResnet, SplitResnet, RPCWorkerNode
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import time


def measure_latency(model, num_images: int, input_shape: list[int]):
    """
    랜덤 이미지를 생성하고 latency를 측정합니다.
    :param model: DistributedResnet 모델
    :param num_images: 생성할 이미지 개수
    :param input_shape: 입력 이미지 텐서의 형태 (e.g., [1, 3, 256, 256])
    :return: 평균 latency
    """
    latencies = []
    model.eval()

    with torch.no_grad():
        progress_bar = tqdm(range(num_images), desc="Measuring Latency", leave=False)
        for _ in progress_bar:
            # 랜덤 입력 이미지 생성
            dummy_input = torch.rand(*input_shape)

            # 시간 측정 시작
            start_time = time.time()

            # 모델 추론
            result = model(dummy_input)

            # 시간 측정 종료
            end_time = time.time()

            # latency 계산 및 저장
            latency = end_time - start_time
            latencies.append(latency)

        progress_bar.close()

    # 평균 latency 계산
    average_latency = sum(latencies) / len(latencies) if latencies else 0
    return average_latency


def main(rank: int, num_workers: int, num_images: int):
    print(f"Master Rank {rank} initializing RPC")
    rpc.init_rpc(
        "master",
        rank=rank,
        world_size=num_workers + 1,  # master + workers
    )
    print("Master RPC initialized")

    # 모델 초기화 및 분산 처리 구성
    orig = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    split = SplitResnet(orig, num_workers)

    # WorkerNode 초기화
    worker_nodes = [RPCWorkerNode(i + 1) for i in range(num_workers)]
    distributed_model = DistributedResnet(split, worker_nodes, is_sequential=True)

    # 랜덤 이미지로 latency 측정
    input_shape = [1, 3, 256, 256]
    print(f"Generating {num_images} random input images for latency measurement...")
    average_latency = measure_latency(distributed_model, num_images, input_shape)

    print(f"Average Latency for {num_images} images: {average_latency:.4f} seconds")
    print("Shutting down RPC from Master")
    rpc.shutdown()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--num_workers", type=int, required=True, help="Number of worker nodes")
    args.add_argument("--num_images", type=int, default=20, help="Number of random images to generate")
    args.add_argument("--addr", type=str, default="localhost", help="Master node address")
    parsed_args = args.parse_args()

    os.environ["MASTER_ADDR"] = parsed_args.addr
    os.environ["MASTER_PORT"] = "29500"

    main(rank=0, num_workers=parsed_args.num_workers, num_images=parsed_args.num_images)
