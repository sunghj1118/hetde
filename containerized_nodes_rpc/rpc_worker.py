import torch.distributed.rpc as rpc
import argparse
import os
from rpc_distributed_utils import rpc_worker_node_inference


def main(rank: int, world_size: int):
    print(f"Worker Rank {rank} initializing RPC")
    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
    )
    print(f"Worker {rank} RPC initialized")

    # 요청 대기 상태로 유지
    rpc.shutdown()
    print(f"Worker {rank} RPC shut down")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--rank", type=int, required=True, help="Rank of the worker node")
    args.add_argument("--addr", type=str, default="localhost", help="Master node address")
    args.add_argument("--world_size", type=int, required=True, help="Total number of nodes (master + workers)")
    parsed_args = args.parse_args()

    os.environ["MASTER_ADDR"] = parsed_args.addr
    os.environ["MASTER_PORT"] = "29500"

    main(rank=parsed_args.rank, world_size=parsed_args.world_size)
