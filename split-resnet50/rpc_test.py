import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import os
import argparse
import time


SIZE = 64 * 1

# Define the CNN models
class SimpleCNNDevice1(nn.Module):
    def __init__(self):
        super(SimpleCNNDevice1, self).__init__()
        self.conv = nn.Conv2d(2, 5, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)

class SimpleCNNDevice2(nn.Module):
    def __init__(self):
        super(SimpleCNNDevice2, self).__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)

class SimpleCNNDevice3(nn.Module):
    def __init__(self):
        super(SimpleCNNDevice3, self).__init__()
        self.fc = nn.Linear(8 * SIZE * SIZE, 10)  # Assume 10 classes for simplicity
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model1 = SimpleCNNDevice1().to("cpu")
model2 = SimpleCNNDevice2().to("cpu")
model3 = SimpleCNNDevice3().to("cpu")

# Define remote function for Device 1
def device1_processing(rref_input):
    a = time.time()
    input_tensor = rref_input.to_here()
    output = model1(input_tensor[:, :2])
    b = time.time()
    print('end device1_processing', b-a, 'seconds')
    return output

# Define remote function for Device 2
def device2_processing(rref_input):
    a = time.time()
    input_tensor = rref_input.to_here()
    output = model2(input_tensor[:, 1:3])
    b = time.time()
    print('end device2_processing', b-a, 'seconds')
    return output

# Define remote function for Device 3
def device3_processing(output1, output2):
    a = time.time()
    combined_output = torch.cat((output1, output2), dim=1)
    final_output = model3(combined_output)
    b = time.time()
    print('end device3_processing', b-a, 'seconds')
    return final_output

def main(rank):
    # Initialize RPC for each device role
    print(f"Rank {rank} initialized RPC")
    # rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
    )
    print('All workers initialized RPC', time.time())

    if rank == 0:
        # Device 1: Process input and send to Device 3
        input_tensor = torch.randn(4, 3, SIZE, SIZE, device='cpu')
        rref_input = RRef(input_tensor)
        # print('sending rref_inputs to worker1 and worker 2')
        a = time.time()
        rref_output1 = rpc.remote("worker1", device1_processing, args=(rref_input,)) # Note: This is a non-blocking call
        b = time.time()
        print('sent rref_inputs to worker1', b-a, 'seconds')
        rref_output2 = rpc.remote("worker2", device2_processing, args=(rref_input,)) # Note: This is a non-blocking call
        c = time.time()
        print('sent rref_inputs to worker 2', c-b, 'seconds')
        d = time.time()
        output1 = rref_output1.to_here()
        output2 = rref_output2.to_here()
        e = time.time()
        print('Got results from worker1 and worker2', e-d, 'seconds')
        print('sending rref_outputs to worker3')
        f = time.time()
        final_result = rpc.rpc_sync("worker3", device3_processing, args=(output1, output2))
        g = time.time()
        print('received final result from worker3', g-f, 'seconds')
        print('Time taken for the whole process:', g-a, 'seconds')

    rpc.shutdown()

if __name__ == "__main__":
    # Usage:
    # python rpc_test.py --rank 0
    # python rpc_test.py --rank 1
    # python rpc_test.py --rank 2
    # python rpc_test.py --rank 3

    args = argparse.ArgumentParser()
    args.add_argument("--rank", type=int, default=0)
    args.add_argument("--addr", type=str, default='127.0.0.1')
    args = args.parse_args()
    rank = args.rank
    master_addr = args.addr

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = '29500'

    world_size = 4
    main(rank=rank)
    # rpc.init_rpc("main", rank=0, world_size=world_size)
    # torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
