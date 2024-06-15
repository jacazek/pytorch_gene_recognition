import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    print(f"torch enabled {torch.cuda.is_available()}")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # dummy input and target
    inputs = torch.randn(20, 10).to(rank)
    targets = torch.randn(20, 10).to(rank)

    for epoch in range(5):
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

def main():
    os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.multiprocessing.spawn(demo_basic,
                                args=(world_size,),
                                nprocs=world_size)

if __name__ == "__main__":
    main()