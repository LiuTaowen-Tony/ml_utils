from typing import Literal
import wandb
import random
import os
import torch
import torch.distributed as dist

class AllGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype

        output = list(torch.empty_like(tensor) for _ in range(dist.get_world_size()))
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype), None

class AllReduceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                tensor: torch.Tensor, 
                reduce_op: Literal["sum", "avg"] = "sum",
                reduce_dtype: torch.dtype = torch.float32,
                ):
        ctx.reduce_dtype = reduce_dtype
        ctx.reduce_op = reduce_op
        ctx.input_dtype = tensor.dtype

        if reduce_op == "sum":
            op = dist.ReduceOp.SUM
        elif reduce_op == "avg":
            op = dist.ReduceOp.AVG
        else:
            raise ValueError(f"Invalid reduce_op: {reduce_op}")
        
        # Create a copy and convert to reduce_dtype to avoid in-place modification
        output = tensor.to(ctx.reduce_dtype).clone()
        dist.all_reduce(output, op)
        
        # Convert back to original dtype
        return output.to(ctx.input_dtype)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # For all-reduce backward pass:
        # - For SUM: gradient flows back unchanged (each process contributed to the sum)
        # - For AVG: gradient should be divided by world_size (since forward averaged)

        # don't need to convert dtype here, because no reduction is done in the backward pass
        if ctx.reduce_op == "sum":
            # For sum reduction, gradient flows back unchanged
            grad_input = grad_output
        elif ctx.reduce_op == "avg":
            # For average reduction, we need to divide gradient by world_size
            # because the forward pass averaged the values
            grad_input = grad_output / dist.get_world_size()
        else:
            raise ValueError(f"Invalid reduce_op: {ctx.reduce_op}")
            
        return grad_input, None, None

def all_gather(tensor):
    if tensor is None:
        return None
    if dist.get_world_size() == 1:
        return tensor
    return AllGatherFunction.apply(tensor)

def all_reduce(tensor, reduce_op: Literal["sum", "avg"] = "sum", reduce_dtype: torch.dtype = torch.float32):
    assert isinstance(tensor, torch.Tensor), "tensor must be a torch.Tensor"
    if dist.get_world_size() == 1:
        return tensor
    return AllReduceFunction.apply(tensor, reduce_op, reduce_dtype)

def scalar_all_reduce(scalar, device: torch.device, reduce_op: Literal["sum", "avg"] = "sum"):
    assert isinstance(scalar, (int, float)), "scalar must be an int or float"
    tensor = torch.tensor(scalar, device=device)
    return all_reduce(tensor, reduce_op, torch.float32).item()

def scalar_all_gather(scalar, device: torch.device):
    assert isinstance(scalar, (int, float)), "scalar must be an int or float"
    tensor = torch.tensor(scalar, device=device)
    return all_gather(tensor).item()

def is_rank_0():
    return not dist.is_initialized() or dist.get_rank() == 0

def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

def seed_init_dist(backend: str = "nccl", seed: int = 1):
    dist.init_process_group(backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(seed + local_rank)
    torch.manual_seed(seed + local_rank)
    return device

def wandb_init(project: str, entity: str, config: dict) -> "wandb.Run":
    if is_rank_0():
        run = wandb.init(
            project=project, 
            entity=entity, 
            config=config
        )
    else:
        class _Dummy:  # makes .log() a no‑op on workers
            def log(self, *_, **__): ...
        run = _Dummy()
    run : "wandb.Run"
    return run

def apply_dist_sampler(dataloader: torch.utils.data.DataLoader, shuffle: bool = True) -> torch.utils.data.DataLoader:
    if dist.get_world_size() == 1:
        return dataloader
    sampler = torch.utils.data.distributed.DistributedSampler(dataloader.dataset, shuffle=shuffle)
    return torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        num_workers=dataloader.num_workers,
        pin_memory=True,
    )

# class DistContext:
#     def __init__(self, backend: str = "nccl", seed: int = 1, project: str = "default", entity: str = "default", config: dict = {}):
#         self.backend = backend
#         self.seed = seed
#         self.device = None
#         self.run = None
#         self.project = project
#         self.entity = entity
#         self.config = config

#     def __enter__(self):
#         self.device = seed_init_dist(self.backend, self.seed)
#         self.run = wandb_init(self.project, self.entity, self.config)
#         return self

#     def __exit__(self, exc_type, exc_value, traceback):
#         if is_rank_0():
#             self.run.finish()
#         dist.destroy_process_group()

# example:
# with DistContext(backend="nccl", seed=1, project="test", entity="test", config={"test": "test"}):
#     model = Model()
#     model.to(self.device)
#     model = FSDP(model, device_id=self.device)
#     train_loader = apply_dist_sampler(train_loader)
#     val_loader = apply_dist_sampler(val_loader)
#     trainer = Trainer(max_epochs=10)
#     trainer.fit(model, train_loader, val_loader)

