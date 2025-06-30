from typing import Literal
import wandb
import random
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

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
        pin_memory=dataloader.pin_memory,
        collate_fn=dataloader.collate_fn,  # Preserve collate function
        drop_last=dataloader.drop_last,    # Preserve drop_last setting
        timeout=dataloader.timeout,        # Preserve timeout
        worker_init_fn=dataloader.worker_init_fn,  # Preserve worker init
    )

def save_fsdp_model(model: FSDP, save_path: str, offload_to_cpu: bool = True) -> None:
    """
    Save FSDP model state dict properly to avoid hanging.
    
    Args:
        model: FSDP wrapped model
        save_path: Path to save the model
        offload_to_cpu: Whether to offload state dict to CPU during save
    """
    rank0_print(f"Saving FSDP model to {save_path}...")
    
    with FSDP.state_dict_type(
        model, 
        StateDictType.FULL_STATE_DICT, 
        FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=True)
    ):
        state_dict = model.state_dict()
        if is_rank_0():
            torch.save(state_dict, save_path)
            rank0_print(f"Model saved successfully to {save_path}")

def cleanup_distributed_training(run=None, finish_wandb: bool = True) -> None:
    """
    Clean up distributed training resources.
    
    Args:
        run: WandB run object to finish (optional)
        finish_wandb: Whether to finish WandB run
    """
    rank0_print("Cleaning up distributed training...")
    
    if finish_wandb and run is not None:
        try:
            if is_rank_0():
                rank0_print("Finishing WandB...")
                if hasattr(run, 'finish'):
                    run.finish()
                rank0_print("WandB finished successfully")
        except Exception as e:
            rank0_print(f"WandB cleanup error: {e}")
    
    rank0_print("Distributed training cleanup completed")

def log_training_metrics(run, global_step: int, loss_accum: float, samples_accum: int, 
                        device: torch.device, optimizer, log_every_n_steps: int = 1) -> None:
    """
    Log training metrics with proper distributed synchronization.
    
    Args:
        run: WandB run object
        global_step: Current training step
        loss_accum: Accumulated loss
        samples_accum: Accumulated samples
        device: Device for synchronization
        optimizer: Optimizer for learning rate
        log_every_n_steps: How often to log
    """
    if global_step % log_every_n_steps != 0:
        return
        
    loss_sum = scalar_all_reduce(loss_accum, device, "sum")
    samples_sum = scalar_all_reduce(samples_accum, device, "sum")
    current_lr = optimizer.param_groups[0]['lr']
    
    run.log({
        "train_loss": loss_sum / samples_sum,
        "learning_rate": current_lr
    }, step=global_step)
    rank0_print(f"Step {global_step:06d} | train_loss={loss_sum / samples_sum:.4f} | lr={current_lr:.2e}")

def create_fsdp_auto_wrap_policy(transformer_layer_cls):
    """
    Create FSDP auto-wrap policy for transformer models.
    
    Args:
        transformer_layer_cls: Set of transformer layer classes to wrap
        
    Returns:
        Configured auto-wrap policy
    """
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
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

