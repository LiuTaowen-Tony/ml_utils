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
        return grad_input.to(grad_dtype)

def all_gather(tensor):
    if dist.get_world_size() == 1:
        return tensor
    return AllGatherFunction.apply(tensor)

def is_rank_0():
    return not dist.is_initialized() or dist.get_rank() == 0

def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def all_gather_concat(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)

def all_gather_concat_pl(self: "pl.LightningModule", values: torch.Tensor, sync_grads:bool = False) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if self.trainer.world_size == 1:
        return values
    all_values = self.all_gather(values, sync_grads=sync_grads)
    # concate the first dimension
    return all_values.view(-1, *all_values.size()[2:])



