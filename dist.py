import torch
import torch.distributed as dist
import pytorch_lightning as pl

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

def all_gather_concat_pl(self: pl.LightningModule, values: torch.Tensor, sync_grads:bool = False) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if self.trainer.world_size == 1:
        return values
    all_values = self.all_gather(values, sync_grads=sync_grads)
    # concate the first dimension
    return all_values.view(-1, *all_values.size()[2:])



