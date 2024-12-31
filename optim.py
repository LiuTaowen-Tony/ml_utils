from torch import nn
import math
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim
from torch.optim.optimizer import Optimizer
import typing
from .dist import rank0_print

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, eta_min=0, last_step=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        self.last_step = 0
        super().__init__(optimizer, last_step)

    def get_lr(self):
        if self.last_step < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_step / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.last_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay for base_lr in self.base_lrs]

    def step(self, step=None):
        """Update the learning rate per step (batch) instead of per epoch."""
        if step is not None:
            self.last_step = step
        else:
            self.last_step += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

Params = typing.Dict[str, torch.Tensor]
OptimizerFactory = typing.Callable[[Params], torch.optim.Optimizer]
SchedulerFactory = typing.Callable[[torch.optim.Optimizer], _LRScheduler]

def escape_non_decay(model: nn.Module, optimizer_factory: OptimizerFactory, scheduler_factory: SchedulerFactory, weight_decay: float) -> Params:
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    rank0_print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    rank0_print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    optimizer = optimizer_factory(optim_groups)
    scheduler = scheduler_factory(optimizer)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }
