from torch import nn
import math
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim
from torch.optim.optimizer import Optimizer
import typing
from .dist import rank0_print


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, 
        optimizer: Optimizer, 
        warmup_steps: typing.Union[int, float], 
        max_steps: int, 
        eta_min: typing.Union[float, list[float]]=0, 
        eta_min_type: str="absolute",
        last_epoch: int=-1):
        """
        Args:
            optimizer: Optimizer
            warmup_steps: int | float
                Number of steps for linear warmup. If float, it is considered as a percentage.
            max_steps: int
                Total number of steps for training
            eta_min: float | list[float] 
                Minimum learning rate.
                as the minimum learning rate. (Default: 0)
            eta_min_type: str
                Type of eta_min. 
                - "absolute": The minimum learning rate is the same for all parameter groups.
                - "relative": The minimum learning rate is a percentage of the initial learning rate.
            last_epoch: 
                The index of last epoch. (Default: -1)
        """
        self.max_steps = max_steps

        assert warmup_steps <= max_steps, "warmup_steps should be less than max_steps"
        assert warmup_steps >= 0, "warmup_steps should be non-negative"
        if 0 < warmup_steps and warmup_steps < 1:
            warmup_steps = int(warmup_steps * max_steps)
        self.warmup_steps = warmup_steps

        self.eta_mins = eta_min
        if isinstance(eta_min, (int, float)):
            self.eta_mins = [eta_min] * len(optimizer.param_groups)

        assert len(self.eta_mins) == len(optimizer.param_groups), "eta_min should have the same length as optimizer.param_groups"

        for i in range(len(self.eta_mins)):
            assert 0 <= self.eta_mins[i], "eta_min should be non-negative"
            if eta_min_type == "relative":
                self.eta_mins[i] = self.eta_mins[i] * optimizer.param_groups[i]["lr"]

        self.last_step = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_step < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_step / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_decay = 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (self.last_step - self.warmup_steps)
                    / (self.max_steps - self.warmup_steps)
                )
            )
            return [
                eta_min + (base_lr - eta_min) * cosine_decay for base_lr, eta_min in zip(self.base_lrs, self.eta_mins)
            ]

    def step(self, step=None):
        """Update the learning rate per step (batch) instead of per epoch."""
        if step is not None:
            self.last_step = step
        else:
            self.last_step += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


Params = typing.Dict[str, torch.Tensor]
OptimizerFactory = typing.Callable[[Params], torch.optim.Optimizer]
SchedulerFactory = typing.Callable[[torch.optim.Optimizer], _LRScheduler]


def get_decay_non_decay(model: nn.Module) -> typing.List[nn.Parameter]:
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return decay_params, nodecay_params


def escape_non_decay(
    model: nn.Module,
    optimizer_factory: OptimizerFactory,
    scheduler_factory: SchedulerFactory,
    weight_decay: float,
) -> Params:
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
