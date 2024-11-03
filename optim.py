import math
from torch.optim.lr_scheduler import _LRScheduler

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
