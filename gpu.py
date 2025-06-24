import torch

def to_device(pytree, device: str):
    if isinstance(pytree, torch.Tensor):
        return pytree.to(device)
    elif isinstance(pytree, list):
        return [to_device(x, device) for x in pytree]
    elif isinstance(pytree, tuple):
        return tuple(to_device(x, device) for x in pytree)
    elif isinstance(pytree, dict):
        return {k: to_device(v, device) for k, v in pytree.items()}
    else:
        return pytree